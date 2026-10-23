"""The base class of the conductance-based LIF networks.

Connectivity, pathways and state are the shared ``ProjectionNetwork``'s (see
``network_simulators.network``); this adds the conductance-based physiology. The time
step itself lives in ``dynamics.py`` (PyTorch) and ``kernels.py`` (Triton).
"""

from __future__ import annotations

import numpy as np
import torch

from wiretorch.network_simulators.conductance import engine
from wiretorch.network_simulators.conductance.dynamics import Physiology
from wiretorch.network_simulators.conductance.synapses import (
    conductance_kernel,
    mg_block_tables,
    neuron_tables,
    saturation_table,
    synapse_tables,
)
from wiretorch.network_simulators.network import (  # noqa: F401 - re-exported
    STORAGE_DTYPES,
    InputSide,
    PairKey,
    Pathway,
    ProjectionNetwork,
    _long,
    check_surrgrad_scale,
    projection_mask,
)


#: What ``learn`` can train per cell type and per synapse type, and which stay
#: positive (learnt as logarithms).
LEARNABLE = {
    "tau_mem": None, "theta": None, "U_reset": None, "E_L": None, "g_L": None,
    "adapt": 0.0, "tau_adapt": 100.0, "E_adapt": -90.0,
}  # fmt: skip
LEARNABLE_SYNAPSE = ("tau_rise", "tau_decay", "E_syn", "g_bar")
POSITIVE = {"tau_mem", "g_L", "tau_rise", "tau_decay", "g_bar", "adapt", "tau_adapt"}


class ConductanceNetwork(ProjectionNetwork):
    """Base class of the conductance-based LIF networks.

    Subclasses describe their input sides and name their outputs; this class owns the
    physiology buffers and the synapse registry. The state ``v`` ``(B, N)`` and the
    conductances ``(B, N, 2, S)`` continue from one ``forward`` call to the next until
    ``reset_state``.

    The physiology is fixed per-neuron and per-synapse buffers unless ``learn`` makes
    some of it trainable, per cell type (``tau_mem``, ``theta``, ``U_reset``, ``E_L``,
    ``g_L``, and the adaptation's ``adapt``, ``tau_adapt``, ``E_adapt``) or synapse type (``tau_rise``, ``tau_decay``, ``E_syn``, ``g_bar``), for
    every type (``True``) or the types named:

        learn={"tau_mem": True, "theta": ["E"], "tau_decay": ["GABA_A"]}

    The learnt values are ``cell_parameters`` and ``synapse_parameters``
    (``TypeParameters``; time constants, ``g_L`` and ``g_bar`` as logarithms), and
    replace the buffers' values for those types. Only what is learnt costs anything in
    the backward pass. The ``tau_decay`` of a saturating synapse type, which its
    presynaptic gating shares, cannot be learnt.
    """

    #: Name of the conductance state (also its checkpoint key).
    state_name = "g"
    _engine = engine
    has_refractory = True

    def __init__(
        self,
        *,
        dt: float,
        cell_params: list[dict],
        cell_type_indices: np.ndarray,
        sides: list[InputSide],
        unify_synapses: bool,
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: str,
        track_variables: bool,
        track_gradients: bool,
        track_batch_idx: int | None,
        storage_dtype: torch.dtype | str = torch.float32,
        learn: dict | None = None,
    ):
        check_surrgrad_scale(surrgrad_scale)
        self.learn = dict(learn or {})
        self.spike_mode = spike_mode
        self._surrgrad_scale = surrgrad_scale
        self._unify_synapses = unify_synapses
        super().__init__(
            dt=dt,
            cell_params=cell_params,
            cell_type_indices=cell_type_indices,
            sides=sides,
            unify_synapses=unify_synapses,
            batch_size=batch_size,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            storage_dtype=storage_dtype,
        )

    @property
    def conductance_name(self) -> str:
        """The former name of ``state_name``."""
        return self.state_name

    @property
    def _conductances(self) -> torch.Tensor:
        return self._synaptic

    @_conductances.setter
    def _conductances(self, value: torch.Tensor) -> None:
        self._synaptic = value

    # ------------------------------------------------------------------
    # Physiology
    # ------------------------------------------------------------------

    def _register_physiology(self) -> None:
        sides = self._sides
        if self._unify_synapses and len(self.registry) < sum(
            len(s.synapse_params) for s in sides
        ):
            shared = sum(len(s.synapse_params) for s in sides) - len(self.registry)
            print(
                f"  Unified synapse registry: {len(self.registry)} types "
                f"({shared} shared between recurrent/feedforward)"
            )
        types = self.cell_type_indices.numpy()
        for name, table in neuron_tables(self.cell_params, types).items():
            self.register_buffer(name, table)
        for name, table in synapse_tables(self.registry).items():
            self.register_buffer(name, table)
        # Not saved with the module (older checkpoints have no block): from the params.
        for name, table in mg_block_tables(self.registry).items():
            self.register_buffer(name, table, persistent=False)
        self._mg_block = bool(self.mg_c.any())
        self.register_buffer(
            "saturation", saturation_table(self.registry), persistent=False
        )
        self._register_refractory()
        # Spike-frequency adaptation: a potassium-like conductance, by E_adapt (mV).
        self._register_adaptation(E_adapt=-90.0)
        self.register_buffer(
            "surrgrad_scale", torch.tensor(self._surrgrad_scale, dtype=torch.float32)
        )
        self._register_synapse_maps(self._unify_synapses)
        self._register_learnt(self.learn, LEARNABLE, LEARNABLE_SYNAPSE, POSITIVE)
        learnt = getattr(self, "_learnt_tau_decay", None)
        if learnt is not None and (learnt & (self.saturation > 0)).any():
            names = [
                s["name"]
                for s, saturates in zip(
                    self.registry.synapses, learnt & (self.saturation > 0)
                )
                if saturates
            ]
            raise ValueError(
                f"learn['tau_decay']: cannot learn the decay of the saturating synapse "
                f"types {names}, which their presynaptic gating shares; name the types "
                "to learn instead"
            )

    def _saturation(self) -> torch.Tensor:
        return self.saturation

    def _register_synapse_maps(self, unified: bool) -> None:
        """Which synapse types each cell type makes, as the original simulators stored
        them — kept so that checkpoints written by them still load.

        ``cell_to_synapse_mask{_FF}``: per side, (source cell type, synapse type) bool.
        ``synapse_to_cell_id``: with a unified registry, the recurrent cell type making
        each synapse type (-1 if none); otherwise ``synapse_to_cell_id_FF``, the input
        cell type making each.
        """
        for side_index, side in enumerate(self._sides):
            made = self.registry.made_by(side_index, side.synapse_params)
            table = torch.zeros((max(made) + 1, len(self.registry)), dtype=torch.bool)
            for cell_type, synapses in made.items():
                table[cell_type] = synapses
            suffix = side.mask_buffer.removeprefix("weights_mask")
            self.register_buffer(f"cell_to_synapse_mask{suffix}", table)
        if unified:
            ids = {c["name"]: c["cell_id"] for c in self._sides[0].cell_params}
            maker = [ids.get(s["signature"][0], -1) for s in self.registry.synapses]
            self.register_buffer("synapse_to_cell_id", torch.tensor(maker))
        else:
            maker = [s["cell_id"] for s in self.registry.synapses]
            self.register_buffer("synapse_to_cell_id_FF", torch.tensor(maker))

    def _set_timestep(self) -> None:
        self.register_buffer("tau_syn", torch.stack((self.tau_rise, self.tau_decay)))
        self.register_buffer("alpha", torch.exp(-self.dt / self.tau_syn))
        self.register_buffer("beta", torch.exp(-self.dt / self.tau_mem))
        self.register_buffer(
            "g_scale", self._kernel(self.tau_rise, self.tau_decay, self.g_bar)
        )
        # The saturating types' gating variables decay as their conductances do.
        self.register_buffer("_gate_decay", self.alpha[1].clone(), persistent=False)
        self._set_refractory()
        self._set_adaptation()

    def _kernel(self, tau_rise, tau_decay, g_bar) -> torch.Tensor:
        return conductance_kernel(tau_rise, tau_decay, g_bar, self.saturation > 0)

    def _physiology(self) -> Physiology:
        learnt = self._learnt
        C_m, decay, jump = self.C_m, self.alpha, self.g_scale
        if self._learns("tau_mem", "g_L"):
            C_m = learnt("tau_mem", self.tau_mem) * learnt("g_L", self.g_L)
        if self._learns("tau_rise", "tau_decay", "g_bar"):
            tau_rise = learnt("tau_rise", self.tau_rise)
            tau_decay = learnt("tau_decay", self.tau_decay)
            decay = torch.exp(-self.dt / torch.stack((tau_rise, tau_decay)))
            jump = self._kernel(tau_rise, tau_decay, learnt("g_bar", self.g_bar))
        return Physiology(
            theta=learnt("theta", self.theta),
            U_reset=learnt("U_reset", self.U_reset),
            E_L=learnt("E_L", self.E_L),
            C_m=C_m,
            beta=learnt("tau_mem", self.beta, decay=True),
            E_syn=learnt("E_syn", self.E_syn),
            g_clip=self.g_clip,
            decay=decay,
            jump=jump,
            dt=self.dt,
            surr_scale=self.surrgrad_scale,
            spike_mode=self.spike_mode,
            storage_dtype=self.storage_dtype,
            n_ref=self.n_ref if self._refractory else None,
            mg_c=self.mg_c if self._mg_block else None,
            mg_a=self.mg_a if self._mg_block else None,
            **(
                dict(
                    adapt=learnt("adapt", self.adapt),
                    adapt_decay=learnt("tau_adapt", self.adapt_decay, decay=True),
                    E_adapt=learnt("E_adapt", self.E_adapt),
                )
                if self._adapting
                else {}
            ),
        )

    def _initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Voltages at reset, conductances zero."""
        v = self.U_reset.expand(self.batch_size, self.n_neurons).clone()
        # (B, N, 2, S), laid out in memory as (B, 2, S, N): see kernels.py.
        g = torch.zeros(
            (self.batch_size, 2, len(self.registry), self.n_neurons), device=self.device
        ).permute(0, 3, 1, 2)
        return v, g
