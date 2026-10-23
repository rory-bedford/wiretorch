"""The base class of the current-based LIF networks.

Connectivity, pathways and state are the shared ``ProjectionNetwork``'s (see
``network_simulators.network``); this adds the current-based physiology. The time
step itself lives in ``dynamics.py`` (PyTorch) and ``kernels.py`` (Triton); networks
written with the former weight-matrix API are built and loaded by ``legacy.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from wiretorch.network_simulators.current import engine, legacy
from wiretorch.network_simulators.current.legacy import (  # noqa: F401
    OptimisableParams,
)
from wiretorch.network_simulators.current.dynamics import Physiology
from wiretorch.network_simulators.network import (
    InputSide,
    ProjectionNetwork,
    check_parameters,
    check_surrgrad_scale,
)

CELL_PARAMETERS = ("cell_id", "name", "tau_mem")
SYNAPSE_PARAMETERS = ("synapse_id", "cell_id", "name", "tau_syn", "type")
#: Defaults of the optional cell parameters: natural units.
THETA, U_RESET, U_REST, R = 1.0, 0.0, 0.0, 1.0
#: What ``learn`` can train per cell type (with defaults) and per synapse type, and
#: which stay positive (learnt as logarithms).
LEARNABLE = {
    "tau_mem": None,
    "theta": THETA,
    "U_reset": U_RESET,
    "U_rest": U_REST,
    "R": R,
    "adapt": 0.0,
    "tau_adapt": 100.0,
}
LEARNABLE_SYNAPSE = ("tau_syn",)
POSITIVE = {"tau_mem", "R", "tau_syn", "adapt", "tau_adapt"}


class CurrentNetwork(ProjectionNetwork):
    """Base class of the current-based LIF networks.

    Subclasses describe their input sides and name their outputs; this class owns the
    physiology buffers. The state ``v`` ``(B, N)`` and the synaptic currents ``I_syn``
    ``(B, N, S)`` continue from one ``forward`` call to the next until
    ``reset_state``.

    The physiology is fixed per-neuron buffers (``tau_mem``, ``theta``, ``U_reset``,
    ``U_rest``, ``R``; per synapse type ``tau_syn``) unless ``learn`` makes some of it
    trainable, per cell type (synapse type for ``tau_syn``), for every type (``True``)
    or the types named:

        learn={"tau_mem": True, "theta": ["excitatory"], "tau_syn": ["inh"]}

    The learnt values are ``cell_parameters`` and ``synapse_parameters``
    (``TypeParameters``; time constants and ``R`` as logarithms), and replace the
    buffers' values for those types. Only what is learnt costs anything in the
    backward pass.
    """

    state_name = "I_syn"
    _engine = engine
    has_refractory = True
    #: What ``from_weights`` was asked to train (the former API's ``optimisable``).
    optimisable = None

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
        check_parameters(cell_params, CELL_PARAMETERS, "cell_params")
        for side in sides:
            check_parameters(side.synapse_params, SYNAPSE_PARAMETERS, side.attribute)
            for synapse in side.synapse_params:
                assert synapse["type"] in ("excitatory", "inhibitory"), (
                    f"{side.attribute}: synapse type must be 'excitatory' or "
                    f"'inhibitory', got {synapse['type']!r}"
                )
                assert synapse["tau_syn"] >= 0, (
                    f"{side.attribute}: tau_syn must be >= 0, got {synapse['tau_syn']}"
                )
        self.spike_mode = spike_mode
        self._surrgrad_scale = surrgrad_scale
        self.learn = dict(learn or {})
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
        self.register_load_state_dict_pre_hook(legacy.convert_state_dict)

    # ------------------------------------------------------------------
    # Physiology
    # ------------------------------------------------------------------

    def _register_physiology(self) -> None:
        self.register_buffer("tau_mem", self._per_neuron("tau_mem"))
        self.register_buffer("theta", self._per_neuron("theta", THETA))
        self.register_buffer("U_reset", self._per_neuron("U_reset", U_RESET))
        self.register_buffer("U_rest", self._per_neuron("U_rest", U_REST))
        self.register_buffer("R", self._per_neuron("R", R))
        self._register_refractory()
        self._register_adaptation()  # a current, subtracted from the input
        self.register_buffer("tau_syn", self.registry.table("tau_syn"))
        self.register_buffer(
            "sign",
            torch.tensor(
                [
                    1.0 if s["type"] == "excitatory" else -1.0
                    for s in self.registry.synapses
                ]
            ),
        )
        self.register_buffer(
            "surrgrad_scale", torch.tensor(self._surrgrad_scale, dtype=torch.float32)
        )
        self._register_learnt(self.learn, LEARNABLE, LEARNABLE_SYNAPSE, POSITIVE)

    def _set_timestep(self) -> None:
        # tau_syn = 0 gives exp(-inf) = 0: a current with no memory.
        self.register_buffer("alpha", torch.exp(-self.dt / self.tau_syn))
        self.register_buffer("beta", torch.exp(-self.dt / self.tau_mem))
        self._set_refractory()
        self._set_adaptation()

    def _physiology(self) -> Physiology:
        learnt = self._learnt
        return Physiology(
            theta=learnt("theta", self.theta),
            U_reset=learnt("U_reset", self.U_reset),
            beta=learnt("tau_mem", self.beta, decay=True),
            U_rest=learnt("U_rest", self.U_rest),
            R=learnt("R", self.R),
            sign=self.sign,
            decay=learnt("tau_syn", self.alpha, decay=True),
            surr_scale=self.surrgrad_scale,
            spike_mode=self.spike_mode,
            storage_dtype=self.storage_dtype,
            n_ref=self.n_ref if self._refractory else None,
            **(
                dict(
                    adapt=learnt("adapt", self.adapt),
                    adapt_decay=learnt("tau_adapt", self.adapt_decay, decay=True),
                )
                if self._adapting
                else {}
            ),
        )

    def _initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Voltages at reset, currents zero."""
        v = self.U_reset.expand(self.batch_size, self.n_neurons).clone()
        # (B, N, S), laid out in memory as (B, S, N): see kernels.py.
        I = torch.zeros(
            (self.batch_size, len(self.registry), self.n_neurons), device=self.device
        ).permute(0, 2, 1)
        return v, I

    def _side_currents(self, I: torch.Tensor, side: int) -> torch.Tensor:
        """The currents ``(..., S_side)`` of the synapse types one input side makes."""
        return I[..., self.registry.side_indices(side)]
