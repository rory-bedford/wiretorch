"""The base class of the rate networks.

Connectivity, pathways and state are the shared ``ProjectionNetwork``'s (see
``network_simulators.network``); this adds the rate physiology, whose per-type values
can be learnt (``learn``). The time step itself lives in ``dynamics.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from wiretorch.network_simulators.network import (
    InputSide,
    ProjectionNetwork,
    check_parameters,
)
from wiretorch.network_simulators.rate import engine
from wiretorch.network_simulators.rate.dynamics import NONLINEARITIES, Physiology
from wiretorch.network_simulators.type_parameters import (
    TypeParameters,
    learnt_synapses,
    learnt_types,
)

CELL_PARAMETERS = ("cell_id", "name", "tau_mem", "nonlinearity")
SYNAPSE_PARAMETERS = ("synapse_id", "cell_id", "name", "tau_syn", "type")
#: Defaults of the optional cell parameters.
THETA, GAIN = 0.0, 1.0
#: What ``learn`` can train: per cell type, and per synapse type.
LEARNABLE_CELL = ("tau_mem", "theta", "gain")
LEARNABLE_SYNAPSE = ("tau_syn",)


class RateNetwork(ProjectionNetwork):
    """Base class of the rate networks.

    Subclasses describe their input sides and name their outputs; this class owns the
    physiology. The state ``v`` ``(B, N)`` and the synaptic currents ``I_syn``
    ``(B, N, S)`` continue from one ``forward`` call to the next until
    ``reset_state``.

    Physiology lives in two ``TypeParameters`` modules: ``cell_parameters`` (per cell
    type: ``tau_mem``, ``theta``, ``gain``) and ``synapse_parameters`` (per synapse
    type, in registry order: ``tau_syn``). ``learn`` makes any of them trainable, for
    every type (``True``) or a list of names (cell type names; synapse names):

        learn={"tau_mem": True, "theta": ["excitatory"], "tau_syn": ["NMDA"]}

    Everything else stays fixed. Weights are trained through the projections.
    """

    state_name = "I_syn"
    output_name = "r"
    _engine = engine

    def __init__(
        self,
        *,
        dt: float,
        cell_params: list[dict],
        cell_type_indices: np.ndarray,
        sides: list[InputSide],
        unify_synapses: bool,
        batch_size: int,
        learn: dict | None,
        track_variables: bool,
        track_gradients: bool,
        track_batch_idx: int | None,
        storage_dtype: torch.dtype | str = torch.float32,
    ):
        check_parameters(cell_params, CELL_PARAMETERS, "cell_params")
        for cell in cell_params:
            assert cell["nonlinearity"] in NONLINEARITIES, (
                f"cell type {cell['name']!r}: nonlinearity must be one of "
                f"{sorted(NONLINEARITIES)}, got {cell['nonlinearity']!r}"
            )
            assert cell["tau_mem"] >= 0, f"cell type {cell['name']!r}: tau_mem < 0"
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
        self.learn = dict(learn or {})
        unknown = set(self.learn) - set(LEARNABLE_CELL) - set(LEARNABLE_SYNAPSE)
        assert not unknown, (
            f"learn: unknown parameters {sorted(unknown)}; "
            f"can learn {LEARNABLE_CELL + LEARNABLE_SYNAPSE}"
        )
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

    # ------------------------------------------------------------------
    # Physiology
    # ------------------------------------------------------------------

    def _register_physiology(self) -> None:
        cells = sorted(self.cell_params, key=lambda c: c["cell_id"])
        names = [c["name"] for c in cells]
        self.cell_parameters = TypeParameters(
            {
                "tau_mem": [c["tau_mem"] for c in cells],
                "theta": [c.get("theta", THETA) for c in cells],
                "gain": [c.get("gain", GAIN) for c in cells],
            },
            learnt={
                name: learnt_types(self.learn.get(name), names, f"learn[{name!r}]")
                for name in LEARNABLE_CELL
            },
            positive={"tau_mem", "gain"},
        )
        synapses = self.registry.synapses
        self.synapse_parameters = TypeParameters(
            {"tau_syn": [s["tau_syn"] for s in synapses]},
            learnt={
                "tau_syn": learnt_synapses(
                    self.learn.get("tau_syn"), synapses, "learn['tau_syn']"
                ),
            },
            positive={"tau_syn"},
        )
        self.register_buffer(
            "sign",
            torch.tensor(
                [1.0 if s["type"] == "excitatory" else -1.0 for s in synapses]
            ),
        )
        self.nonlinearities = [c["nonlinearity"] for c in cells]
        for cell_type in range(len(cells)):
            neurons = (self.cell_type_indices == cell_type).nonzero(as_tuple=True)[0]
            self.register_buffer(f"_neurons_{cell_type}", neurons, persistent=False)

    def _set_timestep(self) -> None:
        pass  # the decays are computed in each forward call, as they may be learnt

    def _physiology(self) -> Physiology:
        types = self.cell_type_indices
        cell = self.cell_parameters
        return Physiology(
            theta=cell.value("theta")[types],
            gain=cell.value("gain")[types],
            beta=torch.exp(-self.dt / cell.value("tau_mem"))[types],
            sign=self.sign,
            decay=torch.exp(-self.dt / self.synapse_parameters.value("tau_syn")),
            groups=[
                (name, getattr(self, f"_neurons_{cell_type}"))
                for cell_type, name in enumerate(self.nonlinearities)
            ],
            storage_dtype=self.storage_dtype,
        )

    def _initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Voltages and currents zero."""
        v = torch.zeros((self.batch_size, self.n_neurons), device=self.device)
        # (B, N, S), laid out in memory as (B, S, N), like the current-based networks.
        I = torch.zeros(
            (self.batch_size, len(self.registry), self.n_neurons), device=self.device
        ).permute(0, 2, 1)
        return v, I

    def _side_currents(self, I: torch.Tensor, side: int) -> torch.Tensor:
        """The currents ``(..., S_side)`` of the synapse types one input side makes."""
        return I[..., self.registry.side_indices(side)]
