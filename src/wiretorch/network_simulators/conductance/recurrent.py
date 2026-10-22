"""Recurrent conductance-based LIF network with connectome-constrained weights."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from wiretorch.network_simulators.conductance.network import (
    ConductanceNetwork,
    InputSide,
    PairKey,
    _long,
)
from wiretorch.network_simulators.projections import Projection


class ConductanceLIFNetwork(ConductanceNetwork):
    """Recurrent conductance-based LIF network.

    Neurons receive recurrent input from each other through ``rec_projections`` and
    external input through ``ff_projections``, both dicts mapping
    ``(source_cell_type_name, target_cell_type_name)`` to a ``Projection``. Synapse
    types with the same (cell type, synapse) signature on both sides share one
    conductance (``unified_synapse_params``).

    The state (``v``, ``g``) continues across ``forward`` calls, so a long simulation
    can be run in chunks; call ``reset_state()`` before an independent one.

    Example:
        >>> model = ConductanceLIFNetwork(..., batch_size=10)
        >>> for chunk in input_chunks:
        ...     spikes = model(chunk)  # (batch, steps, neurons)
        >>> model.reset_state(batch_size=1)
        >>> model.track_variables = True
        >>> out = model(input_spikes)  # dict: spikes, voltages, currents, ...
    """

    def __init__(
        self,
        dt: float,
        rec_projections: dict[PairKey, Projection],
        ff_projections: dict[PairKey, Projection],
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: Literal["deterministic", "sample"] = "deterministic",
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        storage_dtype: torch.dtype | str = torch.float32,
        learn: dict | None = None,
    ):
        """
        Args:
            dt: Simulation timestep (ms).
            rec_projections, ff_projections: Recurrent and feedforward connectivity.
            cell_type_indices: Cell type of each neuron, ``(n_neurons,)``.
            cell_type_indices_FF: Cell type of each input, ``(n_inputs,)``.
            cell_params: Neuron cell types: 'name', 'cell_id', 'tau_mem', 'theta',
                'U_reset', 'E_L', 'g_L'.
            cell_params_FF: Input cell types ('name', 'cell_id').
            synapse_params, synapse_params_FF: Synapses made by each side's cell types:
                'synapse_id', 'cell_id', 'name', 'tau_rise', 'tau_decay', 'E_syn',
                'g_bar', optional 'g_clip'.
            surrgrad_scale: Scale of the surrogate gradient (and of the escape noise).
            batch_size: Number of trials simulated in parallel.
            spike_mode: ``"deterministic"`` (Heaviside) or ``"sample"`` (Bernoulli).
            track_variables: Return voltages, currents and conductances per step.
            track_gradients: Retain per-step gradients (see ``get_tracked_gradients``).
            track_batch_idx: Track only this trial (all if None).
            storage_dtype: What the CUDA kernels store conductances and recurrent
                weights in: float32, or float16 for 1.35x faster recurrent steps with
                the same statistics (see ``ConductanceNetwork.storage_dtype``).
            learn: Physiology to train, per cell type (``tau_mem``, ``theta``,
                ``U_reset``, ``E_L``, ``g_L``) or synapse type (``tau_rise``,
                ``tau_decay``, ``E_syn``, ``g_bar``): True for every type or a list of
                names (see ``ConductanceNetwork``).
        """
        self.synapse_params = synapse_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF
        self.n_synapse_types = len(synapse_params)
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)
        self.n_inputs = int(cell_type_indices_FF.shape[0])
        super().__init__(
            dt=dt,
            cell_params=cell_params,
            cell_type_indices=cell_type_indices,
            sides=[
                InputSide(
                    rec_projections, cell_params, synapse_params, cell_type_indices,
                    recurrent=True, attribute="rec_projections", mask_buffer="weights_mask",
                ),
                InputSide(
                    ff_projections, cell_params_FF, synapse_params_FF, cell_type_indices_FF,
                    recurrent=False, attribute="ff_projections", mask_buffer="weights_mask_FF",
                ),
            ],
            unify_synapses=True,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            spike_mode=spike_mode,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            storage_dtype=storage_dtype,
            learn=learn,
        )  # fmt: skip
        self.register_buffer("cell_type_indices_FF", _long(cell_type_indices_FF))

    def forward(
        self, input_spikes: torch.Tensor | np.ndarray
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Simulate ``input_spikes.shape[1]`` steps, continuing from the current state.

        Args:
            input_spikes: External input, ``(batch_size, n_steps, n_inputs)``.

        Returns:
            Spikes ``(batch_size, n_steps, n_neurons)``; or, with ``track_variables``,
            a dict of ``spikes``, ``voltages``, ``currents`` (per synapse type),
            ``currents_leak`` and ``conductances`` (rise and decay components).
        """
        run = self._simulate(input_spikes)
        if not self.track_variables:
            return run.spikes
        S = self.n_unified_synapse_types
        return {
            "spikes": run.spikes,
            "voltages": run.tracked["v"],
            "currents": run.tracked["I"][..., :S],
            "currents_leak": run.tracked["I"][..., S],
            "conductances": run.tracked["g"],
        }

    # Connectivity as matrices, assembled from the projections.

    @property
    def weights(self) -> torch.Tensor:
        """Recurrent weights ``(n_neurons, n_neurons)``."""
        return self._weight_matrix(0)

    @property
    def weights_FF(self) -> torch.Tensor:
        """Feedforward weights ``(n_inputs, n_neurons)``."""
        return self._weight_matrix(1)

    @property
    def scaling_factors(self) -> torch.Tensor:
        """Recurrent scaling factors ``(n_cell_types, n_cell_types)`` (1 if none)."""
        return self._scaling_matrix(0)

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        """Feedforward scaling factors ``(n_cell_types_FF, n_cell_types)`` (1 if none)."""
        return self._scaling_matrix(1)

    # The synapse registry, under the names analysis code uses.

    @property
    def unified_synapse_params(self) -> list[dict]:
        return self.registry.synapses

    @property
    def n_unified_synapse_types(self) -> int:
        return len(self.registry)

    @property
    def rec_synapse_to_unified(self) -> dict[int, int]:
        return self.registry.to_index[0]

    @property
    def ff_synapse_to_unified(self) -> dict[int, int]:
        return self.registry.to_index[1]

    @property
    def synapse_signature_to_unified_id(self) -> dict[tuple[str, str], int]:
        return {s["signature"]: s["unified_synapse_id"] for s in self.registry.synapses}
