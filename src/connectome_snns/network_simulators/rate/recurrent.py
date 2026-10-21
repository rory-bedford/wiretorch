"""Recurrent rate network with connectome-constrained weights."""

from __future__ import annotations

import numpy as np
import torch

from connectome_snns.network_simulators.network import InputSide, PairKey, _long
from connectome_snns.network_simulators.projections import Projection
from connectome_snns.network_simulators.rate.network import RateNetwork


class RecurrentRateNetwork(RateNetwork):
    """Recurrent rate network: the current-based LIF network with rates for spikes.

    Neurons receive recurrent input from each other through ``rec_projections`` and
    external input through ``ff_projections``, both dicts mapping
    ``(source_cell_type_name, target_cell_type_name)`` to a ``Projection``. See
    ``rate.dynamics`` for the time step, and for the settings that give the classic
    voltage-form and rate-form models.

    The state (``v``, ``I_syn``) continues across ``forward`` calls, so a long
    simulation can be run in chunks; call ``reset_state()`` before an independent one.

    Example:
        >>> model = RecurrentRateNetwork(..., batch_size=10, learn={"tau_mem": True})
        >>> for chunk in input_chunks:
        ...     rates = model(chunk)  # (batch, steps, neurons)
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
        batch_size: int,
        learn: dict | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        storage_dtype: torch.dtype | str = torch.float32,
    ):
        """
        Args:
            dt: Simulation timestep (ms).
            rec_projections, ff_projections: Recurrent and feedforward connectivity
                (non-negative weights: the sign comes from the synapse type).
            cell_type_indices: Cell type of each neuron, ``(n_neurons,)``.
            cell_type_indices_FF: Cell type of each input, ``(n_inputs,)``.
            cell_params: Neuron cell types: 'name', 'cell_id', 'tau_mem' (0 for a
                membrane that follows its input), 'nonlinearity' (one of
                ``rate.dynamics.NONLINEARITIES``), and optionally 'theta' (default 0)
                and 'gain' (default 1).
            cell_params_FF: Input cell types ('name', 'cell_id').
            synapse_params, synapse_params_FF: Synapses made by each side's cell types:
                'synapse_id', 'cell_id', 'name', 'tau_syn' (0 for an instantaneous
                synapse), 'type' ('excitatory' or 'inhibitory', the current's sign).
            batch_size: Number of trials simulated in parallel.
            learn: Physiology to train, ``{parameter: True | [type names]}`` (see
                ``RateNetwork``).
            track_variables: Return voltages and synaptic currents per step.
            track_gradients: Retain per-step gradients (see ``get_tracked_gradients``).
            track_batch_idx: Track only this trial (all if None).
            storage_dtype: What the CUDA kernels multiply rates and weights in:
                float32, or float16 for ~2.7x faster products on tensor cores, with
                relative errors ~1e-3 (see ``rate.engine``).
        """
        self.synapse_params = synapse_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF
        self.n_cell_types_FF = len(cell_params_FF)
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
            batch_size=batch_size,
            learn=learn,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            storage_dtype=storage_dtype,
        )  # fmt: skip
        self.register_buffer("cell_type_indices_FF", _long(cell_type_indices_FF))

    def forward(
        self, inputs: torch.Tensor | np.ndarray
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Simulate ``inputs.shape[1]`` steps, continuing from the current state.

        Args:
            inputs: External input, ``(batch_size, n_steps, n_inputs)``: rates,
                spikes, or any signal (e.g. white noise).

        Returns:
            Rates ``(batch_size, n_steps, n_neurons)``; or, with ``track_variables``,
            a dict of ``rates``, ``voltages``, ``currents`` (per synapse type, as in
            ``unified_synapse_params``), and the same currents split by input side,
            ``currents_recurrent`` and ``currents_feedforward``.
        """
        run = self._simulate(inputs)
        if not self.track_variables:
            return run.output
        I = run.tracked["I"]
        return {
            "rates": run.output,
            "voltages": run.tracked["v"],
            "currents": I,
            "currents_recurrent": self._side_currents(I, 0),
            "currents_feedforward": self._side_currents(I, 1),
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

    @property
    def unified_synapse_params(self) -> list[dict]:
        return self.registry.synapses
