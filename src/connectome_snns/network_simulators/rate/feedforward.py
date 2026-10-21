"""Feedforward-only rate network."""

from __future__ import annotations

import numpy as np
import torch

from connectome_snns.network_simulators.network import InputSide, PairKey, _long
from connectome_snns.network_simulators.projections import Projection
from connectome_snns.network_simulators.rate.network import RateNetwork


class FeedforwardRateNetwork(RateNetwork):
    """Rate neurons driven only by input rows — no recurrence.

    Every input row is feedforward: external inputs, and also recurrent neurons whose
    rates are supplied (teacher forcing). Connectivity is one dict
    ``{(input_cell_type_name, output_cell_type_name): Projection}``.

    The state (``v``, ``I_syn``) continues across ``forward`` calls; call
    ``reset_state()`` before an independent simulation.
    """

    def __init__(
        self,
        dt: float,
        projections: dict[PairKey, Projection],
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
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
            projections: ``{(input_cell_type_name, output_cell_type_name): Projection}``.
            cell_type_indices: Cell type of each output neuron, ``(n_neurons,)``.
            cell_type_indices_FF: Cell type of each input row, ``(n_inputs,)``.
            cell_params: Output cell types (see ``RecurrentRateNetwork``).
            cell_params_FF: Input cell types ('name', 'cell_id').
            synapse_params_FF: Synapses made by the input cell types.
            batch_size, learn, track_*, storage_dtype: As ``RecurrentRateNetwork``.
        """
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
                    projections, cell_params_FF, synapse_params_FF, cell_type_indices_FF,
                    recurrent=False, attribute="projections", mask_buffer="weights_mask_FF",
                )
            ],
            unify_synapses=False,
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

        Returns rates ``(batch_size, n_steps, n_neurons)``; or, with
        ``track_variables``, a dict of ``rates``, ``voltages`` and ``currents``.
        """
        run = self._simulate(inputs)
        if not self.track_variables:
            return run.output
        return {
            "rates": run.output,
            "voltages": run.tracked["v"],
            "currents": run.tracked["I"],
        }

    @property
    def weights_FF(self) -> torch.Tensor:
        """Input weights ``(n_inputs, n_neurons)``, assembled from the projections."""
        return self._weight_matrix(0)

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        """Scaling factors ``(n_cell_types_FF, n_cell_types)`` (1 if none)."""
        return self._scaling_matrix(0)
