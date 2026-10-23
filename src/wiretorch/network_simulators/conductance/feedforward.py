"""Feedforward-only conductance-based LIF network simulator."""

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


class FeedforwardConductanceLIFNetwork(ConductanceNetwork):
    """Conductance-based LIF neurons driven only by input rows — no recurrence.

    Every input row is feedforward: external inputs, and also recurrent neurons whose
    spikes are supplied (teacher forcing, or the hidden layer of ``TwoLayerSNN``).
    Connectivity is one dict ``{(input_cell_type_name, output_cell_type_name):
    Projection}``. Because the whole chunk's input is known in advance, the GPU path
    integrates a chunk in a single kernel launch.

    The state (``v``, ``g_FF``) continues across ``forward`` calls; call
    ``reset_state()`` before an independent simulation.
    """

    state_name = "g_FF"

    def __init__(
        self,
        dt: float,
        projections: dict[PairKey, Projection],
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: Literal["deterministic", "sample"] = "deterministic",
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        storage_dtype: torch.dtype | str = torch.float32,
        return_probabilities: bool = False,
        learn: dict | None = None,
        stp: dict | None = None,
    ):
        """
        Args:
            dt: Simulation timestep (ms).
            projections: ``{(input_cell_type_name, output_cell_type_name): Projection}``.
            cell_type_indices: Cell type of each output neuron, ``(n_neurons,)``.
            cell_type_indices_FF: Cell type of each input row, ``(n_inputs,)``.
            cell_params: Output cell types (see ``ConductanceLIFNetwork``).
            cell_params_FF: Input cell types — the external inputs, plus the recurrent cell
                types whose spikes are supplied as input rows.
            synapse_params_FF: Synapses made by the input cell types.
            surrgrad_scale, batch_size, spike_mode, track_*, storage_dtype: As ``ConductanceLIFNetwork``.
            return_probabilities: Also return the per-step escape-noise spike
                probability ``sigmoid(surrgrad_scale * (v - theta))``, which carries a
                true gradient (rate losses can filter it instead of sampled spikes).
            learn: Physiology to train (see ``ConductanceNetwork``).
            stp: Short-term plasticity per ``(input, output)`` cell type pair:
                ``STP``s (e.g. ``stp.DEPRESSING``) or dicts of ``U``, ``tau_D``,
                ``tau_F``; pairs not named have none (see ``wiretorch.stp``).
        """
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)
        self.n_inputs = int(cell_type_indices_FF.shape[0])
        self.return_probabilities = return_probabilities
        super().__init__(
            dt=dt,
            cell_params=cell_params,
            cell_type_indices=cell_type_indices,
            sides=[
                InputSide(
                    projections, cell_params_FF, synapse_params_FF, cell_type_indices_FF,
                    recurrent=False, attribute="projections", mask_buffer="weights_mask_FF",
                    stp=stp,
                )
            ],
            unify_synapses=False,
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """Simulate ``input_spikes.shape[1]`` steps, continuing from the current state.

        Args:
            input_spikes: Input rows, ``(batch_size, n_steps, n_inputs)``. Usually
                spikes, but real-valued input activations work too.

        Returns:
            Spikes ``(batch_size, n_steps, n_neurons)``; ``(probabilities, spikes)``
            with ``return_probabilities``; or, with ``track_variables``, a dict of
            ``spikes``, ``voltages``, ``currents_feedforward``, ``currents_leak`` and
            ``conductances_feedforward`` (and ``probabilities``).
        """
        run = self._simulate(input_spikes, probabilities=self.return_probabilities)
        if self.track_variables:
            S = self.n_synapse_types_FF
            out = {
                "spikes": run.spikes,
                "voltages": run.tracked["v"],
                **({"adaptation": run.tracked["a"]} if "a" in run.tracked else {}),
                "currents_feedforward": run.tracked["I"][..., :S],
                "currents_leak": run.tracked["I"][..., S],
                "conductances_feedforward": run.tracked["g"],
            }
            if self.return_probabilities:
                out["probabilities"] = run.probabilities
            return out
        if self.return_probabilities:
            return run.probabilities, run.spikes
        return run.spikes

    @property
    def weights_FF(self) -> torch.Tensor:
        """Input weights ``(n_inputs, n_neurons)``, assembled from the projections."""
        return self._weight_matrix(0)

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        """Scaling factors ``(n_cell_types_FF, n_cell_types)`` (1 if none)."""
        return self._scaling_matrix(0)
