"""Feedforward-only current-based LIF network."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from wiretorch.network_simulators.current import legacy
from wiretorch.network_simulators.current.legacy import OptimisableParams
from wiretorch.network_simulators.current.network import CurrentNetwork
from wiretorch.network_simulators.network import InputSide, PairKey, _long
from wiretorch.network_simulators.projections import Projection


class FeedforwardCurrentLIFNetwork(CurrentNetwork):
    """Current-based LIF neurons driven only by input rows — no recurrence.

    Every input row is feedforward: external inputs, and also recurrent neurons whose
    spikes are supplied (teacher forcing). Connectivity is one dict
    ``{(input_cell_type_name, output_cell_type_name): Projection}``. Because the whole
    chunk's input is known in advance, the GPU path integrates a chunk in a single
    kernel launch.

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
            cell_params: Output cell types (see ``CurrentLIFNetwork``).
            cell_params_FF: Input cell types ('name', 'cell_id').
            synapse_params_FF: Synapses made by the input cell types.
            surrgrad_scale, batch_size, spike_mode, track_*, storage_dtype: As
                ``CurrentLIFNetwork``.
            return_probabilities: Also return the per-step escape-noise spike
                probability ``sigmoid(surrgrad_scale * (v - theta))``, which carries a
                true gradient (rate losses can filter it instead of sampled spikes).
            learn: Physiology to train (see ``CurrentNetwork``).
            stp: Short-term plasticity per ``(input, output)`` cell type pair:
                ``STP`` objects (e.g. ``stp.DEPRESSING``) or dicts of ``U``, ``tau_D``,
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

    @classmethod
    def from_weights(
        cls,
        dt: float,
        weights_FF: np.ndarray,
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors_FF: np.ndarray | None = None,
        optimisable: OptimisableParams = None,
        feedforward_mask: np.ndarray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = False,
        learn: dict | None = None,
        cell_params_FF: list[dict] | None = None,
    ) -> FeedforwardCurrentLIFNetwork:
        """The network as the former weight-matrix constructor described it.

        Arguments as ``CurrentLIFNetwork.from_weights``, for the feedforward side only;
        any of the scaling-factor modes of ``optimisable`` trains
        ``scaling_factors_FF``.
        """
        legacy.check_weights(
            weights_FF, len(cell_type_indices_FF), "weights_FF", square=False
        )
        if cell_params_FF is None:
            cell_params_FF = legacy.input_cell_params(
                cell_type_indices_FF, synapse_params_FF
            )
        names = [c["name"] for c in sorted(cell_params, key=lambda c: c["cell_id"])]
        factors = legacy.former_factors(cell_params, dt)
        names_FF = [
            c["name"] for c in sorted(cell_params_FF, key=lambda c: c["cell_id"])
        ]
        projections = legacy.side_projections(
            weights_FF, cell_type_indices_FF, cell_type_indices, names_FF, names,
            scaling_factors_FF, legacy.side_mode(optimisable, False, only_side=True),
            feedforward_mask, factors,
        )  # fmt: skip
        model = cls(
            dt=dt,
            projections=projections,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=legacy.natural_units(cell_params),
            cell_params_FF=cell_params_FF,
            synapse_params_FF=synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            learn=learn,
        )
        model.optimisable = optimisable
        return model

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
            ``spikes``, ``voltages`` and ``currents_syn`` (and ``probabilities``).
        """
        run = self._simulate(input_spikes, probabilities=self.return_probabilities)
        if self.track_variables:
            out = {
                "spikes": run.spikes,
                "voltages": run.tracked["v"],
                **({"adaptation": run.tracked["a"]} if "a" in run.tracked else {}),
                "currents_syn": run.tracked["I"],
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
