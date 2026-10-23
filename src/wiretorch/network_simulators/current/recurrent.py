"""Recurrent current-based LIF network with connectome-constrained weights."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from wiretorch.network_simulators.current import legacy
from wiretorch.network_simulators.current.legacy import (  # noqa: F401
    OptimisableParams,
)
from wiretorch.network_simulators.current.network import CurrentNetwork
from wiretorch.network_simulators.network import InputSide, PairKey, _long
from wiretorch.network_simulators.projections import Projection


class CurrentLIFNetwork(CurrentNetwork):
    """Recurrent current-based LIF network.

    Neurons receive recurrent input from each other through ``rec_projections`` and
    external input through ``ff_projections``, both dicts mapping
    ``(source_cell_type_name, target_cell_type_name)`` to a ``Projection``. See
    ``current.dynamics`` for the time step. Gradients flow through the recurrent spikes
    unless ``recurrent_gradients`` is False (``ConductanceLIFNetwork``'s default).

    A cell type with a refractory period ``tau_ref`` (ms, rounded to whole steps) holds
    its neurons at ``U_reset`` for that long after each spike, unable to spike.

    The state (``v``, ``I_syn``) continues across ``forward`` calls, so a long
    simulation can be run in chunks; call ``reset_state()`` before an independent one.
    ``from_weights`` builds the network from the former weight-matrix arguments.

    Example:
        >>> model = CurrentLIFNetwork(..., batch_size=10)
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
        recurrent_gradients: bool = True,
        rec_stp: dict | None = None,
        ff_stp: dict | None = None,
    ):
        """
        Args:
            dt: Simulation timestep (ms).
            rec_projections, ff_projections: Recurrent and feedforward connectivity
                (non-negative weights: the sign comes from the synapse type).
            cell_type_indices: Cell type of each neuron, ``(n_neurons,)``.
            cell_type_indices_FF: Cell type of each input, ``(n_inputs,)``.
            cell_params: Neuron cell types: 'name', 'cell_id', 'tau_mem', and
                optionally 'theta' (default 1), 'U_reset' (default 0), 'U_rest'
                (default 0), 'R' (default 1) — natural units by default — and
                'tau_ref' (the refractory period, default 0 ms: see the class
                docstring).
            cell_params_FF: Input cell types ('name', 'cell_id').
            synapse_params, synapse_params_FF: Synapses made by each side's cell types:
                'synapse_id', 'cell_id', 'name', 'tau_syn' (0 for a current with no
                memory), 'type' ('excitatory' or 'inhibitory', the current's sign).
            surrgrad_scale: Scale of the surrogate gradient (and of the escape noise).
            batch_size: Number of trials simulated in parallel.
            spike_mode: ``"deterministic"`` (Heaviside) or ``"sample"`` (Bernoulli).
            track_variables: Return voltages and synaptic currents per step.
            track_gradients: Retain per-step gradients (see ``get_tracked_gradients``).
            track_batch_idx: Track only this trial (all if None).
            storage_dtype: What the CUDA kernels store currents and recurrent weights
                in: float32, or float16 (see ``ProjectionNetwork.storage_dtype``).
            learn: Physiology to train, per cell type (``tau_mem``, ``theta``,
                ``U_reset``, ``U_rest``, ``R``) or synapse type (``tau_syn``): True for
                every type or a list of names (see ``CurrentNetwork``).
            recurrent_gradients: Whether gradients flow through the recurrent spikes
                into the presynaptic neurons (via their surrogates), as well as through
                the voltages and synaptic currents; if not, each step's recurrent input
                is treated as given. Default True; False makes the backward pass one
                reverse scan instead of a loop over steps.
            rec_stp, ff_stp: Short-term plasticity per ``(source, target)`` cell type
                pair of each side: ``STP``s (e.g. ``stp.DEPRESSING``) or dicts of
                ``U``, ``tau_D``, ``tau_F``; pairs not named have none (see ``stp``).
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
                    stp=rec_stp,
                ),
                InputSide(
                    ff_projections, cell_params_FF, synapse_params_FF, cell_type_indices_FF,
                    recurrent=False, attribute="ff_projections", mask_buffer="weights_mask_FF",
                    stp=ff_stp,
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
        self.recurrent_gradients = recurrent_gradients

    @classmethod
    def from_weights(
        cls,
        dt: float,
        weights: np.ndarray,
        weights_FF: np.ndarray,
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors: np.ndarray | None = None,
        scaling_factors_FF: np.ndarray | None = None,
        optimisable: OptimisableParams = None,
        connectome_mask: np.ndarray | None = None,
        feedforward_mask: np.ndarray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = False,
        learn: dict | None = None,
        cell_params_FF: list[dict] | None = None,
        recurrent_gradients: bool = True,
        rec_stp: dict | None = None,
        ff_stp: dict | None = None,
    ) -> CurrentLIFNetwork:
        """The network as the former weight-matrix constructor described it.

        Args:
            weights, weights_FF: Recurrent ``(n_neurons, n_neurons)`` and feedforward
                ``(n_inputs, n_neurons)`` non-negative weights.
            scaling_factors, scaling_factors_FF: ``(source type, target type)``
                multipliers (default ones).
            optimisable: "weights" (log-weights within the masks; scaling factors are
                then not applied), "scaling_factors" (both sides),
                "scaling_factors_recurrent", "scaling_factors_feedforward", or None.
            connectome_mask, feedforward_mask: Where weights may be non-zero; required
                when ``optimisable="weights"``.
            use_tqdm: Ignored (the networks no longer show a progress bar).
            learn: Physiology to train (see the constructor).
            cell_params_FF: Names of the input cell types (default ``input_<id>``).
            recurrent_gradients: Whether gradients flow through the recurrent spikes
                (see the constructor).
            rec_stp, ff_stp: Short-term plasticity (see the constructor).
            The rest: as the constructor. The neurons are in natural units (threshold 1,
            reset 0) whatever ``cell_params`` says, as before, and the weights are
            rescaled for the normalised step (``legacy.former_factors``), so this is
            the former network.
        """
        legacy.check_weights(weights, len(cell_type_indices), "weights", square=True)
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
        rec = legacy.side_projections(
            weights, cell_type_indices, cell_type_indices, names, names,
            scaling_factors, legacy.side_mode(optimisable, True, only_side=False),
            connectome_mask, factors,
        )  # fmt: skip
        ff = legacy.side_projections(
            weights_FF, cell_type_indices_FF, cell_type_indices, names_FF, names,
            scaling_factors_FF, legacy.side_mode(optimisable, False, only_side=False),
            feedforward_mask, factors,
        )  # fmt: skip
        model = cls(
            dt=dt,
            rec_projections=rec,
            ff_projections=ff,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=legacy.natural_units(cell_params),
            cell_params_FF=cell_params_FF,
            synapse_params=synapse_params,
            synapse_params_FF=synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            learn=learn,
            recurrent_gradients=recurrent_gradients,
            rec_stp=rec_stp,
            ff_stp=ff_stp,
        )
        model.optimisable = optimisable
        return model

    def forward(
        self, input_spikes: torch.Tensor | np.ndarray
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Simulate ``input_spikes.shape[1]`` steps, continuing from the current state.

        Args:
            input_spikes: External input, ``(batch_size, n_steps, n_inputs)``.

        Returns:
            Spikes ``(batch_size, n_steps, n_neurons)``; or, with ``track_variables``,
            a dict of ``spikes``, ``voltages``, ``currents`` (per synapse type, as in
            ``unified_synapse_params``), and the same currents split by input side,
            ``currents_syn_recurrent`` and ``currents_syn_feedforward``.
        """
        run = self._simulate(input_spikes)
        if not self.track_variables:
            return run.spikes
        I = run.tracked["I"]
        return {
            "spikes": run.spikes,
            "voltages": run.tracked["v"],
            "currents": I,
            "currents_syn_recurrent": self._side_currents(I, 0),
            "currents_syn_feedforward": self._side_currents(I, 1),
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


CurrentLIFNetwork_IO = CurrentNetwork  # the former base-class name
