"""Dataset that replaces feedforward inputs with homogeneous Poisson spikes."""

import torch
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset


class HomogeneousPoissonFFDataset(TeacherSpikeDataset):
    """Replace feedforward inputs with Poisson spikes at a uniform rate.

    The Poisson rate is either provided explicitly or computed as the average
    firing rate across all feedforward neurons and time in the original data.

    Args:
        spike_data_path: Path to ``spike_data.zarr``.
        chunk_size: Number of timesteps per chunk.
        device: Unused: datasets return CPU tensors (see ``TeacherSpikeDataset``).
        firing_rate_override: If provided, use this rate (Hz) instead of
            computing from the data.
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            input spikes.  ``None`` disables smoothing.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        firing_rate_override: Optional[float] = None,
        recurrent_smoothing_tau: Optional[float] = None,
    ):
        super().__init__(
            spike_data_path=spike_data_path,
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=recurrent_smoothing_tau,
        )

        self.input_spike_data = self._root["input_spikes"]
        _, _, self.n_input_neurons = self.input_spike_data.shape

        if firing_rate_override is not None:
            self.avg_firing_rate = firing_rate_override
        else:
            self.avg_firing_rate = self._compute_average_firing_rate()

        self.spike_prob = self.avg_firing_rate * self.dt * 1e-3

    def _compute_average_firing_rate(self) -> float:
        all_input_spikes = self.input_spike_data[:]
        total_spikes = all_input_spikes.sum()
        total_neuron_time_s = (
            self.batch_size * self.total_time * self.n_input_neurons * self.dt / 1000.0
        )
        return float(total_spikes / total_neuron_time_s)

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        length = end_t - start_t
        random_vals = torch.rand(self.batch_size, length, self.n_input_neurons)
        return random_vals < self.spike_prob
