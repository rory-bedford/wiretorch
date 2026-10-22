"""Dataset that loads exact feedforward spikes from disk."""

import torch
import zarr
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset
from ._smoothing import cached_smoothed_rates, read_spikes, sample_spikes


class ExactFFDataset(TeacherSpikeDataset):
    """Load both feedforward and recurrent spikes exactly from zarr.

    Optionally smooth the feedforward input spikes with a Gaussian kernel and
    Bernoulli-resample them at each chunk access. This degrades the precise
    timing information in the FF input, matching the inferring-inputs trick
    for forcing the student off a pure-FF solution. Target spikes stay exact.

    Args:
        spike_data_path: Path to ``spike_data.zarr``.
        chunk_size: Number of timesteps per chunk.
        input_dataset_name: Zarr dataset name for FF spikes.
        target_dataset_name: Zarr dataset name for recurrent spikes.
        device: Unused: datasets return CPU tensors (see ``TeacherSpikeDataset``).
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            input spikes. ``None`` disables smoothing.
        ff_smoothing_tau: Gaussian sigma (ms) for smoothing FF input spikes.
            ``None`` disables smoothing.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        input_dataset_name: str = "input_spikes",
        target_dataset_name: str = "output_spikes",
        device: Union[str, torch.device] = "cpu",
        recurrent_smoothing_tau: Optional[float] = None,
        ff_smoothing_tau: Optional[float] = None,
    ):
        super().__init__(
            spike_data_path=spike_data_path,
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=recurrent_smoothing_tau,
            target_dataset_name=target_dataset_name,
        )
        self.input_spike_data = self._root[input_dataset_name]
        _, _, self.n_input_neurons = self.input_spike_data.shape

        self.ff_smoothing_tau = ff_smoothing_tau
        self._ff_smoothed_rate_data: Optional[zarr.Array] = None
        self._input_dataset_name = input_dataset_name
        if self.ff_smoothing_tau:
            self._ensure_ff_smoothed_rates()

    def _ff_smoothed_rates_key(self) -> str:
        return f"smoothed_ff_rates_tau{self.ff_smoothing_tau}"

    def _ensure_ff_smoothed_rates(self) -> None:
        """Smoothed feedforward rates, from (or into) the archive's cache."""
        self._ff_smoothed_rate_data = cached_smoothed_rates(
            self.spike_data_path,
            self.input_spike_data,
            self._ff_smoothed_rates_key(),
            self.dt,
            self.ff_smoothing_tau,
            self.chunk_size,
        )

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        if self._ff_smoothed_rate_data is not None:
            return sample_spikes(self._ff_smoothed_rate_data, start_t, end_t)
        return read_spikes(self.input_spike_data, start_t, end_t)
