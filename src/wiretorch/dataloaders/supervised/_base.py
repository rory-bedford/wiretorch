"""Base class for supervised spike-train datasets.

All supervised datasets share the same recurrent (target) side: spikes loaded
from a zarr archive, optionally smoothed and Bernoulli-resampled.  Only the
feedforward input strategy varies between subclasses.
"""

import torch
import zarr
from collections import namedtuple
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from typing import Iterator, Optional, Union

from ._smoothing import cached_smoothed_rates, read_spikes, sample_spikes

SpikeData = namedtuple(
    "SpikeData",
    [
        "input_spikes",
        "target_spikes",
        "recurrent_input_spikes",
        "target_calcium",
        "target_rate",
    ],
    defaults=[None, None, None, None],
)


class TeacherSpikeDataset(Dataset):
    """Abstract base for datasets that pair feedforward inputs with teacher spikes.

    Handles:
    * Opening the zarr archive and reading ``output_spikes``.
    * Chunking along the time axis.
    * Optional Gaussian smoothing of recurrent spikes across the **full**
      trial, cached to zarr, with fresh Bernoulli resampling on every access.

    Subclasses must implement :meth:`_get_ff_spikes`.

    Args:
        spike_data_path: Path to ``spike_data.zarr``.
        chunk_size: Number of timesteps per chunk.
        device: Unused, kept for compatibility (stored as ``self.device``). Datasets
            always return CPU tensors, so DataLoader workers never touch CUDA and
            batches can be pinned; the runners move each batch to the device.
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            spikes.  ``None`` disables smoothing.
        target_dataset_name: Name of the zarr dataset holding target spikes.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        recurrent_smoothing_tau: Optional[float] = None,
        target_dataset_name: str = "output_spikes",
    ):
        self.spike_data_path = Path(spike_data_path)
        self.chunk_size = chunk_size
        self.device = device
        self.recurrent_smoothing_tau = recurrent_smoothing_tau

        self._root = zarr.open_group(self.spike_data_path, mode="r")

        if "dt" not in self._root.attrs:
            raise ValueError(
                f"Zarr file at {spike_data_path} does not contain 'dt' attribute."
            )
        self.dt = float(self._root.attrs["dt"])

        self.target_spike_data = self._root[target_dataset_name]
        self.batch_size, self.total_time, self.n_neurons = self.target_spike_data.shape

        self.num_chunks = self.total_time // chunk_size
        if self.total_time % chunk_size != 0:
            print(
                f"Warning: total_time ({self.total_time}) not divisible by "
                f"chunk_size ({chunk_size})"
            )

        # Smoothed-rate cache (lazy — only built when smoothing is enabled)
        self._smoothed_rate_data: Optional[zarr.Array] = None
        if self.recurrent_smoothing_tau:
            self._ensure_smoothed_rates()

    # ------------------------------------------------------------------
    # Smoothing cache
    # ------------------------------------------------------------------

    def _smoothed_rates_key(self) -> str:
        return f"smoothed_rates_tau{self.recurrent_smoothing_tau}"

    def _ensure_smoothed_rates(self) -> None:
        """Smoothed recurrent rates, from (or into) the archive's cache."""
        self._smoothed_rate_data = cached_smoothed_rates(
            self.spike_data_path,
            self.target_spike_data,
            self._smoothed_rates_key(),
            self.dt,
            self.recurrent_smoothing_tau,
            self.chunk_size,
        )

    # ------------------------------------------------------------------
    # Recurrent spike loading
    # ------------------------------------------------------------------

    def _get_recurrent_chunk(self, start_t: int, end_t: int) -> torch.Tensor:
        """Recurrent input spikes: resampled from the smoothed rates if smoothing,
        else exact. Bool ``(batch, chunk_size, n_rec)`` on CPU."""
        if self._smoothed_rate_data is not None:
            return sample_spikes(self._smoothed_rate_data, start_t, end_t)
        return self._get_target_chunk(start_t, end_t)

    def _get_target_chunk(self, start_t: int, end_t: int) -> torch.Tensor:
        """Exact target spikes, bool ``(batch, chunk_size, n_rec)`` on CPU."""
        return read_spikes(self.target_spike_data, start_t, end_t)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        """Return feedforward input spikes for the given time window.

        Must be implemented by subclasses.

        Returns:
            Tensor of shape ``(batch, chunk_size, n_ff)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> SpikeData:
        start_t = idx * self.chunk_size
        end_t = start_t + self.chunk_size

        ff_spikes = self._get_ff_spikes(start_t, end_t)
        target_spikes = self._get_target_chunk(start_t, end_t)

        # When smoothing is enabled, recurrent *input* spikes differ from targets:
        # they are Bernoulli-resampled from the smoothed rates.
        if self._smoothed_rate_data is not None:
            rec_input_spikes = self._get_recurrent_chunk(start_t, end_t)
        else:
            rec_input_spikes = None  # collate will use target_spikes

        return SpikeData(
            input_spikes=ff_spikes,
            target_spikes=target_spikes,
            recurrent_input_spikes=rec_input_spikes,
        )


class CyclicSampler(Sampler):
    """Sampler that cycles through indices infinitely.

    Args:
        data_source: Dataset to sample from.
    """

    def __init__(self, data_source: Dataset):
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from range(self.num_samples)

    def __len__(self) -> int:
        return 2**31 - 1
