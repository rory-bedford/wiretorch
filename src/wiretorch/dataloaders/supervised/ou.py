"""Dataset that reconstructs feedforward inputs from the OU latent trajectory."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset

#: Archive arrays that may hold the static input patterns, newest name first.
PATTERN_KEYS = ("input_patterns", "odourant_patterns")


class OUReconstructedFFDataset(TeacherSpikeDataset):
    """Reconstruct FF rates from OU mixing weights and Poisson-sample spikes.

    Loads the low-dimensional OU latent trajectory (``weights``) and the static
    input patterns from ``spike_data.zarr``, reconstructs per-neuron firing rates via
    ``rates = weights @ patterns``, and Poisson-samples fresh FF spikes on each
    access.

    Args:
        spike_data_path: Path to ``spike_data.zarr`` with ``weights``, the patterns
            (``input_patterns``, or ``odourant_patterns`` in older archives) and
            ``output_spikes``.
        chunk_size: Number of timesteps per chunk.
        device: Unused: datasets return CPU tensors (see ``TeacherSpikeDataset``).
        recurrent_smoothing_tau: Gaussian sigma (ms) for smoothing recurrent
            input spikes.  ``None`` disables smoothing.
    """

    def __init__(
        self,
        spike_data_path: Union[Path, str],
        chunk_size: int,
        device: Union[str, torch.device] = "cpu",
        recurrent_smoothing_tau: Optional[float] = None,
    ):
        super().__init__(
            spike_data_path=spike_data_path,
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=recurrent_smoothing_tau,
        )

        patterns_key = next((k for k in PATTERN_KEYS if k in self._root), None)
        if "weights" not in self._root or patterns_key is None:
            raise ValueError(
                f"Zarr file at {spike_data_path} needs 'weights' (the OU latent "
                f"trajectory) and one of {PATTERN_KEYS} (the static input patterns); "
                "re-generate the teacher activity with both saved."
            )

        self.patterns = torch.from_numpy(self._root[patterns_key][:].astype(np.float32))
        self.weight_data = self._root["weights"]

        _, _, self.n_patterns = self.weight_data.shape
        self.n_input_neurons = self.patterns.shape[1]

        print(
            f"OUReconstructedFFDataset: {self.n_patterns} OU patterns → "
            f"{self.n_input_neurons} FF neurons  |  {self.n_neurons} rec neurons  |  "
            f"{self.num_chunks} chunks  |  dt={self.dt} ms"
        )

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        w_chunk = torch.from_numpy(
            np.array(self.weight_data[:, start_t:end_t, :], dtype=np.float32)
        )

        rates_hz = w_chunk @ self.patterns
        spike_prob = rates_hz * (self.dt / 1000.0)
        return torch.rand_like(spike_prob) < spike_prob
