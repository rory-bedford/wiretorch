"""Dataset that returns raw OU latent trajectories as feedforward input."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from ._base import TeacherSpikeDataset


class LatentOUDataset(TeacherSpikeDataset):
    """The OU latent trajectory as feedforward input: continuous floats, not spikes.

    Where :class:`OUReconstructedFFDataset` Poisson-samples binary spikes from the
    rates the latents imply, this dataset returns the latents themselves (the
    archive's time-varying OU mixing ``weights``), letting the network learn a
    linear map from latents to neurons.

    ``SpikeData.input_spikes`` therefore holds real values, ``(batch, time,
    n_latents)`` float32: the field keeps its name because every dataset fills the
    same tuple. The simulators accept it unchanged -- an input "spike" only ever
    enters as a weight on the conductance update, so a float acts as a continuous
    rate input (the sparse-input path switches itself off for dense inputs).

    Args:
        spike_data_path: Path to ``spike_data.zarr`` with ``weights``
            and ``output_spikes``.
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

        if "weights" not in self._root:
            raise ValueError(
                f"Zarr file at {spike_data_path} is missing 'weights'. "
                "Ensure the teacher activity was generated with OU latents."
            )

        self.weight_data = self._root["weights"]  # (batch, time, n_latents)
        self.n_latents = self.weight_data.shape[2]

        print(
            f"LatentOUDataset: {self.n_latents} OU latents (continuous)  |  "
            f"{self.n_neurons} rec neurons  |  "
            f"{self.num_chunks} chunks  |  dt={self.dt} ms"
        )

    def _get_ff_spikes(self, start_t: int, end_t: int) -> torch.Tensor:
        """The latents in the window, float32 (the hook's name is shared with the
        spiking datasets)."""
        w_chunk = self.weight_data[:, start_t:end_t, :]  # (batch, chunk, n_latents)
        return torch.from_numpy(np.array(w_chunk, dtype=np.float32))
