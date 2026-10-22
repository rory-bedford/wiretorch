"""Poisson input spikes generated on the fly.

- Homogeneous: constant rates, one or several static patterns
  (:class:`HomogeneousPoissonSpikeDataset`, batched by
  :class:`HomogeneousPoissonSpikeDataLoader`).
- Inhomogeneous: time-varying rates from a rate process such as the OU pattern mixture
  of :mod:`.rate_processes` (:class:`InhomogeneousPoissonSpikeDataset`, iterated by
  :class:`InhomogeneousPoissonSpikeDataLoader`).

Rates are in Hz and ``dt`` in ms, so a step spikes with probability
``rate * dt / 1000``.
"""

from collections import namedtuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .rate_processes import OrnsteinUhlenbeckRateProcess

InhomogeneousPoissonData = namedtuple("InhomogeneousPoissonData", ["input_spikes"])
InhomogeneousPoissonDataWithRates = namedtuple(
    "InhomogeneousPoissonDataWithRates", ["input_spikes", "rates", "weights"]
)
HomogeneousPoissonData = namedtuple(
    "HomogeneousPoissonData", ["input_spikes", "pattern_idx"]
)


# ── Homogeneous: constant rates ──────────────────────────────────────────────


class HomogeneousPoissonSpikeDataset(Dataset):
    """Poisson spike trains at constant rates, cycling through one or more patterns.

    Item ``idx`` is a fresh ``(n_steps, n_neurons)`` bool spike train of pattern
    ``idx % n_patterns``; the dataset is effectively infinite.

    Args:
        firing_rates: Rates in Hz, ``(n_neurons,)`` (one pattern) or
            ``(n_patterns, n_neurons)``.
        chunk_size: Duration of each item in ms (``n_steps = int(chunk_size / dt)``).
        dt: Time step in ms.
        device: Device the spikes are generated on.

    Example:
        >>> dataset = HomogeneousPoissonSpikeDataset(
        ...     np.random.uniform(5, 20, (10, 100)), chunk_size=100.0, dt=1.0
        ... )
        >>> spikes, pattern_idx = dataset[15]  # pattern 15 % 10 = 5
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: float,
        dt: float,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        if isinstance(firing_rates, np.ndarray):
            firing_rates = torch.from_numpy(firing_rates)
        elif not isinstance(firing_rates, torch.Tensor):
            raise TypeError("firing_rates must be a numpy array or torch tensor")
        firing_rates = firing_rates.float().to(self.device)
        if firing_rates.ndim == 1:
            firing_rates = firing_rates.unsqueeze(0)
        elif firing_rates.ndim != 2:
            raise ValueError(
                "firing_rates must be 1D (n_neurons,) or 2D (n_patterns, n_neurons)"
            )

        self.firing_rates = firing_rates
        self.n_patterns, self.n_neurons = firing_rates.shape
        self.chunk_size = chunk_size
        self.dt = dt
        self.n_steps = int(chunk_size / dt)
        self.spike_probs = self.firing_rates * dt * 1e-3

    def __len__(self) -> int:
        return 2**31 - 1  # effectively infinite

    def __getitem__(self, idx: int) -> HomogeneousPoissonData:
        pattern_idx = idx % self.n_patterns
        uniform = torch.rand(self.n_steps, self.n_neurons, device=self.device)
        return HomogeneousPoissonData(
            input_spikes=uniform < self.spike_probs[pattern_idx],
            pattern_idx=pattern_idx,
        )


def collate_pattern_batches(batch):
    """Stack ``batch_size`` repeats of every pattern into one tensor.

    The loader fetches ``batch_size * n_patterns`` consecutive items, i.e. patterns
    ``0, 1, ..., n_patterns - 1`` repeated ``batch_size`` times.

    Returns:
        ``HomogeneousPoissonData`` with ``input_spikes`` of shape ``(batch_size,
        n_patterns, n_steps, n_neurons)`` and ``pattern_idx`` the list of patterns
        (in the order of the second axis).
    """
    spikes = torch.stack([item.input_spikes for item in batch])
    patterns = []
    for item in batch:  # one cycle through the patterns
        if item.pattern_idx in patterns:
            break
        patterns.append(item.pattern_idx)
    batch_size = len(batch) // len(patterns)
    return HomogeneousPoissonData(
        input_spikes=spikes.reshape(batch_size, len(patterns), *spikes.shape[1:]),
        pattern_idx=patterns,
    )


def _collate_single_pattern(batch):
    """``(batch_size, n_steps, n_neurons)`` spikes of the only pattern."""
    spikes = torch.stack([item.input_spikes for item in batch])
    return HomogeneousPoissonData(input_spikes=spikes, pattern_idx=[0])


class HomogeneousPoissonSpikeDataLoader(DataLoader):
    """Batches of :class:`HomogeneousPoissonSpikeDataset` spike trains.

    One pattern: batches are ``(batch_size, n_steps, n_neurons)``. Several patterns:
    ``batch_size`` counts repeats of all patterns, and batches are ``(batch_size,
    n_patterns, n_steps, n_neurons)``. Each batch is a ``HomogeneousPoissonData``
    whose ``pattern_idx`` lists the patterns.

    Each item draws its own random numbers, so the spikes a seed produces do not
    depend on how items are batched.

    Args:
        firing_rates: Rates in Hz, ``(n_neurons,)`` or ``(n_patterns, n_neurons)``.
        chunk_size: Duration of each item in ms.
        dt: Time step in ms.
        batch_size: Samples (one pattern) or repeats of all patterns (several).
        device: Device the spikes are generated on.
        num_workers: DataLoader worker processes.
        **kwargs: Passed to :class:`torch.utils.data.DataLoader`.

    Example:
        >>> loader = HomogeneousPoissonSpikeDataLoader(
        ...     np.random.uniform(5, 20, (10, 100)), chunk_size=100.0, dt=1.0,
        ...     batch_size=5,
        ... )
        >>> spikes, patterns = next(iter(loader))  # spikes: (5, 10, 100, 100)
    """

    def __init__(
        self,
        firing_rates: Union[np.ndarray, torch.Tensor],
        chunk_size: float,
        dt: float,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        num_workers: int = 0,
        **kwargs,
    ):
        self.device = torch.device(device)
        dataset = HomogeneousPoissonSpikeDataset(
            firing_rates=firing_rates, chunk_size=chunk_size, dt=dt, device=device
        )
        if dataset.n_patterns == 1:
            collate_fn, items_per_batch = _collate_single_pattern, batch_size
        else:
            collate_fn = collate_pattern_batches
            items_per_batch = batch_size * dataset.n_patterns
        super().__init__(
            dataset=dataset,
            batch_size=items_per_batch,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **kwargs,
        )


# ── Inhomogeneous: time-varying rates ────────────────────────────────────────


class InhomogeneousPoissonSpikeDataset(Dataset):
    """Poisson spikes from the time-varying rates of a rate process.

    Item ``idx`` asks the rate process for its next chunk of rates and samples spikes
    from them. The rate process keeps its own state, so successive items continue one
    trajectory, and one item already holds the process's whole batch of trials.

    Args:
        rate_process: Source of the rates, e.g. an
            :class:`~.rate_processes.OrnsteinUhlenbeckRateProcess`.
        device: Device the spikes are sampled on.
        return_rates: Also return the rates and mixing weights (the rate process must
            have ``return_rates=True``).

    Returns (per item):
        ``InhomogeneousPoissonData(input_spikes)``, or with ``return_rates``
        ``InhomogeneousPoissonDataWithRates(input_spikes, rates, weights)``; spikes and
        rates are ``(batch, chunk_size, n_neurons)``, weights ``(batch, chunk_size,
        n_patterns)``.
    """

    def __init__(
        self,
        rate_process: OrnsteinUhlenbeckRateProcess,
        device: Union[str, torch.device] = "cpu",
        return_rates: bool = False,
    ):
        self.rate_process = rate_process
        self.device = torch.device(device)
        self.return_rates = return_rates
        self.dt = rate_process.dt
        self.n_neurons = rate_process.n_neurons
        self.chunk_size = rate_process.chunk_size

    def __len__(self) -> int:
        return len(self.rate_process)

    def __getitem__(
        self, idx: int
    ) -> Union[InhomogeneousPoissonData, InhomogeneousPoissonDataWithRates]:
        if self.return_rates:
            rates, weights = self.rate_process[idx]
            rates, weights = rates.to(self.device), weights.to(self.device)
        else:
            rates = self.rate_process[idx].to(self.device)

        spike_probs = rates * self.dt / 1000.0
        spikes = torch.rand_like(spike_probs, device=self.device) < spike_probs

        if self.return_rates:
            return InhomogeneousPoissonDataWithRates(spikes, rates, weights)
        return InhomogeneousPoissonData(input_spikes=spikes)


class InhomogeneousPoissonSpikeDataLoader:
    """Endless iterator over :class:`InhomogeneousPoissonSpikeDataset` batches.

    Each step yields the dataset's next item: one chunk of the rate process's
    trajectory for all of its trials, with independent noise per trial.

    Args:
        rate_process: Source of the rates (its ``batch_size`` sets the trials).
        batch_size: Recorded for reference only.
        device: Device the spikes are sampled on.
        return_rates: Also yield the rates and mixing weights (the rate process must
            have ``return_rates=True``).

    Example:
        >>> loader = InhomogeneousPoissonSpikeDataLoader(
        ...     OrnsteinUhlenbeckRateProcess(patterns, chunk_size=1000, dt=0.1,
        ...                                  tau=50.0, batch_size=32),
        ...     batch_size=32,
        ... )
        >>> spikes = next(iter(loader)).input_spikes  # (32, 1000, n_neurons)
    """

    def __init__(
        self,
        rate_process: OrnsteinUhlenbeckRateProcess,
        batch_size: int,
        device: Union[str, torch.device] = "cpu",
        return_rates: bool = False,
    ):
        self.dataset = InhomogeneousPoissonSpikeDataset(
            rate_process=rate_process, device=device, return_rates=return_rates
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.return_rates = return_rates
        self.dt = self.dataset.dt
        self.n_neurons = self.dataset.n_neurons
        self.chunk_size = self.dataset.chunk_size

    def __iter__(self):
        idx = 0
        while True:
            yield self.dataset[idx]
            idx += 1

    def __len__(self) -> int:
        return int(1e9)  # effectively infinite
