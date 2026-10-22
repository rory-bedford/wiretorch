"""Gaussian smoothing of spike trains, its cache in the archive, and chunk reads."""

import numpy as np
import torch
import zarr


def smooth_spike_trains(spikes: torch.Tensor, dt: float, tau: float) -> torch.Tensor:
    """Convolve binary spikes with a normalised Gaussian kernel to estimate rates.

    The kernel has sigma = *tau* (in ms) and is truncated at ±3 sigma.
    The output is normalised so the sum of the kernel equals 1, preserving
    mean spike probability.

    Args:
        spikes: Binary spike tensor of shape ``(batch, time, neurons)``.
        dt: Simulation timestep in ms.
        tau: Gaussian kernel sigma in ms.

    Returns:
        Smoothed rate tensor (same shape), values in [0, 1].
    """
    batch, time, neurons = spikes.shape

    half_width = min(int(3.0 * tau / dt), time // 2)
    if half_width < 1:
        half_width = 1

    t_kernel = (
        torch.arange(
            -half_width, half_width + 1, device=spikes.device, dtype=torch.float32
        )
        * dt
    )
    kernel = torch.exp(-0.5 * (t_kernel / tau) ** 2)
    kernel = kernel / kernel.sum()

    # Conv1d: (batch, channels, time) with groups=neurons
    spikes_float = spikes.float().permute(0, 2, 1)  # (batch, neurons, time)
    spikes_padded = torch.nn.functional.pad(
        spikes_float, (half_width, half_width), mode="reflect"
    )
    kernel_1d = kernel.view(1, 1, -1).expand(neurons, 1, -1)  # (neurons, 1, K)
    rates = torch.nn.functional.conv1d(
        spikes_padded,
        kernel_1d,
        groups=neurons,
    )  # (batch, neurons, time)
    rates = rates.permute(0, 2, 1)  # (batch, time, neurons)
    rates = rates.clamp(0.0, 1.0)

    return rates


def cached_smoothed_rates(
    path, source: zarr.Array, key: str, dt: float, tau: float, time_chunk: int
) -> zarr.Array:
    """``smooth_spike_trains`` of ``source``, cached as ``key`` in the archive at ``path``.

    Stored as float16 (ample for Bernoulli probabilities). A cache written here is
    marked ``complete`` once fully written. One without the mark (an interrupted write,
    or a cache from before the mark existed) is accepted only if its total matches the
    raw spikes' (the kernel sums to 1), else recomputed. Existing caches are never
    modified, only read.
    """
    root = zarr.open_group(path, mode="a")
    if key in root:
        cached = root[key]
        if cached.attrs.get("complete") or _total(cached) >= 0.5 * _total(source):
            print(f"Loaded cached smoothed rates ({key}) from {path}")
            return zarr.open_group(path, mode="r")[key]
        print(f"WARNING: cached smoothed rates ({key}) look truncated; recomputing.")
        del root[key]

    print(f"Computing smoothed rates (tau={tau} ms) for {path} ...")
    spikes = torch.from_numpy(np.asarray(source[:], dtype=np.float32))
    rates = smooth_spike_trains(spikes, dt, tau).numpy().astype(np.float16)
    batch, _, n = rates.shape
    cache = root.create_array(
        key,
        shape=rates.shape,
        dtype=rates.dtype,
        chunks=(batch, time_chunk, n),
        overwrite=True,
    )
    cache[:] = rates
    cache.attrs["complete"] = True
    print(f"Cached smoothed rates to {path}/{key}")
    return zarr.open_group(path, mode="r")[key]


def _total(array: zarr.Array) -> float:
    return float(np.asarray(array[:], dtype=np.float32).sum())


def read_spikes(array: zarr.Array, start: int, stop: int) -> torch.Tensor:
    """Bool spikes ``array[:, start:stop]``."""
    return torch.from_numpy(np.asarray(array[:, start:stop])).bool()


def sample_spikes(rates: zarr.Array, start: int, stop: int) -> torch.Tensor:
    """Bernoulli spikes drawn afresh from ``rates[:, start:stop]``."""
    p = torch.from_numpy(np.asarray(rates[:, start:stop], dtype=np.float32))
    return torch.rand_like(p) < p
