"""Analysis functions for spike train statistics and firing rates."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import zarr
from numpy.typing import NDArray

from .utils import by_cell_type, mean_and_std


def firing_rates_from_spikes(spikes: NDArray, dt: float) -> NDArray:
    """Per-neuron firing rates in Hz from a spike array.

    Args:
        spikes: Shape ``(time, neurons)`` or ``(batch, time, neurons)``.
        dt: Timestep in milliseconds.

    Returns:
        Firing rates in Hz, shape ``(neurons,)`` or ``(batch, neurons)``.
    """
    time_axis = -2  # second-to-last axis is always time
    duration_s = spikes.shape[time_axis] * dt / 1000.0
    return spikes.sum(axis=time_axis) / duration_s


def compute_firing_rates_from_zarr(
    zarr_path: Union[Path, str],
    dataset_name: str,
    dt: float,
) -> np.ndarray:
    """Compute firing rates from Zarr spike data using chunked Dask computation.

    Efficiently computes firing rates without loading the entire array into
    memory. Processes data in chunks using Dask, then returns the final result
    as a NumPy array.

    Args:
        zarr_path: Path to the Zarr store containing spike data.
        dataset_name: Name of the dataset within the Zarr store
            (e.g., ``"input_spikes"``, ``"output_spikes"``).
        dt: Timestep in milliseconds.

    Returns:
        Firing rates in Hz with shape ``(batch_size, n_patterns, n_neurons)``.
    """
    root = zarr.open(zarr_path, mode="r")
    spikes_zarr = root[dataset_name]

    batch_size, n_patterns, n_timesteps, n_neurons = spikes_zarr.shape
    duration_s = n_timesteps * dt * 1e-3

    spikes_dask = da.from_zarr(spikes_zarr)
    spike_counts = spikes_dask.sum(axis=2)
    firing_rates = (spike_counts / duration_s).compute()

    return firing_rates


def compute_spike_train_cv(
    spike_trains: NDArray[np.int32], dt: float = 1.0
) -> NDArray[np.float32]:
    """
    Compute coefficient of variation (CV) for spike trains for each neuron.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons)
            or (batch_size, n_patterns, n_steps, n_neurons).
        dt (float): Time step duration in the desired time units (e.g., seconds or milliseconds).
            Since CV is dimensionless, the units cancel out, but consistent units should be used.
            Defaults to 1.0.

    Returns:
        NDArray[np.float32]: CV values of shape (batch_size, n_neurons) or
            (batch_size, n_patterns, n_neurons). Returns NaN for neurons with fewer than 3 spikes.
    """
    # One row per train: (batch[, pattern], neuron) x time.
    output_shape = spike_trains.shape[:-2] + spike_trains.shape[-1:]
    trains = np.moveaxis(spike_trains, -1, -2).reshape(-1, spike_trains.shape[-2])
    train, step = np.nonzero(trains > 0)  # sorted by train, then time

    times = step.astype(np.float32) * np.float32(dt)
    same = train[1:] == train[:-1]  # consecutive spikes of one train
    isi, owner = (times[1:] - times[:-1])[same], train[1:][same]
    n = np.bincount(owner, minlength=len(trains))
    total = np.bincount(owner, weights=isi, minlength=len(trains))
    squares = np.bincount(
        owner, weights=isi.astype(np.float64) ** 2, minlength=len(trains)
    )

    # At least 2 intervals (3 spikes) for an unbiased standard deviation.
    cv = np.full(len(trains), np.nan)
    ok = (n >= 2) & (total > 0)
    mean = total[ok] / n[ok]
    std = np.sqrt(np.maximum(squares[ok] - n[ok] * mean**2, 0.0) / (n[ok] - 1))
    cv[ok] = std / mean
    return cv.astype(np.float32).reshape(output_shape)


def compute_spike_train_fano_factor(
    spike_trains: NDArray[np.int32], window_size: int
) -> NDArray[np.float32]:
    """
    Compute Fano factor for spike trains for each neuron.

    The Fano factor is computed as the variance of spike counts divided by the
    mean spike count across time windows. Fano factor is computed separately
    for each neuron in each batch.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_size (int): Size of the time window (in steps) for counting spikes.

    Returns:
        NDArray[np.float32]: Fano factor values of shape (batch_size, n_neurons).
            Returns NaN for neurons with zero mean spike count.
    """
    batch_size, n_steps, n_neurons = spike_trains.shape

    # Initialize Fano factor values with NaN
    fano_values = np.full((batch_size, n_neurons), np.nan, dtype=np.float32)

    # Calculate number of complete windows
    n_windows = n_steps // window_size

    if n_windows < 1:
        return fano_values

    # Vectorized computation: reshape all data at once
    # Trim to complete windows and reshape
    trimmed_length = n_windows * window_size
    trimmed_spikes = spike_trains[
        :, :trimmed_length, :
    ]  # (batch_size, trimmed_length, n_neurons)

    # Reshape to windows: (batch_size, n_windows, window_size, n_neurons)
    windowed_spikes = trimmed_spikes.reshape(
        batch_size, n_windows, window_size, n_neurons
    )

    # Sum over window dimension to get spike counts per window
    # Shape: (batch_size, n_windows, n_neurons)
    spike_counts = windowed_spikes.sum(axis=2)

    # Compute mean and variance across windows for each neuron in each batch
    # Shape: (batch_size, n_neurons)
    mean_counts = spike_counts.mean(axis=1)  # Mean across windows
    var_counts = spike_counts.var(axis=1, ddof=1)  # Variance across windows

    # Compute Fano factor = var/mean, avoiding division by zero
    # Only compute where mean > 0 and we have more than 1 window
    valid_mask = (mean_counts > 0) & (n_windows > 1)
    fano_values[valid_mask] = var_counts[valid_mask] / mean_counts[valid_mask]

    return fano_values


def compute_firing_rate_by_cell_type(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    duration: float,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of firing rates by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.
        duration (float): Duration of the spike train in milliseconds.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_firing_rate_hz": Mean firing rate across neurons of this type (Hz)
            - "std_firing_rate_hz": Standard deviation of firing rates (Hz)
            - "n_silent_cells": Number of cells that don't fire at all
    """
    batch_size = spike_trains.shape[0]
    rates = spike_trains.sum(axis=(0, 1)) / (duration / 1000.0 * batch_size)
    stats = {}
    for cell_type, type_rates in by_cell_type(rates, cell_type_indices).items():
        mean, std = mean_and_std(type_rates)
        stats[cell_type] = {
            "mean_firing_rate_hz": mean,
            "std_firing_rate_hz": std,
            "n_silent_cells": int((type_rates == 0).sum()),
        }
    return stats


def compute_cv_by_cell_type(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    dt: float = 1.0,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of coefficient of variation (CV) by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.
        dt (float): Time step duration in the desired time units (e.g., seconds or milliseconds).
            Defaults to 1.0.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_cv": Mean CV across neurons of this type
            - "std_cv": Standard deviation of CV values
    """
    with np.errstate(invalid="ignore"):  # all-NaN neurons (fewer than 3 spikes)
        cv_per_neuron = np.nanmean(compute_spike_train_cv(spike_trains, dt=dt), axis=0)
    stats = {}
    for cell_type, cvs in by_cell_type(cv_per_neuron, cell_type_indices).items():
        mean, std = mean_and_std(cvs)
        stats[cell_type] = {"mean_cv": mean, "std_cv": std}
    return stats
