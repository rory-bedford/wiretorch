"""Shared analysis utilities for experiment notebooks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interleave_spike_trains(
    *spike_arrays: NDArray,
    n_neurons: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Interleave multiple spike train arrays for comparison raster plots.

    Given *k* arrays of shape ``(time, neurons)`` produces:

    * ``interleaved`` of shape ``(1, time, k * n_neurons)`` where neuron
      columns cycle as ``[arr0_n0, arr1_n0, ..., arrk_n0, arr0_n1, ...]``.
    * ``cell_type_indices`` of shape ``(k * n_neurons,)`` cycling
      ``[0, 1, ..., k-1, 0, 1, ..., k-1, ...]``.

    These are ready to pass to
    :func:`visualization.plot_spike_trains` with ``n_compared=k``.

    Args:
        *spike_arrays: Two or more arrays of shape ``(time, neurons)``.
        n_neurons: Number of neurons per array to interleave. Defaults to
            the neuron count of the first array.
    """
    k = len(spike_arrays)
    if k < 2:
        raise ValueError("Need at least two spike arrays to interleave.")
    if n_neurons is None:
        n_neurons = spike_arrays[0].shape[1]
    n_time = spike_arrays[0].shape[0]
    interleaved = np.zeros((1, n_time, k * n_neurons))
    for i in range(n_neurons):
        for j, arr in enumerate(spike_arrays):
            interleaved[0, :, k * i + j] = arr[:, i]
    cell_type_indices = np.tile(np.arange(k), n_neurons)
    return interleaved, cell_type_indices


def make_grid_colormap(
    param_values: list[float] | NDArray,
    cmap_name: str | None = None,
) -> tuple:
    """Create a matplotlib Normalize + color dict for grid-search parameters.

    When *cmap_name* is ``None`` (default) and there are 8 or fewer
    parameter values, a qualitative palette from
    :data:`visualization.colors.QUALITATIVE_COLORS` is used for maximum
    visual distinction.  For more values or when *cmap_name* is given
    explicitly, a matplotlib sequential colormap is used instead.

    Args:
        param_values: Sorted list of parameter values.
        cmap_name: Matplotlib colormap name.  ``None`` → qualitative
            palette (≤8 values) or ``"viridis"`` (>8 values).

    Returns:
        ``(norm, color_dict)`` where *norm* is a
        :class:`~matplotlib.colors.Normalize` and *color_dict* maps each
        parameter value to a color string or RGBA tuple.
    """
    import matplotlib.pyplot as plt
    from wiretorch.visualization.colors import QUALITATIVE_COLORS

    n = len(param_values)
    if cmap_name is None and n <= len(QUALITATIVE_COLORS):
        color_dict = {v: QUALITATIVE_COLORS[i] for i, v in enumerate(param_values)}
        norm = plt.Normalize(min(param_values), max(param_values))
        return norm, color_dict

    if cmap_name is None:
        cmap_name = "viridis"
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(min(param_values), max(param_values))
    color_dict = {v: cmap(norm(v)) for v in param_values}
    return norm, color_dict


def mean_and_std(values: NDArray) -> tuple[float, float]:
    """Mean and unbiased std of the non-NaN values (std 0 for one value, NaN for none)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def by_cell_type(values: NDArray, cell_type_indices: NDArray) -> dict[int, NDArray]:
    """Per-neuron ``values`` grouped by cell type: ``{cell_type: values of its neurons}``."""
    return {
        int(cell_type): values[cell_type_indices == cell_type]
        for cell_type in np.unique(cell_type_indices)
    }
