"""Spiking statistics: firing-rate, CV, ISI and Fano-factor distributions, PSTHs.

Most functions here draw one panel per recurrent cell type (coloured with the shared
cell-type palette) and accept either ``ax=None`` (new figure, returned) or a list of
axes, one per cell type. Population-activity plots (PSTH, assembly activity, OU mixing
weights) draw on a single axes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from wiretorch.analysis.firing_statistics import (
    compute_spike_train_cv,
    compute_spike_train_fano_factor,
)

from ._common import (
    cell_type_palette,
    centered_message,
    figure_and_axes,
    figure_and_axes_grid,
    finish_figure,
    set_time_xlim,
)
from .colors import FEEDFORWARD_COLOR


def _cell_type_row(ax, n_cell_types: int):
    """Resolve ``ax`` into one 6x5-inch panel per cell type, in a row (sharey)."""
    return figure_and_axes_grid(
        ax, n_cell_types, vertical=False, figsize=(6 * n_cell_types, 5)
    )


def _population_rate(
    spike_counts: NDArray, window_size: float, dt: float, n_neurons: int, batch: int
) -> NDArray:
    """Sliding-window population rate (Hz) from summed spike counts per time step.

    A boxcar of ``window_size`` ms is summed (not averaged) over the counts, then
    normalised by window duration, number of neurons and batch size.
    """
    window_steps = int(np.round(window_size / dt))
    convolved = np.convolve(spike_counts, np.ones(window_steps), mode="same")
    return convolved / (window_size * 1e-3) / n_neurons / batch


def _plot_count_histogram(
    ax: plt.Axes,
    values: NDArray,
    bin_edges,
    color,
    *,
    title: str,
    xlabel: str,
    mean_fmt: str,
    xlim: tuple[float, float] | None,
) -> None:
    """Histogram of ``values`` with a dashed mean line and an ``n=`` legend."""
    ax.hist(
        values,
        bins=bin_edges,
        alpha=0.6,
        edgecolor="black",
        linewidth=1.2,
        color=color,
        label=f"n={len(values)}",
    )
    ax.axvline(
        values.mean(),
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label=mean_fmt.format(values.mean()),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params()
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


# ── Fano factor ──────────────────────────────────────────────────────────────


def _mean_std_fano(fano_per_window: list[NDArray], mask: NDArray) -> tuple:
    """Mean and std (over neurons in ``mask``) of the Fano factor for each window."""
    means, stds = [], []
    for fano_factors in fano_per_window:
        fano_cell_type = fano_factors[:, mask].flatten()
        fano_valid = fano_cell_type[~np.isnan(fano_cell_type)]
        if len(fano_valid) > 0:
            means.append(float(fano_valid.mean()))
            stds.append(float(fano_valid.std()))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def _plot_fano_panel(ax, window_sizes_s, mean, std, color) -> None:
    """Dashed coloured mean line, black error bars and a +/- std band."""
    ax.plot(
        window_sizes_s,
        mean,
        linestyle="--",
        linewidth=2,
        color=color,
        alpha=0.8,
        zorder=2,
    )
    ax.errorbar(
        window_sizes_s,
        mean,
        yerr=std,
        fmt="o",
        capsize=5,
        markersize=8,
        color="black",
        ecolor="black",
        capthick=1.5,
        elinewidth=1.5,
        alpha=0.6,
        zorder=3,
    )
    ax.fill_between(
        window_sizes_s, mean - std, mean + std, color=color, alpha=0.2, zorder=1
    )


def plot_fano_factor_vs_window_size(
    spike_trains: NDArray[np.int32],
    window_sizes: list[int] | np.ndarray,
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot mean Fano factor across neurons as a function of window size, split by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        window_sizes (list[int] | np.ndarray): List of window sizes (in steps) to evaluate.
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for converting steps to time.
            Defaults to 1.0.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on: None to
            create a new figure, or a list of axes (one per cell type).

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    window_sizes_s = np.array(window_sizes) * dt * 1e-3  # ms -> s
    n_cell_types = len(cell_type_names)
    colors = cell_type_palette(n_cell_types)
    fig, axes, owns_figure = _cell_type_row(ax, n_cell_types)

    fano_per_window = [
        compute_spike_train_fano_factor(spike_trains, window_size)
        for window_size in window_sizes
    ]

    for cell_type_idx, panel in enumerate(axes):
        mean, std = _mean_std_fano(fano_per_window, cell_type_indices == cell_type_idx)
        has_valid_data = np.any(~np.isnan(mean))
        if has_valid_data:
            _plot_fano_panel(panel, window_sizes_s, mean, std, colors[cell_type_idx])
        else:
            centered_message(panel, "No spikes detected")
        panel.set_xlabel("Window Size (s)")
        panel.set_ylabel("Mean Fano Factor")
        panel.set_title(f"{cell_type_names[cell_type_idx].capitalize()}")
        panel.tick_params()
        if has_valid_data:
            if len(window_sizes_s) > 0 and np.all(np.array(window_sizes_s) > 0):
                panel.set_xscale("log")
            panel.axhline(
                1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1
            )
            panel.grid(True, alpha=0.3, which="both")

    # Shared y range across cell types
    if n_cell_types > 1:
        all_ylims = [panel.get_ylim() for panel in axes]
        global_ymin = min(ylim[0] for ylim in all_ylims)
        global_ymax = max(ylim[1] for ylim in all_ylims)
        for panel in axes:
            panel.set_ylim(global_ymin, global_ymax)

    return finish_figure(fig, owns_figure)


# ── CV and ISI histograms ────────────────────────────────────────────────────


def _valid_cv_for_type(cv_values: NDArray, mask: NDArray) -> NDArray:
    """Non-NaN CVs of the neurons in ``mask`` (``cv_values`` is 1-D or 2-D)."""
    if cv_values.ndim == 1:
        cv_cell_type = cv_values[mask]
    else:
        cv_cell_type = cv_values[:, mask].flatten()
    return cv_cell_type[~np.isnan(cv_cell_type)]


def plot_cv_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot histogram of CV values across all neurons, split by cell type.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for computing ISIs in seconds.
            Defaults to 1.0.
        bins (int): Number of histogram bins.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per cell type), or None to create new figure.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)
    colors = cell_type_palette(n_cell_types)

    # (neurons,) or (patterns, neurons); dt converted from ms to s
    cv_values = compute_spike_train_cv(spike_trains, dt=dt * 1e-3)
    cv_by_type = [
        _valid_cv_for_type(cv_values, cell_type_indices == i)
        for i in range(n_cell_types)
    ]

    # Shared x range (5% padding, floored at 0) and bins across cell types
    all_cv_valid = [cv for cv in cv_by_type if len(cv) > 0]
    if all_cv_valid:
        all_cv_valid = np.concatenate(all_cv_valid)
        x_min, x_max = all_cv_valid.min(), all_cv_valid.max()
        x_range = x_max - x_min
        x_min = max(0, x_min - 0.05 * x_range)
        x_max = x_max + 0.05 * x_range
        bin_edges = np.linspace(x_min, x_max, bins + 1)
    else:
        bin_edges = bins  # automatic binning if there is no data
        x_min, x_max = 0, 2

    fig, axes, owns_figure = _cell_type_row(ax, n_cell_types)
    for i, panel in enumerate(axes):
        title = f"{cell_type_names[i].capitalize()}"
        if len(cv_by_type[i]) > 0:
            _plot_count_histogram(
                panel,
                cv_by_type[i],
                bin_edges,
                colors[i],
                title=title,
                xlabel="Coefficient of Variation (CV)",
                mean_fmt="Mean: {:.2f}",
                xlim=(x_min, x_max),
            )
        else:
            centered_message(panel, "No valid CV values")
            panel.tick_params()
            panel.set_title(title)
            panel.set_xlim(x_min, x_max)

    return finish_figure(fig, owns_figure)


def _isis_by_cell_type(
    spike_trains: NDArray, cell_type_indices: NDArray, n_cell_types: int, dt: float
) -> list[list[NDArray]]:
    """Per cell type, a list of ISI arrays (s), one per neuron/batch with >= 2 spikes."""
    batch_size = spike_trains.shape[0]
    time_step = dt * 1e-3
    isis_by_type = [[] for _ in range(n_cell_types)]
    for cell_type_idx in range(n_cell_types):
        for neuron_idx in np.where(cell_type_indices == cell_type_idx)[0]:
            for batch_idx in range(batch_size):
                spike_indices = np.where(spike_trains[batch_idx, :, neuron_idx] > 0)[0]
                if len(spike_indices) >= 2:
                    isis = np.diff(spike_indices.astype(np.float32)) * time_step
                    isis_by_type[cell_type_idx].append(isis)
    return isis_by_type


def plot_isi_histogram(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float = 1.0,
    bins: int = 50,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> Figure:
    """
    Plot histogram of inter-spike intervals (ISIs) split by cell type.

    The shared x range runs from 0 to the 99.5th percentile of all ISIs.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step duration in milliseconds for converting time steps to seconds.
            Defaults to 1.0.
        bins (int): Number of histogram bins.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on: None to
            create a new figure, or a list of axes (one per cell type).

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)
    colors = cell_type_palette(n_cell_types)
    isis_by_type = _isis_by_cell_type(spike_trains, cell_type_indices, n_cell_types, dt)

    all_isis_combined = [np.concatenate(isis) for isis in isis_by_type if isis]
    if all_isis_combined:
        x_limit = np.percentile(np.concatenate(all_isis_combined), 99.5)
        bin_edges = np.linspace(0, x_limit, bins + 1)
    else:
        x_limit = None
        bin_edges = bins  # automatic binning if there is no data

    fig, axes, owns_figure = _cell_type_row(ax, n_cell_types)
    for i, panel in enumerate(axes):
        title = f"{cell_type_names[i].capitalize()}"
        if isis_by_type[i]:
            _plot_count_histogram(
                panel,
                np.concatenate(isis_by_type[i]),
                bin_edges,
                colors[i],
                title=title,
                xlabel="Inter-Spike Interval (s)",
                mean_fmt="Mean: {:.4f} s",
                xlim=(0, x_limit) if x_limit is not None else None,
            )
        else:
            centered_message(panel, "No ISIs found")
            panel.tick_params()
            panel.set_title(title)

    return finish_figure(fig, owns_figure)


# ── Population activity ──────────────────────────────────────────────────────


def plot_psth(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    window_size: float,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    input_spike_trains: NDArray[np.int32] | None = None,
) -> Figure:
    """
    Plot Peri-Stimulus Time Histogram (PSTH) split by cell type with overlapping windows.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        window_size (float): Window size for PSTH bins in milliseconds.
        dt (float): Time step duration in milliseconds. Defaults to 1.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        title (str | None): Optional custom title. If None, uses default title.
        input_spike_trains (NDArray[np.int32] | None): Optional feedforward input spikes with shape (batch_size, n_steps, n_inputs).

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    batch_size, n_steps, _ = spike_trains.shape
    colors = cell_type_palette(len(cell_type_names))
    time_s = np.arange(n_steps) * dt * 1e-3  # one bin centre per time step
    fig, ax, owns_figure = figure_and_axes(ax, figsize=(12, 6))

    for cell_type_idx, cell_type_name in enumerate(cell_type_names):
        mask = cell_type_indices == cell_type_idx
        n_neurons_type = mask.sum()
        if n_neurons_type == 0:
            continue
        spike_counts = spike_trains[:, :, mask].sum(axis=(0, 2))  # (n_steps,)
        ax.plot(
            time_s,
            _population_rate(spike_counts, window_size, dt, n_neurons_type, batch_size),
            color=colors[cell_type_idx],
            linewidth=1,
            alpha=0.6,
            label=f"{cell_type_name.capitalize()} (n={n_neurons_type})",
        )

    if input_spike_trains is not None:
        n_inputs = input_spike_trains.shape[2]
        spike_counts_ff = input_spike_trains.sum(axis=(0, 2))
        ax.plot(
            time_s,
            _population_rate(spike_counts_ff, window_size, dt, n_inputs, batch_size),
            color=FEEDFORWARD_COLOR,
            linewidth=1,
            alpha=0.6,
            label=f"Feedforward (n={n_inputs})",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.tick_params()
    ax.set_title(title if title is not None else "Population Activity")
    set_time_xlim([ax], 0, (n_steps - 1) * dt * 1e-3)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return finish_figure(fig, owns_figure)


# ── Firing-rate distribution ─────────────────────────────────────────────────


def _set_log_decade_ticks(ax: plt.Axes, x_min: float, x_max: float) -> None:
    """Ticks at the powers of ten within ``[0.9 * x_min, x_max]``, nicely labelled."""
    log_min = np.floor(np.log10(x_min))
    log_max = np.ceil(np.log10(x_max))
    xticks = []
    for exponent in range(int(log_min), int(log_max) + 1):
        tick_val = 10**exponent
        if tick_val > 0 and tick_val >= x_min * 0.9 and tick_val <= x_max:
            xticks.append(tick_val)
    ax.set_xticks(xticks)

    xticklabels = []
    for tick in xticks:
        if tick < 1:
            xticklabels.append(f"{tick:.2f}")
        elif tick < 10:
            xticklabels.append(f"{tick:.0f}")
        else:
            xticklabels.append(f"{int(tick)}")
    ax.set_xticklabels(xticklabels)


def _plot_rate_histogram_panel(
    ax: plt.Axes, rates: NDArray, log_bins: NDArray, x_min: float, x_max: float, color
) -> None:
    """Log-binned rate histogram, mean line and a hatched bar counting silent neurons."""
    rates_nonzero = rates[rates > 0]
    n_silent = (rates == 0).sum()
    legend_handles, legend_labels = [], []

    hist_patches = ax.hist(
        rates_nonzero, bins=log_bins, alpha=0.6, color=color, edgecolor="black"
    )
    legend_handles.append(hist_patches[2][0])
    legend_labels.append(f"n={len(rates)}")

    if len(rates) > 0:  # the mean includes silent neurons
        mean_rate = rates.mean()
        mean_line = ax.axvline(
            mean_rate, alpha=0.6, color="gray", linestyle="--", linewidth=2
        )
        legend_handles.append(mean_line)
        legend_labels.append(f"Mean = {mean_rate:.2f} Hz")

    if n_silent > 0:
        # Bar just left of the visible (log) range, annotated "0" underneath
        bar_position = x_min * 0.6
        bar_width = x_min * 0.2
        silent_bar = ax.bar(
            bar_position,
            n_silent,
            width=bar_width / 2,
            alpha=0.6,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            hatch="//",
        )
        ax.text(bar_position, -ax.get_ylim()[1] * 0.05, "0", ha="center", va="top")
        legend_handles.append(silent_bar)
        legend_labels.append(f"Silent: {n_silent}")
    else:
        legend_handles.append(ax.plot([], [], " ")[0])
        legend_labels.append("Silent: 0")

    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Count")
    ax.tick_params()
    if len(rates_nonzero) > 0 and x_min > 0:
        ax.set_xscale("log")
        ax.set_xlim(x_min * 0.5, x_max)
        _set_log_decade_ticks(ax, x_min, x_max)
    else:
        ax.set_xlim(0, 1)
    ax.legend(legend_handles, legend_labels)
    ax.grid(True, alpha=0.3)


def plot_firing_rate_distribution(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot distribution of firing rates in the recurrent network by cell type.

    Rates are binned logarithmically over the range of nonzero rates shared by all
    cell types; silent neurons are counted in a separate hatched bar.

    Args:
        output_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per cell type), or None to create new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_cell_types = len(cell_type_names)
    colors = cell_type_palette(n_cell_types)

    duration_s = output_spikes.shape[1] * dt * 1e-3
    firing_rates = output_spikes[0].sum(axis=0) / duration_s
    firing_rates_nonzero = firing_rates[firing_rates > 0]

    fig, axes, owns_figure = _cell_type_row(ax, n_cell_types)

    if len(firing_rates_nonzero) > 0:
        x_min, x_max = firing_rates_nonzero.min(), firing_rates_nonzero.max()
    else:
        x_min, x_max = 0.1, 10.0
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 30)

    for i, panel in enumerate(axes):
        _plot_rate_histogram_panel(
            panel,
            firing_rates[cell_type_indices == i],
            log_bins,
            x_min,
            x_max,
            colors[i],
        )
        panel.set_title(cell_type_names[i].capitalize())

    return finish_figure(fig, owns_figure)


def plot_assembly_population_activity(
    spike_trains: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    assembly_ids: NDArray[np.int32],
    window_size: float,
    dt: float = 1.0,
    excitatory_idx: int = 0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    colors: list | None = None,
) -> Figure:
    """
    Plot population activity for excitatory neurons grouped by assembly.

    Similar to PSTH but shows smoothed activity for each assembly separately,
    focusing only on excitatory neurons.

    Args:
        spike_trains (NDArray[np.int32]): Spike trains of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        assembly_ids (NDArray[np.int32]): Array of assembly IDs for each neuron (0-indexed, -1 for unassigned).
        window_size (float): Window size for smoothing in milliseconds.
        dt (float): Time step duration in milliseconds. Defaults to 1.0.
        excitatory_idx (int): Index of excitatory cell type. Defaults to 0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        title (str | None): Optional custom title. If None, uses default title.
        colors (list | None): One colour per assembly. Defaults to ``tab10``.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    batch_size, n_steps, _ = spike_trains.shape

    excitatory_mask = cell_type_indices == excitatory_idx
    excitatory_spikes = spike_trains[:, :, excitatory_mask]
    excitatory_assembly_ids = assembly_ids[excitatory_mask]

    unique_assemblies = np.unique(excitatory_assembly_ids)
    unique_assemblies = unique_assemblies[unique_assemblies >= 0]  # drop unassigned
    n_assemblies = len(unique_assemblies)
    if n_assemblies == 0:
        raise ValueError("No assemblies found with assigned neurons")

    if colors is None:
        cmap = plt.colormaps["tab10"]
        colors = [cmap(i % 10) for i in range(n_assemblies)]

    time_s = np.arange(n_steps) * dt * 1e-3
    fig, ax, owns_figure = figure_and_axes(ax, figsize=(12, 6))

    for assembly_idx, assembly_id in enumerate(unique_assemblies):
        assembly_mask = excitatory_assembly_ids == assembly_id
        n_neurons_assembly = assembly_mask.sum()
        if n_neurons_assembly == 0:
            continue
        spike_counts = excitatory_spikes[:, :, assembly_mask].sum(axis=(0, 2))
        ax.plot(
            time_s,
            _population_rate(
                spike_counts, window_size, dt, n_neurons_assembly, batch_size
            ),
            color=colors[assembly_idx],
            linewidth=1.5,
            alpha=0.7,
            label=f"Assembly {assembly_id}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.tick_params()
    ax.set_title(
        title if title is not None else "Excitatory Assembly Population Activity"
    )
    set_time_xlim([ax], 0, (n_steps - 1) * dt * 1e-3)
    ax.grid(True, alpha=0.3)
    return finish_figure(fig, owns_figure)


# ── Trial-to-trial comparison ────────────────────────────────────────────────


def _windowed_rates(spikes: NDArray, window_steps: int, n_windows: int, dt: float):
    """Rates (Hz) in consecutive non-overlapping windows, shape (n_windows, n_neurons)."""
    n_neurons = spikes.shape[1]
    windowed = spikes[: n_windows * window_steps, :].reshape(
        n_windows, window_steps, n_neurons
    )
    return windowed.sum(axis=1) / (window_steps * dt * 1e-3)


def plot_cross_correlation_scatter(
    spike_trains_trial1: NDArray[np.int32] | NDArray[np.float32],
    spike_trains_trial2: NDArray[np.int32] | NDArray[np.float32],
    window_size: float,
    dt: float = 1.0,
    cell_indices: NDArray[np.int32] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    max_rate: float | None = None,
) -> Figure:
    """
    Create a scatter plot comparing binned firing rates between two trials.

    For each neuron and each time window, computes the spike count and creates
    a scatter plot of trial 1 vs trial 2 firing rates across all windows.

    Args:
        spike_trains_trial1 (NDArray[np.int32] | NDArray[np.float32]): Spike trains for trial 1 of shape (batch_size, n_steps, n_neurons).
        spike_trains_trial2 (NDArray[np.int32] | NDArray[np.float32]): Spike trains for trial 2 of shape (batch_size, n_steps, n_neurons).
        window_size (float): Window size for binning spikes in seconds.
        dt (float): Time step duration in milliseconds. Defaults to 1.0.
        cell_indices (NDArray[np.int32] | None): Indices of cells to include. If None, uses all cells.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        title (str | None): Optional custom title. If None, uses default title.
        x_label (str | None): Custom label for x-axis. If None, uses default "Trial 1 Firing Rate (Hz)".
        y_label (str | None): Custom label for y-axis. If None, uses default "Trial 2 Firing Rate (Hz)".
        max_rate (float | None): Axis limit for both axes. Defaults to the data maximum.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    _, n_steps, n_neurons = spike_trains_trial1.shape
    _, n_steps2, n_neurons2 = spike_trains_trial2.shape
    if n_steps != n_steps2 or n_neurons != n_neurons2:
        raise ValueError(
            f"Spike train dimensions must match. Got trial1: ({n_steps}, {n_neurons}), "
            f"trial2: ({n_steps2}, {n_neurons2})"
        )

    if cell_indices is not None:
        spike_trains_trial1 = spike_trains_trial1[:, :, cell_indices]
        spike_trains_trial2 = spike_trains_trial2[:, :, cell_indices]
        n_neurons = len(cell_indices)

    window_steps = int(window_size * 1000 / dt)  # s -> ms -> steps
    n_windows = n_steps // window_steps
    if n_windows == 0:
        raise ValueError(
            f"Window size ({window_size}s = {window_steps} steps) is larger than "
            f"simulation duration ({n_steps} steps = {n_steps * dt / 1000}s)"
        )

    # Batch-averaged, windowed rates, flattened over (window, neuron) pairs
    rates1 = _windowed_rates(
        spike_trains_trial1.mean(axis=0), window_steps, n_windows, dt
    )
    rates2 = _windowed_rates(
        spike_trains_trial2.mean(axis=0), window_steps, n_windows, dt
    )
    rates1, rates2 = rates1.flatten(), rates2.flatten()

    fig, ax, owns_figure = figure_and_axes(ax, figsize=(8, 8))
    ax.scatter(rates1, rates2, alpha=0.3, s=10, color="steelblue", edgecolors="none")

    plot_max = max_rate if max_rate is not None else max(rates1.max(), rates2.max())
    ax.plot(
        [0, plot_max], [0, plot_max], "k--", alpha=0.5, linewidth=1.5, label="Unity"
    )
    ax.set_xlim(0, plot_max)
    ax.set_ylim(0, plot_max)

    corr_coef = np.corrcoef(rates1, rates2)[0, 1]
    ax.set_xlabel(x_label if x_label is not None else "Trial 1 Firing Rate (Hz)")
    ax.set_ylabel(y_label if y_label is not None else "Trial 2 Firing Rate (Hz)")
    ax.tick_params(labelsize=10)
    if title is None:
        title = "Cross-Correlation Scatter Plot"
    ax.set_title(f"{title} (r={corr_coef:.3f})", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    stats_text = (
        f"Windows: {n_windows}\n"
        f"Cells: {n_neurons}\n"
        f"Points: {len(rates1)}\n"
        f"Window: {window_size}s"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    return finish_figure(fig, owns_figure)


def plot_ou_process_weights(
    weights: NDArray,
    dt: float,
    *,
    ax: plt.Axes | None = None,
    title: str = "Latent Trajectories",
    colors: list | None = None,
    linewidth: float = 0.8,
    alpha: float = 0.85,
    xlim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Plot latent mixing-weight trajectories driving the input rate process.

    Each column of *weights* is one latent dimension (e.g. one input pattern
    pattern) whose value evolves over time according to an OU process.
    The resulting plot shows how strongly each pattern contributes to the
    instantaneous feedforward firing rates.

    Args:
        weights: Mixing weights, shape ``(time, n_patterns)``.
        dt: Simulation timestep in milliseconds.
        ax: Matplotlib axes.  If *None*, uses ``plt.gca()``.
        title: Axes title.
        colors: One color per pattern.  Defaults to ``tab20``.
        linewidth: Line width for each trajectory.
        alpha: Line alpha.
        xlim: X-axis limits in seconds.  Defaults to full duration.

    Returns:
        The axes with the plot.
    """
    if ax is None:
        ax = plt.gca()

    n_time, n_patterns = weights.shape
    time_s = np.arange(n_time) * dt / 1000.0

    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, n_patterns))

    for k in range(n_patterns):
        ax.plot(
            time_s, weights[:, k], color=colors[k], linewidth=linewidth, alpha=alpha
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mixing Weight")
    ax.set_title(title, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0, None)

    return ax
