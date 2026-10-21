"""Single-neuron and raster views of simulated dynamics.

- :func:`plot_membrane_voltages`, :func:`plot_synaptic_currents` and
  :func:`plot_synaptic_conductances` draw time traces, one stacked panel per neuron
  (or per synapse group), aligned to the end of the simulation.
- :func:`plot_spike_trains` draws spike rasters, optionally coloured by cell type;
  :func:`plot_input_spikes` and :func:`plot_network_spikes` are thin wrappers for the
  input and the network.

All functions take arrays shaped ``(batch, time, neurons[, ...])`` and plot batch 0.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

from ._common import (
    figure_and_axes_grid,
    finish_figure,
    floating_legend,
    raster_color_map,
    set_time_xlim,
    split_synapses_by_sign,
    warn_unused,
)
from .colors import (
    EXCITATORY_COLOR,
    INHIBITORY_COLOR,
    RASTER_ROW_SHADING_COLOR,
    SYNAPSE_COLORS,
)


def _round_to_nice_limit(value: float) -> float:
    """Round a value up to a nice round number (power of 10 times 1, 2, or 5).

    Args:
        value (float): The value to round up

    Returns:
        float: Rounded value
    """
    if value <= 0:
        return 1.0

    magnitude = 10 ** np.floor(np.log10(value))
    normalized = value / magnitude  # in [1, 10)
    if normalized <= 1:
        nice_normalized = 1
    elif normalized <= 2:
        nice_normalized = 2
    elif normalized <= 5:
        nice_normalized = 5
    else:
        nice_normalized = 10
    return nice_normalized * magnitude


def _plot_window(n_steps: int, fraction: float, dt: float) -> tuple[int, float, float]:
    """The last ``fraction`` of the simulation: ``(n_steps_plot, start_s, end_s)``."""
    n_steps_plot = int(n_steps * fraction)
    return n_steps_plot, (n_steps - n_steps_plot) * dt * 1e-3, n_steps * dt * 1e-3


def _trace_time_axis(n_steps: int, n_steps_plot: int, dt: float) -> NDArray:
    """Times (s) of the last ``n_steps_plot`` steps, aligned to the simulation end."""
    return np.arange(n_steps - n_steps_plot, n_steps) * dt * 1e-3


def _per_neuron_axes(ax, n_neurons_plot: int, figsize):
    """Resolve ``ax`` into one vertically stacked panel per neuron."""
    return figure_and_axes_grid(ax, n_neurons_plot, vertical=True, figsize=figsize)


# ── Membrane voltages ────────────────────────────────────────────────────────


def _plot_voltage_panel(
    ax: plt.Axes,
    time_s: NDArray,
    voltage: NDArray,
    spike_steps: NDArray,
    first_step: int,
    delta_t: float,
    threshold: float,
    rest: float,
) -> None:
    """Voltage trace, threshold/rest lines and a vertical tick from threshold to 0 mV
    at each spike."""
    ax.plot(time_s, voltage, linewidth=0.5, color="black", rasterized=True)
    ax.axhline(
        y=threshold,
        color=EXCITATORY_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="Threshold",
    )
    ax.axhline(
        y=rest,
        color=INHIBITORY_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="Rest",
    )
    if len(spike_steps) > 0:
        spike_times_s = (first_step + spike_steps) * delta_t * 1e-3
        spike_x = np.repeat(spike_times_s, 2)
        spike_y = np.tile([threshold, 0], len(spike_times_s))
        ax.plot(
            spike_x.reshape(-1, 2).T,
            spike_y.reshape(-1, 2).T,
            color="black",
            linewidth=0.5,
            alpha=0.7,
            zorder=5,
            rasterized=True,
        )


def plot_membrane_voltages(
    voltages: NDArray[np.float32],
    spikes: NDArray[np.int32],
    neuron_types: NDArray[np.int32],
    delta_t: float,
    neuron_params: dict,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    y_min: float = -100.0,
    y_max: float = 0.0,
    y_tick_step: float = 50.0,
    figsize: tuple[float, float] = (12, 12),
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """
    Visualize membrane voltage traces with spike markers.

    Args:
        voltages (NDArray[np.float32]): Voltage array with shape (batch, time, neurons).
        spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        neuron_types (NDArray[np.int32]): Array indicating neuron type indices (0, 1, 2, ...).
        delta_t (float): Time step in milliseconds.
        neuron_params (dict): Dictionary mapping cell type indices to parameters
            {'threshold': float, 'rest': float, 'name': str, 'sign': int}.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        y_min (float): Minimum y-axis value in mV. Defaults to -100.0.
        y_max (float): Maximum y-axis value in mV. Defaults to 0.0.
        y_tick_step (float): Step size for y-axis ticks. Defaults to 50.0.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on.
            Can be a single axis, list of axes (one per neuron), or None to create new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    n_steps = voltages.shape[1]
    n_steps_plot, start_s, end_s = _plot_window(n_steps, fraction, delta_t)
    fig, axes, owns_figure = _per_neuron_axes(ax, n_neurons_plot, figsize)
    time_s = _trace_time_axis(n_steps, n_steps_plot, delta_t)

    for neuron_id, panel in enumerate(axes):
        params = neuron_params[neuron_types[neuron_id]]
        _plot_voltage_panel(
            panel,
            time_s,
            voltages[0, -n_steps_plot:, neuron_id],
            np.where(spikes[0, -n_steps_plot:, neuron_id])[0],
            n_steps - n_steps_plot,
            delta_t,
            params["threshold"],
            params["rest"],
        )
        panel.set_ylabel("Membrane Potential (mV)")
        panel.tick_params()
        panel.set_ylim(y_min, y_max)
        panel.set_yticks(np.arange(y_min, y_max + y_tick_step / 2, y_tick_step))
        panel.grid(True, alpha=0.3)
        if neuron_id == 0:
            floating_legend(panel)

    axes[-1].set_xlabel("Time (s)")
    set_time_xlim(axes, start_s, end_s)
    return finish_figure(fig, owns_figure)


# ── Synaptic currents ────────────────────────────────────────────────────────


def _plot_current_panel(ax, time_s, i_tot, i_exc, i_inh, mean_total) -> None:
    """Total (gray), excitatory and inhibitory+leak traces, a zero line and the
    full-simulation mean total current."""
    for trace, color, label in (
        (i_tot, "gray", "Total"),
        (i_exc, EXCITATORY_COLOR, "Excitatory"),
        (i_inh, INHIBITORY_COLOR, "Inhibitory + Leak"),
    ):
        ax.plot(
            time_s,
            trace,
            linewidth=0.8,
            color=color,
            alpha=0.7,
            label=label,
            rasterized=True,
        )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"mean current = {mean_total:.2f} pA",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )


def plot_synaptic_currents(
    I_exc: NDArray[np.float32],
    I_inh: NDArray[np.float32],
    I_tot: NDArray[np.float32],
    delta_t: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    show_total: bool = False,
    neuron_types: NDArray[np.int32] | None = None,
    neuron_params: dict | None = None,
    figsize: tuple[float, float] = (12, 12),
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """
    Visualize excitatory and inhibitory synaptic currents.

    The symmetric y range is the 98th percentile of the absolute excitatory and
    inhibitory currents (over all plotted neurons), rounded up to a nice number.

    Args:
        I_exc (NDArray[np.float32]): Excitatory current array with shape (batch, time, neurons).
        I_inh (NDArray[np.float32]): Inhibitory current array with shape (batch, time, neurons).
        I_tot (NDArray[np.float32]): Total current array with shape (batch, time, neurons).
        delta_t (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        show_total, neuron_types, neuron_params: Deprecated, no effect (the total
            current is always drawn in grey); passing them warns.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on. If None, creates new figure.
            If list, should contain n_neurons_plot axes.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    warn_unused(
        "plot_synaptic_currents",
        **{
            name: value
            for name, value, default in (
                ("show_total", show_total, False),
                ("neuron_types", neuron_types, None),
                ("neuron_params", neuron_params, None),
            )
            if value is not default
        },
    )
    n_steps = I_exc.shape[1]
    n_steps_plot, start_s, end_s = _plot_window(n_steps, fraction, delta_t)

    all_currents = np.concatenate(
        [
            np.abs(current[0, -n_steps_plot:, neuron_id])
            for neuron_id in range(n_neurons_plot)
            for current in (I_exc, I_inh)
        ]
    )
    y_lim = _round_to_nice_limit(np.percentile(all_currents, 98))

    fig, axes, owns_figure = _per_neuron_axes(ax, n_neurons_plot, figsize)
    time_s = _trace_time_axis(n_steps, n_steps_plot, delta_t)

    for neuron_id, panel in enumerate(axes):
        _plot_current_panel(
            panel,
            time_s,
            I_tot[0, -n_steps_plot:, neuron_id],
            I_exc[0, -n_steps_plot:, neuron_id],
            I_inh[0, -n_steps_plot:, neuron_id],
            mean_total=I_tot[0, :, neuron_id].mean(),  # over the full simulation
        )
        panel.set_ylabel("Input Current (pA)")
        panel.tick_params()
        panel.set_ylim(-y_lim, y_lim)
        panel.grid(True, alpha=0.3)
        if neuron_id == 0:
            floating_legend(panel)

    axes[-1].set_xlabel("Time (s)")
    set_time_xlim(axes, start_s, end_s)
    return finish_figure(fig, owns_figure)


# ── Spike rasters ────────────────────────────────────────────────────────────


def _spike_times(spikes_subset: NDArray, first_step: int, dt: float) -> list[NDArray]:
    """Absolute spike times (s) for each column of a (time, neurons) spike array."""
    return [
        (np.where(spikes_subset[:, neuron_idx])[0] + first_step) * dt * 1e-3
        for neuron_idx in range(spikes_subset.shape[1])
    ]


def _raster_rows(ax: plt.Axes, n_neurons_plot: int, n_compared: int | None):
    """Row y positions, shading alternate rows (or alternate groups of ``n_compared``
    rows, separated by small gaps, when comparing spike trains)."""
    if n_compared is None:
        for i in range(0, n_neurons_plot, 2):
            ax.axhspan(i - 0.5, i + 0.5, color=RASTER_ROW_SHADING_COLOR, zorder=0)
        return np.arange(n_neurons_plot)

    gap_size = 0.3
    n_groups = n_neurons_plot // n_compared
    y_positions = [
        group_idx * (n_compared + gap_size) + within_group
        for group_idx in range(n_groups)
        for within_group in range(n_compared)
    ]
    # Shade every other group, extending half a gap on each side
    for group_idx in range(0, n_groups, 2):
        group_start = group_idx * (n_compared + gap_size)
        ax.axhspan(
            group_start - 0.5 - gap_size / 2,
            group_start + n_compared - 0.5 + gap_size / 2,
            color=RASTER_ROW_SHADING_COLOR,
            zorder=0,
        )
    return np.array(y_positions[:n_neurons_plot])


def _cell_type_legend(ax, colors_map, cell_type_names, recurrent_start_idx) -> None:
    """Legend patch per cell type, first type at the bottom."""
    legend_elements = []
    for i, name in enumerate(cell_type_names):
        if i == 0 and name == "Feedforward":
            color_idx = -1
        else:
            color_idx = i - recurrent_start_idx
        legend_elements.append(
            Patch(facecolor=colors_map[color_idx], label=name.capitalize())
        )
    ax.legend(handles=legend_elements[::-1], loc="upper right")


def _plot_spike_count_bars(ax_hist, y_positions, spike_times, colors) -> None:
    """Horizontal bar of spike count per raster row."""
    spike_counts = np.array([len(times) for times in spike_times])
    ax_hist.barh(y_positions, spike_counts, height=0.6, color=colors, edgecolor="none")
    ax_hist.set_xlabel("Spike Count")
    ax_hist.tick_params()
    ax_hist.yaxis.grid(False)
    ax_hist.xaxis.grid(True, alpha=0.3)
    ax_hist.set_xlim(0, None)


def plot_spike_trains(
    spikes: NDArray[np.int32],
    dt: float,
    cell_type_indices: NDArray[np.int32] | None = None,
    cell_type_names: list[str] | None = None,
    cell_type: str | None = None,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    random_seed: int = 42,
    title: str | None = None,
    ylabel: str = "Neuron ID",
    figsize: tuple[float, float] = (12, 4),
    ax: plt.Axes | None = None,
    n_compared: int | None = None,
    show_spike_histogram: bool = False,
    cell_type_colors: dict[int, str] | None = None,
) -> plt.Figure | None:
    """Plot spike trains with optional cell type coloring.

    Plots one cell type (in black) or several, coloured by type.

    Args:
        spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        cell_type_indices (NDArray[np.int32] | None): Array of cell type indices for each neuron.
            If None, all neurons are treated as the same type. Defaults to None.
        cell_type_names (list[str] | None): Names of cell types. Required if cell_type_indices
            is provided. A leading "Feedforward" name refers to index -1. Defaults to None.
        cell_type (str | None): Name of a single cell type (e.g. "input"), used in
            the default title. Defaults to None.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons when cell_type_indices
            is provided; None takes the first neurons in order. Defaults to 42.
        title (str | None): Custom title for the plot. If None, generates default title
            based on cell type. Defaults to None.
        ylabel (str): Label for y-axis (ignored when colouring by cell type).
            Defaults to "Neuron ID".
        figsize (tuple[float, float]): Figure size. Defaults to (12, 4).
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        n_compared (int | None): Number of spike trains being compared (e.g., 2 for
            teacher vs student). When set, neurons are grouped by index with gaps
            between groups and shared shading within each group. Defaults to None.
        show_spike_histogram (bool): If True, show a horizontal histogram of spike
            counts on the right side of the plot (only when ``ax`` is None).
            Defaults to False.
        cell_type_colors (dict[int, str] | None): Optional mapping from cell type index
            to color string. Overrides default colors when provided. Use -1 for feedforward.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    ax_hist = None
    if ax is None:
        if show_spike_histogram:
            fig, (ax, ax_hist) = plt.subplots(
                1, 2, figsize=figsize, width_ratios=[4, 1], sharey=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
        owns_figure = True
    else:
        fig = ax.get_figure()
        owns_figure = False

    n_steps = spikes.shape[1]
    n_steps_plot, start_s, end_s = _plot_window(n_steps, fraction, dt)
    first_step = n_steps - n_steps_plot
    colour_by_type = cell_type_indices is not None and cell_type_names is not None

    if colour_by_type:
        n_cell_types = len(cell_type_names)
        # A leading "Feedforward" name is index -1, so recurrent types start at 1
        recurrent_start_idx = (
            1 if (n_cell_types > 0 and cell_type_names[0] == "Feedforward") else 0
        )
        colors_map = (
            cell_type_colors
            if cell_type_colors is not None
            else raster_color_map(n_cell_types - recurrent_start_idx)
        )

        total_neurons = spikes.shape[2]
        n_neurons_plot = min(n_neurons_plot, total_neurons)
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            neuron_indices = rng.permutation(total_neurons)[:n_neurons_plot]
        else:
            neuron_indices = np.arange(n_neurons_plot)

        spikes_subset = np.take(spikes[0, -n_steps_plot:, :], neuron_indices, axis=1)
        colors = [colors_map[ct] for ct in cell_type_indices[neuron_indices]]
    else:
        spikes_subset = spikes[0, -n_steps_plot:, :n_neurons_plot]
        colors = "black"

    spike_times = _spike_times(spikes_subset, first_step, dt)
    y_positions = _raster_rows(ax, n_neurons_plot, n_compared)
    ax.eventplot(
        spike_times,
        colors=colors,
        lineoffsets=y_positions,
        linelengths=0.6,
        linewidths=0.8,
        rasterized=True,
    )

    if colour_by_type:
        _cell_type_legend(ax, colors_map, cell_type_names, recurrent_start_idx)
        ax.set_yticks([])
        default_title = "Spike Trains (colored by cell type)"
        ylabel = ""
    else:
        ax.set_yticks([] if n_compared is not None else range(n_neurons_plot))
        default_title = " ".join(
            ["Sample", *([cell_type.title()] if cell_type else []), "Spike Trains"]
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else default_title)
    ax.tick_params()
    ax.set_ylim(-0.5, y_positions[-1] + 0.5)
    ax.yaxis.grid(False)
    set_time_xlim([ax], start_s, end_s)

    if ax_hist is not None:
        _plot_spike_count_bars(ax_hist, y_positions, spike_times, colors)

    return finish_figure(fig, owns_figure)


def plot_input_spikes(
    input_spikes: NDArray[np.int32],
    dt: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
) -> plt.Figure:
    """Plot sample input spike trains (a wrapper around :func:`plot_spike_trains`).

    Args:
        input_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        dt (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.

    Returns:
        plt.Figure: Matplotlib figure object containing the input spike trains.
    """
    return plot_spike_trains(
        spikes=input_spikes,
        dt=dt,
        cell_type="input",
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
    )


def plot_network_spikes(
    output_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    n_neurons_plot: int = 20,
    fraction: float = 1.0,
    random_seed: int = 42,
) -> plt.Figure:
    """Plot sample network spike trains coloured by cell type (a wrapper around
    :func:`plot_spike_trains`).

    Args:
        output_spikes (NDArray[np.int32]): Spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 20.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        random_seed (int): Random seed for shuffling neurons. Defaults to 42.

    Returns:
        plt.Figure: Matplotlib figure object containing the spike trains.
    """
    return plot_spike_trains(
        spikes=output_spikes,
        dt=dt,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=fraction,
        random_seed=random_seed,
        figsize=(12, 6),
    )


#: Deprecated aliases, kept so existing experiments keep importing.
plot_mitral_cell_spikes = plot_input_spikes
plot_dp_network_spikes = plot_network_spikes


# ── Synaptic conductances ────────────────────────────────────────────────────


def _feedforward_synapses(
    input_cell_type_names: list[str], feedforward_synapse_names: dict[str, list[str]]
) -> list[tuple[int, str]]:
    """``(synapse_index, synapse_type)`` for every feedforward synapse, in order."""
    synapses = []
    for input_cell_name in input_cell_type_names:
        for syn_name in feedforward_synapse_names[input_cell_name]:
            synapses.append((len(synapses), syn_name))
    return synapses


def _synapse_color_map(synapse_types) -> dict:
    """Canonical synapse colours, plus ``Set1`` colours for any unknown types."""
    fallback_cmap = plt.colormaps["Set1"]
    unknown_types = sorted(t for t in synapse_types if t not in SYNAPSE_COLORS)
    color_map = {**SYNAPSE_COLORS}
    for i, syn_type in enumerate(unknown_types):
        color_map[syn_type] = fallback_cmap(i % 9)
    return color_map


def _plot_conductance_panel(
    ax, time_s, traces, synapses, color_map, y_max, label, legend_handles
) -> None:
    """One line per synapse (coloured by synapse type) and a group label top-left.

    Records the first line of each synapse type in ``legend_handles``.
    """
    for trace, (_, syn_name) in zip(traces, synapses):
        lines = ax.plot(
            time_s,
            trace,
            color=color_map[syn_name],
            linewidth=1,
            alpha=0.8,
            rasterized=True,
        )
        if syn_name not in legend_handles and len(lines) > 0:
            legend_handles[syn_name] = lines[0]
    ax.set_ylim(0, y_max)
    ax.tick_params()
    ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left")


def plot_synaptic_conductances(
    recurrent_conductances: NDArray[np.float32],
    feedforward_conductances: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
    dt: float,
    neuron_id: int = 0,
    fraction: float = 1.0,
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot synaptic conductances for a single neuron, grouped into 3 subplots (E, I, Feedforward).

    Plots individual synapse traces (AMPA, NMDA, GABA_A, GABA_B, etc.) grouped by type
    with a unified legend. The excitatory and feedforward panels share a y range (98th
    percentile of their traces, rounded up to a nice number); the inhibitory panel's
    range is ten times larger.

    Args:
        recurrent_conductances (NDArray[np.float32]): Recurrent conductances with shape (batch, time, neurons, synapses).
        feedforward_conductances (NDArray[np.float32]): Feedforward conductances with shape (batch, time, neurons, synapses).
        cell_type_indices (NDArray[np.int32]): Unused; kept for backward compatibility.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_synapse_names (dict[str, list[str]]): Synapse names for each recurrent cell type.
        feedforward_synapse_names (dict[str, list[str]]): Synapse names for each feedforward cell type.
        dt (float): Time step in milliseconds.
        neuron_id (int): Index of neuron to plot. Defaults to 0.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        ax (plt.Axes | list[plt.Axes] | None): Matplotlib axes to plot on. If None, creates new figure.
            If list, should contain 3 axes (E, I, Feedforward).

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    # Recurrent and feedforward arrays may differ in length: plot the last
    # n_steps_plot steps of the longer timeline, aligned at the end.
    n_steps_rec = recurrent_conductances.shape[1]
    n_steps_ff = feedforward_conductances.shape[1]
    n_steps = max(n_steps_rec, n_steps_ff)
    n_steps_plot = int(n_steps * fraction)
    rec_start_idx = max(0, n_steps_rec - n_steps_plot)
    ff_start_idx = max(0, n_steps_ff - n_steps_plot)
    time_rec = _trace_time_axis(n_steps, n_steps_rec - rec_start_idx, dt)
    time_ff = _trace_time_axis(n_steps, n_steps_ff - ff_start_idx, dt)

    # The synapse axis is ordered by presynaptic cell type
    exc_synapses, inh_synapses = split_synapses_by_sign(
        cell_type_names, recurrent_synapse_names
    )
    ff_synapses = _feedforward_synapses(
        input_cell_type_names, feedforward_synapse_names
    )
    exc_traces = [
        recurrent_conductances[0, rec_start_idx:, neuron_id, i] for i, _ in exc_synapses
    ]
    inh_traces = [
        recurrent_conductances[0, rec_start_idx:, neuron_id, i] for i, _ in inh_synapses
    ]
    ff_traces = [
        feedforward_conductances[0, ff_start_idx:, neuron_id, i] for i, _ in ff_synapses
    ]

    exc_ff_values = [trace.flatten() for trace in exc_traces + ff_traces]
    max_exc_ff = (
        np.percentile(np.concatenate(exc_ff_values), 98) if exc_ff_values else 1.0
    )
    y_lim_exc_ff = _round_to_nice_limit(max_exc_ff)
    y_lim_inh = y_lim_exc_ff * 10

    fig, axes, owns_figure = figure_and_axes_grid(
        ax, 3, vertical=True, figsize=(14, 6), what=" (E, I, FF)"
    )

    color_map = _synapse_color_map(
        {name for _, name in exc_synapses + inh_synapses + ff_synapses}
    )
    legend_handles = {}
    for panel, time_s, traces, synapses, y_max, label in (
        (axes[0], time_rec, exc_traces, exc_synapses, y_lim_exc_ff, "Excitatory"),
        (axes[1], time_rec, inh_traces, inh_synapses, y_lim_inh, "Inhibitory"),
        (axes[2], time_ff, ff_traces, ff_synapses, y_lim_exc_ff, "Feedforward"),
    ):
        _plot_conductance_panel(
            panel, time_s, traces, synapses, color_map, y_max, label, legend_handles
        )
    axes[1].set_ylabel("Conductance (µS)")

    if legend_handles:
        floating_legend(axes[0], legend_handles.values(), legend_handles.keys())

    axes[-1].set_xlabel("Time (s)")
    max_time = max(
        time_rec[-1] if len(time_rec) > 0 else 0,
        time_ff[-1] if len(time_ff) > 0 else 0,
    )
    min_time = min(
        time_rec[0] if len(time_rec) > 0 else max_time,
        time_ff[0] if len(time_ff) > 0 else max_time,
    )
    for panel in axes[:-1]:
        panel.set_xticklabels([])
    set_time_xlim(axes, min_time, max_time)
    return finish_figure(fig, owns_figure)
