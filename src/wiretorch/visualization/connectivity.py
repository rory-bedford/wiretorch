"""Connectivity plots: weight matrices and per-source summaries of synaptic input.

- Matrix views: :func:`plot_connectivity_by_cell_type`, :func:`plot_assembly_graph`,
  :func:`plot_weighted_connectivity`, :func:`plot_feedforward_connectivity`,
  :func:`plot_charge_transfer_matrix`, :func:`plot_weight_statistics_matrix`.
- Per-source summaries, where each input source (feedforward input types first, then
  recurrent cell types) gets one wedge, box or violin: input counts
  (:func:`plot_input_count_pie_chart`, :func:`plot_input_count_distribution`),
  synaptic drive (:func:`plot_synaptic_drive_pie_chart`,
  :func:`plot_synaptic_drive_distribution`), summed input conductance
  (:func:`plot_synaptic_input_histogram`) and weight values
  (:func:`plot_weight_distribution_by_input_type`).

Weight matrices are indexed ``W[pre, post]``: rows are presynaptic sources.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from wiretorch.configs.network import cell_type_sign

from ._common import (
    color_tick_labels,
    figure_and_axes,
    figure_and_axes_grid,
    title_above_axes,
    warn_unused,
)
from .colors import (
    EXCITATORY_COLOR,
    INHIBITORY_COLOR,
    FEEDFORWARD_SOURCE_COLOR,
    SATURATED_CELL_TYPE_COLORS,
    UNKNOWN_CELL_TYPE_COLOR,
)

# ── Shared helpers ───────────────────────────────────────────────────────────


def _source_color(cell_type_name: str) -> str:
    """Muted E/I colour for a recurrent source (see ``cell_type_sign``)."""
    return {
        "excitatory": EXCITATORY_COLOR,
        "inhibitory": INHIBITORY_COLOR,
    }.get(cell_type_sign(cell_type_name), UNKNOWN_CELL_TYPE_COLOR)


def _saturated_color(cell_type_name: str, is_feedforward: bool = False) -> str:
    """Saturated colour of a source: feedforward green, else E / I; gray if neither."""
    if is_feedforward:
        return SATURATED_CELL_TYPE_COLORS["feedforward"]
    return SATURATED_CELL_TYPE_COLORS.get(cell_type_sign(cell_type_name), "gray")


def _per_source(
    weights: NDArray,
    feedforward_weights: NDArray,
    cell_type_indices: NDArray,
    input_cell_type_indices: NDArray,
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    summarise,
):
    """Apply ``summarise(source_rows, name, is_feedforward)`` to every input source.

    Sources are the feedforward input types, then the recurrent cell types;
    ``source_rows`` is that source's rows of the corresponding weight matrix.

    Returns:
        ``(values, labels, colors)``: one entry per source; colours are green for
        feedforward sources and E/I for recurrent ones.
    """
    values, labels, colors = [], [], []
    for i, name in enumerate(input_cell_type_names):
        values.append(
            summarise(feedforward_weights[input_cell_type_indices == i, :], name, True)
        )
        labels.append(name)
        colors.append(FEEDFORWARD_SOURCE_COLOR)
    for i, name in enumerate(cell_type_names):
        values.append(summarise(weights[cell_type_indices == i, :], name, False))
        labels.append(name)
        colors.append(_source_color(name))
    return values, labels, colors


def _kernel_area_and_peak_norm(tau_rise: float, tau_decay: float):
    """Area (ms) and peak normalisation of a unit-peak difference-of-exponentials
    conductance kernel; charge per spike is ``w * g_bar * area / peak_norm * (V - E)``."""
    r = tau_rise / tau_decay
    peak_norm = r ** (r / (1 - r)) - r ** (1 / (1 - r))
    area = tau_decay - tau_rise
    return area, peak_norm


def _synaptic_drive(weight_sums, g_bar, tau_rise, tau_decay, driving_force):
    """Synaptic drive in charge units (fC): (nS * ms / unitless) * mV."""
    area, peak_norm = _kernel_area_and_peak_norm(tau_rise, tau_decay)
    return (weight_sums * g_bar * area / peak_norm) * driving_force


def _require_synapse_params(
    name: str,
    kind: str,
    g_bar_by_type,
    tau_rise_by_type,
    tau_decay_by_type,
    E_syn_by_type,
):
    """``(g_bar, tau_rise, tau_decay, E_syn)`` for ``name``; ValueError if missing.

    ``kind`` names the source in the error, e.g. ``"feedforward input type"``.
    """
    if g_bar_by_type is None or name not in g_bar_by_type:
        raise ValueError(
            f"g_bar value for {kind} '{name}' is missing. "
            "This is a critical error - all cell types must have g_bar values."
        )
    if name not in tau_rise_by_type or name not in tau_decay_by_type:
        raise ValueError(
            f"Tau values for {kind} '{name}' are missing. "
            "This is a critical error - all cell types must have tau_rise and tau_decay values."
        )
    if name not in E_syn_by_type:
        raise ValueError(
            f"E_syn value for {kind} '{name}' is missing. "
            "This is a critical error - all cell types must have E_syn values."
        )
    return (
        g_bar_by_type[name],
        tau_rise_by_type[name],
        tau_decay_by_type[name],
        E_syn_by_type[name],
    )


def _synaptic_drive_per_source(
    weights,
    feedforward_weights,
    cell_type_indices,
    input_cell_type_indices,
    cell_type_names,
    input_cell_type_names,
    recurrent_g_bar_by_type,
    feedforward_g_bar_by_type,
    recurrent_tau_rise_by_type,
    recurrent_tau_decay_by_type,
    feedforward_tau_rise_by_type,
    feedforward_tau_decay_by_type,
    recurrent_E_syn_by_type,
    feedforward_E_syn_by_type,
    mean_membrane_potential,
):
    """Per source, the synaptic drive (fC) onto every postsynaptic neuron."""

    def drive(source_rows, name, is_feedforward):
        if is_feedforward:
            params = _require_synapse_params(
                name,
                "feedforward input type",
                feedforward_g_bar_by_type,
                feedforward_tau_rise_by_type,
                feedforward_tau_decay_by_type,
                feedforward_E_syn_by_type,
            )
        else:
            params = _require_synapse_params(
                name,
                "recurrent cell type",
                recurrent_g_bar_by_type,
                recurrent_tau_rise_by_type,
                recurrent_tau_decay_by_type,
                recurrent_E_syn_by_type,
            )
        g_bar, tau_rise, tau_decay, E_syn = params
        return _synaptic_drive(
            source_rows.sum(axis=0),
            g_bar,
            tau_rise,
            tau_decay,
            mean_membrane_potential - E_syn,
        )

    return _per_source(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        drive,
    )


def _input_counts_per_source(*type_args):
    """Per source, the number of nonzero inputs onto every postsynaptic neuron."""
    return _per_source(*type_args, lambda rows, *_: (rows != 0).sum(axis=0))


def _pie_of_means(ax, means, labels, colors, value_fmt, title: str) -> None:
    """Pie chart of ``means`` with each wedge labelled by its value (not percentage)."""

    def autopct(pct):
        return value_fmt(pct * sum(means) / 100.0)

    _, _, autotexts = ax.pie(
        means,
        labels=labels,
        colors=[c + "99" if len(c) == 7 else c for c in colors],  # alpha on hex
        autopct=autopct,
        startangle=90,
        textprops={"size": 9, "weight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")


def _style_per_source_axes(ax, labels, ylabel: str, title: str) -> None:
    """Rotated source labels, y from zero, dashed y grid and no top/right spines."""
    positions = np.arange(len(labels))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _signed_by_source(matrix: NDArray, cell_type_indices: NDArray) -> NDArray:
    """Rows of type 0 (excitatory) kept positive, all other rows negated."""
    return matrix * np.where(cell_type_indices == 0, 1, -1)[:, np.newaxis]


def _square_axes(ax: plt.Axes) -> None:
    """Shrink the axes box to a square (its current height)."""
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.height, pos.height])


def _n_neurons_shown(
    num_neurons: int, num_assemblies, plot_num_assemblies: int, fraction: float
) -> int:
    """Leading block size: ``plot_num_assemblies`` assemblies, or a fraction."""
    if num_assemblies is None:
        return int(num_neurons * fraction)
    return (num_neurons // num_assemblies) * plot_num_assemblies


# ── Matrix views ─────────────────────────────────────────────────────────────


def plot_connectivity_by_cell_type(
    connectome: NDArray[np.float32],
    is_exc: NDArray[np.bool_],
    sign_by_presynaptic: bool = True,
    vmax: float | None = None,
    percentile: float = 99.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot a connectivity matrix partitioned into excitatory / inhibitory blocks.

    Neurons are reordered (on both axes) so excitatory cells come first — the
    excitatory→excitatory block lands top-left and inhibitory→inhibitory
    bottom-right. ``C[i, j]`` is the weight from presynaptic ``i`` to postsynaptic
    ``j``, so rows are sources and columns are targets.

    When ``sign_by_presynaptic`` (default), each row is signed by its source cell
    type (excitatory positive, inhibitory negative) and drawn on a diverging
    blue(I)–white–red(E) colormap built from the project ``EXCITATORY_COLOR`` /
    ``INHIBITORY_COLOR``. Otherwise the raw magnitudes are shown on the same map's
    positive (red) half.

    Args:
        connectome (NDArray[np.float32]): Weight matrix (N x N), ``C[i, j]`` = i→j.
        is_exc (bool array): Per-neuron excitatory mask (N,).
        sign_by_presynaptic (bool): Sign each row by its source type. Defaults to True.
        vmax (float | None): Symmetric colour limit. If None, uses ``percentile`` of
            the nonzero weight magnitudes. Defaults to None.
        percentile (float): Percentile of nonzero magnitudes for the auto colour
            limit. Defaults to 99.0.
        ax (plt.Axes | None): Axes to plot on. If None, creates a new figure.

    Returns:
        plt.Figure | None: Figure object if ``ax`` is None, otherwise None.
    """
    is_exc = np.asarray(is_exc, dtype=bool)
    # Excitatory first (top-left), inhibitory last (bottom-right); same order on both axes.
    order = np.concatenate([np.flatnonzero(is_exc), np.flatnonzero(~is_exc)])
    n_exc = int(is_exc.sum())
    n = is_exc.size

    matrix = connectome[np.ix_(order, order)].astype(np.float64)
    if sign_by_presynaptic:
        signs = np.where(is_exc[order], 1.0, -1.0)
        matrix = matrix * signs[:, np.newaxis]

    if vmax is None:
        magnitude = np.abs(connectome[connectome != 0])
        vmax = float(np.percentile(magnitude, percentile)) if magnitude.size else 1.0
    vmax = vmax or 1.0

    # Diverging map from the project E/I colours: inhibitory (blue) → white → excitatory (red).
    cmap = LinearSegmentedColormap.from_list(
        "ei_diverging", [INHIBITORY_COLOR, "white", EXCITATORY_COLOR]
    )

    fig, ax, owns_figure = figure_and_axes(ax, figsize=(7, 6))
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # Block boundary between E and I on both axes.
    boundary = n_exc - 0.5
    ax.axvline(boundary, color="0.3", lw=0.8)
    ax.axhline(boundary, color="0.3", lw=0.8)

    # Label each block at its centre (presynaptic on y, postsynaptic on x).
    exc_centre, inh_centre = (n_exc - 1) / 2, n_exc + (n - n_exc - 1) / 2
    block_labels = [f"E ({n_exc})", f"I ({n - n_exc})"]
    ax.set_xticks([exc_centre, inh_centre])
    ax.set_yticks([exc_centre, inh_centre])
    ax.set_xticklabels(block_labels)
    ax.set_yticklabels(block_labels)
    color_tick_labels(ax.get_xticklabels(), (EXCITATORY_COLOR, INHIBITORY_COLOR))
    color_tick_labels(ax.get_yticklabels(), (EXCITATORY_COLOR, INHIBITORY_COLOR))

    ax.set_xlabel("postsynaptic")
    ax.set_ylabel("presynaptic")
    ax.set_title("Connectivity matrix (E top-left, I bottom-right)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "signed weight (E +, I −)" if sign_by_presynaptic else "weight",
        rotation=270,
        labelpad=15,
    )

    if owns_figure:
        plt.tight_layout()
    return fig if owns_figure else None


def plot_assembly_graph(
    connectivity_graph: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int | None = None,
    plot_num_assemblies: int = 2,
    heatmap_inches: float = 8.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the assembly graph structure.

    Connections are signed by source cell type (type 0 excitatory, +1; others -1).

    Args:
        connectivity_graph (NDArray[np.float32]): Binary connectivity matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int | None): Total number of assemblies in the network.
            If None, shows 10% of the matrix. Defaults to None.
        plot_num_assemblies (int): Number of assemblies to display (ignored if num_assemblies is None). Defaults to 2.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = connectivity_graph.shape[0]
    n_shown = _n_neurons_shown(num_neurons, num_assemblies, plot_num_assemblies, 0.1)
    signed = _signed_by_source(connectivity_graph, cell_type_indices)

    fig, ax, owns_figure = figure_and_axes(
        ax, figsize=(heatmap_inches * 1.3, heatmap_inches)
    )
    im = ax.imshow(
        signed[:n_shown, :n_shown], cmap="bwr", vmin=-1, vmax=1, aspect="equal"
    )
    _square_axes(ax)  # before adding the colorbar

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["Inhibitory (-1)", "No connection (0)", "Excitatory (+1)"])

    if num_assemblies is None:
        title = f"Assembly Graph Structure (showing {n_shown}/{num_neurons} cells)"
    else:
        title = f"Assembly Graph Structure (showing {plot_num_assemblies}/{num_assemblies} assemblies)"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig if owns_figure else None


def plot_weighted_connectivity(
    weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    num_assemblies: int | None = None,
    plot_num_assemblies: int = 2,
    plot_fraction: float = 0.1,
    heatmap_inches: float = 8.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the weighted connectivity matrix.

    Weights are signed by source cell type (type 0 excitatory, positive; others
    negative) and shown on a fixed [-1, 1] colour scale.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        num_assemblies (int | None): Total number of assemblies in the network.
            If None, uses plot_fraction to determine matrix size. Defaults to None.
        plot_num_assemblies (int): Number of assemblies to display (only used if num_assemblies is not None). Defaults to 2.
        plot_fraction (float): Fraction of neurons to display (only used if num_assemblies is None). Defaults to 0.1.
        heatmap_inches (float): Size of the heatmap in inches. Defaults to 8.0.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    num_neurons = weights.shape[0]
    n_shown = _n_neurons_shown(
        num_neurons, num_assemblies, plot_num_assemblies, plot_fraction
    )
    signed = _signed_by_source(weights, cell_type_indices)

    fig, ax, owns_figure = figure_and_axes(
        ax, figsize=(heatmap_inches * 1.3, heatmap_inches)
    )
    im = ax.imshow(
        signed[:n_shown, :n_shown], cmap="bwr", vmin=-1, vmax=1, aspect="equal"
    )
    _square_axes(ax)  # before adding the colorbar

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, -0.5, 0, 0.5, 1], shrink=0.5)
    cbar.ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"])
    cbar.set_label("Weight", rotation=270, labelpad=15)

    if num_assemblies is None:
        title = f"Weighted Connectivity Matrix (showing {n_shown}/{num_neurons} cells)"
    else:
        title = f"Weighted Connectivity Matrix (showing {plot_num_assemblies}/{num_assemblies} assemblies)"
    ax.set_title(title)
    ax.set_xlabel("Postsynaptic Cells")
    ax.set_ylabel("Presynaptic Cells")
    ax.set_xticks([])
    ax.set_yticks([])

    return fig if owns_figure else None


def plot_feedforward_connectivity(
    feedforward_weights: NDArray[np.float32],
    input_cell_type_indices: NDArray[np.int32],
    plot_fraction: float = 0.1,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    section_title: str | None = None,
    section_title_axes: list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot feedforward connectivity matrix.

    Weight magnitudes on a green [0, 1] scale (feedforward inputs are excitatory).

    Args:
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        input_cell_type_indices (NDArray[np.int32]): Unused; kept for backward
            compatibility.
        plot_fraction (float): Fraction of neurons to display. Defaults to 0.1.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        show_legend (bool): Deprecated, no effect (the colorbar is always shown);
            passing it warns.
        section_title (str | None): Optional section title to add above subplot group in dashboard.
        section_title_axes (list[plt.Axes] | None): List of axes to span title across. Required if section_title provided.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    if show_legend is not True:
        warn_unused("plot_feedforward_connectivity", show_legend=show_legend)
    n_input, n_output = feedforward_weights.shape
    n_input_plot = int(n_input * plot_fraction)
    n_output_plot = int(n_output * plot_fraction)

    plot_width = 14
    plot_height = plot_width * n_input_plot / n_output_plot
    fig, ax, owns_figure = figure_and_axes(ax, figsize=(plot_width, plot_height))

    im = ax.imshow(
        feedforward_weights[:n_input_plot, :n_output_plot],
        cmap="Greens",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.set_label("Weight", rotation=270, labelpad=15)
    ax.set_title(
        f"Feedforward Connectivity Matrix ({n_input_plot}/{n_input} input cells × {n_output_plot}/{n_output} network cells)"
    )
    ax.set_xlabel("Postsynaptic Cells")
    ax.set_ylabel("Presynaptic Input Cells")
    ax.set_xticks([])
    ax.set_yticks([])

    if section_title and section_title_axes:
        # Centred over the first of section_title_axes only
        title_above_axes(
            fig,
            section_title_axes[:1],
            section_title,
            pad=0.035,
            fontsize=12,
            fontweight="bold",
        )

    if owns_figure:
        plt.tight_layout()
    return fig if owns_figure else None


def _mean_charge_per_spike(
    weights_to_targets, g_bar, tau_rise, tau_decay, driving_force
) -> float:
    """Mean charge transfer per spike over all source/target pairs."""
    return _synaptic_drive(
        weights_to_targets, g_bar, tau_rise, tau_decay, driving_force
    ).mean()


def plot_charge_transfer_matrix(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    recurrent_V_mem_by_type: dict[str, float],
    ax: plt.Axes,
) -> None:
    """Plot confusion matrix of mean charge transfer per spike for each input-output pair.

    Missing parameters fall back to defaults (g_bar 1, tau_rise 1 ms, tau_decay 5 ms,
    E_syn 0 mV, V_mem -55 mV).

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types (output/target neurons).
        input_cell_type_names: Names of input cell types (source neurons).
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent source cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward source cell type.
        recurrent_V_mem_by_type: Mean membrane potential for each recurrent target cell type.
        ax: Matplotlib axes to plot on.
    """
    # Rows = output (target) cell types; columns = sources, feedforward then recurrent
    sources = [
        (
            feedforward_weights[input_cell_type_indices == i, :],
            name,
            feedforward_g_bar_by_type,
            feedforward_tau_rise_by_type,
            feedforward_tau_decay_by_type,
            feedforward_E_syn_by_type,
        )
        for i, name in enumerate(input_cell_type_names)
    ] + [
        (
            weights[cell_type_indices == i, :],
            name,
            recurrent_g_bar_by_type,
            recurrent_tau_rise_by_type,
            recurrent_tau_decay_by_type,
            recurrent_E_syn_by_type,
        )
        for i, name in enumerate(cell_type_names)
    ]
    n_outputs = len(cell_type_names)
    n_inputs = len(sources)
    input_labels = list(input_cell_type_names) + list(cell_type_names)
    matrix = np.zeros((n_outputs, n_inputs))

    for out_idx, output_cell_type in enumerate(cell_type_names):
        target_mask = cell_type_indices == out_idx
        if target_mask.sum() == 0:
            continue
        V_mem = recurrent_V_mem_by_type.get(output_cell_type, -55.0)
        for col_idx, (rows, name, g_bars, rises, decays, E_syns) in enumerate(sources):
            weights_to_targets = rows[:, target_mask]
            if weights_to_targets.size == 0:
                continue
            matrix[out_idx, col_idx] = _mean_charge_per_spike(
                weights_to_targets,
                g_bars.get(name, 1.0),
                rises.get(name, 1.0),
                decays.get(name, 5.0),
                V_mem - E_syns.get(name, 0.0),
            )

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.abs(matrix).max(),
        vmax=np.abs(matrix).max(),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Charge Transfer (fC/spike)", fontsize=9)

    ax.set_xticks(np.arange(n_inputs))
    ax.set_yticks(np.arange(n_outputs))
    ax.set_xticklabels(input_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(cell_type_names, fontsize=9)
    for i in range(n_outputs):
        for j in range(n_inputs):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_xlabel("Input Source", fontsize=10)
    ax.set_ylabel("Output Target", fontsize=10)
    ax.set_title(
        "Charge Transfer per Spike (I/O Pairs)", fontsize=11, pad=10, fontweight="bold"
    )


def _summed_input_stats(pre_rows: NDArray, post_mask: NDArray) -> tuple:
    """Mean and std over postsynaptic cells of their summed input from ``pre_rows``,
    or None if there are no such cells."""
    sums_per_post_cell = pre_rows[:, post_mask].sum(axis=0)
    if len(sums_per_post_cell) == 0:
        return None
    return sums_per_post_cell.mean(), sums_per_post_cell.std()


def _plot_stat_matrix(ax, matrix, cell_type_names, col_colors, title: str) -> None:
    """viridis heatmap from 0 to the ceiling of its max, coloured x labels, colorbar."""
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=int(np.ceil(matrix.max())),
    )
    ax.set_xticks(np.arange(len(cell_type_names)))
    ax.set_xticklabels(cell_type_names, rotation=45, ha="right")
    ax.set_xlabel("Postsynaptic (Recurrent) Cell Type")
    ax.set_title(title, fontweight="bold")
    color_tick_labels(ax.get_xticklabels(), col_colors)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("", rotation=270, labelpad=15)


def plot_weight_statistics_matrix(
    recurrent_weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax_mean: plt.Axes | None = None,
    ax_std: plt.Axes | None = None,
) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Plot mean and std of weights by presynaptic to postsynaptic cell type.

    For each (presynaptic type, postsynaptic type) pair the weights are summed over
    presynaptic cells, giving each postsynaptic cell's total input; the matrices show
    the mean and std of that total over postsynaptic cells.

    Args:
        recurrent_weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        ax_mean (plt.Axes | None): Matplotlib axes for mean plot. If None, creates new figure.
        ax_std (plt.Axes | None): Matplotlib axes for std plot. If None, creates new figure.

    Returns:
        tuple[plt.Figure | None, plt.Figure | None]: Matplotlib figure objects if axes are None, otherwise None.
    """
    all_pre_types = list(input_cell_type_names) + list(cell_type_names)
    pre_rows = [
        feedforward_weights[input_cell_type_indices == i, :]
        for i in range(len(input_cell_type_names))
    ] + [
        recurrent_weights[cell_type_indices == i, :]
        for i in range(len(cell_type_names))
    ]

    mean_matrix = np.zeros((len(all_pre_types), len(cell_type_names)))
    std_matrix = np.zeros_like(mean_matrix)
    for i, rows in enumerate(pre_rows):
        for j in range(len(cell_type_names)):
            stats = _summed_input_stats(rows, cell_type_indices == j)
            if stats is not None:
                mean_matrix[i, j], std_matrix[i, j] = stats

    row_colors = [
        _saturated_color(name, is_feedforward=i < len(input_cell_type_names))
        for i, name in enumerate(all_pre_types)
    ]
    col_colors = [_saturated_color(name) for name in cell_type_names]

    fig_mean, ax_mean, owns_mean = figure_and_axes(ax_mean, figsize=(8, 6))
    _plot_stat_matrix(
        ax_mean,
        mean_matrix,
        cell_type_names,
        col_colors,
        "Mean Total Input by Connection Type",
    )
    ax_mean.set_yticks(np.arange(len(all_pre_types)))
    ax_mean.set_yticklabels(all_pre_types)
    ax_mean.set_ylabel("Presynaptic Cell Type")
    color_tick_labels(ax_mean.get_yticklabels(), row_colors)

    fig_std, ax_std, owns_std = figure_and_axes(ax_std, figsize=(8, 6))
    _plot_stat_matrix(
        ax_std,
        std_matrix,
        cell_type_names,
        col_colors,
        "Std of Total Input by Connection Type",
    )
    ax_std.set_yticks([])

    if owns_mean:
        fig_mean.tight_layout()
    if owns_std:
        fig_std.tight_layout()

    return (fig_mean if owns_mean else None, fig_std if owns_std else None)


# ── Per-source summaries ─────────────────────────────────────────────────────


def plot_input_count_pie_chart(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax: plt.Axes,
) -> None:
    """Plot pie chart showing mean input counts from each source.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        ax: Matplotlib axes to plot on.
    """
    counts, labels, colors = _input_counts_per_source(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
    )
    _pie_of_means(
        ax,
        np.array([c.mean() for c in counts]),
        labels,
        colors,
        lambda value: f"{int(round(value)):.0f}",
        "Mean Synapse Input Counts",
    )


def plot_input_count_distribution(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    ax: plt.Axes,
) -> None:
    """Plot box and whisker plots showing distribution of input counts.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        ax: Matplotlib axes to plot on.
    """
    counts, labels, colors = _input_counts_per_source(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
    )
    bp = ax.boxplot(
        counts,
        positions=np.arange(len(labels)),
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops=dict(
            marker="D", markerfacecolor="black", markeredgecolor="black", markersize=6
        ),
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
    _style_per_source_axes(
        ax, labels, "Input Count", "Synapse Input Count Distributions"
    )


def plot_synaptic_drive_pie_chart(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    mean_membrane_potential: float,
    ax: plt.Axes,
) -> None:
    """Plot pie chart showing mean synaptic drive from each source.

    Wedges are the mean absolute drive per postsynaptic neuron. Raises ValueError if
    any source lacks g_bar, tau or E_syn values.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward cell type.
        mean_membrane_potential: Mean membrane potential in mV (for driving force calculation).
        ax: Matplotlib axes to plot on.
    """
    drives, labels, colors = _synaptic_drive_per_source(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        recurrent_g_bar_by_type,
        feedforward_g_bar_by_type,
        recurrent_tau_rise_by_type,
        recurrent_tau_decay_by_type,
        feedforward_tau_rise_by_type,
        feedforward_tau_decay_by_type,
        recurrent_E_syn_by_type,
        feedforward_E_syn_by_type,
        mean_membrane_potential,
    )
    _pie_of_means(
        ax,
        np.array([np.abs(drive).mean() for drive in drives]),  # pies can't be negative
        labels,
        colors,
        lambda value: f"{value:.1f}",
        "Mean Synaptic Drive (Charge)",
    )


def plot_synaptic_drive_distribution(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    recurrent_tau_rise_by_type: dict[str, float],
    recurrent_tau_decay_by_type: dict[str, float],
    feedforward_tau_rise_by_type: dict[str, float],
    feedforward_tau_decay_by_type: dict[str, float],
    recurrent_E_syn_by_type: dict[str, float],
    feedforward_E_syn_by_type: dict[str, float],
    mean_membrane_potential: float,
    ax: plt.Axes,
) -> None:
    """Plot violin plots showing distribution of synaptic drive.

    Raises ValueError if any source lacks g_bar, tau or E_syn values.

    Args:
        weights: Recurrent weight matrix (N x N).
        feedforward_weights: Feedforward weight matrix (M x N).
        cell_type_indices: Array of cell type indices for recurrent neurons.
        input_cell_type_indices: Array of cell type indices for input neurons.
        cell_type_names: Names of recurrent cell types.
        input_cell_type_names: Names of input cell types.
        recurrent_g_bar_by_type: Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type: Total g_bar for each feedforward cell type.
        recurrent_tau_rise_by_type: Rise time constant for each recurrent cell type.
        recurrent_tau_decay_by_type: Decay time constant for each recurrent cell type.
        feedforward_tau_rise_by_type: Rise time constant for each feedforward cell type.
        feedforward_tau_decay_by_type: Decay time constant for each feedforward cell type.
        recurrent_E_syn_by_type: Reversal potential for each recurrent cell type.
        feedforward_E_syn_by_type: Reversal potential for each feedforward cell type.
        mean_membrane_potential: Mean membrane potential in mV (for driving force calculation).
        ax: Matplotlib axes to plot on.
    """
    drives, labels, colors = _synaptic_drive_per_source(
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
        recurrent_g_bar_by_type,
        feedforward_g_bar_by_type,
        recurrent_tau_rise_by_type,
        recurrent_tau_decay_by_type,
        feedforward_tau_rise_by_type,
        feedforward_tau_decay_by_type,
        recurrent_E_syn_by_type,
        feedforward_E_syn_by_type,
        mean_membrane_potential,
    )
    parts = ax.violinplot(
        drives,
        positions=np.arange(len(labels)),
        widths=0.6,
        showmeans=True,
        showmedians=True,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(1.5)
    for partname, color, linewidth in (
        ("cmeans", "black", 2),
        ("cmedians", "darkgray", 2),
        ("cbars", "black", 1.5),
        ("cmaxes", "black", 1.5),
        ("cmins", "black", 1.5),
    ):
        parts[partname].set_color(color)
        parts[partname].set_linewidth(linewidth)
    _style_per_source_axes(
        ax, labels, "Synaptic Drive (fC)", "Synaptic Drive Distributions"
    )


def plot_synaptic_input_histogram(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_g_bar_by_type: dict[str, float],
    feedforward_g_bar_by_type: dict[str, float],
    ax: plt.Axes | list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Plot histogram of total synaptic input to each neuron, separated by presynaptic cell type.

    Shows the distribution of input conductances from each presynaptic cell type
    (both recurrent and feedforward) across all postsynaptic neurons in the network.
    Feedforward input is counted onto excitatory (type 0) neurons only.

    Args:
        weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_g_bar_by_type (dict[str, float]): Total g_bar for each recurrent cell type.
        feedforward_g_bar_by_type (dict[str, float]): Total g_bar for each feedforward cell type.
        ax (plt.Axes | list[plt.Axes] | None): None to create a new figure; a list of
            axes, one per presynaptic type; or one subplot, split into them.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    # Feedforward inputs count as excitatory unless their name says otherwise
    conductances_by_type, titles, excitatory = [], [], []
    for cell_type_idx in np.unique(cell_type_indices):
        cell_type_name = cell_type_names[cell_type_idx]
        rows = weights[cell_type_indices == cell_type_idx, :]
        conductances_by_type.append(
            rows.sum(axis=0) * recurrent_g_bar_by_type[cell_type_name]
        )
        titles.append(f"recurrent / {cell_type_name}")
        excitatory.append(cell_type_sign(cell_type_name) == "excitatory")
    excitatory_mask = cell_type_indices == 0
    for cell_type_idx in np.unique(input_cell_type_indices):
        cell_type_name = input_cell_type_names[cell_type_idx]
        rows = feedforward_weights[input_cell_type_indices == cell_type_idx, :]
        conductances_by_type.append(
            rows[:, excitatory_mask].sum(axis=0)
            * feedforward_g_bar_by_type[cell_type_name]
        )
        titles.append(f"feedforward / {cell_type_name}")
        excitatory.append(cell_type_sign(cell_type_name) != "inhibitory")
    n_types = len(conductances_by_type)

    # Shared log-spaced bins over the positive values of all types
    all_positive = np.concatenate([c[c > 0] for c in conductances_by_type])
    x_min = all_positive.min() if len(all_positive) > 0 else 1e-3
    x_max = all_positive.max() if len(all_positive) > 0 else 1.0
    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), 21)
    max_count = max(
        [0] + [np.histogram(c, bins=log_bins)[0].max() for c in conductances_by_type]
    )

    fig, axes, owns_figure = figure_and_axes_grid(
        ax, n_types, vertical=False, figsize=(6 * n_types, 4)
    )
    for panel, conductances, title, is_excitatory in zip(
        axes, conductances_by_type, titles, excitatory
    ):
        mean_conductance = conductances.mean()
        panel.hist(
            conductances,
            bins=log_bins,
            color=EXCITATORY_COLOR if is_excitatory else INHIBITORY_COLOR,
            edgecolor="black",
            alpha=0.6,
        )
        panel.axvline(
            mean_conductance,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Mean = {mean_conductance:.2f} nS",
        )
        panel.set_title(title)
        panel.set_xlabel("Conductance (nS)")
        panel.set_xscale("log")
        panel.set_xlim(x_min, x_max)
        panel.set_ylim(0, max_count * 1.1)
        panel.set_ylabel("Postsynaptic Cell Count")
        panel.legend()

    suptitle = "Total Synaptic Input Conductance Split by Presynaptic Cell Types"
    if owns_figure:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    title_above_axes(fig, axes, suptitle, pad=0.02, fontsize=12, fontweight="bold")
    return None


def plot_weight_distribution_by_input_type(
    recurrent_weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    connectome_mask: NDArray[np.bool_],
    feedforward_mask: NDArray[np.bool_],
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot violin plot showing distribution of weight values by input cell type.

    Only positive, unmasked weights are shown; sources with none are skipped. The y
    range spans the 1st-99th percentile of all shown weights.

    Args:
        recurrent_weights (NDArray[np.float32]): Recurrent weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        connectome_mask (bool array): Binary mask for valid recurrent connections.
        feedforward_mask (bool array): Binary mask for valid feedforward connections.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.

    Returns:
        plt.Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    fig, ax, owns_figure = figure_and_axes(ax, figsize=(8, 6))

    weight_data, labels, colors = [], [], []
    sources = [
        (feedforward_weights, feedforward_mask, input_cell_type_indices, i, name, True)
        for i, name in enumerate(input_cell_type_names)
    ] + [
        (recurrent_weights, connectome_mask, cell_type_indices, i, name, False)
        for i, name in enumerate(cell_type_names)
    ]
    for matrix, mask, type_indices, type_idx, name, is_feedforward in sources:
        type_mask = type_indices == type_idx
        masked_weights = matrix[type_mask, :] * mask[type_mask, :]
        non_zero_weights = masked_weights[masked_weights > 0]
        if len(non_zero_weights) > 0:
            weight_data.append(non_zero_weights)
            labels.append(name)
            colors.append(_saturated_color(name, is_feedforward))

    if weight_data:
        parts = ax.violinplot(
            weight_data,
            positions=np.arange(len(weight_data)),
            showmeans=True,
            showmedians=False,
            widths=0.7,
            bw_method=0.5,  # smaller = smoother
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        if "cmeans" in parts:
            parts["cmeans"].set_edgecolor("black")
            parts["cmeans"].set_linewidth(1.5)
        for partname in ("cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_visible(False)

        # y range from the 1st to 99th percentile, padded by 5%
        all_weights = np.concatenate(weight_data)
        y_min = np.percentile(all_weights, 1)
        y_max = np.percentile(all_weights, 99)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Weight Value")
    ax.set_title("Weight Distribution by Input Type", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if owns_figure:
        plt.tight_layout()
    return fig if owns_figure else None
