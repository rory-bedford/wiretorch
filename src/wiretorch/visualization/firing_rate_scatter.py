"""Teacher-vs-student comparisons: per-neuron firing-rate scatters and R² summaries."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wiretorch.analysis.metrics import r_squared

from .colors import EXCITATORY_COLOR, INHIBITORY_COLOR


def plot_firing_rate_scatter(
    teacher_rates: NDArray,
    student_rates: NDArray,
    cell_type_indices: NDArray,
    *,
    r2: float | None = None,
    cell_type_names: list[str] | None = None,
    cell_type_colors: list[str] | None = None,
    split_panels: bool = False,
    max_rate: float | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str = "Teacher Firing Rate (Hz)",
    ylabel: str = "Student Firing Rate (Hz)",
    marker_size: float = 8,
    alpha: float = 0.5,
    rasterized: bool = False,
    ax: plt.Axes | None = None,
    axes: list[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Scatter plot of teacher vs student per-neuron firing rates.

    ``cell_type_indices`` can be any integer grouping array — not just
    cell types.  For example, combine visibility and cell type into a
    single index (0 = visible-exc, 1 = visible-inh, 2 = hidden-exc,
    3 = hidden-inh) and pass matching *cell_type_names* and
    *cell_type_colors* to get a 4-panel figure with ``split_panels``.

    Args:
        teacher_rates: Per-neuron teacher rates, shape ``(n_neurons,)``.
        student_rates: Per-neuron student rates, shape ``(n_neurons,)``.
        cell_type_indices: Integer group index per neuron.
        r2: Optional R² to annotate on the plot.
        cell_type_names: Names for each group index.  Defaults to
            ``["Excitatory", "Inhibitory"]`` for ≤2 groups, or
            ``["Group 0", "Group 1", ...]`` otherwise.
        cell_type_colors: Colors for each group.  Defaults to canonical
            excitatory/inhibitory colors for ≤2 groups, or ``C0, C1, ...``
            otherwise.
        split_panels: If *True*, plot each group in its own subplot, titled
            with that group's R².
        max_rate: Axis limit. Defaults to 105% of the data max.
        figsize: Figure size.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        marker_size: Scatter marker size.
        alpha: Scatter alpha.
        rasterized: Rasterize scatter points (useful for PDFs).
        ax: Pre-existing axes (only used when *split_panels* is False).
        axes: Pre-existing list of axes (only used when *split_panels* is
            True).  Must contain one axes per unique group.

    Returns:
        The Figure, or *None* if *ax* or *axes* was supplied.
    """
    unique_types = sorted(np.unique(cell_type_indices))
    n_groups = len(unique_types)

    if cell_type_names is None:
        if n_groups <= 2:
            cell_type_names = ["Excitatory", "Inhibitory"][:n_groups]
        else:
            cell_type_names = [f"Group {i}" for i in range(max(unique_types) + 1)]
    if cell_type_colors is None:
        if n_groups <= 2:
            cell_type_colors = [EXCITATORY_COLOR, INHIBITORY_COLOR][:n_groups]
        else:
            cell_type_colors = [f"C{i}" for i in range(max(unique_types) + 1)]
    if max_rate is None:
        max_rate = max(teacher_rates.max(), student_rates.max()) * 1.05

    if split_panels:
        n = len(unique_types)
        if axes is not None:
            axes_arr = axes
            fig = None
        else:
            if figsize is None:
                figsize = (6 * n, 6)
            fig, axes_arr = plt.subplots(1, n, figsize=figsize)
            if n == 1:
                axes_arr = [axes_arr]
        for i, ct in enumerate(unique_types):
            _ax = axes_arr[i]
            mask = cell_type_indices == ct
            _scatter_one(
                _ax,
                teacher_rates[mask],
                student_rates[mask],
                color=cell_type_colors[ct],
                label=cell_type_names[ct],
                max_rate=max_rate,
                marker_size=marker_size,
                alpha=alpha,
                rasterized=rasterized,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            ct_r2 = r_squared(teacher_rates[mask], student_rates[mask])
            _ax.set_title(
                f"{cell_type_names[ct]} (R\u00b2 = {ct_r2:.3f})", fontweight="bold"
            )
        if fig is not None:
            fig.tight_layout()
        return fig

    # Single panel
    fig = None
    if ax is None:
        if figsize is None:
            figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)

    for ct in unique_types:
        mask = cell_type_indices == ct
        name = cell_type_names[ct] if ct < len(cell_type_names) else f"Type {ct}"
        color = cell_type_colors[ct] if ct < len(cell_type_colors) else None
        ax.scatter(
            teacher_rates[mask],
            student_rates[mask],
            s=marker_size,
            alpha=alpha,
            label=name,
            color=color,
            rasterized=rasterized,
        )

    ax.plot([0, max_rate], [0, max_rate], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    elif r2 is not None:
        ax.set_title(f"Firing Rates (R\u00b2 = {r2:.3f})", fontweight="bold")
    ax.legend(fontsize=8)
    return fig


def _scatter_one(
    ax,
    teacher,
    student,
    *,
    color,
    label,
    max_rate,
    marker_size,
    alpha,
    rasterized,
    xlabel,
    ylabel,
):
    """One group's scatter on a square ``[0, max_rate]`` axes with the identity line."""
    ax.scatter(
        teacher, student, s=marker_size, alpha=alpha, color=color, rasterized=rasterized
    )
    ax.plot([0, max_rate], [0, max_rate], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_r2_vs_parameter(
    param_values: list[float],
    r2_dict: dict[str, list[float]],
    *,
    xlabel: str = "Parameter",
    ylabel: str = "R\u00b2",
    figsize: tuple[float, float] = (8, 4),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """R² vs a swept parameter, with optional multiple lines.

    Args:
        param_values: X-axis values.
        r2_dict: ``{line_label: [r2_values]}``.
        xlabel: X label.
        ylabel: Y label.
        figsize: Figure size.
        title: Optional title.
        ax: Pre-existing axes.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, (label, vals) in enumerate(r2_dict.items()):
        ax.plot(
            param_values,
            vals,
            "o-",
            linewidth=1.5,
            markersize=4,
            label=label,
            color=f"C{i}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_r2_comparison_bar(
    condition_labels: list[str],
    r2_dict: dict[str, list[float]],
    *,
    condition_colors: list[str] | None = None,
    ylabel: str = "R\u00b2",
    title: str | None = None,
    ylim: tuple[float, float] = (0, 1.05),
    figsize: tuple[float, float] = (7, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Grouped bar chart comparing R² measures across conditions.

    Args:
        condition_labels: Labels for each condition (x-axis groups).
        r2_dict: ``{measure_name: [r2_per_condition]}``.  Each list
            must have the same length as *condition_labels*.
        condition_colors: One color per condition.  Defaults to
            ``C0, C1, ...``.
        ylabel: Y-axis label.
        title: Optional plot title.
        ylim: Y-axis limits.
        figsize: Figure size.
        ax: Pre-existing axes.

    Returns:
        The Figure, or *None* if *ax* was supplied.
    """
    n_measures = len(r2_dict)
    n_conditions = len(condition_labels)
    if condition_colors is None:
        condition_colors = [f"C{i}" for i in range(n_conditions)]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_measures)
    measure_names = list(r2_dict.keys())
    width = 0.8 / n_conditions

    for i, (cond, color) in enumerate(zip(condition_labels, condition_colors)):
        vals = [r2_dict[m][i] for m in measure_names]
        ax.bar(
            x + (i - (n_conditions - 1) / 2) * width,
            vals,
            width,
            label=cond,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(measure_names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True)
    if ylim[0] < 0:
        ax.axhline(0, color="k", linewidth=0.5)

    if fig is not None:
        fig.tight_layout()
    return fig
