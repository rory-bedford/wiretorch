"""Training curves: loss and firing-rate trajectories from per-run metrics DataFrames."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_trajectories(
    metrics: dict[Any, pd.DataFrame],
    loss_col: str = "total_loss",
    *,
    x_col: str = "epoch",
    x_scale: float = 1.0,
    colors: dict[Any, Any] | None = None,
    label_fn: Any | None = None,
    ylim: tuple[float, float | None] = (0, None),
    figsize: tuple[float, float] = (8, 4),
    title: str = "Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    reference_lines: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot loss curves from multiple runs.

    Args:
        metrics: Mapping from run label to metrics DataFrame.
        loss_col: Column name for the loss values.
        x_col: Column used for the x-axis.
        x_scale: Multiplicative factor applied to x values.
        colors: Mapping from run label to color.
        label_fn: ``(key) -> str`` producing the legend label.
        ylim: Y-axis limits.
        figsize: Figure size (ignored when *ax* is given).
        title: Axes title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        reference_lines: ``{label: y_value}`` for horizontal reference lines.
        ax: Optional pre-existing axes.

    Returns:
        The Figure, or *None* when *ax* was supplied.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for k, df in sorted(metrics.items()):
        if loss_col not in df.columns:
            continue
        x = df[x_col].values * x_scale
        color = colors[k] if colors else None
        label = label_fn(k) if label_fn else str(k)
        ax.plot(
            x, df[loss_col].values, linewidth=1, alpha=0.7, color=color, label=label
        )

    if reference_lines:
        for lbl, y_val in reference_lines.items():
            ax.axhline(y=y_val, linestyle="--", linewidth=1, alpha=0.6, label=lbl)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_loss_mean_std(
    metrics: dict[Any, pd.DataFrame],
    loss_col: str = "total_loss",
    *,
    x_col: str = "epoch",
    x_scale: float = 1.0,
    color: str = "C0",
    ylim: tuple[float, float | None] = (0, None),
    figsize: tuple[float, float] = (8, 4),
    title: str = "Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    reference_lines: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Loss curve with mean \u00b1 std band across runs.

    Args:
        metrics: Mapping from run label to metrics DataFrame.
        loss_col: Column for loss values.
        x_col: Column for x-axis.
        x_scale: Scale factor for x values.
        color: Band/line color.
        ylim: Y-axis limits.
        figsize: Figure size (ignored when *ax* given).
        title: Axes title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        reference_lines: Horizontal reference lines.
        ax: Optional pre-existing axes.

    Returns:
        The Figure, or *None* when *ax* was supplied.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    keys = sorted(metrics.keys())
    ref = metrics[keys[0]]
    x = ref[x_col].values * x_scale
    n = len(x)
    all_vals = np.array([metrics[k][loss_col].values[:n] for k in keys])
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0)

    ax.fill_between(x, np.maximum(mean - std, 0), mean + std, alpha=0.3, color=color)
    ax.plot(x, mean, color=color, linewidth=2)

    if reference_lines:
        for lbl, y_val in reference_lines.items():
            ax.axhline(y=y_val, linestyle="--", linewidth=1, alpha=0.6, label=lbl)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_firing_rate_trajectories(
    metrics: dict[Any, pd.DataFrame],
    panels: list[tuple[str, str]],
    *,
    x_col: str = "epoch",
    x_scale: float = 1.0,
    colors: dict[Any, Any] | None = None,
    label_fn: Any | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    xlabel: str = "Epoch",
) -> plt.Figure:
    """Multi-panel firing rate trajectories.

    Args:
        metrics: Mapping from run label to DataFrame.
        panels: List of ``(column, panel_title)`` pairs.
        x_col: X-axis column.
        x_scale: Scale factor for x.
        colors: Mapping from run label to color.
        label_fn: ``(key) -> str`` for legend labels.
        figsize: Figure size. Defaults to ``(5*n_cols, 4)``.
        suptitle: Optional super-title.
        xlabel: X-axis label.

    Returns:
        The Figure (panels laid out three per row).
    """
    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for idx, (col, panel_title) in enumerate(panels):
        ax = axes.flat[idx]
        for k, df in sorted(metrics.items()):
            if col not in df.columns:
                continue
            x = df[x_col].values * x_scale
            color = colors[k] if colors else None
            label = label_fn(k) if label_fn else str(k)
            ax.plot(x, df[col].values, linewidth=1, alpha=0.7, color=color, label=label)
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Firing Rate (Hz)")
        ax.legend(fontsize=7)

    for idx in range(n, len(axes.flat)):
        axes.flat[idx].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return fig
