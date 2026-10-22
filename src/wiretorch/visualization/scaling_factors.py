"""Scaling-factor plots: per-pathway trajectories over training and final values.

Scaling factors are logged per pathway (e.g. ``excitatory_to_inhibitory``); see
:data:`SF_PATHWAYS` for the canonical pathways and their display labels. A pathway
whose source is :data:`FEEDFORWARD_SOURCE` stands for the input from whichever
feedforward cell type the log has (any source that is not a recurrent type), so the
same plots work whatever the input population is called.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Source name in :data:`SF_PATHWAYS` that matches any feedforward cell type.
FEEDFORWARD_SOURCE = "feedforward"
#: Cell types that are recurrent (any other source of a pathway is feedforward).
RECURRENT_TYPES = ("excitatory", "inhibitory")

# Canonical pathway definitions shared across all experiments.
# Each entry is ``(csv_key_suffix, display_label)``.
SF_PATHWAYS: list[tuple[str, str]] = [
    ("excitatory_to_excitatory", "Exc → Exc"),
    ("excitatory_to_inhibitory", "Exc → Inh"),
    ("inhibitory_to_excitatory", "Inh → Exc"),
    ("inhibitory_to_inhibitory", "Inh → Inh"),
    ("feedforward_to_excitatory", "FF → Exc"),
    ("feedforward_to_inhibitory", "FF → Inh"),
]

# Short-form aliases used in some older experiments (noisy-weights); a feedforward
# source keeps its own name, e.g. ``<input>_to_exc``.
SF_SHORT_KEYS: dict[str, str] = {
    "exc_to_exc": "excitatory_to_excitatory",
    "exc_to_inh": "excitatory_to_inhibitory",
    "inh_to_exc": "inhibitory_to_excitatory",
    "inh_to_inh": "inhibitory_to_inhibitory",
}
_SHORT_TYPES = {"exc": "excitatory", "inh": "inhibitory"}


def _pathway(name: str) -> tuple[str, str] | None:
    """``(source, target)`` of a logged name, long-form; None if it is not a pathway.

    Accepts ``scaling_factors/<pathway>_value`` and bare (long or short) pathways.
    """
    if name.startswith("scaling_factors/"):
        if not name.endswith("_value"):
            return None
        name = name.removeprefix("scaling_factors/").removesuffix("_value")
    if "/" in name or "_to_" not in name:
        return None
    source, target = name.split("_to_", 1)
    return _SHORT_TYPES.get(source, source), _SHORT_TYPES.get(target, target)


def _matching_key(keys: Iterable[str], key: str) -> str | None:
    """The first of ``keys`` (columns, dict keys) that holds pathway ``key``."""
    source, target = key.split("_to_", 1)
    for candidate in keys:
        pathway = _pathway(candidate)
        if pathway is None or pathway[1] != target:
            continue
        if pathway[0] == source or (
            source == FEEDFORWARD_SOURCE and pathway[0] not in RECURRENT_TYPES
        ):
            return candidate
    return None


def _resolve_sf_column(df: pd.DataFrame, key: str) -> str | None:
    """The column of ``df`` holding scaling-factor pathway ``key``, or None.

    Prefers ``scaling_factors/{key}_value``, then bare ``key``, then any other
    spelling of the same pathway (short forms, the feedforward type's own name).
    """
    for column in (f"scaling_factors/{key}_value", key):
        if column in df.columns:
            return column
    return _matching_key(df.columns, key)


def _pathway_value(values: dict, key: str, default=None):
    """``values[key]``, with ``key`` matched as in :func:`_resolve_sf_column`."""
    if key in values:
        return values[key]
    match = _matching_key(values, key)
    return default if match is None else values[match]


# ── Single-run / multi-seed trajectory plots ─────────────────────────


def plot_sf_trajectories(
    metrics: dict[Any, pd.DataFrame],
    *,
    x_col: str = "epoch",
    x_scale: float = 1.0,
    pathways: list[tuple[str, str]] | None = None,
    colors: dict[Any, Any] | None = None,
    label_fn: Any | None = None,
    target: float = 1.0,
    ylim: tuple[float, float] = (0, 2),
    figsize: tuple[float, float] = (14, 8),
    suptitle: str | None = None,
    mean_std: bool = False,
    xlabel: str = "Epoch",
) -> plt.Figure:
    """Plot scaling-factor trajectories on a 3\u00d72 grid.

    Args:
        metrics: Mapping from run label (seed, param value, ...) to its
            metrics :class:`~pandas.DataFrame`.
        x_col: Column used for the x-axis.
        x_scale: Multiplicative factor applied to x values (e.g. ``1/50``
            to convert steps to epochs).
        pathways: Subset of pathways to plot. Defaults to :data:`SF_PATHWAYS`.
        colors: Mapping from run label to matplotlib color.
        label_fn: Callable ``(key) -> str`` producing the legend label.
        target: Horizontal target line (set *None* to hide).
        ylim: Y-axis limits.
        figsize: Figure size.
        suptitle: Optional super-title.
        mean_std: If *True*, plot mean \u00b1 std across keys instead of
            individual lines.
        xlabel: X-axis label.

    Returns:
        The :class:`~matplotlib.figure.Figure`.
    """
    if pathways is None:
        pathways = SF_PATHWAYS

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    keys = sorted(metrics.keys())

    for idx, (sf_key, sf_label) in enumerate(pathways):
        ax = axes.flat[idx]

        if mean_std:
            # Collect trajectories aligned to first run's x values
            ref_df = metrics[keys[0]]
            ref_col = _resolve_sf_column(ref_df, sf_key)
            if ref_col is None:
                ax.set_title(sf_label)
                continue
            x_vals = ref_df[x_col].values * x_scale
            all_traj = []
            for k in keys:
                col = _resolve_sf_column(metrics[k], sf_key)
                if col is not None:
                    all_traj.append(metrics[k][col].values[: len(x_vals)])
            if not all_traj:
                ax.set_title(sf_label)
                continue
            all_traj = np.array(all_traj)
            mean = all_traj.mean(axis=0)
            std = all_traj.std(axis=0)
            color = "C0" if colors is None else list(colors.values())[0]
            ax.fill_between(x_vals, mean - std, mean + std, alpha=0.3, color=color)
            ax.plot(x_vals, mean, color=color, linewidth=2)
        else:
            for k in keys:
                col = _resolve_sf_column(metrics[k], sf_key)
                if col is None:
                    continue
                x_vals = metrics[k][x_col].values * x_scale
                color = colors[k] if colors else None
                label = label_fn(k) if label_fn else str(k)
                ax.plot(
                    x_vals,
                    metrics[k][col].values,
                    linewidth=1,
                    alpha=0.7,
                    color=color,
                    label=label,
                )

        if target is not None:
            ax.axhline(
                y=target,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Target" if idx == 0 else None,
            )

        ax.set_title(sf_label)
        ax.set_ylabel("Scaling Factor")
        ax.set_xlabel(xlabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Hide unused axes
    for idx in range(len(pathways), len(axes.flat)):
        axes.flat[idx].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Cross-parameter comparison plots ─────────────────────────────────


def plot_sf_vs_parameter(
    param_values: list[float],
    final_sfs: dict[str, NDArray],
    *,
    pathways: list[tuple[str, str]] | None = None,
    xlabel: str = "Parameter",
    figsize: tuple[float, float] = (14, 8),
    suptitle: str | None = None,
    ylim: tuple[float, float] = (0, 2),
    target: float = 1.0,
) -> plt.Figure:
    """Final scaling factors vs a swept parameter (3\u00d72 grid).

    Args:
        param_values: Sorted parameter values (x-axis).
        final_sfs: Mapping ``pathway_key -> array(len(param_values))``.
        pathways: Pathway definitions. Defaults to :data:`SF_PATHWAYS`.
        xlabel: X-axis label.
        figsize: Figure size.
        suptitle: Optional super-title.
        ylim: Y-axis limits.
        target: Horizontal target line.
    """
    if pathways is None:
        pathways = SF_PATHWAYS

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for idx, (sf_key, sf_label) in enumerate(pathways):
        ax = axes.flat[idx]
        values = _pathway_value(final_sfs, sf_key)
        if values is not None:
            ax.plot(param_values, values, "o-", linewidth=1.5, markersize=4)
        if target is not None:
            ax.axhline(y=target, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(sf_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final Scaling Factor")
        if ylim is not None:
            ax.set_ylim(*ylim)

    for idx in range(len(pathways), len(axes.flat)):
        axes.flat[idx].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sf_bar_chart(
    sf_values: dict[str, float] | dict[str, dict[str, float]],
    *,
    pathways: list[tuple[str, str]] | None = None,
    condition_labels: list[str] | None = None,
    condition_colors: list[str] | None = None,
    target: float = 1.0,
    figsize: tuple[float, float] = (8, 4),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Bar chart of final scaling-factor values.

    Supports single-condition (``sf_values`` maps pathway key to float)
    or multi-condition grouped bars (``sf_values`` maps condition label
    to ``{pathway_key: value}`` dict).

    Args:
        sf_values: Single condition: ``{pathway_key: value}``.
            Multi-condition: ``{condition_label: {pathway_key: value}}``.
        pathways: Pathway definitions. Defaults to :data:`SF_PATHWAYS`.
        condition_labels: Explicit ordering of conditions (multi-condition
            only).  Defaults to the order of ``sf_values``.
        condition_colors: One color per condition.
        target: Horizontal target line (set *None* to hide).
        figsize: Figure size.
        title: Optional title.
        ax: Pre-existing axes.

    Returns:
        The Figure, or *None* if *ax* was supplied.
    """
    if pathways is None:
        pathways = SF_PATHWAYS

    pw_keys = [k for k, _ in pathways]
    pw_labels = [lbl for _, lbl in pathways]

    # Detect multi-condition: values are dicts, not scalars
    first_val = next(iter(sf_values.values()))
    multi = isinstance(first_val, dict)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(pw_keys))

    if multi:
        if condition_labels is None:
            condition_labels = list(sf_values.keys())
        n_cond = len(condition_labels)
        if condition_colors is None:
            condition_colors = [f"C{i}" for i in range(n_cond)]
        width = 0.8 / n_cond
        for i, cond in enumerate(condition_labels):
            vals = [_pathway_value(sf_values[cond], k, 0.0) for k in pw_keys]
            ax.bar(
                x + (i - (n_cond - 1) / 2) * width,
                vals,
                width,
                label=cond,
                color=condition_colors[i],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
    else:
        vals = [_pathway_value(sf_values, k, 0.0) for k in pw_keys]
        ax.bar(x, vals, color="C0", alpha=0.8)

    if target is not None:
        ax.axhline(
            y=target,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Target",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pw_labels)
    ax.set_ylabel("Scaling Factor")
    ax.set_ylim(0, None)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True)
    if fig is not None:
        fig.tight_layout()
    return fig
