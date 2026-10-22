"""Input-pattern plots: input firing-rate distributions, and how much more network
responses vary across input patterns than across trials of a fixed (control) input."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats

from ._common import figure_and_axes, figure_and_axes_grid, finish_figure
from .colors import TAB_BLUE, TAB_ORANGE


def plot_input_firing_rate_histogram(
    firing_rates: NDArray[np.float64],
    ax: plt.Axes | None = None,
    bins: int = 30,
) -> Figure | None:
    """
    Plot histogram of firing rates for input neurons across input patterns.

    Displays all firing rate values as a histogram with fraction (density) on the y-axis.

    Args:
        firing_rates (NDArray[np.float64]): Firing rates in Hz with shape
            (batch_size, n_patterns, n_input_neurons) or (n_patterns, n_input_neurons).
            A batch dimension is averaged out.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        bins (int): Number of histogram bins. Defaults to 30.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    if firing_rates.ndim == 3:
        firing_rates = firing_rates.mean(axis=0)
    n_patterns, n_input_neurons = firing_rates.shape
    firing_rates_flat = firing_rates.flatten()

    fig, ax, owns_figure = figure_and_axes(ax, figsize=(8, 6))
    ax.hist(
        firing_rates_flat,
        bins=bins,
        alpha=0.7,
        color=TAB_BLUE,
        edgecolor="black",
        linewidth=1.2,
        density=True,
    )
    mean_rate = firing_rates_flat.mean()
    ax.axvline(
        mean_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Mean = {mean_rate:.2f} Hz",
    )

    ax.set_xlabel("Firing Rate (Hz)", fontsize=12)
    ax.set_ylabel("Fraction (Density)", fontsize=12)
    ax.set_title(
        f"Input Neuron Firing Rate Distribution\n({n_input_neurons} neurons × {n_patterns} patterns)",
        fontsize=13,
    )
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    return finish_figure(fig, owns_figure)


def _format_pvalue(p: float) -> tuple[str, str]:
    """``("p = ... (paired t-test)", stars)`` with conventional significance stars."""
    if p < 1e-10:
        p_str, stars = "p < 1e-10", "***"
    elif p < 0.001:
        p_str, stars = f"p = {p:.2e}", "***"
    elif p < 0.01:
        p_str, stars = f"p = {p:.4f}", "**"
    elif p < 0.05:
        p_str, stars = f"p = {p:.4f}", "*"
    else:
        p_str, stars = f"p = {p:.4f}", "n.s."
    return f"{p_str} (paired t-test)", stars


def _paired_violin_panel(
    ax: plt.Axes,
    control: NDArray,
    main: NDArray,
    sampled_indices,
    p_value: float,
    *,
    ylabel: str,
    title: str,
    summary_lines: tuple[str, str],
) -> None:
    """Control vs pattern-modulated violins with paired lines for sampled neurons, a significance
    bracket and a summary box (bottom right)."""
    positions = [1, 2]
    parts = ax.violinplot(
        [control, main],
        positions=positions,
        widths=0.6,
        showmeans=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], (TAB_ORANGE, TAB_BLUE)):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(1.5)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1.5)

    for idx in sampled_indices:
        ax.plot(
            positions,
            [control[idx], main[idx]],
            color="gray",
            alpha=0.5,
            linewidth=0.5,
            zorder=1,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Control (Baseline)", "Pattern-Modulated"], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{title}\n({len(main)} neurons, {len(sampled_indices)} paired samples)",
        fontsize=13,
    )
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Significance bracket (lower for n.s. to avoid overlap)
    p_text, stars = _format_pvalue(p_value)
    y_max = max(control.max(), main.max())
    y_bracket = y_max * (1.05 if stars == "n.s." else 1.1)
    ax.plot(
        [1, 1, 2, 2],
        [y_bracket, y_bracket * 1.02, y_bracket * 1.02, y_bracket],
        "k-",
        linewidth=1.5,
    )
    ax.text(1.5, y_bracket * 1.03, stars, ha="center", va="bottom", fontsize=14)

    ax.text(
        0.98,
        0.02,
        f"{summary_lines[0]}\n{summary_lines[1]}\n{p_text}",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_firing_rate_variance_comparison(
    output_firing_rates: NDArray[np.float64],
    output_firing_rates_control: NDArray[np.float64],
    ax: plt.Axes | list[plt.Axes] | None = None,
    n_spaghetti: int = 50,
) -> Figure | None:
    """
    Compare standard deviation in neuronal firing rates between pattern-modulated and control conditions.

    For each neuron, computes the standard deviation in firing rates across all trials (patterns × repeats)
    and compares it between the experimental (pattern-modulated) and control (baseline) conditions.
    Displays as violin plots with paired spaghetti lines connecting the same neurons.

    Args:
        output_firing_rates (NDArray[np.float64]): Firing rates for main experiment with shape
            (batch_size, n_patterns, n_neurons). Multiple input patterns.
        output_firing_rates_control (NDArray[np.float64]): Firing rates for control experiment
            with shape (batch_size, 1, n_neurons). Single baseline pattern.
        ax (plt.Axes | list[plt.Axes] | None): None for a new figure; two axes (the
            variability and mean panels); or one subplot, split into the two.
        n_spaghetti (int): Number of randomly sampled neurons to show as spaghetti lines
            (drawn with NumPy's global RNG). Defaults to 50.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.

    Notes:
        - Control: std over batch dimension measures trial-to-trial noise with fixed input
        - Patterns: std over patterns per batch (signal variance), averaged across batches
        - This compares noise variance vs signal variance (how much responses differ across inputs)
        - Higher std in the pattern condition indicates neurons respond differently to different inputs
        - Spaghetti lines show paired data (same neuron in both conditions)
    """
    # Control: std over batches (noise with a fixed input), (n_neurons,)
    std_control = np.std(output_firing_rates_control[:, 0, :], axis=0)
    # Patterns: std over patterns within each batch, averaged over batches, (n_neurons,)
    std_main = np.mean(np.std(output_firing_rates, axis=1), axis=0)
    # Mean rates over batch and pattern dimensions, (n_neurons,)
    mean_main = np.mean(output_firing_rates, axis=(0, 1))
    mean_control = np.mean(output_firing_rates_control, axis=(0, 1))
    n_neurons = std_main.shape[0]

    # Paired t-tests on the per-neuron statistics
    _, p_std = stats.ttest_rel(std_control, std_main, alternative="two-sided")
    _, p_mean = stats.ttest_rel(mean_control, mean_main, alternative="two-sided")

    fig, axes, owns_figure = figure_and_axes_grid(
        ax, 2, vertical=False, figsize=(14, 6), share=False
    )

    # Same sampled neurons in both panels
    if n_spaghetti > 0 and n_neurons > 0:
        n_sample = min(n_spaghetti, n_neurons)
        sampled_indices = np.random.choice(n_neurons, size=n_sample, replace=False)
    else:
        sampled_indices = []

    _paired_violin_panel(
        axes[0],
        std_control,
        std_main,
        sampled_indices,
        p_std,
        ylabel="Standard Deviation in Firing Rate (Hz)",
        title="Response Variability (Std Dev)",
        summary_lines=(
            f"Noise (fixed input): {std_control.mean():.2f} Hz",
            f"Signal (varied input): {std_main.mean():.2f} Hz",
        ),
    )
    _paired_violin_panel(
        axes[1],
        mean_control,
        mean_main,
        sampled_indices,
        p_mean,
        ylabel="Mean Firing Rate (Hz)",
        title="Mean Firing Rate",
        summary_lines=(
            f"Mean (control): {mean_control.mean():.2f} Hz",
            f"Mean (pattern-modulated): {mean_main.mean():.2f} Hz",
        ),
    )

    return finish_figure(fig, owns_figure)
