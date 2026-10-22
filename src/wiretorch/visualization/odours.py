"""Deprecated alias of :mod:`wiretorch.visualization.inputs`, kept so existing
experiments keep importing."""

from .inputs import (
    plot_firing_rate_variance_comparison,
    plot_input_firing_rate_histogram,
)

__all__ = ["plot_firing_rate_variance_comparison", "plot_input_firing_rate_histogram"]
