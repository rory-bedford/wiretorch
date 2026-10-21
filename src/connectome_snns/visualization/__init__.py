"""Visualization utilities for connectome-constrained SNN models.

Modules:

- :mod:`.colors` -- the canonical colour palette (import colours from here).
- :mod:`.neuronal_dynamics` -- voltage / current / conductance traces and spike rasters.
- :mod:`.firing_statistics` -- firing-rate, CV, ISI, Fano-factor and PSTH plots.
- :mod:`.connectivity` -- weight matrices and per-input-source summaries.
- :mod:`.dashboards` -- multi-panel connectivity / activity / assembly dashboards.
- :mod:`.inputs` -- input-rate distributions and pattern-vs-control variability.
- :mod:`.scaling_factors`, :mod:`.training_curves`, :mod:`.firing_rate_scatter` --
  training diagnostics and teacher-vs-student comparisons.

Call :func:`use_project_style` to apply the project Matplotlib style. The most used
names are re-exported here; everything else is importable from its module. The
plotting modules (and Matplotlib's pyplot) load on first use, so importing the package
for its colours is cheap.
"""

import importlib
from pathlib import Path as _Path

from .colors import (
    EXCITATORY_COLOR,
    INHIBITORY_COLOR,
    FEEDFORWARD_COLOR,
    CELL_TYPE_COLORS,
    SYNAPSE_COLORS,
    TEACHER_COLOR,
    STUDENT_COLOR,
    NEUTRAL_BAR_COLOR,
    RASTER_BAND_COLOR,
    CMA_PHASE_COLOR,
    CMA_LOSS_COLOR,
    GRADIENT_LOSS_COLOR,
    STRATEGY_COLORS,
    HIGHLIGHT_COLOR,
    SCATTER_NEUTRAL_COLOR,
    BASELINE_BAR_COLOR,
    QUALITATIVE_COLORS,
    FIGURE_BLUE,
    FIGURE_ORANGE,
    FIGURE_TEAL,
    FIGURE_CORAL,
    FIGURE_COLORS,
    FULL_CONNECTOME_COLOR,
    LEARNT_RECURRENCE_COLOR,
    SHUFFLE_WEIGHTS_COLOR,
    CONFIGURATION_MODEL_COLOR,
    FIXED_TOPOLOGY_COLOR,
    OBSERVED_COLOR,
    UNOBSERVED_COLOR,
    FLOOR_COLOR,
    NEURON_REMOVAL_COLOR,
    SYNAPSE_DROPOUT_COLOR,
    DIMENSIONALITY_COLOR,
    SLIDE_BLUE,
    SLIDE_INK,
    SLIDE_RED,
    SLIDE_YELLOW,
    WEIGHT_NOISE_COLOR,
)

#: Name -> the module that defines it, imported on first use.
_LAZY = {
    "plot_membrane_voltages": "neuronal_dynamics",
    "plot_synaptic_currents": "neuronal_dynamics",
    "plot_spike_trains": "neuronal_dynamics",
    "plot_input_spikes": "neuronal_dynamics",
    "plot_network_spikes": "neuronal_dynamics",
    "plot_mitral_cell_spikes": "neuronal_dynamics",
    "plot_dp_network_spikes": "neuronal_dynamics",
    "plot_synaptic_conductances": "neuronal_dynamics",
    "plot_assembly_graph": "connectivity",
    "plot_connectivity_by_cell_type": "connectivity",
    "plot_weighted_connectivity": "connectivity",
    "plot_input_count_pie_chart": "connectivity",
    "plot_input_count_distribution": "connectivity",
    "plot_charge_transfer_matrix": "connectivity",
    "plot_synaptic_drive_pie_chart": "connectivity",
    "plot_synaptic_drive_distribution": "connectivity",
    "plot_synaptic_input_histogram": "connectivity",
    "plot_feedforward_connectivity": "connectivity",
    "plot_firing_rate_distribution": "firing_statistics",
    "plot_input_firing_rate_histogram": "inputs",
    "plot_firing_rate_variance_comparison": "inputs",
    "SF_PATHWAYS": "scaling_factors",
    "SF_SHORT_KEYS": "scaling_factors",
    "plot_sf_trajectories": "scaling_factors",
    "plot_sf_vs_parameter": "scaling_factors",
    "plot_sf_bar_chart": "scaling_factors",
    "plot_loss_trajectories": "training_curves",
    "plot_loss_mean_std": "training_curves",
    "plot_firing_rate_trajectories": "training_curves",
    "plot_firing_rate_scatter": "firing_rate_scatter",
    "plot_r2_vs_parameter": "firing_rate_scatter",
    "plot_r2_comparison_bar": "firing_rate_scatter",
}
#: The submodules, also loaded on first use (``visualization.connectivity``).
_SUBMODULES = (
    "_common",
    "colors",
    "connectivity",
    "dashboards",
    "firing_rate_scatter",
    "firing_statistics",
    "inputs",
    "neuronal_dynamics",
    "odours",
    "scaling_factors",
    "training_curves",
)

_STYLE_PATH = str(_Path(__file__).parent / "connectome_snn.mplstyle")


def use_project_style() -> None:
    """Activate the project matplotlib style sheet."""
    import matplotlib.pyplot as plt

    plt.style.use(["seaborn-v0_8-whitegrid", _STYLE_PATH])


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_LAZY, *_SUBMODULES})


__all__ = [
    "EXCITATORY_COLOR",
    "INHIBITORY_COLOR",
    "FEEDFORWARD_COLOR",
    "CELL_TYPE_COLORS",
    "SYNAPSE_COLORS",
    "TEACHER_COLOR",
    "STUDENT_COLOR",
    "NEUTRAL_BAR_COLOR",
    "RASTER_BAND_COLOR",
    "CMA_PHASE_COLOR",
    "CMA_LOSS_COLOR",
    "GRADIENT_LOSS_COLOR",
    "STRATEGY_COLORS",
    "HIGHLIGHT_COLOR",
    "SCATTER_NEUTRAL_COLOR",
    "BASELINE_BAR_COLOR",
    "QUALITATIVE_COLORS",
    "FIGURE_BLUE",
    "FIGURE_ORANGE",
    "FIGURE_TEAL",
    "FIGURE_CORAL",
    "FIGURE_COLORS",
    "FULL_CONNECTOME_COLOR",
    "LEARNT_RECURRENCE_COLOR",
    "SHUFFLE_WEIGHTS_COLOR",
    "CONFIGURATION_MODEL_COLOR",
    "FIXED_TOPOLOGY_COLOR",
    "OBSERVED_COLOR",
    "UNOBSERVED_COLOR",
    "FLOOR_COLOR",
    "NEURON_REMOVAL_COLOR",
    "SYNAPSE_DROPOUT_COLOR",
    "DIMENSIONALITY_COLOR",
    "SLIDE_BLUE",
    "SLIDE_INK",
    "SLIDE_RED",
    "SLIDE_YELLOW",
    "WEIGHT_NOISE_COLOR",
    "use_project_style",
    "plot_membrane_voltages",
    "plot_synaptic_currents",
    "plot_spike_trains",
    "plot_input_spikes",
    "plot_network_spikes",
    "plot_mitral_cell_spikes",
    "plot_dp_network_spikes",
    "plot_synaptic_conductances",
    "plot_assembly_graph",
    "plot_connectivity_by_cell_type",
    "plot_weighted_connectivity",
    "plot_input_count_pie_chart",
    "plot_input_count_distribution",
    "plot_charge_transfer_matrix",
    "plot_synaptic_drive_pie_chart",
    "plot_synaptic_drive_distribution",
    "plot_synaptic_input_histogram",
    "plot_feedforward_connectivity",
    "plot_firing_rate_distribution",
    "plot_input_firing_rate_histogram",
    "plot_firing_rate_variance_comparison",
    "SF_PATHWAYS",
    "SF_SHORT_KEYS",
    "plot_sf_trajectories",
    "plot_sf_vs_parameter",
    "plot_sf_bar_chart",
    "plot_loss_trajectories",
    "plot_loss_mean_std",
    "plot_firing_rate_trajectories",
    "plot_firing_rate_scatter",
    "plot_r2_vs_parameter",
    "plot_r2_comparison_bar",
]
