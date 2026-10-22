"""Multi-panel dashboards combining the individual plots into single figures.

- :func:`create_connectivity_dashboard`: weighted connectomes plus input-count, weight
  distribution and weight-statistics summaries.
- :func:`create_activity_dashboard`: network raster and PSTH, one example neuron's
  voltage / conductances / currents, and per-cell-type spiking statistics.
- :func:`create_assembly_activity_dashboard`: OU mixing weights next to the
  per-assembly population activity they drive.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wiretorch.configs.network import cell_type_sign

from wiretorch.visualization._common import (
    raster_color_map,
    split_synapses_by_sign,
    title_above_axes,
    warn_unused,
)
from wiretorch.visualization.connectivity import (
    plot_weighted_connectivity,
    plot_input_count_pie_chart,
    plot_weight_distribution_by_input_type,
    plot_weight_statistics_matrix,
    plot_feedforward_connectivity,
)
from wiretorch.visualization.firing_statistics import (
    plot_psth,
    plot_firing_rate_distribution,
    plot_cv_histogram,
    plot_isi_histogram,
    plot_fano_factor_vs_window_size,
    plot_assembly_population_activity,
    plot_ou_process_weights,
)
from wiretorch.visualization.neuronal_dynamics import (
    plot_spike_trains,
    plot_membrane_voltages,
    plot_synaptic_currents,
    plot_synaptic_conductances,
)


# ===================== CONNECTIVITY DASHBOARD =====================


def _scale_by_cell_type(
    weights: NDArray,
    scaling_factors: NDArray,
    source_indices: NDArray,
    n_source_types: int,
    target_indices: NDArray,
    n_target_types: int,
) -> NDArray:
    """Copy of ``weights[source, target]`` with each (source type, target type) block
    scaled by ``scaling_factors[source, target]``."""
    weights = weights.copy()
    for source_idx in range(n_source_types):
        source_mask = source_indices == source_idx
        for target_idx in range(n_target_types):
            target_mask = target_indices == target_idx
            weights[np.ix_(source_mask, target_mask)] *= scaling_factors[
                source_idx, target_idx
            ]
    return weights


def create_connectivity_dashboard(
    weights: NDArray[np.float32],
    feedforward_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    input_cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    connectome_mask: NDArray[np.bool_],
    feedforward_mask: NDArray[np.bool_],
    num_assemblies: int | None = None,
    heatmap_inches: float = 6.0,
    plot_fraction_feedforward: float = 0.1,
    plot_fraction_recurrent: float = 0.1,
    scaling_factors: NDArray[np.float32] | None = None,
    scaling_factors_FF: NDArray[np.float32] | None = None,
) -> plt.Figure:
    """Create a connectivity dashboard showing connectomes and weight distributions.

    Layout: the feedforward and recurrent weighted connectomes stacked on the left; on
    the right a 2x2 grid with the input-count pie chart, the weight violin plot, and
    the mean / std weight-statistics matrices.

    Args:
        weights (NDArray[np.float32]): Weight matrix (N x N).
        feedforward_weights (NDArray[np.float32]): Feedforward weight matrix (M x N).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for recurrent neurons.
        input_cell_type_indices (NDArray[np.int32]): Array of cell type indices for input neurons.
        cell_type_names (list[str]): Names of recurrent cell types.
        input_cell_type_names (list[str]): Names of input cell types.
        connectome_mask (bool array): Binary mask for valid recurrent connections.
        feedforward_mask (bool array): Binary mask for valid feedforward connections.
        num_assemblies (int | None): Number of assemblies in the network. If provided, uses this
            to determine plotting size instead of plot_fraction_recurrent. Defaults to None.
        heatmap_inches (float): Size of heatmaps in inches. Defaults to 6.0.
        plot_fraction_feedforward (float): Fraction of neurons to show in feedforward connectivity. Defaults to 0.1.
        plot_fraction_recurrent (float): Fraction of neurons to show in recurrent connectivity (only used if num_assemblies is None). Defaults to 0.1.
        scaling_factors (NDArray[np.float32] | None): Recurrent scaling factors (n_source_types, n_target_types). If provided, weights will be scaled by cell type.
        scaling_factors_FF (NDArray[np.float32] | None): Feedforward scaling factors (n_source_types, n_target_types). If provided, feedforward weights will be scaled by cell type.

    Returns:
        plt.Figure: Single comprehensive dashboard figure with connectivity plots arranged.
    """
    n_cell_types = len(cell_type_names)
    if scaling_factors is not None:
        weights = _scale_by_cell_type(
            weights,
            scaling_factors,
            cell_type_indices,
            n_cell_types,
            cell_type_indices,
            n_cell_types,
        )
    if scaling_factors_FF is not None:
        feedforward_weights = _scale_by_cell_type(
            feedforward_weights,
            scaling_factors_FF,
            input_cell_type_indices,
            len(input_cell_type_names),
            cell_type_indices,
            n_cell_types,
        )

    fig = plt.figure(figsize=(24, 16))
    gs_main = fig.add_gridspec(
        2,
        2,
        hspace=0.35,
        wspace=0.15,
        height_ratios=[1, 2.4],
        width_ratios=[1, 1],
        left=0.05,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # Left column: feedforward connectome above the (larger) recurrent one
    ax_ff = fig.add_subplot(gs_main[0, 0])
    ax_rec = fig.add_subplot(gs_main[1, 0])
    plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_cell_type_indices,
        plot_fraction=plot_fraction_feedforward,
        ax=ax_ff,
        section_title="Weighted Connectomes",
        section_title_axes=[ax_ff, ax_rec],
    )
    plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
        plot_fraction=plot_fraction_recurrent,
        heatmap_inches=heatmap_inches,
        ax=ax_rec,
    )

    # Right column: 2x2 grid of summaries
    gs_right = gs_main[0:2, 1].subgridspec(
        2, 2, hspace=0.3, wspace=0.15, height_ratios=[1, 1]
    )
    type_args = (
        weights,
        feedforward_weights,
        cell_type_indices,
        input_cell_type_indices,
        cell_type_names,
        input_cell_type_names,
    )

    ax_pie = fig.add_subplot(gs_right[0, 0])
    pos = ax_pie.get_position()  # nudge the pie chart left
    ax_pie.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])
    plot_input_count_pie_chart(*type_args, ax_pie)

    ax_violin = fig.add_subplot(gs_right[0, 1])
    plot_weight_distribution_by_input_type(
        *type_args, connectome_mask, feedforward_mask, ax_violin
    )

    ax_mean = fig.add_subplot(gs_right[1, 0])
    ax_std = fig.add_subplot(gs_right[1, 1])
    plot_weight_statistics_matrix(*type_args, ax_mean, ax_std)

    return fig


# ===================== ACTIVITY DASHBOARD =====================


def _signed_currents(
    recurrent_currents: NDArray,
    feedforward_currents: NDArray,
    leak_currents: NDArray,
    cell_type_names: list[str],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
) -> tuple[NDArray, NDArray]:
    """Total excitatory and inhibitory(+leak) currents, ``(batch, time, neurons)``.

    Both are negated so that depolarising (excitatory) current plots positive and
    hyperpolarising (inhibitory, leak) current negative.
    """
    exc_rec, inh_rec = split_synapses_by_sign(cell_type_names, recurrent_synapse_names)
    exc_ff, inh_ff = split_synapses_by_sign(
        input_cell_type_names, feedforward_synapse_names
    )

    def summed(currents, synapses):
        return currents[:, :, :, [idx for idx, _ in synapses]].sum(axis=-1)

    i_exc = -(
        summed(recurrent_currents, exc_rec) + summed(feedforward_currents, exc_ff)
    )
    i_inh = -(
        summed(recurrent_currents, inh_rec)
        + summed(feedforward_currents, inh_ff)
        + leak_currents
    )
    return i_exc, i_inh


#: Per ``panels`` option: figure size, row height ratios and which rows hold what.
_ACTIVITY_LAYOUTS = {
    "raster": dict(figsize=(16, 8), height_ratios=[1, 1], rows=("spikes", "psth")),
    "neuron": dict(
        figsize=(16, 12),
        height_ratios=[1, 2.0, 1],
        rows=("voltage", "cond", "curr"),
    ),
    "left": dict(
        figsize=(16, 16),
        height_ratios=[1, 1, 1, 2.0, 1],
        rows=("spikes", "psth", "voltage", "cond", "curr"),
    ),
}


def _activity_layout(panels: str):
    """Create the dashboard figure.

    Returns ``(fig, gs_activity, gs_main, rows)``: the grid of activity rows, the outer
    two-column grid (only for ``panels="all"``, else None) and a mapping from row name
    (``spikes``, ``psth``, ``voltage``, ``cond``, ``curr``) to its row index.
    """
    layout = _ACTIVITY_LAYOUTS.get(panels)
    if layout is not None:
        fig = plt.figure(figsize=layout["figsize"])
        gs_activity = fig.add_gridspec(
            len(layout["rows"]),
            1,
            hspace=0.4,
            height_ratios=layout["height_ratios"],
            left=0.07,
            right=0.97,
            top=0.94,
            bottom=0.05,
        )
        rows = {name: i for i, name in enumerate(layout["rows"])}
        return fig, gs_activity, None, rows

    # "all": activity (left, 2/3 width) and spiking statistics (right)
    fig = plt.figure(figsize=(24, 16))
    gs_main = fig.add_gridspec(
        1,
        2,
        hspace=0.3,
        wspace=0.25,
        width_ratios=[2, 1],
        left=0.05,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )
    gs_activity = gs_main[0, 0].subgridspec(
        5, 1, hspace=0.4, height_ratios=[1, 1, 1, 2.0, 1]
    )
    rows = {name: i for i, name in enumerate(_ACTIVITY_LAYOUTS["left"]["rows"])}
    return fig, gs_activity, gs_main, rows


def _sample_raster_neurons(
    output_spikes: NDArray,
    input_spikes: NDArray,
    cell_type_indices: NDArray,
    n_neurons_plot: int,
    dt: float,
    random_seed: int,
) -> tuple[NDArray, NDArray]:
    """Pick feedforward + recurrent neurons for the network raster.

    The two populations are sampled in proportion to their sizes. Recurrent neurons
    are taken per cell type (proportionally), choosing those whose rate is closest to
    their type's mean; input neurons are drawn at random (seeding NumPy's global RNG
    with ``random_seed``).

    Returns:
        ``(spikes, cell_type_indices)`` of the sample, inputs first with index -1.
    """
    n_input_neurons = input_spikes.shape[2]
    n_output_neurons = output_spikes.shape[2]
    n_input_plot = max(
        1, int(n_neurons_plot * n_input_neurons / (n_input_neurons + n_output_neurons))
    )
    n_output_plot = n_neurons_plot - n_input_plot

    duration_s = output_spikes.shape[1] * dt * 1e-3
    output_rates = output_spikes[0].sum(axis=0) / duration_s
    selected = []
    for ct in np.unique(cell_type_indices):
        ct_indices = np.where(cell_type_indices == ct)[0]
        n_ct = max(1, round(n_output_plot * len(ct_indices) / n_output_neurons))
        ct_mean = output_rates[ct_indices].mean()
        closest = ct_indices[np.argsort(np.abs(output_rates[ct_indices] - ct_mean))]
        selected.extend(closest[:n_ct].tolist())
    output_indices = np.array(selected[:n_output_plot])

    np.random.seed(random_seed)
    input_indices = np.random.choice(n_input_neurons, n_input_plot, replace=False)

    spikes = np.concatenate(
        [input_spikes[:, :, input_indices], output_spikes[:, :, output_indices]],
        axis=2,
    )
    indices = np.concatenate(
        [np.full(n_input_plot, -1), cell_type_indices[output_indices]]
    )
    return spikes, indices


def _plot_raster_and_psth(
    fig, gs_activity, rows, inputs: dict, n_neurons_plot, window_size, random_seed
) -> tuple[plt.Axes, plt.Axes]:
    """Network raster (feedforward + recurrent) above the PSTH."""
    output_spikes = inputs["output_spikes"]
    cell_type_names = inputs["cell_type_names"]
    combined_spikes, combined_indices = _sample_raster_neurons(
        output_spikes,
        inputs["input_spikes"],
        inputs["cell_type_indices"],
        n_neurons_plot,
        inputs["dt"],
        random_seed,
    )

    ax_spikes = fig.add_subplot(gs_activity[rows["spikes"], 0])
    plot_spike_trains(
        spikes=combined_spikes,
        dt=inputs["dt"],
        cell_type_indices=combined_indices,
        cell_type_names=["Feedforward"] + cell_type_names,
        n_neurons_plot=n_neurons_plot,
        fraction=1.0,
        random_seed=None,  # already sampled
        title="Network Spike Trains (Feedforward + Recurrent)",
        ylabel="",
        ax=ax_spikes,
        cell_type_colors=raster_color_map(len(cell_type_names)),
    )

    ax_psth = fig.add_subplot(gs_activity[rows["psth"], 0])
    plot_psth(
        spike_trains=output_spikes,
        cell_type_indices=inputs["cell_type_indices"],
        cell_type_names=cell_type_names,
        window_size=window_size,
        dt=inputs["dt"],
        ax=ax_psth,
        input_spike_trains=inputs["input_spikes"],
    )
    return ax_spikes, ax_psth


def _merge_raster_legends(fig, ax_spikes, ax_psth) -> None:
    """Replace the raster and PSTH legends with one figure legend between them."""
    raster_leg = ax_spikes.get_legend()
    psth_leg = ax_psth.get_legend()
    if raster_leg is not None:
        handles = raster_leg.legend_handles
        labels = [t.get_text() for t in raster_leg.get_texts()]
        raster_leg.remove()
    else:
        handles, labels = [], []
    if psth_leg is not None:
        psth_leg.remove()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(0.97, 0.495),
            bbox_transform=fig.transFigure,
            ncol=1,
            frameon=True,
        )


def _select_example_neuron(
    output_spikes: NDArray,
    recurrent_conductances: NDArray,
    cell_type_indices: NDArray,
    cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    dt: float,
    assembly_ids: NDArray | None,
) -> int:
    """Choose a representative excitatory neuron for the single-neuron panels.

    Candidates are excitatory neurons (restricted to assembly 0 if ``assembly_ids`` is
    given) that spiked and received both excitatory and inhibitory recurrent
    conductance. The candidate whose rate is closest to the mean excitatory rate wins;
    failing that, the first excitatory neuron; failing that, neuron 0.
    """
    exc_neuron_indices = [
        i
        for i, cell_type_idx in enumerate(cell_type_indices)
        if cell_type_sign(cell_type_names[cell_type_idx]) == "excitatory"
    ]
    if assembly_ids is not None:
        first_assembly_mask = assembly_ids == 0
        exc_neuron_indices = [
            idx for idx in exc_neuron_indices if first_assembly_mask[idx]
        ]
    if len(exc_neuron_indices) == 0:
        return 0

    n_steps_plot = output_spikes.shape[1]  # the full duration
    g_total = recurrent_conductances.sum(axis=3)  # sum rise/decay components
    exc_synapses, inh_synapses = split_synapses_by_sign(
        cell_type_names, recurrent_synapse_names
    )
    exc_syn_indices = [idx for idx, _ in exc_synapses]
    inh_syn_indices = [idx for idx, _ in inh_synapses]

    def receives(cand, syn_indices):
        if not syn_indices:
            return False
        return g_total[0, -n_steps_plot:, cand, syn_indices].sum() > 0

    candidates = [
        cand
        for cand in exc_neuron_indices
        if output_spikes[0, -n_steps_plot:, cand].sum() > 0
        and receives(cand, exc_syn_indices)
        and receives(cand, inh_syn_indices)
    ]
    if len(candidates) == 0:
        return exc_neuron_indices[0]

    duration_s = output_spikes.shape[1] * dt * 1e-3
    exc_mean_rate = (
        output_spikes[0][:, exc_neuron_indices].sum(axis=0) / duration_s
    ).mean()
    candidate_rates = np.array(
        [output_spikes[0, :, n].sum() / duration_s for n in candidates]
    )
    return candidates[np.argmin(np.abs(candidate_rates - exc_mean_rate))]


def _plot_example_neuron(fig, gs_activity, rows, inputs: dict, neuron: int) -> None:
    """Membrane voltage, conductances (E, I, FF) and input currents of one neuron."""
    dt = inputs["dt"]
    neuron_types = inputs["neuron_types"]

    ax_voltage = fig.add_subplot(gs_activity[rows["voltage"], 0])
    plot_membrane_voltages(
        voltages=inputs["voltages"][:, :, [neuron]],
        spikes=inputs["output_spikes"][:, :, [neuron]],
        neuron_types=neuron_types[[neuron]],
        delta_t=dt,
        neuron_params=inputs["neuron_params"],
        n_neurons_plot=1,
        fraction=1.0,
        y_min=-80.0,
        y_max=-20.0,
        y_tick_step=20.0,
        ax=[ax_voltage],
    )
    ax_voltage.set_title("Membrane Potential", loc="center", pad=10)

    gs_conductances = gs_activity[rows["cond"], 0].subgridspec(3, 1, hspace=0.2)
    conductance_axes = [fig.add_subplot(gs_conductances[i, 0]) for i in range(3)]
    plot_synaptic_conductances(
        # sum the rise/decay components (axis 3) into total conductances
        recurrent_conductances=inputs["recurrent_conductances"].sum(axis=3),
        feedforward_conductances=inputs["feedforward_conductances"].sum(axis=3),
        cell_type_indices=inputs["cell_type_indices"],
        cell_type_names=inputs["cell_type_names"],
        input_cell_type_names=inputs["input_cell_type_names"],
        recurrent_synapse_names=inputs["recurrent_synapse_names"],
        feedforward_synapse_names=inputs["feedforward_synapse_names"],
        dt=dt,
        neuron_id=neuron,
        fraction=1.0,
        ax=conductance_axes,
    )
    conductance_axes[0].set_title("Conductances", loc="center", pad=10)

    i_exc, i_inh = _signed_currents(
        inputs["recurrent_currents"],
        inputs["feedforward_currents"],
        inputs["leak_currents"],
        inputs["cell_type_names"],
        inputs["input_cell_type_names"],
        inputs["recurrent_synapse_names"],
        inputs["feedforward_synapse_names"],
    )
    i_exc, i_inh = i_exc[:, :, [neuron]], i_inh[:, :, [neuron]]
    ax_currents = fig.add_subplot(gs_activity[rows["curr"], 0])
    plot_synaptic_currents(
        I_exc=i_exc,
        I_inh=i_inh,  # includes the leak current
        I_tot=i_exc + i_inh,
        delta_t=dt,
        n_neurons_plot=1,
        fraction=1.0,
        ax=[ax_currents],
    )
    ax_currents.set_title("Input Currents", loc="center", pad=10)


def _plot_spiking_statistics(fig, gs_column, inputs: dict) -> None:
    """Right column: firing-rate, CV, ISI and Fano-factor rows, one panel per type."""
    output_spikes = inputs["output_spikes"]
    common = dict(
        cell_type_indices=inputs["cell_type_indices"],
        cell_type_names=inputs["cell_type_names"],
        dt=inputs["dt"],
    )
    n_cell_types = len(inputs["cell_type_names"])
    gs_statistics = gs_column.subgridspec(4, 1, hspace=0.6)

    def row_axes(row):
        gs_row = gs_statistics[row, 0].subgridspec(1, n_cell_types, wspace=0.3)
        return [fig.add_subplot(gs_row[0, i]) for i in range(n_cell_types)]

    axes = row_axes(0)
    plot_firing_rate_distribution(output_spikes=output_spikes, ax=axes, **common)
    title_above_axes(fig, axes, "Firing Rate Distribution")

    axes = row_axes(1)
    plot_cv_histogram(spike_trains=output_spikes, bins=50, ax=axes, **common)
    title_above_axes(fig, axes, "Coefficient of Variation")

    axes = row_axes(2)
    plot_isi_histogram(spike_trains=output_spikes, bins=50, ax=axes, **common)
    title_above_axes(fig, axes, "Inter-Spike Interval")

    axes = row_axes(3)
    plot_fano_factor_vs_window_size(
        spike_trains=output_spikes,
        window_sizes=np.logspace(0, 3, 20).astype(int),  # 1 to 1000 steps
        ax=axes,
        **common,
    )
    title_above_axes(fig, axes, "Fano Factor")


def create_activity_dashboard(
    output_spikes: NDArray[np.int32],
    input_spikes: NDArray[np.int32],
    cell_type_indices: NDArray[np.int32],
    cell_type_names: list[str],
    dt: float,
    voltages: NDArray[np.float32],
    neuron_types: NDArray[np.int32],
    neuron_params: dict,
    recurrent_currents: NDArray[np.float32],
    feedforward_currents: NDArray[np.float32],
    leak_currents: NDArray[np.float32],
    recurrent_conductances: NDArray[np.float32],
    feedforward_conductances: NDArray[np.float32],
    input_cell_type_names: list[str],
    recurrent_synapse_names: dict[str, list[str]],
    feedforward_synapse_names: dict[str, list[str]],
    window_size: float = 50.0,
    n_neurons_plot: int = 20,
    fraction: float = 1.0,
    random_seed: int = 42,
    assembly_ids: NDArray[np.int32] | None = None,
    neuron_idx: int | None = None,
    panels: str = "all",
) -> plt.Figure:
    """Create a comprehensive activity dashboard with two columns: Activity and Spiking Statistics.

    The left column shows a network raster (feedforward + recurrent), the PSTH and one
    example excitatory neuron's membrane voltage, conductances and currents; the right
    column shows per-cell-type firing-rate, CV, ISI and Fano-factor statistics. The
    selected example neuron index is printed.

    Args:
        output_spikes (NDArray[np.int32]): Output spike array with shape (batch, time, neurons).
        input_spikes (NDArray[np.int32]): Input spike array with shape (batch, time, neurons).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        cell_type_names (list[str]): Names of cell types.
        dt (float): Time step in milliseconds.
        voltages (NDArray[np.float32]): Membrane voltages with shape (batch, time, neurons).
        neuron_types (NDArray[np.int32]): Neuron type indices.
        neuron_params (dict): Dictionary mapping cell type indices to parameters.
        recurrent_currents (NDArray[np.float32]): Recurrent currents with shape (batch, time, neurons, synapses).
        feedforward_currents (NDArray[np.float32]): Feedforward currents with shape (batch, time, neurons, synapses).
        leak_currents (NDArray[np.float32]): Leak currents with shape (batch, time, neurons).
        recurrent_conductances (NDArray[np.float32]): Recurrent conductances with shape (batch, time, neurons, 2, synapses) — the rise/decay component axis (3) is summed internally.
        feedforward_conductances (NDArray[np.float32]): Feedforward conductances with shape (batch, time, neurons, 2, synapses) — the rise/decay component axis (3) is summed internally.
        input_cell_type_names (list[str]): Names of input cell types.
        recurrent_synapse_names (dict[str, list[str]]): Synapse names for each recurrent cell type.
        feedforward_synapse_names (dict[str, list[str]]): Synapse names for each feedforward cell type.
        window_size (float): Window size for PSTH in milliseconds. Defaults to 50.0.
        n_neurons_plot (int): Number of neurons to plot in raster. Defaults to 20.
        fraction (float): Deprecated, no effect (every panel shows the full
            duration); passing it warns.
        random_seed (int): Random seed for sampling input neurons for the raster
            (seeds NumPy's global RNG). Defaults to 42.
        assembly_ids (NDArray[np.int32] | None): Optional array of assembly IDs for each neuron.
            If provided, neuron selection for detailed plots is limited to the first assembly (index 0).
        neuron_idx (int | None): Override auto-selected neuron index for single-neuron plots.
        panels (str): Which panels to render. One of:
            - "all": Full two-column dashboard (rasters + PSTH + single-neuron + statistics). Default.
            - "left": Left column only (rasters + PSTH + single-neuron plots, 16×16 figure).
            - "raster": Rasters + PSTH only (16×8 figure).
            - "neuron": Single-neuron plots only — voltage, conductances, currents (16×12 figure).

    Returns:
        plt.Figure: Matplotlib figure object containing the activity dashboard.
    """
    if fraction != 1.0:
        warn_unused("create_activity_dashboard", fraction=fraction)
    inputs = dict(
        output_spikes=output_spikes,
        input_spikes=input_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=cell_type_names,
        dt=dt,
        voltages=voltages,
        neuron_types=neuron_types,
        neuron_params=neuron_params,
        recurrent_currents=recurrent_currents,
        feedforward_currents=feedforward_currents,
        leak_currents=leak_currents,
        recurrent_conductances=recurrent_conductances,
        feedforward_conductances=feedforward_conductances,
        input_cell_type_names=input_cell_type_names,
        recurrent_synapse_names=recurrent_synapse_names,
        feedforward_synapse_names=feedforward_synapse_names,
    )
    fig, gs_activity, gs_main, rows = _activity_layout(panels)

    if panels != "neuron":
        ax_spikes, ax_psth = _plot_raster_and_psth(
            fig, gs_activity, rows, inputs, n_neurons_plot, window_size, random_seed
        )
    if panels == "raster":
        _merge_raster_legends(fig, ax_spikes, ax_psth)
        return fig

    selected_neuron = _select_example_neuron(
        output_spikes,
        recurrent_conductances,
        cell_type_indices,
        cell_type_names,
        recurrent_synapse_names,
        dt,
        assembly_ids,
    )
    if neuron_idx is not None:
        selected_neuron = neuron_idx
    print(f"Selected neuron index: {selected_neuron}")
    _plot_example_neuron(fig, gs_activity, rows, inputs, selected_neuron)

    if panels in ("left", "neuron"):
        return fig

    _plot_spiking_statistics(fig, gs_main[0, 1], inputs)
    return fig


# ==================== ASSEMBLY ACTIVITY DASHBOARD ====================


def create_assembly_activity_dashboard(
    output_spikes: NDArray[np.int32],
    ou_process_weights: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
    assembly_ids: NDArray[np.int32],
    dt: float,
    excitatory_idx: int = 0,
) -> plt.Figure:
    """Create dashboard combining assembly population activity and OU process weights.

    Args:
        output_spikes (NDArray[np.int32]): Output spike array with shape (batch, time, neurons).
        ou_process_weights (NDArray[np.float32]): OU process weights with shape (batch, time, n_assemblies).
        cell_type_indices (NDArray[np.int32]): Array of cell type indices for each neuron.
        assembly_ids (NDArray[np.int32]): Array of assembly IDs for each neuron.
        dt (float): Time step in milliseconds.
        excitatory_idx (int): Index of excitatory cell type. Defaults to 0.

    Returns:
        plt.Figure: Matplotlib figure object containing the assembly activity dashboard.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Squeeze batch dimension if present: (1, time, patterns) -> (time, patterns)
    if ou_process_weights.ndim == 3:
        ou_process_weights = ou_process_weights[0]

    # One colour per assembly / input pattern, shared by both panels
    n_patterns = ou_process_weights.shape[-1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_patterns))

    plot_ou_process_weights(
        weights=ou_process_weights,
        dt=dt,
        ax=axes[0],
        title="Pattern Mixing Coefficient",
        colors=colors,
    )
    plot_assembly_population_activity(
        spike_trains=output_spikes,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        window_size=200.0,  # smoothing window in ms
        dt=dt,
        excitatory_idx=excitatory_idx,
        title="Assembly Population Activity",
        ax=axes[1],
        colors=colors,
    )

    plt.tight_layout()

    return fig
