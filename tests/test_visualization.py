"""Smoke and characterisation tests for :mod:`wiretorch.visualization`.

Every public plotting function is called on small, seeded synthetic inputs. Each test
checks that the call succeeds and pins down a few cheap properties of what is drawn
(number of axes, titles, axis labels, number of lines / collections / patches), so a
refactor that changes the drawing fails here.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import wiretorch.visualization as viz  # noqa: E402
from wiretorch.visualization import colors as vcolors  # noqa: E402
from wiretorch.visualization import (  # noqa: E402
    connectivity,
    dashboards,
    firing_statistics,
    neuronal_dynamics,
)

DT = 1.0  # ms
N_STEPS = 1000
N_REC = 40
N_FF = 20
CELL_TYPE_NAMES = ["excitatory", "inhibitory"]
INPUT_CELL_TYPE_NAMES = ["input"]
RECURRENT_SYNAPSE_NAMES = {"excitatory": ["AMPA", "NMDA"], "inhibitory": ["GABA_A"]}
FEEDFORWARD_SYNAPSE_NAMES = {"input": ["AMPA", "NMDA"]}
N_SYN_REC = 3
N_SYN_FF = 2
NEURON_PARAMS = {
    0: {"threshold": -50.0, "rest": -70.0, "name": "excitatory", "sign": 1},
    1: {"threshold": -50.0, "rest": -70.0, "name": "inhibitory", "sign": -1},
}
SYNAPSE_PARAMS = dict(
    recurrent_g_bar_by_type={"excitatory": 1.0, "inhibitory": 2.0},
    feedforward_g_bar_by_type={"input": 1.5},
    recurrent_tau_rise_by_type={"excitatory": 1.0, "inhibitory": 0.5},
    recurrent_tau_decay_by_type={"excitatory": 5.0, "inhibitory": 10.0},
    feedforward_tau_rise_by_type={"input": 1.0},
    feedforward_tau_decay_by_type={"input": 5.0},
    recurrent_E_syn_by_type={"excitatory": 0.0, "inhibitory": -70.0},
    feedforward_E_syn_by_type={"input": 0.0},
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _spikes(n_neurons: int, rate: float = 0.03, seed: int = 0, n_steps=N_STEPS):
    return (_rng(seed).random((1, n_steps, n_neurons)) < rate).astype(np.int32)


def _cell_types(n: int = N_REC, n_exc: int = 30) -> np.ndarray:
    return np.array([0] * n_exc + [1] * (n - n_exc), dtype=np.int32)


def _three_cell_types(n: int = N_REC) -> np.ndarray:
    return np.array([i % 3 for i in range(n)], dtype=np.int32)


def _weights(seed: int = 1):
    rng = _rng(seed)
    mask = rng.random((N_REC, N_REC)) < 0.3
    weights = (rng.random((N_REC, N_REC)) * mask).astype(np.float32)
    ff_mask = rng.random((N_FF, N_REC)) < 0.4
    ff_weights = (rng.random((N_FF, N_REC)) * ff_mask).astype(np.float32)
    return weights, ff_weights, mask, ff_mask


def _hex(color) -> str:
    return to_hex(color, keep_alpha=False)


def _check_fig(fig, n_axes: int | None = None) -> list[plt.Axes]:
    assert isinstance(fig, plt.Figure)
    if n_axes is not None:
        assert len(fig.axes) == n_axes
    return fig.axes


def _activity_inputs(seed: int = 3) -> dict:
    rng = _rng(seed)
    cell_types = _cell_types()
    return dict(
        output_spikes=_spikes(N_REC, rate=0.02, seed=seed),
        input_spikes=_spikes(N_FF, rate=0.03, seed=seed + 1),
        cell_type_indices=cell_types,
        cell_type_names=CELL_TYPE_NAMES,
        dt=DT,
        voltages=(-65 + 5 * rng.standard_normal((1, N_STEPS, N_REC))).astype(
            np.float32
        ),
        neuron_types=cell_types,
        neuron_params=NEURON_PARAMS,
        recurrent_currents=rng.standard_normal((1, N_STEPS, N_REC, N_SYN_REC)).astype(
            np.float32
        ),
        feedforward_currents=rng.standard_normal((1, N_STEPS, N_REC, N_SYN_FF)).astype(
            np.float32
        ),
        leak_currents=rng.standard_normal((1, N_STEPS, N_REC)).astype(np.float32),
        recurrent_conductances=rng.random((1, N_STEPS, N_REC, 2, N_SYN_REC)).astype(
            np.float32
        ),
        feedforward_conductances=rng.random((1, N_STEPS, N_REC, 2, N_SYN_FF)).astype(
            np.float32
        ),
        input_cell_type_names=INPUT_CELL_TYPE_NAMES,
        recurrent_synapse_names=RECURRENT_SYNAPSE_NAMES,
        feedforward_synapse_names=FEEDFORWARD_SYNAPSE_NAMES,
    )


# ── Package surface ──────────────────────────────────────────────────────────


def test_public_names_importable():
    for name in viz.__all__:
        assert hasattr(viz, name), name
    assert callable(dashboards.create_activity_dashboard)
    assert callable(dashboards.create_assembly_activity_dashboard)
    assert callable(dashboards.create_connectivity_dashboard)


def test_legacy_names_still_import():
    from wiretorch.visualization import inputs, odours

    assert viz.plot_mitral_cell_spikes is viz.plot_input_spikes
    assert viz.plot_dp_network_spikes is viz.plot_network_spikes
    assert odours.plot_input_firing_rate_histogram is (
        inputs.plot_input_firing_rate_histogram
    )
    assert odours.plot_firing_rate_variance_comparison is (
        inputs.plot_firing_rate_variance_comparison
    )
    assert vcolors.MITRAL_COLOR == vcolors.FEEDFORWARD_SOURCE_COLOR == "#00AA00"


def test_colour_constants_unchanged():
    assert vcolors.EXCITATORY_COLOR == "#CC3333"
    assert vcolors.INHIBITORY_COLOR == "#3366AA"
    assert vcolors.FEEDFORWARD_COLOR == "#808080"
    assert vcolors.CELL_TYPE_COLORS == ["#CC3333", "#3366AA"]
    assert vcolors.SYNAPSE_COLORS == {
        "AMPA": "#CC3333",
        "NMDA": "#339955",
        "GABA_A": "#3366AA",
        "GABA_B": "#884499",
    }
    assert vcolors.FIGURE_COLORS == ["#457b9d", "#e07b2e", "#2a9d8f", "#e76f51"]
    assert vcolors.QUALITATIVE_COLORS[0] == "#4477AA"
    assert viz.SLIDE_INK == "#263640"
    assert viz.WEIGHT_NOISE_COLOR == viz.FIGURE_CORAL


def test_use_project_style():
    with plt.style.context("default"):
        viz.use_project_style()
        assert plt.rcParams["figure.dpi"] == 120
    plt.rcdefaults()


# ── neuronal_dynamics ────────────────────────────────────────────────────────


def test_plot_membrane_voltages():
    rng = _rng(4)
    voltages = (-65 + 5 * rng.standard_normal((1, 200, 5))).astype(np.float32)
    spikes = _spikes(5, rate=0.05, seed=4, n_steps=200)
    fig = viz.plot_membrane_voltages(
        voltages, spikes, _cell_types(5, 3), DT, NEURON_PARAMS, n_neurons_plot=3
    )
    axes = _check_fig(fig, 3)
    for ax in axes:
        assert ax.get_ylabel() == "Membrane Potential (mV)"
        assert ax.get_ylim() == (-100.0, 0.0)
        assert list(ax.get_yticks()) == [-100.0, -50.0, 0.0]
    assert axes[-1].get_xlabel() == "Time (s)"
    assert axes[0].get_legend() is not None
    assert axes[1].get_legend() is None
    # trace + threshold + rest + one line per spike
    n_spikes = int(spikes[0, :, 0].sum())
    assert len(axes[0].lines) == 3 + n_spikes
    assert _hex(axes[0].lines[1].get_color()) == _hex(viz.EXCITATORY_COLOR)
    assert axes[0].get_xlim() == pytest.approx((0.0, 0.2 + 0.01))


def test_plot_membrane_voltages_onto_axes():
    fig, axes = plt.subplots(2, 1)
    voltages = np.full((1, 100, 2), -60.0, dtype=np.float32)
    spikes = np.zeros((1, 100, 2), dtype=np.int32)
    out = viz.plot_membrane_voltages(
        voltages,
        spikes,
        np.array([0, 1]),
        DT,
        NEURON_PARAMS,
        n_neurons_plot=2,
        fraction=0.5,
        ax=list(axes),
    )
    assert out is None
    assert len(axes[0].lines) == 3
    assert axes[0].get_xlim() == pytest.approx((0.05, 0.1 + 0.01))
    with pytest.raises(ValueError):
        viz.plot_membrane_voltages(
            voltages, spikes, np.array([0, 1]), DT, NEURON_PARAMS, 3, ax=list(axes)
        )


def test_plot_synaptic_currents():
    rng = _rng(5)
    shape = (1, 300, 4)
    i_exc = rng.random(shape).astype(np.float32) * 30
    i_inh = -rng.random(shape).astype(np.float32) * 30
    fig = viz.plot_synaptic_currents(i_exc, i_inh, i_exc + i_inh, DT, n_neurons_plot=2)
    axes = _check_fig(fig, 2)
    for ax in axes:
        assert ax.get_ylabel() == "Input Current (pA)"
        assert len(ax.lines) == 4  # total, exc, inh, zero line
        assert ax.get_ylim() == (-50.0, 50.0)
        assert ax.texts[0].get_text().startswith("mean current = ")
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert labels == ["Total", "Excitatory", "Inhibitory + Leak"]
    assert axes[-1].get_xlabel() == "Time (s)"


def test_plot_spike_trains_single_type():
    spikes = _spikes(10, rate=0.05)
    fig = viz.plot_spike_trains(spikes, DT, cell_type="input", n_neurons_plot=6)
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == "Sample Input Spike Trains"
    assert ax.get_ylabel() == "Neuron ID"
    assert ax.get_xlabel() == "Time (s)"
    assert list(ax.get_yticks()) == list(range(6))
    assert len(ax.collections) == 6  # one EventCollection per neuron
    assert len(ax.patches) == 3  # alternating row shading
    assert ax.get_ylim() == (-0.5, 5.5)


def test_plot_spike_trains_cell_types_with_histogram():
    spikes = _spikes(N_REC, rate=0.05)
    fig = viz.plot_spike_trains(
        spikes,
        DT,
        cell_type_indices=_cell_types(),
        cell_type_names=CELL_TYPE_NAMES,
        n_neurons_plot=8,
        show_spike_histogram=True,
    )
    ax, ax_hist = _check_fig(fig, 2)
    assert ax.get_title() == "Spike Trains (colored by cell type)"
    assert ax.get_ylabel() == ""
    assert len(ax.get_yticks()) == 0
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Inhibitory", "Excitatory"]
    assert ax_hist.get_xlabel() == "Spike Count"
    assert len(ax_hist.patches) == 8


def test_plot_spike_trains_three_types_with_feedforward():
    spikes = _spikes(12, rate=0.05)
    indices = np.array([-1, -1, 0, 0, 1, 1, 2, 2, 0, 1, 2, -1])
    fig, ax = plt.subplots()
    out = viz.plot_spike_trains(
        spikes,
        DT,
        cell_type_indices=indices,
        cell_type_names=["Feedforward", "exc", "inh", "other"],
        n_neurons_plot=12,
        random_seed=None,
        ax=ax,
        title="Custom",
    )
    assert out is None
    assert ax.get_title() == "Custom"
    handles = ax.get_legend().legend_handles
    assert [_hex(h.get_facecolor()) for h in handles] == [
        _hex(plt.get_cmap("tab10")(0)),
        _hex(viz.INHIBITORY_COLOR),
        _hex(viz.EXCITATORY_COLOR),
        _hex("#808080"),
    ]


def test_plot_spike_trains_comparison_mode():
    spikes = _spikes(8, rate=0.05)
    fig = viz.plot_spike_trains(spikes, DT, n_neurons_plot=8, n_compared=2)
    (ax,) = _check_fig(fig, 1)
    assert len(ax.get_yticks()) == 0
    assert len(ax.patches) == 2  # groups 0 and 2 shaded
    assert ax.get_ylim() == pytest.approx((-0.5, 3 * 2.3 + 1 + 0.5))
    assert ax.get_title() == "Sample Spike Trains"


def test_plot_input_and_network_wrappers():
    fig = viz.plot_input_spikes(_spikes(N_FF), DT, n_neurons_plot=5)
    assert fig.axes[0].get_title() == "Sample Input Spike Trains"
    assert tuple(fig.get_size_inches()) == (12.0, 4.0)
    fig = viz.plot_network_spikes(
        _spikes(N_REC), _cell_types(), CELL_TYPE_NAMES, DT, n_neurons_plot=10
    )
    assert fig.axes[0].get_title() == "Spike Trains (colored by cell type)"
    assert tuple(fig.get_size_inches()) == (12.0, 6.0)
    assert len(fig.axes[0].collections) == 10


def test_plot_synaptic_conductances():
    rng = _rng(6)
    rec = rng.random((1, 300, 5, N_SYN_REC)).astype(np.float32)
    ff = rng.random((1, 300, 5, N_SYN_FF)).astype(np.float32)
    fig = viz.plot_synaptic_conductances(
        rec,
        ff,
        _cell_types(5, 3),
        CELL_TYPE_NAMES,
        INPUT_CELL_TYPE_NAMES,
        RECURRENT_SYNAPSE_NAMES,
        FEEDFORWARD_SYNAPSE_NAMES,
        DT,
        neuron_id=1,
    )
    axes = _check_fig(fig, 3)
    assert [len(ax.lines) for ax in axes] == [2, 1, 2]
    assert [ax.texts[0].get_text() for ax in axes] == [
        "Excitatory",
        "Inhibitory",
        "Feedforward",
    ]
    assert axes[1].get_ylabel() == "Conductance (µS)"
    assert axes[2].get_xlabel() == "Time (s)"
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert axes[1].get_ylim() == (0.0, 10.0)
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert labels == ["AMPA", "NMDA", "GABA_A"]
    assert _hex(axes[0].lines[1].get_color()) == _hex(viz.SYNAPSE_COLORS["NMDA"])


def test_plot_synaptic_conductances_unknown_synapse_onto_axes():
    rng = _rng(7)
    rec = rng.random((1, 100, 3, 2)).astype(np.float32)
    ff = rng.random((1, 100, 3, 1)).astype(np.float32)
    fig, axes = plt.subplots(3, 1)
    out = viz.plot_synaptic_conductances(
        rec,
        ff,
        np.array([0, 1, 0]),
        CELL_TYPE_NAMES,
        INPUT_CELL_TYPE_NAMES,
        {"excitatory": ["AMPA"], "inhibitory": ["GABA_B"]},
        {"input": ["FOO"]},
        DT,
        ax=list(axes),
    )
    assert out is None
    assert _hex(axes[2].lines[0].get_color()) == _hex(plt.colormaps["Set1"](0))


def test_round_to_nice_limit():
    f = neuronal_dynamics._round_to_nice_limit
    assert [f(v) for v in (0, 0.7, 1.5, 3, 7, 23)] == [1.0, 1.0, 2.0, 5.0, 10.0, 50.0]


# ── connectivity ─────────────────────────────────────────────────────────────


def test_plot_connectivity_by_cell_type():
    weights, *_ = _weights()
    fig = viz.plot_connectivity_by_cell_type(weights, _cell_types() == 0)
    ax, cbar = _check_fig(fig, 2)
    assert ax.get_xlabel() == "postsynaptic"
    assert ax.get_ylabel() == "presynaptic"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["E (30)", "I (10)"]
    assert cbar.get_ylabel() == "signed weight (E +, I −)"
    assert len(ax.images) == 1
    fig2 = viz.plot_connectivity_by_cell_type(
        weights, _cell_types() == 0, sign_by_presynaptic=False, vmax=0.5
    )
    assert fig2.axes[1].get_ylabel() == "weight"
    assert fig2.axes[0].images[0].get_clim() == (-0.5, 0.5)


def test_plot_assembly_graph():
    _, _, mask, _ = _weights()
    fig = viz.plot_assembly_graph(mask.astype(np.float32), _cell_types(), 4)
    ax, cbar = _check_fig(fig, 2)
    assert ax.get_title() == "Assembly Graph Structure (showing 2/4 assemblies)"
    assert ax.images[0].get_array().shape == (20, 20)
    assert [t.get_text() for t in cbar.get_yticklabels()] == [
        "Inhibitory (-1)",
        "No connection (0)",
        "Excitatory (+1)",
    ]
    fig = viz.plot_assembly_graph(mask.astype(np.float32), _cell_types())
    assert fig.axes[0].get_title() == ("Assembly Graph Structure (showing 4/40 cells)")


def test_plot_weighted_connectivity():
    weights, *_ = _weights()
    fig = viz.plot_weighted_connectivity(weights, _cell_types(), plot_fraction=0.25)
    ax, cbar = _check_fig(fig, 2)
    assert ax.get_title() == "Weighted Connectivity Matrix (showing 10/40 cells)"
    assert ax.get_xlabel() == "Postsynaptic Cells"
    assert ax.get_ylabel() == "Presynaptic Cells"
    assert cbar.get_ylabel() == "Weight"
    fig = viz.plot_weighted_connectivity(weights, _cell_types(), num_assemblies=4)
    assert "2/4 assemblies" in fig.axes[0].get_title()


def test_plot_feedforward_connectivity():
    _, ff_weights, *_ = _weights()
    fig = viz.plot_feedforward_connectivity(ff_weights, np.zeros(N_FF, int), 0.5)
    ax, cbar = _check_fig(fig, 2)
    assert ax.get_title() == (
        "Feedforward Connectivity Matrix (10/20 input cells × 20/40 network cells)"
    )
    assert ax.get_ylabel() == "Presynaptic Input Cells"
    assert ax.images[0].get_array().shape == (10, 20)
    assert len(fig.texts) == 0


def _connectivity_args():
    weights, ff_weights, *_ = _weights()
    return (
        weights,
        ff_weights,
        _cell_types(),
        np.zeros(N_FF, int),
        CELL_TYPE_NAMES,
        INPUT_CELL_TYPE_NAMES,
    )


def test_plot_input_count_pie_chart():
    fig, ax = plt.subplots()
    assert viz.plot_input_count_pie_chart(*_connectivity_args(), ax) is None
    assert ax.get_title() == "Mean Synapse Input Counts"
    wedges = [p for p in ax.patches]
    assert len(wedges) == 3
    assert [_hex(w.get_facecolor()) for w in wedges] == [
        "#00aa00",
        _hex(viz.EXCITATORY_COLOR),
        _hex(viz.INHIBITORY_COLOR),
    ]
    assert wedges[0].get_facecolor()[3] == pytest.approx(0x99 / 255)
    # Matplotlib orders the labels and percentages differently between versions.
    texts = [t.get_text() for t in ax.texts]
    assert [t for t in texts if not t.isdigit()] == [
        "input",
        "excitatory",
        "inhibitory",
    ]
    assert sum(t.isdigit() for t in texts) == 3


def test_plot_input_count_distribution():
    fig, ax = plt.subplots()
    assert viz.plot_input_count_distribution(*_connectivity_args(), ax) is None
    assert ax.get_title() == "Synapse Input Count Distributions"
    assert ax.get_ylabel() == "Input Count"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "input",
        "excitatory",
        "inhibitory",
    ]
    assert len(ax.patches) == 3
    assert ax.get_ylim()[0] == 0


def test_plot_charge_transfer_matrix():
    fig, ax = plt.subplots()
    params = dict(SYNAPSE_PARAMS)
    params.pop("recurrent_E_syn_by_type")
    viz.plot_charge_transfer_matrix(
        *_connectivity_args(),
        recurrent_E_syn_by_type={"excitatory": 0.0, "inhibitory": -70.0},
        recurrent_V_mem_by_type={"excitatory": -60.0},
        ax=ax,
        **params,
    )
    assert ax.get_title() == "Charge Transfer per Spike (I/O Pairs)"
    assert ax.images[0].get_array().shape == (2, 3)
    assert len(ax.texts) == 6
    assert fig.axes[1].get_ylabel() == "Mean Charge Transfer (fC/spike)"


def test_plot_synaptic_drive_pie_chart():
    fig, ax = plt.subplots()
    viz.plot_synaptic_drive_pie_chart(
        *_connectivity_args(), **SYNAPSE_PARAMS, mean_membrane_potential=-60.0, ax=ax
    )
    assert ax.get_title() == "Mean Synaptic Drive (Charge)"
    assert len(ax.patches) == 3
    texts = [t.get_text() for t in ax.texts]
    assert sum("." in t for t in texts) == 3  # the percentages, in any order
    with pytest.raises(ValueError, match="g_bar"):
        bad = dict(SYNAPSE_PARAMS, feedforward_g_bar_by_type={})
        viz.plot_synaptic_drive_pie_chart(
            *_connectivity_args(), **bad, mean_membrane_potential=-60.0, ax=ax
        )


def test_plot_synaptic_drive_distribution():
    fig, ax = plt.subplots()
    viz.plot_synaptic_drive_distribution(
        *_connectivity_args(), **SYNAPSE_PARAMS, mean_membrane_potential=-60.0, ax=ax
    )
    assert ax.get_title() == "Synaptic Drive Distributions"
    assert ax.get_ylabel() == "Synaptic Drive (fC)"
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(bodies) == 3
    with pytest.raises(ValueError, match="E_syn"):
        bad = dict(SYNAPSE_PARAMS, recurrent_E_syn_by_type={})
        viz.plot_synaptic_drive_distribution(
            *_connectivity_args(), **bad, mean_membrane_potential=-60.0, ax=ax
        )


def test_plot_synaptic_input_histogram():
    args = _connectivity_args()
    fig = viz.plot_synaptic_input_histogram(
        *args,
        recurrent_g_bar_by_type=SYNAPSE_PARAMS["recurrent_g_bar_by_type"],
        feedforward_g_bar_by_type=SYNAPSE_PARAMS["feedforward_g_bar_by_type"],
    )
    axes = _check_fig(fig, 3)
    assert [ax.get_title() for ax in axes] == [
        "recurrent / excitatory",
        "recurrent / inhibitory",
        "feedforward / input",
    ]
    assert fig._suptitle.get_text() == (
        "Total Synaptic Input Conductance Split by Presynaptic Cell Types"
    )
    assert axes[0].get_xscale() == "log"
    assert len(axes[0].patches) == 20
    assert _hex(axes[1].patches[0].get_facecolor()) == _hex(viz.INHIBITORY_COLOR)
    fig2, axes2 = plt.subplots(1, 3)
    out = viz.plot_synaptic_input_histogram(
        *args,
        recurrent_g_bar_by_type=SYNAPSE_PARAMS["recurrent_g_bar_by_type"],
        feedforward_g_bar_by_type=SYNAPSE_PARAMS["feedforward_g_bar_by_type"],
        ax=list(axes2),
    )
    assert out is None
    assert len(fig2.texts) == 1
    # One subplot is split in place into the three panels.
    fig3, ax3 = plt.subplots()
    out = viz.plot_synaptic_input_histogram(
        *args,
        recurrent_g_bar_by_type=SYNAPSE_PARAMS["recurrent_g_bar_by_type"],
        feedforward_g_bar_by_type=SYNAPSE_PARAMS["feedforward_g_bar_by_type"],
        ax=ax3,
    )
    assert out is None
    assert [ax.get_title() for ax in fig3.axes] == [ax.get_title() for ax in axes]


def test_plot_weight_distribution_by_input_type():
    weights, ff_weights, mask, ff_mask = _weights()
    fig = connectivity.plot_weight_distribution_by_input_type(
        weights,
        ff_weights,
        _cell_types(),
        np.zeros(N_FF, int),
        CELL_TYPE_NAMES,
        INPUT_CELL_TYPE_NAMES,
        mask,
        ff_mask,
    )
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == "Weight Distribution by Input Type"
    assert ax.get_ylabel() == "Weight Value"
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert [_hex(b.get_facecolor()[0]) for b in bodies] == [
        "#00aa00",
        "#dd0000",
        "#0000dd",
    ]


def test_plot_weight_statistics_matrix():
    fig_mean, fig_std = connectivity.plot_weight_statistics_matrix(
        *_connectivity_args()
    )
    ax_mean = fig_mean.axes[0]
    ax_std = fig_std.axes[0]
    assert ax_mean.get_title() == "Mean Total Input by Connection Type"
    assert ax_std.get_title() == "Std of Total Input by Connection Type"
    assert ax_mean.images[0].get_array().shape == (3, 2)
    assert [_hex(t.get_color()) for t in ax_mean.get_yticklabels()] == [
        "#00aa00",
        "#dd0000",
        "#0000dd",
    ]
    assert len(ax_std.get_yticks()) == 0
    for fig in (fig_mean, fig_std):  # both get their tight layout
        assert fig.subplotpars.left != plt.rcParams["figure.subplot.left"]
    fig, (a, b) = plt.subplots(1, 2)
    assert connectivity.plot_weight_statistics_matrix(*_connectivity_args(), a, b) == (
        None,
        None,
    )


# ── firing_statistics ────────────────────────────────────────────────────────


def test_plot_fano_factor_vs_window_size():
    spikes = _spikes(N_REC, rate=0.05)
    fig = firing_statistics.plot_fano_factor_vs_window_size(
        spikes, [5, 10, 50, 2000], _cell_types(), CELL_TYPE_NAMES, DT
    )
    axes = _check_fig(fig, 2)
    assert [ax.get_title() for ax in axes] == ["Excitatory", "Inhibitory"]
    assert axes[0].get_xlabel() == "Window Size (s)"
    assert axes[0].get_ylabel() == "Mean Fano Factor"
    assert axes[0].get_xscale() == "log"
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert _hex(axes[0].lines[0].get_color()) == _hex(viz.EXCITATORY_COLOR)


def test_plot_fano_factor_no_spikes_three_types():
    spikes = np.zeros((1, 200, 9), dtype=np.int32)
    fig = firing_statistics.plot_fano_factor_vs_window_size(
        spikes, [5, 10], _three_cell_types(9), ["a", "b", "c"], DT
    )
    axes = _check_fig(fig, 3)
    assert axes[2].texts[0].get_text() == "No spikes detected"


def test_plot_cv_histogram():
    spikes = _spikes(N_REC, rate=0.05)
    fig = firing_statistics.plot_cv_histogram(
        spikes, _cell_types(), CELL_TYPE_NAMES, DT, bins=10
    )
    axes = _check_fig(fig, 2)
    assert axes[0].get_xlabel() == "Coefficient of Variation (CV)"
    assert len(axes[0].patches) == 10
    labels = [t.get_text() for t in axes[1].get_legend().get_texts()]
    assert labels[0] == "n=10"
    assert labels[1].startswith("Mean: ")
    assert axes[0].get_xlim() == axes[1].get_xlim()


def test_plot_cv_histogram_three_types_onto_axes():
    spikes = _spikes(9, rate=0.05)
    spikes[:, :, 2] = 0
    spikes[:, :, 5] = 0
    spikes[:, :, 8] = 0
    fig, axes = plt.subplots(1, 3)
    out = firing_statistics.plot_cv_histogram(
        spikes, _three_cell_types(9), ["a", "b", "c"], DT, ax=list(axes)
    )
    assert out is None
    assert axes[2].texts[0].get_text() == "No valid CV values"
    assert _hex(axes[0].patches[0].get_facecolor()) == _hex(viz.EXCITATORY_COLOR)


def test_plot_isi_histogram():
    spikes = _spikes(N_REC, rate=0.05)
    fig = firing_statistics.plot_isi_histogram(
        spikes, _cell_types(), CELL_TYPE_NAMES, DT, bins=15
    )
    axes = _check_fig(fig, 2)
    assert axes[0].get_xlabel() == "Inter-Spike Interval (s)"
    assert len(axes[0].patches) == 15
    assert axes[0].get_xlim()[0] == 0
    assert axes[1].get_legend().get_texts()[1].get_text().endswith(" s")


def test_plot_isi_histogram_no_spikes():
    spikes = np.zeros((1, 100, 6), dtype=np.int32)
    fig = firing_statistics.plot_isi_histogram(
        spikes, _three_cell_types(6), ["a", "b", "c"], DT
    )
    assert fig.axes[1].texts[0].get_text() == "No ISIs found"


def test_plot_psth():
    spikes = _spikes(N_REC, rate=0.05)
    fig = firing_statistics.plot_psth(
        spikes,
        _cell_types(),
        CELL_TYPE_NAMES,
        window_size=20.0,
        dt=DT,
        input_spike_trains=_spikes(N_FF, seed=9),
    )
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == "Population Activity"
    assert ax.get_ylabel() == "Firing Rate (Hz)"
    assert len(ax.lines) == 3
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Excitatory (n=30)", "Inhibitory (n=10)", "Feedforward (n=20)"]
    assert _hex(ax.lines[2].get_color()) == "#808080"
    assert ax.get_xlim() == pytest.approx((0.0, 0.999 + 0.01))


def test_plot_firing_rate_distribution():
    spikes = _spikes(N_REC, rate=0.01)
    spikes[:, :, 0] = 0
    fig = viz.plot_firing_rate_distribution(spikes, _cell_types(), CELL_TYPE_NAMES, DT)
    axes = _check_fig(fig, 2)
    assert [ax.get_title() for ax in axes] == ["Excitatory", "Inhibitory"]
    assert axes[0].get_xscale() == "log"
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert labels[0] == "n=30"
    assert labels[2] == "Silent: 1"
    labels = [t.get_text() for t in axes[1].get_legend().get_texts()]
    assert labels[2] == "Silent: 0"


def test_plot_firing_rate_distribution_three_types_onto_axes():
    fig, axes = plt.subplots(1, 3)
    out = viz.plot_firing_rate_distribution(
        _spikes(9, rate=0.02), _three_cell_types(9), ["a", "b", "c"], DT, ax=list(axes)
    )
    assert out is None
    assert _hex(axes[2].patches[0].get_facecolor()) == _hex(plt.get_cmap("tab10")(0))


def test_plot_assembly_population_activity():
    spikes = _spikes(N_REC, rate=0.05)
    assembly_ids = np.array([i % 3 for i in range(N_REC)])
    assembly_ids[-1] = -1
    fig = firing_statistics.plot_assembly_population_activity(
        spikes, _cell_types(), assembly_ids, window_size=50.0, dt=DT
    )
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == "Excitatory Assembly Population Activity"
    assert len(ax.lines) == 3
    assert [line.get_label() for line in ax.lines] == [
        "Assembly 0",
        "Assembly 1",
        "Assembly 2",
    ]
    with pytest.raises(ValueError):
        firing_statistics.plot_assembly_population_activity(
            spikes, _cell_types(), np.full(N_REC, -1), window_size=50.0
        )


def test_plot_cross_correlation_scatter():
    a = _spikes(10, rate=0.05, seed=1)
    b = _spikes(10, rate=0.05, seed=2)
    fig = firing_statistics.plot_cross_correlation_scatter(
        a, b, window_size=0.1, dt=DT, cell_indices=np.arange(5)
    )
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title().startswith("Cross-Correlation Scatter Plot (r=")
    assert len(ax.collections[0].get_offsets()) == 10 * 5
    assert "Cells: 5" in ax.texts[0].get_text()
    with pytest.raises(ValueError):
        firing_statistics.plot_cross_correlation_scatter(a, b, window_size=10.0)


def test_plot_ou_process_weights():
    weights = _rng(2).random((200, 4))
    fig, ax = plt.subplots()
    out = firing_statistics.plot_ou_process_weights(weights, DT, ax=ax, xlim=(0, 0.1))
    assert out is ax
    assert len(ax.lines) == 4
    assert ax.get_title() == "Latent Trajectories"
    assert ax.get_xlim() == (0, 0.1)


# ── inputs ───────────────────────────────────────────────────────────────────


def test_plot_input_firing_rate_histogram():
    rates = _rng(3).random((2, 4, 25)) * 20
    fig = viz.plot_input_firing_rate_histogram(rates, bins=12)
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == (
        "Input Neuron Firing Rate Distribution\n(25 neurons × 4 patterns)"
    )
    assert len(ax.patches) == 12
    assert _hex(ax.patches[0].get_facecolor()) == "#1f77b4"


def test_plot_firing_rate_variance_comparison():
    rng = _rng(4)
    main = rng.random((3, 5, 30)) * 10
    control = rng.random((3, 1, 30)) * 10
    np.random.seed(0)
    fig = viz.plot_firing_rate_variance_comparison(main, control, n_spaghetti=10)
    ax_std, ax_mean = _check_fig(fig, 2)
    assert ax_std.get_ylabel() == "Standard Deviation in Firing Rate (Hz)"
    assert ax_mean.get_ylabel() == "Mean Firing Rate (Hz)"
    assert len(ax_std.lines) == 10 + 1
    bodies = [c for c in ax_std.collections if isinstance(c, PolyCollection)]
    assert [_hex(b.get_facecolor()[0]) for b in bodies] == ["#ff7f0e", "#1f77b4"]
    assert "(paired t-test)" in ax_mean.texts[-1].get_text()
    # On given axes: a list of two, or one subplot split into two.
    fig2, given = plt.subplots(1, 2)
    assert (
        viz.plot_firing_rate_variance_comparison(main, control, ax=list(given)) is None
    )
    assert given[1].get_ylabel() == "Mean Firing Rate (Hz)"
    fig3, single = plt.subplots()
    assert viz.plot_firing_rate_variance_comparison(main, control, ax=single) is None
    assert [ax.get_ylabel() for ax in fig3.axes] == [
        "Standard Deviation in Firing Rate (Hz)",
        "Mean Firing Rate (Hz)",
    ]


def test_deprecated_plot_arguments_warn():
    i_exc = _rng(9).random((1, 50, 2))
    with pytest.warns(DeprecationWarning, match="show_total"):
        viz.plot_synaptic_currents(
            i_exc, -i_exc, 0 * i_exc, DT, n_neurons_plot=2, show_total=True
        )
    _, ff_weights, _, _ = _weights()
    with pytest.warns(DeprecationWarning, match="show_legend"):
        viz.plot_feedforward_connectivity(
            ff_weights, np.zeros(N_FF, int), 0.5, show_legend=False
        )


# ── scaling_factors ──────────────────────────────────────────────────────────


def _sf_metrics():
    rng = _rng(5)
    out = {}
    for seed in (0, 1):
        df = pd.DataFrame({"epoch": np.arange(10)})
        for key, _ in viz.SF_PATHWAYS:
            df[f"scaling_factors/{key}_value"] = 1 + 0.1 * rng.standard_normal(10)
        out[seed] = df
    out[1] = out[1].rename(columns={"scaling_factors/exc_to_exc_value": "x"})
    return out


def test_plot_sf_trajectories():
    fig = viz.plot_sf_trajectories(_sf_metrics(), suptitle="SF")
    axes = _check_fig(fig, 6)
    assert [ax.get_title() for ax in axes] == [lbl for _, lbl in viz.SF_PATHWAYS]
    assert len(axes[0].lines) == 3
    assert axes[0].get_ylim() == (0, 2)
    fig = viz.plot_sf_trajectories(
        _sf_metrics(), mean_std=True, pathways=viz.SF_PATHWAYS[:4]
    )
    assert sum(ax.get_visible() for ax in fig.axes) == 4
    assert len(fig.axes[0].collections) == 1


def test_plot_sf_vs_parameter_and_bar_chart():
    finals = {key: np.ones(3) for key, _ in viz.SF_PATHWAYS}
    fig = viz.plot_sf_vs_parameter([1, 2, 3], finals, xlabel="p")
    assert fig.axes[0].get_xlabel() == "p"
    assert fig.axes[0].get_ylabel() == "Final Scaling Factor"
    fig = viz.plot_sf_bar_chart({key: 1.0 for key, _ in viz.SF_PATHWAYS})
    assert len(fig.axes[0].patches) == 6
    fig = viz.plot_sf_bar_chart(
        {"a": {"excitatory_to_excitatory": 1.0}, "b": {}}, title="t"
    )
    assert len(fig.axes[0].patches) == 12
    assert fig.axes[0].get_title() == "t"


def test_resolve_sf_column_short_alias():
    from wiretorch.visualization.scaling_factors import _resolve_sf_column

    df = pd.DataFrame({"exc_to_exc": [1.0]})
    assert _resolve_sf_column(df, "excitatory_to_excitatory") == "exc_to_exc"


def test_feedforward_pathways_match_any_input_type():
    """``feedforward_to_*`` finds the input population whatever it is called (and
    never a recurrent pathway), in logs and in value dicts."""
    from wiretorch.visualization.scaling_factors import _resolve_sf_column

    df = pd.DataFrame(
        columns=[
            "scaling_factors/excitatory_to_excitatory_value",
            "scaling_factors/inhibitory_to_inhibitory_value",
            "scaling_factors/pn_to_excitatory_grad",
            "scaling_factors/pn_to_excitatory_value",
            "pn_to_inh",
        ]
    )
    assert (
        _resolve_sf_column(df, "feedforward_to_excitatory")
        == "scaling_factors/pn_to_excitatory_value"
    )
    assert _resolve_sf_column(df, "feedforward_to_inhibitory") == "pn_to_inh"
    assert _resolve_sf_column(df.iloc[:, :2], "feedforward_to_excitatory") is None

    values = {"pn_to_excitatory": 0.5, "excitatory_to_excitatory": 2.0}
    fig = viz.plot_sf_bar_chart(values)
    heights = [patch.get_height() for patch in fig.axes[0].patches]
    assert heights == [2.0, 0.0, 0.0, 0.0, 0.5, 0.0]


# ── training_curves ──────────────────────────────────────────────────────────


def _loss_metrics():
    rng = _rng(6)
    return {
        k: pd.DataFrame(
            {
                "epoch": np.arange(8),
                "total_loss": rng.random(8),
                "rate_e": rng.random(8),
            }
        )
        for k in ("a", "b")
    }


def test_training_curves():
    fig = viz.plot_loss_trajectories(_loss_metrics(), reference_lines={"ref": 0.5})
    (ax,) = _check_fig(fig, 1)
    assert len(ax.lines) == 3
    assert ax.get_title() == "Loss"
    fig = viz.plot_loss_mean_std(_loss_metrics())
    assert len(fig.axes[0].collections) == 1
    fig = viz.plot_firing_rate_trajectories(
        _loss_metrics(), [("rate_e", "E"), ("missing", "M")], suptitle="FR"
    )
    assert [ax.get_title() for ax in fig.axes] == ["E", "M"]
    assert len(fig.axes[0].lines) == 2


# ── firing_rate_scatter ──────────────────────────────────────────────────────


def test_firing_rate_scatter():
    rng = _rng(7)
    teacher = rng.random(30) * 10
    student = teacher + rng.standard_normal(30)
    types = np.array([0] * 20 + [1] * 10)
    fig = viz.plot_firing_rate_scatter(teacher, student, types, r2=0.9)
    (ax,) = _check_fig(fig, 1)
    assert ax.get_title() == "Firing Rates (R² = 0.900)"
    assert len(ax.collections) == 2
    assert _hex(ax.collections[0].get_facecolor()[0]) == _hex(viz.EXCITATORY_COLOR)
    fig = viz.plot_firing_rate_scatter(teacher, student, types, split_panels=True)
    axes = _check_fig(fig, 2)
    assert axes[0].get_title().startswith("Excitatory (R² = ")
    fig = viz.plot_firing_rate_scatter(teacher, student, np.arange(30) % 3)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "Group 0",
        "Group 1",
        "Group 2",
    ]


def test_r2_plots():
    fig = viz.plot_r2_vs_parameter([1, 2, 3], {"a": [0.1, 0.2, 0.3]}, title="t")
    assert len(fig.axes[0].lines) == 1
    fig = viz.plot_r2_comparison_bar(
        ["c1", "c2"], {"m1": [0.5, 0.6], "m2": [0.1, -0.2]}, ylim=(-0.5, 1)
    )
    ax = fig.axes[0]
    assert len(ax.patches) == 4
    assert len(ax.lines) == 1


# ── dashboards ───────────────────────────────────────────────────────────────


def test_create_connectivity_dashboard():
    weights, ff_weights, mask, ff_mask = _weights()
    fig = dashboards.create_connectivity_dashboard(
        weights,
        ff_weights,
        _cell_types(),
        np.zeros(N_FF, int),
        CELL_TYPE_NAMES,
        INPUT_CELL_TYPE_NAMES,
        mask,
        ff_mask,
        plot_fraction_feedforward=0.5,
        plot_fraction_recurrent=0.5,
        scaling_factors=np.array([[1.0, 2.0], [0.5, 1.0]]),
        scaling_factors_FF=np.array([[1.5, 1.0]]),
    )
    # ff + cbar, recurrent + cbar, pie, violin, mean + cbar, std + cbar
    axes = _check_fig(fig, 10)
    titles = [ax.get_title() for ax in axes]
    assert "Mean Synapse Input Counts" in titles
    assert "Weight Distribution by Input Type" in titles
    assert "Mean Total Input by Connection Type" in titles
    assert [t.get_text() for t in fig.texts] == ["Weighted Connectomes"]
    assert tuple(fig.get_size_inches()) == (24.0, 16.0)


def test_connectivity_scaling_is_by_source_row_and_target_column():
    """Weights are ``W[pre, post]``; ``scaling_factors[source, target]`` as in the
    simulators, for recurrent and feedforward blocks alike."""
    types = np.array([0, 0, 1])
    scaled = dashboards._scale_by_cell_type(
        np.ones((3, 3)), np.array([[1.0, 2.0], [3.0, 4.0]]), types, 2, types, 2
    )
    np.testing.assert_array_equal(
        scaled, [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 4.0]]
    )


def test_example_neuron_rate_closest_to_the_excitatory_mean():
    """Excitatory rates 1, 2, ..., 6 spikes: the mean is 3.5, so neuron 2 (3 spikes,
    the first of the two closest) is picked."""
    n_exc, n_steps = 6, 100
    spikes = np.zeros((1, n_steps, n_exc + 1), dtype=np.int32)
    for neuron in range(n_exc):
        spikes[0, : neuron + 1, neuron] = 1
    conductances = np.ones((1, n_steps, n_exc + 1, 2, N_SYN_REC), dtype=np.float32)
    neuron = dashboards._select_example_neuron(
        spikes,
        conductances,
        np.array([0] * n_exc + [1]),
        CELL_TYPE_NAMES,
        RECURRENT_SYNAPSE_NAMES,
        DT,
        None,
    )
    assert neuron == 2


@pytest.mark.parametrize(
    ("name", "sign"),
    [
        ("excitatory", "excitatory"),
        ("Inhibitory", "inhibitory"),
        ("L23_exc", "excitatory"),
        ("inh-fast", "inhibitory"),
        ("non-excitatory", None),
        ("input", None),
        ("pinhole", None),
    ],
)
def test_cell_type_sign_reads_whole_words(name, sign):
    from wiretorch.configs.network import cell_type_sign

    assert cell_type_sign(name) == sign
    assert _hex(connectivity._source_color(name)) == _hex(
        {
            "excitatory": vcolors.EXCITATORY_COLOR,
            "inhibitory": vcolors.INHIBITORY_COLOR,
        }.get(sign, vcolors.UNKNOWN_CELL_TYPE_COLOR)
    )


def test_create_activity_dashboard_all(capsys):
    fig = dashboards.create_activity_dashboard(**_activity_inputs(), n_neurons_plot=12)
    # raster, psth, voltage, 3 conductances, currents, then 4 rows x 2 cell types
    axes = _check_fig(fig, 7 + 8)
    titles = [ax.get_title() for ax in axes]
    assert titles[:3] == [
        "Network Spike Trains (Feedforward + Recurrent)",
        "Population Activity",
        "Membrane Potential",
    ]
    assert titles[3] == "Conductances"
    assert titles[6] == "Input Currents"
    assert [t.get_text() for t in fig.texts] == [
        "Firing Rate Distribution",
        "Coefficient of Variation",
        "Inter-Spike Interval",
        "Fano Factor",
    ]
    assert "Selected neuron index:" in capsys.readouterr().out
    raster = axes[0]
    labels = [t.get_text() for t in raster.get_legend().get_texts()]
    assert labels == ["Inhibitory", "Excitatory", "Feedforward"]
    assert len(raster.collections) == 12


@pytest.mark.parametrize(
    ("panels", "n_axes", "size"),
    [("left", 7, (16, 16)), ("raster", 2, (16, 8)), ("neuron", 5, (16, 12))],
)
def test_create_activity_dashboard_panels(panels, n_axes, size):
    fig = dashboards.create_activity_dashboard(
        **_activity_inputs(), panels=panels, neuron_idx=2
    )
    _check_fig(fig, n_axes)
    assert tuple(fig.get_size_inches()) == size
    if panels == "raster":
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert len(fig.legends) == 1


def test_create_activity_dashboard_with_assemblies():
    inputs = _activity_inputs()
    assembly_ids = np.array([i % 2 for i in range(N_REC)])
    fig = dashboards.create_activity_dashboard(
        **inputs, assembly_ids=assembly_ids, panels="neuron"
    )
    assert fig.axes[0].get_title() == "Membrane Potential"


def test_create_assembly_activity_dashboard():
    spikes = _spikes(N_REC, rate=0.05)
    ou = _rng(8).random((1, N_STEPS, 3))
    assembly_ids = np.array([i % 3 for i in range(N_REC)])
    fig = dashboards.create_assembly_activity_dashboard(
        spikes, ou, _cell_types(), assembly_ids, DT
    )
    ax_ou, ax_pop = _check_fig(fig, 2)
    assert ax_ou.get_title() == "Pattern Mixing Coefficient"
    assert ax_pop.get_title() == "Assembly Population Activity"
    assert len(ax_ou.lines) == 3
    assert len(ax_pop.lines) == 3
    assert _hex(ax_ou.lines[1].get_color()) == _hex(ax_pop.lines[1].get_color())


# ── Theme ─────────────────────────────────────────────────────────────────────


def test_theme_figure_html_shows_both_themes_and_leaves_the_figure_unchanged():
    from wiretorch.visualization import theme

    with matplotlib.rc_context():
        theme.use_theme(notebook=False)
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], color=theme.MUTED)
        ax.scatter([0, 1], [1, 0], color=theme.EXCITATORY)
        title = ax.set_title("t")
        html = theme.figure_html(fig)
        assert html.count('class="wt-light"') == html.count('class="wt-dark"') == 1
        # After drawing the dark version the figure is back in the light theme.
        assert to_hex(line.get_color()) == theme.MUTED
        assert to_hex(title.get_color()) == theme.INK
        assert to_hex(fig.get_facecolor()) == theme.LIGHT["background"]
        assert to_hex(ax.collections[0].get_facecolors()[0]) == theme.EXCITATORY
        plt.close(fig)


def test_theme_dark_recolouring_swaps_furniture_but_not_data():
    from wiretorch.visualization import theme

    with matplotlib.rc_context():
        theme.use_theme(notebook=False)
        fig, ax = plt.subplots()
        (ref,) = ax.plot([0, 1], color="k")
        (data,) = ax.plot([0, 1], color=theme.BLUE)
        theme._recolour(fig, theme._swaps(theme.LIGHT, theme.DARK))
        assert to_hex(ref.get_color()) == theme.DARK["ink"]
        assert to_hex(data.get_color()) == theme.BLUE
        assert to_hex(fig.get_facecolor()) == theme.DARK["background"]
        plt.close(fig)
