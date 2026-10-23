"""Spike-frequency adaptation: a slow, spike-triggered inhibitory current onto the
neuron itself.

A cell type with ``adapt > 0`` gives each of its neurons an adaptation variable (part
of the state) that decays with ``tau_adapt`` and rises by ``adapt`` at each of the
neuron's spikes: in conductance-based networks a conductance by ``E_adapt``, in
current-based ones a current subtracted from the input. Checked here: the variable
against its recursion, that it lengthens the intervals under constant drive, that
``adapt = 0`` is the former network, the state across chunks and checkpoints,
validation, and the Triton kernels against the PyTorch reference, gradients of the
adaptation's parameters included.
"""

import math

import numpy as np
import pytest
import torch

from networks import (
    BATCH_SIZE,
    current_feedforward_probabilities,
    current_recurrent,
    feedforward,
    recurrent,
)
from test_learnt_physiology import (
    CONDUCTANCE_LEARN,
    CURRENT_FF_LEARN,
    CURRENT_LEARN,
    _kernels_vs_reference,
)
from wiretorch.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from wiretorch.network_simulators.current.feedforward import (
    FeedforwardCurrentLIFNetwork,
)
from wiretorch.network_simulators.projections import FrozenProjection

SEED = 2026
FAMILIES = ["conductance", "current"]
ADAPT = {"conductance": 2.0, "current": 0.3}  # per spike: nS, or input units
TAU_ADAPT = 150.0
LEARN_ADAPT = {"adapt": ["excitatory"], "tau_adapt": ["excitatory"]}


def _driven(family, adapt, n=3, batch_size=2, **kwargs):
    """``n`` uncoupled neurons with a constant strong input, adapting by ``adapt``."""
    cell = dict(name="e", cell_id=0, tau_mem=20.0, adapt=adapt, tau_adapt=TAU_ADAPT)
    if family == "conductance":
        cell.update(theta=-50.0, U_reset=-65.0, E_L=-65.0, g_L=1.0, E_adapt=-90.0)
        synapse = dict(tau_rise=0.1, tau_decay=1.0, E_syn=0.0, g_bar=1.0)
        network, w = FeedforwardConductanceLIFNetwork, 3.0
    else:
        synapse = dict(tau_syn=0.0, type="excitatory")
        network, w = FeedforwardCurrentLIFNetwork, 2.0
    return network(
        dt=1.0,
        projections={("in", "e"): FrozenProjection(np.full((1, n), w))},
        cell_type_indices=np.zeros(n, dtype=np.int64),
        cell_type_indices_FF=np.zeros(1, dtype=np.int64),
        cell_params=[cell],
        cell_params_FF=[dict(name="in", cell_id=0)],
        synapse_params_FF=[dict(synapse_id=0, cell_id=0, name="s", **synapse)],
        surrgrad_scale=5.0,
        batch_size=batch_size,
        track_variables=True,
        **kwargs,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_adaptation_follows_the_spikes(family, device):
    """Each step it decays by ``exp(-dt / tau_adapt)`` and rises by ``adapt`` at the
    neuron's spike."""
    model = _driven(family, ADAPT[family]).to(device)
    out = model(torch.ones((2, 400, 1), device=device))
    spikes = out["spikes"].cpu().numpy()
    tracked = out["adaptation"].cpu().numpy()
    decay = math.exp(-1.0 / TAU_ADAPT)
    a, expected = np.zeros(spikes[:, 0].shape), []
    for t in range(spikes.shape[1]):
        a = a * decay + ADAPT[family] * spikes[:, t]
        expected.append(a)
    np.testing.assert_allclose(tracked, np.stack(expected, 1), rtol=1e-5, atol=1e-6)
    assert spikes.sum() > 10
    torch.testing.assert_close(model.adaptation, out["adaptation"][:, -1])


@pytest.mark.parametrize("family", FAMILIES)
def test_intervals_lengthen(family, device):
    """Under constant drive the intervals lengthen as the adaptation builds up; without
    it they are regular."""
    intervals = {}
    for adapt in (0.0, ADAPT[family]):
        model = _driven(family, adapt).to(device)
        spikes = model(torch.ones((2, 600, 1), device=device))["spikes"][0, :, 0]
        intervals[adapt] = torch.diff(torch.nonzero(spikes).flatten()).cpu()
    regular, adapting = intervals[0.0], intervals[ADAPT[family]]
    assert regular.max() - regular.min() <= 1
    assert adapting[-1] > 1.5 * adapting[0] and len(adapting) < len(regular)


BUILDERS = {"conductance": recurrent, "current": current_recurrent}


@pytest.mark.parametrize("family", FAMILIES)
def test_no_adaptation_is_the_former_network(family, device):
    build = BUILDERS[family]
    runs = []
    for adapt in (None, 0.0):
        built = build(BATCH_SIZE, SEED, adapt=adapt)
        model = built.model.to(device)
        assert not model._adapting
        runs.append(model(torch.from_numpy(built.input_spikes[:, :60]).to(device)))
    torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)
    built = build(BATCH_SIZE, SEED, adapt=ADAPT[family])
    x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
    assert not torch.equal(built.model.to(device)(x), runs[0])


@pytest.mark.parametrize("family", FAMILIES)
def test_state_carries_across_chunks_and_checkpoints(family, device):
    runs = []
    for mode in ("one", "chunks", "checkpoints"):
        built = BUILDERS[family](BATCH_SIZE, SEED, adapt=ADAPT[family])
        model = built.model.to(device)
        x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
        if mode == "one":
            runs.append((model(x), model.adaptation.clone()))
            continue
        chunks = []
        for start in range(0, 60, 10):
            if mode == "checkpoints":
                state = model.get_checkpoint_state()
                assert "adaptation" in state
                model.reset_state()
                assert not model.adaptation.any()
                model.load_checkpoint_state(state)
            chunks.append(model(x[:, start : start + 10]))
        runs.append((torch.cat(chunks, dim=1), model.adaptation.clone()))
    assert runs[0][1].any()
    for run in runs[1:]:
        torch.testing.assert_close(run, runs[0], rtol=0, atol=1e-5)


def test_checkpoints_without_adaptation_load():
    model = recurrent(BATCH_SIZE, SEED, adapt=2.0).model
    model(torch.from_numpy(recurrent(BATCH_SIZE, SEED).input_spikes[:, :30]))
    state = model.get_checkpoint_state()
    del state["adaptation"]
    model.load_checkpoint_state(state)
    assert not model.adaptation.any()
    assert "adaptation" not in recurrent(BATCH_SIZE, SEED).model.get_checkpoint_state()


def test_validation():
    with pytest.raises(AssertionError, match="adapt must be"):
        _driven("current", -1.0)
    with pytest.raises(AssertionError, match="tau_adapt"):
        FeedforwardCurrentLIFNetwork(
            dt=1.0,
            projections={("in", "e"): FrozenProjection(np.ones((1, 1)))},
            cell_type_indices=np.zeros(1, dtype=np.int64),
            cell_type_indices_FF=np.zeros(1, dtype=np.int64),
            cell_params=[
                dict(name="e", cell_id=0, tau_mem=20.0, adapt=1.0, tau_adapt=0.0)
            ],
            cell_params_FF=[dict(name="in", cell_id=0)],
            synapse_params_FF=[
                dict(synapse_id=0, cell_id=0, name="s", tau_syn=0.0, type="excitatory")
            ],
            surrgrad_scale=5.0,
            batch_size=1,
        )
    with pytest.raises(AssertionError, match="adapt must be positive"):
        recurrent(BATCH_SIZE, SEED, learn={"adapt": True}, adapt=2.0)  # inhibitory: 0
    model = recurrent(BATCH_SIZE, SEED, learn=LEARN_ADAPT, adapt=2.0).model
    assert {"cell_parameters.log_adapt", "cell_parameters.log_tau_adapt"} <= set(
        dict(model.named_parameters())
    )


# =====================================================================
# The Triton kernels against the reference (GPU)
# =====================================================================


def _with_flag(builder, recurrent_gradients):
    def build():
        built = builder()
        built.model.recurrent_gradients = recurrent_gradients
        return built

    return build


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("family", FAMILIES)
def test_recurrent_kernels_match_reference(family, recurrent_gradients, monkeypatch):
    """Spikes, state and every gradient — the adaptation's parameters among them —
    through both backward passes."""
    if family == "conductance":
        learn = {**CONDUCTANCE_LEARN, **LEARN_ADAPT, "E_adapt": ["excitatory"]}
        builder = lambda: recurrent(  # noqa: E731
            BATCH_SIZE, SEED, scaling_factors=True, learn=learn, adapt=ADAPT[family]
        )
    else:
        builder = lambda: current_recurrent(  # noqa: E731
            BATCH_SIZE, SEED, learn={**CURRENT_LEARN, **LEARN_ADAPT},
            adapt=ADAPT[family],
        )  # fmt: skip
    grads = _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)
    assert grads["cell_parameters.log_adapt"].abs().sum() > 0


@pytest.mark.gpu
@pytest.mark.parametrize("family", FAMILIES)
def test_feedforward_kernels_match_reference(family, monkeypatch):
    if family == "conductance":
        learn = {**CONDUCTANCE_LEARN, **LEARN_ADAPT, "E_adapt": ["excitatory"]}
        builder = lambda: feedforward(  # noqa: E731
            BATCH_SIZE, SEED, probabilities=True, learn=learn, adapt=ADAPT[family]
        )
    else:
        builder = lambda: current_feedforward_probabilities(  # noqa: E731
            BATCH_SIZE, SEED, learn={**CURRENT_FF_LEARN, **LEARN_ADAPT},
            adapt=ADAPT[family],
        )  # fmt: skip
    _kernels_vs_reference(builder, monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
def test_every_mechanism_together(recurrent_gradients, monkeypatch):
    """Adaptation, short-term plasticity, saturation, the Mg block and a refractory
    period at once."""
    learn = {
        **CONDUCTANCE_LEARN, **LEARN_ADAPT, "E_adapt": ["excitatory"],
        "tau_decay": ["AMPA", "GABA_A"],
    }  # fmt: skip
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, scaling_factors=True, learn=learn, adapt=2.0, stp=True,
        saturation=0.63, tau_ref=3.0, mg=1.0,
    )  # fmt: skip
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)
