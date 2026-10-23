"""The refractory period ``tau_ref``.

For ``n_ref = round(tau_ref / dt)`` steps after a spike a neuron is held at ``U_reset``
and cannot spike; its synaptic state evolves as usual. A counter per neuron carries the
period across ``forward`` calls (and checkpoints). Checked here: the exact spike times
and voltages of single neurons, in both families, on the CPU and the GPU; that the
state carries over; and that the Triton kernels — forward, both backward passes, and
the spike compaction of the recurrent step — match the PyTorch reference.
"""

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
from wiretorch.network_simulators.conductance.recurrent import ConductanceLIFNetwork
from wiretorch.network_simulators.current.recurrent import CurrentLIFNetwork
from wiretorch.network_simulators.projections import (
    FrozenProjection,
    FullRankProjection,
)

SEED = 2026
FAMILIES = ["conductance", "current"]
#: Threshold and reset of the single-neuron networks, per family: with reset below
#: threshold (a strong input drives the neuron over it in one step), or above it.
THETA = {"conductance": -50.0, "current": 1.0}
RESET = {
    "conductance": {"below": -60.0, "above": -40.0},
    "current": {"below": 0.0, "above": 2.0},
}


def _single(family, regime, tau_ref, n=3, w_rec=None, device="cpu", **kwargs):
    """``n`` identical neurons of one cell type ``e`` with refractory period
    ``tau_ref`` and, with the reset ``"below"`` threshold, an input strong enough to
    make them spike whenever they can. Not coupled, unless ``w_rec`` gives their
    recurrent weights (then learnt)."""
    w_ff = 1000.0 if regime == "below" else 0.0
    cell = dict(name="e", cell_id=0, tau_mem=20.0, theta=THETA[family])
    cell.update(U_reset=RESET[family][regime], tau_ref=tau_ref)
    if family == "conductance":
        cell.update(E_L=-60.0, g_L=10.0)
        synapse = dict(tau_rise=0.1, tau_decay=1.0, E_syn=0.0, g_bar=1.0)
        network = ConductanceLIFNetwork
    else:
        synapse = dict(tau_syn=0.0, type="excitatory")
        network = CurrentLIFNetwork
    model = network(
        dt=1.0,
        rec_projections={
            ("e", "e"): FrozenProjection(np.zeros((n, n)))
            if w_rec is None
            else FullRankProjection(np.full((n, n), w_rec))
        },
        ff_projections={("in", "e"): FrozenProjection(np.full((1, n), w_ff))},
        cell_type_indices=np.zeros(n, dtype=np.int64),
        cell_type_indices_FF=np.zeros(1, dtype=np.int64),
        cell_params=[cell],
        cell_params_FF=[dict(name="in", cell_id=0)],
        synapse_params=[dict(synapse_id=0, cell_id=0, name="rec", **synapse)],
        synapse_params_FF=[dict(synapse_id=0, cell_id=0, name="ff", **synapse)],
        surrgrad_scale=5.0,
        batch_size=2,
        **kwargs,
    ).to(device)
    return model, torch.ones((2, 40, 1), device=device)


@pytest.mark.parametrize("tau_ref", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("regime", ["below", "above"])
@pytest.mark.parametrize("family", FAMILIES)
def test_spike_times_and_held_voltages(family, regime, tau_ref, device):
    """After each spike, ``n_ref`` steps at reset without a spike. Then a neuron
    whose reset is below threshold needs one step of input to spike again (an
    interval of ``n_ref + 2`` steps), one whose reset is above it spikes at once
    (``n_ref + 1``)."""
    model, x = _single(family, regime, tau_ref, device=device, track_variables=True)
    out = model(x)
    spikes, voltages = out["spikes"].cpu(), out["voltages"].cpu()
    n_ref = round(tau_ref)
    expected = n_ref + (2 if regime == "below" else 1)
    trains = spikes.permute(0, 2, 1).reshape(-1, 40)  # (trial x neuron, step)
    for train, v in zip(trains, voltages.permute(0, 2, 1).reshape(-1, 40)):
        times = torch.nonzero(train).flatten()
        assert len(times) >= 40 // expected - 1
        assert (torch.diff(times) == expected).all(), times
        for t in times[:-1]:  # v after the spike's step, and the n_ref after it
            held = v[t : t + n_ref + 1]
            assert (held == RESET[family][regime]).all(), (t, held)


@pytest.mark.parametrize("family", FAMILIES)
def test_tau_ref_in_whole_steps(family):
    """``n_ref = round(tau_ref / dt)``, recomputed with the time step; a period that
    rounds to zero steps is no period at all."""
    model, _ = _single(family, "below", 3.0)
    assert model.n_ref.tolist() == [3, 3, 3] and model._refractory
    assert model._physiology().n_ref is not None
    model.set_timestep(0.5)
    assert model.n_ref.tolist() == [6, 6, 6]
    model, _ = _single(family, "below", 0.4)
    assert model.n_ref.tolist() == [0, 0, 0] and not model._refractory
    assert model._physiology().n_ref is None
    with pytest.raises(AssertionError, match="tau_ref"):
        _single(family, "below", -1.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_no_period_is_the_former_network(family, device):
    """``tau_ref = 0`` (and absent: see the recorded fixtures) changes nothing."""
    build = {"conductance": recurrent, "current": current_recurrent}[family]
    runs = []
    for tau_ref in (None, 0.0):
        built = build(BATCH_SIZE, SEED, tau_ref=tau_ref)
        model = built.model.to(device)
        x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
        runs.append((model(x).detach(), model.v.clone()))
    torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)


def _builder(family, tau_ref=5.0, **kwargs):
    if family == "conductance":
        return recurrent(
            BATCH_SIZE, SEED, scaling_factors=True, tau_ref=tau_ref, **kwargs
        )
    return current_recurrent(BATCH_SIZE, SEED, tau_ref=tau_ref, **kwargs)


@pytest.mark.parametrize("family", FAMILIES)
def test_period_changes_the_network(family):
    """In the test networks the period binds: it changes their spikes."""
    runs = []
    for tau_ref in (None, 5.0):
        built = _builder(family, tau_ref)
        runs.append(built.model(torch.from_numpy(built.input_spikes[:, :100])))
    assert runs[1].sum() > 0
    assert not torch.equal(runs[0], runs[1])


@pytest.mark.parametrize("family", FAMILIES)
def test_counters_carry_across_chunks_and_checkpoints(family, device):
    """One 60-step chunk, six 10-step ones, or six with the state saved and restored
    between them: the same spikes."""
    runs = []
    for mode in ("one", "chunks", "checkpoints"):
        built = _builder(family)
        model = built.model.to(device)
        x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
        if mode == "one":
            runs.append(model(x))
            continue
        chunks = []
        for start in range(0, 60, 10):
            if mode == "checkpoints":
                state = model.get_checkpoint_state()
                assert "refractory" in state
                model.reset_state()
                assert not model.refractory.any()
                model.load_checkpoint_state(state)
            chunks.append(model(x[:, start : start + 10]))
        runs.append(torch.cat(chunks, dim=1))
    assert runs[0].sum() > 0
    for run in runs[1:]:
        torch.testing.assert_close(run, runs[0], rtol=0, atol=0)


def test_checkpoints_without_counters_still_load():
    """A checkpoint from before the refractory period (no counters) loads, with every
    neuron free to spike; and one with no period does not save counters."""
    model = _builder("conductance").model
    model(torch.from_numpy(_builder("conductance").input_spikes[:, :30]))
    state = model.get_checkpoint_state()
    del state["refractory"]
    model.load_checkpoint_state(state)
    assert not model.refractory.any()
    assert (
        "refractory" not in _builder("conductance", None).model.get_checkpoint_state()
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
    """Spikes, state and every gradient — scaling factors and all learnable
    physiology — through a warm-up chunk (the counters carry over) and both backward
    passes."""
    learn = CONDUCTANCE_LEARN if family == "conductance" else CURRENT_LEARN
    _kernels_vs_reference(
        _with_flag(lambda: _builder(family, learn=learn), recurrent_gradients),
        monkeypatch,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", FAMILIES)
def test_feedforward_kernels_match_reference(family, monkeypatch):
    """The fused scan, with spike probabilities (zero while refractory) in the loss."""
    if family == "conductance":
        builder = lambda: feedforward(  # noqa: E731
            BATCH_SIZE, SEED, probabilities=True, learn=CONDUCTANCE_LEARN, tau_ref=5.0
        )
    else:
        builder = lambda: current_feedforward_probabilities(  # noqa: E731
            BATCH_SIZE, SEED, learn=CURRENT_FF_LEARN, tau_ref=5.0
        )
    _kernels_vs_reference(builder, monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("family", FAMILIES)
def test_refractory_neurons_send_no_spikes(family, recurrent_gradients, monkeypatch):
    """Neurons whose reset is above threshold, coupled: were a held neuron listed by
    the spike compaction, its weight row would reach the others, and their state would
    part from the reference's. Gradients included, through the recurrent weights."""
    results = []
    for backend in ("triton", "torch"):
        monkeypatch.setenv("WIRETORCH_BACKEND", backend)
        model, x = _single(
            family, "above", 3.0, n=5, w_rec=0.01, device="cuda",
            track_variables=True, recurrent_gradients=recurrent_gradients,
        )  # fmt: skip
        out = model(x)
        readout = torch.linspace(-1, 1, out["spikes"].numel(), device="cuda")
        (out["spikes"] * readout.view_as(out["spikes"])).sum().backward()
        grad = model.rec_projections["e__e"].log_weights.grad
        results.append((out["spikes"], out["voltages"], out["currents"], grad))
    for kernel, reference in zip(*results):
        torch.testing.assert_close(kernel, reference, rtol=1e-5, atol=1e-5)
    assert results[1][0].sum() > 0
