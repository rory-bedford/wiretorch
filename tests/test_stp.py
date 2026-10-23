"""Short-term plasticity: depression and facilitation of release (Tsodyks–Markram).

STP is set per (source, target) cell type pair. Each source keeps its resources ``x``
and release probability ``u``; a spike's input is its release ``u x`` over ``U``, so a
first spike from rest has its weight. Checked here: the factors against Markram, Wang
& Tsodyks' (1998) event-by-event map, target specificity, that a network without STP
is the former one, the state across chunks and checkpoints, and the Triton kernels
against the PyTorch reference in both spiking families, gradients included.
"""

import math

import numpy as np
import pytest
import torch

import wiretorch as wt
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
from wiretorch.network_simulators.current.feedforward import (
    FeedforwardCurrentLIFNetwork,
)
from wiretorch.network_simulators.projections import FrozenProjection

SEED = 2026
DT = 1.0
WEIGHTS = (0.5, 2.0)  # onto the excitatory target, onto the inhibitory one


def _one_source(stp, batch_size=2, **kwargs):
    """One input onto an ``e`` and an ``i`` neuron (weights ``WEIGHTS``), current-based
    with ``tau_syn = 0``, so each step's current is that step's input, ``w f s``; the
    neurons never spike."""
    return FeedforwardCurrentLIFNetwork(
        dt=DT,
        projections={
            ("in", "e"): FrozenProjection(np.array([[WEIGHTS[0]]])),
            ("in", "i"): FrozenProjection(np.array([[WEIGHTS[1]]])),
        },
        cell_type_indices=np.array([0, 1]),
        cell_type_indices_FF=np.zeros(1, dtype=np.int64),
        cell_params=[
            dict(name="e", cell_id=0, tau_mem=20.0, theta=1e9),
            dict(name="i", cell_id=1, tau_mem=10.0, theta=1e9),
        ],
        cell_params_FF=[dict(name="in", cell_id=0)],
        synapse_params_FF=[
            dict(name="s", synapse_id=0, cell_id=0, tau_syn=0.0, type="excitatory")
        ],
        surrgrad_scale=5.0,
        batch_size=batch_size,
        track_variables=True,
        stp=stp,
        **kwargs,
    )


def markram(times, U, tau_D, tau_F):
    """Release at each spike, by the map of Markram, Wang & Tsodyks (1998):
    ``r_n = u_n R_n``, ``u_{n+1} = U + u_n (1 - U) e^{-dt/tau_F}``,
    ``R_{n+1} = R_n (1 - u_n) e^{-dt/tau_D} + 1 - e^{-dt/tau_D}`` (``u_1 = U``,
    ``R_1 = 1``)."""
    u, R, out = U, 1.0, []
    for n, t in enumerate(times):
        if n:
            gap = (t - times[n - 1]) * DT
            fac = math.exp(-gap / tau_F) if tau_F > 0 else 0.0
            rec = math.exp(-gap / tau_D)
            u, R = U + u * (1 - U) * fac, R * (1 - u) * rec + 1 - rec
        out.append(u * R)
    return np.array(out)


def _spike_train(seed, steps=400, p=0.06):
    rng = np.random.default_rng(seed)
    x = (rng.random((2, steps, 1)) < p).astype(np.float32)
    x[:, 0] = 1.0  # a first spike from rest
    return x


@pytest.mark.parametrize(
    "stp",
    [wt.stp.DEPRESSING, wt.stp.FACILITATING, wt.STP(U=0.2, tau_D=300.0, tau_F=150.0)],
    ids=["depressing", "facilitating", "both"],
)
def test_release_follows_markram(stp, device):
    """Each spike's input is ``w r_n / U``, ``r_n`` from the event-by-event map; a
    first spike from rest has exactly its weight."""
    model = _one_source({("in", "e"): stp, ("in", "i"): stp}).to(device)
    x = _spike_train(0)
    I = model(torch.from_numpy(x).to(device))["currents_syn"][..., 0]
    I = I.cpu().numpy()
    for trial in range(2):
        times = np.flatnonzero(x[trial, :, 0])
        expected = markram(times.tolist(), stp.U, stp.tau_D, stp.tau_F) / stp.U
        assert expected[0] == pytest.approx(1.0)
        for target, w in enumerate(WEIGHTS):
            np.testing.assert_allclose(I[trial, times, target], w * expected, rtol=1e-4)
            silent = np.setdiff1d(np.arange(x.shape[1]), times)
            assert not I[trial, silent, target].any()


def test_target_specific(device):
    """One source, depressing onto ``e`` and facilitating onto ``i``: a regular train
    shrinks onto one and grows onto the other."""
    model = _one_source(
        {("in", "e"): wt.stp.DEPRESSING, ("in", "i"): wt.stp.FACILITATING}
    ).to(device)
    x = np.zeros((2, 200, 1), dtype=np.float32)
    x[:, ::20] = 1.0  # 50 Hz
    I = model(torch.from_numpy(x).to(device))["currents_syn"][0, ::20, :, 0]
    onto_e, onto_i = I[:, 0].cpu() / WEIGHTS[0], I[:, 1].cpu() / WEIGHTS[1]
    assert onto_e[0] == pytest.approx(1.0) and onto_i[0] == pytest.approx(1.0)
    assert (torch.diff(onto_e) < 0).all() and onto_e[-1] < 0.3
    assert onto_i[-1] > 3.0
    times = list(range(0, 200, 20))
    for stp, got in ((wt.stp.DEPRESSING, onto_e), (wt.stp.FACILITATING, onto_i)):
        expected = markram(times, stp.U, stp.tau_D, stp.tau_F) / stp.U
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4)


def test_release_drives_saturation(device):
    """STP and saturation on one connection: each release (``s f``, not ``s``) opens
    receptors, at most those still closed, and the conductance's decay component is
    ``g_bar w g``."""
    from test_saturation import _inputs_onto_one

    stp, c, tau, w = wt.stp.FACILITATING, 0.4, 20.0, 1.5
    model = _inputs_onto_one(
        [w], c, tau, DT, 2, track_variables=True, stp={("in", "e"): stp}
    ).to(device)
    x = _spike_train(1, steps=300, p=0.08)
    out = model(torch.from_numpy(x).to(device))
    d_D, d_F = stp.decays(DT)
    u, r, release = np.full(2, stp.U), np.ones(2), []
    for t in range(x.shape[1]):  # the STP recursion, for each step's release f
        r = 1 - (1 - r) * d_D
        u = stp.U + (u - stp.U) * d_F
        f = u * r / stp.U
        r = r - x[:, t, 0] * u * r
        u = u + x[:, t, 0] * stp.U * (1 - u)
        release.append(f)
    g, gs, d = np.zeros(2), [], math.exp(-DT / tau)
    for t, f in enumerate(release):  # each release opens c f of the closed, at most all
        g = g * d
        g = g + x[:, t, 0] * np.minimum(c * f * (1 - g), 1 - g)
        gs.append(g)
    decay = out["conductances_feedforward"][:, :, 0, 1, 0].cpu().numpy()
    np.testing.assert_allclose(decay, 2.0 * w * np.stack(gs, 1), rtol=1e-4, atol=1e-6)
    assert np.stack(release).max() > 1 / c  # facilitated: a release would open them all


def test_pathways_and_rest_state():
    """A source cell type with two STP sets gets two pathways, sharing its rows, each
    reaching its own target types; the state starts at rest, ``u = U``, ``x = 1``."""
    model = recurrent(BATCH_SIZE, SEED, stp=True).model
    exc = [p for p in model.pathways if p.side == 0 and p.cell_type == 0]
    assert len(exc) == 2 and len({p.source for p in exc}) == 1
    assert {p.stp for p in exc} == {wt.stp.DEPRESSING, wt.stp.FACILITATING}
    assert sorted(p.targets for p in exc) == [(0,), (1,)]
    inh = [p for p in model.pathways if p.side == 0 and p.cell_type == 1]
    assert len(inh) == 1 and inh[0].stp is None and inh[0].targets is None
    for p in exc:
        start, n = model._gate_columns[p.index]
        u, x = (
            model.gating[:, start : start + n],
            model.gating[:, start + n : start + 2 * n],
        )
        assert (u == p.stp.U).all() and (x == 1).all()


@pytest.mark.parametrize(
    "build", [recurrent, feedforward, current_recurrent], ids=["cond", "ff", "current"]
)
def test_no_stp_is_the_former_network(build, device):
    runs = []
    for stp in (False, True):
        built = build(BATCH_SIZE, SEED, stp=stp)
        x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
        runs.append(built.model.to(device)(x))
    assert not torch.equal(runs[0], runs[1])  # STP changes the network
    built = build(BATCH_SIZE, SEED)
    model = built.model.to(device)
    model.reset_state()
    again = model(torch.from_numpy(built.input_spikes[:, :60]).to(device))
    torch.testing.assert_close(again, runs[0], rtol=0, atol=0)


def test_state_carries_across_chunks_and_checkpoints(device):
    runs = []
    for mode in ("one", "chunks", "checkpoints"):
        built = recurrent(BATCH_SIZE, SEED, stp=True, saturation=0.63)
        model = built.model.to(device)
        x = torch.from_numpy(built.input_spikes[:, :60]).to(device)
        if mode == "one":
            runs.append((model(x), model.gating.clone()))
            continue
        chunks = []
        for start in range(0, 60, 10):
            if mode == "checkpoints":
                state = model.get_checkpoint_state()
                model.reset_state()
                model.load_checkpoint_state(state)
            chunks.append(model(x[:, start : start + 10]))
        runs.append((torch.cat(chunks, dim=1), model.gating.clone()))
    for run in runs[1:]:
        torch.testing.assert_close(run, runs[0], rtol=0, atol=1e-6)


def test_checkpoints_without_gating_load_at_rest():
    model = recurrent(BATCH_SIZE, SEED, stp=True).model
    model(torch.from_numpy(recurrent(BATCH_SIZE, SEED).input_spikes[:, :30]))
    state = model.get_checkpoint_state()
    del state["gating"]
    model.load_checkpoint_state(state)
    torch.testing.assert_close(model.gating, model._gating_rest.expand_as(model.gating))


def test_validation():
    with pytest.raises(ValueError, match="U must be"):
        wt.STP(U=0.0, tau_D=100.0)
    with pytest.raises(ValueError, match="tau_D"):
        wt.STP(U=0.5, tau_D=0.0)
    with pytest.raises(ValueError, match="tau_F"):
        wt.STP(U=0.5, tau_D=100.0, tau_F=-1.0)
    with pytest.raises(ValueError, match="no projection"):
        _one_source({("in", "x"): wt.stp.DEPRESSING})
    model = _one_source({("in", "e"): dict(U=0.4, tau_D=200.0)})  # a dict works too
    assert model.pathways[0].stp == wt.STP(U=0.4, tau_D=200.0)


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
@pytest.mark.parametrize("family", ["conductance", "current"])
def test_recurrent_kernels_match_reference(family, recurrent_gradients, monkeypatch):
    if family == "conductance":
        builder = lambda: recurrent(  # noqa: E731
            BATCH_SIZE, SEED, scaling_factors=True, learn=CONDUCTANCE_LEARN, stp=True
        )
    else:
        builder = lambda: current_recurrent(  # noqa: E731
            BATCH_SIZE, SEED, learn=CURRENT_LEARN, stp=True
        )
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("family", ["conductance", "current"])
def test_recurrent_weights_kernels_match_reference(
    family, recurrent_gradients, monkeypatch
):
    """Learnt recurrent weights, whose rows the STP pathways share."""
    if family == "conductance":
        builder = lambda: recurrent(  # noqa: E731
            BATCH_SIZE, SEED, learnt_weights=True, stp=True, n=1000, n_ff=533
        )
    else:
        builder = lambda: current_recurrent(  # noqa: E731
            BATCH_SIZE, SEED, optimisable="weights", stp=True
        )
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)


@pytest.mark.gpu
def test_with_saturation_kernels_match_reference(monkeypatch):
    """STP and saturation on one pathway: the release drives the receptors."""
    learn = {**CONDUCTANCE_LEARN, "tau_decay": ["AMPA", "GABA_A"]}
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, scaling_factors=True, learn=learn, stp=True, saturation=0.63
    )
    _kernels_vs_reference(_with_flag(builder, True), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["conductance", "current"])
def test_feedforward_kernels_match_reference(family, monkeypatch):
    if family == "conductance":
        builder = lambda: feedforward(  # noqa: E731
            BATCH_SIZE, SEED, probabilities=True, learn=CONDUCTANCE_LEARN, stp=True
        )
    else:
        builder = lambda: current_feedforward_probabilities(  # noqa: E731
            BATCH_SIZE, SEED, learn=CURRENT_FF_LEARN, stp=True
        )
    _kernels_vs_reference(builder, monkeypatch)
