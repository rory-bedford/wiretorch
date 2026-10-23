"""Receptor saturation (conductance-based networks).

A synapse type with ``saturation = c > 0`` gives each source neuron a gating variable
``u`` per such type, the fraction of its receptors that are open: every step it decays
with the type's ``tau_decay``, then a spike opens the fraction ``c`` of those still
closed, and the spike's input is scaled by that fraction. The target's conductance then
follows ``g_bar sum_j w_j u_j`` (``g_bar``: every receptor open). Checked here: that
exact recursion, the mean of ``u`` under Poisson input against its closed form and
Brunel & Wang's (2001) ``psi``, that ``saturation = 0`` is the former network, the state
across chunks and checkpoints, and the Triton kernels against the PyTorch reference,
gradients included.
"""

import math

import numpy as np
import pytest
import torch

from networks import BATCH_SIZE, feedforward, recurrent
from test_learnt_physiology import CONDUCTANCE_LEARN, _kernels_vs_reference
from wiretorch.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from wiretorch.network_simulators.current.recurrent import CurrentLIFNetwork
from wiretorch.network_simulators.projections import FrozenProjection

SEED = 2026
C = 1 - math.exp(-1)  # Brunel & Wang's NMDA: alpha = 0.5 / ms over tau_rise = 2 ms
#: Every physiology parameter, but not the saturating (NMDA) types' tau_decay.
LEARN = {**CONDUCTANCE_LEARN, "tau_decay": ["AMPA", "GABA_A"]}


def _inputs_onto_one(weights, saturation, tau_decay, dt, batch_size, **kwargs):
    """One neuron driven by ``len(weights)`` inputs through one synapse type."""
    n_in = len(weights)
    return FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections={("in", "e"): FrozenProjection(np.asarray(weights)[:, None])},
        cell_type_indices=np.zeros(1, dtype=np.int64),
        cell_type_indices_FF=np.zeros(n_in, dtype=np.int64),
        cell_params=[
            dict(
                name="e",
                cell_id=0,
                tau_mem=20.0,
                theta=50.0,
                U_reset=-60.0,
                E_L=-60.0,
                g_L=1.0,
            )
        ],  # fmt: skip
        cell_params_FF=[dict(name="in", cell_id=0)],
        synapse_params_FF=[
            dict(
                name="NMDA",
                synapse_id=0,
                cell_id=0,
                tau_rise=0.5,
                tau_decay=tau_decay,
                E_syn=0.0,
                g_bar=2.0,
                saturation=saturation,
            )
        ],  # fmt: skip
        surrgrad_scale=5.0,
        batch_size=batch_size,
        **kwargs,
    )


def _gating(x, c, decay):
    """The recursion, in numpy: ``u`` after each step of input ``x`` ``(B, T, n)``."""
    u = np.zeros((x.shape[0], x.shape[2]))
    out = []
    for t in range(x.shape[1]):
        u = u * decay
        u = np.minimum(u + x[:, t] * c * (1 - u), 1.0)
        out.append(u)
    return np.stack(out, axis=1)


def test_tables_and_pathways():
    """A saturating type gets a pathway of its own, sharing its cell type's rows; its
    kernel is not normalised, so ``g_bar`` is the conductance with all receptors open."""
    model = recurrent(BATCH_SIZE, SEED, saturation=C).model
    names = [s["name"] for s in model.unified_synapse_params]
    nmda = torch.tensor([name == "NMDA" for name in names])
    torch.testing.assert_close(model.saturation, torch.where(nmda, C, 0.0))
    gated = [p for p in model.pathways if p.gate is not None]
    assert [names[p.gate] for p in gated] == ["NMDA", "NMDA"]  # recurrent, input
    for p in gated:
        twin = next(q for q in model.pathways if q.index == p.source)
        assert twin.gate is None and twin.cell_type == p.cell_type
        assert torch.nonzero(model._made[p.index]).flatten().tolist() == [p.gate]
        assert not model._made[twin.index][nmda].any()
    g_bar = model.g_bar[nmda]
    torch.testing.assert_close(model.g_scale[:, nmda], torch.stack([-g_bar, g_bar]))
    assert model.n_gating == model.gating.shape[1] > 0
    assert recurrent(BATCH_SIZE, SEED).model.n_gating == 0


@pytest.mark.parametrize("build", [recurrent, feedforward], ids=["rec", "ff"])
def test_no_saturation_is_the_former_network(build, device):
    """``saturation = 0`` (and absent: the recorded fixtures) changes nothing."""
    runs = []
    for saturation in (None, 0.0):
        built = build(BATCH_SIZE, SEED, saturation=saturation)
        model = built.model.to(device)
        spikes = model(torch.from_numpy(built.input_spikes[:, :60]).to(device))
        runs.append((spikes.detach(), model.v.clone(), model._synaptic.clone()))
    torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)


def test_saturation_changes_the_network():
    off, on = (
        recurrent(BATCH_SIZE, SEED, saturation=s).model(
            torch.from_numpy(recurrent(BATCH_SIZE, SEED).input_spikes[:, :150])
        )
        for s in (None, C)
    )
    assert on.sum() > 0 and not torch.equal(off, on)


def test_conductance_follows_the_gating(device):
    """Each step, the decay component of the conductance is ``g_bar sum_j w_j u_j``,
    ``u_j`` the recursion over input ``j``'s spikes."""
    weights = [0.5, 1.0, 2.0]
    model = _inputs_onto_one(weights, 0.4, 20.0, 1.0, 2, track_variables=True).to(
        device
    )
    rng = np.random.default_rng(0)
    x = (rng.random((2, 300, 3)) < 0.08).astype(np.float32)
    out = model(torch.from_numpy(x).to(device))
    u = _gating(x, 0.4, math.exp(-1.0 / 20.0))
    expected = 2.0 * u @ np.asarray(weights)
    decay = out["conductances_feedforward"][:, :, 0, 1, 0].cpu().numpy()
    np.testing.assert_allclose(decay, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.gating.cpu().numpy(), u[:, -1], atol=1e-6)


def psi(nu_hz, alpha=0.5, tau_rise=2.0, tau_decay=100.0, terms=60):
    """Brunel & Wang (2001): the mean gating of NMDA receptors under Poisson input."""
    nu = nu_hz / 1000.0
    tau = alpha * tau_rise * tau_decay
    a = tau_rise * (1 + nu * tau)

    def T(n):
        return sum(
            (-1) ** k * math.comb(n, k) * a / (a + k * tau_decay) for k in range(n + 1)
        )

    series = sum(
        (-alpha * tau_rise) ** n * T(n) / math.factorial(n + 1) for n in range(1, terms)
    )
    return nu * tau / (1 + nu * tau) * (1 + series / (1 + nu * tau))


@pytest.mark.parametrize("nu", [3.0, 25.0, 50.0])
def test_mean_gating_under_poisson_input(nu, device):
    """The mean of ``u`` under Poisson input at ``nu`` Hz: exactly ``p c / (1 - d (1 -
    p c))`` (``p`` the spike probability per step, ``d`` the decay), and close to
    Brunel & Wang's ``psi(nu)`` with their NMDA (about 0.17, 0.64 and 0.79). Not
    equal: ``psi`` is their approximation of the gating through a rising transmitter
    concentration, which a fraction opened per spike stands in for."""
    dt, tau_decay, n_in = 0.5, 100.0, 4000
    model = _inputs_onto_one([0.0] * n_in, C, tau_decay, dt, 2).to(device)
    gen = torch.Generator().manual_seed(int(nu))
    p = nu * dt / 1000
    x = (torch.rand((2, round(1000 / dt), n_in), generator=gen) < p).float()
    model(x.to(device))
    mean = model.gating.mean().item()  # stationary, over trials and inputs
    d = math.exp(-dt / tau_decay)
    assert mean == pytest.approx(p * C / (1 - d * (1 - p * C)), abs=0.01)
    assert mean == pytest.approx(psi(nu), abs=0.035)


def test_psi():
    assert [round(psi(nu), 3) for nu in (3, 25, 50)] == [0.167, 0.643, 0.787]


def test_recurrent_gating_follows_the_spikes(device):
    """The gating state after a run is the recursion over the network's own spikes
    (and its inputs), source by source."""
    built = recurrent(BATCH_SIZE, SEED, saturation=C)
    model = built.model.to(device)
    x = built.input_spikes[:, :100]
    spikes = model(torch.from_numpy(x).to(device)).cpu().numpy()
    assert spikes.sum() > 0
    for pathway in model.pathways:
        if pathway.gate is None:
            continue
        rows = getattr(model, f"_rows_{pathway.source}").cpu().numpy()
        recurrent_side = model._sides[pathway.side].recurrent
        pre = (spikes if recurrent_side else x)[:, :, rows]
        decay = math.exp(-model.dt.item() / model.tau_decay[pathway.gate].item())
        start, n = model._gate_columns[pathway.index]
        np.testing.assert_allclose(
            model.gating[:, start : start + n].cpu().numpy(),
            _gating(pre, C, decay)[:, -1],
            rtol=1e-5,
            atol=1e-6,
        )


def test_state_carries_across_chunks_and_checkpoints(device):
    """One 60-step chunk, six 10-step ones, or six with the state saved and restored
    between them: the same spikes and gating."""
    runs = []
    for mode in ("one", "chunks", "checkpoints"):
        built = recurrent(BATCH_SIZE, SEED, saturation=C)
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
                assert not model.gating.any()
                model.load_checkpoint_state(state)
            chunks.append(model(x[:, start : start + 10]))
        runs.append((torch.cat(chunks, dim=1), model.gating.clone()))
    assert runs[0][0].sum() > 0 and runs[0][1].any()
    for run in runs[1:]:
        torch.testing.assert_close(run, runs[0], rtol=0, atol=1e-6)


def test_checkpoints_without_gating_load():
    """A checkpoint from before saturation loads with every receptor closed, and one
    with no saturating type saves no gating."""
    model = recurrent(BATCH_SIZE, SEED, saturation=C).model
    model(torch.from_numpy(recurrent(BATCH_SIZE, SEED).input_spikes[:, :30]))
    state = model.get_checkpoint_state()
    del state["gating"]
    model.load_checkpoint_state(state)
    assert not model.gating.any()
    assert "gating" not in recurrent(BATCH_SIZE, SEED).model.get_checkpoint_state()


def test_tau_decay_of_saturating_types_cannot_be_learnt():
    with pytest.raises(ValueError, match="saturating synapse types"):
        recurrent(BATCH_SIZE, SEED, saturation=C, learn={"tau_decay": True})
    recurrent(BATCH_SIZE, SEED, saturation=C, learn=LEARN)  # the others can


def test_saturation_must_be_a_fraction():
    with pytest.raises(AssertionError, match="saturation"):
        recurrent(BATCH_SIZE, SEED, saturation=1.5)


def test_only_conductance_networks_saturate():
    synapse = dict(synapse_id=0, cell_id=0, name="s", tau_syn=5.0, type="excitatory")
    with pytest.raises(ValueError, match="only conductance-based"):
        CurrentLIFNetwork(
            dt=1.0,
            rec_projections={("e", "e"): FrozenProjection(np.zeros((2, 2)))},
            ff_projections={("in", "e"): FrozenProjection(np.zeros((1, 2)))},
            cell_type_indices=np.zeros(2, dtype=np.int64),
            cell_type_indices_FF=np.zeros(1, dtype=np.int64),
            cell_params=[dict(name="e", cell_id=0, tau_mem=20.0)],
            cell_params_FF=[dict(name="in", cell_id=0)],
            synapse_params=[{**synapse, "saturation": 0.5}],
            synapse_params_FF=[synapse],
            surrgrad_scale=5.0,
            batch_size=1,
        )


def test_configs_carry_saturation():
    from wiretorch.configs.conductance_based import RecurrentLayerConfig

    synapse = dict(tau_rise=[0.1, 2.0], tau_decay=[5.0, 100.0], E_syn=[0.0, 0.0])
    layer = RecurrentLayerConfig(
        cell_types=dict(names=["e"], proportion=[1.0]),
        physiology={
            "e": dict(tau_mem=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0)
        },
        synapses={
            "e": dict(
                names=["AMPA", "NMDA"], g_bar=[1.0, 0.5], saturation=[0.0, C], **synapse
            )
        },
    )
    assert [p["saturation"] for p in layer.get_synapse_params()] == [0.0, C]


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
@pytest.mark.parametrize(
    "extra", [{}, {"tau_ref": 5.0, "mg": 1.0}], ids=["", "refractory-mg"]
)
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
def test_recurrent_kernels_match_reference(recurrent_gradients, extra, monkeypatch):
    """Spikes, state and every gradient — scaling factors and the learnable
    physiology — through both backward passes."""
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, scaling_factors=True, learn=LEARN, saturation=C, **extra
    )
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000], ids=["dense", "sparse"])
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
def test_recurrent_weights_kernels_match_reference(recurrent_gradients, n, monkeypatch):
    """Learnt recurrent weights, whose rows the saturating pathways share."""
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, learnt_weights=True, saturation=C, n=n,
        n_ff=round(n * 80 / 150),
    )  # fmt: skip
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("learnt_weights", [False, True], ids=["sf", "weights"])
def test_feedforward_kernels_match_reference(learnt_weights, monkeypatch):
    """The fused scan, with spike probabilities in the loss."""
    _kernels_vs_reference(
        lambda: feedforward(
            BATCH_SIZE,
            SEED,
            probabilities=True,
            learn=LEARN,
            saturation=C,
            learnt_weights=learnt_weights,
        ),  # fmt: skip
        monkeypatch,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("density", [0.01, 0.5], ids=["sparse", "dense"])
def test_input_gradients_match_reference(density, monkeypatch):
    """The gradient w.r.t. the input (another layer's spikes), scaled by the factors,
    through the sparse and the dense feedforward products."""
    results = []
    for backend in ("triton", "torch"):
        monkeypatch.setenv("WIRETORCH_BACKEND", backend)
        built = feedforward(BATCH_SIZE, SEED, saturation=C)
        model = built.model.to("cuda")
        gen = torch.Generator().manual_seed(0)
        x = torch.rand((BATCH_SIZE, 60, model.n_inputs), generator=gen) < density
        x = x.float().cuda().requires_grad_()
        spikes = model(x)
        readout = torch.randn(spikes.shape, generator=gen).cuda()
        (spikes * readout).sum().backward()
        results.append((spikes.detach(), model.gating.clone(), x.grad))
    (s_k, u_k, g_k), (s_r, u_r, g_r) = results
    assert s_r.sum() > 0 and g_r.abs().sum() > 0
    torch.testing.assert_close(s_k, s_r, rtol=0, atol=0)
    torch.testing.assert_close(u_k, u_r, rtol=0, atol=1e-6)
    torch.testing.assert_close(g_k, g_r, rtol=0, atol=1e-4 * g_r.abs().max().item())
