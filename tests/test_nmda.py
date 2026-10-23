"""The voltage-dependent Mg²⁺ block of NMDA synapses (conductance-based networks).

A synapse type with ``mg > 0`` scales its current by
``B(v) = 1 / (1 + (mg / mg_k) exp(-mg_slope v))`` (Jahr & Stevens); its conductance is
unchanged. Checked here: the tracked currents against the formula, that ``mg = 0`` is
the former network, configs, and the Triton kernels against the PyTorch reference,
gradients included, in both backward passes and with the refractory period.
"""

import pytest
import torch

from networks import BATCH_SIZE, feedforward, recurrent
from test_learnt_physiology import CONDUCTANCE_LEARN, _kernels_vs_reference

SEED = 2026
MG = 1.0  # mM, the physiological concentration


def _run(built, device, steps=60):
    model = built.model.to(device)
    x = torch.from_numpy(built.input_spikes[:, :steps]).to(device)
    return model(x)


def test_block_tables():
    """``mg_c = mg / mg_k`` and ``mg_a = mg_slope``, with the Jahr–Stevens defaults; a
    synapse type without ``mg`` has no block."""
    model = recurrent(BATCH_SIZE, SEED, mg=MG).model
    names = [s["name"] for s in model.unified_synapse_params]
    nmda = torch.tensor([name == "NMDA" for name in names])
    torch.testing.assert_close(model.mg_c, torch.where(nmda, MG / 3.57, 0.0))
    torch.testing.assert_close(model.mg_a, torch.full((len(names),), 0.062))
    assert model._mg_block and model._physiology().mg_c is not None
    assert not recurrent(BATCH_SIZE, SEED).model._mg_block


def test_configs_carry_the_block():
    from wiretorch.configs.conductance_based import RecurrentLayerConfig

    synapse = dict(tau_rise=[0.1, 10.0], tau_decay=[5.0, 60.0], E_syn=[0.0, 0.0])
    layer = RecurrentLayerConfig(
        cell_types=dict(names=["e"], proportion=[1.0]),
        physiology={
            "e": dict(tau_mem=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0)
        },
        synapses={
            "e": dict(names=["AMPA", "NMDA"], g_bar=[1.0, 0.5], mg=[0.0, MG], **synapse)
        },
    )
    params = layer.get_synapse_params()
    assert [p["mg"] for p in params] == [0.0, MG]
    assert all("mg_slope" not in p for p in params)  # unset: the defaults


@pytest.mark.parametrize("build", [recurrent, feedforward], ids=["rec", "ff"])
def test_no_block_is_the_former_network(build, device):
    """``mg = 0`` (and absent: the recorded fixtures) changes nothing."""
    runs = []
    for mg in (None, 0.0):
        built = build(BATCH_SIZE, SEED, mg=mg)
        spikes = _run(built, device)
        runs.append((spikes.detach(), built.model.v.clone()))
    torch.testing.assert_close(runs[0], runs[1], rtol=0, atol=0)


def test_block_changes_the_network():
    """At the test networks' voltages the block binds: it changes their spikes."""
    off, on = (
        _run(recurrent(BATCH_SIZE, SEED, mg=mg), "cpu", 150) for mg in (None, MG)
    )
    assert on.sum() > 0 and not torch.equal(off, on)


def test_tracked_currents_follow_the_formula(device):
    """Each step's tracked current of synapse type ``j`` is
    ``B_j(v) clip(g_j) (v - E_j)`` at the voltage and conductances entering the step
    (the previous step's tracked ones), with ``B = 1`` for the types without a block."""
    built = recurrent(BATCH_SIZE, SEED, mg=MG)
    model = built.model.to(device)
    model.track_variables, model.track_batch_idx = True, None
    out = model(torch.from_numpy(built.input_spikes[:, :60]).to(device))
    v = out["voltages"][:, :-1].double()  # entering steps 1..
    G = out["conductances"][:, :-1].sum(dim=3).double()  # (B, T, N, S)
    G = torch.minimum(G.clamp(min=0.0), model.g_clip.double())
    E, c, a = (getattr(model, k).double() for k in ("E_syn", "mg_c", "mg_a"))
    B = 1 / (1 + c * torch.exp(-a * v[..., None]))
    expected = B * G * (v[..., None] - E)
    actual = out["currents"][:, 1:].double()
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6 * scale)
    nmda = [s["name"] == "NMDA" for s in model.unified_synapse_params]
    assert (B[..., torch.tensor(nmda)] < 0.5).float().mean() > 0.5  # it binds
    assert (B[..., ~torch.tensor(nmda)] == 1).all()


def test_checkpoints_load_either_way():
    """The block comes from the synapse parameters, not the saved module: state dicts
    load between networks with and without it."""
    without, with_block = (
        recurrent(BATCH_SIZE, SEED, mg=mg).model for mg in (None, MG)
    )
    with_block.load_state_dict(without.state_dict())
    without.load_state_dict(with_block.state_dict())
    assert with_block._mg_block and not without._mg_block


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
@pytest.mark.parametrize("tau_ref", [None, 5.0], ids=["", "refractory"])
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
def test_recurrent_kernels_match_reference(recurrent_gradients, tau_ref, monkeypatch):
    """Spikes, state and every gradient — scaling factors and all learnable physiology,
    whose terms the block enters — through both backward passes."""
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, scaling_factors=True, learn=CONDUCTANCE_LEARN, mg=MG,
        tau_ref=tau_ref,
    )  # fmt: skip
    _kernels_vs_reference(_with_flag(builder, recurrent_gradients), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000], ids=["dense", "sparse"])
def test_recurrent_weights_kernels_match_reference(n, monkeypatch):
    """Learnt recurrent weights, the gradient through the recurrent spikes."""
    builder = lambda: recurrent(  # noqa: E731
        BATCH_SIZE, SEED, learnt_weights=True, mg=MG, n=n, n_ff=round(n * 80 / 150)
    )
    _kernels_vs_reference(_with_flag(builder, True), monkeypatch)


@pytest.mark.gpu
def test_feedforward_kernels_match_reference(monkeypatch):
    """The fused scan, with spike probabilities in the loss."""
    _kernels_vs_reference(
        lambda: feedforward(
            BATCH_SIZE, SEED, probabilities=True, learn=CONDUCTANCE_LEARN, mg=MG
        ),
        monkeypatch,
    )
