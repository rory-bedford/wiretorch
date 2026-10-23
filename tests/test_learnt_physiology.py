"""Learnt physiology in the spiking networks.

``learn`` trains exactly the parameters it names, starting from the configured values,
and the Triton kernels' gradients of them match the PyTorch reference's. The reference's
are autograd's, through the time step written out in ``dynamics.py``; the kernels
compute the same adjoints by hand. (Finite differences do not apply: spikes are steps,
and the gradient is the surrogate's.)
"""

import numpy as np
import pytest
import torch

import harness
from networks import (
    BATCH_SIZE,
    current_feedforward_probabilities,
    current_recurrent,
    feedforward,
    recurrent,
)

SEED = 2026
#: Every learnable parameter of the current-based networks, for every type.
CURRENT_LEARN = {
    "tau_mem": True,
    "theta": True,
    "U_reset": True,
    "U_rest": True,
    "R": True,
    "tau_syn": True,
}
#: The same for the feedforward network, whose "instant" synapse (tau_syn = 0) has no
#: time constant to learn.
CURRENT_FF_LEARN = {**CURRENT_LEARN, "tau_syn": ["exc", "inh"]}
#: Every learnable parameter of the conductance-based networks, for every type.
CONDUCTANCE_LEARN = dict.fromkeys(
    ["tau_mem", "theta", "U_reset", "E_L", "g_L"]
    + ["tau_rise", "tau_decay", "E_syn", "g_bar"],
    True,
)


def _physiology(model):
    return {k: v for k, v in vars(model._physiology()).items() if torch.is_tensor(v)}


def test_current_learn_trains_exactly_what_it_names():
    learn = {"tau_mem": ["inhibitory"], "theta": True, "tau_syn": ["inh"]}
    built = current_recurrent(BATCH_SIZE, SEED, learn=learn)
    model = built.model
    parameters = dict(model.named_parameters())
    physiology = {k for k in parameters if k.startswith(("cell_", "synapse_"))}
    assert physiology == {
        "cell_parameters.log_tau_mem",
        "cell_parameters.theta",
        "synapse_parameters.log_tau_syn",
    }
    assert parameters["cell_parameters.log_tau_mem"].shape == (1,)  # inhibitory only
    assert parameters["synapse_parameters.log_tau_syn"].shape == (1,)  # "inh" only

    # Before training, the physiology is the configured one.
    fixed = current_recurrent(BATCH_SIZE, SEED).model
    for name, value in _physiology(fixed).items():
        torch.testing.assert_close(
            _physiology(model)[name].detach(), value, rtol=1e-6, atol=0, msg=name
        )

    spikes = model(torch.from_numpy(built.input_spikes[:, :100]))
    assert spikes.sum() > 0
    (spikes * torch.randn_like(spikes)).sum().backward()
    for name in physiology:
        assert parameters[name].grad.abs().sum() > 0, name


def test_conductance_learn_trains_exactly_what_it_names():
    learn = {"g_L": ["inhibitory"], "E_L": True, "tau_decay": ["NMDA"], "g_bar": True}
    built = recurrent(BATCH_SIZE, SEED, scaling_factors=True, learn=learn)
    model = built.model
    parameters = dict(model.named_parameters())
    physiology = {k for k in parameters if k.startswith(("cell_", "synapse_"))}
    assert physiology == {
        "cell_parameters.log_g_L",
        "cell_parameters.E_L",
        "synapse_parameters.log_tau_decay",
        "synapse_parameters.log_g_bar",
    }
    assert parameters["cell_parameters.log_g_L"].shape == (1,)  # inhibitory only
    # NMDA is made by the excitatory neurons and by the mitral inputs: two synapse types
    # of that name (a type is its source cell type and name), and naming it learns both.
    assert parameters["synapse_parameters.log_tau_decay"].shape == (2,)

    fixed = recurrent(BATCH_SIZE, SEED, scaling_factors=True).model
    for name, value in _physiology(fixed).items():
        torch.testing.assert_close(
            _physiology(model)[name].detach(), value, rtol=1e-5, atol=0, msg=name
        )

    spikes = model(torch.from_numpy(built.input_spikes[:, :100]))
    assert spikes.sum() > 0
    (spikes * torch.randn_like(spikes)).sum().backward()
    for name in physiology:
        assert parameters[name].grad.abs().sum() > 0, name


def test_current_learn_rejects_what_it_cannot_learn():
    with pytest.raises(AssertionError, match="unknown parameters"):
        current_recurrent(BATCH_SIZE, SEED, learn={"gain": True})
    with pytest.raises(AssertionError, match="unknown types"):
        current_recurrent(BATCH_SIZE, SEED, learn={"theta": ["pyramidal"]})
    with pytest.raises(AssertionError, match="positive"):  # the "instant" synapse
        current_feedforward_probabilities(BATCH_SIZE, SEED, learn={"tau_syn": True})


# =====================================================================
# The Triton kernels against the reference (GPU)
# =====================================================================


def _kernels_vs_reference(builder, monkeypatch, warm_up=30, steps=60):
    """Spikes, final state and every gradient, kernels against the PyTorch reference,
    both on the GPU from the same network. The loss reads out the spikes, and the
    spike probabilities if the network returns them.

    The learnt physiology is moved off its defaults (natural units would leave, say,
    ``U_rest`` at 0, where its terms vanish), and a warm-up chunk without gradients
    runs first, so the chunk starts from currents that are not zero.

    A gradient that sums terms which nearly cancel is known to float32 only to a few
    digits, the reference's as much as the kernels': one that misses ``1e-3`` of its
    scale must then be as close as the reference to a float64 run of the reference.
    """
    s_k, v_k, g_k, u_k = _gradients(builder, monkeypatch, "triton", warm_up, steps)
    s_r, v_r, g_r, u_r = _gradients(builder, monkeypatch, "torch", warm_up, steps)
    assert s_r.sum() > 0
    torch.testing.assert_close(s_k, s_r, rtol=0, atol=0)
    torch.testing.assert_close(v_k, v_r, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(u_k, u_r, rtol=1e-5, atol=1e-6)  # presynaptic state
    assert g_k.keys() == g_r.keys() and g_k
    g_64 = None
    for key in g_r:
        scale = g_r[key].abs().max().item() or 1.0
        if torch.allclose(g_k[key], g_r[key], rtol=0, atol=1e-3 * scale):
            continue
        if g_64 is None:
            g_64 = _gradients(builder, monkeypatch, "torch", warm_up, steps, True)[2]
        exact = g_64[key]
        error_k = (g_k[key].double() - exact).abs().max().item()
        error_r = (g_r[key].double() - exact).abs().max().item()
        assert error_k <= 2 * error_r + 1e-4 * scale, (key, error_k, error_r)
    return g_r


def _gradients(builder, monkeypatch, backend, warm_up, steps, float64=False):
    """Spikes, final voltage, parameter gradients and presynaptic state (``gating``,
    empty if none) of ``_kernels_vs_reference``'s run on ``backend``, in float64 if
    asked."""
    monkeypatch.setenv("WIRETORCH_BACKEND", backend)
    built = builder()
    model = (built.model.double() if float64 else built.model).to("cuda")
    rng = np.random.default_rng(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(("cell_", "synapse_")):
                p += torch.from_numpy(
                    0.1 * rng.standard_normal(p.shape).astype(np.float32)
                ).to(p)
        model(torch.from_numpy(built.input_spikes[:, :warm_up]).to("cuda"))
    x = torch.from_numpy(built.input_spikes[:, warm_up : warm_up + steps])
    out = harness.unpack_output(model(x.to("cuda")))
    rng = np.random.default_rng(3)
    loss = 0
    for key in ("spikes", "probabilities"):
        if out.get(key) is not None:
            readout = rng.standard_normal(out[key].shape).astype(np.float32)
            loss = loss + (out[key] * torch.from_numpy(readout).to(out[key])).sum()
    loss.backward()
    grads = {
        k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None
    }
    gating = getattr(model, "gating", torch.zeros(0, device="cuda"))
    return out["spikes"].detach(), model.v.clone(), grads, gating.clone()


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000], ids=["dense", "sparse"])
def test_current_recurrent_kernels_match_reference(n, monkeypatch):
    """Scaling factors and every physiology parameter. At n = 1000 the gradient is
    carried back through the recurrent weights by a sparse product."""
    grads = _kernels_vs_reference(
        lambda: current_recurrent(
            BATCH_SIZE, SEED, n=n, n_ff=round(n * 80 / 150), learn=CURRENT_LEARN
        ),
        monkeypatch,
    )
    learnt = {k for k in grads if k.startswith(("cell_", "synapse_"))}
    assert learnt == {
        "cell_parameters.log_tau_mem",
        "cell_parameters.theta",
        "cell_parameters.U_reset",
        "cell_parameters.U_rest",
        "cell_parameters.log_R",
        "synapse_parameters.log_tau_syn",
    }


@pytest.mark.gpu
def test_current_feedforward_kernels_match_reference(monkeypatch):
    """The fused scan, with spike probabilities in the loss."""
    _kernels_vs_reference(
        lambda: current_feedforward_probabilities(
            BATCH_SIZE, SEED, learn=CURRENT_FF_LEARN
        ),
        monkeypatch,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000])
def test_conductance_recurrent_kernels_match_reference(n, monkeypatch):
    """Scaling factors and every physiology parameter, through the recurrent step and
    the reverse scan."""
    grads = _kernels_vs_reference(
        lambda: recurrent(
            BATCH_SIZE,
            SEED,
            scaling_factors=True,
            n=n,
            n_ff=round(n * 80 / 150),
            learn=CONDUCTANCE_LEARN,
        ),  # fmt: skip
        monkeypatch,
    )
    learnt = {k for k in grads if k.startswith(("cell_", "synapse_"))}
    assert len(learnt) == len(CONDUCTANCE_LEARN)


@pytest.mark.gpu
def test_conductance_feedforward_kernels_match_reference(monkeypatch):
    """The fused scan, with spike probabilities in the loss."""
    _kernels_vs_reference(
        lambda: feedforward(
            BATCH_SIZE, SEED, probabilities=True, learn=CONDUCTANCE_LEARN
        ),
        monkeypatch,
    )
