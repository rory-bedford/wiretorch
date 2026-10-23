"""Gradients through the recurrent spikes, or not (``recurrent_gradients``).

Either family can carry the gradient through the recurrent spikes into the
presynaptic neurons' surrogates, or treat each step's recurrent input as given. The
switch changes the backward pass only, and the Triton kernels' gradients match the
PyTorch reference's both ways: one reverse scan without it, a loop over steps with it.
The defaults (conductance-based off, current-based on) are what the recorded
gradient fixtures check.
"""

import inspect

import numpy as np
import pytest
import torch

from networks import BATCH_SIZE, current_recurrent, recurrent
from test_learnt_physiology import (
    CONDUCTANCE_LEARN,
    CURRENT_LEARN,
    _kernels_vs_reference,
)
from wiretorch.network_simulators.conductance.recurrent import ConductanceLIFNetwork
from wiretorch.network_simulators.current.recurrent import CurrentLIFNetwork

SEED = 2026
STEPS = 60


def _conductance(n=150, **kwargs):
    return recurrent(BATCH_SIZE, SEED, n=n, n_ff=round(n * 80 / 150), **kwargs)


def _current(n=150, **kwargs):
    return current_recurrent(BATCH_SIZE, SEED, n=n, n_ff=round(n * 80 / 150), **kwargs)


#: Recurrent networks whose gradients reach scaling factors and every learnable
#: physiology parameter, or the recurrent weights themselves.
NETWORKS = {
    "conductance": lambda n=150: _conductance(
        n, scaling_factors=True, learn=CONDUCTANCE_LEARN
    ),
    "conductance_weights": lambda n=150: _conductance(n, learnt_weights=True),
    "current": lambda n=150: _current(n, learn=CURRENT_LEARN),
    "current_weights": lambda n=150: _current(n, optimisable="weights"),
}


def _with(builder, recurrent_gradients, n=150):
    """``builder``'s network, with gradients through its recurrent spikes or not."""

    def build():
        built = builder(n)
        built.model.recurrent_gradients = recurrent_gradients
        return built

    return build


def _run(built, device):
    """Spikes and every parameter's gradient of a fixed readout of them."""
    model = built.model.to(device)
    x = torch.from_numpy(built.input_spikes[:, :STEPS]).to(device)
    spikes = model(x)
    readout = np.random.default_rng(3).standard_normal(spikes.shape)
    (spikes * torch.from_numpy(readout.astype(np.float32)).to(device)).sum().backward()
    grads = {
        k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None
    }
    return spikes.detach(), grads


def test_defaults():
    """Off for the conductance-based networks, on for the current-based ones (and
    ``from_weights`` passes it on)."""

    def default(function):
        return inspect.signature(function).parameters["recurrent_gradients"].default

    assert default(ConductanceLIFNetwork) is False
    assert default(CurrentLIFNetwork) is True
    assert default(CurrentLIFNetwork.from_weights) is True
    assert _conductance().model._drive().recurrent_gradients is False
    model = _current().model  # built by from_weights
    assert model._drive().recurrent_gradients is True
    model.recurrent_gradients = False
    assert model._drive().recurrent_gradients is False


@pytest.mark.parametrize("name", NETWORKS)
def test_only_the_gradient_changes(name, device):
    """The same spikes either way; gradients that differ, by the recurrent term."""
    (s_off, g_off), (s_on, g_on) = (
        _run(_with(NETWORKS[name], flag)(), device) for flag in (False, True)
    )
    assert s_off.sum() > 0
    torch.testing.assert_close(s_on, s_off, rtol=0, atol=0)
    assert g_on.keys() == g_off.keys() and g_on
    flat = [torch.cat([g[k].flatten() for k in sorted(g)]) for g in (g_off, g_on)]
    assert not torch.allclose(flat[0], flat[1], rtol=1e-3, atol=0)


@pytest.mark.parametrize("name", ["conductance", "current"])
def test_no_recurrent_weights_no_difference(name, device):
    """With the recurrent weights at zero, nothing reaches the presynaptic spikes:
    the gradients agree to rounding."""
    results = []
    for flag in (False, True):
        built = _with(NETWORKS[name], flag)()
        with torch.no_grad():
            for pathway in built.model.pathways:
                if built.model._sides[pathway.side].recurrent:
                    getattr(built.model, f"_weights_{pathway.index}").zero_()
        results.append(_run(built, device))
    (s_off, g_off), (s_on, g_on) = results
    torch.testing.assert_close(s_on, s_off, rtol=0, atol=0)
    for key in g_off:
        scale = g_off[key].abs().max().item() or 1.0
        torch.testing.assert_close(
            g_on[key], g_off[key], rtol=0, atol=1e-6 * scale, msg=key
        )


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("name", NETWORKS)
def test_kernels_match_reference(name, recurrent_gradients, monkeypatch):
    """Spikes, state and every gradient: the kernels' backward pass either way."""
    _kernels_vs_reference(_with(NETWORKS[name], recurrent_gradients), monkeypatch)


@pytest.mark.gpu
@pytest.mark.parametrize("recurrent_gradients", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("name", ["conductance", "current"])
def test_kernels_match_reference_sparse(name, recurrent_gradients, monkeypatch):
    """At n = 1000 the gradient reaches the presynaptic spikes by a sparse product."""
    _kernels_vs_reference(
        _with(NETWORKS[name], recurrent_gradients, n=1000), monkeypatch
    )
