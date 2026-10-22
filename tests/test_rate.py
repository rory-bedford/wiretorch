"""The rate networks: recorded behaviour, invariants, the classic forms, learning.

Rate networks have no spikes to match exactly, so the recorded runs are compared to a
tolerance; everything else mirrors the spiking tests:

* a free run, and a restart from every recorded state (lockstep), reproduce the
  recording; gradients are checked in ``test_gradients.py`` with the rest;
* chunking, batch independence, ``reset_state`` and tracking change nothing;
* set up as the voltage form or the rate form (``rate.dynamics``), the network is
  exactly a direct implementation of that model;
* each nonlinearity is ``gain f(v - theta)``;
* ``learn`` trains exactly what it names, with gradients that match finite differences.
"""

import numpy as np
import pytest
import torch

import fixture_io
import harness
from wiretorch.network_simulators.projections import FrozenProjection
from wiretorch.network_simulators.rate import (
    NONLINEARITIES,
    RecurrentRateNetwork,
    filter_rates,
)
from wiretorch.network_simulators.rate.dynamics import Physiology, rates
from networks import BATCH_SIZE, DT, SCENARIOS, per_type_blocks

RATE = [name for name, s in SCENARIOS.items() if s.rates]
TOLERANCE = {"cpu": 1e-5, "cuda": 1e-4}  # absolute, against each variable's range
#: The network against itself: BLAS rounds differently for different batch sizes and
#: chunk lengths, on either device (up to ~1.5e-6 here, more on some CPUs). A trial
#: leaking into another would differ by the rates themselves.
SELF_TOLERANCE = {"cpu": 1e-5, "cuda": 1e-5}


def build(name, device, batch_size=BATCH_SIZE):
    built = SCENARIOS[name].build(batch_size)
    return built.model.to(device), torch.from_numpy(built.input_spikes).to(device)


def run(model, x, chunk):
    with torch.no_grad():
        out = [model(x[:, s : s + chunk]) for s in range(0, x.shape[1], chunk)]
    return torch.cat(out, dim=1).cpu().numpy()


# =====================================================================
# Against the recording
# =====================================================================


@pytest.mark.parametrize("name", RATE)
def test_free_run_matches_recording(name):
    scenario = SCENARIOS[name]
    recorded, _ = fixture_io.load(name)
    record = harness.free_run(scenario, scenario.build(BATCH_SIZE))
    for key in ("rates", "voltages"):
        atol = TOLERANCE["cpu"] * np.ptp(recorded[key])
        np.testing.assert_allclose(record[key], recorded[key], rtol=0, atol=atol)


@pytest.mark.parametrize("name", RATE)
def test_lockstep(name, device):
    """Restarted from every recorded state, each window reproduces the recording."""
    scenario = SCENARIOS[name]
    recorded, _ = fixture_io.load(name)
    model, x = build(name, device)
    model.track_variables, model.track_batch_idx = True, None
    for w, start in enumerate(range(0, x.shape[1], harness.WINDOW)):
        stop = start + harness.WINDOW
        model.load_checkpoint_state(
            {k: recorded[f"state.{k}"][w] for k in scenario.state_keys}
        )
        with torch.no_grad():
            out = model(x[:, start:stop])
        for key in ("rates", "voltages"):
            atol = TOLERANCE[device] * np.ptp(recorded[key])
            np.testing.assert_allclose(
                out[key].cpu().numpy(),
                recorded[key][:, start:stop],
                rtol=0,
                atol=atol,
                err_msg=f"{key}, window {w}",
            )


# =====================================================================
# Invariants
# =====================================================================


@pytest.mark.parametrize("name", RATE)
def test_chunking_does_not_matter(name, device):
    model, x = build(name, device)
    whole = run(model, x, x.shape[1])
    for chunk in (1, 7, 64):
        model.reset_state()
        tol = SELF_TOLERANCE[device]
        np.testing.assert_allclose(
            run(model, x, chunk), whole, rtol=tol, atol=tol, err_msg=f"chunk {chunk}"
        )


@pytest.mark.parametrize("name", RATE)
def test_trials_are_independent(name, device):
    model, x = build(name, device)
    together = run(model, x, x.shape[1])
    for b in range(BATCH_SIZE):
        alone, _ = build(name, device, batch_size=1)
        np.testing.assert_allclose(
            run(alone, x[b : b + 1], x.shape[1])[0],
            together[b],
            rtol=SELF_TOLERANCE[device],
            atol=SELF_TOLERANCE[device],
            err_msg=f"trial {b}",
        )


@pytest.mark.parametrize("name", RATE)
def test_reset_restarts(name, device):
    model, x = build(name, device)
    first = run(model, x, 50)
    model.reset_state()
    np.testing.assert_array_equal(run(model, x, 50), first)


@pytest.mark.parametrize("name", RATE)
def test_tracking_does_not_change_dynamics(name, device):
    model, x = build(name, device)
    plain = run(model, x, 50)
    model.reset_state()
    model.track_variables, model.track_batch_idx = True, 0
    with torch.no_grad():
        tracked = model(x[:, :50])
    # Tracking compiles a separate kernel, which may round differently.
    tol = SELF_TOLERANCE[device]
    np.testing.assert_allclose(
        tracked["rates"].cpu().numpy(), plain[:, :50], rtol=tol, atol=tol
    )
    assert tracked["voltages"].shape == (1, 50, model.n_neurons)


# =====================================================================
# The classic forms (rate.dynamics)
# =====================================================================

N_SMALL, N_IN, T = 40, 15, 120
E, I = 0, 1
NAMES = ["excitatory", "inhibitory"]


def small_network(tau_mem, tau_exc, tau_inh, tau_input, nonlinearities):
    """A small ``RecurrentRateNetwork`` with fixed random weights, and its matrices."""
    rng = np.random.default_rng(11)
    types = np.where(np.arange(N_SMALL) < 30, E, I)
    W = (rng.random((N_SMALL, N_SMALL)) < 0.3) * rng.random((N_SMALL, N_SMALL)) * 0.3
    W[types == I] *= 2.0
    W_ff = (rng.random((N_IN, N_SMALL)) < 0.4) * rng.random((N_IN, N_SMALL)) * 0.3
    W, W_ff = W.astype(np.float32), W_ff.astype(np.float32)
    in_types = np.zeros(N_IN, dtype=np.int64)
    cells = [
        dict(name=NAMES[c], cell_id=c, tau_mem=tau_mem[c],
             nonlinearity=nonlinearities[c], theta=0.2 * c, gain=1.0 + c)
        for c in (E, I)
    ]  # fmt: skip
    synapses = [
        dict(name="exc", synapse_id=0, cell_id=E, tau_syn=tau_exc, type="excitatory"),
        dict(name="inh", synapse_id=1, cell_id=I, tau_syn=tau_inh, type="inhibitory"),
    ]
    model = RecurrentRateNetwork(
        dt=DT,
        rec_projections={
            k: FrozenProjection(b)
            for k, b in per_type_blocks(W, types, types, NAMES, NAMES).items()
        },
        ff_projections={
            k: FrozenProjection(b)
            for k, b in per_type_blocks(W_ff, in_types, types, ["input"], NAMES).items()
        },
        cell_type_indices=types,
        cell_type_indices_FF=in_types,
        cell_params=cells,
        cell_params_FF=[dict(name="input", cell_id=0)],
        synapse_params=synapses,
        synapse_params_FF=[
            dict(
                name="ff", synapse_id=0, cell_id=0, tau_syn=tau_input, type="excitatory"
            )
        ],
        batch_size=2,
    )
    x = torch.from_numpy((1.0 + rng.standard_normal((2, T, N_IN))).astype(np.float32))
    sign = np.where(types == E, 1.0, -1.0).astype(np.float32)[:, None]
    signed = torch.from_numpy(W * sign)  # W_ij: from i to j, with i's sign

    def phi(u):
        out = torch.empty_like(u)
        for c in (E, I):
            cell = cells[c]
            neurons = torch.from_numpy(np.flatnonzero(types == c))
            f = NONLINEARITIES[cell["nonlinearity"]]
            out[:, neurons] = cell["gain"] * f(u[:, neurons] - cell["theta"])
        return out

    per_neuron = torch.from_numpy(types)
    return model, x, signed, torch.from_numpy(W_ff), phi, per_neuron


@pytest.mark.parametrize("nonlinearities", [("relu", "relu"), ("softplus", "sigmoid")])
def test_voltage_form(nonlinearities):
    """Every tau_syn = 0: ``tau dx/dt = -x + W phi(x) + W_ff u`` (exponential Euler)."""
    tau = torch.tensor([10.0, 5.0])
    model, x, W, W_ff, phi, types = small_network(
        tau.tolist(), 0.0, 0.0, 0.0, nonlinearities
    )
    beta = torch.exp(-DT / tau)[types]
    v, expected = torch.zeros(2, N_SMALL), []
    for t in range(T):
        r = phi(v)
        expected.append(r)
        v = beta * v + (1 - beta) * (r @ W + x[:, t] @ W_ff)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, torch.stack(expected, 1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("nonlinearities", [("relu", "relu"), ("softplus", "sigmoid")])
def test_rate_form(nonlinearities):
    """tau_mem = 0, recurrent tau_syn = tau (per source type), input tau_syn = 0:
    ``tau dr/dt = -r + phi(W r + W_ff u)`` is the output filtered by tau, with the input
    one step behind."""
    tau = torch.tensor([10.0, 5.0])
    model, x, W, W_ff, phi, types = small_network(
        [0.0, 0.0], tau[E].item(), tau[I].item(), 0.0, nonlinearities
    )
    alpha = torch.exp(-DT / tau)[types]
    r, u_prev, expected = torch.zeros(2, N_SMALL), torch.zeros(2, N_SMALL), []
    for t in range(T):
        r = alpha * r + (1 - alpha) * phi(r @ W + u_prev)
        expected.append(r)
        u_prev = x[:, t] @ W_ff
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(
        filter_rates(out, tau[types], DT),
        torch.stack(expected, 1),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("name", sorted(NONLINEARITIES))
def test_nonlinearity(name):
    v = torch.linspace(-3, 3, 61)[None].repeat(2, 1)
    neurons = torch.arange(61)
    p = Physiology(
        theta=torch.full((61,), 0.5),
        gain=torch.full((61,), 2.0),
        beta=torch.zeros(61),
        sign=torch.ones(1),
        decay=torch.zeros(1),
        groups=[(name, neurons[::2]), ("relu", neurons[1::2])],
    )
    out = rates(v, p)
    f = NONLINEARITIES[name]
    torch.testing.assert_close(out[:, ::2], 2.0 * f(v[:, ::2] - 0.5))
    torch.testing.assert_close(out[:, 1::2], 2.0 * torch.relu(v[:, 1::2] - 0.5))


# =====================================================================
# Learning the physiology
# =====================================================================


def test_learn_trains_exactly_what_it_names():
    model, x = build("rate_recurrent_learnt", "cpu")
    learnt = dict(model.named_parameters())
    physiology = {k for k in learnt if k.startswith(("cell_", "synapse_"))}
    assert physiology == {
        "cell_parameters.log_tau_mem",
        "cell_parameters.theta",
        "cell_parameters.log_gain",
        "synapse_parameters.log_tau_syn",
    }
    assert learnt["cell_parameters.theta"].shape == (1,)  # excitatory only
    assert learnt["synapse_parameters.log_tau_syn"].shape == (1,)  # "slow" only
    out = model(x[:, :50])
    (out * torch.randn_like(out)).sum().backward()
    for name in physiology:
        assert learnt[name].grad is not None and learnt[name].grad.abs().sum() > 0, name
    # Fixed values stay buffers, and the learnt ones start at their configured values.
    cell = model.cell_parameters
    torch.testing.assert_close(cell.value("tau_mem"), torch.tensor([10.0, 5.0]))
    torch.testing.assert_close(cell.value("theta"), torch.tensor([0.5, 0.0]))


def test_learnt_time_constants_must_be_positive():
    with pytest.raises(AssertionError, match="positive"):  # its tau_syn are 0
        RecurrentRateNetwork(**_small_kwargs(learn={"tau_syn": True}))


def _small_kwargs(**extra):
    """``small_network``'s constructor arguments (tau_syn = 0), for variants."""
    model = small_network([10.0, 5.0], 0.0, 0.0, 0.0, ("relu", "relu"))[0]
    return dict(
        dt=DT,
        rec_projections={tuple(k.split("__")): p for k, p in model.rec_projections.items()},
        ff_projections={tuple(k.split("__")): p for k, p in model.ff_projections.items()},
        cell_type_indices=model.cell_type_indices.numpy(),
        cell_type_indices_FF=model.cell_type_indices_FF.numpy(),
        cell_params=model.cell_params,
        cell_params_FF=model.cell_params_FF,
        synapse_params=model.synapse_params,
        synapse_params_FF=model.synapse_params_FF,
        batch_size=2,
        **extra,
    )  # fmt: skip


def test_unknown_learnable_is_rejected():
    with pytest.raises(AssertionError, match="unknown"):
        RecurrentRateNetwork(**_small_kwargs(learn={"tau_rise": True}))
    with pytest.raises(AssertionError, match="unknown types"):
        RecurrentRateNetwork(**_small_kwargs(learn={"theta": ["pyramidal"]}))


def test_physiology_gradients_match_finite_differences():
    """d loss / d(learnt physiology), against central differences, in float64."""
    model, x = build("rate_recurrent_learnt", "cpu")
    model = model.double()
    x = x[:, :40].double()
    readout = torch.randn(BATCH_SIZE, 40, model.n_neurons, dtype=torch.float64)

    def loss():
        model.reset_state()
        model.v, model.I_syn = model.v.double(), model.I_syn.double()
        return (model(x) * readout).sum()

    parameters = {
        k: p for k, p in model.named_parameters() if k.startswith(("cell_", "synapse_"))
    }
    model.zero_grad()
    loss().backward()
    eps = 1e-6
    for name, p in parameters.items():
        for i in range(p.numel()):
            with torch.no_grad():
                p.view(-1)[i] += eps
                up = loss().item()
                p.view(-1)[i] -= 2 * eps
                down = loss().item()
                p.view(-1)[i] += eps
            numeric = (up - down) / (2 * eps)
            assert p.grad.view(-1)[i].item() == pytest.approx(
                numeric, rel=1e-4, abs=1e-6
            ), f"{name}[{i}]"


# =====================================================================
# The Triton kernels against the reference (GPU)
# =====================================================================


def _kernels_vs_reference(builder, monkeypatch, steps=60):
    """Rates, final state and every gradient, kernels against the PyTorch reference,
    both on the GPU from the same network."""
    results = []
    for backend in ("triton", "torch"):
        monkeypatch.setenv("WIRETORCH_BACKEND", backend)
        built = builder()
        model = built.model.to("cuda")
        x = torch.from_numpy(built.input_spikes[:, :steps]).to("cuda")
        out = model(x)
        readout = torch.from_numpy(
            np.random.default_rng(3).standard_normal(out.shape).astype(np.float32)
        ).to("cuda")
        (out * readout).sum().backward()
        grads = {
            k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None
        }
        results.append((out.detach(), model.v.clone(), model.I_syn.clone(), grads))
    (out_k, v_k, I_k, g_k), (out_r, v_r, I_r, g_r) = results
    torch.testing.assert_close(out_k, out_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v_k, v_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(I_k, I_r, rtol=1e-4, atol=1e-5)
    assert g_k.keys() == g_r.keys() and g_k
    for key in g_r:
        scale = g_r[key].abs().max().item() or 1.0
        torch.testing.assert_close(
            g_k[key], g_r[key], rtol=0, atol=1e-4 * scale, msg=key
        )
    return g_r


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000], ids=["dense-I", "sparse"])
def test_recurrent_kernels_match_reference(n, monkeypatch):
    """Learnt recurrent weights (full rank), scaling factors and all four physiology
    parameters. At n = 150 the inhibitory rows are 30% dense (a dense product); at
    1000 every pathway is sparse (cuSPARSE)."""
    from networks import RATE_LEARN, rate_recurrent

    def builder():
        return rate_recurrent(
            BATCH_SIZE, 2026, learn=RATE_LEARN, n=n, n_ff=round(n * 80 / 150),
            learnt_excitatory=True,
        )  # fmt: skip

    grads = _kernels_vs_reference(builder, monkeypatch)
    assert any("rec_projections.excitatory" in k and "log_weights" in k for k in grads)
    assert {"cell_parameters.theta", "synapse_parameters.log_tau_syn"} <= grads.keys()


@pytest.mark.gpu
def test_feedforward_kernels_match_reference(monkeypatch):
    _kernels_vs_reference(
        lambda: SCENARIOS["rate_feedforward"].build(BATCH_SIZE), monkeypatch
    )


@pytest.mark.gpu
@pytest.mark.parametrize("n", [150, 1000])
def test_float16_products(n):
    """``storage_dtype="float16"``: products on tensor cores. Rate networks are not
    chaotic, so the rates stay within ~1% and the gradient keeps its direction."""
    from networks import RATE_LEARN, rate_recurrent

    results = []
    for dtype in ("float32", "float16"):
        built = rate_recurrent(
            BATCH_SIZE, 2026, learn=RATE_LEARN, n=n, n_ff=round(n * 80 / 150),
            learnt_excitatory=True,
        )  # fmt: skip
        model = built.model.to("cuda")
        model.storage_dtype = dtype
        x = torch.from_numpy(built.input_spikes).to("cuda")
        out = model(x)
        readout = torch.from_numpy(
            np.random.default_rng(3).standard_normal(out.shape).astype(np.float32)
        ).to("cuda")
        (out * readout).sum().backward()
        grads = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        results.append((out.detach(), grads))
    (out32, g32), (out16, g16) = results
    assert (out16 - out32).abs().max() <= 0.01 * out32.abs().max()
    assert torch.nn.functional.cosine_similarity(g16, g32, dim=0) > 0.999
