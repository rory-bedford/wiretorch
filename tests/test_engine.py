"""The GPU engine's building blocks agree with their plain-PyTorch definitions."""

import pytest
import torch

from wiretorch.network_simulators.conductance import engine
from wiretorch.network_simulators.conductance.dynamics import Drive

pytestmark = pytest.mark.gpu


def test_sparse_activations_match_dense():
    """The gather kernel computes ``x[..., rows] @ W`` per pathway, gradients included."""
    torch.manual_seed(0)
    B, T, n_in, N = 3, 20, 300, 70
    x = (torch.rand(B, T, n_in, device="cuda") < 0.02).float().requires_grad_()
    columns = torch.randperm(n_in, device="cuda")
    pathways = [
        columns[:120].sort().values,
        columns[120:250].sort().values,
    ]  # 50 unused
    weights = [
        torch.rand(len(r), N, device="cuda", requires_grad=True) for r in pathways
    ]
    drive = Drive(
        made=torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        scale=torch.ones(2, N, device="cuda"),
        feedforward=list(zip(pathways, weights)),
    )
    probe = torch.randn(2, B, T, N, device="cuda")

    results = []
    for A in (engine._activations(drive, x), drive.feedforward_activations(x)):
        grads = torch.autograd.grad((A * probe).sum(), [x, *weights])
        results.append((A, *grads))
    for sparse, dense in zip(*results):
        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)


def test_current_kernels_with_rest_and_resistance(monkeypatch):
    """Physical units (U_rest, R): kernels against the reference, gradients included."""
    import numpy as np

    from test_configs import _current_network

    results = []
    for backend in ("triton", "torch"):
        monkeypatch.setenv("WIRETORCH_BACKEND", backend)
        model, x = _current_network("cuda", spikes_per_step=0.3, weight=0.1)
        model.track_variables = False
        x.requires_grad_()
        spikes = model(x)
        (
            spikes
            * torch.linspace(-1, 1, spikes.numel(), device="cuda").view_as(spikes)
        ).sum().backward()
        results.append((spikes.detach().cpu(), model.v.cpu(), x.grad.cpu()))
    (s_k, v_k, g_k), (s_r, v_r, g_r) = results
    assert s_k.sum() > 0
    torch.testing.assert_close(s_k, s_r)
    torch.testing.assert_close(v_k, v_r, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(
        g_k, g_r, rtol=1e-3, atol=1e-4 * float(np.abs(g_r).max())
    )


def test_current_presynaptic_product_in_float16():
    """With float16 storage, ``dL/da`` reaches the presynaptic spikes through a dense
    tensor-core product: the cuSPARSE one's result, up to float16 rounding."""
    from wiretorch.network_simulators.current.engine import _presynaptic

    torch.manual_seed(0)
    W = (torch.rand(300, 700, device="cuda") < 0.06) * torch.rand(
        300, 700, device="cuda"
    )
    g = torch.randn(5, 700, device="cuda")
    exact = _presynaptic(W)(g)
    half = _presynaptic(W, torch.float16)(g)
    torch.testing.assert_close(exact, g @ W.T, rtol=1e-5, atol=1e-5)
    assert ((half - exact).norm() / exact.norm()).item() < 2e-3


def test_current_recurrent_float16_keeps_the_gradient():
    """float16 storage, end to end, over a window short enough that no spike flips
    (the network is chaotic: one flip changes the gradient wholesale)."""
    import numpy as np

    from networks import BATCH_SIZE, current_recurrent

    results = []
    for dtype in ("float32", "float16"):
        built = current_recurrent(BATCH_SIZE, 2026, n=1000, n_ff=530)
        model = built.model.to("cuda")
        model.storage_dtype = dtype
        x = torch.from_numpy(built.input_spikes[:, :30]).to("cuda")
        spikes = model(x)
        readout = torch.from_numpy(
            np.random.default_rng(3).standard_normal(spikes.shape).astype(np.float32)
        ).to("cuda")
        (spikes * readout).sum().backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        results.append((spikes.detach(), grads))
    (s32, g32), (s16, g16) = results
    assert s32.sum() > 0
    torch.testing.assert_close(s16, s32, rtol=0, atol=0)
    assert torch.nn.functional.cosine_similarity(g16, g32, dim=0) > 0.999


def test_tracked_steps_stack_contiguously():
    """The reference stacks tracked steps with ``stack_steps``: PyTorch 2.8's CUDA
    ``torch.stack`` of more than 128 non-contiguous tensors — as the synaptic state is,
    a permuted view — writes out of bounds (an illegal memory access at this shape)."""
    from wiretorch.network_simulators.network import stack_steps

    state = torch.randn(4, 2, 3, 500, device="cuda").permute(0, 3, 1, 2)
    steps = [state * (i + 1) for i in range(300)]
    stacked = stack_steps(steps)
    torch.cuda.synchronize()
    assert stacked.shape == (4, 300, 500, 2, 3)
    torch.testing.assert_close(stacked[:, 299], steps[299], rtol=0, atol=0)
