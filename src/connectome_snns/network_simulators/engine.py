"""What every family's engine shares: when to use the kernels, and the synaptic input.

Each family's ``engine.simulate`` runs its Triton kernels on CUDA and its PyTorch
reference (``dynamics.simulate``) otherwise. Set ``CONNECTOME_SNNS_BACKEND=torch`` to
force the reference everywhere.

Feedforward input is known for a whole chunk, so its synaptic drive — every pathway's
input times its weights — is computed up front by ``activations``: dense matmuls, or one
cuSPARSE product per pathway when the input is sparse (spike trains).
"""

from __future__ import annotations

import functools
import itertools
import os
import warnings

import torch

from connectome_snns.network_simulators.network import Drive

warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")

try:
    import triton
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

#: Use the sparse path when at most this fraction of inputs is non-zero.
SPARSE_INPUT = 0.1


def uses_kernels(x: torch.Tensor, retain) -> bool:
    """Whether to run the Triton kernels: on CUDA, unless per-step gradients are kept."""
    return (
        triton is not None
        and x.is_cuda
        and retain is None
        and os.environ.get("CONNECTOME_SNNS_BACKEND", "triton") != "torch"
    )


def pad(t: torch.Tensor, SP: int) -> torch.Tensor:
    """``t`` with its last dimension zero-padded to ``SP``, contiguous."""
    out = t.new_zeros((*t.shape[:-1], SP))
    out[..., : t.shape[-1]] = t
    return out.contiguous()


@functools.lru_cache(maxsize=64)
def row_start(sizes: tuple[int, ...], device) -> torch.Tensor:
    """Where each recurrent pathway's rows begin, plus the end: ``(K + 1,)`` int32.

    Cached, so a forward call copies nothing from the host (a small copy would wait
    behind the trainer's prefetch of the next batch).
    """
    return torch.tensor(
        [0, *itertools.accumulate(sizes)], device=device, dtype=torch.int32
    )


class _SparseActivations(torch.autograd.Function):
    """Feedforward activations of sparse input, one cuSPARSE product per pathway.

    Differentiable in the input and the weights; both gradients are dense matmuls
    (they are needed only for learnt weights, or when the input is another layer's
    spikes).
    """

    @staticmethod
    def forward(ctx, x2d, rows, differentiable, *weights):
        ctx.rows, ctx.differentiable = rows, differentiable
        ctx.save_for_backward(x2d, *weights)
        x2d = x2d.detach()
        A = x2d.new_empty((len(rows), x2d.shape[0], weights[0].shape[1]))
        for k, (r, W) in enumerate(zip(rows, weights)):
            torch.mm(x2d[:, r].to_sparse_csr(), W, out=A[k])
        return A

    @staticmethod
    def backward(ctx, grad_A):
        x2d, *weights = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x2d)
            span = ctx.differentiable or slice(0, x2d.shape[1])
            for k, (r, W) in enumerate(zip(ctx.rows, weights)):
                keep = (r >= span.start) & (r < span.stop)  # only columns that need it
                grad_x[:, r[keep]] = grad_A[k] @ W[keep].T
        grad_W = [
            x2d[:, r].T @ grad_A[k] if ctx.needs_input_grad[3 + k] else None
            for k, r in enumerate(ctx.rows)
        ]
        return grad_x, None, None, *grad_W


def activations(drive: Drive, x: torch.Tensor) -> torch.Tensor:
    """``(K_ff, B, T, N)`` feedforward activations, by sparse products if input is sparse.

    Density is judged from the first step, which is cheap and, for spike trains, typical.
    """
    first = x[:, 0]
    if (
        not drive.feedforward
        or torch.count_nonzero(first) > SPARSE_INPUT * first.numel()
    ):
        return drive.feedforward_activations(x)
    B, T, n_in = x.shape
    rows = [r for r, _ in drive.feedforward]
    weights = [W.float() for _, W in drive.feedforward]
    A = _SparseActivations.apply(
        x.reshape(B * T, n_in), rows, drive.differentiable_inputs, *weights
    )
    return A.view(len(rows), B, T, -1)
