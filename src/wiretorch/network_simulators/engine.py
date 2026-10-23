"""What every family's engine shares: when to use the kernels, and the synaptic input.

Each family's ``engine.simulate`` runs its Triton kernels on CUDA and its PyTorch
reference (``dynamics.simulate``) otherwise. Set ``WIRETORCH_BACKEND=torch`` to
force the reference everywhere.

Feedforward input is known for a whole chunk, so its synaptic drive — every pathway's
input times its weights — is computed up front by ``activations``: dense matmuls, or one
cuSPARSE product per pathway when the input is sparse (spike trains).

A recurrent network's backward pass needs the gradient w.r.t. its recurrent weights
(``weight_gradient``) and, when gradients flow through the recurrent spikes
(``Drive.recurrent_gradients``), each step's ``dL/da`` carried back to the presynaptic
spikes (``presynaptic``).
"""

from __future__ import annotations

import functools
import itertools
import os
import warnings

import torch

from wiretorch.network_simulators.network import Drive

warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")

try:
    import triton
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

#: Use the sparse path when at most this fraction of inputs is non-zero.
SPARSE_INPUT = 0.1
#: Carry gradients back through recurrent weights at most this dense by sparse products.
SPARSE_WEIGHTS = 0.25


def _backend() -> str:
    """``WIRETORCH_BACKEND``, default triton."""
    return os.environ.get("WIRETORCH_BACKEND", "triton")


def uses_kernels(x: torch.Tensor, retain) -> bool:
    """Whether to run the Triton kernels: on CUDA, unless per-step gradients are kept."""
    return triton is not None and x.is_cuda and retain is None and _backend() != "torch"


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


def presynaptic(W_k: torch.Tensor, dtype: torch.dtype = torch.float32):
    """``g -> g @ W_k.T``: a pathway's ``dL/da`` ``(B, N)`` carried to its sources.

    Connectomes are sparse, so in float32 a sparse ``W_k`` (built once per backward
    pass) makes this one cuSPARSE product per step instead of a dense
    ``(B, N) x (N, n_k)`` GEMM. With float16 storage it is a dense GEMM on the tensor
    cores instead, 2.8x faster than cuSPARSE on the teacher's connectome (relative
    error ~4e-4, accumulating in float32).
    """
    if dtype == torch.float16:
        W_t = W_k.T.contiguous().half()
        return lambda g: (g.half() @ W_t).float()
    if torch.count_nonzero(W_k) > SPARSE_WEIGHTS * W_k.numel():
        return lambda g: g @ W_k.T
    W_csr = W_k.to_sparse_csr()
    return lambda g: (W_csr @ g.T.contiguous()).T


def presynaptic_blocks(W, rows, sizes, dtype) -> list:
    """Per recurrent pathway: its source neurons and its ``presynaptic`` product.

    ``W`` and ``rows`` hold the pathways' weight rows and source neurons back to back,
    ``sizes`` how many each has.
    """
    starts = [0, *itertools.accumulate(sizes)]
    return [
        (rows[a:b].long(), presynaptic(W[a:b], dtype))
        for a, b in zip(starts, starts[1:])
    ]


def weight_gradient(spikes, rows, sizes, grad_A) -> torch.Tensor:
    """``dL/dW`` of the recurrent pathways, their weight rows back to back.

    ``dL/dW_k`` is the sum over (trial, step) of presynaptic spikes x ``dL/dA_k``;
    ``grad_A`` ``(K, B, T, N)`` has the recurrent pathways first.
    """
    N = grad_A.shape[-1]
    starts = [0, *itertools.accumulate(sizes)]
    pieces = []
    for k, (a, b) in enumerate(zip(starts, starts[1:])):
        pre = spikes[:, :, rows[a:b]]
        pieces.append(pre.reshape(-1, pre.shape[-1]).T @ grad_A[k].reshape(-1, N))
    return torch.cat(pieces)


def refractory_tables(n_ref, counter, dummy) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernels' refractory arguments: the periods in steps ``(N,)`` int32, and a
    copy of the counters ``(B, N)`` int32 that the kernels update in place — dummies
    if no neuron has a refractory period (``n_ref`` None)."""
    if n_ref is None:
        return dummy, dummy
    return (
        n_ref.to(torch.int32).contiguous(),
        counter.to(torch.int32, copy=True).contiguous(),
    )


def held_buffer(B, T, N, refractory, dummy) -> torch.Tensor:
    """Where the forward pass tells the backward pass which steps were refractory:
    ``(B, T, N)`` int8, or a dummy."""
    if not refractory:
        return dummy
    return torch.empty((B, T, N), dtype=torch.int8, device=dummy.device)
