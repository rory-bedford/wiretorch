"""What every family's engine shares: when to use the kernels, and the synaptic input.

Each family's ``engine.simulate`` runs its Triton kernels on CUDA and its PyTorch
reference (``dynamics.simulate``) otherwise. Set ``WIRETORCH_BACKEND=torch`` to
force the reference everywhere.

Feedforward input is known for a whole chunk, so its synaptic drive — every pathway's
input times its weights — is computed up front by ``activations``: dense matmuls, or one
cuSPARSE product per pathway when the input is sparse (spike trains).

A pathway with presynaptic dynamics (see ``network.gate_step``) has as input its rows
times their factors: ``gate_feedforward`` steps the feedforward ones through a chunk on
the GPU, and ``Gating`` holds the recurrent ones' tables for the kernels. Pathways of
one source cell type share its weight rows, so the recurrent weights are laid out once
per source (``recurrent_weights``).

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
from dataclasses import dataclass

import torch

from wiretorch.network_simulators.network import Drive, gate_feedforward as _gate_ref

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
    spikes). A saturating pathway's rows are first scaled by its ``factors``
    ``(B * T, n_rows)`` (None for the others), which carry no gradient.
    """

    @staticmethod
    def forward(ctx, x2d, rows, factors, differentiable, *weights):
        ctx.rows, ctx.factors, ctx.differentiable = rows, factors, differentiable
        ctx.save_for_backward(x2d, *weights)
        x2d = x2d.detach()
        A = x2d.new_empty((len(rows), x2d.shape[0], weights[0].shape[1]))
        for k, (r, W) in enumerate(zip(rows, weights)):
            torch.mm(_rows(x2d, r, factors[k]).to_sparse_csr(), W, out=A[k])
        return A

    @staticmethod
    def backward(ctx, grad_A):
        x2d, *weights = ctx.saved_tensors
        factors = ctx.factors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x2d)
            span = ctx.differentiable or slice(0, x2d.shape[1])
            for k, (r, W) in enumerate(zip(ctx.rows, weights)):
                keep = (r >= span.start) & (r < span.stop)  # only columns that need it
                g = grad_A[k] @ W[keep].T
                if factors[k] is not None:
                    g = g * factors[k][:, keep]
                grad_x[:, r[keep]] += g
        grad_W = [
            _rows(x2d, r, factors[k]).T @ grad_A[k]
            if ctx.needs_input_grad[4 + k]
            else None
            for k, r in enumerate(ctx.rows)
        ]
        return grad_x, None, None, None, *grad_W


def _rows(x2d, rows, factors):
    return x2d[:, rows] if factors is None else x2d[:, rows] * factors


def activations(drive: Drive, x: torch.Tensor, factors=None) -> torch.Tensor:
    """``(K_ff, B, T, N)`` feedforward activations, by sparse products if input is sparse;
    ``factors`` those of the saturating pathways (see ``gate_feedforward``).

    Density is judged from the first step, which is cheap and, for spike trains, typical.
    """
    first = x[:, 0]
    if (
        not drive.feedforward
        or torch.count_nonzero(first) > SPARSE_INPUT * first.numel()
    ):
        return drive.feedforward_activations(x, factors)
    B, T, n_in = x.shape
    rows = [r for r, _ in drive.feedforward]
    weights = [W.float() for _, W in drive.feedforward]
    factors = [
        None if f is None else f.reshape(B * T, -1)
        for f in factors or [None] * len(rows)
    ]
    A = _SparseActivations.apply(
        x.reshape(B * T, n_in), rows, factors, drive.differentiable_inputs, *weights
    )
    return A.view(len(rows), B, T, -1)


GATE_BLOCK = 256  # rows per program of ``gate_scan``


def gate_feedforward(drive: Drive, x: torch.Tensor, gating: torch.Tensor | None):
    """``network.gate_feedforward`` by the ``gate_scan`` kernel (CUDA), one launch per
    pathway with presynaptic dynamics; ``gating`` is updated in place."""
    K_rec = len(drive.recurrent)
    gates = [drive.gate(K_rec + k) for k in range(len(drive.feedforward))]
    if not any(gates):
        return None, gating
    if not x.is_cuda:
        return _gate_ref(drive, x, gating)
    from wiretorch.network_simulators.kernels import gate_scan

    x = x.detach().float().contiguous()
    B, T, n_in = x.shape
    factors = []
    for (rows, _), gate in zip(drive.feedforward, gates):
        if gate is None:
            factors.append(None)
            continue
        f = torch.empty((B, T, gate.n), device=x.device)
        stp_col, sat_col = _gate_columns(gate)
        gate_scan[(triton.cdiv(B * gate.n, GATE_BLOCK),)](
            x, rows.int().contiguous(), gate.params.float().contiguous(), gating, f,
            B, T, n_in, gating.shape[1], gate.n, stp_col, sat_col,
            BLOCK=GATE_BLOCK, STP=gate.stp, SATURATES=gate.saturates,
        )  # fmt: skip
        factors.append(f)
    return factors, gating


def _gate_columns(gate) -> tuple[int, int]:
    """A ``Gate``'s first columns of the gating state: STP's ``u`` (then ``x``) and
    the receptors' gating, -1 where it has none."""
    stp = gate.start if gate.stp else -1
    saturation = gate.start + 2 * gate.n * gate.stp if gate.saturates else -1
    return stp, saturation


@dataclass
class Gating:
    """The recurrent pathways' presynaptic dynamics, as the kernels take them.

    ``state`` ``(B, R)`` is the gating state (updated in place), ``params`` ``(K, 5)``
    each pathway's ``Gate.params`` and ``cols`` ``(K, 3)`` int32 its columns of
    ``state`` and of the factors saved for the backward pass ``(B, T, R_f)`` (see
    ``compact_spikes``), -1 where it has none.
    """

    state: torch.Tensor
    params: torch.Tensor
    cols: torch.Tensor
    R_f: int
    f_cols: tuple[int, ...]

    @property
    def gated(self) -> bool:
        """Whether a recurrent pathway has presynaptic dynamics (``GATED``)."""
        return self.R_f > 0

    def f_saved(self, B, T, dummy) -> torch.Tensor:
        if not self.gated:
            return dummy
        return torch.empty((B, T, self.R_f), device=dummy.device)

    def factors(self, f_saved, sizes) -> list | None:
        """Per recurrent pathway, its factors ``(B, T, n_rows)`` (None if it has no
        presynaptic dynamics), or None if none has."""
        if not self.gated:
            return None
        return [
            None if c < 0 else f_saved[:, :, c : c + n]
            for c, n in zip(self.f_cols, sizes)
        ]


def gating_tables(drive: Drive, gating: torch.Tensor | None, dummy) -> Gating:
    """The ``Gating`` of a call's recurrent pathways."""
    params, cols, f_cols, R_f = [], [], [], 0
    zero = torch.zeros(5, device=dummy.device)
    for k in range(len(drive.recurrent)):
        gate = drive.gate(k)
        if gate is None:
            params.append(zero)
            cols.append((-1, -1, -1))
            f_cols.append(-1)
            continue
        params.append(gate.params.float())
        cols.append((*_gate_columns(gate), R_f))
        f_cols.append(R_f)
        R_f += gate.n
    if not R_f:
        return Gating(dummy, dummy, dummy, 0, tuple(f_cols))
    return Gating(
        gating,
        torch.stack(params).contiguous(),
        _int_table(tuple(cols), dummy.device),
        R_f,
        tuple(f_cols),
    )


@functools.lru_cache(maxsize=64)
def _int_table(rows: tuple, device) -> torch.Tensor:
    """``rows`` as a cached int32 tensor (see ``row_start``)."""
    return torch.tensor(rows, dtype=torch.int32, device=device)


def carry_to_sources(R, blocks, grad_A_t, factors, t, shared) -> None:
    """``R`` ``(B, N)``: what each recurrent pathway's ``dL/da`` of step ``t`` passes
    back to its source spikes (``presynaptic_blocks``), a presynaptic pathway's times
    its factors. ``shared``: whether pathways share sources (then they add up)."""
    if shared:
        R.zero_()
    for k, (source, carry) in enumerate(blocks):
        r = carry(grad_A_t[k])
        if factors is not None and factors[k] is not None:
            r = r * factors[k][:, t]
        if shared:
            R.index_add_(1, source, r)
        else:
            R.index_copy_(1, source, r)


def recurrent_weights(recurrent) -> tuple[torch.Tensor, tuple[int, ...]]:
    """The recurrent pathways' weights, each source's once (pathways that share a
    tensor share its rows), back to back, and the row where each pathway's begin."""
    unique, starts, offsets = [], {}, []
    for _, W in recurrent:
        if id(W) not in starts:
            starts[id(W)] = sum(w.shape[0] for w in unique)
            unique.append(W)
        offsets.append(starts[id(W)])
    return torch.cat(unique).float().contiguous(), tuple(offsets)


@functools.lru_cache(maxsize=64)
def offsets_tensor(offsets: tuple[int, ...], device) -> torch.Tensor:
    """``offsets`` as a cached int32 tensor (see ``row_start``)."""
    return torch.tensor(offsets, device=device, dtype=torch.int32)


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


def presynaptic_blocks(W, rows, sizes, dtype, w_offsets=None) -> list:
    """Per recurrent pathway: its source neurons and its ``presynaptic`` product.

    ``rows`` holds the pathways' source neurons back to back, ``sizes`` how many each
    has; ``W`` their weight rows, from ``w_offsets`` (default: back to back too).
    Pathways that share weight rows share the product.
    """
    starts = [0, *itertools.accumulate(sizes)]
    w_offsets = w_offsets or starts[:-1]
    products = {}
    blocks = []
    for a, b, w in zip(starts, starts[1:], w_offsets):
        if w not in products:
            products[w] = presynaptic(W[w : w + b - a], dtype)
        blocks.append((rows[a:b].long(), products[w]))
    return blocks


def weight_gradient(
    spikes, rows, sizes, grad_A, w_offsets=None, n_weights=None, factors=None
) -> torch.Tensor:
    """``dL/dW`` of the recurrent pathways' weight rows (laid out as ``w_offsets``
    say, ``n_weights`` rows in all; default: each pathway's back to back).

    ``dL/dW_k`` is the sum over (trial, step) of presynaptic spikes x ``dL/dA_k``, a
    saturating pathway's spikes times their ``factors[k]`` ``(B, T, n_rows)``;
    ``grad_A`` ``(K, B, T, N)`` has the recurrent pathways first.
    """
    N = grad_A.shape[-1]
    starts = [0, *itertools.accumulate(sizes)]
    pieces = []
    for k, (a, b) in enumerate(zip(starts, starts[1:])):
        pre = spikes[:, :, rows[a:b]]
        if factors is not None and factors[k] is not None:
            pre = pre * factors[k]
        pieces.append(pre.reshape(-1, pre.shape[-1]).T @ grad_A[k].reshape(-1, N))
    if w_offsets is None:
        return torch.cat(pieces)
    grad = grad_A.new_zeros((n_weights, N))
    for w, piece in zip(w_offsets, pieces):
        grad[w : w + piece.shape[0]] += piece
    return grad


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
