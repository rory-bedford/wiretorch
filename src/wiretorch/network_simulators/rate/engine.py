"""Runs the rate dynamics: the Triton kernels on CUDA, the PyTorch reference otherwise.

``simulate`` is the single entry point the networks call. On a CUDA device (and in
float32) it drives the kernels of ``kernels.py``:

* A network with no recurrent pathways is integrated a chunk at a time by
  ``scan_forward``, after one matmul per pathway computes the whole chunk's synaptic
  input; its backward pass is one reverse-time scan, ``scan_backward``.
* A recurrent network takes, per step, ``rates_forward``, one product per recurrent
  pathway, and ``recurrent_step``. Rates are dense, so the product is a matmul — a
  cuSPARSE one when the pathway's weights are at most ``SPARSE_WEIGHTS`` dense, which
  connectomes are. Backwards: ``inject_backward``, the product with ``W^T``,
  ``output_backward``.
* Learnt physiology (``theta``, ``gain``, ``beta``, synaptic decays) gets its gradient
  from the backward kernels, which accumulate it only when it is learnt.

Rates are dense, so the products dominate: ~85% of a recurrent step, and most of a
feedforward chunk (the input times its weights), at about half the GPU's float32 peak.
With ``storage_dtype=float16`` both run on float16 tensor cores (accumulating in
float32): 2.7x faster products on the teacher, with relative errors ~1e-3; the state
and all the elementwise arithmetic stay float32.

Set ``WIRETORCH_BACKEND=torch`` to force the reference implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from wiretorch.network_simulators.engine import (
    activations,
    pad,
    triton,
    uses_kernels,
)
from wiretorch.network_simulators.network import Drive, Run
from wiretorch.network_simulators.rate import dynamics
from wiretorch.network_simulators.rate.dynamics import Physiology

if triton is not None:
    from wiretorch.network_simulators.rate import kernels

BLOCK = 128  # (trial, neuron) pairs per program
FF_TIME_BLOCK = 256  # steps of feedforward input materialised at once (no gradients)
#: Recurrent products by cuSPARSE for weights at most this dense.
SPARSE_WEIGHTS = 0.25
#: Each nonlinearity's code in the kernels.
CODES = {"relu": 0, "softplus": 1, "sigmoid": 2, "tanh": 3}


def simulate(
    v, I, x, drive: Drive, p: Physiology, *, probabilities=False, track=False,
    track_batch_idx=None, retain=None,
) -> Run:  # fmt: skip
    """Simulate ``x.shape[1]`` steps; see ``dynamics.simulate`` for the arguments."""
    if not uses_kernels(x, retain) or v.dtype != torch.float32:
        return dynamics.simulate(
            v, I, x, drive, p,
            probabilities=probabilities, track=track,
            track_batch_idx=track_batch_idx, retain=retain,
        )  # fmt: skip
    assert not probabilities, "rate networks have no spike probabilities"
    return _simulate_kernels(v, I, x, drive, p, track, track_batch_idx)


# =====================================================================
# Launches
# =====================================================================


@dataclass
class _Launch:
    """Shapes, tables and flags shared by the launches of one call."""

    B: int
    T: int
    N: int
    S: int
    SP: int
    K_rec: int
    K_ff: int
    made: torch.Tensor  # (K, SP) float
    sign: torch.Tensor  # (SP,)
    code: torch.Tensor  # (N,) int32
    track_b: int  # tracked trial, -1 for all, or 0 when not tracking
    tracked: dict[str, torch.Tensor] | None
    dummy: torch.Tensor  # stands in for unused pointer arguments
    product_dtype: torch.dtype = torch.float32
    want_theta: bool = False
    want_gain: bool = False
    want_beta: bool = False
    want_decay: bool = False

    @property
    def K(self) -> int:
        return self.K_rec + self.K_ff

    def saved(self, device):
        """Buffers for what the backward pass needs: ``(v, y, I)`` per step."""
        B, T, N, S = self.B, self.T, self.N, self.S
        return (
            torch.empty((B, T, N), device=device),
            torch.empty((B, T, N), device=device) if self.want_beta else self.dummy,
            torch.empty((B, T, S, N), device=device) if self.want_decay else self.dummy,
        )


def _prepare(v, x, drive, p, track, track_batch_idx) -> _Launch:
    B, T = x.shape[:2]
    N = v.shape[1]
    S = p.sign.shape[0]
    SP = max(2, triton.next_power_of_2(S))
    code = torch.empty(N, dtype=torch.int32, device=x.device)
    for name, neurons in p.groups:
        code[neurons] = CODES[name]
    tracked = None
    if track:
        Bt = B if track_batch_idx is None else 1
        tracked = {
            "v": torch.empty((Bt, T, N), device=x.device),
            "I": torch.empty((Bt, T, N, S), device=x.device),
        }
    return _Launch(
        B=B, T=T, N=N, S=S, SP=SP,
        K_rec=len(drive.recurrent), K_ff=len(drive.feedforward),
        made=pad(drive.made.float(), SP), sign=pad(p.sign, SP), code=code,
        track_b=(-1 if track_batch_idx is None else track_batch_idx) if track else 0,
        tracked=tracked, dummy=torch.empty(1, device=x.device),
        want_theta=p.theta.requires_grad, want_gain=p.gain.requires_grad,
        want_beta=p.beta.requires_grad, want_decay=p.decay.requires_grad,
        product_dtype=p.storage_dtype,
    )  # fmt: skip


def _track_buffers(L: _Launch):
    if L.tracked is None:
        return L.dummy, L.dummy
    return L.tracked["v"], L.tracked["I"]


def _scan(L, A, scale, theta, gain, beta, decay, v, I, rates, saved, t0=0):
    """``scan_forward`` over the ``A.shape[2]`` steps from ``t0``; updates v, I in place."""
    v_saved, y_saved, I_saved = saved or (L.dummy, L.dummy, L.dummy)
    kernels.scan_forward[(triton.cdiv(L.B * L.N, BLOCK),)](
        A, scale, L.made, L.sign, decay, theta, gain, beta, L.code,
        v, I, rates, v_saved, y_saved, I_saved, *_track_buffers(L),
        L.track_b, t0, A.shape[2], L.B, L.T, L.N,
        K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK, SAVE=saved is not None,
        SAVE_Y=bool(saved) and L.want_beta, SAVE_I=bool(saved) and L.want_decay,
        TRACK=L.tracked is not None,
    )  # fmt: skip


class _Product:
    """The recurrent pathways' activations ``a_k = r[:, rows_k] @ W_k``, and back.

    Each pathway's product is a dense matmul, or — if its weights are at most
    ``SPARSE_WEIGHTS`` dense — a cuSPARSE one, with ``W_k`` and ``W_k^T`` converted
    once per forward call. In float16 (``dtype``) every product is a dense one on the
    tensor cores, which beats a float32 sparse one.
    """

    def __init__(
        self,
        W: torch.Tensor,
        rows: torch.Tensor,
        sizes: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ):
        starts = [0, *itertools.accumulate(sizes)]
        self.dtype = dtype
        self.blocks = []
        for k in range(len(sizes)):
            W_k = W[starts[k] : starts[k + 1]].to(dtype)
            rows_k = rows[starts[k] : starts[k + 1]].long()
            sparse = dtype == torch.float32 and (
                torch.count_nonzero(W_k) <= SPARSE_WEIGHTS * W_k.numel()
            )
            self.blocks.append(
                (
                    rows_k,
                    W_k,
                    W_k.T.contiguous().to_sparse_csr() if sparse else None,
                    W_k.to_sparse_csr() if sparse else None,
                )
            )

    def forward(self, r: torch.Tensor, out: torch.Tensor) -> None:
        """``out[k] = r[:, rows_k] @ W_k`` for ``r`` ``(B, N)``; ``out`` ``(K, B, N)``."""
        for k, (rows_k, W_k, W_T, _) in enumerate(self.blocks):
            r_k = r.index_select(1, rows_k)
            if self.dtype != torch.float32:
                out[k].copy_(r_k.to(self.dtype) @ W_k)
            elif W_T is None:
                torch.mm(r_k, W_k, out=out[k])
            else:
                out[k].copy_((W_T @ r_k.T.contiguous()).T)

    def backward(self, g: torch.Tensor, R: torch.Tensor) -> None:
        """``R[:, rows_k] = g[k] @ W_k^T``: each pathway's dL/da to its sources."""
        for k, (rows_k, W_k, _, W_csr) in enumerate(self.blocks):
            if self.dtype != torch.float32:
                R.index_copy_(1, rows_k, (g[k].to(self.dtype) @ W_k.T).float())
            elif W_csr is None:
                R.index_copy_(1, rows_k, g[k] @ W_k.T)
            else:
                R.index_copy_(1, rows_k, (W_csr @ g[k].T.contiguous()).T)


def _recurrent(L, A_ff_blocks, product, scale, beta, decay, theta, gain, v, I, rates,
               saved, A_rec):  # fmt: skip
    """Per step: rates, the recurrent products, the update; updates v, I in place.

    ``A_ff_blocks`` yields ``(t0, A_ff)`` blocks of feedforward activations in order.
    ``A_rec`` ``(T or 1, K_rec, B, N)`` receives each step's recurrent activations (all
    of them, if ``saved``, for the backward pass).
    """
    v_saved, y_saved, I_saved = saved or (L.dummy, L.dummy, L.dummy)
    grid = (triton.cdiv(L.B * L.N, BLOCK),)
    for t0, A_ff in A_ff_blocks:
        T_ff = A_ff.shape[2]
        for i in range(T_ff):
            t = t0 + i
            slab = A_rec[t if saved else 0]
            kernels.rates_forward[grid](
                v, theta, gain, L.code, rates, t, L.B, L.T, L.N, BLOCK=BLOCK
            )
            product.forward(rates[:, t], slab)
            kernels.recurrent_step[grid](
                slab, A_ff, scale, L.made, L.sign, decay, beta,
                v, I, v_saved, y_saved, I_saved, *_track_buffers(L),
                L.track_b, t, i, L.T, T_ff, L.B, L.N,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP, BLOCK=BLOCK,
                SAVE=saved is not None, SAVE_Y=bool(saved) and L.want_beta,
                SAVE_I=bool(saved) and L.want_decay, TRACK=L.tracked is not None,
            )  # fmt: skip


def _physiology_grads(L, theta, gain, beta, decay, needs):
    """Zeroed per-(trial, neuron) accumulators for the physiology gradients wanted."""
    B, N, S, dev = L.B, L.N, L.S, theta.device
    return (
        torch.zeros((B, N), device=dev) if needs[0] else L.dummy,
        torch.zeros((B, N), device=dev) if needs[1] else L.dummy,
        torch.zeros((B, N), device=dev) if needs[2] else L.dummy,
        torch.zeros((B, S, N), device=dev) if needs[3] else L.dummy,
    )


def _reduce(partials, needs):
    """Per-neuron (and, for decays, per-synapse) sums of the accumulators."""
    theta, gain, beta, decay = partials
    return (
        theta.sum(0) if needs[0] else None,
        gain.sum(0) if needs[1] else None,
        beta.sum(0) if needs[2] else None,
        decay.sum((0, 2)) if needs[3] else None,
    )


# =====================================================================
# Autograd
# =====================================================================


class _FeedforwardScan(torch.autograd.Function):
    """Rates of a chunk, differentiable in ``A``, ``scale`` and the physiology."""

    @staticmethod
    def forward(ctx, A, scale, theta, gain, beta, decay, v, I, L: _Launch):
        rates = torch.empty((L.B, L.T, L.N), device=A.device)
        saved = L.saved(A.device)
        decay_p = pad(decay, L.SP)
        _scan(L, A, scale, theta, gain, beta, decay_p, v, I, rates, saved)
        ctx.L = L
        ctx.save_for_backward(A, scale, theta, gain, beta, decay_p, *saved)
        ctx.mark_non_differentiable(v, I)
        return rates, v, I

    @staticmethod
    def backward(ctx, grad_rates, _v, _I):
        A, scale, theta, gain, beta, decay_p, v_saved, y_saved, I_saved = (
            ctx.saved_tensors
        )
        L = ctx.L
        want_A, want_scale = ctx.needs_input_grad[:2]
        needs = ctx.needs_input_grad[2:6]
        grad_A = torch.empty_like(A) if want_A else L.dummy
        partial = (
            torch.empty((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
        )
        physiology = _physiology_grads(L, theta, gain, beta, decay_p, needs)
        kernels.scan_backward[(triton.cdiv(L.B * L.N, BLOCK),)](
            grad_rates.contiguous(), v_saved, y_saved, I_saved, A, scale, L.made,
            L.sign, decay_p, theta, gain, beta, L.code,
            grad_A, partial, *physiology, L.B, L.T, L.N,
            K=L.K, KP=triton.next_power_of_2(max(L.K, 1)), S=L.S, SP=L.SP,
            BLOCK=BLOCK, WANT_A=want_A, WANT_SCALE=want_scale,
            WANT_THETA=needs[0], WANT_GAIN=needs[1], WANT_BETA=needs[2],
            WANT_DECAY=needs[3],
        )  # fmt: skip
        return (
            grad_A if want_A else None,
            partial.sum(dim=1) if want_scale else None,
            *_reduce(physiology, needs),
            None, None, None,
        )  # fmt: skip


class _RecurrentRun(torch.autograd.Function):
    """Rates of a recurrent chunk, differentiable in ``A_ff``, ``scale``, ``W`` and
    the physiology."""

    @staticmethod
    def forward(ctx, A_ff, scale, W, theta, gain, beta, decay, v, I, L, rows, sizes):
        rates = torch.empty((L.B, L.T, L.N), device=A_ff.device)
        saved = L.saved(A_ff.device)
        A_rec = torch.empty((L.T, L.K_rec, L.B, L.N), device=A_ff.device)
        decay_p = pad(decay, L.SP)
        product = _Product(W, rows, sizes, L.product_dtype)
        _recurrent(
            L, [(0, A_ff)], product, scale, beta, decay_p, theta, gain, v, I, rates,
            saved, A_rec,
        )  # fmt: skip
        ctx.L, ctx.rows, ctx.sizes, ctx.product = L, rows, sizes, product
        ctx.save_for_backward(
            A_ff, scale, theta, gain, beta, decay_p, rates, A_rec, *saved
        )
        ctx.mark_non_differentiable(v, I)
        return rates, v, I

    @staticmethod
    def backward(ctx, grad_rates, _v, _I):
        (A_ff, scale, theta, gain, beta, decay_p, rates, A_rec,
         v_saved, y_saved, I_saved) = ctx.saved_tensors  # fmt: skip
        L, rows, sizes = ctx.L, ctx.rows, ctx.sizes
        want_ff, want_scale, want_W = ctx.needs_input_grad[:3]
        needs = ctx.needs_input_grad[3:7]
        dev = A_ff.device
        grad_rates = grad_rates.contiguous()
        lv = torch.zeros((L.B, L.N), device=dev)
        lI = torch.zeros((L.B, L.S, L.N), device=dev)
        R = torch.zeros((L.B, L.N), device=dev)  # dL/dr through the recurrence
        grad_rec = torch.empty_like(A_rec)
        grad_ff = torch.empty_like(A_ff) if want_ff else L.dummy
        partial = torch.zeros((L.K, L.B, L.N), device=dev) if want_scale else L.dummy
        physiology = _physiology_grads(L, theta, gain, beta, decay_p, needs)
        grid = (triton.cdiv(L.B * L.N, BLOCK),)
        for t in reversed(range(L.T)):
            kernels.inject_backward[grid](
                lv, lI, v_saved, y_saved, I_saved, A_rec[t], A_ff, scale, L.made,
                L.sign, decay_p, beta, grad_rec[t], grad_ff, partial,
                physiology[2], physiology[3], t, L.B, L.T, L.N,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP, BLOCK=BLOCK,
                WANT_FF=want_ff, WANT_SCALE=want_scale, WANT_BETA=needs[2],
                WANT_DECAY=needs[3],
            )  # fmt: skip
            ctx.product.backward(grad_rec[t], R)
            kernels.output_backward[grid](
                lv, v_saved, grad_rates, R, theta, gain, L.code,
                physiology[0], physiology[1], t, L.B, L.T, L.N,
                BLOCK=BLOCK, WANT_THETA=needs[0], WANT_GAIN=needs[1],
            )  # fmt: skip
        grad_W = None
        if want_W:
            # dL/dW_k = sum over (trial, step) of presynaptic rates x dL/da_k.
            pieces, starts = [], [0, *itertools.accumulate(sizes)]
            for k in range(L.K_rec):
                pre = rates[:, :, rows[starts[k] : starts[k + 1]]].transpose(0, 1)
                pieces.append(
                    pre.reshape(-1, pre.shape[-1]).T @ grad_rec[:, k].reshape(-1, L.N)
                )
            grad_W = torch.cat(pieces)
        return (
            grad_ff if want_ff else None,
            partial.sum(dim=1) if want_scale else None,
            grad_W,
            *_reduce(physiology, needs),
            None, None, None, None, None,
        )  # fmt: skip


# =====================================================================
# Entry point for the kernels
# =====================================================================


def _activations(L: _Launch, drive: Drive, x: torch.Tensor) -> torch.Tensor:
    """Feedforward activations ``(K_ff, B, T, N)``, float32; the products in
    ``L.product_dtype``."""
    if L.product_dtype == torch.float32:
        return activations(drive, x).contiguous()
    with torch.autocast("cuda", dtype=L.product_dtype):
        A = drive.feedforward_activations(x)
    return A.float().contiguous()


def _simulate_kernels(v, I, x, drive, p, track, track_batch_idx) -> Run:
    L = _prepare(v, x, drive, p, track, track_batch_idx)
    v = v.detach().contiguous().clone()
    I = torch.empty((L.B, L.S, L.N), device=x.device).copy_(I.detach().permute(0, 2, 1))
    scale = drive.scale.float().contiguous()
    theta, gain, beta = (t.contiguous() for t in (p.theta, p.gain, p.beta))
    weights = [W for _, W in drive.recurrent] + [W for _, W in drive.feedforward]
    learnt = (scale, theta, gain, beta, p.decay, x, *weights)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in learnt)

    if L.K_rec == 0:
        if grad:
            A = _activations(L, drive, x)
            rates, v, I = _FeedforwardScan.apply(
                A, scale, theta, gain, beta, p.decay, v, I, L
            )
        else:
            rates = torch.empty((L.B, L.T, L.N), device=x.device)
            decay = pad(p.decay, L.SP)
            with torch.no_grad():
                for t0 in range(0, L.T, FF_TIME_BLOCK):
                    A = _activations(L, drive, x[:, t0 : t0 + FF_TIME_BLOCK])
                    _scan(L, A, scale, theta, gain, beta, decay, v, I, rates, None, t0)
        return Run(rates, v, I.permute(0, 2, 1), None, L.tracked)

    rows = torch.cat([r for r, _ in drive.recurrent]).contiguous()
    sizes = tuple(r.numel() for r, _ in drive.recurrent)
    W = torch.cat([W for _, W in drive.recurrent]).float().contiguous()

    if grad:
        A_ff = _activations(L, drive, x)
        rates, v, I = _RecurrentRun.apply(
            A_ff, scale, W, theta, gain, beta, p.decay, v, I, L, rows, sizes
        )
    else:
        rates = torch.empty((L.B, L.T, L.N), device=x.device)
        A_rec = torch.empty((1, L.K_rec, L.B, L.N), device=x.device)
        decay = pad(p.decay, L.SP)

        def blocks():
            for t0 in range(0, L.T, FF_TIME_BLOCK):
                yield t0, _activations(L, drive, x[:, t0 : t0 + FF_TIME_BLOCK])

        with torch.no_grad():
            _recurrent(
                L, blocks(), _Product(W, rows, sizes, L.product_dtype), scale, beta, decay, theta,
                gain, v, I, rates, None, A_rec,
            )  # fmt: skip
    return Run(rates, v, I.permute(0, 2, 1), None, L.tracked)
