"""Causal double-exponential filtering along time, as used by the spike-train losses.

``double_exponential`` filters ``(batch, time, n)`` trains first with the rise and then
with the decay exponential, continuing from carried state:

    rise[t]   = sum_{k <= t} a_r^k x[t - k]      + carried_rise   a_r^t
    smooth[t] = sum_{k <= t} a_d^k rise[t - k]   + carried_smooth a_d^t

On the CPU this is one lower-triangular Toeplitz matrix product per stage. On CUDA it
is the equivalent recursion ``y[t] = a y[t - 1] + x[t]``, both stages fused in one
Triton kernel, with the reverse recursion as its backward pass: linear in the chunk
length rather than quadratic.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

BLOCK = 256


def double_exponential(
    x: torch.Tensor,
    kernel_rise: torch.Tensor,
    kernel_decay: torch.Tensor,
    carried_rise: torch.Tensor | None,
    carried_smooth: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter ``x`` ``(B, T, N)``; return ``(rise, smooth)``, each ``(B, T, N)``.

    ``kernel_*[k] = exp(-k dt / tau)`` for ``k`` up to at least ``T - 1``; the carried
    states are ``(B, 1, N)`` or None. No gradient flows into the carried states.
    """
    if triton is not None and x.is_cuda:
        rise_carry = _zeros_like_step(x) if carried_rise is None else carried_rise
        smooth_carry = _zeros_like_step(x) if carried_smooth is None else carried_smooth
        return _DoubleExponential.apply(
            x, _step_decay(kernel_rise), _step_decay(kernel_decay), rise_carry,
            smooth_carry,
        )  # fmt: skip
    rise = _toeplitz(x, kernel_rise, carried_rise)
    return rise, _toeplitz(rise, kernel_decay, carried_smooth)


def _step_decay(kernel: torch.Tensor) -> float:
    """``kernel[1]`` as a float, read from the device once per kernel tensor.

    Losses keep their kernels as buffers, so this syncs with the GPU once rather than
    every chunk.
    """
    decay = getattr(kernel, "_step_decay", None)
    if decay is None:
        decay = kernel._step_decay = float(kernel[1])
    return decay


def _zeros_like_step(x):
    return x.new_zeros((x.shape[0], 1, x.shape[2]))


def _toeplitz(x, kernel, carried):
    T = x.shape[1]
    lag = torch.arange(T, device=x.device)
    lag = lag[:, None] - lag[None, :]
    matrix = torch.where(lag >= 0, kernel[:T][lag.clamp(min=0)], 0.0)
    y = matrix @ x
    return y if carried is None else y + carried * kernel[:T, None]


class _DoubleExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_rise, a_decay, carried_rise, carried_smooth):
        x = x.float().contiguous()
        B, T, N = x.shape
        rise, smooth = torch.empty_like(x), torch.empty_like(x)
        _filter_forward[(triton.cdiv(B * N, BLOCK),)](
            x, carried_rise.contiguous(), carried_smooth.contiguous(), rise, smooth,
            a_rise, a_decay, B, T, N, BLOCK=BLOCK,
        )  # fmt: skip
        ctx.shape, ctx.decays = (B, T, N), (a_rise, a_decay)
        ctx.mark_non_differentiable(rise)
        return rise, smooth

    @staticmethod
    def backward(ctx, _grad_rise, grad_smooth):
        B, T, N = ctx.shape
        grad_x = torch.empty((B, T, N), device=grad_smooth.device)
        _filter_backward[(triton.cdiv(B * N, BLOCK),)](
            grad_smooth.contiguous(), grad_x, *ctx.decays, B, T, N, BLOCK=BLOCK
        )
        return grad_x, None, None, None, None


if triton is not None:

    @triton.jit
    def _filter_forward(
        x, carried_rise, carried_smooth, rise, smooth, a_rise, a_decay, B, T, N,
        BLOCK: tl.constexpr,
    ):  # fmt: skip
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (batch, neuron)
        live = idx < B * N
        b, n = idx // N, idx % N
        r = tl.load(carried_rise + idx, mask=live, other=0.0)
        s = tl.load(carried_smooth + idx, mask=live, other=0.0)
        for t in range(T):
            at = (b * T + t) * N + n
            # The carried state enters undecayed at t = 0, then decays like the rest.
            r = tl.where(t == 0, 1.0, a_rise) * r + tl.load(
                x + at, mask=live, other=0.0
            )
            s = tl.where(t == 0, 1.0, a_decay) * s + r
            tl.store(rise + at, r, mask=live)
            tl.store(smooth + at, s, mask=live)

    @triton.jit
    def _filter_backward(
        grad_smooth, grad_x, a_rise, a_decay, B, T, N, BLOCK: tl.constexpr
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < B * N
        b, n = idx // N, idx % N
        gs = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/d rise, through smooth
        gr = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/d x, through rise
        for i in range(T):
            at = (b * T + T - 1 - i) * N + n
            gs = tl.load(grad_smooth + at, mask=live, other=0.0) + a_decay * gs
            gr = gs + a_rise * gr
            tl.store(grad_x + at, gr, mask=live)
