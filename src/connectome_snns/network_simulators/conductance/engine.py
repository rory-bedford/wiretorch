"""Runs the dynamics: the Triton kernels on CUDA, the PyTorch reference otherwise.

``simulate`` is the single entry point the networks call. On a CUDA device it drives
the kernels of ``kernels.py``:

* A network with no recurrent pathways is integrated a chunk at a time by
  ``scan_forward``, after one matmul per pathway computes the whole chunk's synaptic
  input. Without gradients, long chunks are fed in blocks to bound memory.
* A recurrent network takes two launches per step: ``compact_spikes`` lists who fired,
  ``recurrent_step`` gathers just those weight rows and integrates.
* With gradients, the forward pass saves the voltage, conductances and recurrent
  activations of every step, and ``scan_backward`` is the reverse-time adjoint. It also
  gives learnt physiology its gradient, each term compiled in only when that parameter
  is learnt, from nothing more saved than the conductances entering the chunk.

Set ``CONNECTOME_SNNS_BACKEND=torch`` to force the reference implementation. What the
families share — the dispatch rule, sparse feedforward input — is in
``network_simulators.engine``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import torch

from connectome_snns.network_simulators.conductance import dynamics
from connectome_snns.network_simulators.conductance.dynamics import (
    Drive,
    Physiology,
    Run,
)
from connectome_snns.network_simulators.engine import (  # noqa: F401 - re-exported
    SPARSE_INPUT,
    _SparseActivations,
    activations as _activations,
    pad as _pad,
    row_start as _row_start,
    triton,
    uses_kernels,
)

if triton is not None:
    from connectome_snns.network_simulators.conductance import kernels

BLOCK = 128  # neurons per program (scan kernels)
STEP_BLOCK = 256  # neurons per program (recurrent step)
GATHER_ROWS = 16  # weight rows of fired neurons summed per load
COMPACT_BLOCK = 1024  # neurons per iteration of the spike compaction
FF_TIME_BLOCK = 256  # steps of feedforward input materialised at once (no gradients)


def simulate(
    v, g, x, drive: Drive, p: Physiology, *, probabilities=False, track=False,
    track_batch_idx=None, retain=None,
) -> Run:  # fmt: skip
    """Simulate ``x.shape[1]`` steps; see ``dynamics.simulate`` for the arguments."""
    if not uses_kernels(x, retain):
        return dynamics.simulate(
            v, g, x, drive, p,
            probabilities=probabilities, track=track,
            track_batch_idx=track_batch_idx, retain=retain,
        )  # fmt: skip
    return _simulate_kernels(v, g, x, drive, p, probabilities, track, track_batch_idx)


# =====================================================================
# Kernel launches
# =====================================================================


@dataclass
class _Launch:
    """Shapes, tables and output buffers shared by the launches of one call."""

    B: int
    T: int
    N: int
    S: int
    SP: int
    K_rec: int
    K_ff: int
    made: torch.Tensor  # (K, SP) float
    jump: torch.Tensor  # (2, SP)
    decay: torch.Tensor  # (2, SP)
    E_syn: torch.Tensor  # (SP,)
    g_clip: torch.Tensor  # (SP,)
    p: Physiology
    sample: bool
    probabilities: bool
    noise: torch.Tensor  # (B, T, N), or a placeholder
    track_b: int  # tracked trial, -1 for all, or 0 when not tracking
    tracked: dict[str, torch.Tensor] | None
    dummy: torch.Tensor  # stands in for unused pointer arguments
    #: Which physiology gradients are wanted, in ``PHYSIOLOGY`` order.
    wants: tuple[bool, ...] = (False,) * 8

    @property
    def K(self) -> int:
        return self.K_rec + self.K_ff

    def flags(self) -> dict[str, bool]:
        return {f"WANT_{name.upper()}": want for name, want in zip(FLAGS, self.wants)}

    def backward_block(self) -> int:
        """Neurons per program of ``scan_backward``: half the usual when the learnt
        physiology needs six or more per-synapse accumulators (each ``(BLOCK, SP)``),
        which would otherwise spill registers."""
        E_syn, decay, jump = self.wants[5:]
        tiles = E_syn + 4 * decay + 2 * jump  # the decays carry H as well
        return BLOCK // 2 if tiles >= 6 else BLOCK

    def g_init(self, g: torch.Tensor) -> torch.Tensor:
        """A float32 copy of the conductances entering the chunk if the decays are
        learnt, else a dummy."""
        return g.to(torch.float32, copy=True) if self.wants[6] else self.dummy


#: The physiology that can be learnt, as ``Physiology`` names it, and the names of
#: ``scan_backward``'s flags for each.
PHYSIOLOGY = ("theta", "U_reset", "E_L", "C_m", "beta", "E_syn", "decay", "jump")
FLAGS = ("theta", "reset", "E_L", "C_m", "beta", "E_syn", "decay", "jump")


def _physiology_grads(L: _Launch, device) -> list[torch.Tensor]:
    """Zeroed per-(trial, [synapse,] neuron) accumulators of the gradients wanted."""
    shapes = [(L.B, L.N)] * 5 + [(L.B, L.S, L.N)] + [(L.B, 2, L.S, L.N)] * 2
    return [
        torch.empty(shape, device=device) if want else L.dummy
        for shape, want in zip(shapes, L.wants)
    ]


def _reduce(partials, wants) -> list[torch.Tensor | None]:
    """Per-neuron (per-synapse; per component and synapse) sums of the accumulators."""
    dims = [0] * 5 + [(0, 2)] + [(0, 3)] * 2
    return [g.sum(d) if want else None for g, d, want in zip(partials, dims, wants)]


def _prepare(v, x, drive, p, probabilities, track, track_batch_idx) -> _Launch:
    B, T = x.shape[:2]
    N = v.shape[1]
    S = p.E_syn.shape[0]
    SP = triton.next_power_of_2(S + 1)  # > S: tracking stores the leak in column S
    dummy = torch.empty(1, device=x.device)
    sample = p.spike_mode == "sample"
    tracked = None
    if track:
        Bt = B if track_batch_idx is None else 1
        tracked = {
            "v": torch.empty((Bt, T, N), device=x.device),
            "I": torch.empty((Bt, T, N, S + 1), device=x.device),
            "g": torch.empty((Bt, T, N, 2, S), device=x.device),
        }
    return _Launch(
        B=B, T=T, N=N, S=S, SP=SP,
        K_rec=len(drive.recurrent), K_ff=len(drive.feedforward),
        made=_pad(drive.made.float(), SP),
        jump=_pad(p.jump.detach(), SP), decay=_pad(p.decay.detach(), SP),
        E_syn=_pad(p.E_syn.detach(), SP), g_clip=_pad(p.g_clip, SP),
        # The kernels get the physiology detached (the autograd functions take it as
        # inputs): launching with tensors that require grad costs host time per launch.
        p=replace(p, **{name: getattr(p, name).detach() for name in PHYSIOLOGY}),
        sample=sample, probabilities=probabilities,
        noise=torch.rand((B, T, N), device=x.device) if sample else dummy,
        track_b=(-1 if track_batch_idx is None else track_batch_idx) if track else 0,
        tracked=tracked, dummy=dummy,
        wants=tuple(getattr(p, name).requires_grad for name in PHYSIOLOGY),
    )  # fmt: skip


def _neurons(p: Physiology):
    return p.theta, p.U_reset, p.E_L, p.C_m, p.beta


def _track_buffers(L: _Launch):
    if L.tracked is None:
        return L.dummy, L.dummy, L.dummy
    return L.tracked["v"], L.tracked["I"], L.tracked["g"]


def _scan(L, A, scale, v, g, spikes, prob, saved, t0=0):
    """``scan_forward`` over the ``A.shape[2]`` steps from ``t0``; updates v, g in place."""
    v_saved, G_saved = saved if saved else (L.dummy, L.dummy)
    TA = A.shape[2]
    kernels.scan_forward[(triton.cdiv(L.B * L.N, BLOCK),)](
        A, scale, L.made, L.jump, L.decay, L.E_syn, L.g_clip,
        *_neurons(L.p),
        v, g, spikes, L.dummy if prob is None else prob, L.noise,
        v_saved, G_saved, *_track_buffers(L),
        L.p.dt, L.p.surr_scale, L.track_b, t0, TA, L.B, L.T, L.N,
        K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK,
        SAVE=bool(saved), TRACK=L.tracked is not None,
        PROB=prob is not None, SAMPLE=L.sample,
    )  # fmt: skip


def _recurrent(L, A_ff_blocks, W, rows, sizes, scale, v, g, spikes, saved):
    """Per step: ``compact_spikes`` then ``recurrent_step``; updates v, g in place.

    ``A_ff_blocks`` yields ``(t0, A_ff)`` blocks of feedforward activations in order;
    ``rows`` holds the recurrent pathways' source neurons back to back, ``sizes`` how
    many each has.
    """
    row_start = _row_start(sizes, v.device)
    w_start = row_start[:-1]
    v_saved, G_saved, A_saved = saved if saved else (L.dummy, L.dummy, L.dummy)
    W = W.to(L.p.storage_dtype)  # a copy if reduced: gradients stay float32
    capacity = max(sizes) if L.K_rec else 1
    lists = torch.empty(
        (max(L.K_rec, 1), L.B, capacity), dtype=torch.int32, device=v.device
    )
    counts = torch.zeros((max(L.K_rec, 1), L.B), dtype=torch.int32, device=v.device)
    dt, surr = L.p.dt, L.p.surr_scale
    for t0, A_ff in A_ff_blocks:
        T_ff = A_ff.shape[2]
        for i in range(T_ff):
            t = t0 + i
            if L.K_rec:
                kernels.compact_spikes[(L.B, L.K_rec)](
                    v, L.p.theta, L.noise, rows, row_start, lists, counts,
                    surr, t, L.B, L.T, L.N, capacity,
                    BLOCK=COMPACT_BLOCK, SAMPLE=L.sample,
                )  # fmt: skip
            kernels.recurrent_step[(L.B, triton.cdiv(L.N, STEP_BLOCK))](
                A_ff, W, w_start, lists, counts, scale, L.made, L.jump, L.decay,
                L.E_syn, L.g_clip, *_neurons(L.p),
                v, g, spikes, L.noise,
                v_saved, G_saved, A_saved, *_track_buffers(L),
                dt, surr, L.track_b, t, i, L.T, T_ff, L.B, L.N, capacity,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP,
                BLOCK=STEP_BLOCK, ROWS=GATHER_ROWS,
                SAVE=bool(saved), TRACK=L.tracked is not None, SAMPLE=L.sample,
            )  # fmt: skip


def _backward(L, saved, A, scale, grad_spikes, grad_prob, want_A, want_scale):
    """``scan_backward``: returns ``(grad_A, grad_scale, *physiology)`` (None where
    not wanted). ``saved`` is ``(spikes, v_saved, G_saved, g_init)``."""
    grad_A = torch.empty_like(A) if want_A else L.dummy
    partial = torch.empty((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    physiology = _physiology_grads(L, A.device)
    block = L.backward_block()
    kernels.scan_backward[(triton.cdiv(L.B * L.N, block),)](
        grad_spikes.contiguous(),
        L.dummy if grad_prob is None else grad_prob.contiguous(),
        *saved, A, scale, L.made, L.jump, L.decay, L.E_syn, L.g_clip,
        L.p.theta, L.p.E_L, L.p.C_m, L.p.beta, grad_A, partial, *physiology,
        L.p.dt, L.p.surr_scale, L.B, L.T, L.N,
        K=L.K, KP=triton.next_power_of_2(L.K), S=L.S, SP=L.SP, BLOCK=block,
        WANT_A=want_A, WANT_SCALE=want_scale, **L.flags(),
        PROB=grad_prob is not None, SAMPLE=L.sample,
    )  # fmt: skip
    return (
        grad_A if want_A else None,
        partial.sum(dim=1) if want_scale else None,
        *_reduce(physiology, L.wants),
    )


# =====================================================================
# Autograd
# =====================================================================


class _FeedforwardScan(torch.autograd.Function):
    """Spikes (and probabilities) of a chunk, differentiable in ``A``, ``scale`` and
    the physiology (``PHYSIOLOGY``, which the kernels read from ``L``)."""

    @staticmethod
    def forward(ctx, A, scale, v, g, L: _Launch, *physiology):
        spikes = torch.empty((L.B, L.T, L.N), device=A.device)
        prob = torch.empty_like(spikes) if L.probabilities else None
        saved = (
            torch.empty_like(spikes),
            torch.empty((L.B, L.T, L.N, L.S), device=A.device, dtype=L.p.storage_dtype),
        )
        g_init = L.g_init(g)
        _scan(L, A, scale, v, g, spikes, prob, saved)
        ctx.L = L
        ctx.save_for_backward(A, scale, spikes, *saved, g_init)
        ctx.mark_non_differentiable(v, g)
        return spikes, prob, v, g

    @staticmethod
    def backward(ctx, grad_spikes, grad_prob, _v, _g):
        A, scale, *saved = ctx.saved_tensors
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(saved[0])
        grad_A, grad_scale, *physiology = _backward(
            ctx.L, saved, A, scale, grad_spikes, grad_prob,
            want_A=ctx.needs_input_grad[0], want_scale=ctx.needs_input_grad[1],
        )  # fmt: skip
        return grad_A, grad_scale, None, None, None, *physiology


class _RecurrentRun(torch.autograd.Function):
    """Spikes of a recurrent chunk, differentiable in ``A_ff``, ``scale``, ``W`` and
    the physiology (``PHYSIOLOGY``, which the kernels read from ``L``)."""

    @staticmethod
    def forward(ctx, A_ff, scale, W, v, g, L: _Launch, rows, sizes, *physiology):
        spikes = torch.empty((L.B, L.T, L.N), device=A_ff.device)
        saved = (
            torch.empty_like(spikes),
            torch.empty(
                (L.B, L.T, L.N, L.S), device=A_ff.device, dtype=L.p.storage_dtype
            ),
            torch.empty((L.K_rec, L.B, L.T, L.N), device=A_ff.device),
        )
        g_init = L.g_init(g)
        _recurrent(L, [(0, A_ff)], W, rows, sizes, scale, v, g, spikes, saved)
        ctx.L, ctx.rows, ctx.sizes = L, rows, sizes
        ctx.save_for_backward(A_ff, scale, spikes, *saved, g_init)
        ctx.mark_non_differentiable(v, g)
        return spikes, v, g

    @staticmethod
    def backward(ctx, grad_spikes, _v, _g):
        A_ff, scale, spikes, v_saved, G_saved, A_rec, g_init = ctx.saved_tensors
        L = ctx.L
        want_ff, want_scale, want_W = ctx.needs_input_grad[:3]
        A = torch.cat([A_rec, A_ff]) if L.K_ff else A_rec
        grad_A, grad_scale, *physiology = _backward(
            L, (spikes, v_saved, G_saved, g_init), A, scale, grad_spikes, None,
            want_A=want_ff or want_W, want_scale=want_scale,
        )  # fmt: skip
        grad_W = None
        if want_W:
            # dL/dW_k = sum over (trial, step) of presynaptic spikes x dL/dA_k.
            pieces, starts = [], [0, *itertools.accumulate(ctx.sizes)]
            for k in range(L.K_rec):
                pre = spikes[:, :, ctx.rows[starts[k] : starts[k + 1]]]
                pieces.append(
                    pre.reshape(-1, pre.shape[-1]).T @ grad_A[k].reshape(-1, L.N)
                )
            grad_W = torch.cat(pieces)
        grad_ff = grad_A[L.K_rec :] if want_ff else None
        return grad_ff, grad_scale, grad_W, None, None, None, None, None, *physiology


# =====================================================================
# Entry point for the kernels
# =====================================================================


def _simulate_kernels(v, g, x, drive, p, probabilities, track, track_batch_idx) -> Run:
    L = _prepare(v, x, drive, p, probabilities, track, track_batch_idx)
    v = v.detach().float().contiguous().clone()
    # Conductances are stored (B, 2, S, N), so each component is contiguous over neurons,
    # in ``p.storage_dtype`` (as are the saved conductances and the recurrent weights).
    g = torch.empty((L.B, 2, L.S, L.N), dtype=p.storage_dtype, device=x.device).copy_(
        g.detach().permute(0, 2, 3, 1)
    )
    scale = drive.scale.float().contiguous()
    weights = [W for _, W in drive.recurrent] + [W for _, W in drive.feedforward]
    physiology = [getattr(p, name) for name in PHYSIOLOGY]
    grad = torch.is_grad_enabled() and (
        scale.requires_grad
        or x.requires_grad
        or any(W.requires_grad for W in weights)
        or any(L.wants)
    )

    if L.K_rec == 0:
        if grad:
            A = _activations(drive, x).contiguous()
            spikes, prob, v, g = _FeedforwardScan.apply(A, scale, v, g, L, *physiology)
        else:
            spikes = torch.empty((L.B, L.T, L.N), device=x.device)
            prob = torch.empty_like(spikes) if probabilities else None
            with torch.no_grad():
                for t0 in range(0, L.T, FF_TIME_BLOCK):
                    A = _activations(drive, x[:, t0 : t0 + FF_TIME_BLOCK])
                    _scan(L, A.contiguous(), scale, v, g, spikes, prob, None, t0=t0)
        return Run(spikes, v, g.permute(0, 3, 1, 2).float(), prob, L.tracked)

    rows = torch.cat([r for r, _ in drive.recurrent]).int().contiguous()
    sizes = tuple(r.numel() for r, _ in drive.recurrent)
    W = torch.cat([W for _, W in drive.recurrent]).float().contiguous()

    if grad:
        A_ff = _activations(drive, x).contiguous()
        spikes, v, g = _RecurrentRun.apply(
            A_ff, scale, W, v, g, L, rows, sizes, *physiology
        )
    else:
        spikes = torch.empty((L.B, L.T, L.N), device=x.device)

        def blocks():
            for t0 in range(0, L.T, FF_TIME_BLOCK):
                yield (
                    t0,
                    _activations(drive, x[:, t0 : t0 + FF_TIME_BLOCK]).contiguous(),
                )

        with torch.no_grad():
            _recurrent(L, blocks(), W, rows, sizes, scale, v, g, spikes, None)
    return Run(spikes, v, g.permute(0, 3, 1, 2).float(), None, L.tracked)
