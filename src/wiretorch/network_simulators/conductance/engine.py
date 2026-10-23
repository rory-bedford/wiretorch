"""Runs the dynamics: the Triton kernels on CUDA, the PyTorch reference otherwise.

``simulate`` is the single entry point the networks call. On a CUDA device it drives
the kernels of ``kernels.py``:

* A network with no recurrent pathways is integrated a chunk at a time by
  ``scan_forward``, after one matmul per pathway computes the whole chunk's synaptic
  input. Without gradients, long chunks are fed in blocks to bound memory.
* A recurrent network takes two launches per step: ``compact_spikes`` lists who fired,
  ``recurrent_step`` gathers just those weight rows and integrates.
* With gradients, the forward pass saves the voltage, conductances and recurrent
  activations of every step, and ``scan_backward`` is the reverse-time adjoint — unless
  gradients flow through the recurrent spikes (``Drive.recurrent_gradients``), which
  couples the neurons: the backward pass is then a reverse loop over steps,
  ``inject_backward``, a product with ``W^T`` carrying each pathway's ``dL/da`` to the
  presynaptic spikes, and ``spike_backward``. Either also gives learnt physiology its
  gradient, each term compiled in only when that parameter is learnt, from nothing more
  saved than the conductances entering the chunk.

Set ``WIRETORCH_BACKEND=torch`` to force the reference implementation. What the
families share — the dispatch rule, sparse feedforward input — is in
``network_simulators.engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from wiretorch.network_simulators.conductance import dynamics
from wiretorch.network_simulators.conductance.dynamics import (
    Drive,
    Physiology,
    Run,
)
from wiretorch.network_simulators.engine import (  # noqa: F401 - re-exported
    SPARSE_INPUT,
    _SparseActivations,
    activations as _activations,
    held_buffer,
    pad as _pad,
    presynaptic_blocks,
    refractory_tables,
    row_start as _row_start,
    triton,
    uses_kernels,
    weight_gradient,
)

if triton is not None:
    from wiretorch.network_simulators.conductance import kernels

BLOCK = 128  # neurons per program (scan kernels)
STEP_BLOCK = 256  # neurons per program (recurrent step)
GATHER_ROWS = 16  # weight rows of fired neurons summed per load
COMPACT_BLOCK = 1024  # neurons per iteration of the spike compaction
FF_TIME_BLOCK = 256  # steps of feedforward input materialised at once (no gradients)


def simulate(
    v, g, x, drive: Drive, p: Physiology, *, refractory=None, probabilities=False,
    track=False, track_batch_idx=None, retain=None,
) -> Run:  # fmt: skip
    """Simulate ``x.shape[1]`` steps; see ``dynamics.simulate`` for the arguments."""
    if refractory is None:
        refractory = torch.zeros(v.shape, dtype=torch.int32, device=v.device)
    if not uses_kernels(x, retain):
        return dynamics.simulate(
            v, g, x, drive, p, refractory=refractory,
            probabilities=probabilities, track=track,
            track_batch_idx=track_batch_idx, retain=retain,
        )  # fmt: skip
    return _simulate_kernels(
        v, g, x, drive, p, refractory, probabilities, track, track_batch_idx
    )


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
    mg_a: torch.Tensor  # (SP,): the Mg block's slope
    mg_lc: torch.Tensor  # (SP,): log of its mg / mg_k (-inf: no block)
    p: Physiology
    sample: bool
    probabilities: bool
    noise: torch.Tensor  # (B, T, N), or a placeholder
    track_b: int  # tracked trial, -1 for all, or 0 when not tracking
    tracked: dict[str, torch.Tensor] | None
    dummy: torch.Tensor  # stands in for unused pointer arguments
    #: Which physiology gradients are wanted, in ``PHYSIOLOGY`` order.
    wants: tuple[bool, ...] = (False,) * 8
    #: Whether gradients flow through the recurrent spikes (see ``Drive``).
    recurrent_gradients: bool = False
    #: The refractory periods in steps, ``(N,)`` int32, and the counters of the steps
    #: of it left, ``(B, N)`` int32 (updated in place): dummies if no neuron has one.
    n_ref: torch.Tensor | None = None
    counter: torch.Tensor | None = None

    @property
    def K(self) -> int:
        return self.K_rec + self.K_ff

    @property
    def refractory(self) -> bool:
        """Whether any neuron has a refractory period (the kernels' ``REFRACTORY``)."""
        return self.p.n_ref is not None

    @property
    def mg_block(self) -> bool:
        """Whether any synapse type has a Mg block (the kernels' ``MG_BLOCK``)."""
        return self.p.mg_c is not None

    def held(self) -> torch.Tensor:
        return held_buffer(self.B, self.T, self.N, self.refractory, self.dummy)

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
        torch.zeros(shape, device=device) if want else L.dummy
        for shape, want in zip(shapes, L.wants)
    ]


def _reduce(partials, wants) -> list[torch.Tensor | None]:
    """Per-neuron (per-synapse; per component and synapse) sums of the accumulators."""
    dims = [0] * 5 + [(0, 2)] + [(0, 3)] * 2
    return [g.sum(d) if want else None for g, d, want in zip(partials, dims, wants)]


def _prepare(
    v, x, drive, p, refractory, probabilities, track, track_batch_idx
) -> _Launch:
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
    n_ref, counter = refractory_tables(p.n_ref, refractory, dummy)
    unused = torch.zeros_like(p.E_syn.detach())  # Mg block tables when there is none
    return _Launch(
        B=B, T=T, N=N, S=S, SP=SP,
        K_rec=len(drive.recurrent), K_ff=len(drive.feedforward),
        made=_pad(drive.made.float(), SP),
        jump=_pad(p.jump.detach(), SP), decay=_pad(p.decay.detach(), SP),
        E_syn=_pad(p.E_syn.detach(), SP), g_clip=_pad(p.g_clip, SP),
        mg_a=_pad(p.mg_a if p.mg_c is not None else unused, SP),
        mg_lc=_pad(torch.log(p.mg_c) if p.mg_c is not None else unused, SP),
        # The kernels get the physiology detached (the autograd functions take it as
        # inputs): launching with tensors that require grad costs host time per launch.
        p=replace(p, **{name: getattr(p, name).detach() for name in PHYSIOLOGY}),
        sample=sample, probabilities=probabilities,
        noise=torch.rand((B, T, N), device=x.device) if sample else dummy,
        track_b=(-1 if track_batch_idx is None else track_batch_idx) if track else 0,
        tracked=tracked, dummy=dummy,
        wants=tuple(getattr(p, name).requires_grad for name in PHYSIOLOGY),
        recurrent_gradients=drive.recurrent_gradients,
        n_ref=n_ref, counter=counter,
    )  # fmt: skip


def _neurons(p: Physiology):
    return p.theta, p.U_reset, p.E_L, p.C_m, p.beta


def _track_buffers(L: _Launch):
    if L.tracked is None:
        return L.dummy, L.dummy, L.dummy
    return L.tracked["v"], L.tracked["I"], L.tracked["g"]


def _scan(L, A, scale, v, g, spikes, prob, saved, t0=0):
    """``scan_forward`` over the ``A.shape[2]`` steps from ``t0``; updates v, g and the
    refractory counters in place. ``saved`` is ``(v_saved, G_saved, held)``, or None."""
    v_saved, G_saved, held = saved if saved else (L.dummy,) * 3
    TA = A.shape[2]
    kernels.scan_forward[(triton.cdiv(L.B * L.N, BLOCK),)](
        A, scale, L.made, L.jump, L.decay, L.E_syn, L.g_clip, L.mg_a, L.mg_lc,
        *_neurons(L.p),
        v, g, spikes, L.dummy if prob is None else prob, L.noise, L.n_ref, L.counter,
        v_saved, G_saved, held, *_track_buffers(L),
        L.p.dt, L.p.surr_scale, L.track_b, t0, TA, L.B, L.T, L.N,
        K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK,
        SAVE=bool(saved), TRACK=L.tracked is not None,
        PROB=prob is not None, SAMPLE=L.sample, REFRACTORY=L.refractory,
        MG_BLOCK=L.mg_block,
    )  # fmt: skip


def _recurrent(L, A_ff_blocks, W, rows, sizes, scale, v, g, spikes, saved):
    """Per step: ``compact_spikes`` then ``recurrent_step``; updates v, g and the
    refractory counters in place.

    ``A_ff_blocks`` yields ``(t0, A_ff)`` blocks of feedforward activations in order;
    ``rows`` holds the recurrent pathways' source neurons back to back, ``sizes`` how
    many each has. ``saved`` is ``(v_saved, G_saved, A_saved, held)``, or None.
    """
    row_start = _row_start(sizes, v.device)
    w_start = row_start[:-1]
    v_saved, G_saved, A_saved, held = saved if saved else (L.dummy,) * 4
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
                    v, L.p.theta, L.noise, L.counter, rows, row_start, lists, counts,
                    surr, t, L.B, L.T, L.N, capacity,
                    BLOCK=COMPACT_BLOCK, SAMPLE=L.sample, REFRACTORY=L.refractory,
                )  # fmt: skip
            kernels.recurrent_step[(L.B, triton.cdiv(L.N, STEP_BLOCK))](
                A_ff, W, w_start, lists, counts, scale, L.made, L.jump, L.decay,
                L.E_syn, L.g_clip, L.mg_a, L.mg_lc, *_neurons(L.p),
                v, g, spikes, L.noise, L.n_ref, L.counter,
                v_saved, G_saved, A_saved, held, *_track_buffers(L),
                dt, surr, L.track_b, t, i, L.T, T_ff, L.B, L.N, capacity,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP,
                BLOCK=STEP_BLOCK, ROWS=GATHER_ROWS,
                SAVE=bool(saved), TRACK=L.tracked is not None, SAMPLE=L.sample,
                REFRACTORY=L.refractory, MG_BLOCK=L.mg_block,
            )  # fmt: skip


def _backward(L, saved, A, scale, grad_spikes, grad_prob, want_A, want_scale):
    """``scan_backward``: returns ``(grad_A, grad_scale, *physiology)`` (None where
    not wanted). ``saved`` is ``(spikes, v_saved, G_saved, g_init, held)``."""
    grad_A = torch.empty_like(A) if want_A else L.dummy
    partial = torch.empty((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    physiology = _physiology_grads(L, A.device)
    block = L.backward_block()
    kernels.scan_backward[(triton.cdiv(L.B * L.N, block),)](
        grad_spikes.contiguous(),
        L.dummy if grad_prob is None else grad_prob.contiguous(),
        *saved, A, scale, L.made, L.jump, L.decay, L.E_syn, L.g_clip,
        L.mg_a, L.mg_lc,
        L.p.theta, L.p.E_L, L.p.C_m, L.p.beta, grad_A, partial, *physiology,
        L.p.dt, L.p.surr_scale, L.B, L.T, L.N,
        K=L.K, KP=triton.next_power_of_2(L.K), S=L.S, SP=L.SP, BLOCK=block,
        WANT_A=want_A, WANT_SCALE=want_scale, **L.flags(),
        PROB=grad_prob is not None, SAMPLE=L.sample, REFRACTORY=L.refractory,
        MG_BLOCK=L.mg_block,
    )  # fmt: skip
    return (
        grad_A if want_A else None,
        partial.sum(dim=1) if want_scale else None,
        *_reduce(physiology, L.wants),
    )


def _recurrent_backward(L, saved, A, scale, W, rows, sizes, grad_spikes, want_scale):
    """The reverse loop over steps, when gradients flow through the recurrent spikes:
    returns ``(grad_A, grad_scale, *physiology)``.

    ``grad_A`` ``(K, B, T, N)`` is always computed: its recurrent part is what reaches
    the presynaptic spikes. ``grad_scale`` and the physiology are None unless wanted.
    ``saved`` is ``(spikes, v_saved, G_saved, g_init, held)``.
    """
    spikes, v_saved, G_saved, g_init, held = saved
    block = L.backward_block()
    grid = (triton.cdiv(L.B * L.N, block),)
    lv = torch.zeros((L.B, L.N), device=A.device)
    lg = torch.zeros((L.B, 2, L.S, L.N), device=A.device)
    R = torch.zeros((L.B, L.N), device=A.device)  # dL/ds through the recurrence
    grad_A = torch.empty_like(A)
    partial = torch.zeros((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    physiology = _physiology_grads(L, A.device)
    g_theta, *rest = physiology
    H = torch.zeros_like(lg) if L.wants[6] else L.dummy
    flags = L.flags()
    want_theta = flags.pop("WANT_THETA")
    blocks = presynaptic_blocks(W, rows, sizes, L.p.storage_dtype)
    grad_spikes = grad_spikes.contiguous()
    for t in reversed(range(L.T)):
        kernels.inject_backward[grid](
            lv, lg, H, spikes, v_saved, G_saved, held, A, scale, L.made, L.jump,
            L.decay, L.E_syn, L.g_clip, L.mg_a, L.mg_lc, L.p.E_L, L.p.C_m, L.p.beta,
            grad_A, partial, *rest, L.p.dt, t, L.B, L.T, L.N,
            K=L.K, S=L.S, SP=L.SP, BLOCK=block, WANT_SCALE=want_scale, **flags,
            REFRACTORY=L.refractory, MG_BLOCK=L.mg_block,
        )  # fmt: skip
        for k, (source, carry) in enumerate(blocks):
            R.index_copy_(1, source, carry(grad_A[k, :, t]))
        kernels.spike_backward[grid](
            lv, v_saved, held, grad_spikes, R, L.p.theta, g_theta, L.p.surr_scale,
            t, L.B, L.T, L.N, BLOCK=block, SAMPLE=L.sample, WANT_THETA=want_theta,
            REFRACTORY=L.refractory,
        )  # fmt: skip
    if L.wants[6]:
        physiology[6] += g_init * H  # the conductances entering the chunk
    return (
        grad_A,
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
        v_saved = torch.empty_like(spikes)
        G_saved = torch.empty(
            (L.B, L.T, L.N, L.S), device=A.device, dtype=L.p.storage_dtype
        )
        held = L.held()
        g_init = L.g_init(g)
        _scan(L, A, scale, v, g, spikes, prob, (v_saved, G_saved, held))
        ctx.L = L
        ctx.save_for_backward(A, scale, spikes, v_saved, G_saved, g_init, held)
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
        v_saved = torch.empty_like(spikes)
        G_saved = torch.empty(
            (L.B, L.T, L.N, L.S), device=A_ff.device, dtype=L.p.storage_dtype
        )
        A_rec = torch.empty((L.K_rec, L.B, L.T, L.N), device=A_ff.device)
        held = L.held()
        g_init = L.g_init(g)
        _recurrent(
            L, [(0, A_ff)], W, rows, sizes, scale, v, g, spikes,
            (v_saved, G_saved, A_rec, held),
        )  # fmt: skip
        ctx.L, ctx.rows, ctx.sizes = L, rows, sizes
        # W is needed only to carry gradients through the recurrent spikes.
        kept = W if L.recurrent_gradients else L.dummy
        ctx.save_for_backward(
            A_ff, scale, kept, spikes, v_saved, G_saved, A_rec, g_init, held
        )
        ctx.mark_non_differentiable(v, g)
        return spikes, v, g

    @staticmethod
    def backward(ctx, grad_spikes, _v, _g):
        A_ff, scale, W, spikes, v_saved, G_saved, A_rec, g_init, held = (
            ctx.saved_tensors
        )
        L, rows, sizes = ctx.L, ctx.rows, ctx.sizes
        want_ff, want_scale, want_W = ctx.needs_input_grad[:3]
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(spikes)
        A = (torch.cat([A_rec, A_ff]) if L.K_ff else A_rec).contiguous()
        saved = (spikes, v_saved, G_saved, g_init, held)
        if L.recurrent_gradients:
            grad_A, grad_scale, *physiology = _recurrent_backward(
                L, saved, A, scale, W, rows, sizes, grad_spikes, want_scale
            )
        else:  # the recurrent activations are input like any other: one reverse scan
            grad_A, grad_scale, *physiology = _backward(
                L, saved, A, scale, grad_spikes, None,
                want_A=want_ff or want_W, want_scale=want_scale,
            )  # fmt: skip
        grad_W = weight_gradient(spikes, rows, sizes, grad_A) if want_W else None
        grad_ff = grad_A[L.K_rec :] if want_ff else None
        return grad_ff, grad_scale, grad_W, None, None, None, None, None, *physiology


# =====================================================================
# Entry point for the kernels
# =====================================================================


def _simulate_kernels(
    v, g, x, drive, p, refractory, probabilities, track, track_batch_idx
) -> Run:
    L = _prepare(v, x, drive, p, refractory, probabilities, track, track_batch_idx)
    counter = L.counter if L.refractory else refractory
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
        return Run(spikes, v, g.permute(0, 3, 1, 2).float(), prob, L.tracked, counter)

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
    return Run(spikes, v, g.permute(0, 3, 1, 2).float(), None, L.tracked, counter)
