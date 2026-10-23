"""Runs the dynamics: the Triton kernels on CUDA, the PyTorch reference otherwise.

``simulate`` is the single entry point the networks call. On a CUDA device it drives
the kernels of ``kernels.py``:

* A network with no recurrent pathways is integrated a chunk at a time by
  ``scan_forward``, after one matmul per pathway computes the whole chunk's synaptic
  input. Without gradients, long chunks are fed in blocks to bound memory.
* A recurrent network takes two launches per step: ``compact_spikes`` lists who fired,
  ``recurrent_step`` gathers just those weight rows and integrates.
* With gradients, the forward pass saves the voltage and recurrent activations of
  every step. A feedforward network's backward pass is one reverse-time scan,
  ``scan_backward``, and so is a recurrent one's when gradients do not flow through the
  recurrent spikes (the saved activations are then just more input). When they do
  (``Drive.recurrent_gradients``, the default here), it is a reverse loop over steps:
  ``inject_backward`` gives every pathway's ``dL/da``, a product with ``W^T`` (sparse,
  for a sparse connectome) carries it to the presynaptic spikes, and
  ``spike_backward`` adds it to their voltages' adjoints.
* Learnt physiology (``theta``, ``U_reset``, ``beta``, ``U_rest``, ``R``, the synaptic
  decays) gets its gradient from the same backward kernels, each term compiled in only
  when that parameter is learnt, from nothing saved but the chunk's final voltages and
  initial currents (see ``kernels.py``).

Set ``WIRETORCH_BACKEND=torch`` to force the reference implementation. What the
families share — the dispatch rule, sparse feedforward input — is in
``network_simulators.engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from wiretorch.network_simulators.current import dynamics
from wiretorch.network_simulators.current.dynamics import Physiology
from wiretorch.network_simulators.engine import (  # noqa: F401 - re-exported
    Gating,
    activations,
    carry_to_sources,
    gate_feedforward,
    gating_tables,
    held_buffer,
    offsets_tensor,
    pad,
    presynaptic as _presynaptic,
    presynaptic_blocks,
    recurrent_weights,
    refractory_tables,
    row_start,
    triton,
    uses_kernels,
    weight_gradient,
)
from wiretorch.network_simulators.network import Drive, Run

if triton is not None:
    from wiretorch.network_simulators.current import kernels

BLOCK = 128  # neurons per program (scan and backward kernels)
STEP_BLOCK = 256  # neurons per program (recurrent step)
GATHER_ROWS = 16  # weight rows of fired neurons summed per load
COMPACT_BLOCK = 1024  # neurons per iteration of the spike compaction
FF_TIME_BLOCK = 256  # steps of feedforward input materialised at once (no gradients)


def simulate(
    v, I, x, drive: Drive, p: Physiology, *, refractory=None, gating=None,
    adaptation=None, probabilities=False, track=False, track_batch_idx=None,
    retain=None,
) -> Run:  # fmt: skip
    """Simulate ``x.shape[1]`` steps; see ``dynamics.simulate`` for the arguments."""
    if refractory is None:
        refractory = torch.zeros(v.shape, dtype=torch.int32, device=v.device)
    if not uses_kernels(x, retain):
        return dynamics.simulate(
            v, I, x, drive, p, refractory=refractory, gating=gating,
            adaptation=adaptation,
            probabilities=probabilities, track=track,
            track_batch_idx=track_batch_idx, retain=retain,
        )  # fmt: skip
    return _simulate_kernels(
        v, I, x, drive, p, refractory, gating, adaptation, probabilities, track,
        track_batch_idx,
    )  # fmt: skip


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
    sign: torch.Tensor  # (SP,)
    decay: torch.Tensor  # (SP,)
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
    #: The recurrent pathways' presynaptic dynamics (short-term plasticity).
    gating: Gating | None = None
    #: The adaptation currents ``(B, N)`` (updated in place), or a dummy.
    a_state: torch.Tensor | None = None

    @property
    def K(self) -> int:
        return self.K_rec + self.K_ff

    @property
    def refractory(self) -> bool:
        """Whether any neuron has a refractory period (the kernels' ``REFRACTORY``)."""
        return self.p.n_ref is not None

    @property
    def adapting(self) -> bool:
        """Whether any neuron adapts (the kernels' ``ADAPT``)."""
        return self.p.adapt is not None

    def adaptation(self) -> tuple[torch.Tensor, ...]:
        """The kernels' adaptation arguments: ``adapt``, ``adapt_decay`` and the
        state, or dummies."""
        if not self.adapting:
            return (self.dummy,) * 3
        return self.p.adapt, self.p.adapt_decay, self.a_state

    def a_init(self) -> torch.Tensor:
        """A copy of the adaptation entering the chunk if its decay is learnt (call
        before the chunk), else a dummy."""
        return self.a_state.clone() if self.wants[7] else self.dummy

    def held(self) -> torch.Tensor:
        return held_buffer(self.B, self.T, self.N, self.refractory, self.dummy)

    def flags(self) -> dict[str, bool]:
        theta, reset, beta, rest, R, decay, adapt, adapt_decay = self.wants
        return dict(
            WANT_THETA=theta, WANT_RESET=reset, WANT_BETA=beta, WANT_REST=rest,
            WANT_R=R, WANT_DECAY=decay, WANT_ADAPT=adapt, WANT_ADAPT_DECAY=adapt_decay,
        )  # fmt: skip

    def ends(self, v, I) -> tuple[torch.Tensor, torch.Tensor]:
        """What the backward pass keeps of the state: the voltages after the chunk
        (for ``beta``, ``R``) and the currents before it (for the decays), or dummies.

        Call with ``I`` before the chunk and ``v`` after it.
        """
        _, _, beta, _, R, decay = self.wants[:6]
        return (
            v.clone() if beta or R else self.dummy,
            I.to(torch.float32, copy=True) if decay else self.dummy,
        )


#: The physiology that can be learnt, as ``Physiology`` names it.
PHYSIOLOGY = (
    "theta", "U_reset", "beta", "U_rest", "R", "decay", "adapt", "adapt_decay",
)  # fmt: skip


def _physiology_grads(L: _Launch, device) -> list[torch.Tensor]:
    """Zeroed per-(trial, neuron) accumulators of the physiology gradients wanted."""
    shapes = [(L.B, L.N)] * 5 + [(L.B, L.S, L.N)] + [(L.B, L.N)] * 2
    return [
        torch.zeros(shape, device=device) if want else L.dummy
        for shape, want in zip(shapes, L.wants)
    ]


def _reduce(partials, wants) -> list[torch.Tensor | None]:
    """Per-neuron (per-synapse, for the decays) sums of the accumulators."""
    return [
        (g.sum((0, 2)) if name == "decay" else g.sum(0)) if want else None
        for name, g, want in zip(PHYSIOLOGY, partials, wants)
    ]


def _prepare(
    v, x, drive, p, refractory, gating, adaptation, probabilities, track,
    track_batch_idx,
) -> _Launch:  # fmt: skip
    B, T = x.shape[:2]
    N = v.shape[1]
    S = p.sign.shape[0]
    SP = max(2, triton.next_power_of_2(S))
    dummy = torch.empty(1, device=x.device)
    sample = p.spike_mode == "sample"
    tracked = None
    if track:
        Bt = B if track_batch_idx is None else 1
        tracked = {
            "v": torch.empty((Bt, T, N), device=x.device),
            "I": torch.empty((Bt, T, N, S), device=x.device),
        }
        if p.adapt is not None:
            tracked["a"] = torch.empty((Bt, T, N), device=x.device)
    n_ref, counter = refractory_tables(p.n_ref, refractory, dummy)
    return _Launch(
        B=B, T=T, N=N, S=S, SP=SP,
        K_rec=len(drive.recurrent), K_ff=len(drive.feedforward),
        made=pad(drive.made.float(), SP), sign=pad(p.sign, SP),
        decay=pad(p.decay.detach(), SP),
        # The kernels get the physiology detached (the autograd functions take it as
        # inputs): launching with tensors that require grad costs host time per launch.
        p=replace(p, **{name: _detached(getattr(p, name)) for name in PHYSIOLOGY}),
        sample=sample, probabilities=probabilities,
        noise=torch.rand((B, T, N), device=x.device) if sample else dummy,
        track_b=(-1 if track_batch_idx is None else track_batch_idx) if track else 0,
        tracked=tracked, dummy=dummy,
        wants=tuple(_requires_grad(getattr(p, name)) for name in PHYSIOLOGY),
        recurrent_gradients=drive.recurrent_gradients,
        n_ref=n_ref, counter=counter, gating=gating_tables(drive, gating, dummy),
        a_state=adaptation,
    )  # fmt: skip


def _detached(t):
    return None if t is None else t.detach()


def _requires_grad(t) -> bool:
    return t is not None and t.requires_grad


def _neurons(p: Physiology):
    return p.theta, p.U_reset, p.beta, p.U_rest, p.R


def _track_buffers(L: _Launch):
    """The tracked ``v``, ``I`` and adaptation buffers, or dummies."""
    if L.tracked is None:
        return (L.dummy,) * 3
    return L.tracked["v"], L.tracked["I"], L.tracked.get("a", L.dummy)


def _scan(L, A, scale, v, I, spikes, prob, saved, t0=0):
    """``scan_forward`` over the ``A.shape[2]`` steps from ``t0``; updates v, I and the
    refractory counters in place. ``saved`` is ``(v_saved, held)``, or None."""
    v_saved, held = saved if saved else (L.dummy, L.dummy)
    kernels.scan_forward[(triton.cdiv(L.B * L.N, BLOCK),)](
        A, scale, L.made, L.sign, L.decay, *_neurons(L.p),
        v, I, spikes, L.dummy if prob is None else prob, L.noise, L.n_ref, L.counter,
        *L.adaptation(), v_saved, held, *_track_buffers(L),
        L.p.surr_scale, L.track_b, t0, A.shape[2], L.B, L.T, L.N,
        K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK,
        SAVE=bool(saved), TRACK=L.tracked is not None,
        PROB=prob is not None, SAMPLE=L.sample, REFRACTORY=L.refractory,
        ADAPT=L.adapting,
    )  # fmt: skip


def _recurrent(L, A_ff_blocks, W, w_offsets, rows, sizes, scale, v, I, spikes, saved):
    """Per step: ``compact_spikes`` then ``recurrent_step``; updates v, I, the
    refractory counters and the gating in place.

    ``A_ff_blocks`` yields ``(t0, A_ff)`` blocks of feedforward activations in order;
    ``rows`` holds the recurrent pathways' source neurons back to back, ``sizes`` how
    many each has, and their weight rows begin at ``w_offsets`` of ``W``. ``saved`` is
    ``(v_saved, A_saved, held, f_saved)``, or None.
    """
    starts = row_start(sizes, v.device)
    w_start = offsets_tensor(w_offsets, v.device)
    v_saved, A_saved, held, f_saved = saved if saved else (L.dummy,) * 4
    gates = L.gating
    W = W.to(L.p.storage_dtype)  # a copy if reduced: gradients stay float32
    capacity = max(sizes)
    lists = torch.empty((L.K_rec, L.B, capacity), dtype=torch.int32, device=v.device)
    counts = torch.zeros((L.K_rec, L.B), dtype=torch.int32, device=v.device)
    amps = torch.empty(lists.shape, device=v.device) if gates.gated else L.dummy
    R = gates.state.shape[1] if gates.gated else 0
    surr = L.p.surr_scale
    for t0, A_ff in A_ff_blocks:
        T_ff = A_ff.shape[2]
        for i in range(T_ff):
            t = t0 + i
            kernels.compact_spikes[(L.B, L.K_rec)](
                v, L.p.theta, L.noise, L.counter, rows, starts, lists, counts,
                gates.params, gates.cols, gates.state, amps, f_saved,
                surr, t, L.B, L.T, L.N, capacity, R, gates.R_f,
                BLOCK=COMPACT_BLOCK, SAMPLE=L.sample, REFRACTORY=L.refractory,
                GATED=gates.gated, SAVE=bool(saved),
            )  # fmt: skip
            kernels.recurrent_step[(L.B, triton.cdiv(L.N, STEP_BLOCK))](
                A_ff, W, w_start, lists, amps, counts, scale, L.made, L.sign, L.decay,
                *_neurons(L.p), v, I, spikes, L.noise, L.n_ref, L.counter,
                *L.adaptation(), v_saved, A_saved, held, *_track_buffers(L),
                surr, L.track_b, t, i, L.T, T_ff, L.B, L.N, capacity,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP,
                BLOCK=STEP_BLOCK, ROWS=GATHER_ROWS,
                SAVE=bool(saved), TRACK=L.tracked is not None, SAMPLE=L.sample,
                REFRACTORY=L.refractory, GATED=gates.gated, ADAPT=L.adapting,
            )  # fmt: skip


def _scan_backward(
    L, saved, a_init, A, scale, grad_spikes, grad_prob, want_A, want_scale
):
    """``scan_backward``: returns ``(grad_A, grad_scale, *physiology)`` (None where
    not wanted). ``saved`` is ``(spikes, v_saved, v_final, I_init, held)``, ``a_init``
    the adaptation entering the chunk (or a dummy)."""
    grad_A = torch.empty_like(A) if want_A else L.dummy
    partial = torch.empty((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    physiology = _physiology_grads(L, A.device)
    kernels.scan_backward[(triton.cdiv(L.B * L.N, BLOCK),)](
        grad_spikes.contiguous(),
        L.dummy if grad_prob is None else grad_prob.contiguous(),
        *saved, A, scale, L.made, L.sign, L.decay,
        L.p.theta, L.p.beta, L.p.U_rest, L.p.R, *L.adaptation()[:2], a_init,
        grad_A, partial, *physiology,
        L.p.surr_scale, L.B, L.T, L.N,
        K=L.K, KP=triton.next_power_of_2(L.K), S=L.S, SP=L.SP, BLOCK=BLOCK,
        WANT_A=want_A, WANT_SCALE=want_scale, **L.flags(),
        PROB=grad_prob is not None, SAMPLE=L.sample, REFRACTORY=L.refractory,
        ADAPT=L.adapting,
    )  # fmt: skip
    return (
        grad_A if want_A else None,
        partial.sum(dim=1) if want_scale else None,
        *_reduce(physiology, L.wants),
    )


def _recurrent_backward(
    L, saved, a_init, A, scale, W, w_offsets, rows, sizes, factors, grad_spikes,
    want_scale,
):  # fmt: skip
    """The reverse loop over steps: returns ``(grad_A, grad_scale, *physiology)``.

    ``grad_A`` ``(K, B, T, N)`` is always computed: its recurrent part is what reaches
    the presynaptic spikes (a presynaptic pathway's times its ``factors``).
    ``grad_scale`` and the physiology are None unless wanted.
    ``saved`` is ``(spikes, v_saved, v_final, I_init, held)``.
    """
    spikes, v_saved, v_final, I_init, held = saved
    grid = (triton.cdiv(L.B * L.N, BLOCK),)
    lv = torch.zeros((L.B, L.N), device=A.device)
    lI = torch.zeros((L.B, L.S, L.N), device=A.device)
    R = torch.zeros((L.B, L.N), device=A.device)  # dL/ds through the recurrence
    grad_A = torch.empty_like(A)
    partial = torch.zeros((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    physiology = _physiology_grads(L, A.device)
    g_theta, g_reset, g_beta, g_rest, g_R, g_decay, g_adapt, g_adapt_decay = physiology
    H = torch.zeros_like(lI) if L.wants[5] else L.dummy
    lA = torch.zeros((L.B, L.N), device=A.device) if L.adapting else L.dummy
    H_a = torch.zeros((L.B, L.N), device=A.device) if L.wants[7] else L.dummy
    flags = L.flags()
    want_theta = flags.pop("WANT_THETA")
    blocks = presynaptic_blocks(W, rows, sizes, L.p.storage_dtype, w_offsets)
    shared = len(set(w_offsets)) < len(w_offsets)  # sources in more than one pathway
    grad_spikes = grad_spikes.contiguous()
    for t in reversed(range(L.T)):
        kernels.inject_backward[grid](
            lv, lI, H, lA, H_a, spikes, v_saved, v_final, held, A, scale, L.made,
            L.sign, L.decay, L.p.beta, L.p.U_rest, L.p.R, *L.adaptation()[:2],
            grad_A, partial, g_reset, g_beta, g_rest, g_R, g_decay, g_adapt,
            g_adapt_decay, t, L.B, L.T, L.N,
            K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK, WANT_SCALE=want_scale, **flags,
            REFRACTORY=L.refractory, ADAPT=L.adapting,
        )  # fmt: skip
        carry_to_sources(R, blocks, grad_A[:, :, t], factors, t, shared)
        kernels.spike_backward[grid](
            lv, v_saved, held, grad_spikes, R, L.p.theta, g_theta, L.p.surr_scale,
            t, L.B, L.T, L.N, BLOCK=BLOCK, SAMPLE=L.sample, WANT_THETA=want_theta,
            REFRACTORY=L.refractory,
        )  # fmt: skip
    if L.wants[5]:
        g_decay += I_init * H  # the currents entering the chunk
    if L.wants[7]:
        g_adapt_decay += a_init * H_a  # the adaptation entering the chunk
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
    the physiology (``PHYSIOLOGY``, which the kernels read from ``L.p``)."""

    @staticmethod
    def forward(ctx, A, scale, v, I, L: _Launch, *physiology):
        spikes = torch.empty((L.B, L.T, L.N), device=A.device)
        prob = torch.empty_like(spikes) if L.probabilities else None
        v_saved = torch.empty_like(spikes)
        held = L.held()
        I_init = L.ends(v, I)[1]
        a_init = L.a_init()
        _scan(L, A, scale, v, I, spikes, prob, (v_saved, held))
        ctx.L = L
        ctx.save_for_backward(
            A, scale, spikes, v_saved, L.ends(v, I)[0], I_init, held, a_init
        )
        ctx.mark_non_differentiable(v, I)
        return spikes, prob, v, I

    @staticmethod
    def backward(ctx, grad_spikes, grad_prob, _v, _I):
        A, scale, *saved, a_init = ctx.saved_tensors
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(saved[0])
        grad_A, grad_scale, *physiology = _scan_backward(
            ctx.L, saved, a_init, A, scale, grad_spikes, grad_prob,
            want_A=ctx.needs_input_grad[0], want_scale=ctx.needs_input_grad[1],
        )  # fmt: skip
        return grad_A, grad_scale, None, None, None, *physiology


class _RecurrentRun(torch.autograd.Function):
    """Spikes of a recurrent chunk, differentiable in ``A_ff``, ``scale``, ``W`` and
    the physiology (``PHYSIOLOGY``, which the kernels read from ``L.p``)."""

    @staticmethod
    def forward(
        ctx, A_ff, scale, W, v, I, L: _Launch, w_offsets, rows, sizes, *physiology
    ):
        spikes = torch.empty((L.B, L.T, L.N), device=A_ff.device)
        v_saved = torch.empty_like(spikes)
        A_rec = torch.empty((L.K_rec, L.B, L.T, L.N), device=A_ff.device)
        held = L.held()
        f_saved = L.gating.f_saved(L.B, L.T, L.dummy)
        I_init = L.ends(v, I)[1]
        a_init = L.a_init()
        _recurrent(
            L, [(0, A_ff)], W, w_offsets, rows, sizes, scale, v, I, spikes,
            (v_saved, A_rec, held, f_saved),
        )  # fmt: skip
        ctx.L, ctx.w_offsets, ctx.rows, ctx.sizes = L, w_offsets, rows, sizes
        ctx.n_weights = W.shape[0]
        ctx.save_for_backward(
            A_ff, scale, W, spikes, v_saved, L.ends(v, I)[0], I_init, held, A_rec,
            f_saved, a_init,
        )  # fmt: skip
        ctx.mark_non_differentiable(v, I)
        return spikes, v, I

    @staticmethod
    def backward(ctx, grad_spikes, _v, _I):
        A_ff, scale, W, *saved, A_rec, f_saved, a_init = ctx.saved_tensors
        L, w_offsets, rows, sizes = ctx.L, ctx.w_offsets, ctx.rows, ctx.sizes
        factors = L.gating.factors(f_saved, sizes)
        want_ff, want_scale, want_W = ctx.needs_input_grad[:3]
        spikes = saved[0]
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(spikes)
        A = (torch.cat([A_rec, A_ff]) if L.K_ff else A_rec).contiguous()
        if L.recurrent_gradients:
            grad_A, grad_scale, *physiology = _recurrent_backward(
                L, saved, a_init, A, scale, W, w_offsets, rows, sizes, factors,
                grad_spikes, want_scale,
            )  # fmt: skip
        else:  # the recurrent activations are input like any other: one reverse scan
            grad_A, grad_scale, *physiology = _scan_backward(
                L, saved, a_init, A, scale, grad_spikes, None,
                want_A=want_ff or want_W, want_scale=want_scale,
            )  # fmt: skip
        grad_W = (
            weight_gradient(
                spikes, rows, sizes, grad_A, w_offsets, ctx.n_weights, factors
            )
            if want_W
            else None
        )
        grad_ff = grad_A[L.K_rec :] if want_ff else None
        return (
            grad_ff, grad_scale, grad_W, None, None, None, None, None, None,
            *physiology,
        )  # fmt: skip


# =====================================================================
# Entry point for the kernels
# =====================================================================


def _simulate_kernels(
    v, I, x, drive, p, refractory, gating, adaptation, probabilities, track,
    track_batch_idx,
) -> Run:  # fmt: skip
    if gating is not None:  # updated in place by the kernels
        gating = gating.detach().float().contiguous().clone()
    if p.adapt is not None:  # likewise
        adaptation = (
            torch.zeros_like(v, dtype=torch.float32)
            if adaptation is None
            else adaptation.detach().float().contiguous().clone()
        )
    L = _prepare(
        v, x, drive, p, refractory, gating, adaptation, probabilities, track,
        track_batch_idx,
    )  # fmt: skip
    counter = L.counter if L.refractory else refractory
    v = v.detach().float().contiguous().clone()
    # Currents are stored (B, S, N), so each synapse type is contiguous over neurons,
    # in ``p.storage_dtype`` (as are the recurrent weights).
    I = torch.empty((L.B, L.S, L.N), dtype=p.storage_dtype, device=x.device).copy_(
        I.detach().permute(0, 2, 1)
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

    def inputs(x):
        """The feedforward activations of ``x``, the next steps of the chunk (the
        feedforward gating is stepped through them)."""
        factors, _ = gate_feedforward(drive, x, gating)
        return activations(drive, x, factors).contiguous()

    def run(spikes, prob=None):
        state = I.permute(0, 2, 1).float()
        return Run(spikes, v, state, prob, L.tracked, counter, gating, adaptation)

    if L.K_rec == 0:
        if grad:
            A = inputs(x)
            spikes, prob, v, I = _FeedforwardScan.apply(A, scale, v, I, L, *physiology)
        else:
            spikes = torch.empty((L.B, L.T, L.N), device=x.device)
            prob = torch.empty_like(spikes) if probabilities else None
            with torch.no_grad():
                for t0 in range(0, L.T, FF_TIME_BLOCK):
                    A = inputs(x[:, t0 : t0 + FF_TIME_BLOCK])
                    _scan(L, A, scale, v, I, spikes, prob, None, t0=t0)
        return run(spikes, prob)

    rows = torch.cat([r for r, _ in drive.recurrent]).int().contiguous()
    sizes = tuple(r.numel() for r, _ in drive.recurrent)
    W, w_offsets = recurrent_weights(drive.recurrent)

    if grad:
        A_ff = inputs(x)
        spikes, v, I = _RecurrentRun.apply(
            A_ff, scale, W, v, I, L, w_offsets, rows, sizes, *physiology
        )
    else:
        spikes = torch.empty((L.B, L.T, L.N), device=x.device)

        def blocks():
            for t0 in range(0, L.T, FF_TIME_BLOCK):
                yield t0, inputs(x[:, t0 : t0 + FF_TIME_BLOCK])

        with torch.no_grad():
            _recurrent(
                L, blocks(), W, w_offsets, rows, sizes, scale, v, I, spikes, None
            )
    return run(spikes)
