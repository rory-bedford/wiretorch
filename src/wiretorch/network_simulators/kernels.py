"""Triton kernels every spiking family shares: the spike (and its surrogate gradient),
the spike compaction, and the spike's step of a recurrent backward pass.

The refractory period (``_refractory``) is compiled in only when some neuron has one
(``REFRACTORY``): a refractory neuron cannot spike (nor be listed by the compaction)
and passes no gradient back through its spike.

A recurrent step starts with ``compact_spikes``, which lists, per trial and recurrent
pathway, the source neurons that fire now; the family's ``recurrent_step`` then gathers
only those weight rows (spikes are sparse, ~1% of neurons per ms).

A saturating pathway (see ``network.gate_step``) scales each spike by the fraction of
its source's closed receptors it opens, from the source's gating variable: with
``GATED``, ``compact_spikes`` steps the recurrent pathways' gating variables and lists
each fired row's factor beside it, and ``gate_scan`` steps the feedforward ones through
a chunk of input.

When gradients flow through the recurrent spikes, the backward pass steps back one step
at a time: the family's ``inject_backward`` gives each pathway's ``dL/da``, the engine
carries it to the presynaptic spikes (``R = dL/da W^T``), and ``spike_backward`` adds
the spikes' surrogate gradient to the voltages' adjoints.
"""

import triton
import triton.language as tl


@triton.jit
def _spike(v, theta, noise, surr_scale, SAMPLE: tl.constexpr):
    """Spikes from the voltage entering the step (Heaviside, or Bernoulli escape noise)."""
    if SAMPLE:
        p = tl.sigmoid(surr_scale * (v - theta))
        return (noise < p).to(tl.float32)
    return ((v - theta) > 0).to(tl.float32)


@triton.jit
def _refractory(s, counter, n_ref):
    """The refractory period's part of a step (``network.refractory_mask`` and what
    follows it), from ``counter``, the steps of it left entering the step.

    Returns the spikes (none while refractory), what holds the voltage at reset (a
    spike, or the period), the counter after the step, and the refractory mask.
    """
    held = (counter > 0).to(tl.float32)
    s = s * (1 - held)
    counter = tl.where(s > 0, n_ref, tl.maximum(counter - 1, 0))
    return s, tl.maximum(s, held), counter, held


@triton.jit
def _spike_adjoint(x, grad, surr_scale, SAMPLE: tl.constexpr):
    """``grad`` (dL/ds) carried to the voltage by the surrogate, at ``x = v - theta``.

    The fast-sigmoid derivative of ``SurrGradSpike``, or the logistic derivative of
    ``SampleSpike``.
    """
    if SAMPLE:
        p = tl.sigmoid(surr_scale * x)
        return grad * surr_scale * p * (1.0 - p)
    return grad / ((surr_scale * tl.abs(x) + 1.0) * (surr_scale * tl.abs(x) + 1.0))


@triton.jit
def _gate(u, s, saturation, decay):
    """``network.gate_step``: the factor ``f`` of a spike ``s``, and ``u`` after it."""
    u = u * decay
    f = saturation * (1 - u)
    return f, tl.minimum(u + s * f, 1.0)


@triton.jit
def gate_scan(
    x, rows, saturation, decay, gating, factors, B, T, n_in, R, RG, col0,
    BLOCK: tl.constexpr,
):  # fmt: skip
    """Step the gating variables of the saturating feedforward pathways through a
    chunk of input ``x`` ``(B, T, n_in)``.

    The ``RG`` gating columns ``col0 ..`` of ``gating`` ``(B, R)`` (updated in place)
    belong to input rows ``rows`` ``(RG,)`` int32, with their pathway's ``saturation``
    and ``decay`` ``(RG,)``; ``factors`` ``(B, T, RG)`` gets each step's ``f``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, column)
    live = idx < B * RG
    b = idx // RG
    i = idx % RG
    row = tl.load(rows + i, mask=live, other=0)
    c = tl.load(saturation + i, mask=live, other=0.0)
    d = tl.load(decay + i, mask=live, other=0.0)
    u = tl.load(gating + b * R + col0 + i, mask=live, other=0.0)
    for t in range(T):
        s = tl.load(x + (b * T + t) * n_in + row, mask=live, other=0.0)
        f, u = _gate(u, s, c, d)
        tl.store(factors + (b * T + t) * RG + i, f, mask=live)
    tl.store(gating + b * R + col0 + i, u, mask=live)


@triton.jit
def compact_spikes(
    v, theta, noise, counter, rows, row_start, lists, counts,
    gate_saturation, gate_decay, gate_col, gating, amps, f_saved,
    surr_ptr, t, B, T, N, L, R, R_SAVED,
    BLOCK: tl.constexpr, SAMPLE: tl.constexpr, REFRACTORY: tl.constexpr,
    GATED: tl.constexpr, SAVE: tl.constexpr,
):  # fmt: skip
    """List, in order, the rows of recurrent pathway ``k`` whose neuron fires now.

    One program per (trial, pathway). ``rows`` holds every recurrent pathway's source
    neurons back to back, ``row_start`` (K + 1) where each pathway's begin. A neuron
    whose refractory ``counter`` ``(B, N)`` is not zero does not fire.

    With ``GATED`` (some pathway saturates), ``amps`` ``(K, B, L)`` gets each listed
    row's factor beside it: 1, or for a saturating pathway (``gate_col[k]``, its first
    column of ``gating`` ``(B, R)``, is not -1) its ``f``, as its gating variables are
    stepped (in place); with ``SAVE`` also every row's ``f`` into ``f_saved``
    ``(B, T, R_SAVED)``, for the backward pass.
    """
    surr_scale = tl.load(surr_ptr)
    b = tl.program_id(0)
    k = tl.program_id(1)
    start = tl.load(row_start + k)
    n_rows = tl.load(row_start + k + 1) - start
    count = 0
    for base in range(0, n_rows, BLOCK):
        r = base + tl.arange(0, BLOCK)
        m = r < n_rows
        neuron = tl.load(rows + start + r, mask=m, other=0)
        vv = tl.load(v + b * N + neuron, mask=m, other=0.0)
        th = tl.load(theta + neuron, mask=m, other=0.0)
        if SAMPLE:
            u = tl.load(noise + (b * T + t) * N + neuron, mask=m, other=1.0)
        else:
            u = 0.0
        fired = (_spike(vv, th, u, surr_scale, SAMPLE) > 0) & m
        if REFRACTORY:
            fired = fired & (tl.load(counter + b * N + neuron, mask=m, other=0) == 0)
        position = count + tl.cumsum(fired.to(tl.int32), axis=0) - 1
        tl.store(lists + (k * B + b) * L + position, r, mask=fired)
        if GATED:
            col = tl.load(gate_col + k)
            if col >= 0:
                gu = tl.load(gating + b * R + col + r, mask=m, other=0.0)
                f, gu = _gate(
                    gu, fired.to(tl.float32), tl.load(gate_saturation + k),
                    tl.load(gate_decay + k),
                )  # fmt: skip
                tl.store(gating + b * R + col + r, gu, mask=m)
                if SAVE:
                    tl.store(f_saved + (b * T + t) * R_SAVED + col + r, f, mask=m)
            else:
                f = tl.full((BLOCK,), 1.0, tl.float32)
            tl.store(amps + (k * B + b) * L + position, f, mask=fired)
        count += tl.sum(fired.to(tl.int32), axis=0)
    tl.store(counts + k * B + b, count)


@triton.jit
def _accumulate(pointer, value, mask):
    """``*pointer += value`` (one program owns each element)."""
    tl.store(pointer, tl.load(pointer, mask=mask, other=0.0) + value, mask=mask)


@triton.jit
def spike_backward(
    lv, v_saved, held_saved, grad_spikes, R, theta, grad_theta, surr_ptr, t, B, T, N,
    BLOCK: tl.constexpr, SAMPLE: tl.constexpr, WANT_THETA: tl.constexpr,
    REFRACTORY: tl.constexpr,
):  # fmt: skip
    """Add step ``t``'s spike term to dL/dv: the surrogate of the spike's gradient.

    The spike's gradient is the loss's (``grad_spikes``) plus ``R`` ``(B, N)``, what
    the recurrent activations of step ``t`` pass back to it. The spike is a function
    of ``v - theta``, so ``theta``'s gradient is the same term, negated.
    A refractory neuron's spike (``held_saved`` ``(B, T, N)`` int8) is held at zero,
    so it passes nothing back.
    """
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    out = (b * T + t) * N + n
    x = tl.load(v_saved + out, mask=live) - tl.load(theta + n, mask=live)
    gs = tl.load(grad_spikes + out, mask=live, other=0.0)
    gs += tl.load(R + idx, mask=live, other=0.0)
    gx = _spike_adjoint(x, gs, surr_scale, SAMPLE)
    if REFRACTORY:
        gx *= 1.0 - tl.load(held_saved + out, mask=live, other=0).to(tl.float32)
    _accumulate(lv + idx, gx, live)
    if WANT_THETA:
        _accumulate(grad_theta + idx, -gx, live)
