"""Triton kernels every spiking family shares: the spike (and its surrogate gradient),
the spike compaction, and the spike's step of a recurrent backward pass.

The refractory period (``_refractory``) is compiled in only when some neuron has one
(``REFRACTORY``): a refractory neuron cannot spike (nor be listed by the compaction)
and passes no gradient back through its spike.

A recurrent step starts with ``compact_spikes``, which lists, per trial and recurrent
pathway, the source neurons that fire now; the family's ``recurrent_step`` then gathers
only those weight rows (spikes are sparse, ~1% of neurons per ms).

A pathway with presynaptic dynamics (short-term plasticity, receptor saturation: see
``network.gate_step``) scales each spike by a factor from its source's state: with
``GATED``, ``compact_spikes`` steps the recurrent pathways' state and lists each fired
row's factor beside it, and ``gate_scan`` steps the feedforward ones through a chunk.

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
def _saturate(g, s, f, saturation, decay):
    """``network.gate_step``'s saturation: the factor of a release ``f`` (of a spike
    ``s``), and ``g`` after it."""
    g = g * decay
    closed = 1 - g
    f = tl.minimum(saturation * f * closed, closed)
    return f, tl.minimum(g + s * f, 1.0)


@triton.jit
def _stp(u, x, s, U, d_D, d_F):
    """``network.gate_step``'s short-term plasticity: the factor of a spike ``s``, and
    ``u``, ``x`` after it."""
    x = 1 - (1 - x) * d_D
    u = U + (u - U) * d_F
    r = u * x
    f = r / U
    x = x - s * r
    u = u + s * U * (1 - u)
    return f, u, x


@triton.jit
def gate_scan(
    x, rows, gate_params, gating, factors, B, T, n_in, R, RG, stp_col, sat_col,
    BLOCK: tl.constexpr, STP: tl.constexpr, SATURATES: tl.constexpr,
):  # fmt: skip
    """Step a feedforward pathway's presynaptic state through a chunk of input ``x``
    ``(B, T, n_in)``.

    Its ``RG`` source rows are the input rows ``rows`` ``(RG,)`` int32; ``gate_params``
    ``(5,)`` as ``Gate.params``. With ``STP`` its ``u`` and ``x`` are the columns
    ``stp_col ..`` and ``stp_col + RG ..`` of ``gating`` ``(B, R)``, if it
    ``SATURATES`` its receptors' gating ``sat_col ..`` (updated in place);
    ``factors`` ``(B, T, RG)`` gets each step's ``f``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, row)
    live = idx < B * RG
    b = idx // RG
    i = idx % RG
    row = tl.load(rows + i, mask=live, other=0)
    base = gating + b * R
    if STP:
        U, d_D, d_F = (
            tl.load(gate_params + 2),
            tl.load(gate_params + 3),
            tl.load(gate_params + 4),
        )
        su = tl.load(base + stp_col + i, mask=live, other=1.0)
        sx = tl.load(base + stp_col + RG + i, mask=live, other=1.0)
    if SATURATES:
        c, d = tl.load(gate_params), tl.load(gate_params + 1)
        g = tl.load(base + sat_col + i, mask=live, other=0.0)
    for t in range(T):
        s = tl.load(x + (b * T + t) * n_in + row, mask=live, other=0.0)
        f = tl.full((BLOCK,), 1.0, tl.float32)
        if STP:
            f, su, sx = _stp(su, sx, s, U, d_D, d_F)
        if SATURATES:
            f, g = _saturate(g, s, f, c, d)
        tl.store(factors + (b * T + t) * RG + i, f, mask=live)
    if STP:
        tl.store(base + stp_col + i, su, mask=live)
        tl.store(base + stp_col + RG + i, sx, mask=live)
    if SATURATES:
        tl.store(base + sat_col + i, g, mask=live)


@triton.jit
def compact_spikes(
    v, theta, noise, counter, rows, row_start, lists, counts,
    gate_params, gate_cols, gating, amps, f_saved,
    surr_ptr, t, B, T, N, L, R, R_F,
    BLOCK: tl.constexpr, SAMPLE: tl.constexpr, REFRACTORY: tl.constexpr,
    GATED: tl.constexpr, SAVE: tl.constexpr,
):  # fmt: skip
    """List, in order, the rows of recurrent pathway ``k`` whose neuron fires now.

    One program per (trial, pathway). ``rows`` holds every recurrent pathway's source
    neurons back to back, ``row_start`` (K + 1) where each pathway's begin. A neuron
    whose refractory ``counter`` ``(B, N)`` is not zero does not fire.

    With ``GATED`` (some pathway has presynaptic dynamics), ``amps`` ``(K, B, L)`` gets
    each listed row's factor beside it: 1, or for a pathway with presynaptic dynamics
    its ``f``, as its state in ``gating`` ``(B, R)`` is stepped (in place).
    ``gate_params`` ``(K, 5)`` are their ``Gate.params``, ``gate_cols`` ``(K, 3)``
    int32 their columns of ``gating`` (``u`` then ``x``; the receptors' gating) and of
    ``f_saved`` ``(B, T, R_F)``, -1 where they have none; with ``SAVE`` every row's
    ``f`` goes to ``f_saved``, for the backward pass.
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
            f = tl.full((BLOCK,), 1.0, tl.float32)
            f_col = tl.load(gate_cols + k * 3 + 2)
            if f_col >= 0:
                s = fired.to(tl.float32)
                state = gating + b * R
                stp_col = tl.load(gate_cols + k * 3)
                if stp_col >= 0:
                    su = tl.load(state + stp_col + r, mask=m, other=1.0)
                    sx = tl.load(state + stp_col + n_rows + r, mask=m, other=1.0)
                    f, su, sx = _stp(
                        su, sx, s, tl.load(gate_params + k * 5 + 2),
                        tl.load(gate_params + k * 5 + 3),
                        tl.load(gate_params + k * 5 + 4),
                    )  # fmt: skip
                    tl.store(state + stp_col + r, su, mask=m)
                    tl.store(state + stp_col + n_rows + r, sx, mask=m)
                sat_col = tl.load(gate_cols + k * 3 + 1)
                if sat_col >= 0:
                    g = tl.load(state + sat_col + r, mask=m, other=0.0)
                    f, g = _saturate(
                        g, s, f, tl.load(gate_params + k * 5),
                        tl.load(gate_params + k * 5 + 1),
                    )  # fmt: skip
                    tl.store(state + sat_col + r, g, mask=m)
                if SAVE:
                    tl.store(f_saved + (b * T + t) * R_F + f_col + r, f, mask=m)
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
