"""Triton kernels every spiking family shares: the spike (and its surrogate gradient),
and the spike compaction.

A recurrent step starts with ``compact_spikes``, which lists, per trial and recurrent
pathway, the source neurons that fire now; the family's ``recurrent_step`` then gathers
only those weight rows (spikes are sparse, ~1% of neurons per ms).
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
def compact_spikes(
    v, theta, noise, rows, row_start, lists, counts,
    surr_ptr, t, B, T, N, L,
    BLOCK: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """List, in order, the rows of recurrent pathway ``k`` whose neuron fires now.

    One program per (trial, pathway). ``rows`` holds every recurrent pathway's source
    neurons back to back, ``row_start`` (K + 1) where each pathway's begin.
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
        position = count + tl.cumsum(fired.to(tl.int32), axis=0) - 1
        tl.store(lists + (k * B + b) * L + position, r, mask=fired)
        count += tl.sum(fired.to(tl.int32), axis=0)
    tl.store(counts + k * B + b, count)
