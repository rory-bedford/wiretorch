"""Triton kernels for the current-based LIF dynamics.

They implement exactly the time step of ``dynamics.py`` (the readable reference), fused
so that a neuron's state never leaves registers within a chunk:

* ``scan_forward`` / ``scan_backward`` integrate a chunk whose synaptic drive is known
  in advance (feedforward networks, and the backward pass of a recurrent network whose
  gradients do not flow through its spikes) — one launch per chunk, the time loop
  inside the kernel, BPTT as the matching reverse scan.
* ``compact_spikes`` (shared, ``network_simulators.kernels``) + ``recurrent_step``
  advance a recurrent network by one step, gathering only the weight rows of the
  neurons that fired.
* ``inject_backward`` + ``spike_backward`` (shared) step a recurrent network's adjoint
  back by one step when gradients flow through the recurrent spikes (the default),
  which couples the neurons: between the two, the engine carries each pathway's
  ``dL/da`` back to the presynaptic spikes with a matmul by ``W^T``.

Layouts (all contiguous float32 unless noted):

    v (B, N)    I (B, S, N) — the networks expose it as a (B, N, S) view
    refractory counters (B, N) int32, periods n_ref (N,) int32 (``REFRACTORY`` only)
    drive A (K, B, T, N)    spikes (B, T, N)
    saved for backward: v (B, T, N); recurrent activations A (K_rec, B, T, N); with
        ``REFRACTORY``, which neurons were refractory (B, T, N) int8
    per-pathway tables: scale (K, N), made (K, SP)
    per-synapse tables, padded with zeros to SP, a power of two >= S:
        sign (SP), decay (SP)

``I`` and the recurrent weights ``W`` may instead be float16 (the network's
``storage_dtype``): loads convert them to float32, stores round back.

A pathway with short-term plasticity scales each gathered weight row by its factor
(``GATED``: see ``network_simulators.kernels``), and with ``ADAPT`` a neuron's
adaptation current, stepped with the currents, is subtracted from its input.

The backward pass needs no saved currents: the dynamics are linear in them, so their
adjoint is carried back exactly from the voltage's. Nor does learnt physiology need
more saved (each of its terms is compiled in only when it is learnt):

* ``beta`` and ``R`` multiply the signed current ``c = sum_j sign_j I_j`` of a step
  without a spike (after one, the reset makes their gradient zero), and there
  ``beta v + (1 - beta) (U_rest + R c)`` is the next voltage: ``c`` follows from two
  consecutive saved voltages (the last step's next voltage is the final state).
* The decays: ``dL/d decay = sum_t gJ_t I_{t-1}`` (``gJ_t`` the adjoint of the currents
  after step ``t``'s input). The currents are linear in their inputs ``inj_u``, so this
  is ``sum_t inj_t H_t + I_init H_{-1}``, with ``H_{t-1} = gJ_t + decay H_t`` carried
  back alongside the adjoints, ``inj_t`` the injection, recomputed from the
  activations, and ``I_init`` the currents entering the chunk.

Physiology gradients are summed per (trial, neuron) — per (trial, synapse, neuron) for
the decays — and the engine sums over trials.

``K`` pathways (one per source cell type and input side), ``S`` synapse types. A padded
synapse column has zero sign, decay and kernel, so it stays at zero and adds nothing.
"""

import triton
import triton.language as tl

from wiretorch.network_simulators.kernels import (  # noqa: F401 - re-exported
    _accumulate,
    _refractory,
    _spike,
    _spike_adjoint,
    compact_spikes,
    spike_backward,
)


@triton.jit
def _inject(I, a, scale, made):
    """Add one pathway's activation ``a`` times its kernel (scale x made)."""
    return I + a[:, None] * (scale[:, None] * made[None, :])


@triton.jit
def _membrane(v, s, I, sign, U_reset, beta, U_rest, R, I_a, ADAPT: tl.constexpr):
    """The voltage after the step: leak, signed synaptic currents (less the
    adaptation current ``I_a`` with ``ADAPT``), reset."""
    I_syn = tl.sum(I * sign[None, :], axis=1)
    if ADAPT:
        I_syn -= I_a
    return (beta * v + (1 - beta) * (U_rest + R * I_syn)) * (1 - s) + U_reset * s


@triton.jit
def _membrane_adjoint(
    lv, s, v, v_next, beta, U_rest, R, acc_beta, acc_rest, acc_R, acc_reset,
    WANT_BETA: tl.constexpr, WANT_REST: tl.constexpr, WANT_R: tl.constexpr,
    WANT_RESET: tl.constexpr,
):  # fmt: skip
    """Add one step's terms to the gradients of ``_membrane``'s parameters.

    ``lv`` is dL/dv after the step, ``v`` the voltage entering it and ``v_next`` the
    one leaving it. Without a spike ``v_next = beta v + (1 - beta) (U_rest + R c)``,
    which gives the signed current ``c``; with one, only ``U_reset`` has a gradient.
    """
    lm = lv * (1 - s)
    if WANT_BETA:  # d/dbeta = v - U_rest - R c
        acc_beta += lm * (v - v_next) / (1 - beta)
    if WANT_REST:
        acc_rest += lm * (1 - beta)
    if WANT_R:  # d/dR = (1 - beta) c
        acc_R += lm * (v_next - beta * v - (1 - beta) * U_rest) / R
    if WANT_RESET:
        acc_reset += lv * s
    return acc_beta, acc_rest, acc_R, acc_reset


@triton.jit
def _injection(A, scale, made, stride, t_off, n, j, live, N, K: tl.constexpr,
               SP: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """The currents step ``t`` injects, ``sum_k a_k scale_k made_k`` ``(BLOCK, SP)``;
    ``A + t_off`` points at pathway 0's activations of the step."""
    inj = tl.zeros((BLOCK, SP), dtype=tl.float32)
    for k in tl.static_range(K):
        a = tl.load(A + k * stride + t_off, mask=live, other=0.0)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        inj = _inject(inj, a, sc, tl.load(made + k * SP + j))
    return inj


@triton.jit
def _track(
    track_v, track_I, b, n, t, T, N, j, live, v, I, track_b, S: tl.constexpr,
):  # fmt: skip
    """Write tracked variables for the tracked trial (or every trial if track_b < 0)."""
    tracked = live & ((track_b < 0) | (b == track_b))
    bt = tl.where(track_b < 0, b, 0)
    row = (bt * T + t) * N + n
    tl.store(track_v + row, v, mask=tracked)
    tl.store(
        track_I + row[:, None] * S + j[None, :],
        I,
        mask=tracked[:, None] & (j[None, :] < S),
    )


@triton.jit
def _track_adaptation(track_a, b, n, t, T, N, live, I_a, track_b):
    """Write the adaptation after the step, for the tracked trial(s)."""
    tracked = live & ((track_b < 0) | (b == track_b))
    bt = tl.where(track_b < 0, b, 0)
    tl.store(track_a + (bt * T + t) * N + n, I_a, mask=tracked)


@triton.jit
def scan_forward(
    A, scale, made, sign, decay, theta, U_reset, beta, U_rest, R,
    v_state, I_state, spikes, prob, noise, n_ref, counter, adapt, adapt_decay, a_state,
    v_saved, held_saved, track_v, track_I, track_a, surr_ptr, track_b, t0, TA, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    SAVE: tl.constexpr, TRACK: tl.constexpr, PROB: tl.constexpr, SAMPLE: tl.constexpr,
    REFRACTORY: tl.constexpr, ADAPT: tl.constexpr,
):  # fmt: skip
    """Integrate steps ``t0 .. t0 + TA`` of a ``T``-step chunk.

    ``A`` holds the pathway activations of just these ``TA`` steps, so a long chunk can
    be fed in blocks; outputs are indexed by the step within the whole chunk. With
    ``REFRACTORY``, ``counter`` ``(B, N)`` int32 holds the steps of refractory period
    left (``n_ref`` ``(N,)`` int32 the periods), and ``held_saved`` ``(B, T, N)`` int8
    gets which neurons were refractory, for the backward pass. With ``ADAPT``,
    ``a_state`` ``(B, N)`` holds the adaptation currents (updated in place), which rise
    by ``adapt`` at each spike and decay by ``adapt_decay`` ``(N,)``.
    """
    surr_scale = tl.load(surr_ptr)  # read on the device: no host sync to pass it
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, neuron)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    U_reset_n = tl.load(U_reset + n, mask=live)
    beta_n = tl.load(beta + n, mask=live)
    U_rest_n = tl.load(U_rest + n, mask=live)
    R_n = tl.load(R + n, mask=live)
    sg = tl.load(sign + j)
    dc = tl.load(decay + j)

    v = tl.load(v_state + idx, mask=live)
    I_row = I_state + (b * (S * N) + n)[:, None] + j[None, :] * N
    I = tl.load(I_row, mask=sm, other=0.0).to(tl.float32)
    if REFRACTORY:
        n_ref_n = tl.load(n_ref + n, mask=live, other=0)
        c = tl.load(counter + idx, mask=live, other=0)
    if ADAPT:
        I_a = tl.load(a_state + idx, mask=live, other=0.0)
        a_jump = tl.load(adapt + n, mask=live, other=0.0)
        a_decay = tl.load(adapt_decay + n, mask=live, other=0.0)
    else:
        I_a = 0.0

    for i in range(TA):
        t = t0 + i
        out = (b * T + t) * N + n
        if SAMPLE:
            u = tl.load(noise + out, mask=live, other=1.0)
        else:
            u = 0.0
        s = _spike(v, theta_n, u, surr_scale, SAMPLE)
        r = s  # what holds the voltage at reset
        if REFRACTORY:
            s, r, c, held = _refractory(s, c, n_ref_n)
        tl.store(spikes + out, s, mask=live)
        if PROB:
            p = tl.sigmoid(surr_scale * (v - theta_n))
            if REFRACTORY:
                p = p * (1 - held)
            tl.store(prob + out, p, mask=live)
        if SAVE:
            tl.store(v_saved + out, v, mask=live)
            if REFRACTORY:
                tl.store(held_saved + out, held.to(tl.int8), mask=live)

        I = I * dc[None, :]
        for k in tl.static_range(K):
            a = tl.load(
                A + k * (B * TA * N) + (b * TA + i) * N + n, mask=live, other=0.0
            )
            sc = tl.load(scale + k * N + n, mask=live, other=0.0)
            I = _inject(I, a, sc, tl.load(made + k * SP + j))
        if ADAPT:  # the neuron's own spike raises it
            I_a = I_a * a_decay + a_jump * s
        v = _membrane(v, r, I, sg, U_reset_n, beta_n, U_rest_n, R_n, I_a, ADAPT)
        if TRACK:
            _track(track_v, track_I, b, n, t, T, N, j, live, v, I, track_b, S)
            if ADAPT:
                _track_adaptation(track_a, b, n, t, T, N, live, I_a, track_b)

    tl.store(v_state + idx, v, mask=live)
    tl.store(I_row, I, mask=sm)
    if REFRACTORY:
        tl.store(counter + idx, c, mask=live)
    if ADAPT:
        tl.store(a_state + idx, I_a, mask=live)


@triton.jit
def scan_backward(
    grad_spikes, grad_prob, spikes, v_saved, v_final, I_init, held_saved, A, scale,
    made, sign, decay, theta, beta, U_rest, R, adapt, adapt_decay, a_init,
    grad_A, grad_scale, grad_theta, grad_reset, grad_beta, grad_rest, grad_R,
    grad_decay, grad_adapt, grad_adapt_decay, surr_ptr, B, T, N,
    K: tl.constexpr, KP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, WANT_A: tl.constexpr, WANT_SCALE: tl.constexpr,
    WANT_THETA: tl.constexpr, WANT_RESET: tl.constexpr, WANT_BETA: tl.constexpr,
    WANT_REST: tl.constexpr, WANT_R: tl.constexpr, WANT_DECAY: tl.constexpr,
    WANT_ADAPT: tl.constexpr, WANT_ADAPT_DECAY: tl.constexpr,
    PROB: tl.constexpr, SAMPLE: tl.constexpr, REFRACTORY: tl.constexpr,
    ADAPT: tl.constexpr,
):  # fmt: skip
    """Backpropagate through ``T`` steps: the adjoint of ``scan_forward``.

    Carries the adjoints of the voltage and of each current backwards in time. Writes
    ``grad_A`` (gradient w.r.t. the pathway activations) and/or the per-neuron partial
    sums of the gradients w.r.t. each pathway's ``scale`` and the physiology wanted.
    ``v_final`` (B, N) is the voltage after the last step and ``I_init`` (B, S, N) the
    currents before the first (read only for ``beta``/``R`` and the decays).
    ``held_saved`` (B, T, N) int8 marks the refractory steps (with ``REFRACTORY``):
    there the voltage is held at reset, as after a spike, and the spike at zero. With
    ``ADAPT`` the adaptation current's adjoint is carried back too, as a synapse
    type's of sign -1 driven by the neuron's own spikes; ``a_init`` (B, N) is the
    adaptation entering the chunk (read only for its decay).
    """
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    kk = tl.arange(0, KP)

    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    beta_n = tl.load(beta + n, mask=live)
    U_rest_n = tl.load(U_rest + n, mask=live)
    R_n = tl.load(R + n, mask=live)
    sg = tl.load(sign + j)
    dc = tl.load(decay + j)

    lv = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/dv entering the next step
    lI = tl.zeros((BLOCK, SP), dtype=tl.float32)  # dL/dI entering the next step
    acc = tl.zeros((BLOCK, KP), dtype=tl.float32)  # sum_t dL/dscale_k
    acc_theta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_reset = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_beta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_rest = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_R = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_decay = tl.zeros((BLOCK, SP), dtype=tl.float32)
    H = tl.zeros((BLOCK, SP), dtype=tl.float32)  # see the module docstring
    lA = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/dI_a entering the next step
    H_a = tl.zeros((BLOCK,), dtype=tl.float32)  # the adaptation's H
    acc_adapt = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_adapt_decay = tl.zeros((BLOCK,), dtype=tl.float32)
    if ADAPT:
        a_jump = tl.load(adapt + n, mask=live, other=0.0)
        a_decay = tl.load(adapt_decay + n, mask=live, other=0.0)
    if WANT_BETA or WANT_R:
        v_next = tl.load(v_final + idx, mask=live, other=0.0)
    else:
        v_next = lv

    for i in range(T):
        t = T - 1 - i
        out = (b * T + t) * N + n
        v = tl.load(v_saved + out, mask=live)
        s = tl.load(spikes + out, mask=live, other=0.0)
        if REFRACTORY:  # held at reset: s stands for the reset below
            held = tl.load(held_saved + out, mask=live, other=0).to(tl.float32)
            s = tl.maximum(s, held)

        # v_next = (beta v + (1 - beta) (U_rest + R sum_j sign_j J_j)) (1 - s) + U s,
        # with J the currents after the input, which are also the next step's.
        acc_beta, acc_rest, acc_R, acc_reset = _membrane_adjoint(
            lv, s, v, v_next, beta_n, U_rest_n, R_n,
            acc_beta, acc_rest, acc_R, acc_reset,
            WANT_BETA, WANT_REST, WANT_R, WANT_RESET,
        )  # fmt: skip
        v_next = v
        gJ = lI + (lv * (1 - s) * (1 - beta_n) * R_n)[:, None] * sg[None, :]
        if ADAPT:  # it rose by this step's spike (spikes, not the held reset)
            gA = lA - lv * (1 - s) * (1 - beta_n) * R_n
            spike = tl.load(spikes + out, mask=live, other=0.0)
            if WANT_ADAPT:
                acc_adapt += gA * spike
            if WANT_ADAPT_DECAY:
                acc_adapt_decay += a_jump * spike * H_a
                H_a = gA + a_decay * H_a
            lA = gA * a_decay
        if WANT_DECAY:
            inj = _injection(
                A, scale, made, B * T * N, out, n, j, live, N, K, SP, BLOCK
            )
            acc_decay += inj * H
            H = gJ + dc[None, :] * H
        for k in tl.static_range(K):
            q = tl.sum(gJ * tl.load(made + k * SP + j)[None, :], axis=1)
            if WANT_A:
                sc = tl.load(scale + k * N + n, mask=live, other=0.0)
                tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
                acc += tl.where(kk[None, :] == k, (q * a)[:, None], 0.0)
        lI = gJ * dc[None, :]
        lv = lv * (1 - s) * beta_n

        # Spike (surrogate) and probability outputs of step t, both of v - theta.
        x = v - theta_n
        gs = tl.load(grad_spikes + out, mask=live, other=0.0)
        gx = _spike_adjoint(x, gs, surr_scale, SAMPLE)
        if PROB:
            p = tl.sigmoid(surr_scale * x)
            gp = tl.load(grad_prob + out, mask=live, other=0.0)
            gx += gp * surr_scale * p * (1.0 - p)
        if REFRACTORY:  # a refractory neuron's spike and probability are held at 0
            gx *= 1 - held
        lv += gx
        acc_theta -= gx

    if WANT_SCALE:
        km = live[:, None] & (kk[None, :] < K)
        tl.store(grad_scale + kk[None, :] * (B * N) + idx[:, None], acc, mask=km)
    if WANT_THETA:
        tl.store(grad_theta + idx, acc_theta, mask=live)
    if WANT_RESET:
        tl.store(grad_reset + idx, acc_reset, mask=live)
    if WANT_BETA:
        tl.store(grad_beta + idx, acc_beta, mask=live)
    if WANT_REST:
        tl.store(grad_rest + idx, acc_rest, mask=live)
    if WANT_R:
        tl.store(grad_R + idx, acc_R, mask=live)
    if WANT_DECAY:
        rows = (b * (S * N) + n)[:, None] + j[None, :] * N
        acc_decay += tl.load(I_init + rows, mask=sm, other=0.0) * H
        tl.store(grad_decay + rows, acc_decay, mask=sm)
    if WANT_ADAPT:
        tl.store(grad_adapt + idx, acc_adapt, mask=live)
    if WANT_ADAPT_DECAY:  # the adaptation entering the chunk
        acc_adapt_decay += tl.load(a_init + idx, mask=live, other=0.0) * H_a
        tl.store(grad_adapt_decay + idx, acc_adapt_decay, mask=live)


@triton.jit
def recurrent_step(
    A_ff, W, w_start, lists, amps, counts, scale, made, sign, decay, theta, U_reset, beta,
    U_rest, R, v_state, I_state, spikes, noise, n_ref, counter, adapt, adapt_decay,
    a_state, v_saved, A_saved, held_saved, track_v, track_I, track_a, surr_ptr, track_b,
    t, t_ff, T, T_ff, B, N, L,
    K_REC: tl.constexpr, K_FF: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, ROWS: tl.constexpr, SAVE: tl.constexpr, TRACK: tl.constexpr,
    SAMPLE: tl.constexpr, REFRACTORY: tl.constexpr, GATED: tl.constexpr,
    ADAPT: tl.constexpr,
):  # fmt: skip
    """Advance a recurrent network one step, for one trial and a block of neurons.

    Recurrent activations are gathered from the weight rows listed by
    ``compact_spikes`` (pathway ``k``'s from row ``w_start[k]`` of ``W``; pathways of
    one source cell type share its rows), with ``GATED`` each times its factor in
    ``amps`` (short-term plasticity); feedforward ones are read from ``A_ff`` at
    ``t_ff``. The refractory period as in ``scan_forward``.
    """
    surr_scale = tl.load(surr_ptr)
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = n < N
    idx = b * N + n
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    v = tl.load(v_state + idx, mask=live)
    I_row = I_state + (b * (S * N) + n)[:, None] + j[None, :] * N
    I = tl.load(I_row, mask=sm, other=0.0).to(tl.float32)

    out = (b * T + t) * N + n
    if SAMPLE:
        u = tl.load(noise + out, mask=live, other=1.0)
    else:
        u = 0.0
    s = _spike(v, theta_n, u, surr_scale, SAMPLE)
    r = s  # what holds the voltage at reset
    if REFRACTORY:
        c = tl.load(counter + idx, mask=live, other=0)
        s, r, c, held = _refractory(s, c, tl.load(n_ref + n, mask=live, other=0))
        tl.store(counter + idx, c, mask=live)
    tl.store(spikes + out, s, mask=live)
    if SAVE:
        tl.store(v_saved + out, v, mask=live)
        if REFRACTORY:
            tl.store(held_saved + out, held.to(tl.int8), mask=live)

    I = I * tl.load(decay + j)[None, :]
    for k in tl.static_range(K_REC + K_FF):
        if k < K_REC:
            # Sum the weight rows of the neurons that fired, ROWS rows per load.
            a = tl.zeros((BLOCK,), dtype=tl.float32)
            first = tl.load(w_start + k)
            listed = lists + (k * B + b) * L
            fired = tl.load(counts + k * B + b)
            for m in range(0, fired, ROWS):
                m_ = m + tl.arange(0, ROWS)
                row = first + tl.load(listed + m_, mask=m_ < fired, other=0)
                w = tl.load(
                    W + row[:, None] * N + n[None, :],
                    mask=(m_ < fired)[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
                if GATED:
                    amp = tl.load(
                        amps + (k * B + b) * L + m_, mask=m_ < fired, other=0.0
                    )
                    w = w * amp[:, None]
                a += tl.sum(w, axis=0)
            if SAVE:
                tl.store(A_saved + k * (B * T * N) + out, a, mask=live)
        else:
            a = tl.load(
                A_ff + (k - K_REC) * (B * T_ff * N) + (b * T_ff + t_ff) * N + n,
                mask=live,
                other=0.0,
            )
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        I = _inject(I, a, sc, tl.load(made + k * SP + j))
    if ADAPT:  # the neuron's own spike raises it
        I_a = tl.load(a_state + idx, mask=live, other=0.0)
        I_a = I_a * tl.load(adapt_decay + n, mask=live, other=0.0)
        I_a += tl.load(adapt + n, mask=live, other=0.0) * s
        tl.store(a_state + idx, I_a, mask=live)
    else:
        I_a = 0.0
    v_next = _membrane(
        v, r, I, tl.load(sign + j),
        tl.load(U_reset + n, mask=live), tl.load(beta + n, mask=live),
        tl.load(U_rest + n, mask=live), tl.load(R + n, mask=live), I_a, ADAPT,
    )  # fmt: skip

    tl.store(v_state + idx, v_next, mask=live)
    tl.store(I_row, I, mask=sm)
    if TRACK:
        _track(track_v, track_I, b, n, t, T, N, j, live, v_next, I, track_b, S)
        if ADAPT:
            _track_adaptation(track_a, b, n, t, T, N, live, I_a, track_b)


@triton.jit
def inject_backward(
    lv, lI, H_state, lA_state, HA_state, spikes, v_saved, v_final, held_saved, A, scale,
    made, sign, decay, beta, U_rest, R, adapt, adapt_decay, grad_A, grad_scale,
    grad_reset, grad_beta, grad_rest, grad_R, grad_decay, grad_adapt, grad_adapt_decay,
    t, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    WANT_SCALE: tl.constexpr, WANT_RESET: tl.constexpr, WANT_BETA: tl.constexpr,
    WANT_REST: tl.constexpr, WANT_R: tl.constexpr, WANT_DECAY: tl.constexpr,
    WANT_ADAPT: tl.constexpr, WANT_ADAPT_DECAY: tl.constexpr,
    REFRACTORY: tl.constexpr, ADAPT: tl.constexpr,
):  # fmt: skip
    """Step the adjoint back through the injection and integration of step ``t``.

    ``lv`` ``(B, N)`` and ``lI`` ``(B, S, N)`` hold dL/dv and dL/dI after the step.
    Writes ``grad_A[:, :, t]`` (dL/da of every pathway), adds to ``grad_scale`` and the
    physiology gradients wanted, and leaves in ``lv``, ``lI`` the adjoints of the state
    entering the step — ``lv`` still without its spike term, which needs every
    neuron's dL/da (``spike_backward``). ``H_state`` ``(B, S, N)`` carries the decays'
    ``H`` (see the module docstring); the engine adds the initial currents' term.
    ``held_saved`` marks the refractory steps, as in ``scan_backward``; ``lA_state``
    and ``HA_state`` ``(B, N)`` carry the adaptation's adjoint and ``H``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    v_adj = tl.load(lv + idx, mask=live, other=0.0)
    I_row = lI + (b * (S * N) + n)[:, None] + j[None, :] * N
    I_adj = tl.load(I_row, mask=sm, other=0.0)
    out = (b * T + t) * N + n
    spike = tl.load(spikes + out, mask=live, other=0.0)
    s = spike
    if REFRACTORY:  # held at reset: s stands for the reset below
        s = tl.maximum(s, tl.load(held_saved + out, mask=live, other=0).to(tl.float32))
    beta_n = tl.load(beta + n, mask=live)
    R_n = tl.load(R + n, mask=live)

    # The adaptation, as in scan_backward.
    if ADAPT:
        a_decay = tl.load(adapt_decay + n, mask=live, other=0.0)
        gA = tl.load(lA_state + idx, mask=live, other=0.0)
        gA -= v_adj * (1 - s) * (1 - beta_n) * R_n
        if WANT_ADAPT:
            _accumulate(grad_adapt + idx, gA * spike, live)
        if WANT_ADAPT_DECAY:
            H_a = tl.load(HA_state + idx, mask=live, other=0.0)
            a_jump = tl.load(adapt + n, mask=live, other=0.0)
            _accumulate(grad_adapt_decay + idx, a_jump * spike * H_a, live)
            tl.store(HA_state + idx, gA + a_decay * H_a, mask=live)
        tl.store(lA_state + idx, gA * a_decay, mask=live)

    if WANT_RESET or WANT_BETA or WANT_REST or WANT_R:
        v = tl.load(v_saved + out, mask=live, other=0.0)
        if t + 1 < T:
            v_next = tl.load(v_saved + out + N, mask=live, other=0.0)
        else:
            v_next = tl.load(v_final + idx, mask=live, other=0.0)
        zero = tl.zeros((BLOCK,), dtype=tl.float32)
        g_beta, g_rest, g_R, g_reset = _membrane_adjoint(
            v_adj, s, v, v_next, beta_n, tl.load(U_rest + n, mask=live), R_n,
            zero, zero, zero, zero, WANT_BETA, WANT_REST, WANT_R, WANT_RESET,
        )  # fmt: skip
        if WANT_BETA:
            _accumulate(grad_beta + idx, g_beta, live)
        if WANT_REST:
            _accumulate(grad_rest + idx, g_rest, live)
        if WANT_R:
            _accumulate(grad_R + idx, g_R, live)
        if WANT_RESET:
            _accumulate(grad_reset + idx, g_reset, live)

    gJ = (
        I_adj
        + (v_adj * (1 - s) * (1 - beta_n) * R_n)[:, None] * tl.load(sign + j)[None, :]
    )
    for k in tl.static_range(K):
        q = tl.sum(gJ * tl.load(made + k * SP + j)[None, :], axis=1)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
        if WANT_SCALE:
            a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
            partial = grad_scale + k * (B * N) + idx
            tl.store(partial, tl.load(partial, mask=live) + q * a, mask=live)

    dc = tl.load(decay + j)
    if WANT_DECAY:
        H_row = H_state + (b * (S * N) + n)[:, None] + j[None, :] * N
        H = tl.load(H_row, mask=sm, other=0.0)
        inj = _injection(A, scale, made, B * T * N, out, n, j, live, N, K, SP, BLOCK)
        decay_row = grad_decay + (b * (S * N) + n)[:, None] + j[None, :] * N
        _accumulate(decay_row, inj * H, sm)
        tl.store(H_row, gJ + dc[None, :] * H, mask=sm)
    tl.store(I_row, gJ * dc[None, :], mask=sm)
    tl.store(lv + idx, v_adj * (1 - s) * beta_n, mask=live)
