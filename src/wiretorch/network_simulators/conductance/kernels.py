"""Triton kernels for the conductance-based LIF dynamics.

They implement exactly the time step of ``dynamics.py`` (the readable reference), fused
so that a neuron's state never leaves registers within a chunk:

* ``scan_forward`` / ``scan_backward`` integrate a chunk whose synaptic drive is known
  in advance (feedforward networks, and the backward pass of a recurrent network whose
  gradients do not flow through its spikes) — one launch per chunk, the time loop
  inside the kernel, BPTT as the matching reverse scan.
* ``compact_spikes`` (shared, ``network_simulators.kernels``) + ``recurrent_step``
  advance a recurrent network by one step.
  Spikes are sparse (~1% of neurons per ms), so instead of a dense ``spikes @ W``
  each step gathers only the weight rows of the neurons that fired.
* ``inject_backward`` + ``spike_backward`` (shared) step a recurrent network's adjoint
  back by one step when gradients flow through the recurrent spikes, which couples the
  neurons: between the two, the engine carries each pathway's ``dL/da`` back to the
  presynaptic spikes with a matmul by ``W^T``. They share the per-step maths of
  ``scan_backward`` (``_injection``, ``_membrane_adjoint``).

Layouts (all contiguous float32 unless noted):

    v (B, N)    g (B, 2, S, N) — the networks expose it as a (B, N, 2, S) view
    drive A (K, B, T, N)    spikes (B, T, N)
    saved for backward: v (B, T, N), summed conductances G (B, T, N, S)
    per-pathway tables: scale (K, N), made (K, SP)
    per-synapse tables, padded with zeros to SP, the next power of two > S:
        jump (2, SP), decay (2, SP), E_syn (SP), g_clip (SP)

``g``, the saved ``G`` and the recurrent weights ``W`` may instead be float16 (the
network's ``storage_dtype``): loads convert them to float32, stores round back.

``K`` pathways (one per source cell type and input side), ``S`` synapse types. A padded
synapse column has zero jump, decay and clip, so it stays at zero and adds no current.

Learnt physiology gets its gradient from the backward kernels too, each term compiled in
only when it is learnt: per neuron ``theta``, ``U_reset``, ``E_L``, ``C_m`` and ``beta``,
per synapse type ``E_syn`` and per component the ``jump`` and ``decay``, summed per
(trial, [synapse,] neuron) — the engine sums over trials. Nothing more is saved: the
decays' gradient ``sum_t lambda_t g_t`` (``lambda_t`` the adjoint of a component after
step ``t``, ``g_t`` the component entering it) is, the conductances being linear in
their injections ``inj_u``, ``sum_t inj_t H_t + g_init H_{-1}`` with
``H_{t-1} = lambda_t + decay H_t`` carried back alongside the adjoints, and only the
conductances entering the chunk (``g_init``) kept.
"""

import triton
import triton.language as tl

from wiretorch.network_simulators.kernels import (  # noqa: F401 - re-exported
    _accumulate,
    _spike,
    _spike_adjoint,
    compact_spikes,
    spike_backward,
)


@triton.jit
def _membrane(v, s, gr, gd, U_reset, E_L, C_m, beta, E_syn, g_clip, dt):
    """Synaptic and leak currents, and the voltage after the step (with reset)."""
    G = gr + gd
    Gc = tl.minimum(tl.maximum(G, 0.0), g_clip[None, :])
    I = Gc * (v[:, None] - E_syn[None, :])
    I_leak = (v - E_L) * (1 - beta) * C_m / dt
    v_next = (v - (tl.sum(I, axis=1) + I_leak) * dt / C_m) * (1 - s) + U_reset * s
    return G, I, I_leak, v_next


@triton.jit
def _track(
    track_v, track_I, track_g, b, n, t, T, N, j, live,
    v_next, I, I_leak, gr, gd, track_b,
    S: tl.constexpr,
):  # fmt: skip
    """Write tracked variables for the tracked trial (or every trial if track_b < 0)."""
    tracked = live & ((track_b < 0) | (b == track_b))
    bt = tl.where(track_b < 0, b, 0)
    row = (bt * T + t) * N + n
    tl.store(track_v + row, v_next, mask=tracked)
    # The leak current goes in column S, which the padding guarantees is free.
    I_all = tl.where(j[None, :] == S, I_leak[:, None], I)
    tl.store(
        track_I + row[:, None] * (S + 1) + j[None, :],
        I_all,
        mask=tracked[:, None] & (j[None, :] <= S),
    )
    jm = tracked[:, None] & (j[None, :] < S)
    tl.store(track_g + row[:, None] * (2 * S) + j[None, :], gr, mask=jm)
    tl.store(track_g + row[:, None] * (2 * S) + S + j[None, :], gd, mask=jm)


@triton.jit
def scan_forward(
    A, scale, made, jump, decay, E_syn, g_clip,
    theta, U_reset, E_L, C_m, beta,
    v_state, g_state, spikes, prob, noise,
    v_saved, G_saved, track_v, track_I, track_g,
    dt_ptr, surr_ptr, track_b, t0, TA, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    SAVE: tl.constexpr, TRACK: tl.constexpr, PROB: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """Integrate steps ``t0 .. t0 + TA`` of a ``T``-step chunk.

    ``A`` holds the pathway activations of just these ``TA`` steps, so a long chunk can
    be fed in blocks; outputs are indexed by the step within the whole chunk.
    """
    dt = tl.load(dt_ptr)  # scalars read on the device: no host sync to pass them
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, neuron)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    U_reset_n = tl.load(U_reset + n, mask=live)
    E_L_n = tl.load(E_L + n, mask=live)
    C_m_n = tl.load(C_m + n, mask=live, other=1.0)
    beta_n = tl.load(beta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    decay_r, decay_d = tl.load(decay + j), tl.load(decay + SP + j)

    v = tl.load(v_state + idx, mask=live)
    g_row = g_state + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    gr = tl.load(g_row, mask=sm, other=0.0).to(tl.float32)
    gd = tl.load(g_row + S * N, mask=sm, other=0.0).to(tl.float32)

    for i in range(TA):
        t = t0 + i
        out = (b * T + t) * N + n
        if SAMPLE:
            u = tl.load(noise + out, mask=live, other=1.0)
        else:
            u = 0.0
        s = _spike(v, theta_n, u, surr_scale, SAMPLE)
        tl.store(spikes + out, s, mask=live)
        if PROB:
            tl.store(prob + out, tl.sigmoid(surr_scale * (v - theta_n)), mask=live)
        G, I, I_leak, v_next = _membrane(
            v, s, gr, gd, U_reset_n, E_L_n, C_m_n, beta_n, E, clip, dt
        )
        if SAVE:
            tl.store(v_saved + out, v, mask=live)
            tl.store(G_saved + out[:, None] * S + j[None, :], G, mask=sm)

        gr = gr * decay_r[None, :]
        gd = gd * decay_d[None, :]
        for k in tl.static_range(K):
            a = tl.load(
                A + k * (B * TA * N) + (b * TA + i) * N + n, mask=live, other=0.0
            )
            sc = tl.load(scale + k * N + n, mask=live, other=0.0)
            mk = tl.load(made + k * SP + j)
            gr += a[:, None] * (jump_r[None, :] * sc[:, None] * mk[None, :])
            gd += a[:, None] * (jump_d[None, :] * sc[:, None] * mk[None, :])
        v = v_next
        if TRACK:
            _track(
                track_v, track_I, track_g, b, n, t, T, N, j, live,
                v, I, I_leak, gr, gd, track_b, S,
            )  # fmt: skip

    tl.store(v_state + idx, v, mask=live)
    tl.store(g_row, gr, mask=sm)
    tl.store(g_row + S * N, gd, mask=sm)


@triton.jit
def _store_pair(pointer, rise, decay, b, n, j, sm, S: tl.constexpr, N):
    """Store a ``(BLOCK, SP)`` pair of per-component sums into ``(B, 2, S, N)``."""
    row = pointer + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    tl.store(row, rise, mask=sm)
    tl.store(row + S * N, decay, mask=sm)


@triton.jit
def _injection(A, scale, made, stride, out, n, j, live, N, K: tl.constexpr,
               SP: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """What step ``t`` injects per unit jump, ``sum_k a_k scale_k made_k``
    ``(BLOCK, SP)``; ``A + out`` points at pathway 0's activations of the step."""
    base = tl.zeros((BLOCK, SP), dtype=tl.float32)
    for k in tl.static_range(K):
        a = tl.load(A + k * stride + out, mask=live, other=0.0)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        mk = tl.load(made + k * SP + j)
        base += (a * sc)[:, None] * mk[None, :]
    return base


@triton.jit
def _membrane_adjoint(
    dv, s, v, G, E, clip, E_L, C_m, beta, dt,
    acc_reset, acc_E_L, acc_beta, acc_C_m, acc_E,
    WANT_RESET: tl.constexpr, WANT_E_L: tl.constexpr, WANT_BETA: tl.constexpr,
    WANT_C_M: tl.constexpr, WANT_E_SYN: tl.constexpr,
):  # fmt: skip
    """Step ``dv`` (dL/dv after the step) back through the membrane update
    ``v_next = (v - (sum_j I_j + I_leak) dt / C) (1 - s) + U s``.

    Returns dL/dv entering the step (without its spike term), dL/dG of the summed
    conductances it read, and the accumulators with this step's terms added.
    """
    Gc = tl.minimum(tl.maximum(G, 0.0), clip[None, :])
    gx = dv * (1 - s)
    gy = -gx * dt / C_m
    clamp = tl.where(G < clip[None, :], 1.0, tl.where(G == clip[None, :], 0.5, 0.0))
    clamp = tl.where(G >= 0.0, clamp, 0.0)
    dG = gy[:, None] * (v[:, None] - E[None, :]) * clamp
    if WANT_RESET:
        acc_reset += dv * s
    if WANT_E_L:
        acc_E_L += gx * (1 - beta)
    if WANT_BETA:
        acc_beta += gx * (v - E_L)
    if WANT_C_M:  # d/dC_m of -(sum_j I_j) dt / C_m
        I_syn = tl.sum(Gc * (v[:, None] - E[None, :]), axis=1)
        acc_C_m -= gy * I_syn / C_m
    if WANT_E_SYN:
        acc_E -= gy[:, None] * Gc
    dv = gx + gy * (tl.sum(Gc, axis=1) + (1 - beta) * C_m / dt)
    return dv, dG, acc_reset, acc_E_L, acc_beta, acc_C_m, acc_E


@triton.jit
def scan_backward(
    grad_spikes, grad_prob, spikes, v_saved, G_saved, g_init, A, scale, made, jump,
    decay, E_syn, g_clip, theta, E_L, C_m, beta, grad_A, grad_scale,
    grad_theta, grad_reset, grad_E_L, grad_C_m, grad_beta, grad_E_syn, grad_decay,
    grad_jump, dt_ptr, surr_ptr, B, T, N,
    K: tl.constexpr, KP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, WANT_A: tl.constexpr, WANT_SCALE: tl.constexpr,
    WANT_THETA: tl.constexpr, WANT_RESET: tl.constexpr, WANT_E_L: tl.constexpr,
    WANT_C_M: tl.constexpr, WANT_BETA: tl.constexpr, WANT_E_SYN: tl.constexpr,
    WANT_DECAY: tl.constexpr, WANT_JUMP: tl.constexpr,
    PROB: tl.constexpr, SAMPLE: tl.constexpr,
):  # fmt: skip
    """Backpropagate through ``T`` steps: the adjoint of ``scan_forward``.

    Carries the adjoints of the voltage and of each conductance component backwards in
    time. Writes ``grad_A`` (gradient w.r.t. the pathway activations) and/or the
    per-neuron partial sums of the gradients w.r.t. each pathway's ``scale`` and the
    physiology wanted. ``g_init`` (B, 2, S, N) holds the conductances entering the
    chunk (read only for the decays).
    """
    dt = tl.load(dt_ptr)
    surr_scale = tl.load(surr_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)
    kk = tl.arange(0, KP)

    theta_n = tl.load(theta + n, mask=live)
    E_L_n = tl.load(E_L + n, mask=live)
    C_m_n = tl.load(C_m + n, mask=live, other=1.0)
    beta_n = tl.load(beta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    decay_r, decay_d = tl.load(decay + j), tl.load(decay + SP + j)

    dv = tl.zeros((BLOCK,), dtype=tl.float32)  # dL/dv entering the next step
    lr = tl.zeros((BLOCK, SP), dtype=tl.float32)  # dL/dg (rise) entering the next step
    ld = tl.zeros((BLOCK, SP), dtype=tl.float32)  # dL/dg (decay)
    acc = tl.zeros((BLOCK, KP), dtype=tl.float32)  # sum_t dL/dscale_k
    acc_theta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_reset = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_E_L = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_C_m = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_beta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK, SP), dtype=tl.float32)
    acc_decay_r = tl.zeros((BLOCK, SP), dtype=tl.float32)
    acc_decay_d = tl.zeros((BLOCK, SP), dtype=tl.float32)
    acc_jump_r = tl.zeros((BLOCK, SP), dtype=tl.float32)
    acc_jump_d = tl.zeros((BLOCK, SP), dtype=tl.float32)
    H_r = tl.zeros((BLOCK, SP), dtype=tl.float32)  # see the module docstring
    H_d = tl.zeros((BLOCK, SP), dtype=tl.float32)

    for i in range(T):
        t = T - 1 - i
        out = (b * T + t) * N + n
        v = tl.load(v_saved + out, mask=live)
        s = tl.load(spikes + out, mask=live, other=0.0)
        G = tl.load(G_saved + out[:, None] * S + j[None, :], mask=sm, other=0.0)
        G = G.to(tl.float32)

        # The jumps and decays: the injection of step t per unit jump, and the
        # components' adjoints after it (lr, ld).
        if WANT_DECAY or WANT_JUMP:
            base = _injection(
                A, scale, made, B * T * N, out, n, j, live, N, K, SP, BLOCK
            )
            if WANT_JUMP:
                acc_jump_r += lr * base
                acc_jump_d += ld * base
            if WANT_DECAY:
                acc_decay_r += jump_r[None, :] * base * H_r
                acc_decay_d += jump_d[None, :] * base * H_d
                H_r = lr + decay_r[None, :] * H_r
                H_d = ld + decay_d[None, :] * H_d

        # The injections of step t enter the state that step t + 1 sees.
        for k in tl.static_range(K):
            mk = tl.load(made + k * SP + j)
            q = tl.sum(
                (jump_r[None, :] * lr + jump_d[None, :] * ld) * mk[None, :], axis=1
            )
            if WANT_A:
                sc = tl.load(scale + k * N + n, mask=live, other=0.0)
                tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
                acc += tl.where(kk[None, :] == k, (q * a)[:, None], 0.0)

        dv, dG, acc_reset, acc_E_L, acc_beta, acc_C_m, acc_E = _membrane_adjoint(
            dv, s, v, G, E, clip, E_L_n, C_m_n, beta_n, dt,
            acc_reset, acc_E_L, acc_beta, acc_C_m, acc_E,
            WANT_RESET, WANT_E_L, WANT_BETA, WANT_C_M, WANT_E_SYN,
        )  # fmt: skip

        # Spike (surrogate) and probability outputs of step t, both of v - theta.
        x = v - theta_n
        gs = tl.load(grad_spikes + out, mask=live, other=0.0)
        g_x = _spike_adjoint(x, gs, surr_scale, SAMPLE)
        if PROB:
            p = tl.sigmoid(surr_scale * x)
            g_x += (
                tl.load(grad_prob + out, mask=live, other=0.0)
                * surr_scale
                * p
                * (1.0 - p)
            )
        dv += g_x
        acc_theta -= g_x

        # Conductances decay into the next step and feed the currents of this one.
        lr = lr * decay_r[None, :] + dG
        ld = ld * decay_d[None, :] + dG

    if WANT_SCALE:
        km = live[:, None] & (kk[None, :] < K)
        tl.store(grad_scale + kk[None, :] * (B * N) + idx[:, None], acc, mask=km)
    if WANT_THETA:
        tl.store(grad_theta + idx, acc_theta, mask=live)
    if WANT_RESET:
        tl.store(grad_reset + idx, acc_reset, mask=live)
    if WANT_E_L:
        tl.store(grad_E_L + idx, acc_E_L, mask=live)
    if WANT_C_M:
        tl.store(grad_C_m + idx, acc_C_m, mask=live)
    if WANT_BETA:
        tl.store(grad_beta + idx, acc_beta, mask=live)
    if WANT_E_SYN:
        row = grad_E_syn + (b * (S * N) + n)[:, None] + j[None, :] * N
        tl.store(row, acc_E, mask=sm)
    if WANT_JUMP:
        _store_pair(grad_jump, acc_jump_r, acc_jump_d, b, n, j, sm, S, N)
    if WANT_DECAY:  # the conductances entering the chunk
        g_row = g_init + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
        acc_decay_r += tl.load(g_row, mask=sm, other=0.0) * H_r
        acc_decay_d += tl.load(g_row + S * N, mask=sm, other=0.0) * H_d
        _store_pair(grad_decay, acc_decay_r, acc_decay_d, b, n, j, sm, S, N)


@triton.jit
def recurrent_step(
    A_ff, W, w_start, lists, counts, scale, made, jump, decay, E_syn, g_clip,
    theta, U_reset, E_L, C_m, beta,
    v_state, g_state, spikes, noise,
    v_saved, G_saved, A_saved, track_v, track_I, track_g,
    dt_ptr, surr_ptr, track_b, t, t_ff, T, T_ff, B, N, L,
    K_REC: tl.constexpr, K_FF: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, ROWS: tl.constexpr, SAVE: tl.constexpr, TRACK: tl.constexpr,
    SAMPLE: tl.constexpr,
):  # fmt: skip
    """Advance a recurrent network one step, for one trial and a block of neurons.

    Recurrent activations are gathered from the weight rows listed by
    ``compact_spikes``; feedforward ones are read from ``A_ff`` at ``t_ff``.
    """
    dt = tl.load(dt_ptr)
    surr_scale = tl.load(surr_ptr)
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = n < N
    idx = b * N + n
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)

    v = tl.load(v_state + idx, mask=live)
    g_row = g_state + (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    gr = tl.load(g_row, mask=sm, other=0.0).to(tl.float32)
    gd = tl.load(g_row + S * N, mask=sm, other=0.0).to(tl.float32)

    out = (b * T + t) * N + n
    if SAMPLE:
        u = tl.load(noise + out, mask=live, other=1.0)
    else:
        u = 0.0
    s = _spike(v, theta_n, u, surr_scale, SAMPLE)
    tl.store(spikes + out, s, mask=live)
    G, I, I_leak, v_next = _membrane(
        v, s,
        gr, gd,
        tl.load(U_reset + n, mask=live),
        tl.load(E_L + n, mask=live),
        tl.load(C_m + n, mask=live, other=1.0),
        tl.load(beta + n, mask=live),
        E, clip, dt,
    )  # fmt: skip
    if SAVE:
        tl.store(v_saved + out, v, mask=live)
        tl.store(G_saved + out[:, None] * S + j[None, :], G, mask=sm)

    gr = gr * tl.load(decay + j)[None, :]
    gd = gd * tl.load(decay + SP + j)[None, :]
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
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
                )
                a += tl.sum(w.to(tl.float32), axis=0)
            if SAVE:
                tl.store(A_saved + k * (B * T * N) + out, a, mask=live)
        else:
            a = tl.load(
                A_ff + (k - K_REC) * (B * T_ff * N) + (b * T_ff + t_ff) * N + n,
                mask=live,
                other=0.0,
            )
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        mk = tl.load(made + k * SP + j)
        gr += a[:, None] * (jump_r[None, :] * sc[:, None] * mk[None, :])
        gd += a[:, None] * (jump_d[None, :] * sc[:, None] * mk[None, :])

    tl.store(v_state + idx, v_next, mask=live)
    tl.store(g_row, gr, mask=sm)
    tl.store(g_row + S * N, gd, mask=sm)
    if TRACK:
        _track(
            track_v, track_I, track_g, b, n, t, T, N, j, live,
            v_next, I, I_leak, gr, gd, track_b, S,
        )  # fmt: skip


@triton.jit
def inject_backward(
    lv, lg, H_state, spikes, v_saved, G_saved, A, scale, made, jump, decay, E_syn,
    g_clip, E_L, C_m, beta, grad_A, grad_scale, grad_reset, grad_E_L, grad_C_m,
    grad_beta, grad_E_syn, grad_decay, grad_jump, dt_ptr, t, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    WANT_SCALE: tl.constexpr, WANT_RESET: tl.constexpr, WANT_E_L: tl.constexpr,
    WANT_C_M: tl.constexpr, WANT_BETA: tl.constexpr, WANT_E_SYN: tl.constexpr,
    WANT_DECAY: tl.constexpr, WANT_JUMP: tl.constexpr,
):  # fmt: skip
    """Step the adjoint back through the integration and injection of step ``t``.

    ``lv`` ``(B, N)`` and ``lg`` ``(B, 2, S, N)`` hold dL/dv and dL/dg after the step.
    Writes ``grad_A[:, :, t]`` (dL/da of every pathway), adds to ``grad_scale`` and the
    physiology gradients wanted, and leaves in ``lv``, ``lg`` the adjoints of the state
    entering the step — ``lv`` still without its spike term, which needs every
    neuron's dL/da (``spike_backward``). ``H_state`` ``(B, 2, S, N)`` carries the
    decays' ``H`` (see the module docstring); the engine adds the initial
    conductances' term. One step of ``scan_backward``, but for the spike term.
    """
    dt = tl.load(dt_ptr)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    E = tl.load(E_syn + j)
    clip = tl.load(g_clip + j)
    jump_r, jump_d = tl.load(jump + j), tl.load(jump + SP + j)
    decay_r, decay_d = tl.load(decay + j), tl.load(decay + SP + j)

    dv = tl.load(lv + idx, mask=live, other=0.0)
    rows = (b * (2 * S * N) + n)[:, None] + j[None, :] * N
    lr = tl.load(lg + rows, mask=sm, other=0.0)
    ld = tl.load(lg + rows + S * N, mask=sm, other=0.0)
    out = (b * T + t) * N + n
    v = tl.load(v_saved + out, mask=live, other=0.0)
    s = tl.load(spikes + out, mask=live, other=0.0)
    G = tl.load(G_saved + out[:, None] * S + j[None, :], mask=sm, other=0.0)
    G = G.to(tl.float32)

    # The jumps and decays, as in scan_backward.
    if WANT_DECAY or WANT_JUMP:
        base = _injection(A, scale, made, B * T * N, out, n, j, live, N, K, SP, BLOCK)
        if WANT_JUMP:
            _accumulate(grad_jump + rows, lr * base, sm)
            _accumulate(grad_jump + rows + S * N, ld * base, sm)
        if WANT_DECAY:
            H_r = tl.load(H_state + rows, mask=sm, other=0.0)
            H_d = tl.load(H_state + rows + S * N, mask=sm, other=0.0)
            _accumulate(grad_decay + rows, jump_r[None, :] * base * H_r, sm)
            _accumulate(grad_decay + rows + S * N, jump_d[None, :] * base * H_d, sm)
            tl.store(H_state + rows, lr + decay_r[None, :] * H_r, mask=sm)
            tl.store(H_state + rows + S * N, ld + decay_d[None, :] * H_d, mask=sm)

    # The injections of step t enter the state that step t + 1 sees.
    for k in tl.static_range(K):
        mk = tl.load(made + k * SP + j)
        q = tl.sum((jump_r[None, :] * lr + jump_d[None, :] * ld) * mk[None, :], axis=1)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
        if WANT_SCALE:
            a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
            _accumulate(grad_scale + k * (B * N) + idx, q * a, live)

    zero = tl.zeros((BLOCK,), dtype=tl.float32)
    dv, dG, g_reset, g_E_L, g_beta, g_C_m, g_E = _membrane_adjoint(
        dv, s, v, G, E, clip,
        tl.load(E_L + n, mask=live), tl.load(C_m + n, mask=live, other=1.0),
        tl.load(beta + n, mask=live), dt,
        zero, zero, zero, zero, tl.zeros((BLOCK, SP), dtype=tl.float32),
        WANT_RESET, WANT_E_L, WANT_BETA, WANT_C_M, WANT_E_SYN,
    )  # fmt: skip
    if WANT_RESET:
        _accumulate(grad_reset + idx, g_reset, live)
    if WANT_E_L:
        _accumulate(grad_E_L + idx, g_E_L, live)
    if WANT_BETA:
        _accumulate(grad_beta + idx, g_beta, live)
    if WANT_C_M:
        _accumulate(grad_C_m + idx, g_C_m, live)
    if WANT_E_SYN:
        _accumulate(grad_E_syn + (b * (S * N) + n)[:, None] + j[None, :] * N, g_E, sm)

    # Conductances decay into the next step and feed the currents of this one.
    tl.store(lv + idx, dv, mask=live)
    tl.store(lg + rows, lr * decay_r[None, :] + dG, mask=sm)
    tl.store(lg + rows + S * N, ld * decay_d[None, :] + dG, mask=sm)
