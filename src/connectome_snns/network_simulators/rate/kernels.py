"""Triton kernels for the rate dynamics.

They implement exactly the time step of ``dynamics.py`` (the readable reference):

* ``scan_forward`` / ``scan_backward`` integrate a chunk whose synaptic drive is known
  in advance (feedforward networks) — one launch per chunk, the time loop inside the
  kernel, BPTT as the matching reverse scan.
* A recurrent step is ``rates_forward`` (the rates of the voltage entering the step),
  then one product per recurrent pathway (in the engine: rates are dense, so it is a
  matmul — sparse for a sparse connectome), then ``recurrent_step`` (the currents and
  the voltage). Backwards, ``inject_backward`` gives every pathway's ``dL/da``, a
  product with ``W^T`` carries it to the presynaptic rates, and ``output_backward``
  takes it through the nonlinearity.

The backward kernels also accumulate, per neuron, the gradients of the physiology
(``theta``, ``gain``, ``beta``, and per synapse type the decay) when it is learnt; the
engine sums them into the per-type parameters.

Layouts (all contiguous float32):

    v (B, N)    I (B, S, N) — the networks expose it as a (B, N, S) view
    feedforward drive A (K, B, T, N)    recurrent drive A_rec (T, K_rec, B, N)
    rates (B, T, N)
    saved for backward: v (B, T, N); if learning beta, y = sum_s sign_s I_s after the
        input (B, T, N); if learning a decay, I before the input (B, T, S, N)
    per-neuron tables: theta, gain, beta (N), code (N, int32: the nonlinearity)
    per-pathway tables: scale (K, N), made (K, SP)
    per-synapse tables, padded with zeros to SP, a power of two >= S: sign, decay (SP)

A padded synapse column has zero sign, decay and kernel, so it stays at zero.
"""

import triton
import triton.language as tl

#: The nonlinearity of each code (the order of ``CODES`` in the engine).
RELU, SOFTPLUS, SIGMOID, TANH = 0, 1, 2, 3


@triton.jit
def _phi(x, code):
    """``f(x)`` and ``f'(x)``, with each neuron's nonlinearity ``code``."""
    sig = tl.sigmoid(x)
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    th = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    f = tl.where(
        code == 0,
        tl.maximum(x, 0.0),
        tl.where(code == 1, softplus, tl.where(code == 2, sig, th)),
    )
    df = tl.where(
        code == 0,
        (x > 0).to(tl.float32),
        tl.where(code == 1, sig, tl.where(code == 2, sig * (1.0 - sig), 1.0 - th * th)),
    )
    return f, df


@triton.jit
def _integrate(I, v, target, sign, decay, beta):
    """Currents towards their input, then the voltage towards the signed currents.

    Returns ``(I, y, v)``: the currents after the input, ``y = sum_s sign_s I_s``, and
    the voltage after the step.
    """
    I = decay[None, :] * I + (1 - decay[None, :]) * target
    y = tl.sum(I * sign[None, :], axis=1)
    return I, y, beta * v + (1 - beta) * y


@triton.jit
def _track(track_v, track_I, b, n, t, T, N, j, live, v, I, track_b, S: tl.constexpr):
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
def _save_state(v_saved, y_saved, I_saved, b, t, T, N, n, j, live, sm, v, y, I_before,
                S: tl.constexpr, SAVE: tl.constexpr, SAVE_Y: tl.constexpr,
                SAVE_I: tl.constexpr):  # fmt: skip
    out = (b * T + t) * N + n
    if SAVE:
        tl.store(v_saved + out, v, mask=live)
    if SAVE_Y:
        tl.store(y_saved + out, y, mask=live)
    if SAVE_I:
        rows = ((b * T + t) * S)[:, None] + j[None, :]
        tl.store(I_saved + rows * N + n[:, None], I_before, mask=sm)


@triton.jit
def scan_forward(
    A, scale, made, sign, decay, theta, gain, beta, code,
    v_state, I_state, rates, v_saved, y_saved, I_saved, track_v, track_I,
    track_b, t0, TA, B, T, N,
    K: tl.constexpr, S: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr,
    SAVE: tl.constexpr, SAVE_Y: tl.constexpr, SAVE_I: tl.constexpr,
    TRACK: tl.constexpr,
):  # fmt: skip
    """Integrate steps ``t0 .. t0 + TA`` of a ``T``-step chunk.

    ``A`` holds the pathway activations of just these ``TA`` steps, so a long chunk can
    be fed in blocks; outputs are indexed by the step within the whole chunk.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)  # flat (trial, neuron)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    theta_n = tl.load(theta + n, mask=live)
    gain_n = tl.load(gain + n, mask=live)
    beta_n = tl.load(beta + n, mask=live)
    code_n = tl.load(code + n, mask=live)
    sg = tl.load(sign + j)
    dc = tl.load(decay + j)

    v = tl.load(v_state + idx, mask=live)
    I_row = I_state + (b * (S * N) + n)[:, None] + j[None, :] * N
    I = tl.load(I_row, mask=sm, other=0.0)

    for i in range(TA):
        t = t0 + i
        f, _ = _phi(v - theta_n, code_n)
        tl.store(rates + (b * T + t) * N + n, gain_n * f, mask=live)
        target = tl.zeros((BLOCK, SP), dtype=tl.float32)
        for k in tl.static_range(K):
            a = tl.load(
                A + k * (B * TA * N) + (b * TA + i) * N + n, mask=live, other=0.0
            )
            sc = tl.load(scale + k * N + n, mask=live, other=0.0)
            target += a[:, None] * (sc[:, None] * tl.load(made + k * SP + j)[None, :])
        I_before = I
        I, y, v_next = _integrate(I, v, target, sg, dc, beta_n)
        _save_state(
            v_saved, y_saved, I_saved, b, t, T, N, n, j, live, sm, v, y, I_before,
            S, SAVE, SAVE_Y, SAVE_I,
        )  # fmt: skip
        v = v_next
        if TRACK:
            _track(track_v, track_I, b, n, t, T, N, j, live, v, I, track_b, S)

    tl.store(v_state + idx, v, mask=live)
    tl.store(I_row, I, mask=sm)


@triton.jit
def rates_forward(
    v_state, theta, gain, code, rates, t, B, T, N, BLOCK: tl.constexpr,
):  # fmt: skip
    """The rates of the voltage entering step ``t``, into ``rates[:, t]``."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    v = tl.load(v_state + idx, mask=live)
    f, _ = _phi(v - tl.load(theta + n, mask=live), tl.load(code + n, mask=live))
    r = tl.load(gain + n, mask=live) * f
    tl.store(rates + (b * T + t) * N + n, r, mask=live)


@triton.jit
def recurrent_step(
    A_rec, A_ff, scale, made, sign, decay, beta,
    v_state, I_state, v_saved, y_saved, I_saved, track_v, track_I,
    track_b, t, t_ff, T, T_ff, B, N,
    K_REC: tl.constexpr, K_FF: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, SAVE: tl.constexpr, SAVE_Y: tl.constexpr,
    SAVE_I: tl.constexpr, TRACK: tl.constexpr,
):  # fmt: skip
    """The currents and voltage of step ``t``, from this step's activations.

    ``A_rec`` holds this step's recurrent activations ``(K_rec, B, N)``; feedforward
    ones are read from ``A_ff`` at ``t_ff``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)

    v = tl.load(v_state + idx, mask=live)
    I_row = I_state + (b * (S * N) + n)[:, None] + j[None, :] * N
    I = tl.load(I_row, mask=sm, other=0.0)
    target = tl.zeros((BLOCK, SP), dtype=tl.float32)
    for k in tl.static_range(K_REC + K_FF):
        if k < K_REC:
            a = tl.load(A_rec + k * (B * N) + idx, mask=live, other=0.0)
        else:
            a = tl.load(
                A_ff + (k - K_REC) * (B * T_ff * N) + (b * T_ff + t_ff) * N + n,
                mask=live,
                other=0.0,
            )
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        target += a[:, None] * (sc[:, None] * tl.load(made + k * SP + j)[None, :])
    I_before = I
    I, y, v_next = _integrate(
        I, v, target, tl.load(sign + j), tl.load(decay + j),
        tl.load(beta + n, mask=live),
    )  # fmt: skip
    _save_state(
        v_saved, y_saved, I_saved, b, t, T, N, n, j, live, sm, v, y, I_before,
        S, SAVE, SAVE_Y, SAVE_I,
    )  # fmt: skip
    tl.store(v_state + idx, v_next, mask=live)
    tl.store(I_row, I, mask=sm)
    if TRACK:
        _track(track_v, track_I, b, n, t, T, N, j, live, v_next, I, track_b, S)


# =====================================================================
# Backward
# =====================================================================


@triton.jit
def _inject_adjoint(lv, lI, v, y, I_before, beta_n, sg, dc, target):
    """Step the adjoints back through the currents' and voltage's update.

    From ``lv``, ``lI`` (dL/dv, dL/dI after the step), returns ``(lv, lI, gT, g_beta,
    g_decay)``: the adjoints before the step (``lv`` still without the rates' term),
    dL/d(target), and the step's contributions to dL/dbeta and dL/ddecay.
    """
    gJ = lI + (lv * (1 - beta_n))[:, None] * sg[None, :]
    gT = (1 - dc[None, :]) * gJ
    g_beta = lv * (v - y)
    g_decay = gJ * (I_before - target)
    return lv * beta_n, dc[None, :] * gJ, gT, g_beta, g_decay


@triton.jit
def _output_adjoint(lv, g, v, theta_n, gain_n, code_n):
    """Add the rates' term to dL/dv; returns ``(lv, g_theta, g_gain)``."""
    f, df = _phi(v - theta_n, code_n)
    through = gain_n * df * g
    return lv + through, -through, f * g


@triton.jit
def _target(A_at, scale, made, k_stride, n, j, live, B, N,
            K: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """``sum_k a_k scale_k made_k`` from activations at ``A_at + k * k_stride``."""
    target = tl.zeros((BLOCK, SP), dtype=tl.float32)
    for k in tl.static_range(K):
        a = tl.load(A_at + k * k_stride, mask=live, other=0.0)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        target += a[:, None] * (sc[:, None] * tl.load(made + k * SP + j)[None, :])
    return target


@triton.jit
def scan_backward(
    grad_rates, v_saved, y_saved, I_saved, A, scale, made, sign, decay,
    theta, gain, beta, code,
    grad_A, grad_scale, grad_theta, grad_gain, grad_beta, grad_decay, B, T, N,
    K: tl.constexpr, KP: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, WANT_A: tl.constexpr, WANT_SCALE: tl.constexpr,
    WANT_THETA: tl.constexpr, WANT_GAIN: tl.constexpr, WANT_BETA: tl.constexpr,
    WANT_DECAY: tl.constexpr,
):  # fmt: skip
    """Backpropagate through ``T`` steps: the adjoint of ``scan_forward``.

    Writes ``grad_A`` and, per (trial, neuron), the sums over steps of the gradients
    of ``scale`` (K, B, N), ``theta``, ``gain``, ``beta`` (B, N) and ``decay``
    (B, S, N), for those wanted.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)
    kk = tl.arange(0, KP)

    theta_n = tl.load(theta + n, mask=live)
    gain_n = tl.load(gain + n, mask=live)
    beta_n = tl.load(beta + n, mask=live)
    code_n = tl.load(code + n, mask=live)
    sg = tl.load(sign + j)
    dc = tl.load(decay + j)

    lv = tl.zeros((BLOCK,), dtype=tl.float32)
    lI = tl.zeros((BLOCK, SP), dtype=tl.float32)
    acc_scale = tl.zeros((BLOCK, KP), dtype=tl.float32)
    acc_theta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_gain = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_beta = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_decay = tl.zeros((BLOCK, SP), dtype=tl.float32)

    for i in range(T):
        t = T - 1 - i
        out = (b * T + t) * N + n
        v = tl.load(v_saved + out, mask=live, other=0.0)
        if WANT_BETA:
            y = tl.load(y_saved + out, mask=live, other=0.0)
        else:
            y = v
        if WANT_DECAY:
            rows = ((b * T + t) * S)[:, None] + j[None, :]
            I_before = tl.load(I_saved + rows * N + n[:, None], mask=sm, other=0.0)
            target = _target(
                A + out, scale, made, B * T * N, n, j, live, B, N, K, SP, BLOCK
            )
        else:
            I_before = tl.zeros((BLOCK, SP), dtype=tl.float32)
            target = I_before
        lv, lI, gT, g_beta, g_decay = _inject_adjoint(
            lv, lI, v, y, I_before, beta_n, sg, dc, target
        )
        for k in tl.static_range(K):
            q = tl.sum(gT * tl.load(made + k * SP + j)[None, :], axis=1)
            if WANT_A:
                sc = tl.load(scale + k * N + n, mask=live, other=0.0)
                tl.store(grad_A + k * (B * T * N) + out, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(A + k * (B * T * N) + out, mask=live, other=0.0)
                acc_scale += tl.where(kk[None, :] == k, (q * a)[:, None], 0.0)
        if WANT_BETA:
            acc_beta += g_beta
        if WANT_DECAY:
            acc_decay += g_decay
        g = tl.load(grad_rates + out, mask=live, other=0.0)
        lv, g_theta, g_gain = _output_adjoint(lv, g, v, theta_n, gain_n, code_n)
        acc_theta += g_theta
        acc_gain += g_gain

    if WANT_SCALE:
        km = live[:, None] & (kk[None, :] < K)
        tl.store(grad_scale + kk[None, :] * (B * N) + idx[:, None], acc_scale, mask=km)
    if WANT_THETA:
        tl.store(grad_theta + idx, acc_theta, mask=live)
    if WANT_GAIN:
        tl.store(grad_gain + idx, acc_gain, mask=live)
    if WANT_BETA:
        tl.store(grad_beta + idx, acc_beta, mask=live)
    if WANT_DECAY:
        tl.store(
            grad_decay + (b * (S * N) + n)[:, None] + j[None, :] * N, acc_decay, mask=sm
        )


@triton.jit
def inject_backward(
    lv, lI, v_saved, y_saved, I_saved, A_rec, A_ff, scale, made, sign, decay, beta,
    grad_rec, grad_ff, grad_scale, grad_beta, grad_decay, t, B, T, N,
    K_REC: tl.constexpr, K_FF: tl.constexpr, S: tl.constexpr, SP: tl.constexpr,
    BLOCK: tl.constexpr, WANT_FF: tl.constexpr, WANT_SCALE: tl.constexpr,
    WANT_BETA: tl.constexpr, WANT_DECAY: tl.constexpr,
):  # fmt: skip
    """Step the adjoint back through the currents and voltage of step ``t``.

    ``lv`` ``(B, N)`` and ``lI`` ``(B, S, N)`` hold dL/dv and dL/dI after the step;
    leaves in them the adjoints before it (``lv`` without the rates' term: see
    ``output_backward``). Writes dL/da of the recurrent pathways into ``grad_rec``
    ``(K_rec, B, N)`` (this step's slice) and of the feedforward ones into ``grad_ff``,
    and adds to the gradients of ``scale``, ``beta`` and ``decay``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    j = tl.arange(0, SP)
    sm = live[:, None] & (j[None, :] < S)
    out = (b * T + t) * N + n

    v_adj = tl.load(lv + idx, mask=live, other=0.0)
    I_row = lI + (b * (S * N) + n)[:, None] + j[None, :] * N
    I_adj = tl.load(I_row, mask=sm, other=0.0)
    beta_n = tl.load(beta + n, mask=live)
    v = tl.load(v_saved + out, mask=live, other=0.0)
    if WANT_BETA:
        y = tl.load(y_saved + out, mask=live, other=0.0)
    else:
        y = v
    if WANT_DECAY:
        rows = ((b * T + t) * S)[:, None] + j[None, :]
        I_before = tl.load(I_saved + rows * N + n[:, None], mask=sm, other=0.0)
        target = _target(
            A_rec + idx, scale, made, B * N, n, j, live, B, N, K_REC, SP, BLOCK
        )
        target += _target(
            A_ff + out, scale + K_REC * N, made + K_REC * SP, B * T * N, n, j, live,
            B, N, K_FF, SP, BLOCK,
        )  # fmt: skip
    else:
        I_before = tl.zeros((BLOCK, SP), dtype=tl.float32)
        target = I_before
    v_adj, I_adj, gT, g_beta, g_decay = _inject_adjoint(
        v_adj, I_adj, v, y, I_before, beta_n, tl.load(sign + j), tl.load(decay + j),
        target,
    )  # fmt: skip
    for k in tl.static_range(K_REC + K_FF):
        q = tl.sum(gT * tl.load(made + k * SP + j)[None, :], axis=1)
        sc = tl.load(scale + k * N + n, mask=live, other=0.0)
        if k < K_REC:
            tl.store(grad_rec + k * (B * N) + idx, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(A_rec + k * (B * N) + idx, mask=live, other=0.0)
        else:
            if WANT_FF:
                tl.store(grad_ff + (k - K_REC) * (B * T * N) + out, q * sc, mask=live)
            if WANT_SCALE:
                a = tl.load(
                    A_ff + (k - K_REC) * (B * T * N) + out, mask=live, other=0.0
                )
        if WANT_SCALE:
            partial = grad_scale + k * (B * N) + idx
            tl.store(partial, tl.load(partial, mask=live) + q * a, mask=live)
    if WANT_BETA:
        tl.store(
            grad_beta + idx, tl.load(grad_beta + idx, mask=live) + g_beta, mask=live
        )
    if WANT_DECAY:
        decay_row = grad_decay + (b * (S * N) + n)[:, None] + j[None, :] * N
        tl.store(decay_row, tl.load(decay_row, mask=sm, other=0.0) + g_decay, mask=sm)
    tl.store(I_row, I_adj, mask=sm)
    tl.store(lv + idx, v_adj, mask=live)


@triton.jit
def output_backward(
    lv, v_saved, grad_rates, R, theta, gain, code, grad_theta, grad_gain,
    t, B, T, N,
    BLOCK: tl.constexpr, WANT_THETA: tl.constexpr, WANT_GAIN: tl.constexpr,
):  # fmt: skip
    """Add step ``t``'s rates' term to dL/dv, through the nonlinearity.

    The rates' gradient is the loss's (``grad_rates``) plus ``R`` ``(B, N)``, what the
    recurrent activations of step ``t`` pass back to them.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < B * N
    b = idx // N
    n = idx % N
    out = (b * T + t) * N + n
    g = tl.load(grad_rates + out, mask=live, other=0.0)
    g += tl.load(R + idx, mask=live, other=0.0)
    lv_now, g_theta, g_gain = _output_adjoint(
        tl.load(lv + idx, mask=live, other=0.0),
        g,
        tl.load(v_saved + out, mask=live, other=0.0),
        tl.load(theta + n, mask=live),
        tl.load(gain + n, mask=live),
        tl.load(code + n, mask=live),
    )
    tl.store(lv + idx, lv_now, mask=live)
    if WANT_THETA:
        tl.store(
            grad_theta + idx, tl.load(grad_theta + idx, mask=live) + g_theta, mask=live
        )
    if WANT_GAIN:
        tl.store(
            grad_gain + idx, tl.load(grad_gain + idx, mask=live) + g_gain, mask=live
        )
