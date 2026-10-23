"""The conductance-based LIF time step, in plain PyTorch.

This is the reference implementation — the CPU path, the path used when per-step
gradients are tracked, and the specification the Triton kernels in ``kernels.py``
follow line for line. One step, for every neuron of every trial:

1. **Spike** if the voltage is above threshold (Heaviside with a surrogate gradient, or
   Bernoulli escape noise in ``"sample"`` mode) and the neuron is not refractory.
2. **Currents** from the conductances: each synapse type's rise + decay components,
   clipped to ``[0, g_clip]``, times the driving force ``v - E_syn``; plus the leak.
   An NMDA-like synapse type scales its current by its voltage-dependent Mg block.
3. **Integrate** the voltage with exponential Euler — exact for the conductances held
   over the step, so stable however large they are — and reset the neurons that
   spiked; a
   neuron stays at reset for the ``n_ref`` steps after its spike (its refractory
   period), and a counter per neuron (``refractory``, part of the state) tracks them.
4. **Decay** each conductance component, then **inject** this step's input: every
   pathway adds its activation (presynaptic spikes times weights) times its kernel.
   A saturating synapse type's pathway scales each spike by the fraction of its
   source's closed receptors that it opens, from the source's gating variable
   (``network.gate_step``, part of the state ``gating``), so that its conductance
   follows ``g_bar sum_j w_j u_j`` (exactly, as ``tau_rise -> 0``).

Gradients flow through the voltage and conductances (backprop through time), into the
spikes only via the surrogate, and never through the reset. They flow through the
recurrent spikes, into the presynaptic neurons, only if ``drive.recurrent_gradients``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wiretorch.network_simulators.network import (  # noqa: F401
    Drive,
    Run,
    count_down,
    gate_feedforward,
    gate_step,
    held_at_reset,
    refractory_mask,
    stack_steps,
)
from wiretorch.training_utils.surrogate_gradients import (
    SampleSpike,
    SurrGradSpike,
)


@dataclass
class Physiology:
    """Every parameter of the dynamics, as tensors on the simulation device.

    Per neuron ``(N,)``: ``theta``, ``U_reset``, ``E_L``, ``C_m``, ``beta`` (the leak's
    per-step decay). Per synapse type ``(S,)``: ``E_syn``, ``g_clip``; ``decay`` and
    ``jump`` are ``(2, S)`` — per-step decay and per-unit-input jump of the rise and
    decay components.
    ``n_ref`` ``(N,)`` int32 is each neuron's refractory period in steps, or None if no
    neuron has one.
    ``mg_c``, ``mg_a`` ``(S,)`` are each synapse type's Mg block (see
    ``synapses.mg_block_tables``), or None if no synapse type has one.
    """

    theta: torch.Tensor
    U_reset: torch.Tensor
    E_L: torch.Tensor
    C_m: torch.Tensor
    beta: torch.Tensor
    E_syn: torch.Tensor
    g_clip: torch.Tensor
    decay: torch.Tensor
    jump: torch.Tensor
    dt: torch.Tensor
    surr_scale: torch.Tensor
    spike_mode: str = "deterministic"
    #: What the CUDA kernels store conductances and recurrent weights in (they compute
    #: in float32 regardless); the PyTorch reference ignores it.
    storage_dtype: torch.dtype = torch.float32
    n_ref: torch.Tensor | None = None
    mg_c: torch.Tensor | None = None
    mg_a: torch.Tensor | None = None


def jumps(drive: Drive, jump: torch.Tensor) -> list[torch.Tensor]:
    """Per pathway, the ``(N, 2, S)`` conductance jump per unit activation."""
    return [
        jump[None] * drive.scale[k][:, None, None] * drive.made[k]
        for k in range(drive.made.shape[0])
    ]


def spike(v: torch.Tensor, p: Physiology, held=None) -> torch.Tensor:
    """Spikes from the voltage, none where ``held`` (refractory)."""
    fn = SampleSpike if p.spike_mode == "sample" else SurrGradSpike
    s = fn.apply(v - p.theta, p.surr_scale)
    return s if held is None else s * (1 - held)


def mg_block(v, mg_c, mg_a):
    """``(B, N, S)``: ``1 / (1 + mg_c exp(-mg_a v))``, written as a sigmoid so it stays
    finite at any voltage; ``mg_c = 0`` (no block) gives exactly 1."""
    return torch.sigmoid(mg_a * v[:, :, None] - torch.log(mg_c))


def membrane(v, s, g, p: Physiology):
    """Currents and the voltage after the step, reset where ``s`` (a spike, or the
    refractory period: see ``held_at_reset``). Returns ``(v_next, I, I_leak)``.

    With the conductances held over the step, ``C_m dv/dt = -g_tot (v - V_inf)`` is
    linear: ``v`` relaxes towards ``V_inf`` by the factor ``a = exp(-dt g_tot / C_m)``,
    i.e. ``v_next = v - (1 - a) / g_tot (I_leak + sum_j I_j)`` — never past ``V_inf``.
    ``g_L = C_m / tau_mem``, written ``-C_m log(beta) / dt`` (``beta`` is the leak's
    decay, so ``a = beta exp(-dt sum_j G_j / C_m)``).
    """
    g_total = torch.minimum(g.sum(dim=2).clamp(min=0.0), p.g_clip)
    if p.mg_c is not None:  # the Mg block: B(v) = 1 / (1 + mg_c exp(-mg_a v))
        g_total = g_total * mg_block(v, p.mg_c, p.mg_a)
    I = g_total * (v[:, :, None] - p.E_syn)
    rate = -torch.log1p(p.beta - 1) / p.dt  # 1 / tau_mem
    g_L = p.C_m * rate
    I_leak = g_L * (v - p.E_L)
    G = g_total.sum(dim=2)
    x = p.dt * rate + p.dt * G / p.C_m  # dt g_tot / C_m
    step = -torch.expm1(-x) / (g_L + G)  # (1 - a) / g_tot, precisely
    s = s.detach()
    v_next = (v - step * (I.sum(dim=2) + I_leak)) * (1 - s) + p.U_reset * s
    return v_next, I, I_leak


def simulate(
    v: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    drive: Drive,
    p: Physiology,
    *,
    refractory: torch.Tensor | None = None,
    gating: torch.Tensor | None = None,
    probabilities: bool = False,
    track: bool = False,
    track_batch_idx: int | None = None,
    retain: list | None = None,
) -> Run:
    """Simulate ``x.shape[1]`` steps from state ``(v, g)`` with input ``x``.

    Returns a ``Run`` whose ``tracked`` (if tracking) holds, per step: the voltage
    after the step ``v``, the synaptic and leak currents ``I`` ``(..., S + 1)`` and the
    conductances after the input ``g``, for the tracked trial(s).

    Args:
        v, g: State, ``(B, N)`` and ``(B, N, 2, S)``.
        refractory: Steps of refractory period left, ``(B, N)`` int32 (None: none).
        gating: The saturating pathways' gating variables ``(B, R)`` (see ``Drive``),
            if any.
        x: Feedforward input, ``(B, T, n_inputs)``.
        probabilities: Also return the escape-noise spike probability per step.
        track, track_batch_idx: Record ``Run.tracked``, for one trial or all (None).
        retain: If a list, append ``(v, g, s)`` of every step with gradients retained.
    """
    factors, gating = gate_feedforward(drive, x, gating)
    A_ff = drive.feedforward_activations(x, factors)
    kernels = jumps(drive, p.jump)
    gates = [drive.gate(k) for k in range(len(drive.recurrent))]
    if any(gates):
        u = [None if gate is None else gating[:, gate.columns] for gate in gates]
    keep = (
        slice(None)
        if track_batch_idx is None
        else slice(track_batch_idx, track_batch_idx + 1)
    )
    spikes, probs, tracked = [], [], {"v": [], "I": [], "g": []}
    if refractory is None:
        refractory = torch.zeros(v.shape, dtype=torch.int32, device=v.device)

    for t in range(x.shape[1]):
        held = refractory_mask(refractory, p.n_ref)
        s = spike(v, p, held)
        if probabilities:
            prob = torch.sigmoid(p.surr_scale * (v - p.theta))
            probs.append(prob if held is None else prob * (1 - held))
        v_next, I, I_leak = membrane(v, held_at_reset(s, held), g, p)
        refractory = count_down(s, refractory, p.n_ref)
        s_rec = s if drive.recurrent_gradients else s.detach()
        activations = []
        for k, (rows, W) in enumerate(drive.recurrent):
            pre = s_rec[:, rows]
            if gates[k] is not None:  # each spike times the fraction it opens
                f, u[k] = gate_step(u[k], pre, gates[k])
                pre = pre * f
            activations.append(pre @ W)
        activations += list(A_ff[:, :, t])
        g = g * p.decay
        for a, kernel in zip(activations, kernels):
            g = g + a[:, :, None, None] * kernel
        v = v_next

        spikes.append(s)
        if retain is not None:
            for tensor in (v, g, s):
                if tensor.requires_grad:
                    tensor.retain_grad()
            retain.append((v, g, s))
        if track:
            tracked["v"].append(v[keep].detach())
            tracked["I"].append(
                torch.cat([I, I_leak[..., None]], dim=-1)[keep].detach()
            )
            tracked["g"].append(g[keep].detach())

    return Run(
        output=torch.stack(spikes, dim=1),
        v=v,
        syn=g,
        probabilities=torch.stack(probs, dim=1) if probabilities else None,
        tracked={k: stack_steps(val) for k, val in tracked.items()} if track else None,
        refractory=refractory,
        gating=_gating(gating, gates, u) if any(gates) else gating,
    )


def _gating(gating, gates, u):
    """The gating state with the recurrent pathways' columns updated to ``u``."""
    gating = gating.clone()
    for gate, value in zip(gates, u):
        if gate is not None:
            gating[:, gate.columns] = value
    return gating
