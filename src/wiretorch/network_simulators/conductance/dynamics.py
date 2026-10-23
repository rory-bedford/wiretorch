"""The conductance-based LIF time step, in plain PyTorch.

This is the reference implementation — the CPU path, the path used when per-step
gradients are tracked, and the specification the Triton kernels in ``kernels.py``
follow line for line. One step, for every neuron of every trial:

1. **Spike** if the voltage is above threshold (Heaviside with a surrogate gradient, or
   Bernoulli escape noise in ``"sample"`` mode) and the neuron is not refractory.
2. **Currents** from the conductances: each synapse type's rise + decay components,
   clipped to ``[0, g_clip]``, times the driving force ``v - E_syn``; plus the leak.
3. **Integrate** the voltage with forward Euler, and reset the neurons that spiked; a
   neuron stays at reset for the ``n_ref`` steps after its spike (its refractory
   period), and a counter per neuron (``refractory``, part of the state) tracks them.
4. **Decay** each conductance component, then **inject** this step's input: every
   pathway adds its activation (presynaptic spikes times weights) times its kernel.

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


def membrane(v, s, g, p: Physiology):
    """Currents and the voltage after the step, reset where ``s`` (a spike, or the
    refractory period: see ``held_at_reset``). Returns ``(v_next, I, I_leak)``."""
    g_total = torch.minimum(g.sum(dim=2).clamp(min=0.0), p.g_clip)
    I = g_total * (v[:, :, None] - p.E_syn)
    I_leak = (v - p.E_L) * (1 - p.beta) * p.C_m / p.dt
    s = s.detach()
    v_next = (v - (I.sum(dim=2) + I_leak) * p.dt / p.C_m) * (1 - s) + p.U_reset * s
    return v_next, I, I_leak


def simulate(
    v: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    drive: Drive,
    p: Physiology,
    *,
    refractory: torch.Tensor | None = None,
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
        x: Feedforward input, ``(B, T, n_inputs)``.
        probabilities: Also return the escape-noise spike probability per step.
        track, track_batch_idx: Record ``Run.tracked``, for one trial or all (None).
        retain: If a list, append ``(v, g, s)`` of every step with gradients retained.
    """
    A_ff = drive.feedforward_activations(x)
    kernels = jumps(drive, p.jump)
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
        activations = [s_rec[:, rows] @ W for rows, W in drive.recurrent]
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
    )
