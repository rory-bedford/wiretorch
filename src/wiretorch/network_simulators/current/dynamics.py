"""The current-based LIF time step, in plain PyTorch.

This is the reference implementation — the CPU path, the path used when per-step
gradients are tracked, and the specification the Triton kernels in ``kernels.py``
follow line for line. Neurons are in natural units (by default reset 0, threshold 1).
One step, for every neuron of every trial:

1. **Spike** if the voltage is above threshold (Heaviside with a surrogate gradient, or
   Bernoulli escape noise in ``"sample"`` mode).
2. **Decay** each synaptic current, then **inject** this step's input: every pathway
   adds its activation (presynaptic spikes times weights) times its kernel (its
   scaling factor, into each synapse type it makes). A synapse type with
   ``tau_syn = 0`` keeps no memory: its current is just this step's input.
3. **Integrate** ``v <- beta v + (1 - beta) (U_rest + R sum_j sign_j I_j)`` (the sign
   is +1 for excitatory synapse types, -1 for inhibitory), and reset the neurons that
   spiked. In natural units (the default) ``U_rest = 0`` and ``R = 1``.

That is ``tau_m dv/dt = -(v - U_rest) + R sum_j sign_j I_j`` and ``tau_s dI_j/dt = -I_j`` between
inputs, stepped with ``beta = exp(-dt / tau_m)`` and ``alpha = exp(-dt / tau_s)``. An
input is an event: each unit of it (a spike) makes the current jump by its weight, so
one spike moves the voltage by ``R w tau_s / tau_m`` in all, whatever dt. Input rows
therefore hold counts per step — spikes; a continuous rate ``rho`` enters as
``rho dt``. (Before October 2026 the step was ``v <- beta v + sum_j sign_j I_j``, the
same model with weights ``(1 - beta)`` times these at a given dt, but a different one
whenever dt changed; ``from_weights`` converts weights written for it.)

Unlike the conductance-based networks, gradients flow through the recurrent spikes
(into the presynaptic neurons, via the surrogate) as well as through the voltage and
currents; never through the reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wiretorch.network_simulators.network import Drive, Run, stack_steps
from wiretorch.training_utils.surrogate_gradients import (
    SampleSpike,
    SurrGradSpike,
)


@dataclass
class Physiology:
    """Every parameter of the dynamics, as tensors on the simulation device.

    Per neuron ``(N,)``: ``theta``, ``U_reset``, ``beta`` (the leak's per-step decay),
    ``U_rest`` (the voltage the leak relaxes to) and ``R`` (the membrane resistance).
    Per synapse type ``(S,)``: ``sign`` (+1 or -1) and ``decay`` (per-step decay of the
    current).
    """

    theta: torch.Tensor
    U_reset: torch.Tensor
    beta: torch.Tensor
    U_rest: torch.Tensor
    R: torch.Tensor
    sign: torch.Tensor
    decay: torch.Tensor
    surr_scale: torch.Tensor
    spike_mode: str = "deterministic"
    #: What the CUDA kernels store currents and recurrent weights in (they compute
    #: in float32 regardless); the PyTorch reference ignores it.
    storage_dtype: torch.dtype = torch.float32


def kernels(drive: Drive) -> list[torch.Tensor]:
    """Per pathway, the ``(N, S)`` current jump per unit activation."""
    return [drive.scale[k][:, None] * drive.made[k] for k in range(drive.made.shape[0])]


def spike(v: torch.Tensor, p: Physiology) -> torch.Tensor:
    fn = SampleSpike if p.spike_mode == "sample" else SurrGradSpike
    return fn.apply(v - p.theta, p.surr_scale)


def membrane(v, s, I, p: Physiology) -> torch.Tensor:
    """The voltage after the step: leak, the signed synaptic currents, and the reset."""
    # Summed one synapse type at a time, in index order (the kernels sum the same way).
    I_syn = sum(I[..., j] * p.sign[j] for j in range(I.shape[-1]))
    s = s.detach()
    return (p.beta * v + (1 - p.beta) * (p.U_rest + p.R * I_syn)) * (
        1 - s
    ) + p.U_reset * s


def simulate(
    v: torch.Tensor,
    I: torch.Tensor,
    x: torch.Tensor,
    drive: Drive,
    p: Physiology,
    *,
    probabilities: bool = False,
    track: bool = False,
    track_batch_idx: int | None = None,
    retain: list | None = None,
) -> Run:
    """Simulate ``x.shape[1]`` steps from state ``(v, I)`` with input ``x``.

    Args:
        v, I: State, ``(B, N)`` and ``(B, N, S)``.
        x: Feedforward input, ``(B, T, n_inputs)``.
        probabilities: Also return the escape-noise spike probability per step.
        track, track_batch_idx: Record ``Run.tracked``, for one trial or all (None):
            the voltage after each step ``v`` and the currents after the input ``I``.
        retain: If a list, append ``(v, I, s)`` of every step with gradients retained.
    """
    A_ff = drive.feedforward_activations(x)
    jumps = kernels(drive)
    keep = (
        slice(None)
        if track_batch_idx is None
        else slice(track_batch_idx, track_batch_idx + 1)
    )
    spikes, probs, tracked = [], [], {"v": [], "I": []}

    for t in range(x.shape[1]):
        s = spike(v, p)
        if probabilities:
            probs.append(torch.sigmoid(p.surr_scale * (v - p.theta)))
        # Not detached: the gradient reaches the presynaptic neurons' surrogates.
        activations = [s[:, rows] @ W for rows, W in drive.recurrent]
        activations += list(A_ff[:, :, t])
        I = I * p.decay
        for a, jump in zip(activations, jumps):
            I = I + a[:, :, None] * jump
        v = membrane(v, s, I, p)

        spikes.append(s)
        if retain is not None:
            for tensor in (v, I, s):
                if tensor.requires_grad:
                    tensor.retain_grad()
            retain.append((v, I, s))
        if track:
            tracked["v"].append(v[keep].detach())
            tracked["I"].append(I[keep].detach())

    return Run(
        output=torch.stack(spikes, dim=1),
        v=v,
        syn=I,
        probabilities=torch.stack(probs, dim=1) if probabilities else None,
        tracked={k: stack_steps(val) for k, val in tracked.items()} if track else None,
    )
