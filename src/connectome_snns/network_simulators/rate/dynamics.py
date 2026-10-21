"""The rate-network time step, in plain PyTorch.

This is the reference implementation: the specification any faster engine must
follow. The model is the current-based LIF network with a nonlinearity in place of the
spike — a membrane variable driven by exponentially filtered synaptic currents:

    tau_m dv/dt   = -v + sum_s sign_s I_s
    tau_s dI_s/dt = -I_s + (input to synapse type s)
    r             = gain * f(v - theta)

with ``f`` one of ``NONLINEARITIES``, and ``tau_m``, ``theta``, ``gain`` and ``f`` per
cell type, ``tau_s`` per synapse type. One step, for every neuron of every trial:

1. **Output** the rate of the voltage entering the step, ``r = gain f(v - theta)``.
2. **Filter** each synaptic current towards this step's input: every pathway's
   activation (presynaptic rates times weights) times its kernel (its scaling factor,
   into each synapse type it makes), ``I <- alpha I + (1 - alpha) input``.
3. **Integrate** ``v <- beta v + (1 - beta) sum_s sign_s I_s``.

``alpha = exp(-dt / tau_s)`` and ``beta = exp(-dt / tau_m)``. The steps are normalised
(exponential Euler): with constant input, ``v`` and ``I`` settle at that input whatever
dt or the time constants, so weights mean "input per unit rate". A time constant of 0
makes its variable follow its input exactly.

Gradients flow everywhere: through the rates, the recurrence and the state.

The classic rate models are limits of this one (see ``filter_rates``):

* **Voltage form** (Amari; Zhang 1996), ``tau dx/dt = -x + W phi(x) + W_ff u``: every
  ``tau_s = 0``, and ``tau_m = tau``. Then ``v`` is ``x``, and the output ``phi(x)``.
* **Rate form** (Ben-Yishai et al. 1995; Burak & Fiete 2009),
  ``tau dr/dt = -r + phi(W r + W_ff u)``: ``tau_m = 0``, the recurrent synapse types'
  ``tau_s = tau`` (per source cell type), the feedforward ones' ``tau_s = 0``. The
  currents then carry ``W r`` exactly, and the rate-form ``r`` is the output filtered
  by ``tau`` — ``filter_rates(output, tau, dt)`` — one step behind the input.

The model has no noise of its own. Feedforward noise is just input; intrinsic noise
(independent per neuron, as attractor models use) is an input too: one white-noise
input row per neuron, through a diagonal ``FrozenProjection`` of weight ``sigma`` to an
excitatory synapse type with ``tau_syn = 0``, which adds it straight to the membrane's
input. (Negative noise passes through the excitatory sign unchanged.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from connectome_snns.network_simulators.network import Drive, Run

#: The nonlinearities ``f`` a cell type can use (``r = gain f(v - theta)``).
NONLINEARITIES = {
    "relu": F.relu,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}


@dataclass
class Physiology:
    """Every parameter of the dynamics, as tensors on the simulation device.

    Per neuron ``(N,)``: ``theta``, ``gain``, ``beta`` (the membrane's per-step decay).
    Per synapse type ``(S,)``: ``sign`` (+1 or -1) and ``decay`` (the currents'
    per-step decay). ``groups`` lists, per cell type, its nonlinearity's name and its
    neurons.
    """

    theta: torch.Tensor
    gain: torch.Tensor
    beta: torch.Tensor
    sign: torch.Tensor
    decay: torch.Tensor
    groups: list[tuple[str, torch.Tensor]]
    #: What the CUDA kernels multiply rates and weights in (accumulating in float32);
    #: the PyTorch reference ignores it.
    storage_dtype: torch.dtype = torch.float32


def kernels(drive: Drive) -> list[torch.Tensor]:
    """Per pathway, the ``(N, S)`` input to each synapse type per unit activation."""
    return [drive.scale[k][:, None] * drive.made[k] for k in range(drive.made.shape[0])]


def rates(v: torch.Tensor, p: Physiology) -> torch.Tensor:
    """``gain f(v - theta)``, with each cell type's own ``f``."""
    x = v - p.theta
    r = torch.zeros_like(x)
    for name, neurons in p.groups:
        r = r.index_copy(-1, neurons, NONLINEARITIES[name](x.index_select(-1, neurons)))
    return p.gain * r


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
        probabilities: Not available (rate networks have no spikes); must be False.
        track, track_batch_idx: Record ``Run.tracked``, for one trial or all (None):
            the voltage after each step ``v`` and the currents after the input ``I``.
        retain: If a list, append ``(v, I, r)`` of every step with gradients retained.
    """
    assert not probabilities, "rate networks have no spike probabilities"
    A_ff = drive.feedforward_activations(x)
    jumps = kernels(drive)
    keep = (
        slice(None)
        if track_batch_idx is None
        else slice(track_batch_idx, track_batch_idx + 1)
    )
    outputs, tracked = [], {"v": [], "I": []}

    for t in range(x.shape[1]):
        r = rates(v, p)
        activations = [r[:, rows] @ W for rows, W in drive.recurrent]
        activations += list(A_ff[:, :, t])
        target = sum(a[:, :, None] * jump for a, jump in zip(activations, jumps))
        I = p.decay * I + (1 - p.decay) * target
        I_syn = sum(I[..., j] * p.sign[j] for j in range(I.shape[-1]))
        v = p.beta * v + (1 - p.beta) * I_syn

        outputs.append(r)
        if retain is not None:
            for tensor in (v, I, r):
                if tensor.requires_grad:
                    tensor.retain_grad()
            retain.append((v, I, r))
        if track:
            tracked["v"].append(v[keep].detach())
            tracked["I"].append(I[keep].detach())

    return Run(
        output=torch.stack(outputs, dim=1),
        v=v,
        syn=I,
        tracked={k: torch.stack(val, dim=1) for k, val in tracked.items()}
        if track
        else None,
    )


def filter_rates(
    output: torch.Tensor, tau: torch.Tensor | float, dt: float
) -> torch.Tensor:
    """The rate-form rates of a network set up as the rate form (see the docstring).

    ``s <- alpha s + (1 - alpha) r`` from ``s = 0``, with ``alpha = exp(-dt / tau)``
    per neuron (``tau`` a float or ``(N,)``): ``(B, T, N)`` from the output ``r``,
    where step ``t`` holds ``s`` after step ``t``.
    """
    alpha = torch.exp(-dt / torch.as_tensor(tau, dtype=output.dtype))
    alpha = alpha.to(output.device)
    s = torch.zeros_like(output[:, 0])
    out = []
    for t in range(output.shape[1]):
        s = alpha * s + (1 - alpha) * output[:, t]
        out.append(s)
    return torch.stack(out, dim=1)
