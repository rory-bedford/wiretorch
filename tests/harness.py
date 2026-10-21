"""How a network is driven and read out — shared by the fixture generator and the tests,
so a recording and its check do exactly the same thing."""

from __future__ import annotations

import numpy as np
import torch

from networks import Built, Scenario, loss_readout

WINDOW = 10  # steps between recorded states; the lockstep tests restart at each one
GRAD_STEPS = 100  # length of the single chunk the gradient fixtures backpropagate
READOUT_SEED = 7


def unpack_output(out, output: str = "spikes") -> dict[str, torch.Tensor]:
    """Normalise every simulator's return type to a dict.

    Simulators return their output (spikes, or rates: ``output`` names which),
    ``(probabilities, spikes)``, or a dict of tracked variables; the result always has
    the output and whichever of ``"voltages"``, ``"probabilities"`` and
    ``"hidden_spikes"`` the model produced.
    """
    if isinstance(out, torch.Tensor):
        return {output: out}
    if isinstance(out, tuple):
        probabilities, spikes = out
        return {"spikes": spikes, "probabilities": probabilities}
    keep = ("spikes", "rates", "voltages", "probabilities", "hidden_spikes")
    return {k: out[k] for k in keep if k in out}


def tracks_voltages(scenario: Scenario) -> bool:
    """Whether a free run records voltages (single-layer models with a state)."""
    return bool(scenario.state_keys)


def free_run(scenario: Scenario, built: Built, device="cpu") -> dict[str, np.ndarray]:
    """Run the full input in ``WINDOW``-step chunks, recording the state before each.

    Returns stacked arrays: ``spikes`` (and ``voltages``, ``probabilities``,
    ``hidden_spikes`` where produced) over all steps, and ``state.<key>`` with one entry
    per window start plus the final state.
    """
    model = built.model.to(device)
    model.track_variables = tracks_voltages(scenario)
    if model.track_variables:
        model.track_batch_idx = None
    x = torch.from_numpy(built.input_spikes).to(device)

    chunks, states = [], {k: [] for k in scenario.state_keys}
    with torch.no_grad():
        for start in range(0, x.shape[1], WINDOW):
            for key, value in model.get_checkpoint_state().items() if states else ():
                states[key].append(np.array(value))
            out = unpack_output(model(x[:, start : start + WINDOW]), built.output)
            chunks.append({k: v.cpu().numpy() for k, v in out.items()})
        for key, value in model.get_checkpoint_state().items() if states else ():
            states[key].append(np.array(value))

    record = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}
    if "spikes" in record:
        record["spikes"] = record["spikes"] > 0.5
    if "hidden_spikes" in record:
        record["hidden_spikes"] = record["hidden_spikes"] > 0.5
    record.update({f"state.{k}": np.stack(v) for k, v in states.items()})
    return record


def readout_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """``sum(output * R)`` (+ the same for hidden spikes and probabilities).

    A fixed random linear readout: every spike's surrogate gradient contributes, with
    no loss-function machinery between the simulator and the gradient being tested.
    """
    loss = next(iter(out.values())).new_zeros(())
    for i, key in enumerate(("spikes", "hidden_spikes", "probabilities", "rates")):
        if key in out:
            R = loss_readout(READOUT_SEED + i, tuple(out[key].shape))
            loss = loss + (out[key] * R.to(out[key].device)).sum()
    return loss


def is_trainable(built: Built) -> bool:
    return any(p.requires_grad for p in built.model.parameters())


def gradients(
    built: Built, device="cpu", track=False
) -> tuple[float, dict[str, np.ndarray]]:
    """Backpropagate ``readout_loss`` through the first ``GRAD_STEPS`` steps.

    ``track`` turns on variable tracking (of trial 0), which must not change anything.
    Gradients are named as the fixtures record them (``Built.recorded_gradient_names``).
    """
    model = built.model.to(device)
    model.track_variables = track
    if track:
        for layer in getattr(model, "layer1", model), getattr(model, "layer2", model):
            layer.track_batch_idx = 0
    x = torch.from_numpy(built.input_spikes[:, :GRAD_STEPS]).to(device)
    loss = readout_loss(unpack_output(model(x), built.output))
    loss.backward()
    grads = {
        name: p.grad.detach().cpu().numpy()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    if built.recorded_gradient_names is not None:
        grads = built.recorded_gradient_names(model, grads)
    return float(loss.detach()), grads
