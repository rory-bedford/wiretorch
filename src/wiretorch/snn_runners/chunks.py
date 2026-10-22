"""What the runners do with one chunk: move it to the device, read the model's output,
and hand each loss the inputs it asks for.

A loss declares its inputs in ``required_inputs`` (and ``requires_target`` for the
target spikes); ``loss_inputs`` assembles exactly those from the chunk and the model.
A loss without ``required_inputs`` gets the output spikes.
"""

from __future__ import annotations

from typing import Callable

import torch

#: Data fields of a dataloader batch, besides the input spikes.
TARGET_FIELDS = ("target_spikes", "target_calcium", "target_rate")


def to_device(batch, device) -> dict[str, torch.Tensor | None]:
    """The batch's input spikes and targets on ``device`` (None where absent)."""
    out = {"input_spikes": batch.input_spikes.to(device, non_blocking=True)}
    for name in TARGET_FIELDS:
        value = getattr(batch, name, None)
        out[name] = None if value is None else value.to(device, non_blocking=True)
    return out


class Prefetcher:
    """Iterates over batches on the device, copying each one while the last one runs.

    On CUDA the next batch's host-to-device copy is issued on a side stream as soon as
    the current batch is handed out, so it overlaps the current chunk's compute (a
    DataLoader with ``pin_memory=True`` makes the copy asynchronous). Elsewhere it is
    ``to_device(next(batches))``.
    """

    _END = object()

    def __init__(self, batches, device):
        self.batches = batches
        self.device = device
        cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device) if cuda else None
        self._next = None

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor | None]:
        if self.stream is None:
            return to_device(next(self.batches), self.device)
        batch = self._next if self._next is not None else self._copy()
        if batch is self._END:
            raise StopIteration
        compute = torch.cuda.current_stream(self.device)
        compute.wait_stream(self.stream)
        for tensor in batch.values():
            if tensor is not None:  # allocated on the side stream, used on this one
                tensor.record_stream(compute)
        self._next = self._copy()
        return batch

    def _copy(self):
        try:
            batch = next(self.batches)
        except StopIteration:
            return self._END
        with torch.cuda.stream(self.stream):
            return to_device(batch, self.device)


def model_outputs(outputs) -> dict[str, torch.Tensor]:
    """Normalise a model's return value to a dict with at least ``"spikes"``.

    Models return spikes, ``(probabilities, spikes)``, or a dict (tracked variables,
    hidden spikes).
    """
    if isinstance(outputs, dict):
        return dict(outputs)
    if isinstance(outputs, tuple):
        probabilities, spikes = outputs
        return {"spikes": spikes, "probabilities": probabilities}
    return {"spikes": outputs}


def loss_inputs(
    name: str, loss_fn: Callable, chunk: dict, model, masks: dict | None = None
) -> dict:
    """The keyword arguments ``loss_fn`` declares, from the chunk and the model.

    Args:
        name: The loss's name (for error messages).
        chunk: Model outputs (``spikes``, ``probabilities``, ``voltages``,
            ``hidden_spikes``) and data (``target_*``) of this chunk.
        model: Source of parameters a loss may ask for (weights, scaling factors).
        masks: ``connectome_mask`` and ``feedforward_mask`` (None if not given).
    """
    masks = masks or {}
    sources = {
        "output_spikes": lambda: chunk["spikes"],
        "output_probabilities": lambda: chunk["probabilities"],
        "voltages": lambda: chunk["voltages"],
        "hidden_spikes": lambda: chunk.get("hidden_spikes"),
        "dt": lambda: model.dt,
        "recurrent_weights": lambda: model.weights,
        "feedforward_weights": lambda: model.weights_FF,
        "cell_type_indices": lambda: model.cell_type_indices,
        "scaling_factors": lambda: model.scaling_factors,
        "scaling_factors_FF": lambda: model.scaling_factors_FF,
        "connectome_mask": lambda: masks.get("connectome_mask"),
        "feedforward_mask": lambda: masks.get("feedforward_mask"),
    }
    for field in ("target_calcium", "target_rate"):
        sources[field] = lambda field=field: _required(chunk, field, name)

    inputs = {
        key: sources[key]()
        for key in getattr(loss_fn, "required_inputs", ["output_spikes"])
        if key in sources
    }
    if getattr(loss_fn, "requires_target", False):
        inputs["target_spikes"] = _required(chunk, "target_spikes", name)
    return inputs


def _required(chunk: dict, field: str, loss_name: str) -> torch.Tensor:
    value = chunk.get(field)
    if value is None:
        raise ValueError(
            f"Loss '{loss_name}' requires {field}, but the dataloader provided none; "
            "its dataset must return it in SpikeData."
        )
    return value


def weighted_losses(
    loss_functions: dict, loss_weights: dict, chunk: dict, model, masks=None
) -> dict[str, torch.Tensor]:
    """Each loss times its weight, and their sum as ``"total"`` (tensors, no sync).

    Losses weighted 0 (or not weighted at all) are not computed, and not returned. A
    loss that renders an exact prediction (``last_prediction``, e.g. the calcium
    model's dF/F) has it stored in ``chunk["predicted_calcium"]``.
    """
    losses = {}
    for name, loss_fn in loss_functions.items():
        weight = loss_weights.get(name, 0.0)
        if weight == 0:
            continue
        value = loss_fn(**loss_inputs(name, loss_fn, chunk, model, masks))
        if getattr(loss_fn, "last_prediction", None) is not None:
            chunk["predicted_calcium"] = loss_fn.last_prediction
        losses[name] = weight * value
    losses["total"] = sum(
        losses.values(), torch.zeros((), device=chunk["spikes"].device)
    )
    return losses


def warm_up_losses(loss_functions: dict, chunk: dict, model, masks=None) -> None:
    """Advance stateful losses (those with ``reset_state``) through a burn-in chunk."""
    for name, loss_fn in loss_functions.items():
        if hasattr(loss_fn, "reset_state"):
            loss_fn(**loss_inputs(name, loss_fn, chunk, model, masks))


def reset_losses(loss_functions: dict) -> None:
    for loss_fn in loss_functions.values():
        if hasattr(loss_fn, "reset_state"):
            loss_fn.reset_state()


def loss_states(loss_functions: dict) -> dict[str, torch.Tensor | None]:
    """The state the losses carry across chunks, by ``loss/submodule/attribute``.

    Collected from every (sub)module that names its carried attributes in
    ``CARRIED_STATE`` (the van Rossum filters, the calcium model's latent).
    """
    return {
        key: getattr(module, attr) for key, module, attr in _carriers(loss_functions)
    }


def load_loss_states(loss_functions: dict, states: dict) -> None:
    """Restore what ``loss_states`` returned (keys it does not hold are left alone)."""
    for key, module, attr in _carriers(loss_functions):
        if key in states:
            setattr(module, attr, states[key])


def _carriers(loss_functions: dict):
    for name, loss_fn in loss_functions.items():
        modules = (
            loss_fn.named_modules()
            if isinstance(loss_fn, torch.nn.Module)
            else [("", loss_fn)]
        )
        for path, module in modules:
            for attr in getattr(module, "CARRIED_STATE", ()):
                yield "/".join(filter(None, (name, path, attr))), module, attr
