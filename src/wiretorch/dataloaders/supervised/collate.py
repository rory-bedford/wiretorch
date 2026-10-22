"""Collate functions for supervised spike-train dataloaders."""

import torch

from ._base import SpikeData


def recurrent_input(batch: SpikeData) -> torch.Tensor:
    """The recurrent spikes to feed the model: resampled ones if the dataset provides
    them (``recurrent_input_spikes``), else the teacher's own spikes."""
    if batch.recurrent_input_spikes is not None:
        return batch.recurrent_input_spikes
    return batch.target_spikes


class FeedforwardCollate:
    """Concatenate feedforward and recurrent spikes for feedforward-unrolled training.

    Produces model inputs by concatenating ``[FF spikes, recurrent spikes]``
    along the neuron dimension, while keeping the raw target spikes unchanged.

    When the dataset provides ``recurrent_input_spikes`` (e.g. smoothed and
    resampled recurrent activity), those are used as the recurrent component
    of the model input.  Otherwise ``target_spikes`` are used directly.

    Args:
        dt: Simulation timestep in ms (retained for interface compatibility).

    Example:
        >>> collate = FeedforwardCollate(dt=0.1)
        >>> dataloader = DataLoader(dataset, batch_size=None, collate_fn=collate)
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def __call__(self, batch: SpikeData) -> SpikeData:
        rec_input = recurrent_input(batch)
        concatenated_inputs = torch.cat([batch.input_spikes, rec_input], dim=2)
        return SpikeData(
            input_spikes=concatenated_inputs, target_spikes=batch.target_spikes
        )


def feedforward_collate_fn(batch: SpikeData) -> SpikeData:
    """Legacy wrapper — equivalent to ``FeedforwardCollate()``."""
    return FeedforwardCollate()(batch)


def calcium_collate_fn(batch: SpikeData) -> SpikeData:
    """Pass-through collate for calcium-target datasets.

    The dataset's ``input_spikes`` already are the full model input (no FF/recurrent
    concatenation), so this just forwards them together with the first-class
    ``target_calcium`` (dF/F) and ``target_rate`` (smooth rate) targets.
    """
    return SpikeData(
        input_spikes=batch.input_spikes,
        target_calcium=batch.target_calcium,
        target_rate=batch.target_rate,
    )


class VisibleDrivenCollate:
    """Concatenate [FF, teacher_visible] as input; teacher_visible as target.

    Extracts visible neuron spikes from target_spikes, appends them to
    input_spikes, and provides visible-only targets.

    Args:
        visible_indices: Tensor of visible neuron indices into target_spikes.
    """

    def __init__(self, visible_indices: torch.Tensor):
        self.visible_indices = visible_indices

    def __call__(self, batch: SpikeData) -> SpikeData:
        visible_rec = batch.target_spikes[:, :, self.visible_indices]
        return SpikeData(
            input_spikes=torch.cat([batch.input_spikes, visible_rec], dim=2),
            target_spikes=visible_rec,
        )


class VisibleSubsetCollate:
    """FeedforwardCollate variant that subsets targets to ``visible_indices``.

    Used when a subset of the teacher's neurons has been structurally removed
    from the student model: the recurrent input (smoothed rates if the dataset
    provides them, otherwise raw target spikes) and the supervision target are
    both sliced down to the surviving neuron subset before being concatenated
    with the FF input.

    Args:
        visible_indices: Tensor of surviving neuron indices into
            ``target_spikes`` (and ``recurrent_input_spikes`` if present).
    """

    def __init__(self, visible_indices: torch.Tensor):
        self.visible_indices = visible_indices

    def __call__(self, batch: SpikeData) -> SpikeData:
        target_vis = batch.target_spikes[:, :, self.visible_indices]
        rec_input = recurrent_input(batch)
        rec_input_vis = rec_input[:, :, self.visible_indices]
        concatenated_inputs = torch.cat([batch.input_spikes, rec_input_vis], dim=2)
        return SpikeData(input_spikes=concatenated_inputs, target_spikes=target_vis)


def single_neuron_collate_fn(batch: SpikeData) -> SpikeData:
    """Concatenate FF+recurrent as inputs, extract neuron 0 as target.

    Args:
        batch: ``SpikeData`` from a supervised dataset.

    Returns:
        ``SpikeData`` with ``input_spikes`` of shape ``(batch, time, n_ff + n_rec)``
        and ``target_spikes`` of shape ``(batch, time, 1)``.
    """
    rec_input = recurrent_input(batch)
    concatenated_inputs = torch.cat([batch.input_spikes, rec_input], dim=2)
    single_neuron_target = batch.target_spikes[:, :, 0:1]
    return SpikeData(
        input_spikes=concatenated_inputs, target_spikes=single_neuron_target
    )
