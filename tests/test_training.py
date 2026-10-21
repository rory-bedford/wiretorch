"""A short ``SNNTrainer`` run follows the recorded parameter trajectory.

A feedforward network fitted to a recurrent network's spikes: van Rossum loss, burn-in,
gradient accumulation and Adam. Checks the trainer and the simulator's backward pass
together, update by update.
"""

from pathlib import Path

import numpy as np
import pytest

import fixture_io
from fixtures.generate import training_log, training_setup


def test_training_trajectory():
    recorded, _ = fixture_io.load("training")
    trainer, trajectory, names = training_setup()
    trainer.train()

    assert len(trajectory) == len(next(iter(recorded.values())))
    for i, name in enumerate(names):
        np.testing.assert_allclose(
            np.stack([step[i] for step in trajectory]),
            recorded[f"trajectory.{name}"],
            rtol=1e-4,
            atol=1e-6,
            err_msg=name,
        )


def test_plot_data_concatenates_lazily():
    from collections import deque

    from connectome_snns.snn_runners.trainer import PlotData

    chunks = {
        "spikes": deque([np.zeros((1, 2, 3)), np.ones((1, 4, 3))]),
        "voltages": deque(),
    }
    data = PlotData(chunks)
    snapshot = data.copy()
    snapshot["epoch"] = 7
    assert snapshot["spikes"].shape == (1, 6, 3) and snapshot["epoch"] == 7
    assert "voltages" not in data._cache  # never read, never concatenated
    assert set(snapshot) == {"spikes", "voltages", "epoch"} and "epoch" not in data


def test_metrics_log(tmp_path):
    """The metrics CSV — losses, stats and gradient norms — is what it was."""
    recorded, _ = fixture_io.load("training_log")
    logged = training_log(tmp_path)
    assert logged.keys() == recorded.keys()
    for key, value in recorded.items():
        np.testing.assert_allclose(
            logged[key], value, rtol=1e-4, atol=1e-6, err_msg=key
        )


def _assert_same_parameters(a, b):
    for (name, x), y in zip(a.model.named_parameters(), b.model.parameters()):
        np.testing.assert_array_equal(
            x.detach().numpy(), y.detach().numpy(), err_msg=name
        )


def _train_keeping(tmp_path, at, **kwargs):
    """Train uninterrupted, keeping a copy of the checkpoint at chunk ``at``."""
    import shutil

    def keep(chunk, output_dir, model):  # the latest checkpoint is overwritten
        if chunk == at:
            shutil.copy(
                output_dir / "checkpoints" / "checkpoint_latest.pt",
                tmp_path / "kept.pt",
            )

    whole, _, _ = training_setup(scheduler=True, checkpoint_callbacks=[keep], **kwargs)
    whole.train(output_dir=tmp_path)
    return whole


@pytest.mark.parametrize(
    "at, checkpoint_interval", [(2, 2), (4, 2), (6, 2), (8, 2), (3, 3), (9, 3)]
)
def test_resume_matches_uninterrupted(tmp_path, at, checkpoint_interval):
    """Resuming from any checkpoint ends exactly where the uninterrupted run does.

    Five chunks per data epoch, the first of them burn-in, an update every two chunks:
    2 and 4 are inside the first data epoch, 6 after the second's burn-in chunk, 8
    inside it; 3 and 9 fall between updates, with gradients pending. The data must be
    skipped to the checkpoint, and the van Rossum filter state and the pending
    gradients restored.
    """
    whole = _train_keeping(tmp_path, at, checkpoint_interval=checkpoint_interval)
    resumed, _, _ = training_setup(
        scheduler=True, checkpoint_interval=checkpoint_interval
    )
    assert resumed.resume(tmp_path / "kept.pt") == at
    resumed.train()
    _assert_same_parameters(whole, resumed)


def test_resume_with_load_checkpoint(tmp_path):
    """The two-call resume (``load_checkpoint``, ``set_checkpoint_state``) is exact at a
    data-epoch boundary, where the losses' state is reset anyway."""
    from connectome_snns.training_utils.checkpointing import load_checkpoint

    whole = _train_keeping(tmp_path, 5, checkpoint_interval=5, chunks_per_update=1)
    resumed, _, _ = training_setup(scheduler=True, chunks_per_update=1)
    chunk, best = load_checkpoint(
        tmp_path / "kept.pt", resumed.model, resumed.optimizer, resumed.scaler, "cpu",
        scheduler=resumed.scheduler,
    )  # fmt: skip
    resumed.set_checkpoint_state(chunk, best)
    resumed.train()
    _assert_same_parameters(whole, resumed)


def test_best_checkpoint_uses_the_mean_loss_since_the_last(tmp_path):
    import torch

    trainer, _, _ = training_setup(checkpoint_interval=4)
    trainer.train(output_dir=tmp_path)
    saved = torch.load(
        tmp_path / "checkpoints" / "checkpoint_latest.pt", weights_only=False
    )
    # Trained: chunks 1-4 and 6-9 (0 and 5 are burn-in). The last checkpoint, after
    # chunk 7, averages chunks 4, 6 and 7.
    since = list(trainer.loss_history)[3:6]
    assert saved["epoch"] == 8
    assert saved["total"] == pytest.approx(float(sum(h["total"] for h in since) / 3))
    assert saved["training_state"]["loss_states"]["van_rossum/prev_output_smooth"].shape


def test_interrupted_save_keeps_the_last_checkpoint(tmp_path, monkeypatch):
    """A job killed while writing a checkpoint leaves the previous one loadable."""
    import torch

    from connectome_snns.training_utils import checkpointing

    trainer, _, _ = training_setup(checkpoint_interval=4)
    trainer.train(output_dir=tmp_path)
    latest = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    before = latest.read_bytes()

    def killed(obj, path):
        Path(path).write_bytes(b"trunc")
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpointing.torch, "save", killed)
    with pytest.raises(KeyboardInterrupt):
        trainer._checkpoint(12, tmp_path)
    monkeypatch.undo()
    assert latest.read_bytes() == before
    assert torch.load(latest, weights_only=False)["epoch"] == 8


def test_burn_in_checkpoints_are_never_best(tmp_path):
    """A checkpoint with only burn-in since the last has no losses to compare."""
    import torch

    trainer, _, _ = training_setup(checkpoint_interval=1)
    trainer.num_epochs = 1  # chunk 0, burn-in
    trainer.train(output_dir=tmp_path)
    saved = torch.load(
        tmp_path / "checkpoints" / "checkpoint_latest.pt", weights_only=False
    )
    assert saved["epoch"] == 1 and saved["total"] == float("inf")
    assert trainer.best_loss == float("inf")


def test_plots_get_tracked_variables(tmp_path):
    """With a plot generator, checkpoints plot the kept chunks' tracked variables."""
    calls = []

    def plot_generator(**data):
        calls.append(data)
        return {}

    trainer, _, _ = training_setup(checkpoint_interval=4)
    trainer.plot_generator = plot_generator
    trainer.train(output_dir=tmp_path)

    assert calls, "no plots at checkpoints"
    data = calls[-1]
    n_neurons = trainer.model.n_neurons
    assert data["spikes"].shape == (1, 40, n_neurons)  # plot_size 1: the last chunk
    assert data["voltages"].shape == (1, 40, n_neurons)
    assert data["conductances_FF"].shape[:3] == (1, 40, n_neurons)
    assert data["target_spikes"].dtype == bool


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
def test_prefetcher_hands_out_every_batch_in_order(device):
    """The prefetcher (a side-stream copy on CUDA) delivers each batch intact, then
    stops, even while the consumer overwrites what it was given."""
    import torch

    from connectome_snns.dataloaders.supervised import SpikeData
    from connectome_snns.snn_runners.chunks import Prefetcher

    batches = [
        SpikeData(
            input_spikes=torch.full((2, 3, 4), float(i)).pin_memory()
            if device == "cuda"
            else torch.full((2, 3, 4), float(i)),
            target_spikes=torch.full((2, 3, 1), -float(i)),
            recurrent_input_spikes=None,
        )
        for i in range(5)
    ]
    for i, batch in enumerate(Prefetcher(iter(batches), device)):
        assert batch["input_spikes"].device.type == device
        assert batch["input_spikes"].eq(i).all() and batch["target_spikes"].eq(-i).all()
        assert batch["target_calcium"] is None
        batch["input_spikes"].mul_(100)  # on the compute stream, after the wait
    assert i == 4
