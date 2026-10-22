"""The calcium path: rate datasets with dF/F targets, and the calcium MSE loss.

The datasets Poisson-sample input spikes from measured rates and return the dF/F frames
of each chunk; the loss renders spikes through the observation model, frame-averages
them onto the imaging grid and takes a NaN-masked MSE.
"""

import numpy as np
import pytest
import torch
import zarr

from wiretorch.dataloaders.supervised import (
    ParallelSegmentCalciumDataset,
    RateCalciumFFDataset,
)
from wiretorch.observation_models import CalciumObservationModel
from wiretorch.training_utils.losses import CalciumMSELoss

DT, FRAME_HZ = 1.0, 100.0  # 10 steps per frame
N_CELLS, CHUNK, FRAMES = 4, 40, 4


@pytest.fixture
def windows(tmp_path):
    """Pre-cut recording: rates (n_windows, n_cells, n_steps)."""
    path = tmp_path / "windows.zarr"
    root = zarr.open_group(path, mode="w")
    root.attrs.update(dt_ms=DT, frame_rate_hz=FRAME_HZ, batch_duration_s=0.04)
    rates = np.random.default_rng(0).uniform(0, 200, (3, N_CELLS, CHUNK))
    root.create_array("rates", data=rates.astype(np.float32))
    calcium = np.arange(N_CELLS * 3 * FRAMES, dtype=np.float32)
    return path, calcium.reshape(N_CELLS, 3, FRAMES)


@pytest.fixture
def recording(tmp_path):
    """One long recording: rates (n_cells, total_time), stored in time chunks."""
    path = tmp_path / "recording.zarr"
    root = zarr.open_group(path, mode="w")
    root.attrs.update(dt_ms=DT, frame_rate_hz=FRAME_HZ)
    n_chunks = 7
    rates = np.random.default_rng(1).uniform(0, 200, (N_CELLS, n_chunks * CHUNK))
    array = root.create_array(
        "rates", shape=rates.shape, chunks=(N_CELLS, CHUNK), dtype=np.float32
    )
    array[:] = rates
    calcium = np.arange(N_CELLS * n_chunks * FRAMES, dtype=np.float32)
    return path, rates.astype(np.float32), calcium.reshape(N_CELLS, -1)


def test_windows(windows):
    path, calcium = windows
    dataset = RateCalciumFFDataset(path, calcium)
    assert (len(dataset), dataset.chunk_size, dataset.frames_per_chunk) == (3, 40, 4)
    item = dataset[1]
    assert item.input_spikes.shape == (1, CHUNK, N_CELLS)
    assert item.input_spikes.dtype == torch.bool
    torch.testing.assert_close(
        item.target_calcium, torch.from_numpy(calcium[:, 1, :].T[None])
    )
    with pytest.raises(ValueError, match="n_cells, n_windows, frames_per_chunk"):
        RateCalciumFFDataset(path, calcium[:, :2])


def test_parallel_streams(recording):
    path, rates, calcium = recording
    dataset = ParallelSegmentCalciumDataset(path, calcium, n_segments=3)
    # 7 chunks in 3 streams of 2; the last chunk is dropped
    assert (dataset.chunk_size, dataset.stream_len, dataset.n_dropped) == (40, 2, 1)
    assert dataset.whole_chunk_reads
    item = dataset[1]  # chunk 1 of each stream: chunks 1, 3, 5
    assert item.input_spikes.shape == (3, CHUNK, N_CELLS)
    for stream, chunk in enumerate((1, 3, 5)):
        window = slice(chunk * CHUNK, (chunk + 1) * CHUNK)
        torch.testing.assert_close(
            item.target_rate[stream], torch.from_numpy(rates[:, window].T * 1e-3)
        )
        frames = slice(chunk * FRAMES, (chunk + 1) * FRAMES)
        torch.testing.assert_close(
            item.target_calcium[stream], torch.from_numpy(calcium[:, frames].T)
        )


def test_parallel_streams_output_subset_and_override(recording):
    path, rates, calcium = recording
    override = np.full((2, rates.shape[1]), 0.5, dtype=np.float32)
    dataset = ParallelSegmentCalciumDataset(
        path, calcium, 2, output_idx=[0, 2], target_rate_override=override
    )
    item = dataset[0]
    assert item.input_spikes.shape == (2, CHUNK, N_CELLS)  # inputs stay full
    assert item.target_calcium.shape == (2, FRAMES, 2)
    assert torch.equal(item.target_rate, torch.full((2, CHUNK, 2), 0.5))


def test_calcium_loss_frame_averages_and_masks_nan():
    model = CalciumObservationModel(gain=0.3, tau=50.0, dt=DT)
    loss_fn = CalciumMSELoss(model)
    spikes = torch.rand(2, CHUNK, N_CELLS) < 0.1
    target = torch.randn(2, FRAMES, N_CELLS)
    target[0, 1, 2] = float("nan")

    loss = loss_fn(spikes, target)
    reference = CalciumObservationModel(gain=0.3, tau=50.0, dt=DT)(spikes)
    frames = reference.reshape(2, FRAMES, CHUNK // FRAMES, N_CELLS).mean(2)
    valid = torch.isfinite(target)
    torch.testing.assert_close(loss, ((frames - target)[valid] ** 2).mean())
    torch.testing.assert_close(loss_fn.last_prediction, frames)

    assert model.prev_c is not None
    loss_fn.reset_state()
    assert model.prev_c is None
    with pytest.raises(ValueError, match="not divisible"):
        loss_fn(spikes, target[:, :3])


def test_calcium_loss_state_survives_a_checkpoint():
    """The latent calcium is saved by ``loss_states`` and restored by
    ``load_loss_states``, so the next chunk renders as if never interrupted."""
    from wiretorch.snn_runners.chunks import load_loss_states, loss_states

    def loss():
        return CalciumMSELoss(CalciumObservationModel(gain=0.5, tau=50.0, dt=1.0))

    spikes = torch.rand(2, 40, 3, generator=torch.Generator().manual_seed(0)) < 0.1
    target = torch.zeros(2, 4, 3)
    first, second = loss(), loss()
    first(spikes, target)
    states = loss_states({"calcium": first})
    assert list(states) == ["calcium/calcium_model/prev_c"]
    load_loss_states({"calcium": second}, states)
    assert first(spikes, target) == second(spikes, target)
