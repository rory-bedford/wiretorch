"""The supervised datasets serve recorded spikes, chunk by chunk.

Uses a small synthetic ``spike_data.zarr``: exact chunks must equal the stored arrays;
with smoothing, inputs are Bernoulli samples from smoothed rates cached in the archive,
and a cache left incomplete by an interrupted write is rebuilt.
"""

import numpy as np
import pytest
import torch
import zarr

from connectome_snns.dataloaders.supervised import ExactFFDataset
from connectome_snns.dataloaders.supervised._smoothing import smooth_spike_trains

B, T, N_FF, N_REC, CHUNK = 2, 120, 6, 5, 30


@pytest.fixture
def archive(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "spike_data.zarr"
    root = zarr.open_group(path, mode="w")
    root.attrs["dt"] = 1.0
    for name, n in (("input_spikes", N_FF), ("output_spikes", N_REC)):
        root.create_array(name, data=rng.random((B, T, n)) < 0.1)
    return path


def test_exact_chunks(archive):
    root = zarr.open_group(archive, mode="r")
    dataset = ExactFFDataset(archive, CHUNK)
    assert len(dataset) == T // CHUNK and dataset.dt == 1.0
    for i in range(len(dataset)):
        item = dataset[i]
        window = slice(i * CHUNK, (i + 1) * CHUNK)
        np.testing.assert_array_equal(
            item.input_spikes, root["input_spikes"][:, window]
        )
        np.testing.assert_array_equal(
            item.target_spikes, root["output_spikes"][:, window]
        )
        assert item.recurrent_input_spikes is None


@pytest.mark.parametrize("which", ["recurrent", "ff"])
def test_smoothed_inputs_are_samples_of_cached_rates(archive, which):
    root = zarr.open_group(archive, mode="r")
    source = root["output_spikes" if which == "recurrent" else "input_spikes"]
    kwargs = {f"{'recurrent' if which == 'recurrent' else 'ff'}_smoothing_tau": 5.0}
    dataset = ExactFFDataset(archive, CHUNK, **kwargs)

    rates = smooth_spike_trains(
        torch.from_numpy(source[:].astype(np.float32)), 1.0, 5.0
    )
    rates = rates.numpy().astype(np.float16).astype(np.float32)
    torch.manual_seed(3)
    item = dataset[1]
    sampled = item.recurrent_input_spikes if which == "recurrent" else item.input_spikes
    torch.manual_seed(3)
    expected = torch.rand(B, CHUNK, source.shape[2]) < torch.from_numpy(
        rates[:, CHUNK : 2 * CHUNK]
    )
    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(
        item.target_spikes, root["output_spikes"][:, CHUNK : 2 * CHUNK]
    )


def test_cache_is_reused_and_incomplete_cache_rebuilt(archive, capsys):
    ExactFFDataset(archive, CHUNK, recurrent_smoothing_tau=5.0)
    key = "smoothed_rates_tau5.0"
    first = zarr.open_group(archive, mode="r")[key][:]

    ExactFFDataset(archive, CHUNK, recurrent_smoothing_tau=5.0)
    assert "Loaded cached" in capsys.readouterr().out

    # Simulate an interrupted write: zeroed tail, no completion marker.
    root = zarr.open_group(archive, mode="r+")
    root[key][:, T // 2 :] = 0
    root[key].attrs.pop("complete", None)
    ExactFFDataset(archive, CHUNK, recurrent_smoothing_tau=5.0)
    assert "recomputing" in capsys.readouterr().out
    np.testing.assert_array_equal(zarr.open_group(archive, mode="r")[key][:], first)


@pytest.mark.parametrize("alpha", [1 - 1 / 700000, 0.9, 0.5])
def test_ar1_trajectory_matches_the_recurrence(alpha):
    """Including short time constants, where 1 / alpha^T would overflow float32."""
    from connectome_snns.utils.recurrences import ar1_trajectory

    torch.manual_seed(0)
    s0, noise = torch.randn(3, 4), 0.1 * torch.randn(3, 2500, 4)
    states, final = ar1_trajectory(s0, noise, alpha)

    s, expected = s0.double(), []
    for t in range(noise.shape[1]):
        expected.append(s)
        s = alpha * s + noise[:, t].double()
    assert torch.isfinite(states).all()
    torch.testing.assert_close(
        states.double(), torch.stack(expected, 1), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(final.double(), s, rtol=1e-4, atol=1e-5)


def test_collates():
    from connectome_snns.dataloaders.supervised import SpikeData
    from connectome_snns.dataloaders.supervised.collate import (
        FeedforwardCollate,
        VisibleSubsetCollate,
        single_neuron_collate_fn,
    )

    ff, target, resampled = (
        torch.zeros(2, 4, 3),
        torch.ones(2, 4, 5),
        torch.full((2, 4, 5), 2.0),
    )
    plain, smooth = SpikeData(ff, target), SpikeData(ff, target, resampled)
    assert torch.equal(
        FeedforwardCollate()(plain).input_spikes, torch.cat([ff, target], 2)
    )
    assert torch.equal(
        FeedforwardCollate()(smooth).input_spikes, torch.cat([ff, resampled], 2)
    )
    sub = VisibleSubsetCollate(torch.tensor([0, 2]))(smooth)
    assert sub.input_spikes.shape == (2, 4, 5) and sub.target_spikes.shape == (2, 4, 2)
    assert single_neuron_collate_fn(plain).target_spikes.shape == (2, 4, 1)
