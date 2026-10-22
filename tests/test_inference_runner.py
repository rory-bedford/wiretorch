"""``SNNInference`` stores exactly what running the model chunk by chunk produces."""

from collections import namedtuple

import numpy as np
import pytest
import torch
import zarr

from wiretorch.snn_runners import SNNInference
from networks import SCENARIOS

Batch = namedtuple("Batch", ["input_spikes", "rates"])
CHUNK = 40


def chunks(built, device):
    x = torch.from_numpy(built.input_spikes).to(device)
    rates = torch.rand(x.shape, generator=torch.Generator().manual_seed(0)).to(device)
    return [
        Batch(x[:, s : s + CHUNK], rates[:, s : s + CHUNK])
        for s in range(0, x.shape[1], CHUNK)
    ]


def direct(device, track=False):
    built = SCENARIOS["recurrent_frozen"].build(2)
    model = built.model.to(device)
    model.track_variables, model.track_batch_idx = track, None
    with torch.no_grad():
        outs = [model(b.input_spikes) for b in chunks(built, device)]
    if track:
        return {
            k: torch.cat([o[k] for o in outs], dim=1).cpu().numpy() for k in outs[0]
        }
    return {"spikes": torch.cat(outs, dim=1).cpu().numpy()}


def runner(device, tmp_path, mode, track=False):
    built = SCENARIOS["recurrent_frozen"].build(2)
    model = built.model.to(device)
    model.track_variables, model.track_batch_idx = track, None
    data = chunks(built, device)
    result = SNNInference(
        model=model,
        dataloader=data,
        device=device,
        output_mode=mode,
        zarr_path=tmp_path / "out.zarr" if mode == "zarr" else None,
        save_tracked_variables=track,
        max_chunks=len(data),
        progress_bar=False,
    ).run()
    if mode == "zarr":
        root = zarr.open_group(result["zarr_path"], mode="r")
        return {key: np.asarray(root[key]) for key in root.array_keys()}, data
    return result, data


@pytest.mark.parametrize("mode", ["memory", "zarr"])
def test_runner_stores_model_output(device, tmp_path, mode):
    stored, data = runner(device, tmp_path, mode)
    np.testing.assert_array_equal(
        stored["output_spikes"], direct(device)["spikes"] > 0.5
    )
    np.testing.assert_array_equal(
        stored["input_spikes"],
        torch.cat([b.input_spikes for b in data], 1).cpu().numpy(),
    )
    np.testing.assert_array_equal(
        stored["rates"], torch.cat([b.rates for b in data], 1).cpu().numpy()
    )


@pytest.mark.parametrize("mode", ["memory", "zarr"])
def test_runner_stores_tracked_variables(device, tmp_path, mode):
    stored, _ = runner(device, tmp_path, mode, track=True)
    expected = direct(device, track=True)
    for key in ("voltages", "currents", "currents_leak", "conductances"):
        np.testing.assert_array_equal(stored[key], expected[key], err_msg=key)
