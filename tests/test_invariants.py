"""Properties of the simulator that hold exactly, whatever its implementation.

These compare the simulator with itself, so they need no fixture and must hold spike
for spike on every device:

* splitting a simulation into chunks does not change it;
* each trial of a batch evolves independently of the others;
* ``reset_state`` really does restart the simulation;
* turning on variable tracking does not change the dynamics;
* the per-neuron threshold is read at run time (edits to it in place take effect).
"""

import numpy as np
import pytest
import torch

import harness
from networks import BATCH_SIZE, SCENARIOS, SPIKING


def run(model, x, chunk):
    with torch.no_grad():
        out = [
            harness.unpack_output(model(x[:, s : s + chunk]))["spikes"]
            for s in range(0, x.shape[1], chunk)
        ]
    return torch.cat(out, dim=1).cpu().numpy() > 0.5


def build(name, device, batch_size=BATCH_SIZE):
    built = SCENARIOS[name].build(batch_size)
    return built.model.to(device), torch.from_numpy(built.input_spikes).to(device)


@pytest.mark.parametrize("name", SPIKING)
def test_chunking_does_not_matter(name, device):
    model, x = build(name, device)
    whole = run(model, x, x.shape[1])
    for chunk in (1, 7, 64):
        model.reset_state()
        np.testing.assert_array_equal(
            run(model, x, chunk), whole, err_msg=f"chunk {chunk}"
        )


@pytest.mark.parametrize("name", SPIKING)
def test_trials_are_independent(name, device):
    model, x = build(name, device)
    together = run(model, x, x.shape[1])
    for b in range(BATCH_SIZE):
        alone, _ = build(name, device, batch_size=1)
        np.testing.assert_array_equal(
            run(alone, x[b : b + 1], x.shape[1])[0], together[b], err_msg=f"trial {b}"
        )


@pytest.mark.parametrize("name", SPIKING)
def test_reset_restarts(name, device):
    model, x = build(name, device)
    first = run(model, x, 50)
    model.reset_state()
    np.testing.assert_array_equal(run(model, x, 50), first)


def assert_tracks_the_trials_asked_for(model, x):
    """Tracked variables hold trial ``track_batch_idx``, or every trial for None."""
    for index, trials in ((0, 1), (None, BATCH_SIZE)):
        model.track_variables, model.track_batch_idx = True, index
        with torch.no_grad():
            tracked = model(x[:, :10])
        for key, value in tracked.items():
            if isinstance(value, torch.Tensor) and not key.endswith(
                ("spikes", "probabilities")
            ):
                assert value.shape[0] == trials, key


def test_two_layer_tracks_the_trials_asked_for(device):
    assert_tracks_the_trials_asked_for(*build("two_layer", device))


@pytest.mark.parametrize("name", [n for n in SPIKING if SCENARIOS[n].state_keys])
def test_tracking_does_not_change_dynamics(name, device):
    model, x = build(name, device)
    plain = run(model, x, 50)
    model.reset_state()
    model.track_variables, model.track_batch_idx = True, 0
    np.testing.assert_array_equal(run(model, x, 50), plain)
    assert_tracks_the_trials_asked_for(model, x)


@pytest.mark.parametrize(
    "name",
    [
        "recurrent_frozen",
        "feedforward_scaling_factors",
        "current_recurrent",
        "current_feedforward",
    ],
)
def test_threshold_is_read_at_run_time(name, device):
    model, x = build(name, device)
    active = run(model, x, x.shape[1]).any(axis=(0, 1))
    assert active.any()
    model.reset_state()
    model.theta[torch.from_numpy(active).to(device)] += 1000.0  # unreachable
    assert not run(model, x, x.shape[1])[:, :, active].any()


def test_scaling_factor_is_applied_once(device):
    """A scaling factor set at construction scales the weights once, not twice."""
    from wiretorch.network_simulators.projections import ScalingFactorProjection

    def spikes(factor):
        model, x = build("feedforward_scaling_factors", device)
        for projection in model.projections.values():
            projection.log_sf.data.fill_(0.0)
        built = type(model)(
            dt=1.0,
            projections={
                tuple(key.split("__")): ScalingFactorProjection(
                    p.connectome.cpu() * (1.0 if factor else 1.5),
                    init_sf=1.5 if factor else 1.0,
                )
                for key, p in model.projections.items()
            },
            cell_type_indices=model.cell_type_indices.cpu().numpy(),
            cell_type_indices_FF=model.cell_type_indices_FF.cpu().numpy(),
            cell_params=model.cell_params,
            cell_params_FF=model.cell_params_FF,
            synapse_params_FF=model.synapse_params_FF,
            surrgrad_scale=5.0,
            batch_size=BATCH_SIZE,
        ).to(device)
        return run(built, x, x.shape[1])

    np.testing.assert_array_equal(spikes(factor=True), spikes(factor=False))


def test_storage_dtype_is_validated_and_the_reference_ignores_it():
    """Only the kernels store in reduced precision; the PyTorch reference stays float32."""
    model, x = build("recurrent_scaling_factors", "cpu")
    reference = run(model, x, x.shape[1])
    model.reset_state()
    model.storage_dtype = "float16"
    assert model.storage_dtype is torch.float16
    np.testing.assert_array_equal(run(model, x, x.shape[1]), reference)
    with pytest.raises(ValueError, match="storage_dtype"):
        model.storage_dtype = torch.float64
