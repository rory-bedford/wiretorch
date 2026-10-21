"""Surrogate gradients match the recorded ones.

The loss is a fixed random readout of the spikes (``harness.readout_loss``) over the
first ``GRAD_STEPS`` steps, so the gradient reaches every trainable parameter through
the full backward pass through time, and nothing else.
"""

import numpy as np
import pytest

import fixture_io
import harness
from connectome_snns.network_simulators.network import ProjectionNetwork
from networks import BATCH_SIZE, SCENARIOS

TRAINABLE = [
    n for n in SCENARIOS if n != "recurrent_frozen"
]  # the rest have parameters
TOLERANCE = {"cpu": 1e-4, "cuda": 2e-3}  # relative, against each gradient's scale


@pytest.mark.parametrize("track", [False, True], ids=["plain", "tracking"])
@pytest.mark.parametrize("name", TRAINABLE)
def test_gradients_match(name, device, track):
    recorded, _ = fixture_io.load(name)
    loss, grads = harness.gradients(SCENARIOS[name].build(BATCH_SIZE), device, track)

    expected = {
        k.removeprefix("grad."): v for k, v in recorded.items() if k.startswith("grad.")
    }
    assert grads.keys() == expected.keys()
    np.testing.assert_allclose(loss, recorded["loss"], rtol=TOLERANCE[device])
    for key, value in expected.items():
        scale = np.abs(value).max() or 1.0
        np.testing.assert_allclose(
            grads[key], value, rtol=0, atol=TOLERANCE[device] * scale, err_msg=key
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "name",
    [
        "recurrent_scaling_factors",
        "feedforward_scaling_factors",
        "current_feedforward",
    ],
)
def test_float16_storage_keeps_the_gradient(name):
    """Synaptic state and recurrent weights stored in float16: the same gradient.

    On these scenarios float16 flips no spike in the gradient window, so the gradient
    differs only by rounding. (In a network this small one flipped spike changes the
    gradient wholesale, as ``two_layer`` and ``current_recurrent`` show — natural units
    leave a float16 current ~1e-3 of threshold, and that network first flips a spike at
    step 12; on a 5000-neuron conductance-based network the cosine is 0.9999. Recurrent float16
    runs are checked statistically, in ``test_statistics.py``.)
    """
    recorded, _ = fixture_io.load(name)
    built = SCENARIOS[name].build(BATCH_SIZE)
    for module in built.model.modules():
        if isinstance(module, ProjectionNetwork):
            module.storage_dtype = "float16"
    _, grads = harness.gradients(built, "cuda")
    flat = np.concatenate([grads[k].ravel() for k in sorted(grads)])
    expected = np.concatenate([recorded[f"grad.{k}"].ravel() for k in sorted(grads)])
    cosine = flat @ expected / (np.linalg.norm(flat) * np.linalg.norm(expected))
    assert cosine > 0.999, cosine
