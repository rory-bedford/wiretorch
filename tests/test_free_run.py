"""Free runs reproduce the recorded spike trains, spike for spike.

A network run from rest on the recorded input must emit exactly the recorded spikes for
all 200 steps. In a chaotic network one flipped spike spreads to everything, so this is
the strictest test in the suite; it holds on the CPU, where arithmetic changes are tiny,
but is not asked of the GPU (see ``test_lockstep.py`` for the device-independent check).
"""

import numpy as np
import pytest

import fixture_io
import harness
from networks import BATCH_SIZE, SCENARIOS, SPIKING


def first_divergence(a, b):
    """First time step at which two ``(batch, time, neurons)`` spike trains differ."""
    differs = (a != b).any(axis=(0, 2))
    return int(np.argmax(differs)) if differs.any() else None


@pytest.mark.parametrize("name", SPIKING)
def test_spikes_match_exactly(name):
    scenario = SCENARIOS[name]
    recorded, _ = fixture_io.load(name)
    run = harness.free_run(scenario, scenario.build(BATCH_SIZE))

    for key in ("spikes", "hidden_spikes"):
        if key in recorded:
            step = first_divergence(run[key], recorded[key])
            assert step is None, (
                f"{key} diverge from the recording at step {step} of "
                f"{recorded[key].shape[1]} "
                f"({int((run[key] != recorded[key]).sum())} spikes differ in total)"
            )


@pytest.mark.parametrize(
    "name", [n for n in SPIKING if harness.tracks_voltages(SCENARIOS[n])]
)
def test_voltages_match(name):
    scenario = SCENARIOS[name]
    recorded, _ = fixture_io.load(name)
    run = harness.free_run(scenario, scenario.build(BATCH_SIZE))
    # 1e-5 of the voltage range: ~0.3 uV in mV units, 1e-5 in natural units.
    atol = 1e-5 * np.ptp(recorded["voltages"])
    np.testing.assert_allclose(run["voltages"], recorded["voltages"], rtol=0, atol=atol)
    if scenario.returns_probabilities:
        np.testing.assert_allclose(
            run["probabilities"], recorded["probabilities"], rtol=0, atol=1e-5
        )
