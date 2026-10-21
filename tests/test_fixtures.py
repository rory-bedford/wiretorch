"""The fixtures still describe the networks the tests build.

If a rebuilt network's fingerprint differs from the one recorded, every comparison
below it would be against the wrong network — so fail here, loudly, first.
"""

import pytest

import fixture_io
from networks import BATCH_SIZE, SCENARIOS, SEED, fingerprint


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_fingerprint(name):
    _, meta = fixture_io.load(name)
    assert fingerprint(SCENARIOS[name].build(BATCH_SIZE)) == meta["fingerprint"], (
        f"{name}: the network or its input no longer matches the recorded fixture "
        "(a builder in tests/networks.py changed, or numpy changed its random streams)"
    )


@pytest.mark.parametrize("fixture", ["statistics", "statistics_current"])
def test_statistics_fingerprints(fixture):
    from fixtures.generate import STATS_BATCH, STATS_NETWORKS

    _, meta = fixture_io.load(fixture)
    for key, recorded in meta["fingerprints"].items():
        name, input_seed = key.split(".")
        builder, kwargs, n_steps = STATS_NETWORKS[name]
        built = builder(
            STATS_BATCH, SEED, n_steps=n_steps, input_seed=int(input_seed), **kwargs
        )
        assert fingerprint(built) == recorded, key


@pytest.mark.parametrize(
    "name", [*SCENARIOS, "training", "statistics", "statistics_current"]
)
def test_recorded_from_clean_library(name):
    _, meta = fixture_io.load(name)
    assert not meta["library_dirty"], f"{name} was recorded from uncommitted code"
