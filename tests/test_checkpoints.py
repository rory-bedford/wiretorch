"""Checkpoints written before the rewrite still load, with the same values.

``state_dicts.pt`` holds every scenario's ``state_dict`` as master's simulators wrote
it (``generate.py state_dicts``). The rewritten simulators must accept each one
strictly — the same keys, shapes and values (floats to a few ulps) — so existing runs
can be resumed. Buffers since removed (``LEGACY_BUFFERS``) are dropped on load.

The current-based networks have since moved from weight matrices to projections, so
their former checkpoints have other keys: they are converted on load
(``current.legacy``) into exactly the network ``from_weights`` builds.
"""

import pytest
import torch

import fixture_io
from wiretorch.network_simulators.checkpoint_compat import LEGACY_BUFFERS
from networks import (
    BATCH_SIZE,
    SCENARIOS,
    SEED,
    current_recurrent,
)

RECORDED = torch.load(fixture_io.FIXTURES / "state_dicts.pt", weights_only=False)


FORMER_CURRENT = ["current_recurrent", "current_feedforward"]


@pytest.mark.parametrize(
    "name", [n for n in RECORDED["states"] if n not in FORMER_CURRENT]
)
def test_old_state_dict_loads(name):
    recorded = RECORDED["states"][name]
    kept = {k: v for k, v in recorded.items() if k.split(".")[-1] not in LEGACY_BUFFERS}
    model = SCENARIOS[name].build(BATCH_SIZE).model
    current = model.state_dict()
    assert sorted(current) == sorted(kept), (
        f"missing: {sorted(set(kept) - set(current))}, "
        f"new: {sorted(set(current) - set(kept))}"
    )
    for key, value in kept.items():
        assert current[key].dtype == value.dtype, key
        # Floats to a few ulps: values derived with transcendental functions (log
        # weights) round differently on other CPUs' vector units.
        exact = not value.is_floating_point()
        torch.testing.assert_close(
            current[key], value, rtol=0 if exact else 1e-6, atol=0, msg=key
        )
    model.load_state_dict(recorded, strict=True)


@pytest.mark.parametrize("name", FORMER_CURRENT)
def test_former_current_state_dict_converts(name):
    if name == "current_recurrent":  # the checkpoint predates the current regime
        model = current_recurrent(BATCH_SIZE, SEED, former=True).model
    else:
        model = SCENARIOS[name].build(BATCH_SIZE).model
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)  # so the load has to restore them
    model.load_state_dict(RECORDED["states"][name], strict=True)
    loaded = model.state_dict()
    assert sorted(loaded) == sorted(expected)
    for key, value in expected.items():
        exact = not value.is_floating_point()
        torch.testing.assert_close(
            loaded[key], value, rtol=0 if exact else 1e-6, atol=0, msg=key
        )


@pytest.mark.parametrize(
    "former, name, now",
    [
        ("conductance_based.simulator", "ConductanceLIFNetwork", "conductance"),
        ("conductance_based", "ConductanceLIFNetwork", "conductance"),
        (
            "feedforward_conductance_based.simulator",
            "FeedforwardConductanceLIFNetwork",
            "conductance",
        ),
        (
            "feedforward_conductance_based",
            "FeedforwardConductanceLIFNetwork",
            "conductance",
        ),
        ("current_based.simulator", "CurrentLIFNetwork", "current"),
        ("current_based", "CurrentLIFNetwork", "current"),
        ("feedforward_current_based", "FeedforwardCurrentLIFNetwork", "current"),
    ],
)
def test_former_network_import_paths(former, name, now):
    """Experiments (and pickles) naming the networks' former modules still find them."""
    import importlib

    package = "wiretorch.network_simulators"
    old = importlib.import_module(f"{package}.{former}")
    assert getattr(old, name) is getattr(
        importlib.import_module(f"{package}.{now}"), name
    )


def test_former_current_base_class_name():
    from wiretorch.network_simulators.current.network import CurrentNetwork
    from wiretorch.network_simulators.current_based import CurrentLIFNetwork_IO

    assert CurrentLIFNetwork_IO is CurrentNetwork
