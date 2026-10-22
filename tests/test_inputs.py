"""Input generation: assembly rate patterns, their config, and the Poisson loaders.

Also checks that the names older experiments import (``configs.odours``,
``dataloaders.odourants``) still resolve to the same behaviour.
"""

import numpy as np
import pytest
import torch

from wiretorch.configs.input_patterns import (
    InputPatternConfig,
    load_input_patterns,
)
from wiretorch.dataloaders.input_patterns import (
    generate_assembly_firing_rates,
    generate_baseline_firing_rates,
)
from wiretorch.dataloaders.unsupervised import (
    HomogeneousPoissonSpikeDataLoader,
)

N_IN, N_REC = 200, 50
CONFIG = InputPatternConfig(
    baseline_rate=6.0, modulation_rate=9.0, modulation_fraction=0.1
)


def _patterns(**overrides):
    rng = np.random.default_rng(0)
    kwargs = dict(
        feedforward_weights=rng.random((N_IN, N_REC)),
        input_source_indices=np.zeros(N_IN, dtype=int),
        cell_type_indices=np.zeros(N_REC, dtype=int),
        assembly_ids=np.repeat([0, 1], N_REC // 2),
        target_cell_type_idx=0,
        cell_type_names=["input"],
        input_configs={"input": CONFIG.to_dict()},
    )
    kwargs.update(overrides)
    return kwargs, generate_assembly_firing_rates(**kwargs)


def test_config_matches_the_generator():
    _, rates = _patterns()
    up, down = CONFIG.get_modulated_rates(N_IN)
    for pattern in rates:
        assert np.sum(np.isclose(pattern, up)) == CONFIG.get_n_modulated(N_IN)
        assert np.allclose(pattern[~np.isclose(pattern, up)], down)
        assert np.isclose(pattern.mean(), 6.0)


def test_strongest_inputs_to_each_assembly_are_modulated_up():
    kwargs, rates = _patterns()
    weights, assemblies = kwargs["feedforward_weights"], kwargs["assembly_ids"]
    up, _ = CONFIG.get_modulated_rates(N_IN)
    for assembly, pattern in enumerate(rates):
        strength = weights[:, assemblies == assembly].sum(axis=1)
        strongest = np.argsort(-strength)[: CONFIG.get_n_modulated(N_IN)]
        assert np.all(np.isclose(pattern[strongest], up))


def test_baseline_rates():
    types = np.repeat([0, 1], [3, 2])
    configs = {"a": {"baseline_rate": 2.0}, "b": {"baseline_rate": 5.0}}
    np.testing.assert_array_equal(
        generate_baseline_firing_rates(5, types, ["a", "b"], configs),
        [[2.0, 2.0, 2.0, 5.0, 5.0]],
    )
    with pytest.raises(ValueError, match="'c'"):
        generate_baseline_firing_rates(5, types, ["a", "c"], configs)


@pytest.mark.parametrize("table", ["input_patterns", "odours"])
def test_load_input_patterns(table):
    params = {
        table: {"tau": 7.0, "temperature": 0.2, "sigma": 5.0, "input": CONFIG.to_dict()}
    }
    rate_process, configs = load_input_patterns(params)
    assert rate_process == {"tau": 7.0, "temperature": 0.2, "sigma": 5.0}
    assert configs == {"input": CONFIG}


def test_legacy_names_still_import_and_agree():
    from wiretorch.configs.odours import OdourInputConfig
    from wiretorch.dataloaders.odourants import (
        generate_baseline_firing_rates as legacy_baseline,
        generate_odour_firing_rates,
    )

    assert OdourInputConfig is InputPatternConfig
    kwargs, rates = _patterns()
    legacy = dict(kwargs)
    legacy["odour_configs"] = legacy.pop("input_configs")
    np.testing.assert_array_equal(generate_odour_firing_rates(**legacy), rates)
    configs = {"a": {"baseline_rate": 2.0}}
    np.testing.assert_array_equal(
        legacy_baseline(3, np.zeros(3, int), ["a"], odour_configs=configs),
        generate_baseline_firing_rates(3, np.zeros(3, int), ["a"], configs),
    )


@pytest.mark.parametrize("rates", [np.full(7, 50.0), np.full((3, 7), 50.0)])
def test_homogeneous_loader_shapes_and_patterns(rates):
    loader = HomogeneousPoissonSpikeDataLoader(
        rates, chunk_size=20.0, dt=1.0, batch_size=4
    )
    spikes, patterns = next(iter(loader))
    if rates.ndim == 1:
        assert spikes.shape == (4, 20, 7) and patterns == [0]
    else:
        assert spikes.shape == (4, 3, 20, 7) and patterns == [0, 1, 2]
    assert spikes.dtype == torch.bool
    assert 0.0 < spikes.float().mean() < 0.15  # p = 0.05 per step
