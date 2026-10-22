"""Analysis statistics compute what their docstrings define."""

import numpy as np
import pytest

from wiretorch.analysis.firing_statistics import compute_spike_train_cv


def cv_by_definition(trains, dt):
    """std(ISI, ddof=1) / mean(ISI) per train; NaN with fewer than 3 spikes."""
    flat = np.moveaxis(trains, -1, -2).reshape(-1, trains.shape[-2])
    out = []
    for train in flat:
        times = np.flatnonzero(train) * dt
        isi = np.diff(times)
        out.append(isi.std(ddof=1) / isi.mean() if len(times) >= 3 else np.nan)
    return np.array(out).reshape(trains.shape[:-2] + trains.shape[-1:])


@pytest.mark.parametrize("shape", [(3, 500, 7), (2, 2, 500, 5)])
def test_spike_train_cv(shape):
    trains = np.random.default_rng(0).random(shape) < 0.02
    trains[..., 0] = False  # a silent neuron: NaN
    np.testing.assert_allclose(
        compute_spike_train_cv(trains, dt=0.5), cv_by_definition(trains, 0.5), rtol=1e-5
    )


def test_statistics_by_cell_type():
    from wiretorch.analysis.firing_statistics import (
        compute_cv_by_cell_type,
        compute_firing_rate_by_cell_type,
    )
    from wiretorch.analysis.voltage_statistics import (
        compute_membrane_potential_by_cell_type,
    )

    rng = np.random.default_rng(1)
    spikes = rng.random((2, 400, 6)) < 0.03
    spikes[:, :, 5] = False  # one silent neuron of type 1
    types = np.array([0, 0, 0, 1, 1, 1])
    voltages = rng.normal(-60, 2, (2, 400, 6))

    rates = compute_firing_rate_by_cell_type(spikes, types, duration=400.0)
    per_neuron = spikes.sum(axis=(0, 1)) / (0.4 * 2)
    assert rates[1]["n_silent_cells"] == 1
    assert np.isclose(rates[0]["mean_firing_rate_hz"], per_neuron[:3].mean())
    assert np.isclose(rates[0]["std_firing_rate_hz"], per_neuron[:3].std(ddof=1))

    cvs = compute_cv_by_cell_type(spikes, types)
    cv = np.nanmean(compute_spike_train_cv(spikes), axis=0)
    assert np.isclose(cvs[1]["mean_cv"], np.nanmean(cv[3:]))  # silent neuron skipped

    v = compute_membrane_potential_by_cell_type(voltages, types)
    means = voltages.mean(axis=(0, 1))
    assert np.isclose(v[1]["mean_of_means"], means[3:].mean())
    assert np.isclose(v[1]["std_of_means"], means[3:].std(ddof=1))
