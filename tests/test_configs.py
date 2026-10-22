"""Layer configs derive the same parameter tables as when the fixture was recorded
(less the refractory period ``tau_ref``, removed since: it was never simulated)."""

import json

import fixture_io
from fixtures.generate import config_outputs


def test_layer_config_outputs():
    recorded = json.loads((fixture_io.FIXTURES / "configs.json").read_text())["outputs"]
    for cell in recorded["recurrent.get_cell_params"]:
        del cell["tau_ref"]
    assert config_outputs() == recorded


def test_old_configs_with_tau_ref_still_parse():
    from wiretorch.configs.conductance_based import PhysiologyConfig

    config = PhysiologyConfig(
        tau_mem=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0, tau_ref=2.0
    )
    assert "tau_ref" not in config.model_dump()


def test_unknown_keys_are_errors():
    import pytest
    from pydantic import ValidationError

    from wiretorch.configs import SimulationConfig
    from wiretorch.configs.conductance_based import PhysiologyConfig

    with pytest.raises(ValidationError, match="tau_memm"):
        PhysiologyConfig(
            tau_mem=20.0, tau_memm=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0
        )
    with pytest.raises(ValidationError, match="sead"):
        SimulationConfig(dt=1.0, duration=100.0, seed=1, sead=2)


def _current_network(device="cpu", spikes_per_step=0.0, weight=0.02):
    """A current-based network built from its configs, in physical units (mV, MOhm)."""
    import numpy as np
    import torch

    from wiretorch.configs.current_based import (
        FeedforwardLayerConfig,
        RecurrentLayerConfig,
    )
    from wiretorch.network_simulators.current import CurrentLIFNetwork
    from wiretorch.network_simulators.projections import FrozenProjection

    physiology = dict(tau_mem=20.0, tau_syn=5.0, R=50.0, U_rest=-70.0, theta=-50.0, U_reset=-75.0)  # fmt: skip
    rec = RecurrentLayerConfig(
        cell_types=dict(names=["excitatory", "inhibitory"], proportion=[0.5, 0.5]),
        physiology={"excitatory": physiology, "inhibitory": {**physiology, "tau_syn": 10.0}},
    )  # fmt: skip
    ff = FeedforwardLayerConfig(
        cell_types=dict(names=["input"], proportion=[1.0]),
        physiology={"input": physiology},
    )
    types, ff_types = np.array([0, 1, 0, 1]), np.zeros(3, dtype=np.int64)
    names = rec.cell_types.names
    model = CurrentLIFNetwork(
        dt=1.0,
        rec_projections={
            (a, b): FrozenProjection(np.zeros((2, 2), np.float32)) for a in names for b in names
        },
        ff_projections={
            ("input", b): FrozenProjection(np.full((3, 2), weight, np.float32)) for b in names
        },
        cell_type_indices=types,
        cell_type_indices_FF=ff_types,
        cell_params=rec.get_cell_params(),
        cell_params_FF=ff.get_cell_params(),
        synapse_params=rec.get_synapse_params(),
        synapse_params_FF=ff.get_synapse_params(),
        surrgrad_scale=5.0,
        batch_size=1,
        track_variables=True,
    ).to(device)  # fmt: skip
    x = torch.full((1, 400, 3), spikes_per_step, device=device)
    return model, x


def test_current_configs_build_a_network_in_physical_units():
    import numpy as np
    import torch

    model, x = _current_network()
    assert [s["type"] for s in model.synapse_params] == ["excitatory", "inhibitory"]
    with torch.no_grad():
        out = model(x)
    # No input: the voltage relaxes from reset (-75) to rest (-70) and never fires.
    assert out["spikes"].sum() == 0
    np.testing.assert_allclose(out["voltages"][0, -1].numpy(), -70.0, atol=1e-3)

    # Constant input below threshold: v settles at U_rest + R * I, where each step's
    # 3 inputs x 0.1 spikes x weight 0.002 accumulate in a 5 ms current.
    model, x = _current_network(spikes_per_step=0.1, weight=0.002)
    with torch.no_grad():
        out = model(x)
    current = 3 * 0.1 * 0.002 / -np.expm1(-1.0 / 5.0)
    assert out["spikes"].sum() == 0
    np.testing.assert_allclose(
        out["voltages"][0, -1].numpy(), -70.0 + 50.0 * current, atol=1e-3
    )
