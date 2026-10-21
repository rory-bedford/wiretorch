"""Lockstep: from every recorded state, the simulator takes the recorded steps.

This is the suite's definition of "the same simulator". The network is restarted from
the recorded state every ``WINDOW`` steps and run for one window, so floating-point
differences cannot compound through the chaotic dynamics. Within each window:

* spikes must match exactly — except a neuron whose voltage, when it spiked or failed
  to, was within ``THRESHOLD_MARGIN`` of threshold, where either outcome is a legitimate
  rounding of the same equations (that trial's window is excused from then on);
* voltages, and the state at the end of the window, must match to a tolerance.

It runs on every device, so it is also what checks GPU kernels against the CPU.
"""

import numpy as np
import pytest
import torch

import fixture_io
import harness
from networks import BATCH_SIZE, SCENARIOS, SPIKING

THRESHOLD_MARGIN = 1e-3  # mV
TOLERANCE = {  # (voltage atol in mV, state rtol, state atol)
    "cpu": (1e-4, 1e-4, 1e-5),
    "cuda": (1e-3, 1e-3, 1e-4),
}
LOCKSTEP = [n for n in SPIKING if SCENARIOS[n].state_keys]


@pytest.mark.parametrize("name", LOCKSTEP)
def test_lockstep(name, device):
    scenario = SCENARIOS[name]
    recorded, _ = fixture_io.load(name)
    built = scenario.build(BATCH_SIZE)
    model = built.model.to(device)
    model.track_variables, model.track_batch_idx = True, None
    theta = model.theta.cpu().numpy()
    x = torch.from_numpy(built.input_spikes).to(device)
    v_atol, s_rtol, s_atol = TOLERANCE[device]

    excused = []
    for w, start in enumerate(range(0, x.shape[1], harness.WINDOW)):
        stop = start + harness.WINDOW
        model.load_checkpoint_state(
            {k: recorded[f"state.{k}"][w] for k in scenario.state_keys}
        )
        with torch.no_grad():
            out = harness.unpack_output(model(x[:, start:stop]))
        spikes = out["spikes"].cpu().numpy() > 0.5
        voltages = out["voltages"].cpu().numpy()

        # Voltage each spike decision was made from: the state entering each step.
        v_enter = np.concatenate(
            [
                recorded["state.v"][w][:, None],
                recorded["voltages"][:, start : stop - 1],
            ],
            axis=1,
        )
        ok = np.ones(BATCH_SIZE, dtype=bool)
        for b, t, n in zip(*np.nonzero(spikes != recorded["spikes"][:, start:stop])):
            if not ok[b]:
                continue
            margin = abs(v_enter[b, t, n] - theta[n])
            assert margin < THRESHOLD_MARGIN, (
                f"window {w}, step {start + t}, trial {b}, neuron {n}: spike "
                f"{'emitted' if spikes[b, t, n] else 'missed'} with the voltage "
                f"{margin:.2e} mV from threshold"
            )
            ok[b] = False
            excused.append((w, b, start + t, n))

        np.testing.assert_allclose(
            voltages[ok],
            recorded["voltages"][ok, start:stop],
            rtol=0,
            atol=v_atol,
            err_msg=f"voltages, window {w}",
        )
        for key, value in model.get_checkpoint_state().items():
            np.testing.assert_allclose(
                value[ok],
                recorded[f"state.{key}"][w + 1][ok],
                rtol=s_rtol,
                atol=s_atol,
                err_msg=f"state {key!r} at the end of window {w}",
            )
        if scenario.returns_probabilities:
            np.testing.assert_allclose(
                out["probabilities"].cpu().numpy()[ok],
                recorded["probabilities"][ok, start:stop],
                rtol=0,
                atol=10 * s_atol,
                err_msg=f"probabilities, window {w}",
            )

    n_windows = x.shape[1] // harness.WINDOW
    assert len(excused) <= 0.01 * n_windows * BATCH_SIZE, (
        f"{len(excused)} near-threshold flips in {n_windows * BATCH_SIZE} trial-windows"
    )
