"""The conductance-based voltage step: exponential Euler.

With the conductances held over a step, ``C_m dv/dt = -g_tot (v - V_inf)`` is linear,
and the step solves it exactly. Checked here: against that solution, that it is stable
however large the conductances (no overshoot past ``V_inf``, no ``g_clip`` needed), and
that the kernels take the same step.
"""

import math

import numpy as np
import pytest
import torch

from wiretorch.network_simulators.conductance.dynamics import Physiology, membrane
from wiretorch.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from wiretorch.network_simulators.projections import FrozenProjection

TAU, G_L, E_L, DT = 20.0, 2.0, -65.0, 0.5
E_SYN = torch.tensor([0.0, -80.0])


def _physiology(g_clip=math.inf):
    n = torch.ones(1)
    return Physiology(
        theta=n * 1e9, U_reset=n * E_L, E_L=n * E_L, C_m=n * TAU * G_L,
        beta=n * math.exp(-DT / TAU), E_syn=E_SYN, g_clip=torch.full((2,), g_clip),
        decay=torch.ones(2, 2), jump=torch.zeros(2, 2), dt=torch.tensor(DT),
        surr_scale=torch.tensor(1.0),
    )  # fmt: skip


@pytest.mark.parametrize("G", [(0.0, 0.0), (3.0, 1.0), (1e6, 0.0), (2e5, 5e5)])
def test_step_is_the_exact_solution(G):
    """``v`` relaxes towards ``V_inf`` by ``exp(-dt g_tot / C_m)``: for small
    conductances as for ones ``10^5`` times the leak, where forward Euler would
    overshoot by far."""
    v = torch.tensor([[-50.0]])
    g = torch.zeros(1, 1, 2, 2)
    g[0, 0, 1] = torch.tensor(G)
    v_next, I, I_leak = membrane(v, torch.zeros(1, 1), g, _physiology())
    g_tot = G_L + sum(G)
    V_inf = (G_L * E_L + sum(gk * e for gk, e in zip(G, E_SYN.tolist()))) / g_tot
    expected = V_inf + (-50.0 - V_inf) * math.exp(-DT * g_tot / (TAU * G_L))
    assert v_next.item() == pytest.approx(expected, rel=1e-6, abs=1e-4)
    assert I_leak.item() == pytest.approx(G_L * (-50.0 - E_L))
    lo, hi = sorted([-50.0, V_inf])
    assert lo - 1e-4 <= v_next.item() <= hi + 1e-4  # never past V_inf


def test_no_conductance_is_the_leak():
    v = torch.tensor([[-50.0]])
    v_next, _, _ = membrane(
        v, torch.zeros(1, 1), torch.zeros(1, 1, 2, 2), _physiology()
    )
    assert v_next.item() == pytest.approx(E_L + 15.0 * math.exp(-DT / TAU), rel=1e-6)


def test_huge_synapses_stay_between_reversal_potentials(device):
    """Without ``g_clip``, inputs strong enough to send a forward-Euler step hundreds of
    mV past the reversal potentials: every voltage stays within ``[E_inh, E_exc]``."""
    rng = np.random.default_rng(0)
    model = FeedforwardConductanceLIFNetwork(
        dt=1.0,
        projections={
            ("e", "n"): FrozenProjection(np.full((5, 20), 50.0)),
            ("i", "n"): FrozenProjection(np.full((5, 20), 50.0)),
        },
        cell_type_indices=np.zeros(20, dtype=np.int64),
        cell_type_indices_FF=np.repeat([0, 1], 5),
        cell_params=[
            dict(
                name="n",
                cell_id=0,
                tau_mem=20.0,
                theta=-50.0,
                U_reset=-65.0,
                E_L=-65.0,
                g_L=1.0,
            )
        ],  # fmt: skip
        cell_params_FF=[dict(name="e", cell_id=0), dict(name="i", cell_id=1)],
        synapse_params_FF=[
            dict(
                name="AMPA",
                synapse_id=0,
                cell_id=0,
                tau_rise=0.1,
                tau_decay=5.0,
                E_syn=0.0,
                g_bar=40.0,
            ),
            dict(
                name="GABA",
                synapse_id=1,
                cell_id=1,
                tau_rise=0.1,
                tau_decay=8.0,
                E_syn=-80.0,
                g_bar=40.0,
            ),
        ],  # fmt: skip
        surrgrad_scale=5.0,
        batch_size=3,
        track_variables=True,
    ).to(device)
    x = torch.from_numpy((rng.random((3, 200, 10)) < 0.2).astype(np.float32))
    v = model(x.to(device))["voltages"]
    assert v.min().item() >= -80.0 - 1e-3 and v.max().item() <= 0.0 + 1e-3
