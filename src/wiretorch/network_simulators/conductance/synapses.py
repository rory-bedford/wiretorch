"""Physiology tables: per-neuron cell parameters and the synapse registry.

A conductance-based network has one conductance per *synapse type* per neuron (indexed
by the shared ``SynapseRegistry``), each with a rise and a decay component.
"""

from __future__ import annotations

import numpy as np
import torch

from wiretorch.network_simulators.synapses import SynapseRegistry

NEURON_PARAMETERS = ("tau_mem", "theta", "U_reset", "E_L", "g_L")
SYNAPSE_PARAMETERS = ("tau_rise", "tau_decay", "E_syn", "g_bar")


def neuron_tables(
    cell_params: list[dict], cell_type_indices: np.ndarray
) -> dict[str, torch.Tensor]:
    """Per-neuron float32 tensors of each cell parameter, plus ``C_m = tau_mem * g_L``."""
    tables = {}
    for name in NEURON_PARAMETERS:
        lookup = np.zeros(len(cell_params), dtype=np.float32)
        for cell in cell_params:
            lookup[cell["cell_id"]] = cell[name]
        tables[name] = torch.from_numpy(lookup[cell_type_indices])
    tables["C_m"] = tables["tau_mem"] * tables["g_L"]
    return tables


def synapse_tables(registry: SynapseRegistry) -> dict[str, torch.Tensor]:
    """Float32 tensors of each synapse parameter by index; ``g_clip`` is inf if unset."""
    tables = {name: registry.table(name) for name in SYNAPSE_PARAMETERS}
    tables["g_clip"] = registry.table("g_clip", default=float("inf"))
    return tables


#: Defaults of the Mg²⁺ block's parameters: none (``mg``, mM), and the Jahr–Stevens
#: slope (``mg_slope``, 1/mV) and dissociation constant (``mg_k``, mM).
MG, MG_SLOPE, MG_K = 0.0, 0.062, 3.57


def mg_block_tables(registry: SynapseRegistry) -> dict[str, torch.Tensor]:
    """The voltage-dependent Mg²⁺ block of each synapse type (NMDA's),

        B(v) = 1 / (1 + mg_c exp(-mg_a v)),   mg_c = mg / mg_k,   mg_a = mg_slope,

    which scales its current. ``mg = 0`` (the default) is no block: ``B = 1``.
    """
    mg = registry.table("mg", default=MG)
    assert (mg >= 0).all(), "synapse params: mg must be >= 0"
    return {
        "mg_c": mg / registry.table("mg_k", default=MG_K),
        "mg_a": registry.table("mg_slope", default=MG_SLOPE),
    }


def saturation_table(registry: SynapseRegistry) -> torch.Tensor:
    """Each synapse type's ``saturation``: the fraction of a source neuron's closed
    receptors (of that type) that one of its spikes opens. 0 (the default) is no
    saturation: every spike adds the same conductance."""
    saturation = registry.table("saturation", default=0.0)
    assert ((saturation >= 0) & (saturation <= 1)).all(), (
        "synapse params: saturation must be in [0, 1]"
    )
    return saturation


def conductance_kernel(
    tau_rise: torch.Tensor,
    tau_decay: torch.Tensor,
    g_bar: torch.Tensor,
    saturating: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(2, n_synapse_types)`` jump of the rise and decay components per unit input.

    The rise component jumps by ``-g_bar`` and the decay component by ``+g_bar``,
    normalised so the difference-of-exponentials waveform peaks at ``g_bar`` — except
    for the ``saturating`` types, whose input is the fraction of receptors opened, so
    that ``g_bar`` is their conductance with every receptor open.
    """
    r = tau_rise / tau_decay
    peak = r ** (r / (1 - r)) - r ** (1 / (1 - r))
    if saturating is not None:
        peak = torch.where(saturating, torch.ones_like(peak), peak)
    return torch.stack([-g_bar, g_bar], dim=0) / peak
