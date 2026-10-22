"""The former weight-matrix API of the current-based networks, on top of projections.

The current-based networks used to take a weight matrix per input side, an optional
``(source type, target type)`` scaling-factor matrix, and ``optimisable`` to say what
was trained. They now take ``Projection`` dicts like the conductance-based networks;
this module translates:

* ``side_projections`` builds a side's projections from its former arguments (what the
  networks' ``from_weights`` constructors use);
* ``convert_state_dict`` rewrites a checkpoint written by the former networks into the
  projections' parameters, so it loads into a network built by ``from_weights`` with
  the same arguments.

The former networks were in natural units whatever the cell parameters said (reset 0,
threshold 1); ``natural_units`` keeps that. They also integrated ``v <- beta v + I``
where the networks now integrate ``v <- beta v + (1 - beta) I`` (dt-consistent), so
their weights onto a neuron are divided by ``1 - beta`` of its cell type
(``former_factors``): the same network at the same dt.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from wiretorch.network_simulators.projections import (
    FrozenProjection,
    FullRankProjection,
    Projection,
    ScalingFactorProjection,
)

OptimisableParams = Literal[
    "weights",
    "scaling_factors",
    "scaling_factors_recurrent",
    "scaling_factors_feedforward",
    None,
]
EPS = 1e-8  # added before taking logs, as the former networks did
LOG_EPS = float(np.log(np.zeros((), np.float32) + EPS))  # a log-weight outside the mask


def former_factors(cell_params: list[dict], dt: float) -> torch.Tensor:
    """Per target cell type, ``1 / (1 - beta)``: what the former step's weights are
    multiplied by under the normalised step (float32, by cell id)."""
    dt = float(np.float32(dt))
    factors = np.zeros(len(cell_params))
    for cell in cell_params:
        factors[cell["cell_id"]] = 1.0 / -np.expm1(-dt / cell["tau_mem"])
    return torch.from_numpy(factors.astype(np.float32))


def side_mode(optimisable: OptimisableParams, recurrent: bool, only_side: bool) -> str:
    """The caching mode of a side's projections under the former ``optimisable``.

    ``only_side``: the network has this side only (a feedforward network), so every
    scaling-factor option trains it.
    """
    if optimisable == "weights":
        return "weights"
    trained = {"scaling_factors"}
    if only_side:
        trained |= {"scaling_factors_recurrent", "scaling_factors_feedforward"}
    else:
        trained.add(
            "scaling_factors_recurrent" if recurrent else "scaling_factors_feedforward"
        )
    return "scaling_factors" if optimisable in trained else "fixed"


def side_projections(
    weights: np.ndarray,
    source_types: np.ndarray,
    target_types: np.ndarray,
    source_names: list[str],
    target_names: list[str],
    scaling_factors: np.ndarray | None,
    mode: str,
    mask: np.ndarray | None,
    factors: torch.Tensor,
) -> dict[tuple[str, str], Projection]:
    """``{(source, target): Projection}`` for one side, one per non-empty block.

    ``mode`` ``"weights"`` learns the weights within ``mask`` (scaling factors are not
    applied, as before); ``"scaling_factors"`` learns one factor per block;
    ``"fixed"`` bakes the factors into the weights. Every block is multiplied by its
    target type's ``factors`` (``former_factors``).
    """
    weights = np.asarray(weights, dtype=np.float32)
    if scaling_factors is None:
        scaling_factors = np.ones((len(source_names), len(target_names)))
    if mode == "weights":
        assert mask is not None, "a connectivity mask is required to learn weights"
    projections = {}
    for a, src in enumerate(source_names):
        rows = np.flatnonzero(source_types == a)
        for b, tgt in enumerate(target_names):
            cols = np.flatnonzero(target_types == b)
            if not (rows.size and cols.size):
                continue
            block = weights[np.ix_(rows, cols)]
            factor = factors[b]
            if mode == "weights":
                projection = FullRankProjection(block, mask=mask[np.ix_(rows, cols)])
                # The log-weights as the former networks computed them, rescaled.
                former = torch.from_numpy(np.log(block + EPS))
                projection.log_weights.data = former + factor.log()
            elif mode == "scaling_factors":
                projection = ScalingFactorProjection(
                    torch.from_numpy(block) * factor,
                    init_sf=float(scaling_factors[a, b]),
                )
            else:
                sf = torch.tensor(scaling_factors[a, b], dtype=torch.float32)
                projection = FrozenProjection(torch.from_numpy(block) * sf * factor)
            projections[(src, tgt)] = projection
    return projections


def natural_units(cell_params: list[dict]) -> list[dict]:
    """Cell parameters with the former networks' threshold 1, reset 0, rest 0 and
    resistance 1."""
    return [
        {**cell, "theta": 1.0, "U_reset": 0.0, "U_rest": 0.0, "R": 1.0}
        for cell in cell_params
    ]


def input_cell_params(
    cell_type_indices_FF: np.ndarray, synapse_params_FF: list[dict]
) -> list[dict]:
    """Names for the input cell types, which the former API did not have."""
    n_types = max(
        int(np.max(cell_type_indices_FF)) + 1,
        max(s["cell_id"] for s in synapse_params_FF) + 1,
    )
    return [{"name": f"input_{i}", "cell_id": i} for i in range(n_types)]


def check_weights(weights: np.ndarray, n_sources: int, what: str, square: bool) -> None:
    assert weights.ndim == 2, f"{what} must be a 2D matrix"
    assert np.all(weights >= 0), (
        f"{what} must be non-negative (the sign comes from the synapse type)"
    )
    assert weights.shape[0] == n_sources, (
        f"{what} has {weights.shape[0]} rows but {n_sources} sources"
    )
    if square:
        assert weights.shape[0] == weights.shape[1], f"{what} must be square"


# =====================================================================
# Checkpoints written by the former networks
# =====================================================================


def convert_state_dict(module, state_dict, prefix, *args) -> None:
    """``load_state_dict`` pre-hook: a former network's checkpoint, as projections.

    Recognised by its ``beta_mem`` buffer. Its weights, masks and scaling factors move
    into the projections' parameters and buffers; every other entry is the network's
    own (the physiology tables are rebuilt from the same cell and synapse parameters).
    """
    if prefix + "beta_mem" not in state_dict:
        return
    old = {
        key.removeprefix(prefix): state_dict.pop(key)
        for key in list(state_dict)
        if key.startswith(prefix) and "." not in key.removeprefix(prefix)
    }
    new = {key: value.detach().clone() for key, value in module.state_dict().items()}
    for side in module._sides:
        x = side.mask_buffer.removeprefix("weights_mask")
        mask = old[f"weights_mask{x}"]
        if f"log_weights{x}_flat" in old:
            log_w = torch.zeros(mask.shape)
            log_w[mask] = old[f"log_weights{x}_flat"]
            weights = None
        else:
            weights = old[f"_weights{x}_buffer"]
        log_sf = old.get(f"log_scaling_factors{x}")
        sf = old.get(f"_scaling_factors{x}_buffer")
        source_ids = {c["name"]: c["cell_id"] for c in side.cell_params}
        target_ids = {c["name"]: c["cell_id"] for c in module.cell_params}
        factors = former_factors(module.cell_params, module.dt.item())

        for key, projection in getattr(module, side.attribute).items():
            src, tgt = key.split("__", 1)
            rows, cols = (torch.from_numpy(i) for i in module._block(side, src, tgt))
            pair = (source_ids[src], target_ids[tgt])
            factor = factors[target_ids[tgt]]
            name = f"{side.attribute}.{key}."
            if isinstance(projection, FullRankProjection) and weights is None:
                block, inside = log_w[rows][:, cols], mask[rows][:, cols]
                # Outside the mask, as side_projections initialises them.
                former = torch.full_like(block, LOG_EPS)
                former[inside] = block[inside]
                new[name + "log_weights"] = former + factor.log()
            elif isinstance(projection, ScalingFactorProjection) and log_sf is not None:
                new[name + "connectome"] = weights[rows][:, cols] * factor
                new[name + "log_sf"] = log_sf[pair].clone()
            elif isinstance(projection, FrozenProjection) and sf is not None:
                new[name + "_weights"] = weights[rows][:, cols] * sf[pair] * factor
            else:
                raise ValueError(
                    f"checkpoint of a former current-based network does not match "
                    f"{side.attribute}[{key!r}] ({type(projection).__name__}); build "
                    "the network with from_weights and the same optimisable"
                )
    state_dict.update({prefix + key: value for key, value in new.items()})
