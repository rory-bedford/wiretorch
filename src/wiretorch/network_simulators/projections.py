"""Per-(source_cell_type, target_cell_type) connectivity projections.

A ``Projection`` owns the parameters for one biological connectivity block
(e.g. excitatory → excitatory) and produces an ``(n_source, n_target)``
weight matrix on demand. The same Projection object can be referenced by
multiple layers in a multi-layer architecture; gradient sharing across
layers is automatic.

The simulator dispatches caching based on ``Projection.caching_mode``:

* ``"fixed"``           — both connection weights and synapse kernel cached
                          at init (FrozenProjection).
* ``"weights"``         — kernel cached at init, conn weights computed each
                          forward (FullRank, LowRank).
* ``"scaling_factors"`` — conn weights cached at init, kernel scaled by
                          current SF each forward (ScalingFactor).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


# ====================================================================
# Base class
# ====================================================================


class Projection(nn.Module):
    """One (source_cell_type, target_cell_type) connectivity block.

    Subclasses own parameters and implement ``forward()`` to produce the
    full ``(n_source, n_target)`` weight matrix (linear, not log).
    """

    n_source: int
    n_target: int

    def __init__(self, n_source: int, n_target: int):
        super().__init__()
        self.n_source = int(n_source)
        self.n_target = int(n_target)

    @property
    def caching_mode(self) -> str:
        """One of ``"fixed"``, ``"weights"``, ``"scaling_factors"``."""
        raise NotImplementedError

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        """Return the assembled ``(n_source, n_target)`` weight matrix."""
        raise NotImplementedError

    def kernel_factor(self) -> torch.Tensor | float | None:
        """Multiplicative factor applied to the cached synapse kernel.

        Returns ``None`` (or ``1.0``) if there is no kernel-side dynamic
        factor. Used by the ``"scaling_factors"`` caching path.
        """
        return None


# ====================================================================
# Concrete projection types
# ====================================================================


def _to_float_tensor(x: NDArray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


class FrozenProjection(Projection):
    """Buffer-only projection — no trainable parameters.

    The weight matrix is constant. Both conn weights and kernel are cached
    at init in the simulator.
    """

    def __init__(self, weights: NDArray | torch.Tensor):
        w = _to_float_tensor(weights)
        if w.ndim != 2:
            raise ValueError(f"weights must be 2D, got shape {tuple(w.shape)}")
        super().__init__(*w.shape)
        self.register_buffer("_weights", w)

    @property
    def caching_mode(self) -> str:
        return "fixed"

    def forward(self) -> torch.Tensor:
        return self._weights


class ScalingFactorProjection(Projection):
    """Fixed connectome with one trainable scaling factor.

    ``forward()`` returns ``exp(log_sf) * connectome``. The simulator
    caches ``connectome`` as the conn-weights and reads ``kernel_factor()``
    each forward to apply ``exp(log_sf)`` to the synapse kernel.
    """

    def __init__(
        self,
        connectome: NDArray | torch.Tensor,
        init_sf: float = 1.0,
    ):
        c = _to_float_tensor(connectome)
        if c.ndim != 2:
            raise ValueError(f"connectome must be 2D, got shape {tuple(c.shape)}")
        super().__init__(*c.shape)
        self.register_buffer("connectome", c)
        eps = 1e-8
        self.log_sf = nn.Parameter(
            torch.tensor(float(np.log(init_sf + eps)), dtype=torch.float32)
        )

    @property
    def caching_mode(self) -> str:
        return "scaling_factors"

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_sf) * self.connectome

    def kernel_factor(self) -> torch.Tensor:
        return torch.exp(self.log_sf)


class FullRankProjection(Projection):
    """Trainable per-connection log-weights, optionally masked.

    ``forward()`` returns ``exp(log_weights) * mask``. If ``mask`` is None,
    full connectivity is assumed (mask of all-ones is stored as a buffer).
    """

    def __init__(
        self,
        init_weights: NDArray | torch.Tensor,
        mask: NDArray | torch.Tensor | None = None,
    ):
        w = _to_float_tensor(init_weights)
        if w.ndim != 2:
            raise ValueError(f"init_weights must be 2D, got shape {tuple(w.shape)}")
        super().__init__(*w.shape)
        eps = 1e-8
        self.log_weights = nn.Parameter(torch.log(w + eps))
        m = torch.ones_like(w) if mask is None else _to_float_tensor(mask)
        if m.shape != w.shape:
            raise ValueError(
                f"mask shape {tuple(m.shape)} != init_weights shape {tuple(w.shape)}"
            )
        self.register_buffer("mask", m)

    @property
    def caching_mode(self) -> str:
        return "weights"

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_weights) * self.mask


class LowRankProjection(Projection):
    """Low-rank parametrisation of log-weights.

    ``forward()`` returns ``exp(U @ V) * mask`` with ``U: (n_source, rank)``
    and ``V: (rank, n_target)``. If ``U_init``/``V_init`` are supplied,
    they're used as-is; otherwise random init scaled by ``1/sqrt(rank)``.
    """

    def __init__(
        self,
        n_source: int,
        n_target: int,
        rank: int,
        U_init: NDArray | torch.Tensor | None = None,
        V_init: NDArray | torch.Tensor | None = None,
        mask: NDArray | torch.Tensor | None = None,
    ):
        super().__init__(n_source, n_target)
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        scale = 1.0 / np.sqrt(rank)
        if U_init is None:
            U = torch.randn(n_source, rank) * scale
        else:
            U = _to_float_tensor(U_init)
            if U.shape != (n_source, rank):
                raise ValueError(
                    f"U_init shape {tuple(U.shape)} != ({n_source}, {rank})"
                )
        if V_init is None:
            V = torch.randn(rank, n_target) * scale
        else:
            V = _to_float_tensor(V_init)
            if V.shape != (rank, n_target):
                raise ValueError(
                    f"V_init shape {tuple(V.shape)} != ({rank}, {n_target})"
                )
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        m = torch.ones(n_source, n_target) if mask is None else _to_float_tensor(mask)
        if m.shape != (n_source, n_target):
            raise ValueError(f"mask shape {tuple(m.shape)} != ({n_source}, {n_target})")
        self.register_buffer("mask", m)

    @property
    def caching_mode(self) -> str:
        return "weights"

    def forward(self) -> torch.Tensor:
        return torch.exp(self.U @ self.V) * self.mask


# ====================================================================
# Builder helpers
# ====================================================================


PairKey = Tuple[str, str]  # (source_cell_type_name, target_cell_type_name)


def _per_pair_block(
    full_matrix: NDArray,
    src_indices: NDArray,
    tgt_indices: NDArray,
) -> NDArray:
    """Slice ``full_matrix[src_indices][:, tgt_indices]`` (numpy)."""
    return full_matrix[np.ix_(src_indices, tgt_indices)]


def _indices_per_ct(
    cell_type_indices: NDArray,
    cell_type_names: list[str],
) -> Dict[str, NDArray]:
    """For each cell type, return the indices where that type appears."""
    out: Dict[str, NDArray] = {}
    for ct_id, name in enumerate(cell_type_names):
        out[name] = np.flatnonzero(cell_type_indices == ct_id)
    return out


def _resolve_rec_sources(
    rec_weights: NDArray,
    ff_weights: NDArray,
    cell_type_indices: NDArray,
    ff_cell_type_indices: NDArray,
    rec_source_cell_type_indices: NDArray | None,
) -> NDArray:
    """Validate chunked-FF weight blocks and return the recurrent source types.

    ``cell_type_indices`` describes the *output* neurons; ``rec_weights`` and
    ``ff_weights`` are both ``(n_sources, n_outputs)``. The recurrent inputs are
    the same neurons as the outputs unless ``rec_source_cell_type_indices`` says
    otherwise (e.g. a visible-only output layer driven by the full population),
    which is easy to get wrong silently — so check it here rather than letting a
    mis-shaped projection surface downstream in model validation.
    """
    rec_sources = (
        cell_type_indices
        if rec_source_cell_type_indices is None
        else rec_source_cell_type_indices
    )

    n_outputs = len(cell_type_indices)
    for name, block in (("rec_weights", rec_weights), ("ff_weights", ff_weights)):
        if block.shape[1] != n_outputs:
            raise ValueError(
                f"{name} has {block.shape[1]} columns but cell_type_indices "
                f"describes {n_outputs} output neurons."
            )
    if rec_weights.shape[0] != len(rec_sources):
        raise ValueError(
            f"rec_weights has {rec_weights.shape[0]} rows but the recurrent input "
            f"cell types describe {len(rec_sources)} neurons. When the recurrent "
            "inputs are not the same neurons as the outputs, pass their cell types "
            "as rec_source_cell_type_indices."
        )
    if ff_weights.shape[0] != len(ff_cell_type_indices):
        raise ValueError(
            f"ff_weights has {ff_weights.shape[0]} rows but ff_cell_type_indices "
            f"describes {len(ff_cell_type_indices)} inputs."
        )

    return rec_sources


def make_scaling_factor_projections(
    rec_weights: NDArray,
    ff_weights: NDArray,
    cell_type_indices: NDArray,
    ff_cell_type_indices: NDArray,
    cell_type_names: list[str],
    ff_cell_type_names: list[str],
    init_sf: float = 1.0,
) -> Tuple[
    Dict[PairKey, ScalingFactorProjection], Dict[PairKey, ScalingFactorProjection]
]:
    """Build ScalingFactorProjections from teacher matrices.

    Slices ``rec_weights`` and ``ff_weights`` into per-(src_ct, tgt_ct)
    blocks and wraps each as a ScalingFactorProjection. Used when the
    student starts as the teacher's connectome with one trainable scalar
    per pair.

    Args:
        rec_weights:    (n_neurons, n_neurons) recurrent weight matrix.
        ff_weights:     (n_ff, n_neurons) FF weight matrix.
        cell_type_indices:    per-neuron cell type id (ints).
        ff_cell_type_indices: per-FF-input cell type id.
        cell_type_names:      ordered list mapping ids → names for output cells.
        ff_cell_type_names:   ordered list mapping ids → names for FF inputs.
        init_sf:        initial scaling factor (default 1.0).

    Returns:
        ``(rec_projections, ff_projections)`` — each a dict
        ``{(src_name, tgt_name): ScalingFactorProjection}``.
    """
    rec_idx = _indices_per_ct(cell_type_indices, cell_type_names)
    ff_idx = _indices_per_ct(ff_cell_type_indices, ff_cell_type_names)

    rec_projs: Dict[PairKey, ScalingFactorProjection] = {}
    for src in cell_type_names:
        for tgt in cell_type_names:
            block = _per_pair_block(rec_weights, rec_idx[src], rec_idx[tgt])
            rec_projs[(src, tgt)] = ScalingFactorProjection(block, init_sf=init_sf)

    ff_projs: Dict[PairKey, ScalingFactorProjection] = {}
    for src in ff_cell_type_names:
        for tgt in cell_type_names:
            block = _per_pair_block(ff_weights, ff_idx[src], rec_idx[tgt])
            ff_projs[(src, tgt)] = ScalingFactorProjection(block, init_sf=init_sf)

    return rec_projs, ff_projs


def make_frozen_projections(
    rec_weights: NDArray,
    ff_weights: NDArray,
    cell_type_indices: NDArray,
    ff_cell_type_indices: NDArray,
    cell_type_names: list[str],
    ff_cell_type_names: list[str],
    scaling_factors_rec: NDArray | None = None,
    scaling_factors_ff: NDArray | None = None,
) -> Tuple[Dict[PairKey, FrozenProjection], Dict[PairKey, FrozenProjection]]:
    """Build FrozenProjections (no trainable params) per pair, baking SFs.

    For inference / fixed-weights use cases. ``scaling_factors_*`` matrices
    (shape ``(n_src_ct, n_tgt_ct)``) are multiplied into each pair block;
    omit them to use unscaled weights.
    """
    rec_idx = _indices_per_ct(cell_type_indices, cell_type_names)
    ff_idx = _indices_per_ct(ff_cell_type_indices, ff_cell_type_names)

    def _baked_block(weights, src_indices, tgt_indices, sf_matrix, src_id, tgt_id):
        block = weights[np.ix_(src_indices, tgt_indices)].astype(np.float32)
        if sf_matrix is not None:
            block = block * float(sf_matrix[src_id, tgt_id])
        return block

    rec_projs: Dict[PairKey, FrozenProjection] = {}
    for src_id, src in enumerate(cell_type_names):
        for tgt_id, tgt in enumerate(cell_type_names):
            blk = _baked_block(
                rec_weights,
                rec_idx[src],
                rec_idx[tgt],
                scaling_factors_rec,
                src_id,
                tgt_id,
            )
            rec_projs[(src, tgt)] = FrozenProjection(blk)

    ff_projs: Dict[PairKey, FrozenProjection] = {}
    for src_id, src in enumerate(ff_cell_type_names):
        for tgt_id, tgt in enumerate(cell_type_names):
            blk = _baked_block(
                ff_weights,
                ff_idx[src],
                rec_idx[tgt],
                scaling_factors_ff,
                src_id,
                tgt_id,
            )
            ff_projs[(src, tgt)] = FrozenProjection(blk)

    return rec_projs, ff_projs


def make_frozen_chunked_ff_projections(
    rec_weights: NDArray,
    ff_weights: NDArray,
    cell_type_indices: NDArray,
    ff_cell_type_indices: NDArray,
    cell_type_names: list[str],
    ff_cell_type_names: list[str],
    scaling_factors: NDArray | None = None,  # (n_ff_ct + n_rec_ct, n_rec_ct)
    rec_source_cell_type_indices: NDArray | None = None,
) -> Dict[PairKey, FrozenProjection]:
    """Frozen version for the chunked-FF model. Single combined dict.

    ``scaling_factors`` indexes the combined-input cell types: rows
    ``[0:n_ff_ct]`` are FF, ``[n_ff_ct:]`` are recurrent (offset by
    n_ff_ct relative to ``cell_type_names``).

    ``rec_source_cell_type_indices`` gives the cell types of the recurrent
    *input* rows when they are not the same neurons as the outputs — e.g. a
    visible-only output layer driven by the full recurrent population, where
    ``rec_weights`` is ``(n_all, n_visible)``. Defaults to
    ``cell_type_indices`` (inputs and outputs are the same neurons).
    """
    rec_sources = _resolve_rec_sources(
        rec_weights,
        ff_weights,
        cell_type_indices,
        ff_cell_type_indices,
        rec_source_cell_type_indices,
    )
    rec_tgt_idx = _indices_per_ct(cell_type_indices, cell_type_names)
    rec_src_idx = _indices_per_ct(rec_sources, cell_type_names)
    ff_idx = _indices_per_ct(ff_cell_type_indices, ff_cell_type_names)
    n_ff_ct = len(ff_cell_type_names)

    projs: Dict[PairKey, FrozenProjection] = {}
    for src_id, src in enumerate(ff_cell_type_names):
        for tgt_id, tgt in enumerate(cell_type_names):
            block = ff_weights[np.ix_(ff_idx[src], rec_tgt_idx[tgt])].astype(np.float32)
            if scaling_factors is not None:
                block = block * float(scaling_factors[src_id, tgt_id])
            projs[(src, tgt)] = FrozenProjection(block)
    for src_id, src in enumerate(cell_type_names):
        for tgt_id, tgt in enumerate(cell_type_names):
            block = rec_weights[np.ix_(rec_src_idx[src], rec_tgt_idx[tgt])].astype(
                np.float32
            )
            if scaling_factors is not None:
                # Recurrent rows are offset by n_ff_ct
                block = block * float(scaling_factors[n_ff_ct + src_id, tgt_id])
            projs[(src, tgt)] = FrozenProjection(block)
    return projs


def make_chunked_ff_projections(
    rec_weights: NDArray,
    ff_weights: NDArray,
    cell_type_indices: NDArray,
    ff_cell_type_indices: NDArray,
    cell_type_names: list[str],
    ff_cell_type_names: list[str],
    init_sf: float = 1.0,
    rec_source_cell_type_indices: NDArray | None = None,
) -> Dict[PairKey, ScalingFactorProjection]:
    """Build a single combined projection dict for the chunked-FF model.

    The chunked-FF simulator treats both true-FF inputs and recurrent
    inputs (from previous chunk's spikes) uniformly as feedforward. This
    helper returns a single dict combining both into one map keyed by
    ``(input_ct_name, output_ct_name)``.

    The input-cell-type space is the union of ``ff_cell_type_names`` and
    ``cell_type_names`` (recurrent cell types appear as input rows).

    ``rec_source_cell_type_indices`` gives the cell types of the recurrent
    *input* rows when they are not the same neurons as the outputs — e.g. a
    visible-only output layer driven by the full recurrent population, where
    ``rec_weights`` is ``(n_all, n_visible)``. Defaults to
    ``cell_type_indices`` (inputs and outputs are the same neurons).
    """
    rec_sources = _resolve_rec_sources(
        rec_weights,
        ff_weights,
        cell_type_indices,
        ff_cell_type_indices,
        rec_source_cell_type_indices,
    )
    rec_tgt_idx = _indices_per_ct(cell_type_indices, cell_type_names)
    rec_src_idx = _indices_per_ct(rec_sources, cell_type_names)
    ff_idx = _indices_per_ct(ff_cell_type_indices, ff_cell_type_names)

    projs: Dict[PairKey, ScalingFactorProjection] = {}
    for src in ff_cell_type_names:
        for tgt in cell_type_names:
            block = _per_pair_block(ff_weights, ff_idx[src], rec_tgt_idx[tgt])
            projs[(src, tgt)] = ScalingFactorProjection(block, init_sf=init_sf)
    for src in cell_type_names:
        for tgt in cell_type_names:
            block = _per_pair_block(rec_weights, rec_src_idx[src], rec_tgt_idx[tgt])
            projs[(src, tgt)] = ScalingFactorProjection(block, init_sf=init_sf)

    return projs
