"""The synapse registry: every synapse type of every input side, under one index.

Each network keeps one synaptic state per *synapse type* per neuron (a conductance, a
current). Synapse types come from the cell types that make them, on each input side
(recurrent neurons, feedforward inputs). The registry gives every synapse type a single
index used throughout the simulator; with ``unify=True`` types with the same signature —
(source cell type name, synapse name) — share an index, so e.g. excitatory AMPA from
recurrent and from teacher-forced inputs is one conductance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SynapseRegistry:
    """Synapse types of every input side, under one index.

    Attributes:
        synapses: one parameter dict per synapse type, in index order, each with its
            ``unified_synapse_id`` and ``signature``.
        to_index: per input side, ``synapse_id -> index``.
    """

    synapses: list[dict]
    to_index: list[dict[int, int]]

    @classmethod
    def build(
        cls, sides: list[tuple[list[dict], list[dict]]], unify: bool
    ) -> SynapseRegistry:
        """Index the synapse types of each ``(cell_params, synapse_params)`` side.

        Sides are indexed in order, so the first side's types come first.
        """
        synapses, to_index, by_signature = [], [], {}
        for side, (cells, side_synapses) in enumerate(sides):
            names = {c["cell_id"]: c["name"] for c in cells}
            mapping = {}
            if not unify:  # then each side's indices follow its synapse ids
                side_synapses = sorted(side_synapses, key=lambda s: s["synapse_id"])
            for synapse in side_synapses:
                signature = (names[synapse["cell_id"]], synapse["name"])
                key = signature if unify else (side, synapse["synapse_id"])
                if key not in by_signature:
                    by_signature[key] = len(synapses)
                    synapses.append(
                        {
                            **synapse,
                            "unified_synapse_id": len(synapses),
                            "signature": signature,
                        }
                    )
                mapping[synapse["synapse_id"]] = by_signature[key]
            to_index.append(mapping)
        return cls(synapses, to_index)

    def __len__(self) -> int:
        return len(self.synapses)

    def table(self, name: str, default: float | None = None) -> torch.Tensor:
        """Float32 tensor of one synapse parameter by index (``default`` if unset)."""
        return torch.tensor(
            [
                s[name] if default is None else s.get(name, default)
                for s in self.synapses
            ],
            dtype=torch.float32,
        )

    def side_indices(self, side: int) -> list[int]:
        """Indices of the synapse types made on one input side, in order."""
        return sorted(set(self.to_index[side].values()))

    def made_by(self, side: int, synapse_params: list[dict]) -> dict[int, torch.Tensor]:
        """For each source cell type on ``side``, a bool mask of the types it makes."""
        masks = {}
        for synapse in synapse_params:
            mask = masks.setdefault(
                synapse["cell_id"], torch.zeros(len(self), dtype=torch.bool)
            )
            mask[self.to_index[side][synapse["synapse_id"]]] = True
        return dict(sorted(masks.items()))
