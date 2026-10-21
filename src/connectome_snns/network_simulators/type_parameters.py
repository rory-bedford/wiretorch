"""Physiology parameters with one value per type, each fixed or learnt.

A network's physiology is mostly per cell type (membrane time constant, threshold, ...)
or per synapse type (synaptic time constant). ``TypeParameters`` holds such values and
makes any of them trainable — for every type, or only some — while the rest stay
buffers, so only what is asked for costs a gradient:

    >>> params = TypeParameters(
    ...     {"tau_mem": [20.0, 10.0], "theta": [0.0, 0.0]},
    ...     learnt={"tau_mem": [0]},  # only type 0's time constant
    ...     positive={"tau_mem"},
    ... )
    >>> params.value("tau_mem")  # (2,): learnt for type 0, fixed for type 1

Positive parameters are learnt as their logarithm (``log_<name>``), so they stay
positive; the others as themselves (``<name>``). Either way the parameter holds only
the learnt types' values.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TypeParameters(nn.Module):
    """One value per type of each named parameter; any of them learnt, for any types.

    Args:
        values: ``{name: values}``, one value per type (index = type id).
        learnt: ``{name: type ids}`` to train (others stay fixed).
        positive: Names that must stay positive (learnt as logarithms).
    """

    def __init__(
        self,
        values: dict[str, Sequence[float]],
        learnt: dict[str, Sequence[int]] | None = None,
        positive: set[str] | frozenset[str] = frozenset(),
    ):
        super().__init__()
        learnt = learnt or {}
        unknown = set(learnt) - set(values)
        assert not unknown, f"cannot learn unknown parameters {sorted(unknown)}"
        self.names = tuple(values)
        self.positive = frozenset(positive)
        self.learnt = {}
        for name, value in values.items():
            value = torch.tensor(list(value), dtype=torch.float32)
            types = sorted(set(learnt.get(name, ())))
            if not types:
                self.register_buffer(name, value)
                continue
            self.learnt[name] = types
            index = torch.tensor(types, dtype=torch.long)
            self.register_buffer(f"_{name}_learnt", index, persistent=False)
            self.register_buffer(f"_{name}_fixed", value)
            initial = value[index]
            if name in self.positive:
                assert torch.all(initial > 0), (
                    f"a learnt {name} must be positive, got {initial.tolist()}"
                )
                self.register_parameter(f"log_{name}", nn.Parameter(initial.log()))
            else:
                self.register_parameter(name, nn.Parameter(initial))

    def value(self, name: str) -> torch.Tensor:
        """``(n_types,)`` values of ``name``, differentiable in the learnt ones."""
        if name not in self.learnt:
            return getattr(self, name)
        if name in self.positive:
            learnt = getattr(self, f"log_{name}").exp()
        else:
            learnt = getattr(self, name)
        fixed = getattr(self, f"_{name}_fixed")
        # index_put, unlike masked_scatter, needs no host synchronisation on CUDA.
        return fixed.index_put((getattr(self, f"_{name}_learnt"),), learnt)


def learnt_types(
    spec: bool | Sequence[str] | None, names: Sequence[str], what: str
) -> list[int]:
    """Type ids from a ``learn`` entry: True (every type), names, or None/False."""
    if not spec:
        return []
    if spec is True:
        return list(range(len(names)))
    unknown = set(spec) - set(names)
    assert not unknown, f"{what}: unknown types {sorted(unknown)} (have {list(names)})"
    return [i for i, name in enumerate(names) if name in spec]


def learnt_synapses(
    spec: bool | Sequence[str] | None, synapses: list[dict], what: str = "learn"
) -> list[int]:
    """Registry indices from a ``learn`` entry for a synapse parameter: True (every
    synapse type), synapse names, or None/False."""
    if not spec:
        return []
    if spec is True:
        return list(range(len(synapses)))
    names = {s["name"] for s in synapses}
    unknown = set(spec) - names
    assert not unknown, f"{what}: unknown synapses {sorted(unknown)}"
    return [i for i, s in enumerate(synapses) if s["name"] in spec]
