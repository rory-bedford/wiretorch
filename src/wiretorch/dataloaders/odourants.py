"""Deprecated alias of :mod:`wiretorch.dataloaders.input_patterns`, kept so
existing experiments keep importing."""

from .input_patterns import (
    generate_assembly_firing_rates,
    modulated_rates,
)
from .input_patterns import (
    generate_baseline_firing_rates as _generate_baseline_firing_rates,
)


def generate_odour_firing_rates(*args, odour_configs=None, **kwargs):
    """:func:`~.input_patterns.generate_assembly_firing_rates`, with ``odour_configs``
    for ``input_configs``."""
    if odour_configs is not None:
        kwargs["input_configs"] = odour_configs
    return generate_assembly_firing_rates(*args, **kwargs)


def generate_baseline_firing_rates(*args, odour_configs=None, **kwargs):
    """:func:`~.input_patterns.generate_baseline_firing_rates`, with ``odour_configs``
    for ``input_configs``."""
    if odour_configs is not None:
        kwargs["input_configs"] = odour_configs
    return _generate_baseline_firing_rates(*args, **kwargs)


__all__ = [
    "generate_baseline_firing_rates",
    "generate_odour_firing_rates",
    "modulated_rates",
]
