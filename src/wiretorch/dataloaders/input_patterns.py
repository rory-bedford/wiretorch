"""Static firing-rate patterns for the feedforward inputs.

:func:`generate_assembly_firing_rates` makes one pattern per assembly of the recurrent
network: the inputs most strongly connected to the assembly fire above baseline, the
rest below it, keeping the population mean at baseline. :func:`generate_baseline_firing_rates`
is the unmodulated control. The patterns feed a rate process (e.g. the OU mixture in
:mod:`.rate_processes`) or a Poisson spike generator (:mod:`.unsupervised`).

Rate parameters come per feedforward cell type, as dicts with the fields of
:class:`~wiretorch.configs.input_patterns.InputPatternConfig`.
"""

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def modulated_rates(
    baseline_rate: float, modulation_rate: float, modulation_fraction: float, n: int
) -> tuple[int, float, float]:
    """How a pattern modulates ``n`` inputs: ``(n_up, up_rate, down_rate)``.

    ``int(n * modulation_fraction)`` inputs fire at ``baseline + modulation``; the rest
    share the rate that keeps the population mean at ``baseline_rate``.
    """
    n_up = int(n * modulation_fraction)
    up_rate = baseline_rate + modulation_rate
    n_down = n - n_up
    down_rate = (
        (n * baseline_rate - n_up * up_rate) / n_down if n_down else baseline_rate
    )
    return n_up, up_rate, down_rate


def generate_assembly_firing_rates(
    feedforward_weights: ArrayLike,
    input_source_indices: ArrayLike,
    cell_type_indices: ArrayLike,
    assembly_ids: ArrayLike,
    target_cell_type_idx: int,
    cell_type_names: list[str],
    input_configs: dict[str, dict],
) -> np.ndarray:
    """One firing-rate pattern per assembly, modulated by connection strength.

    An input's strength to an assembly is the sum of its feedforward weights onto the
    assembly's neurons of the target cell type (e.g. only excitatory cells). In each
    pattern, the strongest ``modulation_fraction`` of each input cell type fire at
    ``baseline + modulation_rate`` and the rest at the rate that keeps the type's mean
    at baseline (:func:`modulated_rates`).

    Args:
        feedforward_weights: ``(n_inputs, n_recurrent)`` weight matrix.
        input_source_indices: Cell type index of each input.
        cell_type_indices: Cell type index of each recurrent neuron.
        assembly_ids: Assembly of each recurrent neuron (-1 for unassigned).
        target_cell_type_idx: Recurrent cell type whose incoming weights define the
            connection strengths.
        cell_type_names: Names of the input cell types, in index order.
        input_configs: Per input cell type name: 'baseline_rate', 'modulation_rate',
            'modulation_fraction', and optionally 'baseline_variance' (default 0; if
            > 0, each pattern's baseline is drawn from N(baseline_rate,
            baseline_variance) with NumPy's global RNG, clipped at 0.1 Hz).

    Returns:
        ``(n_assemblies, n_inputs)`` firing rates in Hz, assemblies in sorted order.

    Example:
        >>> rates = generate_assembly_firing_rates(
        ...     feedforward_weights=np.random.rand(1000, 5000),
        ...     input_source_indices=np.zeros(1000, dtype=int),
        ...     cell_type_indices=np.zeros(5000, dtype=int),
        ...     assembly_ids=np.random.randint(-1, 20, 5000),
        ...     target_cell_type_idx=0,
        ...     cell_type_names=["input"],
        ...     input_configs={"input": {"baseline_rate": 6.0, "modulation_rate": 1.0,
        ...                              "modulation_fraction": 0.1}},
        ... )
        >>> rates.shape
        (20, 1000)
    """
    feedforward_weights = _numpy(feedforward_weights)
    input_source_indices = _numpy(input_source_indices)
    cell_type_indices = _numpy(cell_type_indices)
    assembly_ids = _numpy(assembly_ids)

    assemblies = np.unique(assembly_ids[assembly_ids >= 0])
    if len(assemblies) == 0:
        raise ValueError("No assemblies found in assembly_ids")
    n_patterns, n_inputs = len(assemblies), feedforward_weights.shape[0]

    # (n_patterns, n_inputs): each input's summed weight onto each assembly's targets
    targets = cell_type_indices == target_cell_type_idx
    strengths = np.zeros((n_patterns, n_inputs))
    for pattern, assembly in enumerate(assemblies):
        strengths[pattern] = feedforward_weights[
            :, (assembly_ids == assembly) & targets
        ].sum(axis=1)

    firing_rates = np.zeros((n_patterns, n_inputs))
    for ct_idx, ct_name in enumerate(cell_type_names):
        config = _input_config(input_configs, ct_name)
        baselines = _pattern_baselines(config, n_patterns)
        inputs = np.where(input_source_indices == ct_idx)[0]
        for pattern in range(n_patterns):
            n_up, up_rate, down_rate = modulated_rates(
                baselines[pattern],
                config["modulation_rate"],
                config["modulation_fraction"],
                len(inputs),
            )
            by_strength = inputs[np.argsort(-strengths[pattern, inputs])]
            firing_rates[pattern, by_strength[:n_up]] = up_rate
            firing_rates[pattern, by_strength[n_up:]] = down_rate
    return firing_rates


def generate_baseline_firing_rates(
    n_input_neurons: int,
    input_source_indices: ArrayLike,
    cell_type_names: list[str],
    input_configs: dict[str, dict],
) -> np.ndarray:
    """A single unmodulated pattern: every input at its cell type's baseline rate.

    The control for :func:`generate_assembly_firing_rates`; ``input_configs`` needs
    only 'baseline_rate'. Returns ``(1, n_input_neurons)`` rates in Hz.
    """
    input_source_indices = _numpy(input_source_indices)
    firing_rates = np.zeros((1, n_input_neurons))
    for ct_idx, ct_name in enumerate(cell_type_names):
        config = _input_config(input_configs, ct_name)
        firing_rates[0, input_source_indices == ct_idx] = config["baseline_rate"]
    return firing_rates


def _numpy(array: ArrayLike) -> np.ndarray:
    return array.cpu().numpy() if isinstance(array, torch.Tensor) else array


def _input_config(input_configs: dict[str, dict], cell_type_name: str) -> dict:
    if cell_type_name not in input_configs:
        raise ValueError(
            f"No input pattern configuration found for cell type '{cell_type_name}'"
        )
    return input_configs[cell_type_name]


def _pattern_baselines(config: dict, n_patterns: int) -> np.ndarray:
    """Each pattern's baseline rate: fixed, or drawn if 'baseline_variance' > 0."""
    variance = config.get("baseline_variance", 0.0)
    if variance > 0:
        baselines = np.random.normal(
            loc=config["baseline_rate"], scale=np.sqrt(variance), size=n_patterns
        )
        return np.clip(baselines, 0.1, None)
    return np.full(n_patterns, config["baseline_rate"])
