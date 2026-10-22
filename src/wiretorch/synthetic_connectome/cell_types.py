"""Cell type assignment utilities.

This module provides functions for assigning neurons to different cell types
based on given proportions.

Features:
- Random assignment based on proportions
- Deterministic seeding for reproducibility
- Support for multiple cell types

Typical workflow:
    >>> cell_proportions = [0.8, 0.2]  # 80% type 0, 20% type 1
    >>> cell_types = assign_cell_types(100, cell_proportions, seed=42)
    >>> # cell_types contains integers 0 and 1 indicating cell type
"""

import numpy as np
from numpy.typing import NDArray
from typing import List

# Type aliases for clarity
IntArray = NDArray[np.int_]


def assign_cell_types(
    num_neurons: int,
    cell_type_proportions: List[float],
    num_assemblies: int | None = None,
) -> IntArray:
    """
    Randomly assign neurons to cell types based on given proportions.

    Args:
        num_neurons (int): Total number of neurons to assign.
        cell_type_proportions (List[float]): List of proportions for each cell type. Must sum to 1.0.
        num_assemblies (int | None): If provided, balance cell types evenly across assemblies.

    Returns:
        IntArray: Array of shape (num_neurons,) with cell type indices.

    Raises:
        ValueError: If proportions don't sum to approximately 1.0.
    """
    if abs(sum(cell_type_proportions) - 1.0) > 1e-6:
        raise ValueError(
            f"Cell type proportions must sum to 1.0, got {sum(cell_type_proportions)}"
        )

    # Calculate number of neurons for each cell type
    num_cell_types = len(cell_type_proportions)
    cell_type_counts = np.zeros(num_cell_types, dtype=int)

    # Assign neurons ensuring we use all neurons
    remaining_neurons = num_neurons
    for i in range(num_cell_types - 1):
        count = int(np.round(num_neurons * cell_type_proportions[i]))
        cell_type_counts[i] = count
        remaining_neurons -= count

    # Last cell type gets remaining neurons
    cell_type_counts[-1] = remaining_neurons

    # Create array of cell type assignments
    cell_type_indices = np.zeros(num_neurons, dtype=int)
    start_idx = 0

    for cell_type_idx, count in enumerate(cell_type_counts):
        cell_type_indices[start_idx : start_idx + count] = cell_type_idx
        start_idx += count

    # If num_assemblies specified, balance cell types across assemblies
    if num_assemblies is not None:
        assembly_size = num_neurons // num_assemblies
        n_neurons_used = assembly_size * num_assemblies

        # Get unique cell types and their counts (only use neurons that fit into assemblies)
        unique_types, counts = np.unique(
            cell_type_indices[:n_neurons_used], return_counts=True
        )

        # Create new balanced array
        balanced = np.zeros(n_neurons_used, dtype=int)

        # For each assembly, distribute cell types evenly
        for assembly_id in range(num_assemblies):
            start_idx_asm = assembly_id * assembly_size
            end_idx_asm = (assembly_id + 1) * assembly_size

            # Calculate how many of each type should go in this assembly
            neurons_per_type = counts // num_assemblies

            pos = start_idx_asm
            for cell_type, count_per_assembly in zip(unique_types, neurons_per_type):
                balanced[pos : pos + count_per_assembly] = cell_type
                pos += count_per_assembly

            # Shuffle within each assembly to randomize positions
            np.random.shuffle(balanced[start_idx_asm:end_idx_asm])

        return balanced
    else:
        # Shuffle to randomize assignment
        np.random.shuffle(cell_type_indices)
        return cell_type_indices
