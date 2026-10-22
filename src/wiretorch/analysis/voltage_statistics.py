"""Analysis functions for membrane voltage statistics."""

import numpy as np
from numpy.typing import NDArray

from .utils import mean_and_std


def compute_membrane_potential_by_cell_type(
    voltages: NDArray[np.float32],
    cell_type_indices: NDArray[np.int32],
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of membrane potential statistics by cell type.

    For each neuron, computes the mean and standard deviation of its membrane potential
    across time. Then, for each cell type, computes the mean and standard deviation
    of these per-neuron statistics.

    Args:
        voltages (NDArray[np.float32]): Membrane voltages of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (NDArray[np.int32]): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_of_means": Mean of per-neuron mean voltages for this cell type
            - "std_of_means": Standard deviation of per-neuron mean voltages
            - "mean_of_stds": Mean of per-neuron voltage standard deviations
            - "std_of_stds": Standard deviation of per-neuron voltage standard deviations
    """
    means = voltages.mean(axis=(0, 1))
    stds = voltages.std(axis=(0, 1), ddof=1)
    stats = {}
    for cell_type in np.unique(cell_type_indices).tolist():
        mask = cell_type_indices == cell_type
        mean_of_means, std_of_means = mean_and_std(means[mask])
        mean_of_stds, std_of_stds = mean_and_std(stds[mask])
        stats[cell_type] = {
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
            "mean_of_stds": mean_of_stds,
            "std_of_stds": std_of_stds,
        }
    return stats
