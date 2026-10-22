"""Perturbing connectome weights by cell-type scaling factors, to test whether a
student can recover them."""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


def perturb_weights_scaling_factor(
    weights: FloatArray,
    feedforward_weights: FloatArray,
    cell_type_indices: IntArray,
    feedforward_cell_type_indices: IntArray,
    variance: float,
    optimisable: str = "scaling_factors",
):
    """
    Perturb weights by cell type-specific scaling factors drawn from a log-normal distribution.

    Only perturbs the weights corresponding to scaling factors that will be optimized.

    Args:
        weights: Recurrent weight matrix of shape (N, N).
        feedforward_weights: Feedforward weight matrix of shape (N_in, N).
        cell_type_indices: Array of shape (N,) indicating cell type index for each neuron.
        feedforward_cell_type_indices: Array of shape (N_in,) indicating cell type index for each input neuron.
        variance: Variance of the log-normal distribution used for scaling.
        optimisable: Optimization mode - one of "scaling_factors", "scaling_factors_recurrent",
            or "scaling_factors_feedforward". Determines which weights to perturb.

    Returns:
        perturbed_weights: Perturbed recurrent weight matrix (or original if not optimizing recurrent).
        perturbed_feedforward_weights: Perturbed feedforward weight matrix (or original if not optimizing feedforward).
        target_scaling_factors: Reciprocal of applied scaling factors, or None if not optimizing recurrent.
        target_feedforward_scaling_factors: Reciprocal of applied scaling factors, or None if not optimizing feedforward.
    """
    n_types = len(np.unique(cell_type_indices))
    n_ff_types = len(np.unique(feedforward_cell_type_indices))
    # Log-normal with mean 1: mu = -sigma^2 / 2.
    sigma = np.sqrt(variance)
    mu = -(sigma**2) / 2.0

    def perturb(w, pre_types, n_pre_types, enabled):
        """Scale each (pre type, post type) block by one draw; return the targets."""
        if not enabled:
            return w.copy(), None
        scalings = np.random.lognormal(
            mean=mu, sigma=sigma, size=(n_pre_types, n_types)
        )
        # The student must learn the reciprocal to undo the perturbation.
        return w * scalings[np.ix_(pre_types, cell_type_indices)], 1.0 / scalings

    perturbed_weights, target_scaling_factors = perturb(
        weights,
        cell_type_indices,
        n_types,
        optimisable in ("scaling_factors", "scaling_factors_recurrent"),
    )
    perturbed_feedforward_weights, target_feedforward_scaling_factors = perturb(
        feedforward_weights,
        feedforward_cell_type_indices,
        n_ff_types,
        optimisable in ("scaling_factors", "scaling_factors_feedforward"),
    )
    return (
        perturbed_weights,
        perturbed_feedforward_weights,
        target_scaling_factors,
        target_feedforward_scaling_factors,
    )
