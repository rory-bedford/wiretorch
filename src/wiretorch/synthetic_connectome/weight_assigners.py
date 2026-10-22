"""Assign continuous weights to graph edges.

This module provides functions to assign continuous weight values to boolean
adjacency matrices while preserving Dale's law (excitatory/inhibitory signs).
Parameters are specified using matrices for different cell type combinations.

Typical workflow:
    >>> connectivity_graph, cell_indices = sparse_graph_generator(...)
    >>> w_mu_matrix = [[1.0, 1.2], [2.4, 1.6]]  # 2x2 for 2 cell types
    >>> weights = assign_weights_lognormal(connectivity_graph, cell_indices, cell_indices,
    ...                                   [1, -1], w_mu_matrix, w_sigma_matrix)
"""

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def assign_weights_lognormal(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: IntArray | None,
    w_mu_matrix: FloatArray,
    w_sigma_matrix: FloatArray,
    parameter_space: str = "log",
) -> FloatArray:
    """
    Assign log-normal distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (IntArray | None): Signs for each SOURCE cell type (+1/-1) as a numpy array.
            This is used to determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types.
            If None, weights will be unsigned (conductance-based model).
        w_mu_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) for parameters:
            - If parameter_space="log" (default): mu parameters of log-normal distribution (in log-space)
            - If parameter_space="linear": desired mean of resulting distribution (in linear space)
        w_sigma_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) for parameters:
            - If parameter_space="log" (default): sigma parameters of log-normal distribution (in log-space)
            - If parameter_space="linear": desired variance of resulting distribution (in linear space)
        parameter_space (str, optional): Specifies how to interpret the input parameters:
            - "log": Parameters are in log space (mu, sigma)
            - "linear": Parameters are in linear space (mean, variance)
            Defaults to "log".

    Returns:
        FloatArray: Weighted adjacency matrix with log-normal magnitudes and appropriate signs.
    """
    assert parameter_space.lower() in ("log", "linear"), (
        'parameter_space must be either "log" or "linear"'
    )
    w_mu_matrix, w_sigma_matrix = np.asarray(w_mu_matrix), np.asarray(w_sigma_matrix)
    if parameter_space.lower() == "linear":
        # Mean m and variance v of a log-normal give its log-space parameters:
        # sigma^2 = log(1 + v / m^2), mu = log(m) - sigma^2 / 2.
        mean = np.maximum(w_mu_matrix, np.finfo(float).eps)
        log_sigma_squared = np.log(1 + np.maximum(w_sigma_matrix / (mean * mean), 0))
        w_mu_matrix = np.log(mean) - log_sigma_squared / 2
        w_sigma_matrix = np.sqrt(log_sigma_squared)
    return _assign(
        connectivity_graph,
        source_cell_indices,
        target_cell_indices,
        cell_type_signs,
        {"w_mu_matrix": w_mu_matrix, "w_sigma_matrix": w_sigma_matrix},
        np.random.lognormal,
    )


def assign_weights_gamma(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: IntArray | None,
    shape_matrix: FloatArray,
    scale_matrix: FloatArray,
) -> FloatArray:
    """
    Assign gamma-distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (IntArray | None): Signs for each SOURCE cell type (+1/-1) as a numpy array.
            This is used to determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types.
            If None, weights will be unsigned (conductance-based model).
        shape_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of gamma shape parameters (k).
        scale_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of gamma scale parameters (θ).

    Returns:
        FloatArray: Weighted adjacency matrix with gamma-distributed magnitudes and appropriate signs.
    """
    return _assign(
        connectivity_graph,
        source_cell_indices,
        target_cell_indices,
        cell_type_signs,
        {"shape_matrix": shape_matrix, "scale_matrix": scale_matrix},
        np.random.gamma,
    )


def assign_weights_uniform(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: IntArray | None,
    low_matrix: FloatArray,
    high_matrix: FloatArray,
) -> FloatArray:
    """
    Assign uniformly distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (IntArray | None): Signs for each SOURCE cell type (+1/-1) as a numpy array.
            This is used to determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types.
            If None, weights will be unsigned (conductance-based model).
        low_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of uniform distribution lower bounds.
        high_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of uniform distribution upper bounds.

    Returns:
        FloatArray: Weighted adjacency matrix with uniformly distributed magnitudes and appropriate signs.
    """
    return _assign(
        connectivity_graph,
        source_cell_indices,
        target_cell_indices,
        cell_type_signs,
        {"low_matrix": low_matrix, "high_matrix": high_matrix},
        np.random.uniform,
    )


def _assign(graph, source_types, target_types, signs, matrices, sample):
    """Draw one weight per possible edge from ``sample(*per-edge parameters)``.

    ``matrices`` maps names to ``(n_source_types, n_target_types)`` parameter matrices;
    each is looked up per edge by its source and target types. Magnitudes are signed
    by source type (Dale's law) if ``signs`` is given, then masked by ``graph``.
    """
    n_source, n_target = graph.shape
    assert source_types.shape[0] == n_source, (
        "Mismatch between source indices and graph rows."
    )
    assert target_types.shape[0] == n_target, (
        "Mismatch between target indices and graph columns."
    )
    if signs is not None:
        assert isinstance(signs, np.ndarray), (
            "cell_type_signs must be a numpy array or None."
        )
        n_source_types = len(signs)
    else:
        n_source_types = max(source_types) + 1
    n_target_types = max(target_types) + 1
    for name, matrix in matrices.items():
        assert np.shape(matrix) == (n_source_types, n_target_types), (
            f"{name} must be {n_source_types}x{n_target_types} for {n_source_types} "
            f"source types and {n_target_types} target types."
        )
    per_edge = [np.asarray(m)[source_types][:, target_types] for m in matrices.values()]
    weights = sample(*per_edge)
    if signs is not None:
        weights = weights * signs[source_types][:, None]
    return weights * graph
