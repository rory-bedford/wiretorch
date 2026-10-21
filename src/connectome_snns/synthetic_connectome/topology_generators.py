"""Generate graph topologies with cell type assignments.

Each generator returns a boolean adjacency matrix ``A[source, target]`` whose edge
probabilities come from a matrix of connection probabilities between cell types
(any number of types, not just excitatory/inhibitory). Randomness comes from NumPy's
global generator, so ``np.random.seed`` fixes the graph.

Typical workflow:
    >>> # First assign cell types
    >>> cell_proportions = [0.8, 0.2]  # 80% type 0, 20% type 1
    >>> source_cell_types = assign_cell_types(100, cell_proportions, seed=42)
    >>> target_cell_types = assign_cell_types(100, cell_proportions, seed=43)
    >>>
    >>> # Then generate connectivity with matrix-based probabilities
    >>> conn_probs = [[0.05, 0.05], [0.3, 0.05]]  # 2x2 matrix for 2 cell types
    >>> adj = sparse_graph_generator(
    ...     source_cell_types, target_cell_types, conn_probs
    ... )
    >>> # adj is a boolean adjacency matrix
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

METHODS = ("erdos-renyi", "configuration")

#: Rounds of stub swapping before the configuration model gives up.
MAX_SWAP_ROUNDS = 10_000


def sparse_graph_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    conn_matrix: List[List[float]],
    allow_self_loops: bool = False,
    method: str = "erdos-renyi",
) -> BoolArray:
    """
    Generate a sparse random graph between typed neurons.

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        conn_matrix (List[List[float]]): Matrix of connection probabilities from source to target cell types.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself.
            Only applies to a recurrent graph (see Note); a feedforward graph keeps
            its edges from input i to neuron i.
        method (str): Method for graph generation. Either "erdos-renyi" or "configuration".

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Raises:
        ValueError: For an unknown method, a cell type without a row/column of
            ``conn_matrix``, or a configuration graph that asks for more edges than
            exist.
        RuntimeError: If the configuration model cannot remove repeated edges and
            self-loops (a nearly complete graph).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and
        target_cell_types: a graph is recurrent when the two are equal.
        "erdos-renyi" samples each edge independently. "configuration" places exactly
        ``round(p * n_source * n_target)`` edges per cell-type pair, with in- and
        out-degrees as even as possible.
    """
    _check_method(method)
    probabilities = _probabilities(
        conn_matrix, source_cell_types, target_cell_types, "conn_matrix"
    )
    allow_self_loops = allow_self_loops or not _is_recurrent(
        source_cell_types, target_cell_types
    )
    return _sparse_graph(
        source_cell_types, target_cell_types, probabilities, allow_self_loops, method
    )


def _sparse_graph(
    source_cell_types, target_cell_types, probabilities, allow_self_loops, method
) -> BoolArray:
    """``sparse_graph_generator`` on checked probabilities; ``allow_self_loops=False``
    removes every edge from position i to position i."""
    n_source, n_target = len(source_cell_types), len(target_cell_types)
    if method == "erdos-renyi":
        prob_matrix = probabilities[source_cell_types[:, None], target_cell_types]
        adjacency = np.random.random((n_source, n_target)) < prob_matrix
        if not allow_self_loops:  # as in the configuration model: no i -> i edges
            np.fill_diagonal(adjacency, False)
        return adjacency

    adjacency = np.zeros((n_source, n_target), dtype=bool)
    for src_type, tgt_type in np.ndindex(probabilities.shape):
        conn_prob = probabilities[src_type, tgt_type]
        src_indices = np.flatnonzero(source_cell_types == src_type)
        tgt_indices = np.flatnonzero(target_cell_types == tgt_type)
        if conn_prob == 0 or len(src_indices) == 0 or len(tgt_indices) == 0:
            continue
        out_stubs, in_stubs = _configuration_edges(
            src_indices, tgt_indices, conn_prob, allow_self_loops
        )
        adjacency[out_stubs, in_stubs] = True
    return adjacency


def dense_graph_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    allow_self_loops: bool = False,
) -> BoolArray:
    """
    Generate a fully connected (dense) graph with cell types.

    All possible edges exist except self-loops (unless allow_self_loops=True).
    This is implemented as calling sparse_graph_generator with probability 1.0.

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself.

    Returns:
        BoolArray: Boolean adjacency matrix of shape (len(source_cell_types), len(target_cell_types)).

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
    """
    n_source_types = source_cell_types.max() + 1
    n_target_types = target_cell_types.max() + 1
    conn_matrix = [[1.0] * n_target_types for _ in range(n_source_types)]

    return sparse_graph_generator(
        source_cell_types,
        target_cell_types,
        conn_matrix,
        allow_self_loops,
        method="erdos-renyi",
    )


def assembly_generator(
    source_cell_types: IntArray,
    target_cell_types: IntArray,
    num_assemblies: int,
    conn_within: List[List[float]],
    conn_between: List[List[float]],
    allow_self_loops: bool = False,
    method: str = "erdos-renyi",
    between_diagonal: bool = False,
) -> Tuple[BoolArray, IntArray]:
    """
    Generate a graph with assembly structure and cell types.

    Neurons are split, in index order, into ``num_assemblies`` equal assemblies;
    pairs in the same assembly connect with ``conn_within``, pairs in different
    assemblies with ``conn_between`` (both cell type x cell type probabilities).

    Args:
        source_cell_types (IntArray): Cell type indices for source neurons.
        target_cell_types (IntArray): Cell type indices for target neurons.
        num_assemblies (int): Number of assemblies to create.
        conn_within (List[List[float]]): Matrix of connection probabilities within assemblies between cell types.
        conn_between (List[List[float]]): Matrix of connection probabilities between assemblies between cell types.
        allow_self_loops (bool): Whether to allow connections from neuron i to itself
            (in a recurrent graph only, as for ``sparse_graph_generator``).
        method (str): Method for graph generation. Either "erdos-renyi" or "configuration".
        between_diagonal (bool): "configuration" only: also place the edges k -> k of
            each between-assembly block (neuron ``i*size + k`` to ``j*size + k``, two
            different cells). Off by default, which never places them, so existing
            graphs regenerate identically.

    Returns:
        Tuple[BoolArray, IntArray]:
            - Boolean adjacency matrix of shape ``(n_source_used, n_target_used)``:
              only the neurons that fit into equal assemblies,
              ``n_used = (n // num_assemblies) * num_assemblies``
            - 0-indexed array of assembly IDs for all source neurons (shape:
              ``len(source_cell_types)``); neurons left out of the assemblies get -1

    Raises:
        ValueError: For an unknown method, more assemblies than neurons, mismatched
            ``conn_within`` / ``conn_between`` shapes, or cell types without
            probabilities.

    Note:
        For recurrent connections, pass the same array for both source_cell_types and target_cell_types.
        The "configuration" method places within- and between-assembly edges with
        the configuration model, one block per (source assembly, target assembly);
        between-assembly blocks drop their diagonal unless ``between_diagonal``.
    """
    _check_method(method)
    allow_self_loops = allow_self_loops or not _is_recurrent(
        source_cell_types, target_cell_types
    )
    n_source, n_target = len(source_cell_types), len(target_cell_types)
    if num_assemblies > min(n_source, n_target):
        raise ValueError(
            f"num_assemblies ({num_assemblies}) cannot exceed "
            f"min(n_source, n_target) = {min(n_source, n_target)}"
        )
    within = _probabilities(
        conn_within, source_cell_types, target_cell_types, "conn_within"
    )
    between = np.array(conn_between)
    if within.shape != between.shape:
        raise ValueError(
            f"conn_within {within.shape} and conn_between {between.shape} must have "
            "the same shape"
        )

    source_size = n_source // num_assemblies
    target_size = n_target // num_assemblies
    source_assembly_id = np.repeat(np.arange(num_assemblies), source_size)
    target_assembly_id = np.repeat(np.arange(num_assemblies), target_size)
    n_source_used, n_target_used = len(source_assembly_id), len(target_assembly_id)

    if method == "erdos-renyi":
        source_types = source_cell_types[:n_source_used, None]
        target_types = target_cell_types[:n_target_used]
        prob_matrix = np.where(
            source_assembly_id[:, None] == target_assembly_id,
            within[source_types, target_types],
            between[source_types, target_types],
        )
        adjacency = np.random.random((n_source_used, n_target_used)) < prob_matrix
    else:
        adjacency = np.zeros((n_source_used, n_target_used), dtype=bool)
        sources = [
            np.arange(i * source_size, (i + 1) * source_size)
            for i in range(num_assemblies)
        ]
        targets = [
            np.arange(i * target_size, (i + 1) * target_size)
            for i in range(num_assemblies)
        ]
        # Within-assembly blocks first, then every between-assembly block
        blocks = [(i, i, within, allow_self_loops) for i in range(num_assemblies)]
        blocks += [
            (i, j, between, between_diagonal)
            for i in range(num_assemblies)
            for j in range(num_assemblies)
            if i != j
        ]
        for i, j, conn, self_loops in blocks:
            adjacency[np.ix_(sources[i], targets[j])] = _sparse_graph(
                source_cell_types[sources[i]],
                target_cell_types[targets[j]],
                conn,
                self_loops,
                "configuration",
            )

    if not allow_self_loops and n_source_used == n_target_used:
        np.fill_diagonal(adjacency, False)

    assembly_ids = np.full(n_source, -1, dtype=np.int32)
    assembly_ids[:n_source_used] = source_assembly_id
    return adjacency, assembly_ids


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_recurrent(source_cell_types: IntArray, target_cell_types: IntArray) -> bool:
    """Whether source and target are one population (so i -> i is a self-loop)."""
    return source_cell_types is target_cell_types or np.array_equal(
        source_cell_types, target_cell_types
    )


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(
            f"method must be 'erdos-renyi' or 'configuration', got '{method}'"
        )


def _probabilities(conn, source_cell_types, target_cell_types, name) -> FloatArray:
    """``conn`` as an array, checked to cover every source and target cell type."""
    probabilities = np.array(conn)
    n_source_types, n_target_types = probabilities.shape
    if source_cell_types.max() >= n_source_types:
        raise ValueError(
            f"source_cell_types max {source_cell_types.max()} exceeds {name} rows "
            f"{n_source_types}"
        )
    if target_cell_types.max() >= n_target_types:
        raise ValueError(
            f"target_cell_types max {target_cell_types.max()} exceeds {name} columns "
            f"{n_target_types}"
        )
    return probabilities


def _configuration_edges(
    src_indices: IntArray, tgt_indices: IntArray, conn_prob: float, allow_self_loops
) -> Tuple[IntArray, IntArray]:
    """Edges ``(out_stubs[k], in_stubs[k])`` of one cell-type block.

    Each neuron gets an (as near as possible) equal share of the
    ``round(p * n_src * n_tgt)`` stubs: the shuffled indices are tiled and cut to
    length, then both stub lists are shuffled and paired. Repeated edges and
    self-loops are then broken up by swapping their in-stubs with random others.
    """
    n_src, n_tgt = len(src_indices), len(tgt_indices)
    total_edges = int(np.round(conn_prob * n_src * n_tgt))
    available = n_src * n_tgt
    if not allow_self_loops:
        available -= len(np.intersect1d(src_indices, tgt_indices))
    if total_edges > available:
        raise ValueError(
            f"A configuration graph block with {n_src} x {n_tgt} neurons cannot have "
            f"{total_edges} edges without self-loops (at most {available})"
        )

    # Stubs hold positions in src_indices / tgt_indices (shuffling positions draws
    # the same permutation as shuffling the indices themselves).
    shuffled_src = np.arange(n_src)
    np.random.shuffle(shuffled_src)
    shuffled_tgt = np.arange(n_tgt)
    np.random.shuffle(shuffled_tgt)
    out_stubs = np.tile(shuffled_src, int(np.ceil(total_edges / n_src)))[:total_edges]
    in_stubs = np.tile(shuffled_tgt, int(np.ceil(total_edges / n_tgt)))[:total_edges]
    np.random.shuffle(out_stubs)
    np.random.shuffle(in_stubs)

    own_target = None  # position in tgt_indices of each source neuron itself, or -1
    if not allow_self_loops:
        own_target = np.full(n_src, -1)
        shared, src_pos, tgt_pos = np.intersect1d(
            src_indices, tgt_indices, return_indices=True
        )
        own_target[src_pos] = tgt_pos

    conflicts = _Conflicts(out_stubs, in_stubs, n_tgt, own_target)
    for _ in range(MAX_SWAP_ROUNDS):
        positions = conflicts.positions
        if len(positions) == 0:
            return src_indices[out_stubs], tgt_indices[in_stubs]
        partners = np.random.randint(0, len(in_stubs), size=len(positions))
        touched = np.unique(np.concatenate([positions, partners]))
        before = conflicts.keys[touched]
        _swap_in_order(in_stubs, positions, partners)
        conflicts.update(touched, before)
    raise RuntimeError(
        f"The configuration model could not remove repeated edges and self-loops "
        f"after {MAX_SWAP_ROUNDS} rounds ({total_edges} edges in a {n_src} x {n_tgt} "
        "block); use a lower connection probability or 'erdos-renyi'."
    )


class _Conflicts:
    """The edges a configuration graph must still move, kept up to date as stubs swap.

    ``positions`` lists the self-loops (if not allowed), then the repeats of an
    earlier edge, each in ascending order. After the first round only the edges that
    share a key with a swapped edge are re-examined, so a round costs one pass over
    the edges rather than a sort of them.
    """

    #: Blocks with more possible edges than this re-sort every round instead (the
    #: incremental update keeps one flag per possible edge).
    MAX_TRACKED_EDGES = 1 << 28

    def __init__(self, out_stubs, in_stubs, n_tgt, own_target):
        self.out_stubs, self.in_stubs, self.n_tgt = out_stubs, in_stubs, n_tgt
        self.own_target = own_target
        self.keys = out_stubs.astype(np.int64) * n_tgt + in_stubs
        self.self_loops = self._self_loops(np.arange(len(in_stubs)))
        self.repeats = _repeats(np.arange(len(in_stubs)), self.keys)
        n_keys = (int(out_stubs.max()) + 1) * n_tgt if len(out_stubs) else 0
        self.affected = (
            np.zeros(n_keys, dtype=bool) if n_keys <= self.MAX_TRACKED_EDGES else None
        )

    @property
    def positions(self) -> IntArray:
        return np.concatenate([self.self_loops, self.repeats])

    def update(self, touched: IntArray, keys_before: IntArray) -> None:
        """Account for new in-stubs at the (sorted, unique) ``touched`` positions."""
        self.keys[touched] = (
            self.out_stubs[touched].astype(np.int64) * self.n_tgt
            + self.in_stubs[touched]
        )
        self.self_loops = np.union1d(
            self.self_loops[~np.isin(self.self_loops, touched)],
            self._self_loops(touched),
        )
        if self.affected is None:
            self.repeats = _repeats(np.arange(len(self.keys)), self.keys)
            return
        # Only edges sharing a key with a touched edge can change status.
        changed = np.concatenate([keys_before, self.keys[touched]])
        self.affected[changed] = True
        candidates = np.flatnonzero(self.affected[self.keys])
        self.affected[changed] = False
        self.repeats = np.union1d(
            self.repeats[~np.isin(self.repeats, candidates)],
            _repeats(candidates, self.keys[candidates]),
        )

    def _self_loops(self, positions: IntArray) -> IntArray:
        if self.own_target is None:
            return np.array([], dtype=int)
        own = self.own_target[self.out_stubs[positions]]
        return positions[own == self.in_stubs[positions]]


def _repeats(positions: IntArray, keys: IntArray) -> IntArray:
    """The ``positions`` (ascending) whose key equals that of an earlier position."""
    order = np.argsort(keys, kind="stable")  # equal keys keep their order
    repeated = order[1:][keys[order[1:]] == keys[order[:-1]]]
    return np.sort(positions[repeated])


def _swap_in_order(stubs, positions, partners) -> None:
    """Swap ``stubs[positions[k]]`` with ``stubs[partners[k]]`` for k = 0, 1, ...

    Vectorised when no position is touched twice (the usual case), else in order.
    """
    touched = np.concatenate([positions, partners])
    if len(np.unique(touched)) == len(touched):
        stubs[positions], stubs[partners] = stubs[partners], stubs[positions]
        return
    for i, j in zip(positions.tolist(), partners.tolist()):
        stubs[i], stubs[j] = stubs[j], stubs[i]
