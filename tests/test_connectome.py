"""Synthetic connectome generators respect their topology arguments."""

import numpy as np
import pytest

from connectome_snns.synthetic_connectome.topology_generators import (
    dense_graph_generator,
    sparse_graph_generator,
)

TYPES = np.repeat([0, 1], [40, 10])


@pytest.mark.parametrize("method", ["erdos-renyi", "configuration"])
@pytest.mark.parametrize("allow", [False, True])
def test_self_loops(method, allow):
    np.random.seed(0)
    adjacency = sparse_graph_generator(
        TYPES, TYPES, [[0.5, 0.5], [0.5, 0.5]], allow, method
    )
    if allow:
        assert np.diag(adjacency).any()
    else:
        assert not np.diag(adjacency).any()


def test_erdos_renyi_draws_unchanged_off_the_diagonal():
    np.random.seed(1)
    with_loops = sparse_graph_generator(TYPES, TYPES, [[0.3, 0.3], [0.3, 0.3]], True)
    np.random.seed(1)
    without = sparse_graph_generator(TYPES, TYPES, [[0.3, 0.3], [0.3, 0.3]], False)
    off = ~np.eye(len(TYPES), dtype=bool)
    np.testing.assert_array_equal(with_loops[off], without[off])


def test_dense_graph():
    adjacency = dense_graph_generator(TYPES, TYPES)
    assert adjacency.sum() == len(TYPES) ** 2 - len(TYPES)


@pytest.mark.parametrize("assigner", ["lognormal", "gamma", "uniform"])
def test_weight_assigners_sign_and_mask(assigner):
    from connectome_snns.synthetic_connectome import weight_assigners as wa

    rng = np.random.default_rng(0)
    graph = rng.random((50, 50)) < 0.3
    params = {
        "lognormal": (np.full((2, 2), -3.0), np.full((2, 2), 0.5)),
        "gamma": (np.full((2, 2), 2.0), np.full((2, 2), 0.01)),
        "uniform": (np.full((2, 2), 0.01), np.full((2, 2), 0.02)),
    }[assigner]
    w = getattr(wa, f"assign_weights_{assigner}")(
        graph, TYPES, TYPES, np.array([1, -1]), *params
    )
    assert np.all(w[~graph] == 0)
    assert np.all(w[(TYPES == 0)[:, None] & graph] > 0)
    assert np.all(w[(TYPES == 1)[:, None] & graph] < 0)


# ── Fingerprints: a seed generates the same connectome as before the tidy ────

_E_I = np.repeat([0, 1], [160, 40])
_FF = np.zeros(120, dtype=int)
_P = [[0.1, 0.2], [0.3, 0.15]]
_WITHIN = [[0.3, 0.3], [0.3, 0.3]]

GENERATED = {
    "er_rec": (
        lambda g: g.sparse_graph_generator(_E_I, _E_I, _P),
        ["ee80a16cd8abd81b"],
    ),
    "er_rec_loops": (
        lambda g: g.sparse_graph_generator(_E_I, _E_I, _P, True),
        ["bac390c4691123ca"],
    ),
    "er_ff": (
        lambda g: g.sparse_graph_generator(_FF, _E_I, [[0.05, 0.1]]),
        ["580f2c1b7f2b435c"],  # = the old code with allow_self_loops=True,
    ),
    "conf_rec": (
        lambda g: g.sparse_graph_generator(_E_I, _E_I, _P, False, "configuration"),
        ["7e544c24073facf5"],
    ),
    "conf_rec_loops": (
        lambda g: g.sparse_graph_generator(_E_I, _E_I, _P, True, "configuration"),
        ["0dcc3989a74c0851"],
    ),
    "conf_dense": (
        lambda g: g.sparse_graph_generator(
            _E_I, _E_I, [[0.4, 0.45], [0.4, 0.4]], False, "configuration"
        ),
        ["08dbc33072f2f06e"],
    ),
    "conf_ff": (
        lambda g: g.sparse_graph_generator(
            _FF, _E_I, [[0.05, 0.1]], False, "configuration"
        ),
        ["106acec74f7975ac"],  # likewise,
    ),
    "dense": (
        lambda g: g.dense_graph_generator(_E_I, _E_I),
        ["6098f0c1d63555f4"],
    ),
    "asm_er": (
        lambda g: g.assembly_generator(_E_I, _E_I, 3, _WITHIN, _P),
        ["822a9135eb5f4148", "7e2718bc37df6f98"],
    ),
    "asm_conf": (
        lambda g: g.assembly_generator(
            _E_I, _E_I, 3, _WITHIN, _P, False, "configuration"
        ),
        ["86b92471e35722f7", "7e2718bc37df6f98"],
    ),
    "asm_conf_loops": (
        lambda g: g.assembly_generator(
            _E_I, _E_I, 4, [[0.5, 0.3], [0.3, 0.3]], _P, True, "configuration"
        ),
        ["a10329de722bf43e", "76a7ff6bd5951857"],
    ),
}


def _fingerprint(array: np.ndarray) -> str:
    import hashlib

    data = np.ascontiguousarray(array).tobytes() + str(array.shape).encode()
    return hashlib.sha256(data + str(array.dtype).encode()).hexdigest()[:16]


def test_between_diagonal_places_the_dropped_edges():
    """Between-assembly blocks drop edges k -> k unless asked: they join two cells."""
    from connectome_snns.synthetic_connectome.topology_generators import (
        assembly_generator,
    )

    types = np.zeros(8, dtype=int)

    def between_block(probability, flag):
        adjacency, _ = assembly_generator(
            types, types, 2, [[0.0]], [[probability]], False, "configuration", flag
        )
        assert not np.any(np.diag(adjacency))
        return adjacency[:4, 4:]

    np.random.seed(0)
    assert np.all(between_block(1.0, True))
    with pytest.raises(ValueError, match="at most 12"):
        between_block(1.0, False)
    assert np.array_equal(between_block(0.75, False), ~np.eye(4, dtype=bool))


@pytest.mark.parametrize("method", ["erdos-renyi", "configuration"])
def test_feedforward_graphs_keep_edges_from_input_i_to_neuron_i(method):
    """Self-loops are a recurrent notion: input i and neuron i are different cells."""
    from connectome_snns.synthetic_connectome import topology_generators

    graphs = []
    for allow in (False, True):
        np.random.seed(3)
        graphs.append(
            topology_generators.sparse_graph_generator(
                _FF, _E_I, [[0.5, 0.5]], allow, method
            )
        )
    np.testing.assert_array_equal(*graphs)
    assert graphs[0].diagonal().any()


@pytest.mark.parametrize("case", GENERATED)
def test_generated_connectomes_unchanged(case):
    from connectome_snns.synthetic_connectome import topology_generators

    make, expected = GENERATED[case]
    np.random.seed(7)
    out = make(topology_generators)
    parts = out if isinstance(out, tuple) else (out,)
    assert [_fingerprint(p) for p in parts] == expected


@pytest.mark.parametrize(
    "kwargs",
    [dict(method="random"), dict(conn_matrix=[[0.1]])],  # [[0.1]]: no row for type 1
)
def test_sparse_graph_rejects_bad_arguments(kwargs):
    args = dict(
        source_cell_types=TYPES, target_cell_types=TYPES, conn_matrix=[[0.1] * 2] * 2
    )
    with pytest.raises(ValueError):
        sparse_graph_generator(**{**args, **kwargs})


def test_assembly_generator_rejects_bad_arguments():
    from connectome_snns.synthetic_connectome.topology_generators import (
        assembly_generator,
    )

    probabilities = [[0.1] * 2] * 2
    with pytest.raises(ValueError):  # mismatched within / between shapes
        assembly_generator(TYPES, TYPES, 2, probabilities, [[0.1]])
    with pytest.raises(ValueError):  # more assemblies than neurons
        assembly_generator(TYPES, TYPES, len(TYPES) + 1, probabilities, probabilities)


def test_impossible_configuration_graph_raises_instead_of_hanging():
    """A type pair asking for more edges than exist without self-loops."""
    with pytest.raises(ValueError, match="without self-loops"):
        sparse_graph_generator(
            TYPES, TYPES, [[1.0, 0.0], [0.0, 0.0]], False, "configuration"
        )


def test_configuration_model_large_block_path_matches(monkeypatch):
    """Blocks too big to track incrementally re-sort every round: same graph."""
    from connectome_snns.synthetic_connectome import topology_generators

    args = (_E_I, _E_I, [[0.4, 0.45], [0.4, 0.4]], False, "configuration")
    np.random.seed(5)
    tracked = topology_generators.sparse_graph_generator(*args)
    monkeypatch.setattr(topology_generators._Conflicts, "MAX_TRACKED_EDGES", 0)
    np.random.seed(5)
    np.testing.assert_array_equal(
        topology_generators.sparse_graph_generator(*args), tracked
    )
