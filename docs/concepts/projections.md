# Projections

A network's connectivity is given per **(source cell type, target cell type)** pair:
a dict of {py:class}`~wiretorch.network_simulators.projections.Projection` objects,

```python
rec_projections = {
    ("excitatory", "excitatory"): wt.FrozenProjection(w_ee),   # (n_E, n_E)
    ("excitatory", "inhibitory"): wt.FrozenProjection(w_ei),   # (n_E, n_I)
    ("inhibitory", "excitatory"): wt.ScalingFactorProjection(w_ie),
    ...
}
```

Each projection holds a non-negative block of weights of shape `(sources of that type,
targets of that type)`, and its class decides what is trained. Pairs that are left out
have no connections.

## Kinds of projection

`FrozenProjection(weights)`
: Fixed weights, no parameters.

`ScalingFactorProjection(connectome, init_sf=1.0)`
: The weights are `exp(log_sf) × connectome`, a fixed connectome with one trainable
  gain for the whole block.

`LowRankProjection(n_source, n_target, rank, U_init=None, V_init=None, mask=None)`
: The weights are `exp(U V) × mask`, with `U` of shape `(n_source, rank)` and `V` of
  shape `(rank, n_target)` trained.

`FullRankProjection(init_weights, mask=None)`
: The weights are `exp(log_weights) × mask`, with every weight in the mask trained.

Trained weights are parametrised through an exponential so that they stay positive.
The mask, usually the connectome's nonzero entries, keeps absent connections at zero.

## Building them from full matrices

If you have full weight matrices, these helpers split them into blocks by cell type:

```python
rec, ff = wt.make_frozen_projections(
    w_rec, w_ff, cell_type_indices, ff_cell_type_indices, names, ff_names
)
rec, ff = wt.make_scaling_factor_projections(
    w_rec, w_ff, cell_type_indices, ff_cell_type_indices, names, ff_names
)
```

## Implementation

The network groups projections by source cell type. Frozen weights are prepared once,
when the network is built. Scaling-factor projections keep the connectome fixed and
apply the gains to the synaptic kernel on each forward call, which costs almost
nothing. Low- and full-rank weights are rebuilt on every forward call so that gradients
can reach them.

For this reason, all projections from one source cell type on one side of the network
must be of the same kind. A projection object can be shared between layers, for example
both layers of a `TwoLayerSNN`, and then its parameters and gradients are shared too.
