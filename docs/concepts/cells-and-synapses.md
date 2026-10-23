# Cell types and synapse types

## Cell types

Every neuron and every input row belongs to a **cell type**, given as an integer per
neuron (`cell_type_indices`, `cell_type_indices_FF`). Each type is described once, in
`cell_params` (the network's neurons) or `cell_params_FF` (its inputs):

```python
cell_params = [
    dict(name="excitatory", cell_id=0, tau_mem=20.0, ...),
    dict(name="inhibitory", cell_id=1, tau_mem=10.0, ...),
]
```

`cell_id` matches the integers in `cell_type_indices`, and projections and `learn=`
refer to the type by `name`. Input cell types only need a name and an id. The other
entries depend on the neuron model:

| Neuron model | Required | Optional (default) |
|---|---|---|
| Conductance-based | `tau_mem`, `theta`, `U_reset`, `E_L`, `g_L` | `tau_ref` (0), `adapt` (0), `tau_adapt` (100), `E_adapt` (−90) |
| Current-based | `tau_mem` | `theta` (1), `U_reset` (0), `U_rest` (0), `R` (1), `tau_ref` (0), `adapt` (0), `tau_adapt` (100) |
| Rate | `tau_mem`, `nonlinearity` | `theta` (0), `gain` (1) |

`tau_ref` is the refractory period in ms, and `adapt`, `tau_adapt` (ms) and `E_adapt`
(mV) set spike-frequency adaptation (see {doc}`neuron-models`). Short-term plasticity
is not a cell or synapse parameter: it belongs to a (source, target) pair, and the
networks take it alongside the projections (`rec_stp`, `ff_stp`).

## Synapse types

The effect of a spike on its targets is set by the synapse types of its cell type,
listed in `synapse_params` (recurrent) and `synapse_params_FF` (inputs):

```python
synapse_params = [
    dict(name="AMPA", synapse_id=0, cell_id=0, tau_rise=0.1, tau_decay=5.0, E_syn=0.0, g_bar=4.0),
    dict(name="NMDA", synapse_id=1, cell_id=0, tau_rise=10.0, tau_decay=60.0, E_syn=0.0, g_bar=0.5, mg=1.0),
    dict(name="GABA_A", synapse_id=2, cell_id=1, tau_rise=0.5, tau_decay=6.0, E_syn=-70.0, g_bar=8.0),
]
```

`cell_id` is the cell type that makes the synapse. A cell type can make several, as the
excitatory cells make AMPA and NMDA here. Each neuron keeps one synaptic variable per
synapse type (a conductance with rise and decay components, or a current), and a
connection's weight drives every synapse type its source makes.

| Neuron model | Synapse parameters |
|---|---|
| Conductance-based | `tau_rise`, `tau_decay`, `E_syn`, `g_bar`, optional `g_clip` (a cap on the conductance), `mg`, `mg_slope`, `mg_k` (a voltage-dependent Mg²⁺ block, for NMDA) and `saturation` (receptor saturation) |
| Current-based | `tau_syn` (0: no memory), `type` (`"excitatory"` or `"inhibitory"`) |
| Rate | `tau_syn` (0: no memory), `type` |

`g_clip` is not needed for numerical stability: the voltage step is exact for the
conductances of the step, however large ({doc}`neuron-models`). It remains an optional
cap on a synapse type's conductance.

An NMDA synapse type gets its voltage dependence from `mg`, the extracellular
[Mg²⁺] in mM (1.0 is physiological; the default, 0, means no block), and its
saturation from `saturation`, the fraction of a source's closed receptors one spike
opens (0.63 for Brunel & Wang's NMDA; the default, 0, means none). With saturation,
`g_bar` is the conductance with every receptor open. See {doc}`neuron-models`.

## Dale's law

Weights are non-negative. Whether a connection excites or inhibits is a property of its
synapse type: the reversal potential `E_syn` for conductance-based synapses, and `type`
for current-based and rate synapses. A neuron is excitatory or inhibitory through the
synapse types of its cell type.

A model with signed weights has to be split into excitatory and inhibitory parts carried
by different cell types. The {doc}`../tutorials/ring_attractor` shows how.

## Shared synaptic variables

In a recurrent network, a synapse type is identified by its source cell type and its
name. If the same synapse type arrives both from the network's own neurons and through
`ff_projections` from inputs of the same cell type, the two share one variable. A
feedforward network keeps them separate.
