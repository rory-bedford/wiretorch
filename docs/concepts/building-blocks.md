# Choosing your building blocks

A model is defined by three choices: the neuron model, whether the network is recurrent
or feedforward, and, for each block of connections between two cell types, whether the
weights are fixed or trained. You can also choose to learn physiological parameters
such as time constants and thresholds.

## The network class

The neuron model and the topology together give the class:

|  | Recurrent | Feedforward |
|---|---|---|
| Conductance-based LIF | {py:class}`~wiretorch.network_simulators.conductance.recurrent.ConductanceLIFNetwork` | {py:class}`~wiretorch.network_simulators.conductance.feedforward.FeedforwardConductanceLIFNetwork` |
| Current-based LIF | {py:class}`~wiretorch.network_simulators.current.recurrent.CurrentLIFNetwork` | {py:class}`~wiretorch.network_simulators.current.feedforward.FeedforwardCurrentLIFNetwork` |
| Rate | {py:class}`~wiretorch.network_simulators.rate.recurrent.RecurrentRateNetwork` | {py:class}`~wiretorch.network_simulators.rate.feedforward.FeedforwardRateNetwork` |

Conductance-based neurons have reversal potentials and separate rise and decay
kinetics for each synapse type, so their input depends on the membrane voltage.
Current-based neurons are simpler and use natural units by default (threshold 1, reset
0). Rate neurons have a continuous output with a nonlinearity chosen per cell type. With
their time constants set to zero they reduce to the standard rate equations. The
spiking models can add a refractory period, spike-frequency adaptation and short-term
depression and facilitation; the conductance-based ones also an NMDA Mg²⁺ block and
receptor saturation ({doc}`neuron-models`).

A recurrent network takes `rec_projections` between its own neurons and
`ff_projections` from its inputs. A feedforward network only has `projections` from its
inputs ({doc}`topologies`).

## Projections

| Projection | Weights | Trained |
|---|---|---|
| {py:class}`~wiretorch.network_simulators.projections.FrozenProjection` | as given | nothing |
| {py:class}`~wiretorch.network_simulators.projections.ScalingFactorProjection` | connectome × gain | one gain per block |
| {py:class}`~wiretorch.network_simulators.projections.LowRankProjection` | `exp(U V) × mask` | `U`, `V` |
| {py:class}`~wiretorch.network_simulators.projections.FullRankProjection` | `exp(log w) × mask` | every weight in the mask |

Frozen and scaling-factor projections are the cheapest, since their weights are
prepared once. Low- and full-rank weights are rebuilt on every forward call so that
gradients can reach them ({doc}`projections`).

## Example: the ring attractor

The {doc}`../tutorials/ring_attractor` uses a
{py:class}`~wiretorch.network_simulators.rate.recurrent.RecurrentRateNetwork` with
ReLU neurons. With its synaptic time constants at zero, it is the standard ring model
`τ dx/dt = −x + W[x]₊ + b`. All of its connections are
{py:class}`~wiretorch.network_simulators.projections.FrozenProjection`s, because the
weights are set from theory and nothing is trained. It has two cell types because
weights must be non-negative: the cosine kernel is split into excitation between ring
neurons and inhibition through a single inhibitory neuron.

## Common cases

| To... | use |
|---|---|
| simulate a fixed network from a connectome | a recurrent network with `make_frozen_projections` |
| fit one gain per pair of cell types | `make_scaling_factor_projections` and `SNNTrainer` |
| fit every weight the connectome allows | `FullRankProjection(weights, mask=connectome != 0)` |
| model unrecorded neurons | `TwoLayerSNN` ({doc}`topologies`) |
| fit time constants or thresholds | the `learn=` argument ({doc}`learning`) |
| generate a synthetic connectome | `wiretorch.synthetic_connectome` |
