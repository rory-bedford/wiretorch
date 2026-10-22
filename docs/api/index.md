# API reference

The most common classes and functions can be imported from the top level, as in
`wt.RecurrentRateNetwork`. They are documented in the subpackages where they are
defined:

**Networks**
: {py:class}`~wiretorch.network_simulators.conductance.recurrent.ConductanceLIFNetwork`,
  {py:class}`~wiretorch.network_simulators.conductance.feedforward.FeedforwardConductanceLIFNetwork`,
  {py:class}`~wiretorch.network_simulators.current.recurrent.CurrentLIFNetwork`,
  {py:class}`~wiretorch.network_simulators.current.feedforward.FeedforwardCurrentLIFNetwork`,
  {py:class}`~wiretorch.network_simulators.rate.recurrent.RecurrentRateNetwork`,
  {py:class}`~wiretorch.network_simulators.rate.feedforward.FeedforwardRateNetwork`,
  {py:class}`~wiretorch.network_simulators.two_layer.TwoLayerSNN`

**Projections**
: {py:class}`~wiretorch.network_simulators.projections.FrozenProjection`,
  {py:class}`~wiretorch.network_simulators.projections.ScalingFactorProjection`,
  {py:class}`~wiretorch.network_simulators.projections.LowRankProjection`,
  {py:class}`~wiretorch.network_simulators.projections.FullRankProjection`,
  {py:func}`~wiretorch.network_simulators.projections.make_frozen_projections`,
  {py:func}`~wiretorch.network_simulators.projections.make_scaling_factor_projections`

**Training and inference**
: {py:class}`~wiretorch.snn_runners.trainer.SNNTrainer`,
  {py:class}`~wiretorch.snn_runners.inference_runner.SNNInference`,
  {py:class}`~wiretorch.snn_runners.evolutionary_search.EvolutionarySearch`

**Losses**
: {py:class}`~wiretorch.training_utils.losses.VanRossumLoss`,
  {py:class}`~wiretorch.training_utils.losses.VanRossumRateLoss`,
  {py:class}`~wiretorch.training_utils.losses.PoissonKLLoss`,
  {py:class}`~wiretorch.training_utils.losses.FiringRateLoss`,
  {py:class}`~wiretorch.training_utils.losses.SilentNeuronLoss`,
  {py:class}`~wiretorch.training_utils.losses.CalciumMSELoss`

**Everything, by subpackage**
: {doc}`reference/wiretorch/network_simulators/index` (the networks, projections, and
  each neuron model's time step in `dynamics`),
  {doc}`reference/wiretorch/synthetic_connectome/index`,
  {doc}`reference/wiretorch/snn_runners/index`,
  {doc}`reference/wiretorch/training_utils/index`,
  {doc}`reference/wiretorch/dataloaders/index`,
  {doc}`reference/wiretorch/observation_models/index`,
  {doc}`reference/wiretorch/analysis/index`,
  {doc}`reference/wiretorch/visualization/index`,
  {doc}`reference/wiretorch/configs/index`,
  {doc}`reference/wiretorch/utils/index`

```{toctree}
:hidden:
:maxdepth: 2

reference/wiretorch/index
```
