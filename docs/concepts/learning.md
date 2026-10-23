# What can be trained

## Connectivity

Connectivity is trained through the projections ({doc}`projections`): one gain per
pair of cell types (`ScalingFactorProjection`), a low-rank factorisation
(`LowRankProjection`), or every weight within a mask (`FullRankProjection`).

## Physiology

The `learn=` argument makes physiological parameters trainable, per cell type or
synapse type. Pass `True` to learn a parameter for every type, or a list of type
names:

```python
model = wt.ConductanceLIFNetwork(
    ...,
    learn={"tau_mem": True, "theta": ["excitatory"], "tau_decay": ["NMDA"]},
)
```

| Neuron model | Per cell type | Per synapse type |
|---|---|---|
| Conductance-based | `tau_mem`, `theta`, `U_reset`, `E_L`, `g_L` | `tau_rise`, `tau_decay`, `E_syn`, `g_bar` |
| Current-based | `tau_mem`, `theta`, `U_reset`, `U_rest`, `R` | `tau_syn` |
| Rate | `tau_mem`, `theta`, `gain` | `tau_syn` |

The `tau_decay` of a saturating synapse type cannot be learnt, since the
presynaptic gating shares it ({doc}`neuron-models`).

The learnt values start at the configured ones and are stored in
`model.cell_parameters` and `model.synapse_parameters`. Time constants, conductances,
`R` and `gain` are learnt as logarithms so that they stay positive. Parameters that
aren't learnt stay fixed and add no cost, since the GPU kernels only compute the
gradients that are needed.

## Surrogate gradients

A spike is a step function of the voltage, so its derivative is zero almost everywhere.
The spiking networks use a surrogate derivative in the backward pass, that of a fast
sigmoid, `1 / (surrgrad_scale |v - θ| + 1)²`. With `spike_mode="sample"`, spikes are
instead drawn with probability `sigmoid(surrgrad_scale (v - θ))`. Feedforward networks
can return this probability (`return_probabilities=True`), which has a true gradient.

## Losses and training

The networks are ordinary PyTorch modules, so any differentiable loss works, and so
does a hand-written training loop ({doc}`../tutorials/learning_physiology`). wiretorch
also provides losses for comparing with recordings (`VanRossumLoss`, `PoissonKLLoss`,
`CalciumMSELoss`) and for regularising activity (`FiringRateLoss`,
`SilentNeuronLoss`).

{py:class}`~wiretorch.snn_runners.trainer.SNNTrainer` handles long training runs. It
feeds the data chunk by chunk with the state carried over, accumulates gradients over
several chunks per update, and takes care of burn-in, logging, plots and checkpoints
({doc}`../tutorials/teacher_student`).
