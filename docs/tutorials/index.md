# Tutorials

Each tutorial is a notebook, which you can download from the top of its page. They
take seconds on a GPU and up to two minutes on a CPU.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} A ring attractor
:link: ring_attractor
:link-type: doc

A rate network wired by hand from theory, using `RecurrentRateNetwork` and
`FrozenProjection`.
:::

:::{grid-item-card} A spiking network from a synthetic connectome
:link: synthetic_connectomes
:link-type: doc

Generate a connectome with assemblies, put conductance-based neurons on it and
simulate it.
:::

:::{grid-item-card} Fitting a network to recorded spikes
:link: teacher_student
:link-type: doc

Recover a network's connection strengths from its spikes with
`ScalingFactorProjection` and `SNNTrainer`.
:::

:::{grid-item-card} Learning physiology
:link: learning_physiology
:link-type: doc

Fit membrane time constants and thresholds of a current-based network with a plain
PyTorch loop.
:::
::::

```{toctree}
:hidden:

ring_attractor
synthetic_connectomes
teacher_student
learning_physiology
```
