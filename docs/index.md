# wiretorch

wiretorch builds neural network models from connectomes in PyTorch. You give it a
wiring diagram and the cell type of each neuron, and it builds a network with exactly
that connectivity. The networks run on GPUs with custom kernels and are differentiable,
so they can be trained with any loss: matching recorded spikes, rates or calcium
signals, performing a task, or anything else you can write in PyTorch.

It supports a range of single-neuron models, including conductance- and current-based
spiking neurons and rate neurons. Weights can be fixed, scaled per cell-type pair,
factorised or learnt one by one, and physiological parameters such as time constants
and thresholds can be learnt too.

```bash
pip install wiretorch
```

```python
import wiretorch as wt
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Installation and a first simulation.
:::

:::{grid-item-card} Concepts
:link: concepts/index
:link-type: doc

Cell types, synapse types, projections and neuron models.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

A ring attractor, a spiking network on a synthetic connectome, and two fitting examples.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

Checkpoints, float16, experiments, zarr output.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Generated from the docstrings.
:::

:::{grid-item-card} Performance
:link: performance
:link-type: doc

Benchmarks.
:::
::::

```{toctree}
:hidden:
:caption: Learn

getting-started/index
concepts/index
tutorials/index
how-to/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Background

performance
development
```
