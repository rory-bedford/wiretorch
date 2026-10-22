<p align="center">
  <img src="https://raw.githubusercontent.com/rory-bedford/wiretorch/main/docs/_static/logo.png" alt="wiretorch" width="260">
</p>

<h1 align="center">wiretorch</h1>

<p align="center">
  <a href="https://wiretorch.readthedocs.io"><img src="https://img.shields.io/readthedocs/wiretorch" alt="docs"></a>
  <a href="https://pypi.org/project/wiretorch/"><img src="https://img.shields.io/pypi/v/wiretorch.svg" alt="PyPI"></a>
  <a href="https://github.com/rory-bedford/wiretorch/actions/workflows/ci.yml"><img src="https://github.com/rory-bedford/wiretorch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/rory-bedford/wiretorch"><img src="https://codecov.io/gh/rory-bedford/wiretorch/graph/badge.svg" alt="coverage"></a>
  <a href="https://wiretorch.readthedocs.io/en/latest/performance.html"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/rory-bedford/wiretorch/main/docs/badges/inference.json" alt="inference speed"></a>
  <a href="https://wiretorch.readthedocs.io/en/latest/performance.html"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/rory-bedford/wiretorch/main/docs/badges/training.json" alt="training speed"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.7+-ee4c2c.svg" alt="PyTorch 2.7+"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/linting-ruff-261230.svg" alt="Ruff"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/packaging-uv-de5fe9.svg" alt="uv"></a>
  <a href="https://github.com/rory-bedford/wiretorch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

wiretorch builds neural network models from connectomes in PyTorch. You give it a
wiring diagram and the cell type of each neuron, and it builds a network with exactly
that connectivity. The networks run on GPUs with custom kernels and are differentiable,
so they can be trained with any loss: matching recorded spikes, rates or calcium
signals, performing a task, or anything else you can write in PyTorch.

It supports a range of single-neuron models, including conductance- and current-based
spiking neurons and rate neurons. Weights can be fixed, scaled per cell-type pair,
factorised or learnt one by one, and physiological parameters such as time constants
and thresholds can be learnt too.

## Installation

Install [PyTorch](https://pytorch.org/get-started/locally/) first, choosing the CUDA
version your driver supports (`nvidia-smi` shows it), then wiretorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install wiretorch            # or "wiretorch[wandb]" to log to Weights & Biases
```

Without the first step pip installs PyPI's default PyTorch, a CUDA 13 build that needs
a recent driver. The [installation guide](https://wiretorch.readthedocs.io/en/latest/getting-started/install.html)
explains how to choose, and which GPUs are supported.

```python
import wiretorch as wt
```

## Documentation

[wiretorch.readthedocs.io](https://wiretorch.readthedocs.io) has a
[quickstart](https://wiretorch.readthedocs.io/en/latest/getting-started/quickstart.html),
[tutorials](https://wiretorch.readthedocs.io/en/latest/tutorials/index.html) (a ring
attractor, a spiking network on a synthetic connectome, fitting connection strengths
and physiology), and the
[API reference](https://wiretorch.readthedocs.io/en/latest/api/index.html).
Setting up a development environment and running the tests is described under
[Development](https://wiretorch.readthedocs.io/en/latest/development.html).

wiretorch used to be called `connectome_snns`. That name still imports, with a
deprecation warning.
