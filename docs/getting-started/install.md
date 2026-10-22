# Installation

wiretorch needs Python 3.13 or later and PyTorch 2.7 or later. Install PyTorch first,
choosing the build for your machine (see below), then wiretorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install wiretorch
```

pip keeps the PyTorch you already have. If you skip the first step, it installs PyPI's
default PyTorch, which on Linux is a CUDA 13 build. That works on CPU and on machines
with a recent NVIDIA driver, but with an older driver PyTorch won't see the GPU:
`torch.cuda.is_available()` returns False and everything runs on the CPU.

To log training runs to Weights & Biases, install the extra: `pip install
"wiretorch[wandb]"`.

## Choosing a PyTorch build

PyTorch wheels bring their own copy of CUDA, so you don't need to install CUDA
yourself. What your machine provides is the NVIDIA driver, and the driver limits which
CUDA versions can run. `nvidia-smi` shows that limit in its header, as "CUDA Version".
This is the newest CUDA the driver supports, not a CUDA installation. Any PyTorch
build for that version or older will work.

| `nvidia-smi` shows | Install |
|---|---|
| CUDA 13.0 or later | the default: `pip install torch` |
| CUDA 12.8 or 12.9 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` (or `cu129`) |
| older | update the driver, or use an older PyTorch for your CUDA version |
| no NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

[pytorch.org](https://pytorch.org/get-started/locally/) lists the builds for each
PyTorch release. With uv, `uv pip install torch --torch-backend=auto` reads your driver
and picks a matching build.

In a uv project, point torch at PyTorch's index in your `pyproject.toml`:

```toml
[project]
dependencies = ["wiretorch", "torch"]

[tool.uv.sources]
torch = { index = "pytorch-cu129" }

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true
```

uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) shows how to
switch between CPU and CUDA builds with extras.

## GPUs

On an NVIDIA GPU the networks use [Triton](https://triton-lang.org) kernels for both
simulation and gradients. Triton compiles them for your GPU the first time they run,
so they don't depend on a particular GPU or CUDA version. They work on any GPU that
recent PyTorch supports, which means compute capability 7.5 or higher:

| Generation | Examples |
|---|---|
| Turing (2018) | RTX 20 series, Quadro RTX, T4 |
| Ampere (2020) | RTX 30 series, A100, A40 |
| Ada Lovelace (2022) | RTX 40 series, L4, L40 |
| Hopper (2022) | H100, H200 |
| Blackwell (2024) | RTX 50 series, B200 |

Older GPUs (Volta, such as the V100, and Pascal, such as the GTX 10 series) need an
older PyTorch and are untested. The benchmarks on {doc}`../performance` were run on a
Quadro RTX 5000, a Turing card. Newer GPUs should be faster, but the kernels' block
sizes were chosen on Turing, so they may not reach their peak. AMD GPUs are untested.

Without a GPU, or without Triton (it comes with PyTorch's Linux CUDA builds only), the
networks run a plain PyTorch implementation. It gives the same results but is about 10
times slower to simulate and 40 times slower to train. The PyTorch implementation also
runs on the GPU if you set `WIRETORCH_BACKEND=torch`, or if you request per-step
gradients.

## From source

To work on wiretorch itself, see {doc}`../development`.
