# Installation

wiretorch needs Python 3.13 or later and PyTorch 2.7 or later. Install PyTorch first,
choosing the build for your machine ({ref}`see below <pytorch-build>`), then wiretorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install wiretorch
```

To log training runs to Weights & Biases, install the extra: `pip install
"wiretorch[wandb]"`.

Linux with an NVIDIA GPU is recommended. There the networks run on fast GPU kernels
written in [Triton](https://triton-lang.org), which PyTorch ships only for Linux. On
macOS and Windows wiretorch runs, but on a plain PyTorch implementation, which is about
10 times slower to simulate and 40 times slower to train: on Windows on the GPU, on
macOS on the CPU (Macs have no NVIDIA GPU).

(pytorch-build)=
## Choosing a PyTorch build

PyTorch wheels bring their own copy of CUDA, so you don't need to install CUDA
yourself. What your machine provides is the NVIDIA driver, and the driver limits which
CUDA versions can run. `nvidia-smi` shows that limit in its header, as "CUDA Version".
This is the newest CUDA the driver supports, not a CUDA installation. Any PyTorch
build for that version or older will work.

pip keeps the PyTorch you already have. If you don't install one first, pip installs
PyPI's default, which on Linux is a CUDA 13 build. That needs a recent driver; with an
older one PyTorch won't see the GPU (`torch.cuda.is_available()` returns False) and
everything runs on the CPU.

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

Without a GPU or without Triton, the networks run the plain PyTorch implementation,
which gives the same results more slowly. It is also used on Linux GPUs if you set
`WIRETORCH_BACKEND=torch`, or if you request per-step gradients.

## From source

To work on wiretorch itself, see {doc}`../development`.
