# GPU kernels and float16

## Kernels and the PyTorch implementation

On a CUDA device, simulation and gradients use [Triton](https://triton-lang.org)
kernels. A feedforward chunk is one kernel launch, with a matching reverse-time kernel
for its gradient. A recurrent step is two launches, and only reads the weights of
neurons that spiked. On the CPU, or when per-step gradients are requested
(`track_gradients=True`), the networks use a plain PyTorch implementation, which the
kernels are tested against. Set `WIRETORCH_BACKEND=torch` to use it on the GPU too.

Benchmarks are on the {doc}`../performance` page.

## float16 storage

With `model.storage_dtype = "float16"` (or the `storage_dtype=` argument), the synaptic
state and recurrent weights are stored in half precision on the GPU, while arithmetic
stays in float32. The rate networks and the current-based backward pass then also run
their large matrix products on tensor cores. This is up to 2 to 3 times faster for rate
networks, but it rounds every step, so a spiking simulation becomes a different
realisation of the same network. Its statistics agree with float32, while individual
spike times diverge after a while. The default is float32 ({doc}`../how-to/float16`).
