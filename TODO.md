# TODO

## Test and tune the kernels on other GPUs

The Triton kernels have only been run on a Quadro RTX 5000 (Turing, compute
capability 7.5). For each newer generation (Ampere, e.g. A100 or RTX 30; Ada, e.g. L40
or RTX 40; Hopper, e.g. H100; Blackwell, e.g. B200 or RTX 50):

- Run the test suite (`uv run pytest`) and check the kernels match the PyTorch
  reference.
- Run `benchmarks/speed.py` and add the results to `docs/performance.md`.
- Tune the block sizes, which were chosen on Turing, e.g. with `triton.autotune` or a
  per-architecture table, and check that state held in registers doesn't spill.
- Check float16 storage on each, including tensor-core use in rate networks.

Also try AMD GPUs (ROCm), including the sparse spike input path.

## Recover the conductance-based backward pass's speed

The exponential-Euler voltage step made `scan_backward` 2.7x slower on the teacher
(7 → 19 ms per 160-step chunk; training +23%), with inference unchanged. Its adjoint
needs a second reduction over synapse types and more per-neuron factors broadcast back
over them, and Triton converts layouts through shared memory for each (SASS: 944 → 1472
instructions, `BAR` 21 → 33). Options: save the step's `phi = (1 - a) / g_tot` (and the
total current) in the forward pass, or restructure the adjoint to keep per-neuron
values in the reductions' layout.

## Dynamics to add later

Discussed and deferred (October 2026):

- **Synaptic delays**: per pathway (source type → target type), a ring buffer of the
  last D steps of spikes; per connection would be expensive.
- **Gap junctions** (electrical synapses, in C. elegans and Drosophila connectomes):
  `I_i = sum_j g_ij (v_j - v_i)` needs every voltage every step — a sparse product per
  step instead of the spike gather.
- **Intrinsic membrane noise** (Gaussian or OU current per neuron): Philox in the kernels.
- **Per-neuron heterogeneity** of the physiology (not only per cell type); the tables
  are already per neuron.
- **Graded (non-spiking) neurons mixed with spiking ones** in one network.
- **Subtractive reset** (`v <- v - theta`).
- **Target-side receptor saturation** (a smooth `g_clip`):
  `dG_i/dt = -G_i / tau + (1 - G_i / g_max) input_i`.
- **Gradients through the presynaptic gating** (saturation, and STP when it lands): the
  state depends only on the source's own spikes, so a per-source reverse scan.
