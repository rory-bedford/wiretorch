# Performance log

Wall-clock seconds to simulate **one second of network time** for a batch of 50 trials
(dt = 1 ms), from `python benchmarks/speed.py`, on a Quadro RTX 5000 (Turing, 16 GB),
torch 2.8.0+cu128. Lower is better. Compilation and the first data epoch are excluded.

* **Teacher simulation** — the 5000-neuron Bernstein teacher (20 assemblies, 1500 feedforward
  inputs), compiled, `inference_mode`, 2500-step chunks.
* **Student training** — the fully observed student (6500 input rows, six scaling
  factors) through `SNNTrainer` with the Figure 1 settings: 160-step chunks, van Rossum
  loss, an update every 6 chunks, logging every 6.
* **Half-observed training** — the same with half the neurons observed, Figure 1's
  actual setting: a `TwoLayerSNN` whose 2500-neuron hidden layer is recurrent. Added
  after the fact; its baseline was measured by running the same benchmark against
  master's code from a git worktree. ("–": not measured at that commit.)

Every row passes the full test suite (`tests/`), including GPU lockstep against the
fixtures recorded from the original simulator.

| Commit | Change | Teacher simulation | Student training | Half-observed training |
|---|---|---|---|---|
| `36182f4` | Baseline (master) | 1.63 s/s | 9.65 s/s | 18.6 s/s |
| `dd51ce0` | Triton engine: fused per-chunk scan, sparse recurrent steps, adjoint BPTT | 0.214 s/s (7.6x) | 1.63 s/s (5.9x) | – |
| `e6972a3` | van Rossum filter as batched triangular matmuls | 0.214 s/s | 0.83 s/s (11.6x) | – |
| `6675848` | Trainer: track variables only for plots, concatenate logged data lazily | 0.214 s/s | 0.60 s/s (16.1x) | – |
| `0e870e9` | Sparse feedforward input: one gather kernel instead of dense matmuls | 0.166 s/s (9.8x) | 0.465 s/s (20.8x) | – |
| `f539413` | Sparse input through cuSPARSE; coalesced conductance layout; tiled spike gather | 0.158 s/s (10.3x) | 0.357 s/s (27.0x) | – |
| `1353e75` | van Rossum filter as a fused linear-time recursion on CUDA | 0.158 s/s | 0.280 s/s (34.5x) | 0.61 s/s (30.5x) |
| `0daf2ee` | Two-layer backward: input gradient only for the hidden-spike columns | 0.158 s/s | 0.280 s/s | 0.47 s/s (39.8x) |
| `2806668` | No host syncs per chunk; kept chunks copied without blocking | 0.158 s/s | 0.246 s/s (39.2x) | 0.435 s/s (42.8x) |

Opt-in, not in the rows above: `storage_dtype="float16"` (conductances, saved
conductances and recurrent weights stored in float16; `speed.py --storage-dtype float16`)
gives **0.117 s/s** teacher simulation (1.35x), 0.246 s/s student training (1.14x),
0.44 s/s half-observed training (~1.05x, 10% less memory). See below.

The benchmark reproduces the real runs: the actual Bernstein teacher measured 1.69 s/s
and the Figure 1 seed-44 training run 9.56 s/s on the same GPU.

## What changed, and why it was faster

### Triton engine

The old step was ~30 PyTorch kernels per time step, launched from Python: a dense
`spikes @ W` per source cell type (a 50 x 4000 x 5000 GEMM every millisecond, for ~1%
non-zero spikes), boolean-mask indexing that copied the conductance tensor several
times, and autograd recording every intermediate. `torch.compile` gave no speed-up
(compiled and eager both 1.6 ms/step).

* **Feedforward networks** (the fully observed student): the whole chunk's synaptic
  input is one matmul per pathway, then a single kernel integrates all 160 steps with
  each neuron's state in registers. The backward pass is the matching reverse-time
  adjoint, also one kernel, saving only the voltage and summed conductances per step.
* **Recurrent networks** (the teacher): two kernels per step. The first lists, per
  trial, which neurons fired; the second gathers only those weight rows (~35 of 5000)
  and integrates. Feedforward input is still one matmul per block of steps.
* Results are unchanged to within rounding: every test passes, including GPU lockstep
  against the fixtures recorded from the original code.

### van Rossum loss as matrix products

With the simulator fast, the loss took 46% of a training chunk: each filter stage was a
`conv1d` with one channel and a kernel as long as the chunk (160 taps), which cuDNN
runs as a slow grouped direct convolution. Within a chunk the causal exponential filter
is a fixed lower-triangular Toeplitz matrix, so each stage is now one batched
`(T, T) @ (B, T, N)` matrix product — the same kernel values, the same chunk carry,
~10x less time. `PoissonKLLoss`, which duplicated the filter, now inherits it.

### Trainer bookkeeping

The trainer tracked voltages, currents and conductances whenever a log or checkpoint was
within `plot_size` chunks — with Figure 1's settings (log every 6, `plot_size` 48), on
every chunk — copied them to the CPU, and at every log concatenated 48 chunks of them
(~3 GB) into the stats snapshot. Only plots read them, and Figure 1 has no plot
generator; its stats read only spike trains. Tracking now happens only when there is a
plot generator, and the snapshot concatenates a variable only when it is read. Besides
the median, this removed the 0.6 s stall every sixth chunk (the mean fell from 1.51 to
0.60 s/s).

### Sparse feedforward input

Computing every pathway's synaptic input was a dense `(B T, n_in) @ (n_in, N)` matmul
per pathway, half of all training time — for spike input that is ~0.5% non-zero.
Input that is at most 10% non-zero now goes through one kernel that walks each row's
non-zeros (CSR) and sums the matching weight rows for all pathways at once, written
straight into the layout the scan reads. Its backward (for learnt weights, or when the
input is another layer's spikes) stays a dense matmul. Continuous inputs keep the
dense path.

### cuSPARSE for sparse input, and kernel layout

My gather kernel walked each row's non-zeros one load at a time; cuSPARSE's CSR product
is 2x faster on these shapes (12 ms vs 24 ms per training chunk for all three
pathways), so sparse input now goes through `to_sparse_csr() @ W` per pathway, written
in place, with density judged from the first step. The conductance state is stored
`(B, 2, S, N)` — each component contiguous over neurons — and exposed as the same
`(B, N, 2, S)` view as before; the recurrent step gathers fired neurons' weight rows 16
at a time, 256 neurons per program. The step is now memory-bound: ~22 MB of state per
step, read and written.

### van Rossum filter as a recursion

The matrix-product filter was still ~27 ms per chunk once its backward (broadcast
batched matmuls) and the carried-state additions were counted. Each stage is a
first-order recursion, `y[t] = a y[t-1] + x[t]`, so on CUDA one Triton kernel runs
both stages per neuron in a single pass over time, and the reverse recursion is its
backward — linear in the chunk length, memory-bound. The CPU keeps the exact Toeplitz
products; the CUDA recursion agrees with them to ~1e-5 relative.

### Two-layer backward

Layer 2 of a `TwoLayerSNN` gets `[external inputs, hidden spikes, observed spikes]` as input,
and only the hidden spikes carry a gradient — yet the sparse-input backward computed
the gradient for every column, a 25 ms dense matmul per chunk. `TwoLayerSNN` now marks
the hidden columns (`differentiable_inputs`) and the backward computes only those.
With that, the recurrent step itself is ~10% of a half-observed chunk.

### `SNNInference`

Teacher generation as `fig00` runs it — OU-modulated Poisson input, the teacher, every
input and output stored — was 0.72 s/s in memory and 4.4 s/s to zarr, against 0.16 s/s
for the simulation itself: every chunk's tensors were copied to the host with blocking,
pageable copies, memory mode concatenated everything again at the end, and each zarr
array was one chunk per time block, compressed on one thread. Chunks are now copied to
reused pinned buffers without blocking, filed by a background thread into
preallocated arrays or zarr (chunked per trial, so compression runs in parallel), while
the GPU simulates the next chunk: **0.46 s/s** in memory, **3.3 s/s** to zarr. The
zarr figure is bounded by writing the float rate arrays to network storage, not by
compressing them: OU rates compress 1.2x with any codec (zstd, the default; blosc with
lz4 or zstd, byte- or bitshuffle) and are written at ~0.12 GB/s to the cluster's network storage
whichever codec — or none — is used, against ~1 GB/s to a local disk. Only fewer bytes
would be faster (e.g. rates stored in float16, which would change what is saved).
Spike arrays compress 44x with the default.

### Trainer data copies

`benchmarks/speed.py` keeps its data on the GPU, which hid the cost of the copies a real
run makes: a teacher-sized training batch (50 trials × 160 steps, bool spikes) is 92 MB,
and this GPU sits on a PCIe gen3 ×4 link (~3 GB/s), so each chunk waited 30 ms for its
data before ~40 ms of compute. With whole batches in pinned host memory, as a
DataLoader with `pin_memory=True` delivers them, fully observed training ran at
0.43 s/s against 0.25 s/s with the data already on the GPU. The trainer now copies
the next batch on a side stream while the current chunk runs (`chunks.Prefetcher`), and
the copy disappears: **0.25 s/s** fully observed, 0.515 → **0.43 s/s** half-observed.
(Float spikes would be 4x larger and cost more than the compute: keep datasets bool.)

Three smaller host syncs went with it. The chunks kept for stats and plots (every
chunk, with the teacher-student settings `plot_size = checkpoint_interval`) were copied
to the host with a blocking `.cpu()`; they now go to pinned buffers and are synced only
when read (0.284 → 0.270 s/s). The engine read `dt` and the surrogate scale with
`float()` of device scalars at every launch, and built the recurrent pathways' row
offsets from the host every forward call; the kernels now load the scalars themselves
and the offsets are cached (0.270 → 0.246 s/s). The van Rossum filter reads its decay
factors once per kernel tensor instead of every chunk.

### float16 storage (opt-in)

A recurrent step reads and writes the conductance state (8 MB for the teacher) and
gathers the weight rows of the neurons that fired (~35 per trial, 35 MB) — memory-bound
at ~350 GB/s. `storage_dtype="float16"` on the conductance networks stores the state,
the conductances saved for the backward pass and the recurrent weights in float16;
loads convert to float32, so arithmetic and gradients are unchanged in kind. Teacher
simulation 0.157 → 0.117 s/s; float16 state alone gave 1.19x, weights alone 1.12x.
Training gains less (the recurrent step is a small part of a chunk): fully observed
0.280 → 0.246 s/s, half-observed 0.47 → 0.44 s/s and 4.05 → 3.65 GB peak.

The default stays float32, bit-identical to before. A float16 run is a different
realisation of the same network. Over 10 s of the teacher (20 trials, same input):
firing rates +0.35% (E) and +0.26% (I), per-neuron rate correlation 0.9996 with float32,
Fano factor −0.3%, population-rate fluctuations +1.6%; spike agreement with float32
99% in the first 50 ms, 68% at 50–200 ms, then chance (~14%). (A float32 run with the
initial voltages nudged by 1e-4 mV keeps 96% agreement: float16 rounds every step, a
nudge only once.) Half-observed student gradients: cosine 0.99993 with float32.
bfloat16 was as fast but rates moved +2% and, without any spike flipped, a small
network's gradient only kept cosine 0.98 — it is not offered.

### Learnt physiology

`learn=` trains `tau_mem`, `theta`, `U_reset`, `E_L`, `g_L` (per cell type) and
`tau_rise`, `tau_decay`, `E_syn`, `g_bar` (per synapse type). `scan_backward` gives the
kernel-level gradients (`theta`, `U_reset`, `E_L`, `C_m`, `beta`, `E_syn`, and each
component's jump and decay; autograd takes them on to the time constants and
conductances) from the voltages and conductances it already reads, plus the
conductances entering the chunk: as in the current-based kernels, the decays' gradient
is carried back as one more recursion over the injections instead of saving every
step's components. Each term is compiled in only when its parameter is learnt. Fully
observed student training (`speed.py training`):

| | s/s |
|---|---|
| Fixed physiology | 0.247 (unchanged) |
| `theta`, `U_reset`, `E_L` | 0.254 |
| `tau_mem`, `g_L` | 0.259 |
| `E_syn`, `g_bar` | 0.265 |
| `tau_rise`, `tau_decay` | 0.263 |
| All nine | 0.278 (+13%) |

With everything learnt the backward kernel holds seven per-synapse accumulators and
spilled registers (0.297 s/s); it then runs half-size blocks.

### Saved conductances

The backward pass reads the summed conductances of every step, which the forward pass
saves (`B x T x N x S`: 800 MB per 160-step chunk of the fully observed student). Their
traffic is ~4.6 ms of the chunk's 28 ms of GPU time. Recomputing them instead would mean
checkpointing the state every few steps and re-running each segment inside the
backward kernel, held in registers or scratch memory, re-reading the activations:
an estimated ≤7% net, for much harder kernels. Not done; `storage_dtype="float16"`
already halves that traffic.

### The recurrent step, measured

A teacher step takes 127 µs (`recurrent_step`; `compact_spikes` another 10 µs): 73 µs
without any spike — the state update, ~24 MB at ~320 GB/s — and ~50 µs to sum the ~37
weight rows each trial's fired neurons contribute. Those rows are 37 MB per step, more
than DRAM could deliver in that time: neurons firing in several trials are read from L2.
A gather from the rows' non-zeros (an occupancy bitmask per 32 targets and a popcount
into packed values, bit-identical to the dense sum and ~8x fewer bytes) ran at 0.147
s/s against 0.152 in float32 and was slower in float16 (0.139 against 0.111): the
gather waits on dependent loads, not on bytes. Larger row tiles spilled registers. The
host enqueues a step in less time than the GPU runs it, so CUDA graphs would not help
either; float16 storage (above) remains the lever.


## Current-based networks

The current-based networks used to be a Python loop over time steps with a dense
`spikes @ W` per step, on PyTorch only. They now share the conductance-based networks'
projection API and engine design (`network_simulators/current/`): a fused feedforward
scan, a sparse recurrent step (compaction, then a gather of the fired neurons' weight
rows), and a reverse-time adjoint. From `python benchmarks/speed.py current`: the
teacher's connectome in natural units (weights rescaled for E 4.6 Hz, I 7.3 Hz), batch
50, eager, same GPU; old = master before this change on the same inputs.

| | Old | New | New, `storage_dtype="float16"` |
|---|---|---|---|
| Recurrent simulation (`inference_mode`, 1000-step chunks) | 0.64 s/s | 0.12 s/s (5.3x) | 0.095 s/s |
| Recurrent training (forward + backward, 160 steps, scaling factors) | 1.98 s/s | 0.68 s/s (2.9x) | 0.444 s/s |
| Feedforward simulation | – | 0.115 s/s | 0.115 s/s |
| Feedforward training | 1.09 s/s | 0.137 s/s (8.0x) | 0.137 s/s |

The feedforward network's input rows are the external inputs plus 5000 rows of spikes
from another population (as in teacher forcing). Training rows are one forward and
backward pass of a random spike readout, not a full `SNNTrainer` run; the
conductance-based figures above (0.246 s/s) include the trainer and the van Rossum
loss.

Unlike the conductance-based networks, gradients flow through the recurrent spikes, so
the recurrent backward pass cannot be one per-neuron reverse scan: every step carries
each pathway's `dL/da` back to the presynaptic spikes through `W^T`. As a dense GEMM that
was two-thirds of a training chunk; connectomes are sparse (the teacher's 6%), so it is a
cuSPARSE product per step when the weights are at most 25% dense, which took recurrent
training from 1.14 to 0.68 s/s. It is gather-bound: a Triton kernel walking each row's
non-zeros was no faster. With float16 storage it is instead a dense product on the
tensor cores, `W^T` stored contiguous: 152 µs per step for the teacher against cuSPARSE's
422 µs (relative error ~4e-4, accumulating in float32), so recurrent training runs at
0.444 s/s. (A batch of 50 keeps the tensor cores far from their peak; the other
orientation, `W @ g^T`, took 205 µs.) Correctness: the PyTorch reference reproduces the original
current-based simulator spike for spike; the kernels are held to it by lockstep,
gradient and large-network statistics fixtures (`tests/README.md`).

### Learnt physiology

`learn=` trains `tau_mem`, `theta`, `U_reset`, `U_rest`, `R` (per cell type) and
`tau_syn` (per synapse type). The backward kernels give their gradients from nothing
saved beyond what the weights' gradients already need, plus the chunk's final voltages
and initial currents: `beta` and `R` multiply the signed current of a step without a
spike, which follows from two consecutive voltages, and the decays' gradient
`sum_t gJ_t I_{t-1}` is rewritten, the currents being linear in their inputs, as one more
reverse recursion over the injections (`current/kernels.py`). A first version that
saved the currents of every step cost 40% each for `tau_mem`/`R` and `tau_syn` in
feedforward training. Each term is compiled in only when its parameter is learnt, so
fixed physiology costs nothing (the rows above are unchanged).

| Training, s/s | Fixed | One parameter | All six |
|---|---|---|---|
| Recurrent | 0.69 | 0.69–0.72 | 0.77 (+12%) |
| Feedforward | 0.138 | – | 0.155 (+12%) |

Two host-side costs were found on the way, each worth ~10% while any parameter was
learnt: `TypeParameters` placed the learnt values with `masked_scatter`, whose CUDA
implementation (and its backward, `masked_select`) synchronises with the host — now
`index_put`, which does not; and Triton launches with tensors that require grad cost
more host time, so the kernels get the physiology detached (the autograd functions take
it as inputs). The rate networks, which also use `TypeParameters`, gained 1–6% in
training (below).

## Rate networks

`network_simulators/rate/` runs on Triton kernels behind the PyTorch reference:
a fused scan for feedforward networks, and per recurrent step a rates kernel, one
product per recurrent pathway and an update kernel, with matching backward kernels
that also give the learnt physiology its gradient. From `python benchmarks/speed.py
rate`: the teacher's connectome with ReLU rates (E 1.4, I 0.8), white-noise input,
batch 50, learnt scaling factors and physiology, same GPU.

| | Reference | Kernels | Kernels, `storage_dtype="float16"` |
|---|---|---|---|
| Recurrent simulation (`inference_mode`, 1000-step chunks) | 0.74 s/s | 0.54 s/s | 0.245 s/s |
| Recurrent training (forward + backward, 160 steps) | 2.50 s/s | 1.14 s/s | 0.69 s/s |
| Feedforward simulation | 0.64 s/s | 0.37 s/s | 0.128 s/s |
| Feedforward training | 1.34 s/s | 0.444 s/s | 0.201 s/s |

float32 is the default: float16's ~1e-3 relative error would floor precise fits, and
small weights lose precision, for 2-3x.

Rates are dense, so unlike spikes there is nothing to skip: every step multiplies all
rates by the recurrent weights. That product is ~85% of a recurrent step, and the
input product most of a feedforward chunk, both at about half the GPU's float32 peak
(dense GEMM 0.42 ms per teacher step; cuSPARSE on the 6%-dense connectome is no
faster, so it is used only where it wins; one sparse product fusing every pathway,
measured but not adopted, would save ~10%). float16 runs them on the tensor cores — 2.7x faster products,
relative error ~1e-3, accumulating in float32 — with the state and elementwise
arithmetic in float32. Rate networks are not chaotic, so float16 rates agree with
float32 to ~1e-3 and gradients keep their direction (cosine > 0.999, tested).

The kernels are tested against the reference directly (rates, state, every gradient),
not through large-network statistics as the spiking networks are: rate networks are not
chaotic, so the two stay close over whole runs.
