# Performance

Seconds of wall time per second of simulated time (lower is better), for a batch of 50
trials at dt = 1 ms on a Quadro RTX 5000. The network has 5000 neurons (80% excitatory,
6% connectivity) and 1500 Poisson inputs. Training means a forward and backward pass
over 160-step chunks with trainable scaling factors. The numbers come from
`benchmarks/speed.py`.

| | float32 | float16 storage |
|---|---|---|
| Conductance-based, recurrent simulation | 0.158 | 0.117 |
| Conductance-based, recurrent training | 0.256 | 0.212 |
| Conductance-based, feedforward training* | 0.246 | 0.246 |
| Current-based, recurrent simulation | 0.12 | 0.095 |
| Current-based, recurrent training | 0.68 | 0.44 |
| Current-based, feedforward training | 0.137 | 0.137 |
| Rate, recurrent simulation | 0.54 | 0.245 |
| Rate, recurrent training | 1.14 | 0.69 |
| Rate, feedforward training | 0.444 | 0.201 |

\* Through `SNNTrainer` with the van Rossum loss. The other training rows are a
forward and backward pass with a random readout as the loss.

For comparison, the plain PyTorch implementation takes 1.63 s/s to simulate the
conductance-based network and 9.65 s/s to train it.

The speed comes from a few things:

- A feedforward chunk is integrated by a single Triton kernel, with each neuron's state
  held in registers. The backward pass is one reverse-time kernel.
- A recurrent spiking step only reads the weight rows of neurons that spiked, roughly
  35 of 5000 per step.
- Sparse spike input goes through cuSPARSE rather than dense matrix products.
- The van Rossum loss filters with a linear-time recursion on the GPU.
- The trainer copies the next batch to the GPU while the current one runs, and avoids
  synchronising with the host.

With `storage_dtype="float16"` the state and weights are stored in half precision, and
rate networks run their matrix products on tensor cores ({doc}`how-to/float16`).
