# Quickstart

This example builds a random connectome of 500 conductance-based neurons (80%
excitatory, 20% inhibitory) and drives it with 200 Poisson inputs. It runs on a CPU in a
few seconds.

```python
import numpy as np
import torch
import wiretorch as wt
from wiretorch.synthetic_connectome import cell_types, topology_generators, weight_assigners

np.random.seed(0)
N, N_IN = 500, 200

# Cell types: 80% excitatory (0), 20% inhibitory (1); the inputs are one type.
types = cell_types.assign_cell_types(N, [0.8, 0.2])
input_types = np.zeros(N_IN, dtype=int)

# Wiring: random graphs with type-to-type connection probabilities, then
# log-normal weights with type-to-type means (and variances).
recurrent = topology_generators.sparse_graph_generator(types, types, [[0.045, 0.045], [0.09, 0.09]])
w_rec = weight_assigners.assign_weights_lognormal(
    recurrent, types, types, None, [[0.3, 0.3], [1.8, 1.8]], [[0.09, 0.09], [3.24, 3.24]],
    parameter_space="linear",
)
feedforward = topology_generators.sparse_graph_generator(input_types, types, [[0.12, 0.12]])
w_in = weight_assigners.assign_weights_lognormal(
    feedforward, input_types, types, None, [[0.18, 0.18]], [[0.0324, 0.0324]], parameter_space="linear"
)

# One projection per (source type, target type) pair: here all fixed.
names, input_names = ["E", "I"], ["input"]
rec_projections, in_projections = wt.make_frozen_projections(
    w_rec, w_in, types, input_types, names, input_names
)

model = wt.ConductanceLIFNetwork(
    dt=1.0,  # ms
    rec_projections=rec_projections,
    ff_projections=in_projections,
    cell_type_indices=types,
    cell_type_indices_FF=input_types,
    cell_params=[
        dict(name="E", cell_id=0, tau_mem=85.0, theta=-38.0, U_reset=-60.0, E_L=-60.0, g_L=1.35),
        dict(name="I", cell_id=1, tau_mem=50.0, theta=-45.0, U_reset=-60.0, E_L=-60.0, g_L=0.9),
    ],
    cell_params_FF=[dict(name="input", cell_id=0)],
    synapse_params=[
        dict(name="AMPA", synapse_id=0, cell_id=0, tau_rise=0.1, tau_decay=5.0, E_syn=0.0, g_bar=4.0, g_clip=40.0),
        dict(name="GABA", synapse_id=1, cell_id=1, tau_rise=0.5, tau_decay=6.0, E_syn=-70.0, g_bar=8.0, g_clip=80.0),
    ],
    synapse_params_FF=[
        dict(name="AMPA", synapse_id=0, cell_id=0, tau_rise=0.1, tau_decay=5.0, E_syn=0.0, g_bar=6.0, g_clip=40.0),
    ],
    surrgrad_scale=5.0,
    batch_size=4,
)

# One second of 20 Hz Poisson input, (batch, time, inputs).
inputs = torch.rand(4, 1000, N_IN) < 20.0 / 1000
spikes = model(inputs)  # (batch, time, neurons)
rates = spikes.mean(dim=(0, 1)) * 1000
print(f"E: {rates[types == 0].mean():.1f} Hz, I: {rates[types == 1].mean():.1f} Hz")
```

```text
E: 3.9 Hz, I: 29.5 Hz
```

Each neuron has an integer cell type, and each type has a name and its physiology in
`cell_params`. The input rows have cell types too.

The connectome is a non-negative weight matrix of shape `(sources, targets)`. Whether a
connection excites or inhibits depends on the synapse type of its source, not on the
sign of the weight, so Dale's law holds automatically.

The weight matrices are split into one block per pair of source and target cell types.
Each block is a *projection*, and the kind of projection decides whether and how its
weights are trained. `make_frozen_projections` makes fixed ones.

Each source cell type makes one or more synapse types (`synapse_params`), with their
own kinetics and reversal potential. `g_clip` caps the conductance, which keeps the
time step stable.

`model(inputs)` takes input of shape `(batch, time, inputs)` and returns spikes of shape
`(batch, time, neurons)`. The state carries over to the next call, so a long simulation
can be run in chunks.

{doc}`../concepts/building-blocks` explains how to pick these pieces for your own
model, and the {doc}`../tutorials/index` go through complete examples.
