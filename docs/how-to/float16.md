# Run faster with float16 storage

```python
model.storage_dtype = "float16"     # or storage_dtype="float16" when constructing
```

On the GPU, the synaptic state and recurrent weights are then stored in half precision,
while arithmetic stays in float32. The rate networks and the current-based backward pass
also run their large matrix products on tensor cores. Some timings from the
{doc}`../performance` page:

| | float32 | float16 |
|---|---|---|
| Conductance-based recurrent simulation | 0.157 s/s | 0.117 s/s |
| Current-based recurrent training | 0.68 s/s | 0.44 s/s |
| Rate recurrent training | 1.14 s/s | 0.69 s/s |

(Seconds of wall time per second of simulated time, for a 5000-neuron network and a
batch of 50.)

The stored state is rounded at every step, so a spiking simulation in float16 is a
different realisation of the same network. Firing rates agree with float32 to within
about 0.4% and gradients point the same way, but individual spike times diverge after
about 100 ms. Rate networks aren't chaotic, and their rates agree with float32 to about
1e-3. Keep the default, float32, if you need exact reproducibility. The CPU always
computes in float32.
