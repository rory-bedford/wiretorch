# Store simulations to zarr

{py:class}`~wiretorch.snn_runners.inference_runner.SNNInference` runs a model over a dataloader and records every
input and output:

```python
inference = wt.SNNInference(
    model,
    dataloader,               # named tuples with at least `input_spikes`
    device="cuda",
    output_mode="zarr",       # or "memory" to get numpy arrays back
    zarr_path=out / "spike_data.zarr",
    max_chunks=200,
    save_tracked_variables=False,
)
result = inference.run()
```

Each chunk is copied to the host and written by a background thread while the GPU
simulates the next one. The arrays have shape `(batch, time, ...)`, concatenated over
chunks along time and chunked on disk by trial and time block. Spikes are stored as
booleans, and the model's `dt` as an attribute.

Float arrays such as input rates and voltages take up most of the space and hardly
compress, and writing them to network storage can take longer than the simulation.
Leave out what you don't need (`save_tracked_variables=False`), or write to a local disk
and copy the result.
