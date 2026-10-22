# State, chunks and tracking

## The state persists between calls

A network keeps its state (voltages `v` and the synaptic variables `g` or `I_syn`)
between calls, so a long simulation can be run in chunks:

```python
for chunk in chunks:          # each (batch, steps, inputs)
    spikes = model(chunk)     # continues from where the last call stopped
```

This gives the same result as one long call. To start a new simulation, call
`model.reset_state()`, or `reset_state(batch_size=...)` to change the number of trials.
Gradients don't cross chunk boundaries: the backward pass of each chunk starts from the
state it was given.

## Trials

`batch_size` independent trials run in parallel. Inputs and outputs have the batch as
their first dimension.

## Tracking variables

By default `forward` returns spikes or rates. With `model.track_variables = True` it
returns a dict that also contains the voltages and synaptic variables at every step,
for trial `model.track_batch_idx`, or for every trial if it is `None`. Tracking doesn't
change the dynamics.

## Saving the state

`model.get_checkpoint_state()` and `load_checkpoint_state(...)` save and restore the
state, so a simulation or a training run can be resumed exactly. The trainer includes
this in its checkpoints ({doc}`../how-to/checkpoints`).
