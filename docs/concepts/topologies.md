# Recurrent, feedforward and two-layer networks

## Recurrent

The neurons of a recurrent network receive input from each other (`rec_projections`)
and from external inputs (`ff_projections`). `forward` takes the external input, of
shape `(batch, time, n_inputs)`, and the recurrent input is computed as the network
runs.

## Feedforward

A feedforward network only receives input from its input rows (`projections`). The
rows can be external stimuli, but they can also be recorded spikes of other neurons.
When a model is fitted to recordings this way, each neuron is driven by the recorded
spikes of the others rather than by the model's own, so every neuron can be fitted
independently. This is known as teacher forcing.

Since the whole input of a chunk is known in advance, a feedforward network runs the
chunk as a single kernel on the GPU, which is much faster than stepping a recurrent
network.

## Two layers

When only some neurons in a circuit are recorded,
{py:class}`~wiretorch.network_simulators.two_layer.TwoLayerSNN` models the others. Layer
1 is a recurrent network of the unrecorded neurons. Layer 2 is a feedforward network of
the recorded neurons, driven by their recordings and by the spikes of layer 1.
Gradients reach layer 1 through its effect on layer 2. Projections can be shared
between the layers, so that a connection type has the same parameters in both.
