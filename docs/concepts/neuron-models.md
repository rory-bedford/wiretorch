# Neuron models

wiretorch has three families of neuron, each building on the one before:

1. {doc}`neuron-models/rate`: a leaky voltage driven by exponential synaptic currents,
   read out through a nonlinearity.
2. {doc}`neuron-models/current-lif`: the same neuron, with a threshold, spikes and a
   reset instead of the nonlinearity.
3. {doc}`neuron-models/conductance-lif`: the synaptic currents become conductances
   times a driving force, with reversal potentials and rise and decay kinetics.

Each page starts from the model of the one before and describes only what changes,
then the mechanisms you can add. A mechanism is described in full where it first
appears, and every mechanism is off by default and costs nothing when off. Each page
ends with the complete time step.

| Mechanism | Rate | Current-based | Conductance-based | Set by |
|---|:-:|:-:|:-:|---|
| Exponential synaptic currents | ✓ | ✓ | | `tau_syn` |
| Conductances with rise and decay, reversal potentials | | | ✓ | `tau_rise`, `tau_decay`, `E_syn`, `g_bar` |
| Spikes, reset, surrogate gradients | | ✓ | ✓ | `theta`, `U_reset`, `surrgrad_scale` |
| [Refractory period](neuron-models/current-lif.md#refractory-period) | | ✓ | ✓ | `tau_ref` |
| [Spike-frequency adaptation](neuron-models/current-lif.md#spike-frequency-adaptation) | | ✓ | ✓ | `adapt`, `tau_adapt` (`E_adapt`) |
| [Short-term depression and facilitation](neuron-models/current-lif.md#short-term-depression-and-facilitation) | | ✓ | ✓ | `rec_stp`, `ff_stp` |
| [Gradients through recurrent spikes](neuron-models/current-lif.md#gradients-through-recurrent-spikes) | | on | off | `recurrent_gradients` |
| [Clipping](neuron-models/conductance-lif.md#clipping) | | | ✓ | `g_clip` |
| {ref}`NMDA Mg²⁺ block <nmda-mg-block>` | | | ✓ | `mg`, `mg_slope`, `mg_k` |
| [Receptor saturation](neuron-models/conductance-lif.md#receptor-saturation) | | | ✓ | `saturation` |

"on" and "off" are the defaults of an option both families have. The parameters are
listed in {doc}`cells-and-synapses`. Each model's time step is written in plain PyTorch
in its `dynamics.py`, and the GPU kernels are tested against it. Every family
integrates exactly what it can: the synaptic variables decay exactly between inputs,
and the voltage takes an exponential-Euler step, exact for the inputs held over the
step.

```{toctree}
:hidden:

neuron-models/rate
neuron-models/current-lif
neuron-models/conductance-lif
```
