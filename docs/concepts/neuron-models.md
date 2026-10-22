# Neuron models

wiretorch has conductance-based and current-based leaky integrate-and-fire neurons, and
rate neurons. They share the same interface: cell types, synapse types, projections, and
a state that persists across calls. Each model's time step is written in plain PyTorch
in its `dynamics.py`, and the GPU kernels are tested against it. The equations are
summarised below.

## Conductance-based LIF

Each synapse type $s$ has a conductance made of a rise and a decay component. The
synaptic current depends on the voltage through the reversal potential:

$$
C_m \frac{dv}{dt} = -g_L (v - E_L) - \sum_s \min(\max(g_s, 0), g^{\text{clip}}_s)\,(v - E_s)
$$

with $C_m = \tau_m g_L$. When $v$ crosses $\theta$ the neuron spikes and $v$ is reset to
$U_\text{reset}$. An input spike increases the conductance of each synapse type its
source makes by the weight times the peak conductance $\bar g$ (the difference of
exponentials is normalised to peak at $\bar g$). In each step the leak and the
conductances decay exactly, and the synaptic currents take a forward-Euler step.

Gradients flow through the voltage and the conductances using a surrogate for the
spike. They do not flow through recurrent spikes: the effect of a neuron's spike on
other neurons is treated as input.

## Current-based LIF

Each synapse type $s$ carries a current that decays with $\tau_s$ and increases by the
weight on each input spike:

$$
\tau_m \frac{dv}{dt} = -(v - U_\text{rest}) + R \sum_s \sigma_s I_s, \qquad
\tau_s \frac{dI_s}{dt} = -I_s
$$

where $\sigma_s = \pm 1$ is the sign of the synapse type. The step uses exponential
Euler, normalised so that one spike moves the voltage by $R w \tau_s / \tau_m$
independent of the time step. The defaults are natural units: $\theta = 1$,
$U_\text{reset} = U_\text{rest} = 0$, $R = 1$. Gradients also flow through recurrent
spikes.

## Rate

The rate neuron is the current-based neuron with a nonlinearity in place of the spike:

$$
\tau_m \frac{dv}{dt} = -v + \sum_s \sigma_s I_s, \qquad
\tau_s \frac{dI_s}{dt} = -I_s + \text{input}_s, \qquad
r = \text{gain}\cdot f(v - \theta)
$$

with $f$ one of `relu`, `softplus`, `sigmoid` or `tanh`, chosen per cell type. The steps
use exponential Euler, normalised so that with constant input $v$ and $I$ settle at that
input for any time step. A time constant of 0 makes the variable equal its input. This
gives the two standard rate models as special cases:

- Voltage form, $\tau \dot x = -x + W \phi(x) + W_\text{ff} u$: set every $\tau_s = 0$
  and $\tau_m = \tau$. Then $v$ is $x$ and the output is $\phi(x)$. The
  {doc}`../tutorials/ring_attractor` uses this form.
- Rate form, $\tau \dot r = -r + \phi(W r + W_\text{ff} u)$: set $\tau_m = 0$, the
  recurrent synapse types' $\tau_s = \tau$ and the input ones' to 0. Then
  `rate.filter_rates` converts the output to $r$.

The model has no noise of its own. To add intrinsic noise, give each neuron its own
noisy input row through a diagonal projection.
