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
C_m \frac{dv}{dt} = -g_L (v - E_L) - \sum_s B_s(v)\,\min(\max(g_s, 0), g^{\text{clip}}_s)\,(v - E_s)
$$

with $C_m = \tau_m g_L$. $B_s(v) = 1$ unless the synapse type has a voltage-dependent
Mg²⁺ block, as NMDA receptors do (Jahr & Stevens, 1990):

$$
B_s(v) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]_s}{K_s}\, e^{-a_s v}}
$$

set per synapse type by `mg` ($[\text{Mg}^{2+}]$ in mM, default 0: no block; 1.0 is
physiological), `mg_slope` ($a$, default 0.062 /mV) and `mg_k` ($K$, default 3.57 mM).
The block scales the current, not the conductance, so NMDA's rise and decay are set by
`tau_rise` and `tau_decay` as for any synapse type. The Mg parameters are fixed: they
cannot be learnt.

When $v$ crosses $\theta$ the neuron spikes and $v$ is reset to
$U_\text{reset}$. An input spike increases the conductance of each synapse type its
source makes by the weight times the peak conductance $\bar g$ (the difference of
exponentials is normalised to peak at $\bar g$). In each step the leak and the
conductances decay exactly, and the synaptic currents take a forward-Euler step.

`g_clip` caps a synapse type's conductance (hard clipping). Saturation, below, is the
dynamical alternative: it limits how much of a source's receptors its spikes can open.

### Receptor saturation

A synapse type with `saturation` $c > 0$ saturates, as NMDA receptors do (Brunel &
Wang, 2001). Each source neuron $j$ keeps a gating variable $u_j \in [0, 1]$ per such
type, the fraction of its receptors that are open. In each step it decays with the
type's $\tau_\text{decay}$, and a spike then opens the fraction $c$ of the receptors
still closed:

$$
u_j \leftarrow u_j e^{-\Delta t / \tau_\text{decay}}, \qquad
f_j = c\,(1 - u_j), \qquad u_j \leftarrow u_j + s_j f_j
$$

The spike's input is $s_j f_j$ instead of $s_j$, so the target's conductance follows
$\bar g \sum_j w_{ij} u_j$, exactly as $\tau_\text{rise} \to 0$ and filtered by the
rise otherwise. For a saturating type $\bar g$ is therefore the conductance with every
receptor open: its kernel is not normalised to peak at $\bar g$. A first spike from rest
gives about $c \bar g$, and sustained firing approaches $\bar g$.

Brunel & Wang's NMDA gating opens receptors at rate $\alpha x (1 - u)$ through a
transmitter concentration $x$ that jumps at each spike and decays with
$\tau_{\text{rise}}$. Here each spike opens its share at once, the limit for spikes
further apart than $\tau_{\text{rise}}$. Their $\alpha = 0.5$ /ms over
$\tau_\text{rise} = 2$ ms corresponds to $c = 1 - e^{-\alpha \tau_\text{rise}} \approx
0.63$. Under Poisson input the mean gating is then within 0.03 of their $\psi(\nu)$
(about 0.17, 0.64 and 0.79 at 3, 25 and 50 Hz).

The gating variables are part of the state, carried across `forward` calls and saved
in checkpoints. They carry no gradient themselves: a spike's gradient is scaled by its
$f_j$. A saturating type's $\tau_\text{decay}$, which the gating shares, cannot be
learnt. Saturation needs its own copy of the source's synaptic input, so it costs about
15% in simulation and 40% in training on the teacher network.

Gradients flow through the voltage and the conductances using a surrogate for the
spike. By default they do not flow through recurrent spikes: the effect of a neuron's
spike on other neurons is treated as input (see [below](#gradients-through-recurrent-spikes)).

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
$U_\text{reset} = U_\text{rest} = 0$, $R = 1$. By default gradients also flow
through recurrent spikes.

## Refractory period

Both spiking models take an optional refractory period `tau_ref` per cell type, in ms.
It is rounded to a whole number of steps,
$n_\text{ref} = \operatorname{round}(\tau_\text{ref} / \Delta t)$. For the $n_\text{ref}$
steps after a spike, $v$ is held at $U_\text{reset}$ and the neuron cannot spike, even
with escape noise. Its synaptic conductances or currents keep evolving. A counter per
neuron is part of the state, so the period continues across `forward` calls and is
saved in checkpoints. Gradients do not flow through a held voltage, as with the reset.
The default, 0, gives the model without a refractory period.

## Gradients through recurrent spikes

In a recurrent spiking network, a spike reaches the loss directly and also through the
neurons it drives. `recurrent_gradients=True` carries the gradient along that second
path too, back through the recurrent weights into the presynaptic neuron's surrogate.
With `False`, each step's recurrent input is treated as given; the gradients still reach
the recurrent weights and scaling factors, and the voltages and synaptic states, but
not the spiking neurons upstream.

Both families support both settings. The default is `False` for
`ConductanceLIFNetwork` and `True` for `CurrentLIFNetwork`. `False` is about 3x faster
to train on a GPU, because the backward pass is a single reverse scan. With `True`, all
neurons are coupled at every step, so the backward pass steps back one step at a time.

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
