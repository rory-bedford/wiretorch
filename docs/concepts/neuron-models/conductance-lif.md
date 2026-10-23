# Conductance-based LIF

The {doc}`current-based neuron <current-lif>` with conductances in place of currents
(`ConductanceLIFNetwork`, `FeedforwardConductanceLIFNetwork`): a synapse's effect
depends on the voltage through its reversal potential.

## The model

Each synapse type $s$ has a conductance $g_s$ made of a rise and a decay component
(time constants `tau_rise` and `tau_decay`). Its current is the conductance times the
driving force $v - E_s$:

$$
C_m \frac{dv}{dt} = -g_L (v - E_L) - \sum_s g_s (v - E_s)
$$

with $C_m = \tau_m g_L$. What changes from the current-based neuron:

- **Kinetics.** An input spike raises the conductance of each synapse type its source
  makes by a difference of exponentials, normalised so that one spike of weight $w$
  peaks at $w \bar g$ (`g_bar`, nS).
- **Reversal potentials.** Whether a synapse excites or inhibits is set by $E_s$
  (`E_syn`) relative to the voltage, not by a sign.
- **Units.** Voltages are in mV, conductances in nS, $\tau_m$ in ms, and the leak's
  reversal potential is `E_L`.

The voltage takes an exponential-Euler step. With the conductances held over the step
the equation is linear, $C_m \dot v = -g_\text{tot} (v - V_\infty)$, where
$g_\text{tot} = g_L + \sum_s g_s$ and $V_\infty$ is the conductance-weighted mean of the
reversal potentials, so $v$ relaxes towards $V_\infty$ by $e^{-\Delta t\, g_\text{tot} /
C_m}$. The step is stable however large the conductances: $v$ never passes $V_\infty$,
which lies between the reversal potentials.

By default gradients do not flow through the recurrent spikes (`recurrent_gradients`,
see {doc}`current-lif`).

## Clipping

A synapse type's optional `g_clip` caps its conductance: $\min(\max(g_s, 0),
g^\text{clip}_s)$ enters the current. It is not needed for stability, which the
exponential-Euler step gives. [Receptor saturation](#receptor-saturation) is the
dynamical alternative.

(nmda-mg-block)=
## NMDA Mg²⁺ block

NMDA receptors pass current only when the membrane is depolarised enough to expel the
Mg²⁺ ion blocking their pore (Jahr & Stevens 1990). A synapse type with `mg` > 0 scales
its current by

$$
B_s(v) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]_s}{K_s}\, e^{-a_s v}}
$$

set by `mg` ($[\text{Mg}^{2+}]$ in mM, default 0: no block; 1.0 is physiological),
`mg_slope` ($a$, default 0.062 /mV) and `mg_k` ($K$, default 3.57 mM). The block scales
the current, not the conductance, so NMDA's rise and decay are set by `tau_rise` and
`tau_decay` as for any synapse type. The Mg parameters are fixed.

## Receptor saturation

A synapse type with `saturation` $c > 0$ saturates, as NMDA receptors do (Brunel &
Wang 2001). Each source neuron $j$ keeps a gating variable $g_j \in [0, 1]$ per such
type, the fraction of its receptors that are open. In each step it decays with the
type's $\tau_\text{decay}$, and a spike then opens the fraction $c$ of the receptors
still closed:

$$
g_j \leftarrow g_j e^{-\Delta t / \tau_\text{decay}}, \qquad
f_j = c\,(1 - g_j), \qquad g_j \leftarrow g_j + s_j f_j
$$

The spike's input is $s_j f_j$ instead of $s_j$, so the target's conductance follows
$\bar g \sum_j w_{ij} g_j$, exactly as $\tau_\text{rise} \to 0$ and filtered by the
rise otherwise. For a saturating type $\bar g$ is therefore the conductance with every
receptor open: its kernel is not normalised to peak at $\bar g$. A first spike from rest
gives about $c \bar g$, and sustained firing approaches $\bar g$.

Brunel & Wang's NMDA gating opens receptors at rate $\alpha x (1 - g)$ through a
transmitter concentration $x$ that jumps at each spike and decays with
$\tau_\text{rise}$. Here each spike opens its share at once, the limit for spikes
further apart than $\tau_\text{rise}$. Their $\alpha = 0.5$ /ms over
$\tau_\text{rise} = 2$ ms corresponds to $c = 1 - e^{-\alpha \tau_\text{rise}} \approx
0.63$. Under Poisson input the mean gating is then within 0.03 of their $\psi(\nu)$
(about 0.17, 0.64 and 0.79 at 3, 25 and 50 Hz).

The gating variables are part of the state, carried across `forward` calls and saved
in checkpoints. They carry no gradient themselves: a spike's gradient is scaled by its
$f_j$. A saturating type's $\tau_\text{decay}$, which the gating shares, cannot be
learnt. Saturation needs its own copy of the source's synaptic input, so it costs about
15% in simulation and 40% in training on a 5000-neuron network.

## Mechanisms shared with current-based neurons

- **[Refractory period](current-lif.md#refractory-period)**: as there.
- **[Spike-frequency adaptation](current-lif.md#spike-frequency-adaptation)**: here a
  potassium-like conductance onto the neuron itself, $g_a (v - E_\text{adapt})$, one
  more term of $g_\text{tot}$ in the voltage step. `adapt` is its rise per spike in nS,
  `E_adapt` its reversal potential (default −90 mV); `E_adapt` can be learnt too.
- **[Short-term depression and facilitation](current-lif.md#short-term-depression-and-facilitation)**:
  as there. On a saturating synapse type the release, not the spike, opens the
  receptors: $f_j = \min(c\, f^\text{STP}_j (1 - g_j),\, 1 - g_j)$, at most those still
  closed.

## The time step

1. **Spike** if $v > \theta$ and the neuron is not refractory.
2. **Voltage.** The conductances entering the step — clipped, times their Mg block —
   and the adaptation conductance give $g_\text{tot}$ and $V_\infty$; $v \leftarrow
   V_\infty + (v - V_\infty) e^{-\Delta t\, g_\text{tot} / C_m}$, then reset where the
   neuron spiked, held at reset while it is refractory.
3. **Adaptation** decays and rises by the spike: $g_a \leftarrow e^{-\Delta t /
   \tau_a} g_a + \text{adapt}\cdot s$.
4. **Inputs.** Each conductance component decays, then each source's spike adds its
   weight times the kernel to the synapse types it makes — times its STP factor and
   saturation factor, with the source's presynaptic state stepped as above.
