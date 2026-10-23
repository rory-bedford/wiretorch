# Current-based LIF

The {doc}`rate neuron <rate>` with a threshold, spikes and a reset in place of the
nonlinearity (`CurrentLIFNetwork`, `FeedforwardCurrentLIFNetwork`).

## The model

Between inputs, each synapse type's current decays and the voltage leaks towards
$U_\text{rest}$:

$$
\tau_s \frac{dI_s}{dt} = -I_s, \qquad
\tau_m \frac{dv}{dt} = -(v - U_\text{rest}) + R \sum_s \sigma_s I_s
$$

What changes from the rate neuron:

- **Spikes.** When $v$ crosses the threshold $\theta$ the neuron spikes and $v$ is
  reset to $U_\text{reset}$. The spike is a step function of $v$; for gradients it has
  a surrogate derivative, and in `spike_mode="sample"` spikes are drawn with
  probability $\text{sigmoid}(\text{surrgrad\_scale}\,(v - \theta))$
  ({doc}`../learning`). Gradients never flow through the reset.
- **Inputs are events.** Each input spike makes the current of every synapse type its
  source makes jump by the weight, so one spike moves the voltage by $R w \tau_s /
  \tau_m$ in all, whatever the time step. Input rows hold counts per step (spikes); a
  continuous rate $\rho$ enters as $\rho\,\Delta t$.

The defaults are natural units: $\theta = 1$, $U_\text{reset} = U_\text{rest} = 0$,
$R = 1$.

## Refractory period

A cell type's `tau_ref` (ms, default 0) is rounded to a whole number of steps,
$n_\text{ref} = \operatorname{round}(\tau_\text{ref} / \Delta t)$. For the $n_\text{ref}$
steps after a spike, $v$ is held at $U_\text{reset}$ and the neuron cannot spike, even
with escape noise; its synaptic currents keep evolving. A counter per neuron is part of
the state, so the period continues across `forward` calls and is saved in checkpoints.
Gradients do not flow through a held voltage, as with the reset.

## Spike-frequency adaptation

A cell type with `adapt > 0` has, per neuron, a slow inhibitory current onto itself
that each of its own spikes raises: under constant drive its intervals lengthen. The
adaptation current $a$ decays with `tau_adapt` (ms, default 100) and rises by `adapt`
at each spike, and the voltage feels it as a negative input:

$$
\tau_a \frac{da}{dt} = -a + \text{adapt} \sum_k \delta(t - t_k), \qquad
\tau_m \frac{dv}{dt} = -(v - U_\text{rest}) + R \left(\sum_s \sigma_s I_s - a\right)
$$

`adapt` is in the units of the input (in natural units, a fraction of the threshold
per unit $R$). The adaptation is part of the state, saved in checkpoints, and tracked
(`track_variables`) as `adaptation`. `adapt` and `tau_adapt` can be learnt for the
cell types that adapt; as with the reset, no gradient flows through the spike that
raises the adaptation, but it does through the adaptation's effect on the voltage.

## Short-term depression and facilitation

Release is presynaptic, and how it changes with use depends on the connection: the same
pyramidal axon makes depressing synapses onto some cells and facilitating ones onto
others (Markram, Wang & Tsodyks 1998; Reyes et al. 1998). Short-term plasticity is
therefore set per (source cell type, target cell type) pair, like the projections, and
applies to every synapse type that connection makes:

```python
import wiretorch as wt

model = wt.CurrentLIFNetwork(
    ...,
    rec_stp={("E", "E"): wt.stp.DEPRESSING, ("E", "SST"): wt.stp.FACILITATING},
    ff_stp={("input", "E"): wt.STP(U=0.4, tau_D=300.0, tau_F=0.0)},
)
```

(`FeedforwardCurrentLIFNetwork` takes `stp`.) Each source neuron keeps, per pair with
STP, its available resources $x$ (at rest 1) and its release probability $u$ (at rest
$U$), as in the Tsodyks–Markram model:

$$
\tau_D \frac{dx}{dt} = 1 - x - \tau_D\, u x \sum_k \delta(t - t_k), \qquad
\tau_F \frac{du}{dt} = U - u + \tau_F\, U (1 - u) \sum_k \delta(t - t_k)
$$

A spike releases $u x$ of the resources, then facilitates $u$. Its input is the release
over $U$, so **a first spike from rest has exactly its weight**: switching STP on does
not rescale a network, it shapes what follows. Event by event this is the map of
Markram, Wang & Tsodyks (1998).

The two effects compete through $U$: a high release probability drains the resources
(depression), a low one leaves them and lets the release probability grow
(facilitation). In rat neocortex:

| Connection | Typically | Preset |
|---|---|---|
| Pyramidal → pyramidal | depressing | `stp.DEPRESSING`: $U = 0.5$, $\tau_D = 800$ ms, $\tau_F = 0$ (Tsodyks & Markram 1997) |
| Pyramidal → PV (fast-spiking) interneuron | depressing | `stp.DEPRESSING` |
| Pyramidal → SST (Martinotti) interneuron | facilitating | `stp.FACILITATING`: $U = 0.03$, $\tau_D = 130$ ms, $\tau_F = 530$ ms (Tsodyks, Pawelzik & Markram 1998) |

The state is part of the network's, saved in checkpoints, and carries no gradient: a
spike's gradient is scaled by its factor. The STP parameters are fixed. Each distinct
STP a source cell type has costs one more read of its weight rows per step, only when
used.

## Gradients through recurrent spikes

In a recurrent network a spike reaches the loss directly and also through the neurons it
drives. `recurrent_gradients=True` (the default for current-based networks) carries the
gradient along that second path too, back through the recurrent weights into the
presynaptic neuron's surrogate. With `False`, each step's recurrent input is treated as
given: the gradients still reach the recurrent weights and scaling factors, the
voltages and synaptic state, but not the spiking neurons upstream. `False` is about 3x
faster to train on a GPU, because the backward pass is a single reverse scan; with
`True` all neurons are coupled at every step, so it steps back one step at a time.

## The time step

1. **Spike** if $v > \theta$ and the neuron is not refractory.
2. **Inputs.** Each current decays, $I_s \leftarrow \alpha_s I_s$, then each source's
   spike adds its weight to the currents of the synapse types it makes — times its STP
   factor $u x / U$, with the source's $x$ and $u$ stepped as above.
3. **Adaptation** decays and rises by the spike: $a \leftarrow e^{-\Delta t /
   \tau_a} a + \text{adapt}\cdot s$.
4. **Voltage**: $v \leftarrow \beta v + (1 - \beta) \left(U_\text{rest} + R
   \left(\sum_s \sigma_s I_s - a\right)\right)$, with $\beta = e^{-\Delta t / \tau_m}$;
   then reset where the neuron spiked, held at reset while it is refractory.

{doc}`conductance-lif` turns the currents into conductances times a driving force.
