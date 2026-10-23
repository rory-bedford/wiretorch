# Rate neurons

The simplest of the three models ({doc}`../neuron-models`): a leaky voltage driven by
exponential synaptic currents, and a nonlinearity in place of the spike.

## The model

Each synapse type $s$ carries a current $I_s$ that follows its input with time constant
$\tau_s$, and the voltage follows the signed sum of the currents with time constant
$\tau_m$:

$$
\tau_s \frac{dI_s}{dt} = -I_s + \text{input}_s, \qquad
\tau_m \frac{dv}{dt} = -v + \sum_s \sigma_s I_s, \qquad
r = \text{gain}\cdot f(v - \theta)
$$

where $\sigma_s = \pm 1$ is the sign of the synapse type (its `type`, excitatory or
inhibitory), and $\text{input}_s$ is the weighted sum of the rates of the sources that
make synapse type $s$. The nonlinearity $f$ is one of `relu`, `softplus`, `sigmoid` or
`tanh`, chosen per cell type, with `theta` and `gain` per cell type too.

## The two standard rate models

A time constant of 0 makes the variable equal its input, which gives the two standard
rate models as special cases:

- Voltage form, $\tau \dot x = -x + W \phi(x) + W_\text{ff} u$: set every $\tau_s = 0$
  and $\tau_m = \tau$. Then $v$ is $x$ and the output is $\phi(x)$. The
  {doc}`../../tutorials/ring_attractor` uses this form.
- Rate form, $\tau \dot r = -r + \phi(W r + W_\text{ff} u)$: set $\tau_m = 0$, the
  recurrent synapse types' $\tau_s = \tau$ and the input ones' to 0. Then
  `rate.filter_rates` converts the output to $r$.

The model has no noise of its own. To add intrinsic noise, give each neuron its own
noisy input row through a diagonal projection.

## The time step

Each step uses exponential Euler, normalised so that with constant input $v$ and $I$
settle at that input for any time step:

1. **Output** the rate of the voltage entering the step, $r = \text{gain}\cdot f(v -
   \theta)$: the presynaptic rates of this step's input.
2. **Currents** relax towards their input: $I_s \leftarrow \alpha_s I_s + (1 - \alpha_s)\,
   \text{input}_s$, with $\alpha_s = e^{-\Delta t / \tau_s}$.
3. **Voltage** relaxes towards the signed currents: $v \leftarrow \beta v + (1 -
   \beta) \sum_s \sigma_s I_s$, with $\beta = e^{-\Delta t / \tau_m}$.

Gradients flow everywhere: through the rates, the recurrence and the state.
{doc}`current-lif` replaces step 1 with a threshold, a spike and a reset.
