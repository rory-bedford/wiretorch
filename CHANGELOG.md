# Changelog

The release notes of each version on GitHub are taken from its section here.

## [Unreleased]

- **Breaking (numerics):** the conductance-based voltage takes an exponential-Euler step
  instead of a forward-Euler one, exact for the conductances held over the step and
  stable however large they are, so `g_clip` is no longer needed for stability. Spike
  trains differ slightly from earlier versions. The tracked leak current is now
  `g_L (v - E_L)`.

- Removed the former name `connectome_snns` (the import alias, `CONNECTOME_SNNS_BACKEND`,
  and reading `connectome_snns_commit` from old experiment metadata): use `wiretorch`.
- `recurrent_gradients` on `ConductanceLIFNetwork` and `CurrentLIFNetwork`: whether
  gradients flow through the recurrent spikes into the presynaptic neurons. Both
  families now support both settings, with Triton kernels for each. The defaults are
  unchanged: off for conductance-based networks, on for current-based ones.
- Refractory period `tau_ref` (ms, per cell type, default 0) in both spiking families,
  with Triton kernels: for that long after a spike a neuron is held at `U_reset` and
  cannot spike. The counters are part of the state, carried across `forward` calls and
  checkpoints. `tau_ref` in a config was accepted and ignored before: **configs that
  set it now get the refractory period they name.**
- Voltage-dependent NMDA in conductance-based networks: a synapse type's optional `mg`
  ([Mg²⁺], mM), `mg_slope` and `mg_k` give its current the Jahr–Stevens Mg²⁺ block.
  Fused into the existing kernels and compiled in only when a synapse type has a block.
- Receptor saturation in conductance-based networks: a synapse type's optional
  `saturation` gives each source neuron a gating variable (the fraction of its receptors
  open, decaying with `tau_decay`), and each spike opens that fraction of the closed
  ones, as in Brunel & Wang's (2001) NMDA. Its `g_bar` is then the conductance with every
  receptor open. The gating is part of the state and the checkpoints; Triton kernels for
  every path.

## [0.1.0b1] - 2026-10-22

First public beta.

- Recurrent, feedforward and two-layer networks built from a connectome, with
  conductance-based and current-based spiking neurons and rate neurons.
- Triton kernels for simulation and gradients on NVIDIA GPUs, and a PyTorch
  implementation for the CPU.
- Fixed, scaling-factor, low-rank and full-rank projections; learnable time constants,
  thresholds and other per-type parameters.
- `SNNTrainer` and `SNNInference`, with losses for spikes, rates and calcium signals.
- Renamed from `connectome_snns`, which still imports with a deprecation warning.
