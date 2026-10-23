# Changelog

The release notes of each version on GitHub are taken from its section here.

## [Unreleased]

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
