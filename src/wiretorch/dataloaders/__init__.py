"""Input and target data for the networks.

- :mod:`.unsupervised`: Poisson input spikes generated on the fly, from constant rates or
  a time-varying rate process.
- :mod:`.rate_processes`: time-varying rate processes (e.g. the OU pattern mixture).
- :mod:`.input_patterns`: static firing-rate patterns for the feedforward inputs.
- :mod:`.supervised`: recorded spike trains (a teacher's) as targets, and collate
  functions that arrange them as inputs and targets.
"""
