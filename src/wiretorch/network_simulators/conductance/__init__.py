"""Conductance-based LIF networks: ``recurrent`` and ``feedforward``, and their engine.

``network`` (the ``nn.Module`` base class) organises connectivity into pathways and owns
the state; ``dynamics`` is the time step in plain PyTorch; ``kernels`` the same step as
fused Triton kernels; ``engine`` chooses between them and wraps the kernels for autograd;
``synapses`` builds the per-neuron and per-synapse parameter tables.
"""

from wiretorch.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from wiretorch.network_simulators.conductance.recurrent import (
    ConductanceLIFNetwork,
)

__all__ = ["ConductanceLIFNetwork", "FeedforwardConductanceLIFNetwork"]
