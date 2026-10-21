"""Current-based LIF networks: ``recurrent`` and ``feedforward``, and their engine.

The same layout as ``conductance``: ``network`` (the ``nn.Module`` base class, on the
shared ``network_simulators.network``) owns the physiology; ``dynamics`` is the time
step in plain PyTorch; ``kernels`` the same step as fused Triton kernels; ``engine``
chooses between them and wraps the kernels for autograd; ``legacy`` builds and loads
networks written with the former weight-matrix API (``from_weights``).
"""

from connectome_snns.network_simulators.current.feedforward import (
    FeedforwardCurrentLIFNetwork,
)
from connectome_snns.network_simulators.current.recurrent import CurrentLIFNetwork

__all__ = ["CurrentLIFNetwork", "FeedforwardCurrentLIFNetwork"]
