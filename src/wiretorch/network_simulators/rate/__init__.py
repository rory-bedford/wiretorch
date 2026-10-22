"""Rate networks: ``recurrent`` and ``feedforward``, and their engine.

The same layout as ``conductance`` and ``current``: ``network`` (the ``nn.Module``
base class, on the shared ``network_simulators.network``) owns the physiology, whose
per-type values can be learnt; ``dynamics`` is the time step in plain PyTorch, and
documents how the classic voltage-form and rate-form models are set up; ``engine``
runs it: Triton kernels on CUDA, the reference otherwise.
"""

from wiretorch.network_simulators.rate.dynamics import (
    NONLINEARITIES,
    filter_rates,
)
from wiretorch.network_simulators.rate.feedforward import FeedforwardRateNetwork
from wiretorch.network_simulators.rate.recurrent import RecurrentRateNetwork

__all__ = [
    "NONLINEARITIES",
    "FeedforwardRateNetwork",
    "RecurrentRateNetwork",
    "filter_rates",
]
