"""Conductance-based model configurations."""

from .physiology import (
    PhysiologyConfig,
    SynapseConfig,
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)
from .network import RecurrentLayerConfig, FeedforwardLayerConfig

__all__ = [
    "PhysiologyConfig",
    "SynapseConfig",
    "EXCITATORY_SYNAPSE_TYPES",
    "INHIBITORY_SYNAPSE_TYPES",
    "RecurrentLayerConfig",
    "FeedforwardLayerConfig",
]
