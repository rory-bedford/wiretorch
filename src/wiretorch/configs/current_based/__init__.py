"""Current-based model configurations."""

from .physiology import PhysiologyConfig
from .network import RecurrentLayerConfig, FeedforwardLayerConfig

__all__ = [
    "PhysiologyConfig",
    "RecurrentLayerConfig",
    "FeedforwardLayerConfig",
]
