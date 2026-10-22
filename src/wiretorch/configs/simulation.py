"""Shared simulation configuration parameters.

Configuration for simulation settings that are common across all model types.
"""

from typing import Literal, Optional
from pydantic import BaseModel
from .base import Config

#: What a model trains: its weights, or its scaling factors on one or both sides.
Optimisable = Literal[
    "weights",
    "scaling_factors",
    "scaling_factors_recurrent",
    "scaling_factors_feedforward",
]


class SimulationConfig(Config):
    """Basic simulation parameters."""

    dt: float
    duration: float
    seed: int
    chunk_size: int = 1000
    batch_size: Optional[int] = None
    plot_size: Optional[int] = None
    optimisable: Optimisable | None = None

    @property
    def num_chunks(self) -> int:
        """Total number of chunks"""
        return int(self.duration / (self.chunk_size * self.dt))

    @property
    def chunk_duration_s(self) -> float:
        """Duration of a single chunk in seconds."""
        return self.chunk_size * self.dt / 1000.0

    @property
    def total_duration_s(self) -> float:
        """Total simulation duration in seconds."""
        return self.num_chunks * self.chunk_duration_s


class StudentSimulationConfig(BaseModel):
    """Minimal simulation config for student training.

    Unlike SimulationConfig, this doesn't have dt or duration in TOML.
    Instead, dt comes from the dataset and num_chunks is computed from
    epochs * chunks_per_epoch. Call setup() after loading the dataset.

    Only archived run scripts use it, and they read keys it does not declare from the
    same table, so unlike the current configs it ignores unknown keys.
    """

    seed: int
    chunk_size: int

    # These are set by setup() method - must call setup() after loading dataset
    dt: float | None = None
    num_chunks: int = 0

    def setup(self, dt: float, chunks_per_epoch: int, epochs: int) -> None:
        """Initialize computed simulation parameters from dataset and training config.

        Args:
            dt: Timestep in milliseconds (from dataset zarr file).
            chunks_per_epoch: Number of chunks in one epoch (from dataset).
            epochs: Number of training epochs (from training config).
        """
        self.dt = dt
        self.num_chunks = epochs * chunks_per_epoch

    @property
    def chunk_duration_s(self) -> float:
        """Duration of a single chunk in seconds."""
        if self.dt is None:
            raise ValueError("dt not initialized - call setup() after loading dataset")
        return self.chunk_size * self.dt / 1000.0

    @property
    def total_duration_s(self) -> float:
        """Total training duration in seconds."""
        if self.dt is None:
            raise ValueError("dt not initialized - call setup() after loading dataset")
        return self.num_chunks * self.chunk_duration_s
