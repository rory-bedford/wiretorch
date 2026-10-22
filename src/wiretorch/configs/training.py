"""Training configuration of the archived student runs.

Only archived run scripts use these classes. They read keys the classes do not declare
from the same tables, so unlike the current configs (``base.Config``) these ignore
unknown keys.
"""

from typing import Dict, Optional
from pydantic import BaseModel

from .simulation import Optimisable


class StudentTrainingConfig(BaseModel):
    """Training config for student network training."""

    epochs: Optional[int] = None
    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    plot_size: int
    mixed_precision: bool
    weight_perturbation_variance: float
    burn_in_chunks: int = 0
    recurrent_smoothing_tau: Optional[float] = None
    optimisable: Optimisable | Dict[str, Optional[str]] | None = None

    def total_chunks(self, num_chunks_per_epoch: int) -> int:
        """Total number of chunks across all epochs.

        Args:
            num_chunks_per_epoch: Number of chunks in one epoch.

        Returns:
            Total number of training chunks.

        Raises:
            ValueError: If epochs is not set.
        """
        if self.epochs is None:
            raise ValueError(
                "epochs not set in [training] — use phase-specific configs instead"
            )
        return self.epochs * num_chunks_per_epoch

    def log_interval_s(self, chunk_duration_s: float) -> float:
        """Time interval between logging in seconds.

        Args:
            chunk_duration_s: Duration of one chunk in seconds.

        Returns:
            Log interval in seconds.
        """
        return self.log_interval * chunk_duration_s

    def checkpoint_interval_s(self, chunk_duration_s: float) -> float:
        """Time interval between checkpoints in seconds.

        Args:
            chunk_duration_s: Duration of one chunk in seconds.

        Returns:
            Checkpoint interval in seconds.
        """
        return self.checkpoint_interval * chunk_duration_s


class EMTrainingConfig(BaseModel):
    """Training config for EM-based training (no epochs needed)."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    plot_size: int
    mixed_precision: bool
    weight_perturbation_variance: float
    burn_in_chunks: int = 0
    optimisable: Optimisable


class StudentLossWeights(BaseModel):
    """Loss function weights for student training."""

    van_rossum: float
    firing_rate: float = 0.0
    silence_penalty: float = 0.0
    van_rossum_rate: float = 0.0
    hidden_rate: float = 0.0  # legacy — prefer hidden_rate_mean
    hidden_rate_mean: float = 0.0
    hidden_rate_std: float = 0.0
    ff_matrix_l1: float = 0.0  # fallback for all cell types
    ff_l1_excitatory: float = 0.0  # per-type override (0 = use ff_matrix_l1)
    ff_l1_inhibitory: float = 0.0


class StudentHyperparameters(BaseModel):
    """Optimization hyperparameters for student training."""

    surrgrad_scale: Optional[float] = None
    learning_rate: Optional[float] = None
    lr_min: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.999
    grad_norm_clip: Optional[float] = None
    van_rossum_tau_rise: Optional[float] = None
    van_rossum_tau_decay: Optional[float] = None
    van_rossum_rate_tau_rise: Optional[float] = None
    van_rossum_rate_tau_decay: Optional[float] = None
    loss_weight: StudentLossWeights


class Targets(BaseModel):
    """Target values for homeostatic plasticity training."""

    firing_rate: Dict[str, float]
    alpha: Dict[str, float]
    threshold_ratio: Dict[str, float]
    weight_ratio: float


# Unused, kept importable from here (see legacy.py).
from .legacy import (  # noqa: E402, F401
    Hyperparameters,
    LossWeights,
    PoissonHyperparameters,
    PoissonKLHyperparameters,
    PoissonKLLossWeights,
    PoissonLossWeights,
    _StudentHyperparametersBase,
    TrainingConfig,
    VanRossumFiringRateHyperparameters,
    VanRossumFiringRateLossWeights,
    VanRossumHyperparameters,
    VanRossumLossWeights,
    VanRossumRateHyperparameters,
    VanRossumRateLossWeights,
)
