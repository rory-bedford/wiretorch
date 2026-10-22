"""Configuration classes no current code uses, kept so old scripts still import them.

``TrainingConfig`` and the homeostatic-plasticity ``Hyperparameters``/``LossWeights``
predate the student experiments; the ``VanRossum*``/``Poisson*`` classes were the
solving-bias experiment's per-loss hyperparameters. All are still importable from
``wiretorch.configs`` and ``wiretorch.configs.training``.
"""

from typing import Optional

from pydantic import BaseModel

from .simulation import Optimisable


class TrainingConfig(BaseModel):
    """Basic training parameters."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    plot_size: int
    optimisable: Optimisable | None = None


class LossWeights(BaseModel):
    """Loss function weights for homeostatic plasticity."""

    firing_rate: float
    cv: float
    silent_penalty: float
    membrane_variance: float
    weight_ratio: float


class Hyperparameters(BaseModel):
    """Optimization hyperparameters for homeostatic plasticity."""

    surrgrad_scale: float
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    grad_norm_clip: Optional[float] = None
    loss_weight: LossWeights


class _StudentHyperparametersBase(BaseModel):
    model_config = {"extra": "forbid"}

    surrgrad_scale: float
    learning_rate: float
    lr_min: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.999
    grad_norm_clip: Optional[float] = None


class VanRossumLossWeights(BaseModel):
    model_config = {"extra": "forbid"}

    van_rossum: float


class VanRossumFiringRateLossWeights(BaseModel):
    model_config = {"extra": "forbid"}

    van_rossum: float
    firing_rate: float


class VanRossumRateLossWeights(BaseModel):
    model_config = {"extra": "forbid"}

    van_rossum: float
    van_rossum_rate: float


class PoissonLossWeights(BaseModel):
    model_config = {"extra": "forbid"}

    poisson: float


class PoissonKLLossWeights(BaseModel):
    model_config = {"extra": "forbid"}

    poisson_kl: float


class VanRossumHyperparameters(_StudentHyperparametersBase):
    van_rossum_tau_rise: float
    van_rossum_tau_decay: float
    loss_weight: VanRossumLossWeights


class VanRossumFiringRateHyperparameters(_StudentHyperparametersBase):
    van_rossum_tau_rise: float
    van_rossum_tau_decay: float
    loss_weight: VanRossumFiringRateLossWeights


class VanRossumRateHyperparameters(_StudentHyperparametersBase):
    van_rossum_tau_rise: float
    van_rossum_tau_decay: float
    van_rossum_rate_tau_rise: float
    van_rossum_rate_tau_decay: float
    loss_weight: VanRossumRateLossWeights


class PoissonHyperparameters(_StudentHyperparametersBase):
    loss_weight: PoissonLossWeights


class PoissonKLHyperparameters(_StudentHyperparametersBase):
    van_rossum_tau_rise: float
    van_rossum_tau_decay: float
    loss_weight: PoissonKLLossWeights
