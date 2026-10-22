"""Configuration modules for spiking neural network simulations.

This module provides a compositional configuration system where scripts load only
the configuration sections they need, rather than requiring monolithic parameter classes.

Example::

    import toml
    from wiretorch.configs import SimulationConfig, StudentTrainingConfig
    from wiretorch.configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig

    data = toml.load(params_file)
    simulation = SimulationConfig(**data['simulation'])
    recurrent = RecurrentLayerConfig(**data['recurrent'])
"""

# Shared configs
from .simulation import SimulationConfig, StudentSimulationConfig
from .training import (
    StudentTrainingConfig,
    EMTrainingConfig,
    StudentHyperparameters,
    Targets,
)
from .legacy import (
    TrainingConfig,
    LossWeights,
    Hyperparameters,
    VanRossumHyperparameters,
    VanRossumFiringRateHyperparameters,
    VanRossumRateHyperparameters,
    PoissonHyperparameters,
    PoissonKLHyperparameters,
)
from .network import (
    SimpleCellTypesConfig,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    ActivityConfig,
)

# Shared DataLoader keyword arguments for all experiment scripts.
# Using a single worker with forkserver avoids CUDA context inheritance
# while still overlapping zarr I/O with GPU compute. pin_memory speeds
# up CPU→GPU transfers. Thread limits are set in run_experiment.py.
DATALOADER_KWARGS = {
    "num_workers": 1,
    "pin_memory": True,
    "multiprocessing_context": "forkserver",
}

# Model-specific configs are imported from their respective modules:
# - configs.conductance_based
# - configs.current_based

__all__ = [
    # Simulation
    "SimulationConfig",
    "StudentSimulationConfig",
    # Training
    "TrainingConfig",
    "StudentTrainingConfig",
    "EMTrainingConfig",
    "LossWeights",
    "Hyperparameters",
    "StudentHyperparameters",
    "VanRossumHyperparameters",
    "VanRossumFiringRateHyperparameters",
    "VanRossumRateHyperparameters",
    "PoissonHyperparameters",
    "PoissonKLHyperparameters",
    "Targets",
    # Network
    "SimpleCellTypesConfig",
    "CellTypesConfig",
    "TopologyConfig",
    "WeightsConfig",
    "ActivityConfig",
    # DataLoader
    "DATALOADER_KWARGS",
]
