"""wiretorch: connectome-constrained network models in PyTorch.

    import wiretorch as wt

    model = wt.RecurrentRateNetwork(...)        # or ConductanceLIFNetwork, ...
    projection = wt.FrozenProjection(weights)   # or ScalingFactorProjection, ...

The most used names are here; everything else is importable from its subpackage:

- ``network_simulators``: the networks (``conductance``, ``current``, ``rate``
  families; ``projections``; ``two_layer``).
- ``synthetic_connectome``: generators of synthetic connectomes.
- ``snn_runners``: training (``SNNTrainer``), inference (``SNNInference``) and
  evolutionary search.
- ``training_utils``: losses, checkpointing, logging, surrogate gradients.
- ``dataloaders``, ``observation_models``, ``analysis``, ``visualization``, ``configs``,
  ``utils``.

Names and subpackages load on first use, so ``import wiretorch`` is quick and needs
neither torch nor triton until a model is touched.
"""

import importlib

#: Top-level name -> the module that defines it.
_NAMES = {
    # Networks
    "ConductanceLIFNetwork": "network_simulators.conductance",
    "FeedforwardConductanceLIFNetwork": "network_simulators.conductance",
    "CurrentLIFNetwork": "network_simulators.current",
    "FeedforwardCurrentLIFNetwork": "network_simulators.current",
    "RecurrentRateNetwork": "network_simulators.rate",
    "FeedforwardRateNetwork": "network_simulators.rate",
    "TwoLayerSNN": "network_simulators.two_layer",
    # Projections
    "FrozenProjection": "network_simulators.projections",
    "ScalingFactorProjection": "network_simulators.projections",
    "FullRankProjection": "network_simulators.projections",
    "LowRankProjection": "network_simulators.projections",
    "make_frozen_projections": "network_simulators.projections",
    "make_scaling_factor_projections": "network_simulators.projections",
    # Running
    "SNNTrainer": "snn_runners",
    "SNNInference": "snn_runners",
    "EvolutionarySearch": "snn_runners",
    # Losses
    "VanRossumLoss": "training_utils.losses",
    "VanRossumRateLoss": "training_utils.losses",
    "PoissonKLLoss": "training_utils.losses",
    "FiringRateLoss": "training_utils.losses",
    "SilentNeuronLoss": "training_utils.losses",
    "CalciumMSELoss": "training_utils.losses",
}
_SUBPACKAGES = (
    "analysis",
    "configs",
    "dataloaders",
    "network_simulators",
    "observation_models",
    "snn_runners",
    "synthetic_connectome",
    "training_utils",
    "utils",
    "visualization",
)

__all__ = [*_NAMES, *_SUBPACKAGES]


def __getattr__(name: str):
    if name in _NAMES:
        value = getattr(importlib.import_module(f"{__name__}.{_NAMES[name]}"), name)
    elif name in _SUBPACKAGES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # load once
    return value


def __dir__():
    return sorted([*globals(), *__all__])
