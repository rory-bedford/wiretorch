"""Supervised spike-train datasets and collate functions."""

from ._base import CyclicSampler, SpikeData, TeacherSpikeDataset
from .collate import (
    FeedforwardCollate,
    VisibleDrivenCollate,
    VisibleSubsetCollate,
    feedforward_collate_fn,
    calcium_collate_fn,
    single_neuron_collate_fn,
)
from .exact import ExactFFDataset
from .latent_ou import LatentOUDataset
from .ou import OUReconstructedFFDataset
from .poisson import HomogeneousPoissonFFDataset
from .probabilistic import ProbabilisticSpikeDataset
from .rate_calcium import ParallelSegmentCalciumDataset, RateCalciumFFDataset

__all__ = [
    "SpikeData",
    "TeacherSpikeDataset",
    "ExactFFDataset",
    "HomogeneousPoissonFFDataset",
    "LatentOUDataset",
    "OUReconstructedFFDataset",
    "ProbabilisticSpikeDataset",
    "RateCalciumFFDataset",
    "ParallelSegmentCalciumDataset",
    "FeedforwardCollate",
    "VisibleDrivenCollate",
    "VisibleSubsetCollate",
    "feedforward_collate_fn",
    "calcium_collate_fn",
    "single_neuron_collate_fn",
    "CyclicSampler",
]
