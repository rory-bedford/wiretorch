"""Shared network configuration parameters.

Configuration for network topology and structure that is common across model types.
"""

import re
from typing import List, Literal, Optional

import numpy as np
from pydantic import ConfigDict, Field
from .base import Config


def cell_type_sign(name: str) -> Optional[Literal["excitatory", "inhibitory"]]:
    """Whether a cell type is excitatory or inhibitory, read from its name.

    A word of the name starting with "exc" or "inh" decides ("Excitatory",
    "L23_exc", "inh-fast"); a word preceded by "non"/"not" is skipped, so
    "non-excitatory" is neither. None if no word decides.
    """
    words = re.findall(r"[a-z]+", name.lower())
    for previous, word in zip([""] + words, words):
        if previous in ("non", "not"):
            continue
        if word.startswith("exc"):
            return "excitatory"
        if word.startswith("inh"):
            return "inhibitory"
    return None


class SimpleCellTypesConfig(Config):
    """Cell types without proportions (for pre-defined networks)."""

    names: List[str]


class CellTypesConfig(SimpleCellTypesConfig):
    """Cell types with proportions (for generating networks)."""

    proportion_list: List[float] = Field(alias="proportion")

    @property
    def proportion(self) -> np.ndarray:
        """Cell type proportions as numpy array."""
        return np.array(self.proportion_list)

    model_config = ConfigDict(populate_by_name=True)


class TopologyConfig(Config):
    """Network topology configuration."""

    num_neurons: int
    num_assemblies: Optional[int] = None
    neurons_per_assembly: Optional[int] = None
    conn_within_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_within"
    )
    conn_between_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_between"
    )
    conn_inputs_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_inputs"
    )

    @property
    def conn_within(self) -> Optional[np.ndarray]:
        """Within-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_within_list)
            if self.conn_within_list is not None
            else None
        )

    @property
    def conn_between(self) -> Optional[np.ndarray]:
        """Between-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_between_list)
            if self.conn_between_list is not None
            else None
        )

    @property
    def conn_inputs(self) -> Optional[np.ndarray]:
        """Input connectivity as numpy array."""
        return (
            np.array(self.conn_inputs_list)
            if self.conn_inputs_list is not None
            else None
        )

    model_config = ConfigDict(populate_by_name=True)


class WeightsConfig(Config):
    """Weight distribution parameters."""

    w_mu_list: List[List[float]] = Field(alias="w_mu")
    w_sigma_list: List[List[float]] = Field(alias="w_sigma")

    @property
    def w_mu(self) -> np.ndarray:
        """Mean weights as numpy array."""
        return np.array(self.w_mu_list)

    @property
    def w_sigma(self) -> np.ndarray:
        """Weight standard deviations as numpy array."""
        return np.array(self.w_sigma_list)

    model_config = ConfigDict(populate_by_name=True)


class ActivityConfig(Config):
    """Input activity configuration."""

    firing_rate: float
