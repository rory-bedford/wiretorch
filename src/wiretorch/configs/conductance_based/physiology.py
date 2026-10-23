"""Conductance-based model physiology configuration."""

import numpy as np
from typing import List, Optional
from pydantic import ConfigDict, Field
from ..base import Config


class PhysiologyConfig(Config):
    """Neuronal physiology parameters for conductance-based models."""

    tau_mem: float  # Membrane time constant (ms)
    theta: float  # Spike threshold voltage (mV)
    U_reset: float  # Reset voltage after spike (mV)
    E_L: float  # Leak reversal potential (mV)
    g_L: float  # Leak conductance (nS)
    tau_ref: float = 0.0  # Refractory period (ms): held at U_reset, no spikes


# Synapse type constants
EXCITATORY_SYNAPSE_TYPES = ["AMPA", "NMDA"]
INHIBITORY_SYNAPSE_TYPES = ["GABA_A", "GABA_B"]


class SynapseConfig(Config):
    """Synapse parameters for conductance-based models.

    Supports multiple synapse types per cell type (e.g., AMPA and NMDA for excitatory cells).
    """

    names: List[str]  # Synapse type names (e.g., ["AMPA", "NMDA"])
    tau_rise_list: List[float] = Field(alias="tau_rise")  # Rise time constants (ms)
    tau_decay_list: List[float] = Field(alias="tau_decay")  # Decay time constants (ms)
    E_syn_list: List[float] = Field(alias="E_syn")  # Reversal potentials (mV)
    g_bar_list: List[float] = Field(
        alias="g_bar"
    )  # Conductance waveform scaling factors (nS)
    g_clip_list: Optional[List[float]] = Field(default=None, alias="g_clip")

    @property
    def tau_rise(self) -> np.ndarray:
        """Rise time constants as numpy array."""
        return np.array(self.tau_rise_list)

    @property
    def tau_decay(self) -> np.ndarray:
        """Decay time constants as numpy array."""
        return np.array(self.tau_decay_list)

    @property
    def E_syn(self) -> np.ndarray:
        """Reversal potentials as numpy array."""
        return np.array(self.E_syn_list)

    @property
    def g_bar(self) -> np.ndarray:
        """Conductance waveform scaling factors as numpy array."""
        return np.array(self.g_bar_list)

    @property
    def g_clip(self) -> Optional[np.ndarray]:
        """Maximum summed conductance (nS) for saturation, or None if no clipping."""
        if self.g_clip_list is None:
            return None
        return np.array(self.g_clip_list)

    model_config = ConfigDict(populate_by_name=True)
