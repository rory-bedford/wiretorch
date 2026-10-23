"""Current-based model physiology configuration."""

from ..base import Config


class PhysiologyConfig(Config):
    """Neuronal physiology parameters for current-based models.

    Current-based models use simplified single-exponential synapses
    with tau_syn instead of detailed conductance-based synapses.
    """

    tau_mem: float  # Membrane time constant (ms)
    tau_syn: float  # Synaptic time constant (ms)
    R: float  # Membrane resistance (MOhm)
    U_rest: float  # Resting membrane potential (mV)
    theta: float  # Spike threshold voltage (mV)
    U_reset: float  # Reset voltage after spike (mV)
    tau_ref: float = 0.0  # Refractory period (ms): held at U_reset, no spikes
