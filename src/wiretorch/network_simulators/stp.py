"""Short-term plasticity: depression and facilitation of release (Tsodyks–Markram).

Release is presynaptic, and its dynamics are set per connection type: the same axon
makes depressing synapses onto some targets and facilitating ones onto others (Markram,
Wang & Tsodyks 1998; Reyes et al. 1998). So the spiking networks take STP per
``(source cell type, target cell type)`` pair, as they take projections, and it scales
every synapse type that connection makes.

Each source neuron keeps, per pair with STP, its available resources ``x`` (at rest 1)
and its release probability ``u`` (at rest ``U``). Between spikes ``x`` recovers to 1
with ``tau_D`` and ``u`` relaxes to ``U`` with ``tau_F``. A spike releases ``u x``,
which leaves ``x``; then ``u`` jumps by ``U (1 - u)``. Its input is the release over
``U``, so a first spike from rest has exactly its weight: depression weakens the spikes
that follow, facilitation strengthens them. This is the model of Markram, Wang &
Tsodyks (1998), event by event.

Typical (rat neocortex): pyramidal → pyramidal and → PV (fast-spiking) interneurons
depress (``DEPRESSING``); pyramidal → SST (Martinotti) interneurons facilitate
(``FACILITATING``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class STP:
    """Short-term plasticity of one connection type.

    Attributes:
        U: Release probability at rest, in (0, 1]: high depresses, low facilitates.
        tau_D: Recovery time constant of the resources (ms), > 0.
        tau_F: Facilitation time constant (ms); 0 is pure depression.
    """

    U: float
    tau_D: float
    tau_F: float = 0.0

    def __post_init__(self):
        if not 0 < self.U <= 1:
            raise ValueError(f"STP: U must be in (0, 1], got {self.U}")
        if not self.tau_D > 0:
            raise ValueError(f"STP: tau_D must be > 0 ms, got {self.tau_D}")
        if not self.tau_F >= 0:
            raise ValueError(f"STP: tau_F must be >= 0 ms, got {self.tau_F}")

    def decays(self, dt: float) -> tuple[float, float]:
        """Per-step decay of the resources' deficit and of the facilitation."""
        recovery = math.exp(-dt / self.tau_D)
        return recovery, math.exp(-dt / self.tau_F) if self.tau_F > 0 else 0.0


#: Depressing synapses (Tsodyks & Markram 1997, pyramidal → pyramidal).
DEPRESSING = STP(U=0.5, tau_D=800.0, tau_F=0.0)
#: Facilitating synapses (Tsodyks, Pawelzik & Markram 1998, pyramidal → interneuron).
FACILITATING = STP(U=0.03, tau_D=130.0, tau_F=530.0)


def as_stp(value: STP | dict) -> STP:
    """An ``STP``, from one or from a dict of its fields."""
    return value if isinstance(value, STP) else STP(**value)
