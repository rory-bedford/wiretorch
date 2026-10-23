"""Conductance-based network layer configuration.

A layer config describes a population of sources: its cell types, the synapses each
cell type makes, and (optionally) how its connectivity and weights are generated. It
turns them into the parameter lists the simulators take (``get_cell_params``,
``get_synapse_params``). A recurrent layer also has physiology; a feedforward layer's
cells are pure sources.
"""

from typing import Dict, List, Optional


from ..network import (
    cell_type_sign,
    ActivityConfig,
    CellTypesConfig,
    SimpleCellTypesConfig,
    TopologyConfig,
    WeightsConfig,
)
from .physiology import (
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
    PhysiologyConfig,
    SynapseConfig,
)
from ..base import Config


class _SynapticLayerConfig(Config):
    """What every layer has: cell types and the synapses they make."""

    cell_types: CellTypesConfig | SimpleCellTypesConfig
    synapses: Dict[str, SynapseConfig]
    topology: Optional[TopologyConfig] = None
    weights: Optional[WeightsConfig] = None

    def get_cell_params(self) -> List[Dict]:
        """One dict per cell type, with at least 'name' and 'cell_id'."""
        return [
            {"name": name, "cell_id": i} for i, name in enumerate(self.cell_types.names)
        ]

    def get_synapse_params(self) -> List[Dict[str, float | int | str]]:
        """One dict per synapse type, numbered across cell types in config order.

        Keys: 'name', 'synapse_id', 'cell_id', 'tau_rise', 'tau_decay', 'E_syn',
        'g_bar', and 'g_clip' if the cell type sets clipping.
        """
        cell_ids = {name: i for i, name in enumerate(self.cell_types.names)}
        synapse_params = []
        for cell_type_name, synapse in self.synapses.items():
            for i, synapse_name in enumerate(synapse.names):
                params = {
                    "name": synapse_name,
                    "synapse_id": len(synapse_params),
                    "cell_id": cell_ids[cell_type_name],
                    "tau_rise": synapse.tau_rise[i],
                    "tau_decay": synapse.tau_decay[i],
                    "E_syn": synapse.E_syn[i],
                    "g_bar": synapse.g_bar[i],
                }
                if synapse.g_clip is not None:
                    params["g_clip"] = synapse.g_clip[i]
                synapse_params.append(params)
        return synapse_params

    def get_synapse_names(self) -> Dict[str, List[str]]:
        """Synapse names, by cell type."""
        return {
            cell_type: synapse.names for cell_type, synapse in self.synapses.items()
        }

    def get_g_bar_by_type(self) -> Dict[str, float]:
        """Total maximal conductance of each cell type's synapses."""
        return {ct: float(sum(s.g_bar)) for ct, s in self.synapses.items()}

    def get_tau_rise_by_type(self) -> Dict[str, float]:
        """Rise time constant of each cell type's first synapse."""
        return {ct: float(s.tau_rise[0]) for ct, s in self.synapses.items()}

    def get_tau_decay_by_type(self) -> Dict[str, float]:
        """Decay time constant of each cell type's first synapse."""
        return {ct: float(s.tau_decay[0]) for ct, s in self.synapses.items()}

    def get_E_syn_by_type(self) -> Dict[str, float]:
        """Reversal potential of each cell type's first synapse."""
        return {ct: float(s.E_syn[0]) for ct, s in self.synapses.items()}

    def get_excitatory_synapse_indices(self) -> List[int]:
        """Indices (numbered as in ``get_synapse_params``) of excitatory synapses."""
        return self._synapse_indices(EXCITATORY_SYNAPSE_TYPES)

    def get_inhibitory_synapse_indices(self) -> List[int]:
        """Indices (numbered as in ``get_synapse_params``) of inhibitory synapses."""
        return self._synapse_indices(INHIBITORY_SYNAPSE_TYPES)

    def _synapse_indices(self, kinds: List[str]) -> List[int]:
        """Synapses whose name is in ``kinds``, numbered by cell type in type order."""
        names = [
            name
            for cell_type in self.cell_types.names
            if cell_type in self.synapses
            for name in self.synapses[cell_type].names
        ]
        return [i for i, name in enumerate(names) if name in kinds]


class RecurrentLayerConfig(_SynapticLayerConfig):
    """Recurrent layer configuration for conductance-based models."""

    physiology: Dict[str, PhysiologyConfig]

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """One dict per cell type: 'name', 'cell_id', 'tau_mem', 'theta', 'U_reset',
        'E_L', 'g_L', 'tau_ref'."""
        return [
            {"name": name, "cell_id": i, **self.physiology[name].model_dump()}
            for i, name in enumerate(self.cell_types.names)
        ]

    def get_neuron_params_for_plotting(self) -> Dict[int, Dict[str, float | str]]:
        """Per cell type index: 'threshold', 'rest', 'name' and 'sign' (+1 excitatory)."""
        return {
            idx: {
                "threshold": self.physiology[cell_name].theta,
                "rest": self.physiology[cell_name].U_reset,
                "name": cell_name,
                "sign": 1 if cell_type_sign(cell_name) == "excitatory" else -1,
            }
            for idx, cell_name in enumerate(self.cell_types.names)
        }


class FeedforwardLayerConfig(_SynapticLayerConfig):
    """Feedforward layer configuration for conductance-based models."""

    activity: Optional[Dict[str, ActivityConfig]] = None
