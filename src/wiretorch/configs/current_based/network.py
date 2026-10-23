"""Current-based network layer configuration.

``get_cell_params`` and ``get_synapse_params`` give the arguments of the current-based
networks (``network_simulators.current``): each cell type's physiology, and one
synapse type per source cell type (its ``tau_syn``, its sign read from its name).
"""

from typing import Dict, List, Optional

from ..network import (
    cell_type_sign,
    SimpleCellTypesConfig,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    ActivityConfig,
)
from .physiology import PhysiologyConfig
from ..base import Config


class RecurrentLayerConfig(Config):
    """Recurrent layer configuration for current-based models."""

    cell_types: CellTypesConfig | SimpleCellTypesConfig
    physiology: Dict[str, PhysiologyConfig]
    topology: Optional[TopologyConfig] = None
    weights: Optional[WeightsConfig] = None

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_mem', 'tau_syn',
            'R', 'U_rest', 'theta', 'U_reset', 'tau_ref'.
        """
        cell_params = []
        for i, cell_type_name in enumerate(self.cell_types.names):
            physiology = self.physiology[cell_type_name]
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": i,
                    "tau_mem": physiology.tau_mem,
                    "tau_syn": physiology.tau_syn,
                    "R": physiology.R,
                    "U_rest": physiology.U_rest,
                    "theta": physiology.theta,
                    "U_reset": physiology.U_reset,
                    "tau_ref": physiology.tau_ref,
                }
            )
        return cell_params

    def get_synapse_params(self) -> List[Dict[str, float | int | str]]:
        """One synapse type per cell type, for ``synapse_params`` of the networks."""
        return _synapse_params(self.cell_types.names, self.physiology)

    def get_neuron_params_for_plotting(self) -> Dict[int, Dict[str, float | str]]:
        """Get neuron parameters formatted for visualization functions.

        Returns:
            Dict mapping cell type indices to parameter dicts with keys:
            'threshold', 'rest', 'name', 'sign'.
        """
        return {
            idx: {
                "threshold": self.physiology[cell_name].theta,
                "rest": self.physiology[cell_name].U_reset,
                "name": cell_name,
                "sign": 1 if cell_type_sign(cell_name) == "excitatory" else -1,
            }
            for idx, cell_name in enumerate(self.cell_types.names)
        }


class FeedforwardLayerConfig(Config):
    """Feedforward layer configuration for current-based models."""

    cell_types: CellTypesConfig | SimpleCellTypesConfig
    physiology: Dict[str, PhysiologyConfig]
    topology: Optional[TopologyConfig] = None
    weights: Optional[WeightsConfig] = None
    activity: Optional[Dict[str, ActivityConfig]] = None

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_syn'.
        """
        cell_params = []
        for i, cell_type_name in enumerate(self.cell_types.names):
            physiology = self.physiology[cell_type_name]
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": i,
                    "tau_syn": physiology.tau_syn,
                }
            )
        return cell_params

    def get_synapse_params(self) -> List[Dict[str, float | int | str]]:
        """One synapse type per input cell type, for ``synapse_params_FF``."""
        return _synapse_params(self.cell_types.names, self.physiology)


def _synapse_params(names, physiology) -> List[Dict[str, float | int | str]]:
    """Synapse type ``i`` made by cell type ``i``: its ``tau_syn``, and inhibitory only
    if its name says so."""
    return [
        {
            "synapse_id": i,
            "cell_id": i,
            "name": name,
            "tau_syn": physiology[name].tau_syn,
            "type": "inhibitory"
            if cell_type_sign(name) == "inhibitory"
            else "excitatory",
        }
        for i, name in enumerate(names)
    ]
