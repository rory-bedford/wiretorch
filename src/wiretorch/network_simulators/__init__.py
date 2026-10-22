"""The spiking-network models.

``conductance``, ``current`` and ``rate`` each hold one model family: its physiology,
its time step (``dynamics``, plain PyTorch; ``kernels``, Triton) and its two public
networks, ``recurrent`` and ``feedforward``. What the families share is here:
``network`` (connectivity organised into pathways, and the state), ``engine`` and
``kernels`` (what their engines and kernels share), ``synapses`` (the synapse
registry). ``projections`` builds the connectivity they take; ``two_layer`` stacks two
of them.

The networks used to live one package each (``conductance_based.simulator``,
``feedforward_conductance_based.simulator``, ``current_based.simulator``,
``feedforward_current_based.simulator``); those import paths still work.
"""

import importlib
import importlib.abc
import importlib.util
import sys

_PACKAGE = __name__
#: Former module -> the module that now holds its contents.
_MOVED = {
    "conductance_based": "conductance.recurrent",
    "conductance_based.simulator": "conductance.recurrent",
    "feedforward_conductance_based": "conductance.feedforward",
    "feedforward_conductance_based.simulator": "conductance.feedforward",
    "current_based": "current.recurrent",
    "current_based.simulator": "current.recurrent",
    "feedforward_current_based": "current.feedforward",
    "feedforward_current_based.simulator": "current.feedforward",
}


class _FormerPaths(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports a former module path as a view of the module that replaced it."""

    def find_spec(self, name, path=None, target=None):
        moved = _MOVED.get(name.removeprefix(_PACKAGE + "."))
        if not name.startswith(_PACKAGE + ".") or moved is None:
            return None
        # Every former path was a package or inside one, so all can hold submodules.
        return importlib.util.spec_from_loader(name, self, is_package=True)

    def exec_module(self, module):
        target = importlib.import_module(
            f"{_PACKAGE}.{_MOVED[module.__name__.removeprefix(_PACKAGE + '.')]}"
        )
        module.__dict__.update(
            {n: v for n, v in vars(target).items() if not n.startswith("__")}
        )


if not any(isinstance(finder, _FormerPaths) for finder in sys.meta_path):
    # First, so a leftover directory of a former package cannot shadow it.
    sys.meta_path.insert(0, _FormerPaths())
