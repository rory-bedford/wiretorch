"""The former name of ``wiretorch``, kept so that existing code and checkpoints work.

``connectome_snns.X`` is ``wiretorch.X``, the same module object: imports, attribute
access and unpickling (``torch.load`` of whole models saved under the former name)
all resolve to wiretorch's classes. New code should ``import wiretorch as wt``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import wiretorch

warnings.warn(
    "connectome_snns is now wiretorch: use `import wiretorch as wt`",
    DeprecationWarning,
    stacklevel=2,
)


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports ``connectome_snns.X`` as the module ``wiretorch.X``."""

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(__name__ + "."):
            return None
        return importlib.util.spec_from_loader(name, self, is_package=True)

    def create_module(self, spec):
        # The module keeps its own name and spec: it is wiretorch's, under two names.
        return importlib.import_module("wiretorch" + spec.name.removeprefix(__name__))

    def exec_module(self, module):
        pass  # already executed, as wiretorch.X


if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())


def __getattr__(name: str):
    return getattr(wiretorch, name)
