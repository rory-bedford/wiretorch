"""The package's names: ``wiretorch``'s top level, and the former name ``connectome_snns``.

``wt.<name>`` loads on first use and is the object its subpackage defines; code,
configs and pickles that name ``connectome_snns`` get wiretorch's own modules.
"""

import importlib
import pickle
import subprocess
import sys

import pytest

import wiretorch as wt


@pytest.mark.parametrize("name", wt._NAMES)
def test_top_level_names_are_their_modules(name):
    module = importlib.import_module(f"wiretorch.{wt._NAMES[name]}")
    assert getattr(wt, name) is getattr(module, name)


def test_top_level_subpackages():
    for name in wt._SUBPACKAGES:
        assert getattr(wt, name) is importlib.import_module(f"wiretorch.{name}")
    assert set(wt.__all__) <= set(dir(wt))
    with pytest.raises(AttributeError):
        wt.not_a_name


def test_import_is_light():
    """``import wiretorch`` loads nothing heavy until a name is used."""
    code = "import sys, wiretorch; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_former_name_is_the_same_package():
    with pytest.warns(DeprecationWarning, match="wiretorch"):
        importlib.reload(importlib.import_module("connectome_snns"))
    import connectome_snns.network_simulators.rate as rate
    from connectome_snns.network_simulators.conductance_based import (
        ConductanceLIFNetwork,
    )
    from connectome_snns.snn_runners import SNNTrainer

    assert rate is importlib.import_module("wiretorch.network_simulators.rate")
    assert ConductanceLIFNetwork is wt.ConductanceLIFNetwork  # a former path, too
    assert SNNTrainer is wt.SNNTrainer
    assert importlib.import_module("connectome_snns").FrozenProjection is (
        wt.FrozenProjection
    )


def test_pickles_naming_the_former_package_load():
    """Checkpoints of whole models saved as connectome_snns name its modules."""
    importlib.import_module("connectome_snns")
    data = pickle.dumps(wt.ScalingFactorProjection, protocol=0)
    former = data.replace(b"wiretorch", b"connectome_snns")
    assert former != data
    assert pickle.loads(former) is wt.ScalingFactorProjection
