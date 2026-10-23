"""The package's top-level names: ``wt.<name>`` loads on first use and is the object
its subpackage defines.
"""

import importlib
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
