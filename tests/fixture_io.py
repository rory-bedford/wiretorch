"""Reading and writing the recorded reference fixtures (``tests/fixtures/*.npz``).

Spike trains are stored bit-packed along the neuron axis; everything else as float32.
Each file carries a JSON ``meta`` entry: the commit it was recorded at, the network
fingerprint, and the recording parameters.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parents[1]


def pack_spikes(spikes) -> dict[str, np.ndarray]:
    spikes = np.asarray(spikes).astype(bool)
    return {"bits": np.packbits(spikes, axis=-1), "shape": np.array(spikes.shape)}


def unpack_spikes(bits: np.ndarray, shape: np.ndarray) -> np.ndarray:
    return np.unpackbits(bits, axis=-1, count=int(shape[-1])).astype(bool)


def library_commit() -> dict[str, str | bool]:
    """The commit of the imported library, and whether its source differs from it."""
    import wiretorch

    repo = Path(wiretorch.__file__).parents[2]  # <repo>/src/wiretorch

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "library_dirty": bool(git("status", "--porcelain", "--", "src")),
    }


def save(name: str, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    """Write ``fixtures/<name>.npz``; keys holding spikes are packed automatically."""
    out = {}
    for key, value in arrays.items():
        if key.endswith("spikes"):
            packed = pack_spikes(value)
            out[f"{key}.bits"], out[f"{key}.shape"] = packed["bits"], packed["shape"]
        else:
            out[key] = np.asarray(value)
    out["meta"] = np.array(json.dumps(meta, indent=1))
    path = FIXTURES / f"{name}.npz"
    np.savez_compressed(path, **out)
    return path


def load(name: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read ``fixtures/<name>.npz`` back into ``(arrays, meta)``, spikes unpacked."""
    with np.load(FIXTURES / f"{name}.npz") as data:
        raw = {key: data[key] for key in data.files}
    meta = json.loads(str(raw.pop("meta")))
    arrays = {}
    for key, value in raw.items():
        if key.endswith(".bits"):
            stem = key.removesuffix(".bits")
            arrays[stem] = unpack_spikes(value, raw[f"{stem}.shape"])
        elif not key.endswith(".shape"):
            arrays[key] = value
    return arrays, meta
