"""Re-execute the tutorial notebooks in place, so their committed outputs are current.

    uv run python docs/run_tutorials.py                 # every tutorial
    uv run python docs/run_tutorials.py ring_attractor  # just the ones named

The notebooks are synthetic examples: they read no data, and each runs in seconds on a
GPU, up to two minutes on a CPU. Commit them executed: the docs build does not run
them.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TUTORIALS = Path(__file__).parent / "tutorials"


def main(names):
    paths = sorted(TUTORIALS.glob("*.ipynb"))
    if names:
        paths = [p for p in paths if p.stem in names]
    for path in paths:
        start = time.perf_counter()
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook, timeout=1200, kernel_name="python3",
            resources={"metadata": {"path": str(TUTORIALS)}},
        ).execute()  # fmt: skip
        nbformat.write(notebook, path)
        print(f"{path.name}: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
