"""What a tracked experiment leaves in its output directory besides its own results:
the linked or copied input data, ``metadata.json`` and a ``README.md``."""

import json
import shutil
from pathlib import Path

INPUT_STRATEGIES = ("symlink", "copy")


def link_inputs(entries: list, inputs_dir: Path, resolve) -> None:
    """Link or copy each ``[[data.inputs]]`` entry into ``inputs_dir``.

    Each entry has a ``path`` (made absolute by ``resolve``), a ``strategy``
    ("symlink" or "copy") and optionally a ``name`` that replaces the source's
    basename, so scripts can load fixed, generic filenames (e.g. "connectome.npy")
    whatever the dated source file is called.
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        source, strategy = _validated_entry(i, entry)
        source_path = resolve(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input data not found: {source_path}")
        if not (source_path.is_file() or source_path.is_dir()):
            raise ValueError(f"Input path is neither file nor directory: {source_path}")

        dest_name = entry.get("name") or source_path.name
        dest_path = inputs_dir / dest_name
        if dest_path.exists() or dest_path.is_symlink():
            raise ValueError(
                f"Duplicate input destination '{dest_name}' in inputs/ "
                f"(from {source_path}); give one entry a unique 'name'."
            )

        slash = "/" if source_path.is_dir() else ""
        if strategy == "symlink":
            dest_path.symlink_to(
                source_path.absolute(), target_is_directory=source_path.is_dir()
            )
            print(f"  ✓ Linked: {source_path.name}{slash} -> {dest_name}{slash}")
        else:
            copy = shutil.copytree if source_path.is_dir() else shutil.copy2
            copy(source_path, dest_path)
            print(f"  ✓ Copied: {source_path.name}{slash} -> {dest_name}{slash}")


def _validated_entry(i: int, entry) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"Input entry {i} must be a dict with 'path' and 'strategy' keys"
        )
    source = entry.get("path")
    if not source:
        raise ValueError(f"Input entry {i} missing 'path' key")
    strategy = entry.get("strategy")
    if not strategy:
        raise ValueError(
            f"Input entry {i} missing 'strategy' key. Must be 'symlink' or 'copy'."
        )
    if strategy not in INPUT_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}' for {source}. Must be 'symlink' or 'copy'."
        )
    return source, strategy


def write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_json_line(data: dict, path: Path) -> None:
    """Append ``data`` as one JSON line (the experiments' central log)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def experiment_readme(metadata: dict, output_dir: Path) -> str:
    """The README of an experiment's output folder, from its metadata."""
    status = metadata.get("status", "unknown")
    status_emoji = "✓" if status == "completed" else "✗"
    repo_name = metadata["git_remote_url"].split("/")[-1].removesuffix(".git")
    experiment_toml = (
        Path(metadata.get("output_dir", output_dir)) / "experiment.toml"
    ).resolve()
    params_suffix = Path(metadata["parameters_file"]).suffix[1:]
    library_commit = metadata.get("wiretorch_commit") or "n/a"
    library_dirty = metadata.get("wiretorch_dirty")

    readme = f"""# Experiment Run - {status_emoji} {status.upper()}

## Description
{metadata.get("description", "No description provided.")}

## Execution Details
- **Script**: `{metadata["script"]}`
- **Start Time**: {metadata["start_time"]}
- **End Time**: {metadata.get("end_time", "N/A")}
- **Duration**: {metadata.get("duration_seconds", 0):.2f} seconds
- **Status**: {status_emoji} **{status.upper()}**
"""
    for resume in metadata.get("resumes", []):
        readme += (
            f"- **Resumed**: {resume['start_time']} (commit {resume['git_commit']})\n"
        )

    if status == "failed" and "error_file" in metadata:
        readme += f"""
## Error Information
⚠️ **This experiment failed during execution.**

Error details saved to: `{Path(metadata["error_file"]).name}`

See the error file in this directory for the full traceback.
"""

    readme += f"""
## Git Information
- **Repository**: {metadata["git_remote_url"]}
- **Commit**: {metadata["git_commit"]}
- **Branch**: {metadata["git_branch"]}
- **wiretorch library commit**: {library_commit}{" (DIRTY — uncommitted library changes)" if library_dirty else ""}

## Parameters
Parameters loaded from: `{metadata["parameters_file"]}`

See `parameters.{params_suffix}` in this directory for the exact parameters used.

## Reproducibility

To reproduce this experiment exactly:

### 1. Clone the repository
```bash
git clone {metadata["git_remote_url"]}
cd {repo_name}
```

### 2. Check out the exact commit
```bash
git checkout {metadata["git_commit"]}
```

This experiment also depends on the **wiretorch** library at commit
`{library_commit}`{" (which had uncommitted changes — see warning above)" if library_dirty else ""}; check that out in your wiretorch clone too.

### 3. Set up the Python environment
```bash
# Using uv (recommended)
uv sync --extra cpu  # or --extra gpu

# Or using pip
pip install -e .
```

### 4. Run the experiment
```bash
# Using the executable wrapper (recommended)
./run {experiment_toml}

# Or using uv directly
uv run python run_experiment.py {experiment_toml}

You will most likeley be prompted to enter a new output directory to avoid overwriting existing results.
```

**Note**: The command above uses the absolute path to the `experiment.toml` file in this output directory, which contains the exact parameters and paths for this specific run.

## Files in This Directory
- `parameters.{params_suffix}` - Exact parameters used for this run
- `experiment.toml` - Configuration file (with paths updated to this directory)
- `metadata.json` - Machine-readable metadata
- `README.md` - This file
"""
    if status == "failed":
        readme += "- `log.err` - Error traceback from failed run\n"
    return readme
