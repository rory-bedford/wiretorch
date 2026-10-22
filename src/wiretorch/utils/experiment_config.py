"""Reading an ``experiment.toml``: its paths are relative to the file's own directory."""

from pathlib import Path

import toml

#: Top-level keys of an experiment config that hold paths.
PATH_KEYS = ("script", "parameters_file", "output_dir", "log_file")


def get_repo_root() -> Path:
    """The root of the repository this library is installed from."""
    return Path(__file__).resolve().parents[3]  # <root>/src/wiretorch/utils/


def resolve_path(path_str, base_dir) -> Path:
    """``path_str`` if absolute, else resolved against ``base_dir``.

    Relative paths in an ``experiment.toml`` are relative to the file's own directory.
    """
    path = Path(path_str)
    return path if path.is_absolute() else (Path(base_dir) / path).resolve()


def resolve_relative_to_repo(path_str) -> Path:
    """``path_str`` if absolute, else resolved against the repository root."""
    return resolve_path(path_str, get_repo_root())


def absolute_config(config: dict, base_dir) -> dict:
    """A copy of an experiment config with every path made absolute.

    Paths (``script``, ``parameters_file``, ``output_dir``, ``log_file`` and each
    ``[[data.inputs]]`` path) are resolved against ``base_dir``, the directory of the
    file the config came from — so the config can be written elsewhere.
    """
    out = {**config}
    for key in PATH_KEYS:
        if key in out:
            out[key] = str(resolve_path(out[key], base_dir))
    if "data" in out:
        inputs = [
            {**entry, "path": str(resolve_path(entry["path"], base_dir))}
            for entry in out["data"].get("inputs", [])
        ]
        out["data"] = {**out["data"], "inputs": inputs}
    return out


def load_experiment_config(toml_path):
    """
    Load an experiment.toml and return its paths as a dictionary.

    Useful in analysis notebooks to avoid hardcoding paths::

        from wiretorch.utils.reproducibility import load_experiment_config
        config = load_experiment_config("experiment.toml")
        results = Path(config["output_dir"])

    Args:
        toml_path: Path to experiment.toml; relative paths inside it are resolved
            against its directory.

    Returns:
        Dict with keys: output_dir, parameters_file, script, and optionally
        data_inputs (list of input paths from [[data.inputs]]).
    """
    raw = absolute_config(toml.load(Path(toml_path)), Path(toml_path).parent)

    config = {
        "output_dir": Path(raw["output_dir"]),
        "parameters_file": Path(raw["parameters_file"]),
        "script": Path(raw["script"]),
    }

    if "data" in raw:
        config["data_inputs"] = [
            Path(entry["path"]) for entry in raw["data"].get("inputs", [])
        ]

    return config
