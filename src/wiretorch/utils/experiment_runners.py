"""Run an experiment's script under an ``ExperimentTracker``, start or resume.

The script's ``main(input_dir, output_dir, params_file[, wandb_config][,
resume_from])`` is called inside the tracker, which records git state, parameters and
outcome. Grid searches live in ``grid_search`` (re-exported here).
"""

import importlib.util
import os
import sys
from pathlib import Path

from wiretorch.utils.grid_search import (  # noqa: F401 (re-exported)
    SNAPSHOT_ENV,
    check_git_status_for_grid_search,
    run_custom_search,
    run_on_gpu,
)
from wiretorch.utils.reproducibility import (  # noqa: F401 (re-exported)
    ExperimentTracker,
    absolute_config,
    get_repo_root,
    resolve_relative_to_repo,
    uncommitted_files,
)


def run_experiment(config_path=None, skip_git_check=False):
    """
    Load and run an experiment from a TOML config file.

    Args:
        config_path: Path to experiment.toml file. Defaults to workspace/experiment.toml.
        skip_git_check: If True, skip git status validation (useful for development).
    """
    tracker = ExperimentTracker(config_path=config_path, skip_git_check=skip_git_check)
    _run_script(tracker)


def resume_experiment(config_path, resume_dir, skip_git_check=False):
    """
    Resume an experiment from the latest checkpoint in an existing output directory.

    Reuses the existing output directory and symlinked inputs, with the config's
    parameters file; the script's ``main`` gets ``resume_from=resume_dir``. The
    tracker checks git status (unless ``skip_git_check``) and records the resumed
    session in the run's metadata, README and the central log.

    Args:
        config_path: Path to experiment.toml file.
        resume_dir: Path to the existing output directory to resume from.
        skip_git_check: If True, skip git status validation.
    """
    tracker = ExperimentTracker(
        config_path=config_path, skip_git_check=skip_git_check, resume_dir=resume_dir
    )
    print(f"Resuming experiment from: {resume_dir}")
    _run_script(tracker, resume_from=Path(resume_dir))


def _run_script(tracker: ExperimentTracker, **extra_kwargs) -> None:
    """Load the config's script and call its ``main`` inside the tracker."""
    script_path = snapshot_path(tracker.resolve(tracker.config["script"]))
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)
    print(f"Loading experiment script: {script_path}")

    # Register the module so torch.compile / dynamo can resolve classes defined in
    # the script (e.g. LowRankParam).
    spec = importlib.util.spec_from_file_location("experiment_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_script"] = module

    with tracker:
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            print(f"ERROR: Script {script_path} has no main() function")
            sys.exit(1)
        kwargs = {
            "input_dir": tracker.input_dir,
            "output_dir": tracker.output_dir,
            "params_file": tracker.params_file,
            **extra_kwargs,
        }
        if tracker.wandb_config and tracker.wandb_config.get("enabled", False):
            kwargs["wandb_config"] = {
                k: v for k, v in tracker.wandb_config.items() if k != "enabled"
            }
        module.main(**kwargs)


def snapshot_path(path: Path) -> Path:
    """``path``, or its copy in a grid search's snapshot when it lies in a
    snapshotted repository (``SNAPSHOT_ENV`` holds ``original=snapshot`` roots)."""
    path = Path(path).resolve()
    for pair in filter(None, os.environ.get(SNAPSHOT_ENV, "").split(os.pathsep)):
        original, snapshot = pair.split("=", 1)
        if path.is_relative_to(original):
            return Path(snapshot) / path.relative_to(original)
    return path
