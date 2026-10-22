"""
Reproducibility module for running experiments with tracked parameters and metadata.

Usage in your scripts:
    from wiretorch.utils.reproducibility import ExperimentTracker

    tracker = ExperimentTracker()

    with tracker:
        output_dir, params_file = tracker.output_dir, tracker.params_file

        # Your code here...
        # Errors are automatically caught and logged

The pieces live in their own modules and are importable from here too: reading
experiment configs (``experiment_config``), git facts (``git``), and the inputs,
metadata and README written to the output directory (``experiment_outputs``).
"""

import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import toml

from .experiment_config import (  # noqa: F401 (re-exported)
    absolute_config,
    get_repo_root,
    load_experiment_config,
    resolve_path,
    resolve_relative_to_repo,
)
from .experiment_outputs import (
    append_json_line,
    experiment_readme,
    link_inputs,
    write_json,
)
from .git import (  # noqa: F401 (re-exported)
    code_changes,
    library_info,
    repository_info,
    uncommitted_files,
)

REQUIRED_FIELDS = ("script", "output_dir", "parameters_file", "log_file")


def _is_configuration(path: str) -> bool:
    """Files an experiment may run with uncommitted: parameters, workspace, TOML."""
    return (
        path.startswith("parameters/")
        or path.startswith("workspace/")
        or path.endswith(".toml")
    )


def _banner(title: str, *lines: str) -> None:
    rule = "=" * len(title)
    print("\n".join(["", rule, title, *lines, rule, ""]))


class ExperimentTracker:
    """
    Tracks experiment reproducibility by checking git status, logging metadata,
    and managing output directories.

    With ``resume_dir``, the run continues in that existing output directory: its
    inputs and files are kept, and ``metadata.json``, the README and the central log
    record the resumed session (under ``resumes``).
    """

    def __init__(self, config_path=None, skip_git_check=False, resume_dir=None):
        """
        Initialize the experiment tracker.

        Args:
            config_path: Path to the TOML configuration file.
                        Defaults to workspace/experiment.toml in the repository root.
            skip_git_check: If True, skip git status validation (useful for development).
            resume_dir: An existing output directory to continue in (see the class
                docstring). None starts a new run.
        """
        if config_path is None:
            # Default to workspace/experiment.toml in repository root
            config_path = get_repo_root() / "workspace" / "experiment.toml"
        self.config_path = Path(config_path)
        self.skip_git_check = skip_git_check
        self.resume_dir = None if resume_dir is None else Path(resume_dir)
        self.config = self._load_config()
        self.start_time = None
        self.git_info = None
        self.calling_script = None
        self.output_dir = None
        self.completed_successfully = False
        self.initial_metadata = None
        self.params_file = None
        self.wandb_config = None
        self.input_dir = None  # Will be populated if data section exists

    def __enter__(self):
        """Context manager entry - start tracking."""
        self._start_tracking()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit - handle success or failure."""
        if exc_type is None:
            self._end_tracking_success()
        else:
            self._end_tracking_failure(exc_type, exc_value, exc_tb)
            return False  # re-raise the exception

    def resolve(self, path_str) -> Path:
        """A path from the config: absolute, or relative to the config's directory."""
        return resolve_path(path_str, self.config_path.resolve().parent)

    # ── Checks ───────────────────────────────────────────────────────────────

    def _load_config(self):
        """Load experiment configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        return toml.load(self.config_path)

    def _validate_config(self):
        """Required fields are present; the script and parameters file exist."""
        for field in REQUIRED_FIELDS:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in config")
        script_path = self.resolve(self.config["script"])
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        params_path = self.resolve(self.config["parameters_file"])
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

    def _check_git_status(self):
        """The repository's commit, branch and remote; exits if code is uncommitted.

        Uncommitted parameters/*, workspace/* and .toml files are allowed; with
        ``skip_git_check`` nothing is checked.
        """
        try:
            if self.skip_git_check:
                print("⚠️  Skipping git status validation (--no-commit flag used)")
            else:
                self._require_committed_code()
            return repository_info()
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Git command failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("ERROR: Git not found. Is this a git repository?")
            sys.exit(1)

    def _require_committed_code(self) -> None:
        dirty_files = uncommitted_files()
        code_dirty_files = code_changes(dirty_files, _is_configuration)
        if code_dirty_files:
            print("ERROR: You have uncommitted changes in code files:")
            for f in code_dirty_files:
                print(f"  {f}")
            print("\nPlease commit code changes before running experiments.")
            print(
                "Note: Changes to .toml files, parameters/*, and workspace/* are allowed."
            )
            print("Or use --no-commit flag to skip this check entirely.")
            sys.exit(1)
        if dirty_files:
            print("ℹ️  Uncommitted config files detected (allowed):")
            for f in dirty_files:
                print(f"  {f}")

    def _get_library_info(self):
        """Git commit (+ dirty flag) and version of the imported wiretorch.

        A dirty library is a reproducibility hole, so it is flagged (but does not
        block — the library is co-developed).
        """
        lib = library_info()
        if lib["dirty"] and not self.skip_git_check:
            print(
                "⚠️  wiretorch library has uncommitted changes — recording "
                "wiretorch_dirty=true (result not fully reproducible)."
            )
        return lib

    def _extract_wandb_config(self):
        """Extract wandb configuration from config if present."""
        return self.config.get("wandb")

    # ── Output directory ─────────────────────────────────────────────────────

    def _choose_output_dir(self) -> Path:
        """The configured output directory, asking what to do if it exists."""
        output_dir = self.resolve(self.config["output_dir"])
        while output_dir.exists():
            print(f"\n⚠️  WARNING: Output directory already exists: {output_dir}")
            print("\nChoose an option:")
            print("  [c] Cancel experiment")
            print("  [o] Overwrite existing folder")
            print("  [n] Specify a new output folder")
            response = input("\nYour choice (c/o/n): ").lower().strip()

            if response == "c":
                print("Experiment cancelled.")
                sys.exit(0)
            elif response == "o":
                print(f"⚠️  Overwriting: {output_dir}")
                shutil.rmtree(output_dir)
                break
            elif response == "n":
                new_path = input("Enter new output folder path: ").strip()
                if new_path:
                    output_dir = Path(new_path)
                    self.config["output_dir"] = str(output_dir)
                    print(f"Updated output directory to: {output_dir}")
                else:
                    print("Invalid path. Please try again.")
            else:
                print("Invalid choice. Please enter 'c', 'o', or 'n'.")
        return output_dir

    def _setup_input_data(self, output_dir):
        """Link or copy the config's ``[[data.inputs]]`` into ``output_dir/inputs``.

        Returns the inputs directory, or None if the config lists no inputs.
        """
        entries = self.config.get("data", {}).get("inputs", [])
        if not entries:
            return None
        inputs_dir = output_dir / "inputs"
        print("\nSetting up input data...")
        link_inputs(entries, inputs_dir, self.resolve)
        print(f"✓ Input data ready: {inputs_dir}")
        return inputs_dir

    def _prepare_new_output(self) -> None:
        """Create the output directory with the parameters, inputs and config."""
        output_dir = self._choose_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")

        # The run reads its own copy of the parameters, so concurrent runs sharing a
        # source parameters.toml don't race on whatever it says when main() loads it.
        params_path = self.resolve(self.config["parameters_file"])
        params_filename = f"parameters{params_path.suffix}"
        self.params_file = output_dir / params_filename
        shutil.copy2(params_path, self.params_file)
        print(f"✓ Copied parameters file to: {self.params_file}")

        self.input_dir = self._setup_input_data(output_dir)

        # The config, with paths pointing at the copies in the output directory
        copied_config = self.config.copy()
        copied_config["parameters_file"] = str(self.params_file.resolve())
        copied_config["output_dir"] = str(output_dir.resolve())
        copied_config["log_file"] = str(self.resolve(self.config["log_file"]))
        if "data" in copied_config and self.input_dir:
            copied_config["data"]["inputs"] = [str(self.input_dir.resolve())]
        dest_toml = output_dir / "experiment.toml"
        with open(dest_toml, "w") as f:
            toml.dump(copied_config, f)
        print(f"✓ Copied experiment config to: {dest_toml}")
        self.output_dir = output_dir

    def _prepare_resumed_output(self) -> None:
        """Continue in ``resume_dir``: its inputs, with the config's parameters."""
        if not self.resume_dir.exists():
            print(f"ERROR: Resume directory not found: {self.resume_dir}")
            sys.exit(1)
        self.output_dir = self.resume_dir
        self.params_file = self.resolve(self.config["parameters_file"])
        input_dir = self.resume_dir / "inputs"
        if self.config.get("data", {}).get("inputs") and not input_dir.exists():
            print(f"ERROR: No inputs directory found in {self.resume_dir}")
            sys.exit(1)
        self.input_dir = input_dir if input_dir.exists() else None
        print(f"✓ Resuming in: {self.output_dir}")

    # ── Metadata ─────────────────────────────────────────────────────────────

    def _create_initial_metadata(self):
        """Create initial metadata at experiment start."""
        lib = self._get_library_info()
        return {
            "script": self.config["script"],
            "description": self.config.get("description", ""),
            "parameters_file": self.config["parameters_file"],
            "output_dir": str(self.config["output_dir"]),
            "git_commit": self.git_info["commit_hash"],
            "git_branch": self.git_info["branch"],
            "git_remote_url": self.git_info["remote_url"],
            "wiretorch_commit": lib["commit"],
            "wiretorch_dirty": lib["dirty"],
            "wiretorch_version": lib["version"],
            "start_time": self.start_time.isoformat(),
            "status": "running",
        }

    def _resumed_metadata(self) -> dict:
        """The run's metadata with this session appended to ``resumes``."""
        session = self._create_initial_metadata()
        path = self.output_dir / "metadata.json"
        metadata = json.loads(path.read_text()) if path.exists() else dict(session)
        resume = {
            key: session[key]
            for key in (
                "start_time",
                "parameters_file",
                "git_commit",
                "wiretorch_commit",
                "wiretorch_dirty",
            )
        }
        metadata["resumes"] = [*metadata.get("resumes", []), resume]
        metadata["status"] = "running"
        return metadata

    def _finalize_metadata(self, end_time, success=True):
        """Finalize metadata with end time and status (of this session)."""
        metadata = self.initial_metadata.copy()
        metadata["end_time"] = end_time.isoformat()
        metadata["duration_seconds"] = (end_time - self.start_time).total_seconds()
        metadata["status"] = "completed" if success else "failed"
        return metadata

    def _log_metadata(self, metadata, log_file):
        """Append metadata to log file."""
        append_json_line(metadata, log_file)
        print(f"✓ Logged metadata to: {log_file}")

    def _create_readme(self, metadata):
        """Generate a README for the experiment output folder."""
        return experiment_readme(metadata, self.output_dir)

    def _write_results_record(self, metadata) -> None:
        """``metadata.json`` and ``README.md`` in the output directory."""
        write_json(metadata, self.output_dir / "metadata.json")
        (self.output_dir / "README.md").write_text(self._create_readme(metadata))

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _start_tracking(self):
        """Start tracking the experiment (called by context manager)."""
        self._validate_config()
        self.wandb_config = self._extract_wandb_config()

        _banner(
            "Resuming Experiment Tracking"
            if self.resume_dir
            else "Starting Experiment Tracking"
        )
        print(f"Script: {self.config['script']}")
        print(f"Parameters: {self.config['parameters_file']}")
        print(f"Output: {self.resume_dir or self.config['output_dir']}")
        print(f"Description: {self.config.get('description', 'N/A')}")
        if self.wandb_config and self.wandb_config.get("enabled", False):
            print(f"W&B Project: {self.wandb_config.get('project', 'N/A')}")

        print(
            "\nChecking git status"
            + (" (--no-commit mode)..." if self.skip_git_check else "...")
        )
        self.git_info = self._check_git_status()
        commit = self.git_info["commit_hash"][:8]
        if self.skip_git_check:
            print(f"⚠️  Git check skipped (commit: {commit})")
        else:
            print(f"✓ Git status clean (commit: {commit})")

        if self.resume_dir:
            self._prepare_resumed_output()
        else:
            self._prepare_new_output()

        self.start_time = datetime.now()
        self.initial_metadata = (
            self._resumed_metadata()
            if self.resume_dir
            else self._create_initial_metadata()
        )
        write_json(self.initial_metadata, self.output_dir / "metadata.json")
        print(f"✓ Saved initial metadata to: {self.output_dir / 'metadata.json'}")

        _banner("Experiment tracking initialized. Running your code now.")

    def _end_tracking_success(self):
        """End tracking for successful experiment completion."""
        _banner("Finalizing Experiment Tracking")
        metadata = self._finalize_metadata(datetime.now(), success=True)
        self._write_results_record(metadata)
        print(f"✓ Updated metadata: {self.output_dir / 'metadata.json'}")
        print(f"✓ Generated README: {self.output_dir / 'README.md'}")
        self._log_metadata(metadata, self.resolve(self.config["log_file"]))

        title = f"Experiment complete! Results saved to: {self.output_dir}"
        _banner(title, f"Duration: {metadata['duration_seconds']:.2f} seconds")

    def _end_tracking_failure(self, exc_type, exc_value, exc_tb):
        """Handle experiment failure by logging error and saving partial results."""
        error_file = self.output_dir / "log.err"
        error_file.write_text(
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        print(f"\n✗ Error trace saved to: {error_file}")

        metadata = self._finalize_metadata(datetime.now(), success=False)
        metadata["error_file"] = str(error_file)
        self._write_results_record(metadata)

        log_file = self.resolve(self.config["log_file"])
        append_json_line(metadata, log_file)
        print(
            f"✗ Experiment failed. Partial results and error log saved to: {self.output_dir}"
        )
        print(f"✗ Failure logged to: {log_file}\n")
