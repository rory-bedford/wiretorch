"""Parallel grid searches over an experiment's parameters, run from a code snapshot.

``run_custom_search`` writes one experiment config per parameter set and runs them in
parallel subprocesses (one per GPU, or a pool of single-threaded CPU workers). Every
job runs the code as committed when the search started, so you can keep editing:

- the experiment repository (the working directory's) is checked out at ``HEAD`` in
  a temporary git worktree, the jobs' working directory, and each job runs the
  snapshot's copy of the config's ``script``;
- if the wiretorch library is a separate git checkout, it is snapshotted too
  and put first on the jobs' ``PYTHONPATH`` (its uncommitted changes, if any, are
  not included; a warning says so).

Results go where the configs say (not into the snapshot), and the snapshot is
removed when the search ends.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import toml
from tqdm import tqdm

from .experiment_config import absolute_config
from .git import code_changes, git, library_checkout, toplevel, uncommitted_files

#: Environment variable telling a job which repositories it runs from a snapshot of:
#: ``original=snapshot`` root pairs, separated by ``os.pathsep``.
SNAPSHOT_ENV = "WIRETORCH_SNAPSHOT"

#: Environment of CPU jobs: GPUs hidden, BLAS/OpenMP on one thread per worker.
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _allowed_in_grid_search(path: str) -> bool:
    return path.endswith(".toml") or path.endswith("run_grid_search.py")


def check_git_status_for_grid_search():
    """
    Check that all code changes are committed before running a grid search.

    Allows uncommitted changes to .toml files (parameters, experiment configs) and
    run_grid_search.py, but requires all other code to be committed since the grid
    search runs from a git worktree snapshot.

    Returns:
        str: The current commit hash if validation passes

    Raises:
        SystemExit: If there are uncommitted code changes
    """
    try:
        dirty_files = uncommitted_files()
    except subprocess.CalledProcessError:
        print("ERROR: Failed to check git status")
        sys.exit(1)

    code_dirty_files = code_changes(dirty_files, _allowed_in_grid_search)
    if code_dirty_files:
        print("ERROR: You have uncommitted changes in code files:")
        for f in code_dirty_files:
            print(f"  {f}")
        print("\nGrid search runs from a git worktree snapshot, so uncommitted")
        print("changes will NOT be included. Please commit your changes first.")
        print("\nNote: Changes to .toml files are allowed.")
        sys.exit(1)
    if dirty_files:
        print("ℹ️  Uncommitted .toml files detected (allowed):")
        for f in dirty_files:
            print(f"    {f}")

    return git("rev-parse", "HEAD")


# ── Snapshot ─────────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """Worktrees of the experiment repository and (if separate) the library."""

    root: Path  # temporary directory holding the worktrees
    experiment: Path  # the experiment repository's worktree: the jobs' cwd
    repos: dict[Path, Path] = field(default_factory=dict)  # original -> worktree
    library_src: Path | None = None  # the library worktree's src/, if snapshotted

    def env(self) -> dict[str, str]:
        """What a job's environment adds: the snapshot map and ``PYTHONPATH``."""
        env = {SNAPSHOT_ENV: os.pathsep.join(f"{o}={s}" for o, s in self.repos.items())}
        if self.library_src is not None:
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(self.library_src), os.environ.get("PYTHONPATH")])
            )
        return env


@contextmanager
def snapshot(commit_hash: str, parent_dir=None):
    """Snapshot the experiment repository (and library) at ``HEAD`` while open.

    Args:
        commit_hash: The experiment repository's ``HEAD`` (names the directory).
        parent_dir: Where to create the snapshot (default: the system temp dir).
    """
    root = Path(
        tempfile.mkdtemp(
            prefix=f"grid-search-{commit_hash[:8]}-{datetime.now():%H%M%S}-",
            dir=parent_dir,
        )
    )
    experiment_repo = toplevel()
    library_repo = library_checkout()
    repos = {experiment_repo: root / experiment_repo.name}
    if library_repo is not None and library_repo != experiment_repo:
        repos[library_repo] = root / "wiretorch-library"
        if code_changes(uncommitted_files(cwd=library_repo), _allowed_in_grid_search):
            print(
                f"⚠️  wiretorch ({library_repo}) has uncommitted changes; the "
                "grid search runs its last commit."
            )

    print(f"Creating worktree snapshot at {root}...")
    created = []
    try:
        for repo, worktree in repos.items():
            git("-C", str(repo), "worktree", "add", "--detach", str(worktree), "HEAD")
            created.append((repo, worktree))
        print(f"✓ Worktree created (commit: {commit_hash[:8]})")
        print("  You can continue working on the main repo while grid search runs.\n")
        library_src = None if library_repo is None else repos[library_repo] / "src"
        yield Snapshot(root, repos[experiment_repo], repos, library_src)
    finally:
        print(f"Cleaning up worktree {root}...")
        for repo, worktree in created:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                check=False,
            )
        shutil.rmtree(root, ignore_errors=True)
        print("✓ Worktree removed")


# ── Jobs ─────────────────────────────────────────────────────────────────────


def run_on_gpu(args):
    """Run one experiment as a subprocess.

    ``args`` is ``(config_path, gpu_id, description, output_dir, cwd[, env])``.
    ``gpu_id`` is a GPU index, or ``None`` for CPU mode. In CPU mode the GPUs are
    hidden and BLAS/OpenMP are pinned to a single thread per worker, so N parallel
    workers don't oversubscribe the cores. ``env`` adds to the job's environment
    (the snapshot's, see ``Snapshot.env``).
    """
    experiment_config_path, gpu_id, description, output_dir, cwd, *rest = args
    extra_env = rest[0] if rest else {}
    if gpu_id is None:
        device_label = "CPU"
        env = {**os.environ, **CPU_ENV, **extra_env}
    else:
        device_label = f"GPU {gpu_id}"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), **extra_env}
    print(f"Running: {description} on {device_label}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wiretorch.run_experiment",
            str(experiment_config_path),
            "--no-commit",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "grid_search.log").write_text(
        f"Description: {description}\n"
        f"Device: {device_label}\n"
        f"Return Code: {result.returncode}\n"
        f"\n--- STDOUT ---\n{result.stdout}"
        f"\n--- STDERR ---\n{result.stderr}"
    )
    return {
        "config": Path(experiment_config_path).stem,
        "gpu": gpu_id,
        "description": description,
        "success": result.returncode == 0,
        "output": result.stdout if result.returncode == 0 else result.stderr,
    }


def _write_run_configs(experiment_config: dict, config_list, grid_parent: Path):
    """One parameters file and experiment config per run, and the runs' records."""
    workspace = grid_parent / "temp_grid_configs"
    workspace.mkdir(parents=True, exist_ok=True)
    config_files, runs = [], []
    for i, (params, description) in enumerate(config_list):
        params_file = workspace / f"params_{i:03d}.toml"
        params_file.write_text(toml.dumps(params))

        run_config = deepcopy(experiment_config)
        run_config["parameters_file"] = str(params_file)
        run_config["output_dir"] = str(grid_parent / description)
        if run_config.get("wandb", {}).get("enabled", False):
            run_config["wandb"]["notes"] = description
        config_file = workspace / f"experiment_{i:03d}.toml"
        config_file.write_text(toml.dumps(run_config))

        config_files.append(config_file)
        runs.append(
            {
                "run_id": i,
                "description": description,
                "output_dir": run_config["output_dir"],
                "parameters": params,
                "success": None,  # pending
            }
        )
    return config_files, runs


def _write_summary(grid_parent: Path, runs, config_path, parameters_file) -> None:
    """``README.md`` (status of every run) and ``grid_parameters.toml``."""
    succeeded = sum(run["success"] is True for run in runs)
    failed = sum(run["success"] is False for run in runs)
    status = {True: "✓", False: "✗", None: "..."}
    readme = f"""# Grid Search Run

**Timestamp**: {datetime.now():%Y-%m-%d %H:%M:%S}
**Base Experiment**: {config_path}
**Base Parameters**: {parameters_file}

## Summary
- **Total Runs**: {len(runs)}
- **Successful**: {succeeded}
- **Failed**: {failed}
- **Pending**: {len(runs) - succeeded - failed}

## Runs

"""
    readme += "".join(
        f"- {status[run['success']]} `{run['description']}/`\n" for run in runs
    )
    readme += "\n## Details\nSee `grid_parameters.toml` for complete parameter configuration of each run.\n"
    (grid_parent / "README.md").write_text(readme)
    (grid_parent / "grid_parameters.toml").write_text(toml.dumps({"runs": runs}))


def _devices(cuda_devices, max_workers, n_jobs) -> tuple[list, int]:
    """Each job's device (GPUs round-robin, or None for CPU) and the worker count."""
    if cuda_devices:
        devices = [cuda_devices[i % len(cuda_devices)] for i in range(n_jobs)]
        n_workers = len(cuda_devices)
    else:
        devices = [None] * n_jobs
        n_workers = max_workers or os.cpu_count() or 1
    return devices, max(1, min(n_workers, n_jobs))


def _run_jobs(jobs, snap: Snapshot, n_workers: int):
    """Run ``(config, device, description, output_dir)`` jobs from the snapshot in
    ``n_workers`` processes, yielding each result in job order."""
    jobs = [(*job, str(snap.experiment), snap.env()) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(run_on_gpu, jobs)
        yield from tqdm(results, total=len(jobs), desc="Experiments Progress")


def run_custom_search(
    experiment_config_path,
    config_generator,
    cuda_devices,
    grid_script_path=None,
    max_workers=None,
    snapshot_dir=None,
):
    """
    Run a custom grid search in parallel, on GPUs or CPU.

    Runs every job from a git worktree snapshot (see the module docstring), so you
    can continue working on the main repo while the grid search runs.

    Args:
        experiment_config_path: Path to experiment.toml
        config_generator: Generator function(base_params) that yields (params_dict, description)
        cuda_devices: List of GPU IDs (e.g., [0, 1]) for GPU mode. Pass ``None`` or
            an empty list for CPU mode — jobs then run with the GPUs hidden and
            BLAS/OpenMP pinned to one thread each (see ``run_on_gpu``).
        grid_script_path: Path to the grid search script (for copying to output dir)
        max_workers: Number of parallel workers in CPU mode (defaults to
            ``os.cpu_count()``). Ignored in GPU mode, where the worker count is
            ``len(cuda_devices)``.
        snapshot_dir: Where to create the snapshot (default: the system temp dir).

    Returns:
        One record per run: ``run_id``, ``description``, ``output_dir``,
        ``parameters`` and ``success``.
    """
    print("Checking git status...")
    commit_hash = check_git_status_for_grid_search()
    print(f"✓ All code changes committed (commit: {commit_hash[:8]})\n")

    with snapshot(commit_hash, snapshot_dir) as snap:
        # Absolute paths: the per-run configs are written to another directory.
        experiment_config = absolute_config(
            toml.load(experiment_config_path), Path(experiment_config_path).parent
        )
        base_params = toml.load(experiment_config["parameters_file"])

        # Each run gets a subfolder of the config's output directory.
        grid_parent = Path(experiment_config["output_dir"])
        grid_parent.mkdir(parents=True, exist_ok=True)
        if grid_script_path and Path(grid_script_path).exists():
            shutil.copy2(grid_script_path, grid_parent / "run_grid_search.py")
            print(f"✓ Copied {grid_script_path} to {grid_parent}")
        print(f"\nGrid search parent: {grid_parent}")

        config_list = list(config_generator(base_params))
        devices, n_workers = _devices(cuda_devices, max_workers, len(config_list))
        print(f"\n{'=' * 60}")
        print(f"GRID SEARCH: {len(config_list)} simulations will be run")
        if cuda_devices:
            print(f"GPUs: {cuda_devices} ({len(cuda_devices)} workers)")
        else:
            print(f"CPU: {max_workers or os.cpu_count()} workers")
        print(f"{'=' * 60}\n")

        print("Generating configurations...")
        config_files, runs = _write_run_configs(
            experiment_config, config_list, grid_parent
        )
        print(f"✓ Generated {len(config_files)} configurations\n")

        def write_summary():
            _write_summary(
                grid_parent,
                runs,
                experiment_config_path,
                experiment_config["parameters_file"],
            )

        write_summary()
        jobs = [
            (config, device, run["description"], run["output_dir"])
            for config, device, run in zip(config_files, devices, runs)
        ]
        print(
            f"Running experiments ({'GPU' if cuda_devices else 'CPU'}, "
            f"{n_workers} workers)...\n"
        )
        for run, result in zip(runs, _run_jobs(jobs, snap, n_workers)):
            run["success"] = result["success"]
            write_summary()

        succeeded = sum(run["success"] is True for run in runs)
        print(f"\nComplete: {succeeded}/{len(runs)} successful")
        print(f"Results: {grid_parent}\n")
        return runs
