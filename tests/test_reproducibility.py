"""Experiment configs: relative paths are relative to the config file's directory."""

import json
from pathlib import Path

import pytest
import toml

from wiretorch.utils.experiment_runners import run_experiment
from wiretorch.utils.reproducibility import (
    absolute_config,
    get_repo_root,
    load_experiment_config,
)

SCRIPT = """
from pathlib import Path

def main(input_dir, output_dir, params_file):
    (Path(output_dir) / "ran.json").write_text(Path(params_file).read_text())
"""


def experiment(tmp_path):
    folder = tmp_path / "project" / "my-experiment"
    folder.mkdir(parents=True)
    (folder / "train.py").write_text(SCRIPT)
    (folder / "parameters.toml").write_text("x = 1\n")
    (tmp_path / "data.npz").write_text("data")
    config = {
        "script": "train.py",
        "parameters_file": "parameters.toml",
        "output_dir": "../results/run",
        "log_file": "../results/log.jsonl",
        "description": "test",
        "data": {"inputs": [{"path": "../../data.npz", "strategy": "symlink"}]},
    }
    path = folder / "experiment.toml"
    path.write_text(toml.dumps(config))
    return path


def test_relative_paths_resolve_against_the_config(tmp_path, monkeypatch):
    path = experiment(tmp_path)
    monkeypatch.chdir(tmp_path)  # anywhere but the config's directory
    run_experiment(path, skip_git_check=True)

    results = tmp_path / "project" / "results"
    assert (results / "run" / "ran.json").read_text() == "x = 1\n"
    assert (results / "run" / "inputs" / "data.npz").resolve() == tmp_path / "data.npz"
    assert json.loads((results / "log.jsonl").read_text().splitlines()[-1])


def test_load_and_absolute_config(tmp_path):
    path = experiment(tmp_path)
    config = load_experiment_config(path)
    assert config["script"] == path.parent / "train.py"
    assert config["data_inputs"] == [tmp_path / "data.npz"]
    absolute = absolute_config(toml.load(path), path.parent)
    assert absolute["output_dir"] == str(tmp_path / "project" / "results" / "run")


def test_repo_root():
    assert (get_repo_root() / "pyproject.toml").exists()


def test_uncommitted_files_parses_renames(tmp_path):
    import subprocess

    from wiretorch.utils.reproducibility import uncommitted_files

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-qm", "a")
    git("mv", "a.py", "b.py")
    (tmp_path / "c.toml").write_text("")
    assert sorted(uncommitted_files(cwd=tmp_path)) == ["b.py", "c.toml"]


def _git(cwd, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _commit_all(repo):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "snapshot me")


WHERE_SCRIPT = """
import json
from pathlib import Path
from pathlib import Path

import wiretorch

def main(input_dir, output_dir, params_file):
    (Path(output_dir) / "where.json").write_text(json.dumps({
        "script": __file__, "library": wiretorch.__file__,
    }))
"""


def test_grid_search_jobs_run_the_snapshot(tmp_path, monkeypatch):
    """Each job runs the snapshot's script and imports the snapshot's library, even
    when both checkouts change after the search started; results land where the
    config says; the snapshot is gone afterwards."""
    import shutil

    from wiretorch.utils import grid_search
    from wiretorch.utils.experiment_config import get_repo_root

    library = tmp_path / "library"
    shutil.copytree(
        get_repo_root() / "src" / "wiretorch",
        library / "src" / "wiretorch",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    _commit_all(library)

    project = tmp_path / "project"
    (project / "exp").mkdir(parents=True)
    (project / "exp" / "train.py").write_text(WHERE_SCRIPT)
    (project / "exp" / "parameters.toml").write_text("x = 1\n")
    config = {
        "script": "train.py",
        "parameters_file": "parameters.toml",
        "output_dir": str(tmp_path / "results"),
        "log_file": str(tmp_path / "results" / "log.jsonl"),
    }
    (project / "exp" / "experiment.toml").write_text(toml.dumps(config))
    _commit_all(project)

    def edit_checkouts_then_yield(base):
        # The jobs start after these edits; they must not see them.
        (project / "exp" / "train.py").write_text("raise RuntimeError('live')\n")
        (library / "src" / "wiretorch" / "run_experiment.py").write_text(
            "raise RuntimeError('live library')\n"
        )
        yield {**base, "x": 2}, "two"

    monkeypatch.chdir(project)
    monkeypatch.setattr(grid_search, "library_checkout", lambda: library)
    monkeypatch.setattr(
        grid_search, "check_git_status_for_grid_search", lambda: "f" * 40
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    runs = grid_search.run_custom_search(
        project / "exp" / "experiment.toml",
        edit_checkouts_then_yield,
        cuda_devices=None,
        max_workers=1,
        snapshot_dir=snapshots,
    )

    log = (tmp_path / "results" / "two" / "grid_search.log").read_text()
    assert runs[0]["success"], log
    where = json.loads((tmp_path / "results" / "two" / "where.json").read_text())
    (snapshot,) = {Path(where["script"]).relative_to(snapshots).parts[0]}
    root = snapshots / snapshot
    assert Path(where["script"]) == root / "project" / "exp" / "train.py"
    assert Path(where["library"]) == (
        root / "wiretorch-library" / "src" / "wiretorch" / "__init__.py"
    )
    parameters = tmp_path / "results" / "two" / "parameters.toml"
    assert toml.loads(parameters.read_text()) == {"x": 2}
    copied = toml.load(tmp_path / "results" / "two" / "experiment.toml")
    assert copied["script"] == str(project / "exp" / "train.py")  # re-runnable
    assert list(snapshots.iterdir()) == []


def test_resume_is_tracked(tmp_path, monkeypatch):
    """Resuming checks nothing with --no-commit but records the session."""
    path = experiment(tmp_path)
    (path.parent / "train.py").write_text(
        SCRIPT.replace("params_file):", "params_file, resume_from=None):")
    )
    run_experiment(path, skip_git_check=True)

    from wiretorch.utils.experiment_runners import resume_experiment

    run_dir = tmp_path / "project" / "results" / "run"
    resume_experiment(path, run_dir, skip_git_check=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert len(metadata["resumes"]) == 1
    log = (tmp_path / "project" / "results" / "log.jsonl").read_text().splitlines()
    assert len(log) == 2 and "resumes" in json.loads(log[-1])
    assert "Resumed" in (run_dir / "README.md").read_text()


def test_resume_checks_git_status(tmp_path, monkeypatch):
    path = experiment(tmp_path)
    run_experiment(path, skip_git_check=True)
    from wiretorch.utils import reproducibility
    from wiretorch.utils.experiment_runners import resume_experiment

    monkeypatch.setattr(reproducibility, "uncommitted_files", lambda: ["model.py"])
    with pytest.raises(SystemExit):
        resume_experiment(path, tmp_path / "project" / "results" / "run")


def test_run_experiment_cli(tmp_path, monkeypatch):
    from wiretorch import run_experiment as cli

    calls = []
    monkeypatch.setattr(
        cli, "run_experiment", lambda c, **k: calls.append(("run", c, k))
    )
    monkeypatch.setattr(
        cli, "resume_experiment", lambda c, r, **k: calls.append(("resume", c, r, k))
    )
    cli.main(["e.toml", "--no-commit"])
    cli.main(["e.toml", "--resume", "out", "--unknown"])
    assert calls == [
        ("run", "e.toml", {"skip_git_check": True}),
        ("resume", "e.toml", "out", {"skip_git_check": False}),
    ]
