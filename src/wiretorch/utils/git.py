"""The git facts experiments record and check: commits, uncommitted files, and where
the wiretorch library is checked out."""

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

#: The wiretorch package directory.
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def git(*args, cwd=None) -> str:
    """Output of ``git args...`` (stripped); raises ``CalledProcessError`` on failure."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
    )
    return result.stdout.strip()


def uncommitted_files(cwd=None) -> list[str]:
    """Paths with uncommitted changes (``git status --porcelain``; renames: new path)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
    )
    files = []
    for line in result.stdout.splitlines():
        path = line[2:].split(" -> ")[-1].strip()  # "XY path" or "XY old -> new"
        if path:
            files.append(path)
    return files


def code_changes(files: list[str], allowed) -> list[str]:
    """The ``files`` for which ``allowed(path)`` is false (code, not configuration)."""
    return [path for path in files if not allowed(path)]


def repository_info(cwd=None) -> dict[str, str]:
    """``commit_hash``, ``branch`` and ``remote_url`` (empty without an origin)."""
    try:
        remote_url = git("config", "--get", "remote.origin.url", cwd=cwd)
    except subprocess.CalledProcessError:
        remote_url = ""
    return {
        "commit_hash": git("rev-parse", "HEAD", cwd=cwd),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        "remote_url": remote_url,
    }


def toplevel(cwd=None) -> Path:
    """Root of the git repository containing ``cwd``."""
    return Path(git("rev-parse", "--show-toplevel", cwd=cwd)).resolve()


def library_checkout() -> Path | None:
    """Root of the git checkout the library is imported from
    (``<root>/src/wiretorch``), or None if it is not imported from one (a wheel)."""
    try:
        root = toplevel(PACKAGE_DIR)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return root if (root / "src" / "wiretorch").resolve() == PACKAGE_DIR else None


def library_info() -> dict:
    """Git ``commit`` and ``dirty`` flag of the imported library, and its ``version``.

    The library is an editable dependency, so a result depends on both the experiment
    repository's commit and the library's. ``commit`` and ``dirty`` are None when the
    library is not a git checkout.
    """
    commit = dirty = None
    try:
        commit = git("-C", str(PACKAGE_DIR), "rev-parse", "HEAD")
        dirty = bool(git("-C", str(PACKAGE_DIR), "status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass  # not an editable git checkout: no commit info available
    try:
        pkg_version = version("wiretorch")
    except PackageNotFoundError:
        pkg_version = None
    return {"commit": commit, "dirty": dirty, "version": pkg_version}
