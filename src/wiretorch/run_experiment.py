"""
Universal experiment runner that loads and executes experiments from TOML configs.

Usage:
    python -m wiretorch.run_experiment path/to/experiment.toml [--no-commit]
    python -m wiretorch.run_experiment path/to/experiment.toml --resume OUTPUT_DIR

Use --no-commit to skip git status checks (useful for development/debugging).
Use --resume to continue training from the latest checkpoint in an existing output directory.

Examples:
    python -m wiretorch.run_experiment path/to/experiment.toml
    python -m wiretorch.run_experiment path/to/experiment.toml --no-commit
    python -m wiretorch.run_experiment path/to/experiment.toml --resume /path/to/output
"""

import argparse
import os

# Limit PyTorch threads so multiple experiments can share the machine.
# Must be set before importing torch (which experiment scripts will do).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


from wiretorch.utils.experiment_runners import run_experiment, resume_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run (or resume) the experiment an experiment.toml describes."
    )
    parser.add_argument("config", help="path to experiment.toml")
    parser.add_argument(
        "--no-commit", action="store_true", help="skip the git cleanliness check"
    )
    parser.add_argument(
        "--resume", metavar="OUTPUT_DIR", help="continue from this run's checkpoint"
    )
    args, _ = parser.parse_known_args(argv)  # other flags are ignored, as before
    if args.resume is not None:
        resume_experiment(args.config, args.resume, skip_git_check=args.no_commit)
    else:
        run_experiment(args.config, skip_git_check=args.no_commit)


if __name__ == "__main__":
    main()
