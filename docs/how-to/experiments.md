# Run reproducible experiments

wiretorch includes a small run framework for experiment repositories. An experiment is
a script with a `main` function and an `experiment.toml` describing a run:

```toml
script = "train.py"
parameters_file = "parameters.toml"
output_dir = "../results/my-experiment"
log_file = "../results/experiments_log.jsonl"
description = "What this run is for."

[[data.inputs]]
path = "../results/teacher/spike_data.zarr"
strategy = "symlink"
```

Relative paths are relative to the `experiment.toml`. The script provides

```python
def main(input_dir, output_dir, params_file, resume_from=None):
    ...
```

It can also take `wandb_config`, the config's `[wandb]` table, if it logs to Weights &
Biases. Run it with

```bash
python -m wiretorch.run_experiment path/to/experiment.toml
python -m wiretorch.run_experiment path/to/experiment.toml --resume OUTPUT_DIR
```

Before running, the framework checks that the experiment repository has no uncommitted
code (`--no-commit` skips the check while developing). It records in the output
directory the commits of the experiment repository and of wiretorch (with a warning if
wiretorch has uncommitted changes), the parameters, the inputs, the timing and the
outcome, and writes a README explaining how to reproduce the run.

`run_grid_search.py` runs a grid of parameters as separate experiments, in parallel
over GPUs, from a snapshot of the code taken when the search started.

Parameter files are TOML, validated by the pydantic classes in `wiretorch.configs`.
