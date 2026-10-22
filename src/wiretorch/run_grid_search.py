"""
Simple parallel grid search with GPU assignment and custom config modification.

This is a template file. Each experiment can have its own run_grid_search.py
with a custom_config_generator function.

Setup:
    1. Copy this file to your experiment directory (e.g., my-experiment/run_grid_search.py)
    2. Edit the custom_config_generator function to define your parameter sweep
    3. Edit CUDA_VISIBLE_DEVICES to match your available GPUs

Run:
    ./run --grid experiments/your-experiment/experiment.toml

The ./run script automatically uses run_grid_search.py from the experiment directory if it exists,
otherwise falls back to this template in src/.
"""

import sys
from copy import deepcopy


from wiretorch.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """
    Define your custom grid search here.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    # Example 1: repeat the experiment over seeds.
    for seed in [44, 45, 46]:
        params = deepcopy(base_params)
        params["simulation"]["seed"] = seed
        yield params, f"seed-{seed}"

    # Example 2: a parameter crossed with seeds. Keys are your parameters.toml's.
    for observed in [0.1, 0.5]:
        for seed in [44, 45]:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["observed_fraction"] = observed
            yield params, f"obs-{observed}__seed-{seed}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: Experiment config file required")
        print("Usage: python run_grid_search.py path/to/experiment.toml")
        sys.exit(1)

    config_path = sys.argv[1]
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
