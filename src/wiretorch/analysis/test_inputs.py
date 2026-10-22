"""Generate (or load cached) test teacher activity for an analysis notebook.

The experiment's output directory already contains a snapshot of the teacher
parameters (in the ``teacher-activity`` dir that its ``inputs/`` symlinks into)
and the saved ``network_structure.npz``. This helper runs only the spike
generation half of the teacher pipeline: it loads the saved network as-is,
re-seeds with ``teacher_seed + SEED_BUMP``, forces ``batch_size=1``, and runs
for one training trial's duration. Result is written as a zarr cache so
subsequent calls are cheap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import toml
import torch
import zarr

from wiretorch.configs import SimulationConfig
from wiretorch.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from wiretorch.configs.input_patterns import load_input_patterns
from wiretorch.dataloaders.input_patterns import generate_assembly_firing_rates
from wiretorch.dataloaders.rate_processes import OrnsteinUhlenbeckRateProcess
from wiretorch.dataloaders.unsupervised import InhomogeneousPoissonSpikeDataLoader
from wiretorch.network_simulators.conductance.recurrent import (
    ConductanceLIFNetwork,
)
from wiretorch.network_simulators.projections import make_frozen_projections
from wiretorch.snn_runners import SNNInference


SEED_BUMP = 10
CACHE_FILENAME = "test_inputs.zarr"


def _resolve_teacher_dir(run_dir: Path) -> Path:
    """Walk the spike_data.zarr symlink in ``run_dir/inputs/`` to the teacher dir."""
    spike_link = run_dir / "inputs" / "spike_data.zarr"
    resolved = spike_link.resolve()
    # .../teacher-activity/results/spike_data.zarr  →  .../teacher-activity/
    return resolved.parents[1]


def _config_hash(teacher_params: dict, test_seed: int) -> str:
    payload = json.dumps({"params": teacher_params, "seed": test_seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def ensure_test_inputs(
    run_dir: str | Path,
    *,
    device: str = "cpu",
    regenerate: bool = False,
) -> Path:
    """Return path to a cached test-inputs zarr; regenerate if missing/stale.

    Args:
        run_dir: Experiment output directory (the trained student's run dir).
        device: Torch device for spike generation.
        regenerate: Force regeneration even if the cache's config hash matches.

    Returns:
        Path to ``run_dir / test_inputs.zarr`` containing ``input_spikes`` and
        ``output_spikes`` (shape ``(1, total_timesteps, n_neurons)``), with
        ``dt``, ``config_hash`` and ``test_seed`` stored in ``attrs``.
    """
    run_dir = Path(run_dir)
    cache_path = run_dir / CACHE_FILENAME

    teacher_dir = _resolve_teacher_dir(run_dir)
    teacher_params = toml.load(teacher_dir / "parameters.toml")
    test_seed = int(teacher_params["simulation"]["seed"]) + SEED_BUMP
    cfg_hash = _config_hash(teacher_params, test_seed)

    if cache_path.exists() and not regenerate:
        try:
            existing = zarr.open_group(str(cache_path), mode="r")
            if existing.attrs.get("config_hash") == cfg_hash:
                return cache_path
            print("[test_inputs] cache hash mismatch — regenerating")
        except Exception as e:
            print(f"[test_inputs] could not read cache ({e}) — regenerating")

    print(f"[test_inputs] generating (seed={test_seed}, single trial)")
    _generate(
        cache_path=cache_path,
        teacher_params=teacher_params,
        teacher_dir=teacher_dir,
        test_seed=test_seed,
        cfg_hash=cfg_hash,
        device=device,
    )
    return cache_path


def _generate(
    *,
    cache_path: Path,
    teacher_params: dict,
    teacher_dir: Path,
    test_seed: int,
    cfg_hash: str,
    device: str,
) -> None:
    """Run teacher simulation with batch_size=1 and write ``cache_path``."""
    # Seed first so any downstream RNG matches the recorded test_seed.
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    sim = SimulationConfig(**teacher_params["simulation"])
    rec_cfg = RecurrentLayerConfig(**teacher_params["recurrent"])
    ff_cfg = FeedforwardLayerConfig(**teacher_params["feedforward"])

    ou_params, pattern_configs = load_input_patterns(teacher_params)

    # Saved teacher network — do NOT regenerate.
    ns = np.load(teacher_dir / "results" / "network_structure.npz")
    cell_type_indices = ns["cell_type_indices"]
    input_source_indices = ns["feedforward_cell_type_indices"]
    feedforward_weights = ns["feedforward_weights"]
    weights = ns["recurrent_weights"]
    assembly_ids = ns["assembly_ids"]

    # One input rate pattern per assembly (derived from the saved weights).
    input_patterns = generate_assembly_firing_rates(
        feedforward_weights=feedforward_weights,
        input_source_indices=input_source_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=ff_cfg.cell_types.names,
        input_configs={name: cfg.to_dict() for name, cfg in pattern_configs.items()},
    )

    batch_size = 1  # single test trial
    chunk_size = sim.chunk_size
    num_chunks = sim.num_chunks
    dt = sim.dt

    rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_patterns,
        chunk_size=int(chunk_size),
        dt=dt,
        **ou_params,
        a_init=None,
        return_rates=True,
        batch_size=batch_size,
        device=device,
    )
    spike_dataloader = InhomogeneousPoissonSpikeDataLoader(
        rate_process=rate_process,
        batch_size=batch_size,
        device=device,
        return_rates=True,
    )

    rec_projections, ff_projections = make_frozen_projections(
        rec_weights=weights,
        ff_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=input_source_indices,
        cell_type_names=list(rec_cfg.cell_types.names),
        ff_cell_type_names=list(ff_cfg.cell_types.names),
    )
    model = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projections,
        ff_projections=ff_projections,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=input_source_indices,
        cell_params=rec_cfg.get_cell_params(),
        cell_params_FF=ff_cfg.get_cell_params(),
        synapse_params=rec_cfg.get_synapse_params(),
        synapse_params_FF=ff_cfg.get_synapse_params(),
        surrgrad_scale=1.0,
        batch_size=batch_size,
        track_variables=False,
        track_batch_idx=0,
    )
    model.to(device)
    model.eval()

    # SNNInference opens the zarr with mode="w" — stale caches get overwritten.
    runner = SNNInference(
        model=model,
        dataloader=spike_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=cache_path,
        save_tracked_variables=False,
        max_chunks=num_chunks,
        progress_bar=True,
    )
    runner.run()

    # Stamp attrs used by the cache-validity check and downstream consumers.
    root = zarr.open_group(str(cache_path), mode="a")
    root.attrs["dt"] = float(dt)
    root.attrs["config_hash"] = cfg_hash
    root.attrs["test_seed"] = int(test_seed)
