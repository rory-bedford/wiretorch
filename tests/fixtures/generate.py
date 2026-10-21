"""Record the reference fixtures the test suite compares against.

    python tests/fixtures/generate.py              # CPU fixtures (every scenario + training)
    python tests/fixtures/generate.py statistics [fixture ...]  # GPU (large networks)
    python tests/fixtures/generate.py scenarios [name ...]      # CPU, chosen scenarios
    python tests/fixtures/generate.py state_dicts  # every scenario's state_dict
    python tests/fixtures/generate.py training_log # SNNTrainer's metrics CSV
    python tests/fixtures/generate.py configs      # layer-config method outputs

The fixtures *define* correct behaviour: they were recorded from the simulators before
any optimisation, and every later change is checked against them. Regenerating them is
therefore a deliberate act — only after a change that is meant to alter the dynamics,
and never to make a failing test pass. Each file records the commit it came from.
"""

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

import fixture_io  # noqa: E402
import harness  # noqa: E402
from networks import (  # noqa: E402
    BATCH_SIZE,
    SCENARIOS,
    SEED,
    current_feedforward,
    current_recurrent,
    feedforward,
    fingerprint,
    recurrent,
)


def meta(**extra):
    return {
        **fixture_io.library_commit(),
        "torch": torch.__version__,
        "recorded": date.today().isoformat(),
        **extra,
    }


# =====================================================================
# Small networks, on CPU: free run, lockstep states and gradients
# =====================================================================


def record_scenario(scenario):
    built = scenario.build(BATCH_SIZE)
    arrays = harness.free_run(scenario, built)
    if harness.is_trainable(built):
        loss, grads = harness.gradients(scenario.build(BATCH_SIZE))
        arrays["loss"] = np.array(loss)
        arrays.update({f"grad.{name}": g for name, g in grads.items()})
    return fixture_io.save(
        scenario.name,
        arrays,
        meta(
            fingerprint=fingerprint(built),
            batch_size=BATCH_SIZE,
            window=harness.WINDOW,
            grad_steps=harness.GRAD_STEPS,
            device="cpu",
        ),
    )


# =====================================================================
# A short training run through SNNTrainer, on CPU
# =====================================================================

TRAIN_CHUNK = 40  # steps per chunk; the 200-step input gives 5 chunks per data epoch
TRAIN_DATA_EPOCHS = 2


def training_setup(
    log_interval=10**9,
    checkpoint_interval=10**9,
    stats=False,
    scheduler=False,
    checkpoint_callbacks=None,
    chunks_per_update=2,
):
    """A feedforward network fitted to a recurrent one's spikes, its data and its
    trainer — shared with the tests.

    The target is ``recurrent_frozen``'s spikes; the model is
    ``feedforward_scaling_factors`` built on the same seed, so it has that network's
    weights and perturbed scaling factors. Its input rows are the recurrent network's
    external input and spikes. The defaults are the configuration ``training.npz`` was
    recorded with. (The logged stats keep the names ``training_log.npz`` recorded.)
    """
    from connectome_snns.dataloaders.supervised import SpikeData
    from connectome_snns.snn_runners import SNNTrainer
    from connectome_snns.training_utils.losses import VanRossumLoss
    from torch.amp import GradScaler

    source = recurrent(BATCH_SIZE, SEED)
    external = torch.from_numpy(source.input_spikes).float()
    with torch.no_grad():
        target = source.model(external)
    model = feedforward(BATCH_SIZE, SEED).model
    inputs = torch.cat([external, target], dim=2)

    n_chunks = inputs.shape[1] // TRAIN_CHUNK
    chunks = [
        SpikeData(
            input_spikes=inputs[:, c * TRAIN_CHUNK : (c + 1) * TRAIN_CHUNK],
            target_spikes=target[:, c * TRAIN_CHUNK : (c + 1) * TRAIN_CHUNK],
            recurrent_input_spikes=None,
            target_calcium=None,
            target_rate=None,
        )
        for c in range(n_chunks)
    ]
    trajectory = []
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

    def spike_rates(snapshot):
        return {
            "rate/student": float(snapshot["spikes"].mean() * 1000),
            "rate/teacher": float(snapshot["target_spikes"].mean() * 1000),
        }

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=GradScaler("cuda", enabled=False),
        dataloader=chunks * TRAIN_DATA_EPOCHS,
        loss_functions={
            "van_rossum": VanRossumLoss(
                tau_rise=10.0, tau_decay=100.0, dt=1.0, window_size=TRAIN_CHUNK
            )
        },
        loss_weights={"van_rossum": 1.0},
        device="cpu",
        num_epochs=n_chunks * TRAIN_DATA_EPOCHS,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=1,
        mixed_precision=False,
        chunks_per_data_epoch=n_chunks,
        burn_in_chunks=1,
        stats_computer=spike_rates if stats else None,
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.5)
        if scheduler
        else None,
        update_callbacks=[
            lambda: trajectory.append(
                [p.detach().clone().numpy() for p in model.parameters()]
            )
        ],
        checkpoint_callbacks=checkpoint_callbacks,
    )
    names = [name for name, _ in model.named_parameters()]
    return trainer, trajectory, names


def record_training():
    trainer, trajectory, names = training_setup()
    trainer.train()
    arrays = {
        f"trajectory.{name}": np.stack([step[i] for step in trajectory])
        for i, name in enumerate(names)
    }
    return fixture_io.save("training", arrays, meta(device="cpu"))


TRAIN_LOG = dict(log_interval=2, stats=True, scheduler=True)


def training_log(output_dir):
    """Train with ``TRAIN_LOG`` into ``output_dir``; return its metrics CSV as arrays."""
    import pandas as pd

    trainer, _, _ = training_setup(**TRAIN_LOG)
    trainer.train(output_dir=output_dir)
    trainer.metrics_logger.close()
    table = pd.read_csv(output_dir / "training_metrics.csv")
    return {f"log.{column}": table[column].to_numpy() for column in table.columns}


def record_training_log():
    import tempfile

    with tempfile.TemporaryDirectory(
        dir=fixture_io.FIXTURES.parent / ".pytest_tmp"
    ) as d:
        arrays = training_log(Path(d))
    return fixture_io.save("training_log", arrays, meta(device="cpu", **TRAIN_LOG))


# =====================================================================
# Large networks, on GPU: firing statistics
# =====================================================================

STATS_BATCH = 4
STATS_BURN_IN = 500  # steps discarded before measuring rates
STATS_NETWORKS = {
    # name: (builder, kwargs, steps)
    "recurrent": (recurrent, dict(n=2000, n_ff=600), 3000),
    "feedforward": (feedforward, dict(n=1000, n_ff=400), 2000),
    "current_recurrent": (current_recurrent, dict(n=2000, n_ff=600), 3000),
    "current_feedforward": (current_feedforward, dict(n=1000, n_ff=400), 2000),
}
STATS_INPUT_SEEDS = (0, 1)
#: The fixture each family's networks are recorded in, and the backend recording them.
#: The conductance-based ones were recorded from the original simulator; the
#: current-based ones from the PyTorch reference (``current/dynamics.py``), which
#: reproduces the original current-based simulator spike for spike.
STATS_FIXTURES = {
    "statistics": (("recurrent", "feedforward"), "triton"),
    "statistics_current": (("current_recurrent", "current_feedforward"), "torch"),
}


def stats_run(name, input_seed, device="cuda", compile=False, storage_dtype="float32"):
    """Free-run one statistics network; return its spikes after the burn-in, on CPU."""
    builder, kwargs, n_steps = STATS_NETWORKS[name]
    built = builder(STATS_BATCH, SEED, n_steps=n_steps, input_seed=input_seed, **kwargs)
    model = built.model.to(device)
    model.storage_dtype = storage_dtype
    if compile:
        model = torch.compile(model)
    x = torch.from_numpy(built.input_spikes).to(device)
    out = []
    with torch.inference_mode():
        for start in range(0, n_steps, 250):
            out.append(
                harness.unpack_output(model(x[:, start : start + 250]))["spikes"]
            )
    spikes = torch.cat(out, dim=1)[:, STATS_BURN_IN:]
    return built, (spikes > 0.5).cpu().numpy()


def record_statistics(fixture="statistics"):
    names, backend = STATS_FIXTURES[fixture]
    os.environ["CONNECTOME_SNNS_BACKEND"] = backend
    arrays, fingerprints = {}, {}
    for name in names:
        for seed in STATS_INPUT_SEEDS:
            built, spikes = stats_run(name, seed)
            arrays[f"{name}.rates.{seed}"] = spikes.mean(axis=(0, 1)) * 1000.0
            arrays[f"{name}.population_rate.{seed}"] = spikes.mean(axis=(0, 2)) * 1000.0
            fingerprints[f"{name}.{seed}"] = fingerprint(built)
            if name.endswith("feedforward") and seed == 0:
                arrays[f"{name}.spikes"] = spikes
            types = built.structure["types"]
            rates = arrays[f"{name}.rates.{seed}"]
            print(
                f"  {name} input {seed}: E {rates[types == 0].mean():.2f} Hz, "
                f"I {rates[types == 1].mean():.2f} Hz"
            )
    return fixture_io.save(
        fixture,
        arrays,
        meta(
            device=torch.cuda.get_device_name(),
            backend=backend,
            batch_size=STATS_BATCH,
            burn_in=STATS_BURN_IN,
            fingerprints=fingerprints,
        ),
    )


# =====================================================================
# State dicts, for checkpoint compatibility
# =====================================================================


def record_state_dicts():
    """Every scenario's ``state_dict``: checkpoints from this code must keep loading."""
    states = {
        name: s.build(BATCH_SIZE).model.state_dict() for name, s in SCENARIOS.items()
    }
    path = fixture_io.FIXTURES / "state_dicts.pt"
    torch.save({"meta": meta(), "states": states}, path)
    return path


# =====================================================================
# Config methods
# =====================================================================

CONFIG_METHODS = (
    "get_cell_params", "get_synapse_params", "get_g_bar_by_type",
    "get_tau_rise_by_type", "get_tau_decay_by_type", "get_E_syn_by_type",
    "get_synapse_names", "get_excitatory_synapse_indices",
    "get_inhibitory_synapse_indices", "get_neuron_params_for_plotting",
)  # fmt: skip


def config_outputs():
    """Every layer-config method's output, on a realistic parameter set (the teacher
    network of the teacher-student experiments)."""
    import json

    from connectome_snns.configs.conductance_based import (
        FeedforwardLayerConfig,
        RecurrentLayerConfig,
    )

    sys.path.insert(0, str(fixture_io.REPO_ROOT / "benchmarks"))
    import speed

    layers = {
        "recurrent": RecurrentLayerConfig(**speed.RECURRENT),
        "feedforward": FeedforwardLayerConfig(**speed.FEEDFORWARD),
    }
    out = {}
    for name, layer in layers.items():
        for method in CONFIG_METHODS:
            if hasattr(layer, method):
                out[f"{name}.{method}"] = getattr(layer, method)()
    return json.loads(json.dumps(out, default=float))


def record_configs():
    import json

    path = fixture_io.FIXTURES / "configs.json"
    path.write_text(json.dumps({"meta": meta(), "outputs": config_outputs()}, indent=1))
    return path


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)  # CPU matmul results can depend on the thread count
    if sys.argv[1:2] == ["statistics"]:
        for fixture in sys.argv[2:] or STATS_FIXTURES:
            print(f"Wrote {record_statistics(fixture)}")
    elif sys.argv[1:2] == ["scenarios"]:
        for name in sys.argv[2:] or SCENARIOS:
            print(f"Wrote {record_scenario(SCENARIOS[name])}")
    elif sys.argv[1:] == ["state_dicts"]:
        print(f"Wrote {record_state_dicts()}")
    elif sys.argv[1:] == ["training_log"]:
        print(f"Wrote {record_training_log()}")
    elif sys.argv[1:] == ["configs"]:
        print(f"Wrote {record_configs()}")
    else:
        for scenario in SCENARIOS.values():
            print(f"Wrote {record_scenario(scenario)}")
        print(f"Wrote {record_training()}")
