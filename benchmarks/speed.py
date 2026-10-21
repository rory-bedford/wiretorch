"""Simulator speed on a network the size of the Bernstein teacher.

    python benchmarks/speed.py                 # both benchmarks
    python benchmarks/speed.py inference       # teacher simulation only
    python benchmarks/speed.py training        # fully observed student training
    python benchmarks/speed.py partial         # half-observed (two-layer) training
    python benchmarks/speed.py current         # the current-based networks
    python benchmarks/speed.py rate            # the rate networks (kernels and reference)
    python benchmarks/speed.py --badges        # also refresh docs/badges/*.json
    python benchmarks/speed.py --storage-dtype float16   # synaptic state stored in fp16

The teacher is rebuilt from its own parameters (``fig00-teacher-activity`` in the
teacher-student repo: 5000 conductance-based LIF neurons in 20 assemblies, 1500 mitral
inputs) with the library's connectome generators, so nothing outside this repo is read.

* **inference** — the teacher, compiled, under ``torch.inference_mode``, in 2500-step
  chunks: how the teacher's spike data is generated.
* **training** — the fully observed student (every teacher neuron's spikes as input,
  six scaling factors) through ``SNNTrainer`` with the Figure 1 settings: 160-step
  chunks, van Rossum loss, an update every 6, logging every 6.
* **partial** — the same with half the neurons observed (Figure 1's actual setting):
  a ``TwoLayerSNN`` whose hidden layer is recurrent.

* **current** — current-based networks on the teacher's connectome (natural units,
  weights rescaled for teacher-like rates): simulation and training (one forward +
  backward of a random spike readout, learnt scaling factors) of a recurrent network,
  and of a feedforward one whose input rows are the inputs plus a population of the
  same size (as in teacher forcing).

* **rate** — rate networks on the teacher's connectome (ReLU, white-noise input): the
  same four cases, training with learnt scaling factors and physiology — the Triton
  kernels (float32 and float16 products), and the PyTorch reference on the same GPU.

All use a batch of 50 trials and dt = 1 ms, and report wall-clock seconds per second
of network time, after the warm-up (compilation and the first data epoch) is discarded.
Results are appended to ``benchmarks/results/``.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.network_simulators.conductance.recurrent import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    ScalingFactorProjection,
    make_frozen_projections,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN
from connectome_snns.synthetic_connectome import (
    cell_types,
    topology_generators,
    weight_assigners,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = HERE / ".cache"
BADGES = HERE.parent / "docs" / "badges"

BATCH = 50
DT = 1.0
SEED = 43

# The teacher's parameters (fig00-teacher-activity/parameters.toml), minus the odours.
SYNAPSE_E = dict(names=["AMPA", "NMDA"], tau_rise=[0.1, 10.0], tau_decay=[5.0, 60.0],
                 E_syn=[0.0, 0.0], g_bar=[4.0, 0.5], g_clip=[40.0, 4.0])  # fmt: skip
RECURRENT = dict(
    cell_types=dict(names=["excitatory", "inhibitory"], proportion=[0.8, 0.2]),
    topology=dict(num_neurons=5000, num_assemblies=20,
                  conn_within=[[0.3, 0.3], [0.4, 0.3]],
                  conn_between=[[0.05, 0.05], [0.05, 0.05]]),
    weights=dict(w_mu=[[0.02, 0.02], [0.08, 0.08]], w_sigma=[[0.05, 0.05], [0.05, 0.05]]),
    physiology=dict(
        excitatory=dict(tau_mem=85.0, theta=-38.0, U_reset=-60.0, E_L=-60.0,
                        g_L=1.35),
        inhibitory=dict(tau_mem=50.0, theta=-45.0, U_reset=-60.0, E_L=-60.0,
                        g_L=0.9),
    ),
    synapses=dict(
        excitatory=SYNAPSE_E,
        inhibitory=dict(names=["GABA_A"], tau_rise=[0.5], tau_decay=[6.0],
                        E_syn=[-70.0], g_bar=[8.0], g_clip=[80.0]),
    ),
)  # fmt: skip
FEEDFORWARD = dict(
    cell_types=dict(names=["mitral"], proportion=[1]),
    topology=dict(num_neurons=1500, conn_inputs=[[0.05, 0.05]]),
    weights=dict(w_mu=[[0.02, 0.01]], w_sigma=[[0.05, 0.05]]),
    synapses=dict(mitral={**SYNAPSE_E, "tau_decay": [5.0, 70.0], "g_bar": [6.0, 0.1]}),
)
# Mean rates of the teacher's inputs and neurons (Hz), for the Poisson stand-ins.
RATE_MITRAL, RATE_E, RATE_I = 6.9, 4.3, 19.0


# =====================================================================
# The teacher
# =====================================================================


def teacher_structure():
    """The teacher's cell types and weights, generated once and cached."""
    path = CACHE / "teacher_structure.npz"
    if path.exists():
        return dict(np.load(path))
    print("Generating the teacher connectome (cached for next time)...")
    np.random.seed(SEED)
    rec, ff = RecurrentLayerConfig(**RECURRENT), FeedforwardLayerConfig(**FEEDFORWARD)
    types = cell_types.assign_cell_types(
        rec.topology.num_neurons, rec.cell_types.proportion
    )
    graph, _ = topology_generators.assembly_generator(
        source_cell_types=types,
        target_cell_types=types,
        num_assemblies=rec.topology.num_assemblies,
        conn_within=rec.topology.conn_within,
        conn_between=rec.topology.conn_between,
        method="configuration",
    )
    w_rec = weight_assigners.assign_weights_lognormal(
        graph, types, types, None, rec.weights.w_mu, rec.weights.w_sigma, "linear"
    )
    ff_types = cell_types.assign_cell_types(
        ff.topology.num_neurons, ff.cell_types.proportion
    )
    ff_graph = topology_generators.sparse_graph_generator(
        source_cell_types=ff_types,
        target_cell_types=types,
        conn_matrix=ff.topology.conn_inputs,
        allow_self_loops=True,
        method="configuration",
    )
    w_ff = weight_assigners.assign_weights_lognormal(
        ff_graph, ff_types, types, None, ff.weights.w_mu, ff.weights.w_sigma, "linear"
    )
    structure = dict(
        types=types,
        ff_types=ff_types,
        w_rec=w_rec.astype(np.float32),
        w_ff=w_ff.astype(np.float32),
    )
    CACHE.mkdir(exist_ok=True)
    np.savez(path, **structure)
    return structure


def poisson(n_steps, rates_hz, device, generator):
    p = torch.as_tensor(rates_hz, dtype=torch.float32, device=device) * DT / 1000.0
    u = torch.rand(BATCH, n_steps, len(p), device=device, generator=generator)
    return (u < p).float()


def per_second(durations, chunk_steps):
    d = np.asarray(durations) / (chunk_steps * DT / 1000.0)
    return {
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p10": float(np.percentile(d, 10)),
        "p90": float(np.percentile(d, 90)),
        "n_chunks": int(d.size),
    }


# =====================================================================
# Inference
# =====================================================================


def bench_inference(device, chunk=2500, n_warmup=2, n_timed=6, storage_dtype="float32"):
    s = teacher_structure()
    rec, ff = RecurrentLayerConfig(**RECURRENT), FeedforwardLayerConfig(**FEEDFORWARD)
    rec_proj, ff_proj = make_frozen_projections(
        s["w_rec"], s["w_ff"], s["types"], s["ff_types"],
        rec.cell_types.names, ff.cell_types.names,
    )  # fmt: skip
    model = ConductanceLIFNetwork(
        dt=DT,
        rec_projections=rec_proj,
        ff_projections=ff_proj,
        cell_type_indices=s["types"],
        cell_type_indices_FF=s["ff_types"],
        cell_params=rec.get_cell_params(),
        cell_params_FF=ff.get_cell_params(),
        synapse_params=rec.get_synapse_params(),
        synapse_params_FF=ff.get_synapse_params(),
        surrgrad_scale=1.0,
        batch_size=BATCH,
        storage_dtype=storage_dtype,
    ).to(device)
    model = torch.compile(model)

    generator = torch.Generator(device).manual_seed(SEED)
    durations, n_spikes = [], 0
    with torch.inference_mode():
        for c in range(n_warmup + n_timed):
            x = poisson(chunk, [RATE_MITRAL] * s["ff_types"].size, device, generator)
            torch.cuda.synchronize()
            start = time.perf_counter()
            spikes = model(x)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            if c >= n_warmup:
                n_spikes += int(spikes.sum())
    timed = durations[n_warmup:]
    rate = n_spikes / (BATCH * s["types"].size * n_timed * chunk * DT / 1000.0)
    return {
        "wall_s_per_network_s": per_second(timed, chunk),
        "chunk_steps": chunk,
        "warmup_s": durations[:n_warmup],
        "mean_rate_hz": rate,
    }


# =====================================================================
# Training
# =====================================================================


class ChunkTimer:
    """Stands in for the trainer's progress bar: ``update`` fires as each chunk starts."""

    def __init__(self):
        self.stamps = []

    def update(self, n=1):
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())

    close = update

    def set_postfix(self, *args, **kwargs):
        pass

    clear = refresh = set_postfix


def student_physiology():
    """Cells and synapses of a teacher-forced layer's inputs: mitral, then E and I."""
    rec, ff = RecurrentLayerConfig(**RECURRENT), FeedforwardLayerConfig(**FEEDFORWARD)
    rec_cells, ff_cells = rec.get_cell_params(), ff.get_cell_params()
    rec_syn, ff_syn = rec.get_synapse_params(), ff.get_synapse_params()
    shift = len(ff_cells)
    cells = ff_cells + [{**c, "cell_id": c["cell_id"] + shift} for c in rec_cells]
    synapses = ff_syn + [
        {**y, "cell_id": y["cell_id"] + shift, "synapse_id": y["synapse_id"] + len(ff_syn)}
        for y in rec_syn
    ]  # fmt: skip
    return rec_cells, rec_syn, cells, synapses


def scaling_projections(weights, src_types, tgt_types, src_names, tgt_names, log_sf):
    """One ``ScalingFactorProjection`` per type pair, sharing the ``log_sf`` parameters."""
    out = {}
    for a, src in enumerate(src_names):
        for b, tgt in enumerate(tgt_names):
            rows, cols = np.flatnonzero(src_types == a), np.flatnonzero(tgt_types == b)
            if rows.size and cols.size:
                projection = ScalingFactorProjection(weights[np.ix_(rows, cols)])
                projection.log_sf = log_sf[(src, tgt)]
                out[(src, tgt)] = projection
    return out


def build_student(observed: float, device):
    """The Bernstein student observing a fraction of the teacher's neurons.

    Fully observed: one teacher-forced ``FeedforwardConductanceLIFNetwork``. Otherwise
    a ``TwoLayerSNN``: the unobserved neurons as a recurrent layer, the observed ones
    teacher-forced, one scaling factor per type pair shared by both. Scaling factors
    start at 1 and are then moved, as training would.
    """
    s = teacher_structure()
    rec_cells, rec_syn, cells, synapses = student_physiology()
    names = [c["name"] for c in cells]
    rec_names = [c["name"] for c in rec_cells]
    rng = np.random.default_rng(SEED)
    log_sf = {
        (a, b): torch.nn.Parameter(torch.zeros(())) for a in names for b in rec_names
    }
    types, ff_types = s["types"], s["ff_types"]
    shifted = types + 1  # the teacher's neurons as input rows
    n = types.size
    seen = np.sort(rng.choice(n, round(observed * n), replace=False))
    hidden = np.setdiff1d(np.arange(n), seen)

    def feedforward(rows_w, rows_types, targets):
        return FeedforwardConductanceLIFNetwork(
            dt=DT,
            projections=scaling_projections(
                rows_w, rows_types, types[targets], names, rec_names, log_sf
            ),
            cell_type_indices=types[targets],
            cell_type_indices_FF=rows_types,
            cell_params=rec_cells,
            cell_params_FF=cells,
            synapse_params_FF=synapses,
            surrgrad_scale=5.0,
            batch_size=BATCH,
        )

    if hidden.size == 0:
        model = feedforward(
            np.concatenate([s["w_ff"], s["w_rec"]]),
            np.concatenate([ff_types, shifted]),
            np.arange(n),
        )
    else:
        l1_rows = np.concatenate(
            [s["w_ff"][:, hidden], s["w_rec"][np.ix_(seen, hidden)]]
        )
        l1_types = np.concatenate([ff_types, shifted[seen]])
        layer1 = ConductanceLIFNetwork(
            dt=DT,
            rec_projections=scaling_projections(
                s["w_rec"][np.ix_(hidden, hidden)], types[hidden], types[hidden],
                rec_names, rec_names, log_sf,
            ),
            ff_projections=scaling_projections(
                l1_rows, l1_types, types[hidden], names, rec_names, log_sf
            ),
            cell_type_indices=types[hidden],
            cell_type_indices_FF=l1_types,
            cell_params=rec_cells,
            cell_params_FF=cells,
            synapse_params=rec_syn,
            synapse_params_FF=synapses,
            surrgrad_scale=5.0,
            batch_size=BATCH,
        )  # fmt: skip
        layer2 = feedforward(
            np.concatenate(
                [s["w_ff"][:, seen], s["w_rec"][np.ix_(hidden, seen)], s["w_rec"][np.ix_(seen, seen)]]
            ),
            np.concatenate([ff_types, shifted[hidden], shifted[seen]]),
            seen,
        )  # fmt: skip
        model = TwoLayerSNN(
            layer1, layer2, n_ff=ff_types.size, return_hidden_spikes=True
        )
    with torch.no_grad():  # move the scaling factors away from the truth
        for parameter in log_sf.values():
            parameter.fill_(float(rng.normal(0.0, 0.5)))
    return model.to(device), seen


def bench_training(
    device, observed=1.0, chunk=160, chunks_per_epoch=30, burn_in=5, n_epochs=2,
    storage_dtype="float32",
):  # fmt: skip
    from connectome_snns.dataloaders.supervised import SpikeData
    from connectome_snns.snn_runners import SNNTrainer
    from connectome_snns.training_utils.losses import VanRossumLoss
    from torch.amp import GradScaler

    s = teacher_structure()
    model, seen = build_student(observed, device)
    for module in model.modules():
        if hasattr(module, "storage_dtype"):
            module.storage_dtype = storage_dtype
    generator = torch.Generator(device).manual_seed(SEED)
    teacher_rates = np.where(s["types"][seen] == 0, RATE_E, RATE_I)
    data = []
    for _ in range(4):  # cycled: speed does not depend on which data a chunk gets
        target = poisson(chunk, teacher_rates, device, generator)
        mitral = poisson(chunk, [RATE_MITRAL] * s["ff_types"].size, device, generator)
        data.append(
            SpikeData(torch.cat([mitral, target], dim=2), target, None, None, None)
        )

    timer = ChunkTimer()
    trainer = SNNTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.008, betas=(0.95, 0.999)),
        scaler=GradScaler("cuda", enabled=False),
        dataloader=[data[c % len(data)] for c in range(chunks_per_epoch * n_epochs)],
        loss_functions={
            "van_rossum": VanRossumLoss(10.0, 100.0, DT, chunk, device=device)
        },
        loss_weights={"van_rossum": 1.0},
        device=device,
        num_epochs=chunks_per_epoch * n_epochs,
        chunks_per_update=6,
        log_interval=6,
        checkpoint_interval=48,
        plot_size=48,
        mixed_precision=False,
        progress_bar=timer,
        chunks_per_data_epoch=chunks_per_epoch,
        burn_in_chunks=burn_in,
    )
    trainer.train()

    durations = np.diff(timer.stamps)
    position = np.arange(durations.size) % chunks_per_epoch
    epoch = np.arange(durations.size) // chunks_per_epoch
    timed = epoch >= 1  # the first data epoch carries compilation
    return {
        "wall_s_per_network_s": per_second(
            durations[timed & (position >= burn_in)], chunk
        ),
        "burn_in_wall_s_per_network_s": per_second(
            durations[timed & (position < burn_in) & (position > 0)], chunk
        ),
        "observed": observed,
        "chunk_steps": chunk,
        "warmup_s": float(durations[epoch == 0].sum()),
    }


# =====================================================================
# Current-based networks
# =====================================================================

#: What the current-based and rate benchmarks time, for each network.
GRID = (
    "recurrent_simulation",
    "recurrent_training",
    "feedforward_simulation",
    "feedforward_training",
)

CURRENT_CELLS = [
    dict(name="excitatory", cell_id=0, tau_mem=20.0),
    dict(name="inhibitory", cell_id=1, tau_mem=10.0),
]
CURRENT_SYNAPSES = [
    dict(name="exc", synapse_id=0, cell_id=0, tau_syn=5.0, type="excitatory"),
    dict(name="inh", synapse_id=1, cell_id=1, tau_syn=10.0, type="inhibitory"),
]
CURRENT_SYNAPSES_FF = [
    dict(name="ff", synapse_id=0, cell_id=0, tau_syn=5.0, type="excitatory")
]
# The teacher's weights in natural units: recurrent x2 (inhibitory rows x20 in all),
# feedforward x4, for E ~5 Hz, I ~7 Hz.
CURRENT_SCALE = dict(recurrent=2.0, inhibitory=10.0, feedforward=4.0)


def bench_current(device, n_warmup=2, n_timed=5, storage_dtype="float32"):
    from connectome_snns.network_simulators.current import (
        CurrentLIFNetwork,
        FeedforwardCurrentLIFNetwork,
    )

    s = teacher_structure()
    types, ff_types = s["types"], s["ff_types"]
    w_rec = s["w_rec"] * CURRENT_SCALE["recurrent"]
    w_rec[types == 1] *= CURRENT_SCALE["inhibitory"]
    w_ff = s["w_ff"] * CURRENT_SCALE["feedforward"]
    common = dict(
        dt=DT, cell_type_indices=types, cell_params=CURRENT_CELLS,
        surrgrad_scale=5.0, batch_size=BATCH,
    )  # fmt: skip

    def stored(model):
        model.storage_dtype = storage_dtype
        return model

    recurrent = dict(
        weights=w_rec, weights_FF=w_ff, cell_type_indices_FF=ff_types,
        synapse_params=CURRENT_SYNAPSES, synapse_params_FF=CURRENT_SYNAPSES_FF,
        **common,
    )  # fmt: skip
    generator = torch.Generator(device).manual_seed(SEED)

    def timed(fn, chunk):
        durations = []
        for _ in range(n_warmup + n_timed):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
        return per_second(durations[n_warmup:], chunk), out

    result = {}
    model = stored(CurrentLIFNetwork.from_weights(**recurrent).to(device))
    chunk = 1000
    with torch.inference_mode():
        result["recurrent_simulation"], spikes = timed(
            lambda: model(poisson(chunk, [RATE_MITRAL] * ff_types.size, device, generator)),
            chunk,
        )  # fmt: skip
    rates = spikes.mean(dim=(0, 1)).cpu().numpy() * 1000.0
    result["rates_hz"] = {"E": float(rates[types == 0].mean()), "I": float(rates[types == 1].mean())}  # fmt: skip

    chunk = 160
    readout = torch.randn(BATCH, chunk, types.size, device=device, generator=generator)

    def train_step(model, x):
        model.reset_state()
        model.zero_grad()
        (model(x) * readout).sum().backward()

    model = stored(
        CurrentLIFNetwork.from_weights(**recurrent, optimisable="scaling_factors")
    ).to(device)
    x = poisson(chunk, [RATE_MITRAL] * ff_types.size, device, generator)
    result["recurrent_training"], _ = timed(lambda: train_step(model, x), chunk)

    def feedforward(optimisable=None):
        return stored(FeedforwardCurrentLIFNetwork.from_weights(
            weights_FF=np.concatenate([w_ff, w_rec]),
            cell_type_indices_FF=np.concatenate([ff_types, types + 1]),
            synapse_params_FF=CURRENT_SYNAPSES_FF + [
                {**y, "cell_id": y["cell_id"] + 1, "synapse_id": y["synapse_id"] + 1}
                for y in CURRENT_SYNAPSES
            ],
            optimisable=optimisable,
            **common,
        )).to(device)  # fmt: skip

    def rows(chunk):
        return torch.cat(
            [
                poisson(chunk, [RATE_MITRAL] * ff_types.size, device, generator),
                poisson(chunk, np.where(types == 0, RATE_E, RATE_I), device, generator),
            ],
            dim=2,
        )

    model = feedforward()
    with torch.inference_mode():
        result["feedforward_simulation"], _ = timed(lambda: model(rows(1000)), 1000)
    model = feedforward("scaling_factors")
    x = rows(chunk)
    result["feedforward_training"], _ = timed(lambda: train_step(model, x), chunk)
    return result


# =====================================================================
# Rate networks
# =====================================================================

RATE_CELLS = [
    dict(name="excitatory", cell_id=0, tau_mem=10.0, nonlinearity="relu"),
    dict(name="inhibitory", cell_id=1, tau_mem=5.0, nonlinearity="relu"),
]
RATE_SYNAPSES = [
    dict(name="exc", synapse_id=0, cell_id=0, tau_syn=5.0, type="excitatory"),
    dict(name="inh", synapse_id=1, cell_id=1, tau_syn=5.0, type="inhibitory"),
]
RATE_SYNAPSES_FF = [
    dict(name="ff", synapse_id=0, cell_id=0, tau_syn=2.0, type="excitatory")
]
# The teacher's weights rescaled for ReLU rates of order 1 (inhibitory rows x4).
RATE_SCALE = dict(recurrent=0.1, inhibitory=2.0, feedforward=1.0)
RATE_LEARN = {"tau_mem": True, "theta": True, "gain": True, "tau_syn": True}


def bench_rate(device, n_warmup=2, n_timed=5):
    import os

    from connectome_snns.network_simulators.rate import (
        FeedforwardRateNetwork,
        RecurrentRateNetwork,
    )

    s = teacher_structure()
    types, ff_types = s["types"], s["ff_types"]
    w_rec = s["w_rec"] * RATE_SCALE["recurrent"]
    w_rec[types == 1] *= RATE_SCALE["inhibitory"]
    w_ff = s["w_ff"] * RATE_SCALE["feedforward"]
    names = ["excitatory", "inhibitory"]

    def blocks(weights, src_types, src_names):
        out = {}
        for a, src in enumerate(src_names):
            for b, tgt in enumerate(names):
                rows, cols = np.flatnonzero(src_types == a), np.flatnonzero(types == b)
                out[(src, tgt)] = ScalingFactorProjection(weights[np.ix_(rows, cols)])
        return out

    def recurrent(learn=None):
        return RecurrentRateNetwork(
            dt=DT, rec_projections=blocks(w_rec, types, names),
            ff_projections=blocks(w_ff, ff_types, ["mitral"]),
            cell_type_indices=types, cell_type_indices_FF=ff_types,
            cell_params=RATE_CELLS, cell_params_FF=[dict(name="mitral", cell_id=0)],
            synapse_params=RATE_SYNAPSES, synapse_params_FF=RATE_SYNAPSES_FF,
            batch_size=BATCH, learn=learn,
        ).to(device)  # fmt: skip

    def feedforward():
        cells = [dict(name="mitral", cell_id=0)] + [
            dict(name=c["name"], cell_id=c["cell_id"] + 1) for c in RATE_CELLS
        ]
        synapses = RATE_SYNAPSES_FF + [
            {**y, "cell_id": y["cell_id"] + 1, "synapse_id": y["synapse_id"] + 1}
            for y in RATE_SYNAPSES
        ]
        row_types = np.concatenate([ff_types, types + 1])
        return FeedforwardRateNetwork(
            dt=DT, projections=blocks(np.concatenate([w_ff, w_rec]), row_types,
                                      [c["name"] for c in cells]),
            cell_type_indices=types, cell_type_indices_FF=row_types,
            cell_params=RATE_CELLS, cell_params_FF=cells, synapse_params_FF=synapses,
            batch_size=BATCH, learn=RATE_LEARN,
        ).to(device)  # fmt: skip

    generator = torch.Generator(device).manual_seed(SEED)

    def noise(chunk, n):
        return 1.0 + torch.randn(BATCH, chunk, n, device=device, generator=generator)

    def timed(fn, chunk):
        durations = []
        for _ in range(n_warmup + n_timed):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
        return per_second(durations[n_warmup:], chunk), out

    def train_step(model, x, readout):
        model.reset_state()
        model.zero_grad()
        (model(x) * readout).sum().backward()

    result = {}
    for label, backend, dtype in (
        ("triton", "triton", "float32"),
        ("triton-float16", "triton", "float16"),
        ("torch", "torch", "float32"),
    ):
        os.environ["CONNECTOME_SNNS_BACKEND"] = backend
        out = result[label] = {}
        model = recurrent()
        model.storage_dtype = dtype
        chunk = 1000
        with torch.inference_mode():
            out["recurrent_simulation"], rates = timed(
                lambda: model(noise(chunk, ff_types.size)), chunk
            )
        r = rates[:, 200:].mean(dim=(0, 1)).cpu().numpy()
        out["rates"] = {"E": float(r[types == 0].mean()), "I": float(r[types == 1].mean())}  # fmt: skip
        chunk = 160
        readout = torch.randn(
            BATCH, chunk, types.size, device=device, generator=generator
        )
        model = recurrent(RATE_LEARN)
        model.storage_dtype = dtype
        x = noise(chunk, ff_types.size)
        out["recurrent_training"], _ = timed(
            lambda: train_step(model, x, readout), chunk
        )

        def rows(chunk):
            return torch.cat(
                [noise(chunk, ff_types.size), noise(chunk, types.size).clamp(min=0)],
                dim=2,
            )

        model = feedforward()
        model.storage_dtype = dtype
        with torch.inference_mode():
            out["feedforward_simulation"], _ = timed(lambda: model(rows(1000)), 1000)
        model.reset_state()
        x = rows(chunk)
        out["feedforward_training"], _ = timed(
            lambda: train_step(model, x, readout), chunk
        )
    os.environ.pop("CONNECTOME_SNNS_BACKEND")
    return result


# =====================================================================
# Badges
# =====================================================================


def write_badge(name, label, seconds):
    BADGES.mkdir(parents=True, exist_ok=True)
    badge = {
        "schemaVersion": 1,
        "label": label,
        "message": f"{seconds:.2f} s per network s",
        "color": "blue",
    }
    (BADGES / f"{name}.json").write_text(json.dumps(badge, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "which",
        nargs="?",
        choices=["inference", "training", "partial", "current", "rate", "all"],
        default="all",
    )
    parser.add_argument(
        "--badges", action="store_true", help="refresh docs/badges/*.json"
    )
    parser.add_argument("--label", default="", help="a note stored with the result")
    parser.add_argument(
        "--storage-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="what the kernels store conductances and recurrent weights in",
    )
    args = parser.parse_args()

    device = "cuda"
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "date": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "batch": BATCH,
        "storage_dtype": args.storage_dtype,
    }
    if args.which in ("inference", "all"):
        result["inference"] = bench_inference(device, storage_dtype=args.storage_dtype)
        print("inference:", json.dumps(result["inference"]["wall_s_per_network_s"]))
    if args.which in ("training", "all"):
        result["training"] = bench_training(device, storage_dtype=args.storage_dtype)
        print("training:", json.dumps(result["training"]["wall_s_per_network_s"]))
    if args.which in ("partial", "all"):
        result["partial"] = bench_training(
            device, observed=0.5, storage_dtype=args.storage_dtype
        )
        print("partial:", json.dumps(result["partial"]["wall_s_per_network_s"]))

    if args.which in ("current", "all"):
        result["current"] = bench_current(device, storage_dtype=args.storage_dtype)
        for key in GRID:
            print(f"current {key}: {result['current'][key]['median']:.3f} s/s")
        print("current rates:", result["current"]["rates_hz"])

    if args.which in ("rate", "all"):
        result["rate"] = bench_rate(device)
        for backend, values in result["rate"].items():
            for key in GRID:
                print(f"rate {backend} {key}: {values[key]['median']:.3f} s/s")
            print(f"rate {backend} rates:", values["rates"])

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{args.which}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"Wrote {out}")
    if args.badges:
        if "inference" in result:
            sec = result["inference"]["wall_s_per_network_s"]["median"]
            write_badge("inference", "simulation", sec)
        if "training" in result:
            sec = result["training"]["wall_s_per_network_s"]["median"]
            write_badge("training", "training", sec)
