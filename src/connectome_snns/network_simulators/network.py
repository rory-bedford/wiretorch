"""The machinery every network family shares: connectivity, pathways and state.

A network of ``N`` neurons receives synaptic input from one or more *input sides*: its
own neurons (recurrent) and external inputs (feedforward). Connectivity on each side
is a dict of ``Projection`` objects, one per (source cell type, target cell type).

Internally the input is organised into **pathways** — one per source cell type that
has projections, on each side. A pathway has a single caching mode, shared by all its
projections, which decides what is computed when:

* ``"fixed"`` — weights fixed at construction (``FrozenProjection``).
* ``"scaling_factors"`` — the connectome is fixed at construction; each forward call
  scales the pathway's synaptic kernel by the current scaling factor of each
  target neuron's type (``ScalingFactorProjection``).
* ``"weights"`` — the weight matrix is assembled from the projections on every forward
  call, so gradients reach their parameters (``FullRankProjection``,
  ``LowRankProjection`` and subclasses).

What a family adds is its physiology: the per-neuron and per-synapse tables, the
synaptic state, and an ``engine`` whose ``simulate`` runs the time steps (each
family's ``dynamics.py`` in PyTorch, its ``kernels.py`` in Triton).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from connectome_snns.network_simulators.checkpoint_compat import drop_legacy_buffers
from connectome_snns.network_simulators.projections import Projection
from connectome_snns.network_simulators.synapses import SynapseRegistry
from connectome_snns.network_simulators.type_parameters import (
    TypeParameters,
    learnt_synapses,
    learnt_types,
)

PairKey = tuple[str, str]  # (source cell type name, target cell type name)
#: Dtypes the kernels can store the synaptic state and recurrent weights in.
STORAGE_DTYPES = (torch.float32, torch.float16)


@dataclass
class InputSide:
    """One source of synaptic input and its connectivity.

    Attributes:
        projections: ``{(source_name, target_name): Projection}``.
        cell_params, synapse_params: The source cell types and the synapses they make.
        cell_type_indices: Cell type of each source (each input row).
        recurrent: Whether the sources are the network's own neurons.
        attribute: Name of the ``nn.ModuleDict`` holding the projections.
        mask_buffer: Name of the buffer holding the side's connectivity mask.
    """

    projections: dict[PairKey, Projection]
    cell_params: list[dict]
    synapse_params: list[dict]
    cell_type_indices: np.ndarray
    recurrent: bool
    attribute: str
    mask_buffer: str


@dataclass(frozen=True)
class Pathway:
    """One source cell type on one input side (see the module docstring)."""

    index: int
    side: int
    cell_type: int
    mode: str  # "fixed" | "scaling_factors" | "weights"


@dataclass
class Drive:
    """The synaptic drive of one forward call, pathway by pathway.

    A *pathway* is one source cell type on one input side. Recurrent pathways come
    first; each lists its source neurons (``rows`` of the network's own output) and its
    ``(n_rows, N)`` weights, and likewise feedforward pathways for the input columns.

    ``made`` ``(K, S)`` marks the synapse types each pathway makes; ``scale`` ``(K, N)``
    multiplies its kernel per target neuron (a scaling factor, or 1).
    """

    made: torch.Tensor
    scale: torch.Tensor
    recurrent: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    feedforward: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    #: Input columns that can carry a gradient (None: any); the rest are data.
    differentiable_inputs: slice | None = None

    def feedforward_activations(self, x: torch.Tensor) -> torch.Tensor:
        """``(K_ff, B, T, N)``: each feedforward pathway's input times its weights."""
        B, T = x.shape[:2]
        if not self.feedforward:
            return x.new_zeros((0, B, T, 0))
        return torch.stack([x[:, :, rows] @ W for rows, W in self.feedforward])


@dataclass
class Run:
    """What a simulation produced.

    ``output`` is the network's output per step, ``(B, T, N)``: spikes, or rates for a
    rate network. ``v`` and ``syn`` are the state after the last step. ``tracked`` (if
    tracking) holds per-step variables of the tracked trial(s), named by the family's
    dynamics.
    """

    output: torch.Tensor
    v: torch.Tensor
    syn: torch.Tensor
    probabilities: torch.Tensor | None = None
    tracked: dict[str, torch.Tensor] | None = None

    @property
    def spikes(self) -> torch.Tensor:
        """The output of a spiking network."""
        return self.output

    @property
    def g(self) -> torch.Tensor:
        """The conductances (the former name of ``syn``)."""
        return self.syn


class ProjectionNetwork(nn.Module):
    """Base class of the networks: connectivity organised into pathways, and the state.

    Subclasses describe their input sides, register their physiology tables
    (``_register_physiology``), derive what depends on the time step
    (``_set_timestep``), and supply the engine and its parameters (``_engine``,
    ``_physiology``). The state — ``v`` ``(B, N)`` and the synaptic state named
    ``state_name`` ``(B, N, ..., S)`` — continues from one ``forward`` call to the next
    until ``reset_state``.
    """

    #: Name of the synaptic state (also its checkpoint key).
    state_name = "syn"
    #: Name of the per-step output in ``get_tracked_gradients`` (``grad_<name>``).
    output_name = "s"
    #: The module whose ``simulate`` runs the time steps.
    _engine = None

    def __init__(
        self,
        *,
        dt: float,
        cell_params: list[dict],
        cell_type_indices: np.ndarray,
        sides: list[InputSide],
        unify_synapses: bool,
        batch_size: int,
        track_variables: bool,
        track_gradients: bool,
        track_batch_idx: int | None,
        storage_dtype: torch.dtype | str = torch.float32,
    ):
        super().__init__()
        self.register_load_state_dict_pre_hook(drop_legacy_buffers)
        validate(dt, cell_params, cell_type_indices, sides)
        self.batch_size = batch_size
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
        self.storage_dtype = storage_dtype
        #: Input columns that can carry a gradient — e.g. another layer's spikes —
        #: as a slice; None means any. Lets the backward pass skip pure data columns.
        self.differentiable_inputs = None
        self.cell_params = cell_params
        self.n_cell_types = len(cell_params)
        self.n_neurons = int(cell_type_indices.shape[0])
        self._sides = sides
        self._names = {c["cell_id"]: c["name"] for c in cell_params}

        self.registry = SynapseRegistry.build(
            [(s.cell_params, s.synapse_params) for s in sides], unify=unify_synapses
        )
        self.register_buffer("cell_type_indices", _long(cell_type_indices))
        for side in sides:
            setattr(
                self,
                side.attribute,
                nn.ModuleDict(
                    {f"{s}__{t}": p for (s, t), p in side.projections.items()}
                ),
            )
            self.register_buffer(side.mask_buffer, torch.from_numpy(self._mask(side)))
        self._register_physiology()
        self.pathways = self._pathways()
        self.set_timestep(dt)

    # ------------------------------------------------------------------
    # What a family provides
    # ------------------------------------------------------------------

    def _register_physiology(self) -> None:
        """Register the per-neuron and per-synapse tables (called once, at init)."""
        raise NotImplementedError

    def _set_timestep(self) -> None:
        """Register what depends on ``self.dt`` (decay factors, kernels)."""
        raise NotImplementedError

    def _physiology(self):
        """The engine's parameters for one forward call."""
        raise NotImplementedError

    def _initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(v, syn)`` of a network at rest, for ``self.batch_size`` trials."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def storage_dtype(self) -> torch.dtype:
        """What the CUDA kernels keep the synaptic state and recurrent weights in.

        ``torch.float16`` (or ``"float16"``) nearly halves the memory traffic of a
        recurrent step and halves the state saved for the backward pass; arithmetic
        stays in float32, and gradients are float32. The run is then a different
        realisation of the same network: on the conductance-based teacher, firing
        rates agree with float32 within 0.4% and gradients point the same way (cosine
        0.9999), but individual spike times part after ~100 ms. Rounding happens
        wherever the state is stored, so splitting a feedforward network's simulation
        into chunks is no longer exactly neutral.

        Default float32. The PyTorch reference (CPU) always computes in float32.
        """
        return self._storage_dtype

    @storage_dtype.setter
    def storage_dtype(self, dtype: torch.dtype | str) -> None:
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"storage_dtype must be one of {STORAGE_DTYPES}, got {dtype}"
            )
        self._storage_dtype = dtype

    def _mask(self, side: InputSide) -> np.ndarray:
        """``(n_sources, N)`` bool: where the side's projections have connections."""
        mask = np.zeros((len(side.cell_type_indices), self.n_neurons), dtype=bool)
        for (src, tgt), projection in side.projections.items():
            rows, cols = self._block(side, src, tgt)
            if rows.size and cols.size:
                mask[np.ix_(rows, cols)] = projection_mask(projection)
        return mask

    def _block(self, side: InputSide, src: str, tgt: str):
        """Rows (sources) and columns (targets) of the block of pair ``(src, tgt)``."""
        src_id = next(c["cell_id"] for c in side.cell_params if c["name"] == src)
        tgt_id = next(c["cell_id"] for c in self.cell_params if c["name"] == tgt)
        rows = np.flatnonzero(side.cell_type_indices == src_id)
        cols = np.flatnonzero(np.asarray(self.cell_type_indices.cpu()) == tgt_id)
        return rows, cols

    def _per_neuron(self, name: str, default: float | None = None) -> torch.Tensor:
        """``(N,)`` float32: a cell parameter of each neuron (``default`` if unset)."""
        lookup = np.zeros(self.n_cell_types, dtype=np.float32)
        for cell in self.cell_params:
            lookup[cell["cell_id"]] = (
                cell[name] if default is None else cell.get(name, default)
            )
        return torch.from_numpy(lookup[np.asarray(self.cell_type_indices.cpu())])

    def _register_learnt(
        self,
        learn: dict,
        cell: dict[str, float | None],
        synapse: tuple[str, ...],
        positive: set[str],
    ) -> None:
        """Parameters for the physiology ``learn`` names, initialised from the config.

        ``learn`` maps a cell parameter (a key of ``cell``, with its default, None if
        required) or a synapse parameter (in ``synapse``) to the types to learn: True for
        all, or their names (cell types; synapse names). Learnt cell parameters live in
        ``cell_parameters``, learnt synapse parameters in ``synapse_parameters`` (both
        ``TypeParameters``; ``positive`` ones as logarithms); ``_learnt`` puts them in
        place of the fixed per-neuron (or per-synapse) values.
        """
        unknown = set(learn) - set(cell) - set(synapse)
        assert not unknown, (
            f"learn: unknown parameters {sorted(unknown)}; "
            f"can learn {(*cell, *synapse)}"
        )
        cells = sorted(self.cell_params, key=lambda c: c["cell_id"])
        names = [c["name"] for c in cells]
        learnt = {
            name: learnt_types(learn[name], names, f"learn[{name!r}]")
            for name in cell
            if learn.get(name)
        }
        if learnt:
            self.cell_parameters = TypeParameters(
                {
                    name: [
                        c[name] if cell[name] is None else c.get(name, cell[name])
                        for c in cells
                    ]
                    for name in learnt
                },
                learnt=learnt,
                positive=positive & set(learnt),
            )
            for name, types in learnt.items():
                mask = torch.isin(self.cell_type_indices, torch.tensor(types))
                self.register_buffer(f"_learnt_{name}", mask, persistent=False)
        synapses = self.registry.synapses
        learnt = {
            name: learnt_synapses(learn[name], synapses, f"learn[{name!r}]")
            for name in synapse
            if learn.get(name)
        }
        if learnt:
            self.synapse_parameters = TypeParameters(
                {name: self.registry.table(name).tolist() for name in learnt},
                learnt=learnt,
                positive=positive & set(learnt),
            )
            for name, types in learnt.items():
                mask = torch.zeros(len(synapses), dtype=torch.bool)
                mask[types] = True
                self.register_buffer(f"_learnt_{name}", mask, persistent=False)
        self._synapse_learnables = synapse

    def _learnt(self, name: str, fixed: torch.Tensor, decay: bool = False):
        """``fixed`` (per neuron, or per synapse for a synapse parameter), with the
        learnt types' values in place; ``decay`` takes a time constant to its per-step
        decay ``exp(-dt / tau)`` (``fixed`` is then the decay)."""
        mask = getattr(self, f"_learnt_{name}", None)
        if mask is None:
            return fixed
        if name in self._synapse_learnables:
            value = self.synapse_parameters.value(name)
        else:
            value = self.cell_parameters.value(name)[self.cell_type_indices]
        if decay:
            value = torch.exp(-self.dt / value)
        return torch.where(mask, value, fixed)

    def _learns(self, *names: str) -> bool:
        """Whether any of ``names`` is learnt."""
        return any(hasattr(self, f"_learnt_{name}") for name in names)

    def _pathways(self) -> list[Pathway]:
        """One pathway per source cell type with synapses and projections, per side."""
        pathways, made = [], []
        for side_index, side in enumerate(self._sides):
            names = {c["cell_id"]: c["name"] for c in side.cell_params}
            modes = {}
            for (src, _), projection in side.projections.items():
                previous = modes.setdefault(src, projection.caching_mode)
                if previous != projection.caching_mode:
                    raise ValueError(
                        f"source cell type {src!r} has projections with mixed caching "
                        f"modes ({previous!r} and {projection.caching_mode!r}); all "
                        "projections from one source must use the same mode"
                    )
            for cell_type, synapses in self.registry.made_by(
                side_index, side.synapse_params
            ).items():
                mode = modes.get(names[cell_type])
                if mode is None:
                    continue  # no projections: contributes nothing
                pathway = Pathway(len(pathways), side_index, cell_type, mode)
                rows = np.flatnonzero(side.cell_type_indices == cell_type)
                self.register_buffer(
                    f"_rows_{pathway.index}", _long(rows), persistent=False
                )
                pathways.append(pathway)
                made.append(synapses)
        self.register_buffer(
            "_made",
            torch.stack(made)
            if made
            else torch.zeros((0, len(self.registry)), dtype=torch.bool),
            persistent=False,
        )
        # Recurrent pathways first: the order the dynamics inject them in.
        return sorted(pathways, key=lambda p: not self._sides[p.side].recurrent)

    def _assemble(self, pathway: Pathway, connectome: bool = False) -> torch.Tensor:
        """The ``(n_rows, N)`` weights of a pathway, from its projections' ``forward``.

        With ``connectome=True``, scaling-factor projections contribute their unscaled
        connectome instead (the scaling factor is applied to the kernel).
        """
        side = self._sides[pathway.side]
        rows = getattr(self, f"_rows_{pathway.index}")
        W = torch.zeros((rows.numel(), self.n_neurons), **self._float)
        projections = getattr(self, side.attribute)
        src = next(
            c["name"] for c in side.cell_params if c["cell_id"] == pathway.cell_type
        )
        for tgt_id, tgt in self._names.items():
            key = f"{src}__{tgt}"
            if key not in projections:
                continue
            cols = (self.cell_type_indices == tgt_id).nonzero(as_tuple=True)[0]
            if cols.numel():
                projection = projections[key]
                W[:, cols] = projection.connectome if connectome else projection()
        return W

    def _cache_weights(self) -> None:
        """Snapshot the weights of fixed and scaling-factor pathways."""
        for pathway in self.pathways:
            if pathway.mode != "weights":
                W = self._assemble(
                    pathway, connectome=pathway.mode == "scaling_factors"
                )
                self.register_buffer(
                    f"_weights_{pathway.index}", W.detach(), persistent=False
                )

    def set_timestep(self, dt: float) -> None:
        """Set the time step, recompute what depends on it, and reset the state."""
        assert isinstance(dt, float), "dt must be a float"
        assert dt > 0, "Timestep dt must be positive"
        self.register_buffer("dt", torch.tensor(dt, device=self.device))
        self._set_timestep()
        self._cache_weights()
        self.reset_state()

    # ------------------------------------------------------------------
    # Parameters as matrices
    # ------------------------------------------------------------------

    def _weight_matrix(self, side_index: int) -> torch.Tensor:
        """``(n_sources, N)`` weights of a side, assembled from its projections."""
        side = self._sides[side_index]
        W = torch.zeros((len(side.cell_type_indices), self.n_neurons), **self._float)
        for (src, tgt), projection in side.projections.items():
            rows, cols = self._block(side, src, tgt)
            if rows.size and cols.size:
                W[np.ix_(rows, cols)] = getattr(self, side.attribute)[f"{src}__{tgt}"]()
        return W

    def _scaling_matrix(self, side_index: int) -> torch.Tensor:
        """``(n_source_types, n_target_types)`` scaling factors of a side (1 if none)."""
        side = self._sides[side_index]
        ids = {c["name"]: c["cell_id"] for c in side.cell_params}
        tgt_ids = {c["name"]: c["cell_id"] for c in self.cell_params}
        sf = torch.ones((len(side.cell_params), self.n_cell_types), **self._float)
        for key, projection in getattr(self, side.attribute).items():
            if projection.caching_mode == "scaling_factors":
                src, tgt = key.split("__", 1)
                sf[ids[src], tgt_ids[tgt]] = projection.kernel_factor()
        return sf

    @property
    def device(self) -> torch.device:
        return self.cell_type_indices.device

    @property
    def _float(self) -> dict:
        """Device and float dtype of the model (float32 unless converted, e.g. by
        ``.double()``), for the tensors it assembles."""
        return {"device": self.device, "dtype": self.dt.dtype}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _drive(self) -> Drive:
        """The pathways' current weights and kernel scales, for one forward call."""
        scaling = {}
        ones = torch.ones(self.n_neurons, **self._float)
        scales, recurrent, feedforward = [], [], []
        for pathway in self.pathways:
            if pathway.mode == "weights":
                W = self._assemble(pathway)
            else:
                W = getattr(self, f"_weights_{pathway.index}")
            if pathway.mode == "scaling_factors":
                if pathway.side not in scaling:
                    scaling[pathway.side] = self._scaling_matrix(pathway.side)
                scales.append(
                    scaling[pathway.side][pathway.cell_type, self.cell_type_indices]
                )
            else:
                scales.append(ones)
            rows = getattr(self, f"_rows_{pathway.index}")
            (recurrent if self._sides[pathway.side].recurrent else feedforward).append(
                (rows, W)
            )
        order = [p.index for p in self.pathways]
        return Drive(
            made=self._made[order],
            scale=torch.stack(scales)
            if scales
            else ones.new_zeros((0, self.n_neurons)),
            recurrent=recurrent,
            feedforward=feedforward,
            differentiable_inputs=self.differentiable_inputs,
        )

    # The time loop runs in the engine (Triton kernels on CUDA): there is nothing for
    # torch.compile to fuse here, so a compiled model calls this eagerly.
    @torch.compiler.disable
    def _simulate(self, input_spikes, probabilities: bool = False) -> Run:
        """Validate the input, simulate it from the current state, and keep the state."""
        if isinstance(input_spikes, np.ndarray):
            input_spikes = torch.from_numpy(input_spikes).to(self.device)
        self._validate_forward(input_spikes)
        retain = [] if self.track_gradients else None
        run = self._engine.simulate(
            self.v,
            self._synaptic,
            input_spikes.to(self.dt.dtype),
            self._drive(),
            self._physiology(),
            probabilities=probabilities,
            track=self.track_variables,
            track_batch_idx=self.track_batch_idx,
            retain=retain,
        )
        if retain is not None:
            self._retained = retain
        # Detach: the next chunk must not backpropagate into this one's graph.
        self.v = run.v.detach()
        self._synaptic = run.syn.detach()
        return run

    def _validate_forward(self, input_spikes: torch.Tensor) -> None:
        assert isinstance(input_spikes, torch.Tensor), "input_spikes must be a Tensor"
        assert input_spikes.device == self.device, (
            f"input_spikes on {input_spikes.device}, model on {self.device}"
        )
        assert input_spikes.ndim == 3, (
            f"input_spikes must be 3D (batch, time, n_inputs), got {input_spikes.shape}"
        )
        assert input_spikes.shape[2] == self.n_inputs, (
            f"input_spikes shape[2]={input_spikes.shape[2]}, expected {self.n_inputs}"
        )
        if input_spikes.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size {input_spikes.shape[0]} != model batch size "
                f"{self.batch_size}; call reset_state(batch_size=...) first"
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def _synaptic(self) -> torch.Tensor:
        return getattr(self, self.state_name)

    @_synaptic.setter
    def _synaptic(self, value: torch.Tensor) -> None:
        setattr(self, self.state_name, value)

    def reset_state(self, batch_size: int | None = None) -> None:
        """Back to rest (see ``_initial_state``); optionally change the batch size."""
        if batch_size is not None:
            self.batch_size = batch_size
        v, syn = self._initial_state()
        self.register_buffer("v", v, persistent=False)
        self.register_buffer(self.state_name, syn, persistent=False)

    def get_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {
            "v": self.v.cpu().numpy(),
            self.state_name: self._synaptic.cpu().numpy(),
        }

    def load_checkpoint_state(self, state: dict[str, np.ndarray]) -> None:
        self.v = torch.from_numpy(state["v"]).to(self.device)
        self._synaptic = torch.from_numpy(state[self.state_name]).to(self.device)

    def get_tracked_gradients(self) -> dict[str, torch.Tensor]:
        """Gradients of the tracked per-step ``v``, synaptic state and output.

        Call after ``backward()`` on a loss from a forward pass with
        ``track_gradients=True``. Shapes ``(T, B, ...)``; zeros where no gradient.
        """
        if not self.track_gradients:
            raise RuntimeError("track_gradients must be True during forward pass")
        if not getattr(self, "_retained", None):
            raise RuntimeError(
                "No tracked tensors found. Did you call forward() with track_gradients=True?"
            )
        names = ("grad_v", f"grad_{self.state_name}", f"grad_{self.output_name}")
        return {
            name: torch.stack(
                [t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t)
                 for t in tensors]
            )
            for name, tensors in zip(names, zip(*self._retained))
        }  # fmt: skip


# =====================================================================
# Helpers
# =====================================================================


def _long(array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).long()


def projection_mask(projection: Projection) -> np.ndarray:
    """Where a projection has connections, as a bool array (at construction)."""
    for name in ("connectome", "mask", "_weights"):
        tensor = getattr(projection, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy() != 0
    raise TypeError(f"Unknown Projection subclass: {type(projection).__name__}")


def check_surrgrad_scale(surrgrad_scale) -> None:
    assert isinstance(surrgrad_scale, (int, float)), "surrgrad_scale must be numeric"
    assert surrgrad_scale > 0, "surrgrad_scale must be positive"


def check_parameters(params: list[dict], required: tuple[str, ...], what: str) -> None:
    """Every entry of ``params`` has every ``required`` key."""
    for entry in params:
        for name in required:
            assert name in entry, f"{what}: entry {entry} is missing {name!r}"


def validate(dt, cell_params, cell_type_indices, sides) -> None:
    """Check the construction arguments, with messages naming what is wrong."""
    assert isinstance(dt, (int, float)), "dt must be numeric"
    assert dt > 0, f"dt must be positive, got {dt}"
    assert dt <= 100.0, f"dt unusually large ({dt} ms), check units"

    def contiguous_ids(params, key, what):
        ids = sorted(p[key] for p in params)
        assert ids == list(range(len(ids))), (
            f"{what} must be 0-indexed contiguous, got {ids}"
        )

    contiguous_ids(cell_params, "cell_id", "cell_ids in cell_params")
    assert cell_type_indices.ndim == 1, "cell_type_indices must be 1D"
    assert cell_type_indices.shape[0] > 0, "n_neurons must be positive"
    assert np.all((cell_type_indices >= 0) & (cell_type_indices < len(cell_params))), (
        "cell_type_indices out of range"
    )
    targets = {p["name"]: p["cell_id"] for p in cell_params}
    n_per_target = np.bincount(cell_type_indices, minlength=len(cell_params))

    for side in sides:
        contiguous_ids(side.cell_params, "cell_id", f"{side.attribute} source cell_ids")
        assert len(side.synapse_params) > 0, (
            f"{side.attribute}: synapse_params must not be empty"
        )
        contiguous_ids(
            side.synapse_params, "synapse_id", f"{side.attribute} synapse_ids"
        )
        types = side.cell_type_indices
        assert types.ndim == 1, f"{side.attribute}: source cell types must be 1D"
        assert np.all((types >= 0) & (types < len(side.cell_params))), (
            f"{side.attribute}: source cell type indices out of range"
        )
        sources = {p["name"]: p["cell_id"] for p in side.cell_params}
        n_per_source = np.bincount(types, minlength=len(side.cell_params))
        for (src, tgt), projection in side.projections.items():
            assert src in sources, (
                f"{side.attribute}: source {src!r} not in its cell_params"
            )
            assert tgt in targets, (
                f"{side.attribute}: target {tgt!r} not in cell_params"
            )
            expected = (
                int(n_per_source[sources[src]]),
                int(n_per_target[targets[tgt]]),
            )
            assert (projection.n_source, projection.n_target) == expected, (
                f"{side.attribute} projection ({src!r}, {tgt!r}) shape "
                f"({projection.n_source}, {projection.n_target}) != expected {expected}"
            )
