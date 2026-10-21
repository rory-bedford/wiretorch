"""The training loop for chunked spiking-network training.

A long recording is cut into chunks of time steps. The trainer runs the model through
the chunks in order (its state carries over), computes the losses of each chunk,
backpropagates, and updates the parameters every ``chunks_per_update`` chunks. When
the dataset is cycled, ``chunks_per_data_epoch`` marks where it starts again: model
and loss states are reset there (as when the data was generated), the learning-rate
scheduler steps, and the first ``burn_in_chunks`` chunks only rebuild state.

Vocabulary: a *chunk* is one iteration of the loop; a *data epoch* is one pass over
the dataset. (For compatibility, the constructor's ``num_epochs``, the checkpoints'
``epoch`` and the metrics CSV's ``epoch`` column count chunks.)

Checkpoints hold the model (and its simulation state), optimiser, scaler, scheduler,
random number generators, the losses' carried filter state, the recent loss history and
any gradients accumulated since the last update. ``resume`` restores all of it and
moves the data to the checkpoint's chunk, so a resumed run continues exactly as the
uninterrupted one would have, from any checkpoint (burn-in chunks included). The
"best" checkpoint is the one with the lowest mean loss over the chunks trained since
the previous checkpoint.
"""

from __future__ import annotations

from collections import deque
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.amp import autocast

from connectome_snns.snn_runners.chunks import (
    Prefetcher,
    load_loss_states,
    loss_states,
    model_outputs,
    reset_losses,
    warm_up_losses,
    weighted_losses,
)
from connectome_snns.training_utils import AsyncLogger, AsyncPlotter, save_checkpoint
from connectome_snns.training_utils.checkpointing import restore_checkpoint
from connectome_snns.training_utils.logging import save_figures

#: Recorded for logs and plots (trial 0 only), keyed by the name plots use.
PLOT_KEYS = (
    "spikes",
    "voltages",
    "currents",
    "currents_FF",
    "currents_leak",
    "conductances",
    "conductances_FF",
    "input_spikes",
    "target_spikes",
    "target_calcium",
    "target_rate",
    "predicted_calcium",
    "hidden_spikes",
)
#: Model output names of the tracked variables, by plot name.
TRACKED_AS = {
    "currents": ("currents", "currents_recurrent"),
    "currents_FF": ("currents_feedforward",),
    "conductances": ("conductances", "conductances_recurrent"),
    "conductances_FF": ("conductances_feedforward",),
}
SPIKE_KEYS = ("spikes", "target_spikes", "hidden_spikes")


class SNNTrainer:
    """Trains a spiking network chunk by chunk (see the module docstring).

    Args:
        model: The network. Compiled with ``torch.compile`` on CUDA.
        optimizer, scaler, scheduler: Optimiser, ``GradScaler`` and an optional
            learning-rate scheduler (stepped once per data epoch).
        dataloader: Iterable of named tuples (``SpikeData``): ``input_spikes`` and
            any of ``target_spikes``, ``target_calcium``, ``target_rate``.
        loss_functions, loss_weights: Losses by name, and their weights. Each loss
            declares its inputs in ``required_inputs`` (see ``chunks.loss_inputs``).
        device: 'cpu' or 'cuda'.
        num_epochs: Total number of chunks to train for.
        chunks_per_update: Chunks of gradient accumulated per optimiser step.
        log_interval: Log averaged losses, stats and gradient norms every this many
            chunks (to the metrics CSV and wandb).
        checkpoint_interval: Save a checkpoint (and plots) every this many chunks.
        plot_size: How many chunks before each log/checkpoint are kept for stats and
            plots (trial 0 only). Default 1.
        mixed_precision: Run forward and losses under autocast (CUDA only).
        grad_norm_clip: Clip the global gradient norm (ignored with ``grad_clip_fn``).
        grad_clip_fn: Custom clipping, ``fn(model)``, called before each step.
        wandb_config: Starts a wandb run in ``train`` if given (keys of
            ``wandb.init``, plus ``config``).
        progress_bar: A tqdm bar, advanced once per chunk.
        plot_generator: ``fn(spikes=..., voltages=..., ...) -> {name: figure}``. Only
            with one are voltages, currents and conductances tracked.
        stats_computer: ``fn(snapshot) -> {name: value}`` at each log; ``snapshot``
            holds the kept chunks (spikes, targets, ...) and the scaling factors and
            feedforward weights.
        connectome_mask, feedforward_mask: Where the recurrent and feedforward
            connections are, for losses and plots that ask for them.
        chunks_per_data_epoch: Chunks in one pass over the dataset.
        burn_in_chunks: Chunks after each data-epoch start that only rebuild state.
        epoch_callbacks: ``fn(data_epoch)`` at each data-epoch start.
        update_callbacks: ``fn()`` after each optimiser step.
        checkpoint_callbacks: ``fn(chunk, output_dir, model)`` after each checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        dataloader,
        loss_functions: dict[str, Callable],
        loss_weights: dict[str, float],
        device: str,
        num_epochs: int,
        chunks_per_update: int,
        log_interval: int,
        checkpoint_interval: int,
        plot_size: int | None = None,
        mixed_precision: bool = True,
        grad_norm_clip: float | None = None,
        wandb_config: dict[str, Any] | None = None,
        progress_bar: Any | None = None,
        plot_generator: Callable | None = None,
        stats_computer: Callable | None = None,
        connectome_mask: torch.Tensor | None = None,
        feedforward_mask: torch.Tensor | None = None,
        chunks_per_data_epoch: int | None = None,
        burn_in_chunks: int = 0,
        scheduler: Any | None = None,
        grad_clip_fn: Callable | None = None,
        epoch_callbacks: list[Callable] | None = None,
        update_callbacks: list[Callable] | None = None,
        checkpoint_callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.spike_iter = iter(dataloader)
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.device = device
        self.num_epochs = num_epochs
        self.chunks_per_update = chunks_per_update
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.plot_size = 1 if plot_size is None else plot_size
        self.mixed_precision = mixed_precision
        self.grad_norm_clip = grad_norm_clip
        self.grad_clip_fn = grad_clip_fn
        self.wandb_config = wandb_config
        self.pbar = progress_bar
        self.plot_generator = plot_generator
        self.stats_computer = stats_computer
        self.connectome_mask = connectome_mask
        self.feedforward_mask = feedforward_mask
        self.chunks_per_data_epoch = chunks_per_data_epoch
        self.burn_in_chunks = burn_in_chunks
        self.epoch_callbacks = epoch_callbacks or []
        self.update_callbacks = update_callbacks or []
        self.checkpoint_callbacks = checkpoint_callbacks or []

        if str(device).startswith("cuda"):
            try:
                self.model = torch.compile(model)
                print("  ✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠ torch.compile failed ({e}), running in eager mode")

        # Set by train() unless the caller set them first.
        self.metrics_logger = None
        self.wandb_logger = None
        self.async_plotter = None
        self._defined_wandb_metrics: set[str] = set()

        self.plot_accumulators = {
            key: deque(maxlen=self.plot_size) for key in PLOT_KEYS
        }
        self.loss_history: deque[dict[str, torch.Tensor]] = deque(maxlen=log_interval)
        self._since_checkpoint: list[dict[str, torch.Tensor]] = []
        self.accumulated_chunks = 0
        self.chunk = 0  # the next chunk to run
        self.best_loss = float("inf")
        self._gradient_norms: dict[str, torch.Tensor] = {}
        self._optimizer_has_stepped = False
        self._batches_to_skip = 0  # set on resuming: data already used
        self._batches: Prefetcher | None = None  # the data on the device, from train()
        self._pending_gradients: dict[str, torch.Tensor] = {}  # set on resuming

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def train(self, output_dir: Path | None = None) -> float:
        """Train from the current chunk to ``num_epochs``; return the best loss.

        With ``output_dir``, metrics, checkpoints and plots are written there.
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self._pending_gradients:
            for name, p in self._parameters():
                if name in self._pending_gradients:
                    p.grad = self._pending_gradients[name].to(p.device)
            self._pending_gradients = {}
        self._start_logging(output_dir)
        if self.chunk == 0 and output_dir:
            self._save_initial_checkpoint(output_dir)
        for _ in range(self._batches_to_skip):
            next(self.spike_iter)
        self._batches_to_skip = 0
        if self._batches is None:
            self._batches = Prefetcher(self.spike_iter, self.device)

        for chunk in range(self.chunk, self.num_epochs):
            self.chunk = chunk
            if self._starts_data_epoch(chunk):
                self._start_data_epoch(chunk)
            if self.pbar:
                self.pbar.update(1)

            if self._is_burn_in(chunk):
                self._burn_in()
            else:
                self._train_on_chunk()
                if self._is_due(chunk, self.chunks_per_update):
                    self._update_weights()
                if self._is_due(chunk, self.log_interval):
                    self._log(chunk)
            if output_dir and self._is_due(chunk, self.checkpoint_interval):
                self._checkpoint(chunk, output_dir)
            if self.async_plotter and self._is_due(chunk, self.log_interval * 10):
                if self.async_plotter.has_pending():  # safety net
                    print(f"  ⚠ Flushing pending plots at chunk {chunk + 1}...")
                    self.async_plotter.flush()

        if self.pbar:
            self.pbar.close()
        # The metrics logger is left open: callers may share it across runs.
        if self.async_plotter:
            print("Shutting down async plotter...")
            self.async_plotter.close()
        return self.best_loss

    def resume(self, checkpoint_path: Path | str) -> int:
        """Continue from a checkpoint file (or an output directory's latest one).

        Restores everything the checkpoint holds (see the module docstring) and returns
        the chunk training continues at. The dataloader must start at the beginning of
        the data, as a new one does: ``train`` skips the batches already used.
        """
        model = getattr(self.model, "_orig_mod", self.model)
        checkpoint = restore_checkpoint(
            checkpoint_path,
            model,
            self.optimizer,
            self.scaler,
            self.device,
            scheduler=self.scheduler,
        )
        self.set_checkpoint_state(
            checkpoint["epoch"], checkpoint.get("best_loss", float("inf"))
        )
        state = checkpoint.get("training_state", {})
        load_loss_states(self.loss_functions, state.get("loss_states", {}))
        self._pending_gradients = state.get("gradients", {})
        self.loss_history.extend(
            {name: value.to(self.device) for name, value in losses.items()}
            for losses in state.get("loss_history", [])
        )
        return self.chunk

    def set_checkpoint_state(self, epoch: int, best_loss: float) -> None:
        """Resume at chunk ``epoch`` (as stored in a checkpoint) with ``best_loss``.

        The model's simulation state is restored by ``load_checkpoint``; ``resume``
        does both, and restores the losses' state too. The dataloader is taken to
        start at the beginning of the data: ``train`` skips the batches already used.
        """
        self.chunk = epoch
        self.best_loss = best_loss
        # Each chunk, burn-in included, takes one batch; a cycled dataset starts again
        # at each data epoch.
        self._batches_to_skip = (
            epoch % self.chunks_per_data_epoch if self.chunks_per_data_epoch else epoch
        )
        # A checkpoint after training started means the optimizer has stepped, so the
        # scheduler must step at the next data-epoch boundary as it would have.
        self._optimizer_has_stepped = epoch > 0

    @property
    def current_epoch(self) -> int:
        """The current chunk (the name predates the chunk/data-epoch vocabulary)."""
        return self.chunk

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self.chunk = value

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    @staticmethod
    def _is_due(chunk: int, interval: int) -> bool:
        """Whether chunk ``chunk`` completes an ``interval``."""
        return (chunk + 1) % interval == 0

    def _starts_data_epoch(self, chunk: int) -> bool:
        return (
            self.chunks_per_data_epoch is not None
            and chunk % self.chunks_per_data_epoch == 0
        )

    def _is_burn_in(self, chunk: int) -> bool:
        if self.chunks_per_data_epoch is None or self.burn_in_chunks == 0:
            return False
        return chunk % self.chunks_per_data_epoch < self.burn_in_chunks

    def _keeps(self, chunk: int) -> bool:
        """Whether this chunk is within ``plot_size`` of the next log or checkpoint."""
        to_log = -(chunk + 1) % self.log_interval
        to_checkpoint = -(chunk + 1) % self.checkpoint_interval
        return min(to_log, to_checkpoint) < self.plot_size

    def _start_data_epoch(self, chunk: int) -> None:
        """Reset model and loss states (as the data was generated), step the scheduler."""
        if hasattr(self.model, "reset_state"):
            self.model.reset_state(batch_size=getattr(self.model, "batch_size", 1))
        reset_losses(self.loss_functions)
        if self.scheduler is not None and self._optimizer_has_stepped:
            self.scheduler.step()
        for callback in self.epoch_callbacks:
            callback(chunk // self.chunks_per_data_epoch)

    # ------------------------------------------------------------------
    # One chunk
    # ------------------------------------------------------------------

    @property
    def _masks(self) -> dict:
        return {
            "connectome_mask": self.connectome_mask,
            "feedforward_mask": self.feedforward_mask,
        }

    def _autocast(self):
        return autocast(
            device_type="cuda", enabled=self.mixed_precision and self.device == "cuda"
        )

    def _train_on_chunk(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """Forward, losses and backward for the next chunk; keep it if a log is near."""
        data = next(self._batches)
        keep = self._keeps(self.chunk)
        # Voltages, currents and conductances are only for plots, so only tracked with
        # a plot generator; losses and stats need just the spikes.
        track = keep and self.plot_generator is not None
        self.model.track_variables = track
        self.model.track_batch_idx = 0 if track else None

        with self._autocast():  # exp() of log-weights autocasts to float32
            chunk = {
                **data,
                **model_outputs(self.model.forward(input_spikes=data["input_spikes"])),
            }
            losses = weighted_losses(
                self.loss_functions, self.loss_weights, chunk, self.model, self._masks
            )
        self.scaler.scale(losses["total"] / self.chunks_per_update).backward()

        losses = {name: value.detach() for name, value in losses.items()}
        losses["total"] = losses["total"] / self.chunks_per_update
        self.loss_history.append(losses)
        self._since_checkpoint.append(losses)
        if keep:
            self._keep(chunk)
        return chunk, losses

    def _burn_in(self) -> None:
        """Run the next chunk forward only, to rebuild model and loss state."""
        data = next(self._batches)
        self.model.track_variables = False
        self.model.track_batch_idx = None
        with torch.no_grad():
            chunk = {
                **data,
                **model_outputs(self.model.forward(input_spikes=data["input_spikes"])),
            }
            warm_up_losses(self.loss_functions, chunk, self.model, self._masks)

    def _keep(self, chunk: dict) -> None:
        """Store trial 0 of this chunk's data and outputs for stats and plots."""
        for key in PLOT_KEYS:
            value = next(
                (
                    chunk[name]
                    for name in TRACKED_AS.get(key, (key,))
                    if chunk.get(name) is not None
                ),
                None,
            )
            if value is None:
                continue
            value = value[0:1].detach()
            if key in SPIKE_KEYS:
                value = value.bool()
            self.plot_accumulators[key].append(_to_host(value))
        self.accumulated_chunks = min(self.accumulated_chunks + 1, self.plot_size)

    def _parameters(self):
        """The model's named parameters (of the original module if compiled)."""
        return getattr(self.model, "_orig_mod", self.model).named_parameters()

    def _update_weights(self) -> None:
        """Clip, step the optimiser and scheduler state."""
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip_fn is not None:
            self.grad_clip_fn(self.model)
        elif self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_norm_clip
            )

        # Gradient norms for the next log, kept on the device until then.
        self._gradient_norms = {
            f"gradients/{name}_norm": p.grad.detach().norm()
            for name, p in self._parameters()
            if p.grad is not None
        }
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._optimizer_has_stepped = True
        for callback in self.update_callbacks:
            callback()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _start_logging(self, output_dir: Path | None) -> None:
        """Create the wandb run, metrics logger and plotter unless already present."""
        if self.wandb_config and self.wandb_logger is None:
            import wandb

            config = {
                **self.wandb_config.pop("config", {}),
                "num_epochs": self.num_epochs,
                "chunks_per_update": self.chunks_per_update,
                "log_interval": self.log_interval,
                "checkpoint_interval": self.checkpoint_interval,
                "mixed_precision": self.mixed_precision,
                "grad_norm_clip": self.grad_norm_clip,
                "output_dir": str(output_dir) if output_dir else None,
                "device": self.device,
            }
            name = self.wandb_config.pop("name", None) or (
                output_dir.name if output_dir else "snn_training"
            )
            print("\n" + "=" * 60)
            self.wandb_logger = wandb.init(
                name=name,
                config=config,
                dir=str(output_dir) if output_dir else None,
                **self.wandb_config,
            )
            wandb.watch(self.model, log="parameters", log_freq=self.log_interval)
            wandb.define_metric("epoch")
            print("=" * 60 + "\n")

        if output_dir and self.metrics_logger is None:
            self.metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)
        if self.plot_generator and output_dir and self.async_plotter is None:
            self.async_plotter = AsyncPlotter(
                plot_generator=self.plot_generator,
                output_dir=output_dir,
                wandb_logger=self.wandb_logger,
                max_queue_size=1,
                blocking_mode=False,
                chunks_per_data_epoch=self.chunks_per_data_epoch,
            )

    def _wandb_log(self, metrics: dict[str, float], chunk: int) -> None:
        """Log to wandb against the (fractional) data epoch, if a run is active."""
        if not self.wandb_logger:
            return
        import wandb

        for name in metrics:
            if name not in self._defined_wandb_metrics:
                wandb.define_metric(name, step_metric="epoch")
                self._defined_wandb_metrics.add(name)
        epoch = (
            chunk / self.chunks_per_data_epoch if self.chunks_per_data_epoch else chunk
        )
        wandb.log({"epoch": epoch, **metrics})

    def _average_losses(self) -> dict[str, float]:
        """Mean of each loss over the last ``log_interval`` trained chunks."""
        return _mean_losses(self.loss_history)

    def _log(self, chunk: int) -> None:
        """Averaged losses, stats and gradient norms to the CSV, wandb and the bar."""
        losses = self._average_losses()
        stats = {}
        if self.stats_computer:
            snapshot = self._plot_data().copy()
            snapshot["epoch"] = chunk
            for name in ("scaling_factors", "scaling_factors_FF", "weights_FF"):
                if hasattr(self.model, name):
                    snapshot[name] = getattr(self.model, name).detach().cpu().numpy()
            stats = self.stats_computer(snapshot)
        gradients = {name: float(norm) for name, norm in self._gradient_norms.items()}

        if self.metrics_logger:
            row = {f"{name}_loss": value for name, value in losses.items()}
            self.metrics_logger.log(epoch=chunk + 1, **row, **stats, **gradients)
        self._wandb_log(
            {**{f"loss/{name}": v for name, v in losses.items()}, **stats, **gradients},
            chunk + 1,
        )
        if self.pbar:
            self.pbar.set_postfix(_short_names(losses))

    def _plot_data(self) -> PlotData:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()  # the kept chunks' copies to the host
        return PlotData(self.plot_accumulators)

    # Kept for callers of the previous name.
    _concatenate_plot_data = _plot_data

    # ------------------------------------------------------------------
    # Checkpoints and plots
    # ------------------------------------------------------------------

    def _save_initial_checkpoint(self, output_dir: Path) -> None:
        """The untrained state, as chunk 0 (losses infinite)."""
        losses = {name: float("inf") for name in [*self.loss_functions, "total"]}
        save_checkpoint(
            output_dir=output_dir,
            epoch=0,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            best_loss=float("inf"),
            scheduler=self.scheduler,
            **losses,
        )
        self._wandb_log({f"loss/{name}": value for name, value in losses.items()}, 0)
        print(f"Initial checkpoint (epoch 0) saved to {output_dir / 'checkpoints'}")

    def _checkpoint(self, chunk: int, output_dir: Path) -> None:
        """Save a checkpoint (best if its mean loss is the lowest) and plots.

        The losses saved and compared are the means over the chunks trained since the
        previous checkpoint; with none (only burn-in since), they are infinite and
        there are no new plots.
        """
        if self.pbar:
            self.pbar.clear()
        print(
            f"\n{'=' * 60}\nCheckpoint at chunk {chunk + 1}/{self.num_epochs}\n{'=' * 60}"
        )
        trained = bool(self._since_checkpoint)
        if trained:
            losses = _mean_losses(self._since_checkpoint)
        else:
            losses = {name: float("inf") for name in [*self.loss_functions, "total"]}
        self._since_checkpoint = []
        if save_checkpoint(
            output_dir=output_dir,
            epoch=chunk + 1,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            best_loss=self.best_loss,
            scheduler=self.scheduler,
            training_state=self._training_state(),
            **losses,
        ):
            self.best_loss = losses["total"]
        for callback in self.checkpoint_callbacks:
            callback(chunk + 1, output_dir, self.model)
        if self.plot_generator and trained:
            self._plot(chunk, output_dir)

        for name, value in losses.items():
            label = "Total" if name == "total" else name.replace("_", " ").title()
            print(f"  {label} Loss: {value:.6f}")
        print("=" * 60)
        if self.pbar:
            self.pbar.refresh()

    def _training_state(self) -> dict:
        """What ``resume`` needs beyond model, optimiser and RNGs (on the CPU)."""

        def cpu(value):
            return value.detach().cpu() if isinstance(value, torch.Tensor) else value

        return {
            "loss_states": {
                key: cpu(value)
                for key, value in loss_states(self.loss_functions).items()
            },
            "loss_history": [
                {name: cpu(value) for name, value in losses.items()}
                for losses in self.loss_history
            ],
            "gradients": {
                name: cpu(p.grad)
                for name, p in self._parameters()
                if p.grad is not None
            },
        }

    def _plot(self, chunk: int, output_dir: Path) -> None:
        """Hand the kept chunks, weights and masks to the plot generator."""
        plot_data = dict(self._plot_data())

        def numpy(value):
            return None if value is None else value.detach().cpu().numpy()

        extras = {
            "weights": numpy(getattr(self.model, "weights", None)),
            "feedforward_weights": numpy(getattr(self.model, "weights_FF", None)),
            "connectome_mask": numpy(self.connectome_mask),
            "feedforward_mask": numpy(self.feedforward_mask),
        }
        if self.async_plotter:
            print("  📊 Submitting plot (will wait if queue is full)...")
            submitted = self.async_plotter.submit_plot(
                plot_data={**plot_data, **extras}, epoch=chunk, block=True, timeout=90.0
            )
            print(
                "  ✓ Plot job submitted"
                if submitted
                else "  ⚠ Plot submission timed out after 90s"
            )
        else:
            self._plot_now(plot_data, extras, chunk, output_dir)

    def _plot_now(
        self, plot_data: dict, extras: dict, chunk: int, output_dir: Path
    ) -> None:
        """Make and save the plots in this thread (no async plotter)."""
        print("  Generating plots...")
        figures_dir = save_figures(
            self.plot_generator,
            {**plot_data, "epoch": chunk, **extras},
            chunk,
            output_dir,
            log_images=self._wandb_images if self.wandb_logger else None,
        )
        print(f"  ✓ Saved plots to {figures_dir}")

    def _wandb_images(self, images: dict, chunk: int) -> None:
        self._wandb_log(images, chunk + 1)


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    """A host copy of ``tensor``; from CUDA without waiting for it (sync before reading)."""
    if tensor.device.type != "cuda":
        return tensor.cpu()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    return host.copy_(tensor, non_blocking=True)


def _mean_losses(history) -> dict[str, float]:
    """Mean of each loss over a sequence of per-chunk loss dicts."""
    names = {name for losses in history for name in losses}
    return {
        name: float(
            torch.stack(
                [losses.get(name, torch.zeros(())) for losses in history]
            ).mean()
        )
        for name in names
    }


def _short_names(losses: dict[str, float]) -> dict[str, str]:
    """Progress-bar labels: 'Total', then initials of the other losses."""
    out = {"Total": f"{losses['total']:.4f}"} if "total" in losses else {}
    for name, value in losses.items():
        if name != "total":
            words = name.split("_")
            label = (
                "".join(w[0].upper() for w in words)
                if len(words) > 1
                else name.upper()[:3]
            )
            out[label] = f"{value:.4f}"
    return out


class PlotData(MutableMapping):
    """Accumulated per-chunk arrays (or host tensors), concatenated along time into one
    array only when a key is read.

    Stats usually read only the spike trains, so the (much larger) tracked voltages,
    currents and conductances are never concatenated unless a plot needs them. Keys
    set on the mapping (``snapshot["epoch"] = ...``) are stored alongside.
    """

    def __init__(self, accumulators: dict[str, deque], extra=None, cache=None):
        self._accumulators = accumulators
        self._extra = dict(extra or {})
        self._cache = {} if cache is None else cache

    def __getitem__(self, key):
        if key in self._extra:
            return self._extra[key]
        if key not in self._cache:
            arrays = self._accumulators[key]
            self._cache[key] = (
                np.concatenate([np.asarray(a) for a in arrays], axis=1)
                if arrays
                else np.array([])
            )
        return self._cache[key]

    def __setitem__(self, key, value):
        self._extra[key] = value

    def __delitem__(self, key):
        del self._extra[key]

    def __iter__(self):
        yield from self._accumulators
        yield from (k for k in self._extra if k not in self._accumulators)

    def __len__(self):
        return len(set(self._accumulators) | set(self._extra))

    def copy(self) -> PlotData:
        return PlotData(self._accumulators, self._extra, self._cache)
