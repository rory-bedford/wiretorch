"""
Asynchronous logging utilities for training metrics and visualizations.

This module provides thread-safe asynchronous logging for CSV metrics and plot
generation, allowing training loops to continue without waiting for I/O operations.
"""

import csv
import threading
import time
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Any, Optional, Dict, Callable

import numpy as np


#: Plot inputs that describe the model (copied when a plot is submitted), as opposed
#: to the recorded activity.
MODEL_KEYS = (
    "weights",
    "feedforward_weights",
    "connectome_mask",
    "feedforward_mask",
    "scaling_factors",
    "scaling_factors_FF",
)


def accepted_kwargs(fn: Callable, kwargs: dict) -> dict:
    """The ``kwargs`` that ``fn`` takes (all of them if it has ``**kwargs``)."""
    import inspect

    parameters = inspect.signature(fn).parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return kwargs
    names = {p.name for p in parameters}
    return {k: v for k, v in kwargs.items() if k in names}


def save_figures(
    plot_generator: Callable,
    plot_kwargs: dict,
    epoch: int,
    output_dir: str | Path,
    log_images: Optional[Callable] = None,
) -> Path:
    """Make figures with ``plot_generator`` and save them as PNGs.

    The generator gets the ``plot_kwargs`` it accepts and returns ``{name: figure}``;
    each is saved to ``output_dir/figures/chunk_{epoch + 1:06d}/{name}.png``, handed
    to ``log_images({"plots/name": wandb.Image, ...}, epoch)`` if given, and closed.
    Returns the figures directory.
    """
    from matplotlib import pyplot as plt

    figures = plot_generator(**accepted_kwargs(plot_generator, plot_kwargs))
    figures_dir = Path(output_dir) / "figures" / f"chunk_{epoch + 1:06d}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    if log_images is not None:
        import wandb

        log_images(
            {f"plots/{name}": wandb.Image(fig) for name, fig in figures.items()},
            epoch,
        )
    for fig in figures.values():
        plt.close(fig)
    return figures_dir


def _column_order(names) -> list[str]:
    """CSV columns: ``epoch`` first, the rest sorted."""
    return sorted(names, key=lambda name: (name != "epoch", name))


class AsyncLogger:
    """Asynchronous logger for training metrics.

    This logger uses a background thread to handle CSV I/O operations without blocking
    the training loop. Each metric entry is written to disk immediately when processed.

    Args:
        log_dir (str | Path): Directory where log files will be saved
        max_queue_size (int): Maximum queue size (blocks when full for backpressure)

    Example:
        >>> logger = AsyncLogger(log_dir='training_logs')
        >>> for step in range(1000):
        ...     loss, accuracy = train_step()
        ...     logger.log(epoch=step, loss=loss, accuracy=accuracy)
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        max_queue_size: int = 1,
    ):
        """Initialize the async logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.queue: Queue = Queue(maxsize=max_queue_size)

        # CSV file tracking
        self.csv_path = self.log_dir / "training_metrics.csv"
        self.csv_fieldnames: Optional[list[str]] = None

        # Load existing fieldnames from CSV on disk (for resume)
        if self.csv_path.exists():
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self.csv_fieldnames = list(reader.fieldnames)

        # Start worker thread
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        """Background worker thread that writes data to disk."""
        while True:
            try:
                item = self.queue.get()
                if item is None:
                    break

                item_type, data = item

                if item_type == "csv_row":
                    self._write_csv_row(data)
                elif item_type == "weights":
                    self._save_weights(data)

                self.queue.task_done()
            except Exception:
                # Daemon threads fail silently, so we need to handle errors gracefully
                import traceback

                traceback.print_exc()
                self.queue.task_done()

    def _write_csv_row(self, data: dict[str, Any]):
        """Append a row; if it brings new columns, rewrite the file with them.

        Columns are sorted with ``epoch`` first. Rows written before a column existed
        get it blank.
        """
        columns = _column_order(set(data) | set(self.csv_fieldnames or ()))
        if columns == self.csv_fieldnames:
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=columns).writerow(data)
            return

        rows = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                columns = _column_order(set(columns) | set(reader.fieldnames or ()))
        self.csv_fieldnames = columns
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([*rows, data])

    def _save_weights(self, data: dict[str, Any]):
        """``weights/{kind}_weights_epoch_{epoch:06d}.npz`` for each kind given."""
        weights_dir = self.log_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        for kind in ("recurrent", "feedforward"):
            if f"{kind}_weights" in data:
                path = weights_dir / f"{kind}_weights_epoch_{data['epoch']:06d}.npz"
                np.savez_compressed(path, weights=data[f"{kind}_weights"])

    def log(self, epoch: int, **metrics: float):
        """Log metrics for a given epoch."""
        data = {"epoch": epoch, **metrics}
        self.queue.put(("csv_row", data), block=True)

    def save_weights(
        self,
        epoch: int,
        recurrent_weights: np.ndarray,
        feedforward_weights: Optional[np.ndarray] = None,
    ):
        """Save network weights to disk asynchronously (``weights/`` in the log
        directory). Not used by the trainer; kept for scripts that snapshot weights."""
        data = {"epoch": epoch, "recurrent_weights": recurrent_weights}
        if feedforward_weights is not None:
            data["feedforward_weights"] = feedforward_weights
        self.queue.put(("weights", data), block=True)

    def close(self):
        """Close the logger and wait for all pending writes."""
        self.queue.put(None, block=True)
        self.worker.join(timeout=10.0)


class AsyncPlotter:
    """Asynchronous plotter for training visualizations.

    This plotter uses background threads to handle CPU-intensive plotting operations
    without blocking the training loop. Plot jobs are queued and processed asynchronously,
    with automatic cleanup and graceful shutdown.

    Args:
        plot_generator (Callable): Function that generates matplotlib figures
        output_dir (str | Path): Base directory where plots will be saved
        wandb_logger (Optional[Any]): Wandb logger for uploading plots (optional)
        max_queue_size (int): Maximum number of pending plot jobs
        blocking_mode (bool): If True, run plots synchronously for debugging (default: False)
        chunks_per_data_epoch (Optional[int]): Number of chunks per data epoch for fractional epoch calculation in wandb

    Example:
        >>> plotter = AsyncPlotter(my_plot_function, output_dir='./plots')
        >>> for epoch in range(1000):
        ...     activity_data = copy_from_gpu(train_step())  # Copy to numpy first
        ...     plotter.submit_plot(activity_data, epoch, weights.cpu().numpy())
        >>> plotter.close()
    """

    def __init__(
        self,
        plot_generator: Callable,
        output_dir: str | Path,
        wandb_logger: Optional[Any] = None,
        max_queue_size: int = 3,
        blocking_mode: bool = False,
        chunks_per_data_epoch: Optional[int] = None,
    ):
        """Initialize the async plotter.

        Args:
            plot_generator (Callable): Function that generates matplotlib figures
            output_dir (str | Path): Base directory where plots will be saved
            wandb_logger (Optional[Any]): Wandb logger for uploading plots
            max_queue_size (int): Maximum number of pending plot jobs
            blocking_mode (bool): If True, run plots synchronously for debugging
            chunks_per_data_epoch (Optional[int]): Number of chunks per data epoch for fractional epoch calculation
        """
        self.plot_generator = plot_generator
        self.output_dir = Path(output_dir)
        self.wandb_logger = wandb_logger
        self.blocking_mode = blocking_mode
        self.chunks_per_data_epoch = chunks_per_data_epoch

        if self.blocking_mode:
            # Don't initialize queue or thread in blocking mode
            return

        # Thread-safe queue for plot jobs
        self.plot_queue: Queue = Queue(maxsize=max_queue_size)
        self.shutdown_event = threading.Event()

        # Start background plotting thread
        self.plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        self.plot_thread.start()

    def submit_plot(
        self,
        plot_data: Dict[str, np.ndarray],
        epoch: int,
        block: bool = True,
        timeout: Optional[float] = 90.0,
    ) -> bool:
        """Submit a plotting job to the background thread.

        Args:
            plot_data (Dict[str, np.ndarray]): Dictionary containing activity data and weights as numpy arrays.
                Should include keys: spikes, voltages, currents, etc., plus:
                - weights: Current network weights
                - feedforward_weights: Current feedforward weights
                - connectome_mask (optional): Binary mask for recurrent connections
                - feedforward_mask (optional): Binary mask for feedforward connections
                - scaling_factors (optional): Recurrent scaling factors
                - scaling_factors_FF (optional): Feedforward scaling factors
                The dict is not modified; the model arrays are copied, so training
                can go on updating them.
            epoch (int): Current epoch number
            block (bool): If True, wait for space in queue. If False, skip if queue full.
            timeout (Optional[float]): Maximum time to wait if blocking (None = wait forever)

        Returns:
            bool: True if job was queued, False if queue full and not blocking or timeout
        """
        plot_job = {
            "plot_data": {k: v for k, v in plot_data.items() if k not in MODEL_KEYS},
            "model": {
                key: np.copy(plot_data[key])
                for key in MODEL_KEYS
                if plot_data.get(key) is not None
            },
            "epoch": epoch,
            "timestamp": time.time(),
        }

        if self.blocking_mode:  # plot now, in this thread
            try:
                self._process_plot_job(plot_job)
                return True
            except Exception:
                import traceback

                traceback.print_exc()
                return False

        try:
            if block:
                self.plot_queue.put(plot_job, block=True, timeout=timeout)
            else:
                self.plot_queue.put_nowait(plot_job)
            return True
        except Full:  # timed out, or full and not blocking
            return False

    def _plot_worker(self) -> None:
        """Background worker that processes plotting jobs."""
        while not self.shutdown_event.is_set():
            try:
                job = self.plot_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                self._process_plot_job(job)
            except Exception:
                # Plotting errors must not kill the thread
                import traceback

                traceback.print_exc()
            finally:
                self.plot_queue.task_done()

    def _process_plot_job(self, job: Dict[str, Any]) -> None:
        """Make, save (and upload) the figures of one job."""
        epoch = job["epoch"]
        kwargs = {
            **job["plot_data"],
            "epoch": epoch,
            "weights": job["model"].get("weights"),
            **job["model"],
        }
        save_figures(
            self.plot_generator,
            kwargs,
            epoch,
            self.output_dir,
            log_images=self._upload if self.wandb_logger else None,
        )

    def _upload(self, images: Dict[str, Any], epoch: int) -> None:
        """Log images to wandb against the fractional data epoch."""
        import wandb

        if self.chunks_per_data_epoch and self.chunks_per_data_epoch > 0:
            fractional_epoch = (epoch + 1) / self.chunks_per_data_epoch
        else:
            fractional_epoch = epoch + 1
        wandb.log({"epoch": fractional_epoch, **images})

    def has_pending(self) -> bool:
        """Check if there are pending plots in the queue.

        Returns:
            bool: True if plots are pending, False otherwise
        """
        if self.blocking_mode:
            return False
        return not self.plot_queue.empty()

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending plots to complete.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds (None = wait forever)

        Returns:
            bool: True if all plots completed, False if timeout
        """
        if self.blocking_mode:
            return True

        try:
            if timeout is None:
                self.plot_queue.join()
                return True
            else:
                # Queue.join() doesn't support timeout, so we poll
                start_time = time.time()
                while not self.plot_queue.empty():
                    if time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
                # Brief wait for last task to complete
                time.sleep(0.5)
                return True
        except Exception:
            return False

    def close(self, timeout: float = 120.0) -> None:
        """Gracefully shutdown the async plotter.

        Args:
            timeout (float): Maximum time to wait for shutdown in seconds
        """
        if self.blocking_mode:
            return

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for remaining jobs to complete
        try:
            self.plot_queue.join()
            self.plot_thread.join(timeout=timeout)
        except Exception:
            pass
