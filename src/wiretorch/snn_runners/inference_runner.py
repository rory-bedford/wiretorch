"""Run a model over a dataloader, storing its inputs and outputs in memory or zarr.

Each chunk's tensors are copied to reused pinned host buffers without blocking, and a
background thread files them away (into preallocated arrays, or zarr), so the GPU is
already simulating the next chunk while the previous one is being stored.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import zarr
from tqdm import tqdm


class SNNInference:
    """Forward passes over a dataloader, with every input and output recorded.

    Recorded per chunk: every tensor field of the dataloader's named tuples, the model's
    spikes (as ``output_spikes``, bool) and, with ``save_tracked_variables``, every
    other tensor the model returns when tracking. All are ``(batch, time, ...)`` and are
    stacked along time.

    Args:
        model: The SNN; its state continues from chunk to chunk.
        dataloader: Iterable of named tuples with at least ``input_spikes``.
        device: Where the model runs ('cpu' or 'cuda').
        output_mode: ``"memory"`` (returned as numpy arrays) or ``"zarr"``.
        zarr_path: Output group for zarr mode (overwritten).
        save_tracked_variables: Also store the model's tracked variables.
        max_chunks: Chunks to run (None: until the dataloader ends; required for zarr).
        progress_bar: Show a progress bar.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str,
        output_mode: Literal["zarr", "memory"] = "memory",
        zarr_path: Path | None = None,
        save_tracked_variables: bool = True,
        max_chunks: int | None = None,
        progress_bar: bool = True,
    ):
        if output_mode == "zarr" and zarr_path is None:
            raise ValueError("zarr_path must be provided when output_mode='zarr'")
        if output_mode == "zarr" and max_chunks is None:
            raise ValueError("max_chunks must be specified when using zarr output mode")
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_mode = output_mode
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self.save_tracked_variables = save_tracked_variables
        self.max_chunks = max_chunks
        self.show_progress = progress_bar

    def run(self) -> dict[str, Any]:
        """Run every chunk.

        Returns:
            Memory mode: ``{name: array}`` over all chunks. Zarr mode: ``{"zarr_path",
            "num_chunks", "metadata"}``.
        """
        sink = (
            _ZarrSink(self.zarr_path, self.max_chunks, getattr(self.model, "dt", None))
            if self.output_mode == "zarr"
            else _MemorySink(self.max_chunks)
        )
        chunks = islice(self.dataloader, self.max_chunks)
        if self.show_progress:
            chunks = tqdm(chunks, total=self.max_chunks, desc="Running inference")
        n_chunks = 0
        with torch.inference_mode(), _Stager() as stager:
            for batch in chunks:
                x = batch.input_spikes.to(self.device, non_blocking=True)
                record = self._record(batch, self.model.forward(input_spikes=x))
                sink.write(n_chunks, stager.stage(record))
                n_chunks += 1
        return sink.close(n_chunks)

    def _record(self, batch, outputs) -> dict[str, torch.Tensor]:
        """Everything to store from one chunk, by name."""
        record = {
            name: value
            for name, value in zip(batch._fields, batch)
            if isinstance(value, torch.Tensor)
        }
        tracked = outputs if isinstance(outputs, dict) else {"spikes": outputs}
        record["output_spikes"] = tracked["spikes"].bool()
        if self.save_tracked_variables:
            record.update(
                (k, v)
                for k, v in tracked.items()
                if k != "spikes" and isinstance(v, torch.Tensor)
            )
        return record


# =====================================================================
# Moving chunks to the host, and storing them
# =====================================================================


class _Stager:
    """Copies a chunk's tensors to reused pinned host buffers without blocking.

    ``stage`` returns a future of the chunk as numpy arrays. A buffer is reused only
    once the previous chunk's consumer has finished with it, so at most one chunk is in
    flight while the next is simulated.
    """

    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}
        self._pending: Future | None = None

    def stage(self, record: dict[str, torch.Tensor]) -> Future:
        if self._pending is not None:
            self._pending.result()  # the previous chunk is done with the buffers
        host = {}
        for name, tensor in record.items():
            buffer = self._buffers.get(name)
            if (
                buffer is None
                or buffer.shape != tensor.shape
                or buffer.dtype != tensor.dtype
            ):
                buffer = torch.empty(
                    tensor.shape, dtype=tensor.dtype, pin_memory=tensor.is_cuda
                )
                self._buffers[name] = buffer
            buffer.copy_(tensor, non_blocking=True)
            host[name] = buffer
        done = torch.cuda.Event() if any(t.is_cuda for t in record.values()) else None
        if done is not None:
            done.record()
        self._pending = _Chunk(host, done)
        return self._pending

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pending is not None:
            self._pending.result()


class _Chunk(Future):
    """A staged chunk: its host arrays, readable once the copies have landed."""

    def __init__(self, host: dict[str, torch.Tensor], done):
        super().__init__()
        self.host, self.done = host, done

    def arrays(self) -> dict[str, np.ndarray]:
        if self.done is not None:
            self.done.synchronize()
        return {name: t.numpy() for name, t in self.host.items()}


class _Sink:
    """Stores staged chunks in order, in a background thread."""

    def __init__(self):
        self._writer = ThreadPoolExecutor(max_workers=1)
        self._last: Future | None = None

    def write(self, index: int, chunk: _Chunk) -> None:
        if self._last is not None:
            self._last.result()  # already finished (see _Stager); re-raises its errors

        def store():
            try:
                self._store(index, chunk.arrays())
            finally:
                chunk.set_result(None)  # releases the host buffers

        self._last = self._writer.submit(store)

    def close(self, n_chunks: int):
        if self._last is not None:
            self._last.result()
        self._writer.shutdown()
        return self._result(n_chunks)

    def _store(self, index: int, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def _result(self, n_chunks: int):
        raise NotImplementedError


class _MemorySink(_Sink):
    """Into whole-run numpy arrays (preallocated when the chunk count is known)."""

    def __init__(self, max_chunks: int | None):
        super().__init__()
        self.max_chunks = max_chunks
        self.arrays: dict[str, np.ndarray | list] = {}

    def _store(self, index, arrays):
        for name, chunk in arrays.items():
            if self.max_chunks is None:
                self.arrays.setdefault(name, []).append(chunk.copy())
                continue
            if name not in self.arrays:
                shape = (
                    chunk.shape[0],
                    chunk.shape[1] * self.max_chunks,
                    *chunk.shape[2:],
                )
                self.arrays[name] = np.empty(shape, dtype=chunk.dtype)
            T = chunk.shape[1]
            self.arrays[name][:, index * T : (index + 1) * T] = chunk

    def _result(self, n_chunks):
        out = {}
        for name, value in self.arrays.items():
            if isinstance(value, list):
                out[name] = np.concatenate(value, axis=1)
            else:  # fewer chunks than max_chunks if the dataloader ran out
                out[name] = value[:, : value.shape[1] // self.max_chunks * n_chunks]
        return out


class _ZarrSink(_Sink):
    """Into a zarr group, one array per name; chunked per trial and time block."""

    def __init__(self, path: Path, max_chunks: int, dt):
        super().__init__()
        self.path, self.max_chunks = path, max_chunks
        self.root = zarr.open_group(path, mode="w")
        if dt is not None:
            self.root.attrs["dt"] = float(dt)
        self.arrays: dict[str, zarr.Array] = {}

    def _store(self, index, arrays):
        for name, chunk in arrays.items():
            B, T = chunk.shape[:2]
            if name not in self.arrays:
                # One zarr chunk per (trial, time block): compressed in parallel.
                self.arrays[name] = self.root.create_array(
                    name,
                    shape=(B, T * self.max_chunks, *chunk.shape[2:]),
                    chunks=(1, T, *chunk.shape[2:]),
                    dtype=chunk.dtype,
                )
            self.arrays[name][:, index * T : (index + 1) * T] = chunk

    def _result(self, n_chunks):
        return {
            "zarr_path": self.path,
            "num_chunks": n_chunks,
            "metadata": dict(self.root.attrs),
        }
