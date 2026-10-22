"""
Gradient-free parameter search using CMA-ES.

Searches over a parameter vector using CMA-ES, evaluating candidates
by running the model forward and computing losses. The caller provides
the initial parameter values and an update function that writes candidate
values into the model.

Loss functions use the same required_inputs/requires_target interface
as SNNTrainer.
"""

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from wiretorch.snn_runners.chunks import (
    reset_losses,
    to_device,
    warm_up_losses,
    weighted_losses,
)


class EvolutionarySearch:
    """
    Gradient-free parameter search using CMA-ES.

    Args:
        model: The initialized SNN model (should use optimisable=None for speed).
        initial_params: 1D numpy array of initial parameter values for the search.
        update_fn: Callable that takes a 1D numpy array and writes candidate
            values into the model (e.g. reshaping and calling
            model.update_scaling_factors).
        dataloader: Iterator providing input spike data (must return named tuples).
        loss_functions: Dict with individual loss functions.
        loss_weights: Dict with weights for each loss function.
        device: Device string ('cpu' or 'cuda').
        n_chunks: Number of chunks to preload from the dataloader.
        initial_sigma: Initial step size for CMA-ES.
        max_evaluations: Maximum number of function evaluations.
        popsize: Population size per generation (default: CMA-ES auto).
        batch_size: Number of batch elements to use per evaluation. If None,
            uses the full batch from the dataloader.
        seed: Random seed for CMA-ES.
        callback: Optional function called after each generation with a dict
            containing: generation, n_evals, best_loss, mean_loss, sigma.
        burn_in_chunks: Number of initial chunks to skip for loss computation.
            The model still runs forward (to build up internal state) but those
            chunks don't contribute to the fitness value.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_params: np.ndarray,
        update_fn: Callable[[np.ndarray], None],
        dataloader,
        loss_functions: dict[str, Callable],
        loss_weights: dict[str, float],
        device: str,
        n_chunks: int,
        initial_sigma: float = 0.1,
        max_evaluations: int = 200,
        popsize: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        callback: Callable | None = None,
        burn_in_chunks: int = 0,
    ):
        self.model = model
        self.update_fn = update_fn
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.device = device
        self.initial_sigma = initial_sigma
        self.max_evaluations = max_evaluations
        self.popsize = popsize
        self.seed = seed
        self.callback = callback
        self.burn_in_chunks = burn_in_chunks

        self._initial_params = initial_params

        # Compile model for speed on GPU
        if str(device).startswith("cuda"):
            try:
                self.model = torch.compile(model)
                print("  ✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠ torch.compile failed ({e}), running in eager mode")

        # Preload data chunks. Move to the target device (GPU) to avoid
        # costly CPU↔GPU round-trips during the evaluation loop. Batch slicing
        # is applied first to limit memory usage.
        self._chunks = []
        for _, batch in zip(range(n_chunks), dataloader):
            if batch_size is not None:  # slice before moving, to move less
                batch = batch._replace(
                    **{
                        k: v[:batch_size]
                        for k, v in batch._asdict().items()
                        if isinstance(v, torch.Tensor)
                    }
                )
            self._chunks.append(to_device(batch, device))

        self._n_chunks = len(self._chunks)
        self._eval_batch_size = self._chunks[0]["input_spikes"].shape[0]

    def _evaluate(self, flat_params: np.ndarray) -> dict[str, float]:
        """Mean weighted losses of a candidate over the chunks after burn-in."""
        self.update_fn(flat_params)
        if hasattr(self.model, "reset_state"):
            self.model.reset_state(batch_size=self._eval_batch_size)
        reset_losses(self.loss_functions)
        self.model.track_variables = False
        self.model.track_batch_idx = None

        totals = {}
        with torch.inference_mode():
            for i, data in enumerate(self._chunks):
                spikes = self.model.forward(input_spikes=data["input_spikes"])
                chunk = {**data, "spikes": spikes}
                if i < self.burn_in_chunks:
                    warm_up_losses(self.loss_functions, chunk, self.model)
                    continue
                losses = weighted_losses(
                    self.loss_functions, self.loss_weights, chunk, self.model
                )
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + value
        n = self._n_chunks - self.burn_in_chunks
        return {name: float(value) / n for name, value in totals.items()}

    def search(self) -> float:
        """
        Run CMA-ES search.

        Uses the initial parameter values as the starting point, runs the
        search, writes the best values back via update_fn, and returns the
        best loss.
        """
        x0 = self._initial_params
        n_evals = 0
        generation = 0

        opts = {
            "maxfevals": self.max_evaluations,
            "verbose": -1,  # suppress CMA-ES output, we print our own
        }
        if self.popsize is not None:
            opts["popsize"] = self.popsize
        if self.seed is not None:
            opts["seed"] = self.seed

        import cma  # only needed here, so not a dependency of importing the runners

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.initial_sigma, opts)

        pbar = tqdm(
            total=self.max_evaluations,
            desc="CMA-ES",
            unit="eval",
        )

        while not es.stop():
            candidates = es.ask()
            fitnesses = []
            all_losses = []
            for candidate in candidates:
                losses = self._evaluate(np.array(candidate))
                fitnesses.append(losses["total"])
                all_losses.append(losses)
                n_evals += 1
                pbar.update(1)

            es.tell(candidates, fitnesses)
            generation += 1

            best_loss = es.result.fbest
            mean_losses = {
                k: float(np.mean([l[k] for l in all_losses])) for k in all_losses[0]
            }
            pbar.set_postfix(best=f"{best_loss:.6f}", sigma=f"{es.sigma:.4f}")

            if self.callback is not None:
                # Write best-so-far params to model so callback can inspect/evaluate
                self.update_fn(np.array(es.result.xbest))

                # Run forward pass with best params to get output spikes (streamed)
                if hasattr(self.model, "reset_state"):
                    self.model.reset_state(batch_size=self._eval_batch_size)
                self.model.track_variables = False
                spike_chunks = []
                with torch.inference_mode():
                    for data in self._chunks:
                        spikes = self.model.forward(input_spikes=data["input_spikes"])
                        spike_chunks.append(spikes.cpu())
                output_spikes = torch.cat(spike_chunks, dim=1)

                self.callback(
                    {
                        "generation": generation,
                        "n_evals": n_evals,
                        "best_loss": best_loss,
                        "mean_losses": mean_losses,
                        "sigma": es.sigma,
                        "output_spikes": output_spikes,
                    }
                )

        pbar.close()

        # Write best parameters back to the model
        self.update_fn(np.array(es.result.xbest))

        best_loss = es.result.fbest
        print(f"  CMA-ES finished: {n_evals} evaluations, best loss: {best_loss:.6f}")
        return best_loss
