"""A short CMA-ES search runs, and finishes no worse than where it started."""

import numpy as np
import torch

from wiretorch.snn_runners import EvolutionarySearch
from wiretorch.training_utils.losses import VanRossumLoss
from fixtures.generate import training_setup


def test_search_improves_or_keeps_the_loss():
    trainer, _, _ = training_setup()
    model = trainer.model
    projections = list(model.projections.values())
    chunks = trainer.spike_iter

    def update(params):
        with torch.no_grad():
            for projection, value in zip(projections, params):
                projection.log_sf.fill_(float(value))

    start = np.array([float(p.log_sf) for p in projections])
    search = EvolutionarySearch(
        model=model,
        initial_params=start,
        update_fn=update,
        dataloader=chunks,
        loss_functions={"van_rossum": VanRossumLoss(10.0, 100.0, 1.0, 40)},
        loss_weights={"van_rossum": 1.0},
        device="cpu",
        n_chunks=5,
        initial_sigma=0.1,
        max_evaluations=12,
        popsize=4,
        seed=1,
        burn_in_chunks=1,
    )
    initial = search._evaluate(start)["total"]
    best = search.search()
    assert np.isfinite(best) and best <= initial
