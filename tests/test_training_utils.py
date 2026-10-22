"""Training utilities: weight perturbation and surrogate spikes."""

import numpy as np
import torch

from wiretorch.training_utils.surrogate_gradients import SurrGradSpike
from wiretorch.training_utils.weight_perturbers import (
    perturb_weights_scaling_factor,
)


def test_perturbation_scales_each_type_block_once():
    rng = np.random.default_rng(0)
    types, ff_types = rng.integers(0, 2, 40), np.zeros(10, dtype=int)
    w, w_ff = rng.random((40, 40)), rng.random((10, 40))
    pw, pw_ff, target, target_ff = perturb_weights_scaling_factor(
        w, w_ff, types, ff_types, 0.5
    )
    for a in range(2):
        for b in range(2):
            block = np.ix_(types == a, types == b)
            np.testing.assert_allclose(pw[block] * target[a, b], w[block])
    np.testing.assert_allclose(pw_ff * target_ff[0, types], w_ff)

    _, unchanged, _, none = perturb_weights_scaling_factor(
        w, w_ff, types, ff_types, 0.5, optimisable="scaling_factors_recurrent"
    )
    assert none is None and np.array_equal(unchanged, w_ff)


def test_surrogate_spike():
    x = torch.tensor([-1.0, 0.0, 0.5], requires_grad=True)
    s = SurrGradSpike.apply(x, 2.0)
    assert s.tolist() == [0.0, 0.0, 1.0]
    s.sum().backward()
    torch.testing.assert_close(x.grad, 1 / (2 * x.detach().abs() + 1) ** 2)


def test_calcium_model_matches_its_recurrence():
    from wiretorch.observation_models.calcium import CalciumObservationModel

    torch.manual_seed(0)
    gain = torch.rand(6) + 0.5
    model = CalciumObservationModel(gain=gain, tau=torch.linspace(200, 800, 6), dt=33.0)
    spikes = (torch.rand(2, 300, 6) < 0.2).float().requires_grad_()
    chunks = [model(spikes[:, :150]), model(spikes[:, 150:])]  # state carried across
    pred = torch.cat(chunks, dim=1)

    c, expected = torch.zeros(2, 6), []
    for t in range(300):
        c = model.gamma * c + model.a * spikes[:, t].detach()
        expected.append(c)
    torch.testing.assert_close(pred, torch.stack(expected, 1), rtol=1e-4, atol=1e-5)
    pred.sum().backward()
    assert torch.isfinite(spikes.grad).all()


def test_async_logger_columns(tmp_path):
    import pandas as pd

    from wiretorch.training_utils import AsyncLogger

    logger = AsyncLogger(tmp_path)
    logger.log(epoch=1, b=1.0)
    logger.log(epoch=2, b=2.0, a=0.5)  # a new column: earlier rows get it blank
    logger.close()
    resumed = AsyncLogger(tmp_path)  # resuming appends with the known columns
    resumed.log(epoch=3, a=1.5, b=3.0)
    resumed.close()

    table = pd.read_csv(tmp_path / "training_metrics.csv")
    assert list(table.columns) == ["epoch", "a", "b"]
    assert table["epoch"].tolist() == [1, 2, 3]
    assert table["b"].tolist() == [1.0, 2.0, 3.0]
    assert pd.isna(table["a"][0]) and table["a"][1:].tolist() == [0.5, 1.5]


def test_load_checkpoint_from_directory_takes_latest(tmp_path):
    from wiretorch.training_utils.checkpointing import (
        load_checkpoint,
        save_checkpoint,
    )
    from torch.amp import GradScaler

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(()))

        def get_checkpoint_state(self):
            return {}

        def load_checkpoint_state(self, state):
            pass

    model = Model()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    scaler = GradScaler("cuda", enabled=False)
    for epoch, value, loss in [(1, 1.0, 0.5), (2, 2.0, 0.9)]:  # best at 1, latest at 2
        model.w.data.fill_(value)
        save_checkpoint(tmp_path, epoch, model, opt, scaler, best_loss=0.5, total=loss)
    fresh = Model()
    epoch, _ = load_checkpoint(tmp_path, fresh, opt, scaler, "cpu")
    assert epoch == 2 and float(fresh.w) == 2.0
    # The old keyword is still accepted
    epoch, _ = load_checkpoint(
        tmp_path, fresh, optimiser=opt, scaler=scaler, device="cpu"
    )
    assert epoch == 2


def test_async_plotter_leaves_plot_data_alone(tmp_path):
    """Blocking mode plots in the caller's thread; the dict is not modified, model
    arrays are copied, and the generator gets only the arguments it takes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wiretorch.training_utils import AsyncPlotter

    received = {}

    def plot_generator(spikes, weights, epoch):
        received.update(spikes=spikes, weights=weights, epoch=epoch)
        return {"raster": plt.figure()}

    weights = np.ones((3, 3))
    plot_data = {"spikes": np.zeros((1, 5, 3)), "weights": weights, "voltages": None}
    before = dict(plot_data)
    plotter = AsyncPlotter(plot_generator, tmp_path, blocking_mode=True)
    assert plotter.submit_plot(plot_data, epoch=4)

    assert plot_data.keys() == before.keys() and plot_data["weights"] is weights
    assert received["epoch"] == 4 and received["weights"] is not weights
    np.testing.assert_array_equal(received["weights"], weights)
    assert (tmp_path / "figures" / "chunk_000005" / "raster.png").exists()
