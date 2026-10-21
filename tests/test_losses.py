"""The spike-train losses compute what their docstrings define.

The van Rossum filter is written out here as the literal sum over past inputs, with
the documented chunk carry: the last filtered value of a chunk enters the next one
decayed by ``exp(-t dt / tau)`` for ``t = 0, 1, ...``, with no gradient through it. The library computes the same
thing with batched triangular matrix products.
"""

import numpy as np
import pytest
import torch

from connectome_snns.training_utils.losses import (
    FiringRateLoss,
    PoissonKLLoss,
    SilentNeuronLoss,
    VanRossumLoss,
    VanRossumRateLoss,
)

TAU_RISE, TAU_DECAY, DT, CHUNK = 10.0, 100.0, 1.0, 40
# CPU: the same sums as the definition. CUDA: a recursion, whose rounding differs.
RTOL = {"cpu": 1e-5, "cuda": 1e-4}


def exponential_filter(x, tau, carried):
    """``y[t] = sum_{k <= t} exp(-k dt / tau) x[t - k] + carried * exp(-t dt / tau)``."""
    time = x.shape[1]
    kernel = torch.exp(-torch.arange(time, dtype=x.dtype, device=x.device) * DT / tau)
    y = torch.stack(
        [
            (kernel[: t + 1].flip(0)[None, :, None] * x[:, : t + 1]).sum(1)
            for t in range(time)
        ],
        dim=1,
    )
    return y if carried is None else y + carried * kernel[None, :, None]


def reference_smooth(chunks):
    """Filter a sequence of chunks, carrying state, as the docstring defines."""
    rise = smooth = None
    out = []
    for x in chunks:
        r = exponential_filter(x, TAU_RISE, rise)
        y = exponential_filter(r, TAU_DECAY, smooth)
        rise, smooth = (
            r[:, -1:].detach(),
            y[:, -1:].detach(),
        )  # no gradient across chunks
        out.append(y)
    return out


@pytest.fixture
def trains():
    rng = np.random.default_rng(0)

    def train():
        return torch.from_numpy(
            (rng.random((3, 3 * CHUNK, 7)) < 0.05).astype(np.float32)
        )

    return train(), train()


@pytest.mark.parametrize("loss_cls", [VanRossumLoss, PoissonKLLoss])
def test_loss_matches_definition(trains, loss_cls, device):
    output, target = (x.to(device) for x in trains)
    output.requires_grad_(True)
    loss_fn = loss_cls(TAU_RISE, TAU_DECAY, DT, CHUNK, device=device)
    chunks = [slice(c, c + CHUNK) for c in range(0, output.shape[1], CHUNK)]

    o_ref = reference_smooth([output[:, c] for c in chunks])
    t_ref = reference_smooth([target[:, c] for c in chunks])
    for c, o, t in zip(chunks, o_ref, t_ref):
        loss = loss_fn(output[:, c], target[:, c])
        if loss_cls is VanRossumLoss:
            expected = ((o - t) ** 2).mean()
        else:
            eps, t = loss_fn.eps, t.detach()
            expected = (t * torch.log(t / (o + eps) + eps) - t + o).mean() * DT
        torch.testing.assert_close(loss, expected, rtol=RTOL[device], atol=1e-7)
        (grad,) = torch.autograd.grad(loss, output, retain_graph=True)
        (expected_grad,) = torch.autograd.grad(expected, output, retain_graph=True)
        torch.testing.assert_close(
            grad, expected_grad, rtol=1e-4, atol=1e-4 * float(expected_grad.abs().max())
        )


def test_reset_state_forgets(trains):
    output, target = trains
    loss_fn = VanRossumLoss(TAU_RISE, TAU_DECAY, DT, CHUNK)
    first = loss_fn(output[:, :CHUNK], target[:, :CHUNK])
    loss_fn(output[:, CHUNK : 2 * CHUNK], target[:, CHUNK : 2 * CHUNK])
    loss_fn.reset_state()
    torch.testing.assert_close(loss_fn(output[:, :CHUNK], target[:, :CHUNK]), first)


@pytest.mark.parametrize("use_probability", [False, True])
def test_rate_loss_filters_the_rate_like_a_train(trains, use_probability):
    """The target rate goes through the van Rossum filter as a target train would."""
    output, _ = trains
    rate = torch.full_like(output, 0.05)
    loss_fn = VanRossumRateLoss(
        TAU_RISE, TAU_DECAY, DT, CHUNK, use_probability=use_probability
    )
    spike_loss = VanRossumLoss(TAU_RISE, TAU_DECAY, DT, CHUNK)
    signal = "output_probabilities" if use_probability else "output_spikes"
    assert loss_fn.required_inputs == [signal, "target_rate"]
    assert not loss_fn.requires_target
    for c in range(0, output.shape[1], CHUNK):
        chunk = slice(c, c + CHUNK)
        torch.testing.assert_close(
            loss_fn(target_rate=rate[:, chunk], **{signal: output[:, chunk]}),
            spike_loss(output[:, chunk], rate[:, chunk]),
        )


def test_firing_rate_loss(trains, tmp_path):
    """Normalised squared rate error, against a fixed target or the target's spikes."""
    output, target = trains
    seconds = output.shape[1] * DT / 1000
    rate, target_rate = output.sum(1) / seconds, target.sum(1) / seconds
    expected = ((rate - target_rate) ** 2 / (target_rate + 1.0)).mean()

    per_chunk = FiringRateLoss(DT)
    assert per_chunk.requires_target
    torch.testing.assert_close(per_chunk(output, target), expected)

    fixed = FiringRateLoss(DT, target_rate=target_rate)
    assert not fixed.requires_target
    torch.testing.assert_close(fixed(output), expected)

    import zarr

    zarr.open_group(str(tmp_path), mode="w").create_array(
        "output_spikes", data=target.numpy()
    )
    loaded = FiringRateLoss.from_spike_data(tmp_path, DT)
    torch.testing.assert_close(loaded(output), expected)


def test_silent_neuron_loss():
    """``exp(-alpha * rate)``: 1 for a silent neuron, falling as it fires."""
    spikes = torch.zeros(1, 1000, 3)
    spikes[0, ::100, 1] = 1  # 10 Hz
    spikes[0, ::10, 2] = 1  # 100 Hz
    loss = SilentNeuronLoss(alpha=0.1, dt=DT)(spikes)
    expected = (1 + np.exp(-1.0) + np.exp(-10.0)) / 3
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=torch.float32))


def test_zero_weight_losses_are_not_computed():
    from connectome_snns.snn_runners.chunks import weighted_losses

    calls = []

    class Counted(torch.nn.Module):
        def forward(self, output_spikes):
            calls.append(1)
            return output_spikes.float().mean()

    chunk = {"spikes": torch.ones(2, 3, 4)}
    losses = weighted_losses(
        {"used": Counted(), "off": Counted(), "unweighted": Counted()},
        {"used": 2.0, "off": 0.0},
        chunk,
        model=None,
    )
    assert len(calls) == 1
    assert sorted(losses) == ["total", "used"]
    assert losses["total"].item() == 2.0
