r"""Calcium observation model: render spike trains into predicted dF/F traces.

This is a *forward* (observation) model, not a neural simulator. It takes a
network's spikes and renders the dF/F trace a calcium indicator would produce,
so the network can be trained in observable space — fitting the measured dF/F
directly, with no spike-deconvolution step in between.

Forward model (per cell)::

    a           = (1 + gamma) * gain           # peak ΔF/F per spike
    c[t]        = gamma * c[t-1] + a * s[t]     # latent calcium; s[t] = spikes in step t
    pred_dff[t] = b0 + c[t]                     # observed dF/F

- ``gamma`` — AR(1) decay of the indicator, ``gamma = exp(-dt / tau)``. Taken
  from the literature for the indicator and (at present) shared across cells.
- ``a = (1 + gamma) * gain`` — the measured gain is ``var/mean = a/(1+gamma)``,
  so the per-spike *peak* jump is ``(1 + gamma) * gain``. Do **not**
  unit-normalise the kernel: gain is peak ΔF/F per spike, not integrated area.
  Dropping the ``(1 + gamma)`` makes every rendered transient ~``(1+gamma)``×
  too small (≈2× for a slow indicator) and the network silently mis-scales its
  firing to compensate.
- ``b0`` — resting dF/F. Defaults to 0: the cells are active almost all the
  time, so the true silent baseline is unobservable; we adopt the data's
  8th-percentile zero as the model's zero and let ongoing spiking explain the
  elevated level of each trace. This is a convention, not a claim that rest sits
  at zero. ``b0`` is exposed per cell so a different convention can be supplied.
- **No noise term** — the model predicts the *expected* (noise-free) dF/F.
  Measurement noise lives in the residual the loss is fitting; injecting it here
  would only corrupt the target.
"""

import torch
import torch.nn as nn

from wiretorch.utils.recurrences import ar1_trajectory


def _as_param_tensor(value, name):
    """Coerce a scalar or 1-D array-like into a float32 tensor for a buffer."""
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.ndim > 1:
        raise ValueError(
            f"{name} must be a scalar or 1-D (per-cell) array, got shape {tuple(t.shape)}"
        )
    return t


class CalciumObservationModel(nn.Module):
    r"""AR(1) calcium forward (observation) model — renders spikes to dF/F.

    The model is **stateful**: the latent calcium ``c[t-1]`` is carried across
    successive ``forward`` calls so a long recording can be rendered
    chunk-by-chunk (as in chunked training). Because AR(1) dynamics are
    Markovian, carrying this single latent reproduces the full-recording trace
    exactly. Call :meth:`reset_state` at the start of each independent sequence.
    The carried latent is detached between chunks (truncated BPTT through time),
    matching the convention in ``VanRossumLoss``.

    Parameters are predefined (not learned) and per cell, so different cells —
    and, in future, different cell types — can carry different gain, baseline,
    and kinetics. ``gain``, ``tau`` and ``baseline`` may each be a scalar (shared
    across cells) or a per-cell 1-D tensor of length ``n_cells``.

    Args:
        gain: Measured per-cell gain (``var/mean``). Scalar or ``(n_cells,)``.
            The per-spike *peak* dF/F is ``(1 + gamma) * gain``.
        tau: Indicator decay time constant in ms (``gamma = exp(-dt / tau)``).
            Scalar or ``(n_cells,)``. Held constant across cells at present;
            pass a per-cell tensor to model cell-type-specific kinetics.
        dt: Time step of the spike trains it renders, in ms (the simulation dt
            under :class:`~wiretorch.training_utils.losses.CalciumMSELoss`,
            which frame-averages the rendered trace afterwards).
        baseline: Per-cell resting dF/F ``b0``. Scalar or ``(n_cells,)``.
            Defaults to 0.0 (the 8th-percentile-zero convention).
        trainable_baseline: If True, ``baseline`` becomes a learnable
            ``nn.Parameter`` (pass a per-cell init) instead of a fixed buffer, so
            the optimiser can place each cell's resting dF/F. Default False.
    """

    #: The latent calcium carried from chunk to chunk.
    CARRIED_STATE = ("prev_c",)

    def __init__(self, gain, tau, dt, baseline=0.0, trainable_baseline=False):
        super().__init__()
        self.dt = float(dt)
        self.trainable_baseline = bool(trainable_baseline)

        gain = _as_param_tensor(gain, "gain")
        tau = _as_param_tensor(tau, "tau")
        baseline = _as_param_tensor(baseline, "baseline")

        gamma = torch.exp(-self.dt / tau)  # AR(1) decay per cell (or scalar)
        a = (1.0 + gamma) * gain  # peak ΔF/F per spike per cell

        # Predefined, non-learnable parameters: register as buffers so they
        # move with .to(device) and are saved/loaded with the module.
        self.register_buffer("gain", gain)
        self.register_buffer("tau", tau)
        self.register_buffer("gamma", gamma)
        self.register_buffer("a", a)
        # Resting dF/F b0: a fixed buffer by default, or a learnable per-cell
        # offset when trainable_baseline=True (e.g. let the fit place each cell's
        # baseline rather than adopting a percentile-zero convention).
        if self.trainable_baseline:
            self.baseline = nn.Parameter(baseline.clone())
        else:
            self.register_buffer("baseline", baseline)

        # Latent calcium carried across chunks; initialised on first forward.
        self.prev_c = None

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Render a chunk of spikes into a predicted dF/F trace.

        Args:
            spikes: Spike train, shape ``(batch, time, n_cells)``. Bool or float.

        Returns:
            Predicted dF/F, shape ``(batch, time, n_cells)``, same dtype as the
            float-cast input.
        """
        spikes = spikes.float()
        if spikes.ndim != 3:
            raise ValueError(
                f"expected spikes of shape (batch, time, n_cells), got {tuple(spikes.shape)}"
            )
        batch, n_time, n_cells = spikes.shape

        if self.prev_c is None:
            c = spikes.new_zeros(batch, n_cells)
        else:
            c = self.prev_c

        # c[t] = gamma c[t-1] + a s[t]: an AR(1) recurrence, evaluated in closed form.
        # Its states s[0..T-1] lag c by one step, so c[0..T-1] = (s[1..T-1], s[T]).
        states, c = ar1_trajectory(c, self.a * spikes, self.gamma)
        c_traj = torch.cat([states[:, 1:], c[:, None]], dim=1)  # (batch, time, n_cells)
        pred_dff = self.baseline + c_traj

        # Carry the final latent to the next chunk, detached (truncated BPTT).
        self.prev_c = c.detach()

        return pred_dff

    def reset_state(self):
        """Clear the carried latent calcium.

        Call at the start of a new sequence / epoch so the next chunk starts
        from rest rather than inheriting the previous recording's state.
        """
        self.prev_c = None
