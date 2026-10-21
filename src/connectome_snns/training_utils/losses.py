"""Loss functions for training spiking networks.

Each loss declares the inputs it needs in ``required_inputs`` (and ``requires_target``
for the target spikes); the trainer supplies exactly those
(``snn_runners.chunks.loss_inputs``). Stateful losses — the van Rossum family, which
filters across chunks — implement ``reset_state`` and name the attributes they carry
in ``CARRIED_STATE`` (saved in checkpoints).

* Spike-train distances: ``VanRossumLoss``, ``VanRossumRateLoss``, ``PoissonKLLoss``.
* Rate penalties: ``FiringRateLoss``, ``SilentNeuronLoss``.
* Observation-space losses: ``CalciumMSELoss``.
"""

import torch
import torch.nn as nn

from connectome_snns.observation_models import CalciumObservationModel
from connectome_snns.training_utils.filtering import double_exponential


def _per_neuron(x: torch.Tensor) -> torch.Tensor:
    """``(batch, time, n)`` -> ``(batch * n, 1, time)``, the layout debug output uses."""
    return x.permute(0, 2, 1).reshape(-1, 1, x.shape[1])


def firing_rates(spikes: torch.Tensor, dt: float) -> torch.Tensor:
    """Firing rates in Hz over the time axis (second to last) of ``spikes``."""
    return spikes.float().sum(dim=-2) / (spikes.shape[-2] * dt / 1000.0)


class VanRossumLoss(nn.Module):
    """Van Rossum distance between spike trains, with a double-exponential filter.

    Both trains are filtered first with the rise and then with the decay exponential,
    and the loss is the mean squared difference of the results. The filter is causal
    and carries its state across chunks: the last filtered value of each chunk enters
    the next chunk decayed by ``exp(-t dt / tau)``, ``t = 0, 1, ...`` (undecayed at its
    first step).

    The filtering itself is ``filtering.double_exponential``: a triangular matrix
    product per stage on the CPU, a fused linear-time recursion on CUDA.

    Expects ``(batch, time, n_neurons)`` or ``(batch, patterns, time, n_neurons)``.
    """

    required_inputs = ["output_spikes", "target_spikes"]
    requires_target = True
    #: The filter state carried from chunk to chunk.
    CARRIED_STATE = (
        "prev_output_rise",
        "prev_target_rise",
        "prev_output_smooth",
        "prev_target_smooth",
    )

    def __init__(
        self,
        tau_rise: float,
        tau_decay: float,
        dt: float,
        window_size: int,
        device: str = "cpu",
        debug: bool = False,
    ):
        """
        Args:
            tau_rise (float): Rise time constant of the first filter (ms).
            tau_decay (float): Decay time constant of the second filter (ms).
            dt (float): Simulation time step (ms).
            window_size (int): Steps per chunk (the longest chunk accepted).
            device (str, optional): Device of the filter matrices. Defaults to "cpu".
            debug (bool, optional): If True, return ``(loss, output_smooth,
                target_smooth)``, the filtered trains shaped
                ``(batch * n_neurons, 1, time)``. Defaults to False.
        """
        super().__init__()
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.dt = dt
        self.kernel_size = window_size
        self.debug = debug

        # kernel[k]: the weight of the input k steps ago.
        t = torch.arange(window_size, dtype=torch.float32, device=device) * dt
        self.register_buffer("kernel_rise", torch.exp(-t / tau_rise))
        self.register_buffer("kernel_decay", torch.exp(-t / tau_decay))

        # Filter state carried from the previous chunk, (batch, 1, n_neurons).
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None

    def smooth(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter both trains, continuing from (and updating) the carried state.

        Returns the filtered trains, ``(batch, time, n_neurons)``.
        """
        output_spikes = output_spikes.float()
        target_spikes = target_spikes.float()
        if output_spikes.ndim == 4:  # (batch, patterns, ...) -> (batch * patterns, ...)
            output_spikes = output_spikes.flatten(0, 1)
            target_spikes = target_spikes.flatten(0, 1)

        kernels = self.kernel_rise, self.kernel_decay
        output_rise, output_smooth = double_exponential(
            output_spikes, *kernels, self.prev_output_rise, self.prev_output_smooth
        )
        target_rise, target_smooth = double_exponential(
            target_spikes, *kernels, self.prev_target_rise, self.prev_target_smooth
        )

        # Carry the last step to the next chunk (detached: no gradient across chunks).
        self.prev_output_rise = output_rise[:, -1:].detach()
        self.prev_target_rise = target_rise[:, -1:].detach()
        self.prev_output_smooth = output_smooth[:, -1:].detach()
        self.prev_target_smooth = target_smooth[:, -1:].detach()
        return output_smooth, target_smooth

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean squared difference of the filtered output and target trains."""
        output_smooth, target_smooth = self.smooth(output_spikes, target_spikes)
        loss = ((output_smooth - target_smooth) ** 2).mean()
        if self.debug:
            return loss, _per_neuron(output_smooth), _per_neuron(target_smooth)
        return loss

    def reset_state(self):
        """Forget the carried filter state (call at the start of a new sequence)."""
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None


class VanRossumRateLoss(VanRossumLoss):
    """Van Rossum distance against a smooth firing-rate target (native target_rate).

    Identical double-exponential filter to :class:`VanRossumLoss` — the stochastic
    output spikes are double-filtered exactly as before — but the target is the
    measured **smooth firing rate** (expected spikes per bin, ``rate * dt``) put
    through the *same* filter, rather than a target spike train. Because the filter
    is linear, ``E[filter(spikes)] = filter(E[spikes]) = filter(rate * dt)``, so
    the MSE between the filtered stochastic output and the filtered rate is
    unbiased in expectation — a lower-variance objective than matching a sampled
    target spike train.

    The target is supplied natively as ``SpikeData.target_rate`` (routed by the
    trainer via ``required_inputs``), not through the spike-target slot.

    With ``use_probability=True`` (escape-noise / sample-mode models only) the loss
    filters the network's per-step spike *probability* ``p`` instead of the sampled
    spikes ``s``. Since ``E[s] = p``, this removes the sampling-variance term of the
    objective: filtering ``s`` makes the expected loss ``bias² + r·E_k``, whose
    minimum is silence whenever the target rate ``r < E_k/K²``; filtering ``p`` makes
    it pure ``bias²`` so the minimum sits at the target rate. The trainer supplies
    ``p`` via the ``output_probabilities`` snapshot key.
    """

    requires_target = False

    def __init__(self, *args, use_probability: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_probability = use_probability
        # Instance-level so the two modes never collide across loss instances.
        self.required_inputs = (
            ["output_probabilities", "target_rate"]
            if use_probability
            else ["output_spikes", "target_rate"]
        )

    def forward(
        self,
        target_rate: torch.Tensor,
        output_spikes: torch.Tensor | None = None,
        output_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Double-filter the output signal and the smooth rate, MSE between them.

        Args:
            target_rate: Smooth target rate as expected spikes per bin
                (``rate * dt``), shape ``(batch, time, n_cells)``.
            output_spikes: Sampled network spikes (default signal).
            output_probabilities: Per-step spike probability ``p``; used instead of
                ``output_spikes`` when ``use_probability=True``.
        """
        signal = (
            output_probabilities if output_probabilities is not None else output_spikes
        )
        return super().forward(signal, target_rate)


class FiringRateLoss(nn.Module):
    required_inputs = ["output_spikes"]

    def __init__(
        self,
        dt: float,
        target_rate: torch.Tensor | None = None,
        epsilon: float = 1.0,
    ):
        """
        Firing rate loss for spike trains.

        Computes per-chunk firing rates for the student, then compares to target
        values using normalized MSE loss. Each neuron's squared error is divided
        by its target rate (plus epsilon) to give equal relative importance.
        The loss is averaged over both trials and neurons.

        Two modes:
        - Fixed target: provide target_rate at init. Uses precomputed rates that
          stay constant across chunks. Shape can be (n_neurons,) for a single
          global target, or (batch, n_neurons) for per-trial targets.
        - Per-chunk target: omit target_rate. Computes target rates from
          target_spikes each forward call (requires_target=True).

        Use ``from_spike_data`` to precompute per-trial, per-neuron target rates
        from a full zarr dataset (averaged across the entire trial).

        Args:
            dt (float): Simulation time step (ms).
            target_rate (torch.Tensor | None): Target firing rates in Hz.
                Shape: (n_neurons,), (batch, n_neurons), or (batch, patterns, n_neurons).
                If None, target rates are computed from target_spikes each forward call.
            epsilon (float): Regularization term added to target_rate in denominator to prevent
                numerical instability. Default 1.0 Hz provides reasonable normalization even for
                near-zero target rates.
        """
        super().__init__()
        self.dt = dt
        self.epsilon = epsilon

        if target_rate is not None:
            self.register_buffer("target_rate", target_rate)
            self.requires_target = False
        else:
            self.target_rate = None
            self.requires_target = True

    @classmethod
    def from_spike_data(
        cls,
        spike_data,
        dt: float,
        epsilon: float = 1.0,
    ) -> "FiringRateLoss":
        """Create a FiringRateLoss with per-trial, per-neuron target rates.

        Computes target firing rates averaged over the entire trial duration,
        giving a stable per-trial, per-neuron target that doesn't fluctuate
        chunk-to-chunk.

        Args:
            spike_data: Target spike data, either a numpy/zarr array of shape
                (batch, total_time, n_neurons) or a path to a zarr group
                containing an 'output_spikes' dataset.
            dt (float): Simulation time step (ms).
            epsilon (float): Regularization epsilon. Default 1.0.

        Returns:
            FiringRateLoss with target_rate shape (batch, n_neurons).
        """
        import numpy as np
        from pathlib import Path

        if isinstance(spike_data, (str, Path)):
            import zarr

            root = zarr.open_group(str(spike_data), mode="r")
            spike_data = root["output_spikes"]

        # spike_data shape: (batch, total_time, n_neurons)
        # Sum over time axis, compute rate in Hz
        spike_counts = np.array(spike_data).sum(axis=1)  # (batch, n_neurons)
        total_time_s = spike_data.shape[1] * dt / 1000.0
        target_rate = torch.from_numpy((spike_counts / total_time_s).astype(np.float32))

        return cls(dt=dt, target_rate=target_rate, epsilon=epsilon)

    def _compute_rates(self, spikes: torch.Tensor) -> torch.Tensor:
        """Per-trial rates in Hz: (batch, neurons) or (batch, patterns, neurons)."""
        return firing_rates(spikes, self.dt)

    def forward(
        self,
        output_spikes: torch.Tensor,
        target_spikes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute firing rate loss for spike trains.

        Computes per-trial, per-neuron normalized squared error between student
        and target firing rates, then averages over all trials and neurons.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).
            target_spikes (torch.Tensor | None): Target spike trains (same shape as
                output_spikes). Required when target_rate was not provided at init.

        Returns:
            torch.Tensor: Mean normalized MSE loss between actual and target firing rates.
        """
        firing_rates = self._compute_rates(output_spikes)

        if self.target_rate is not None:
            target_rates = self.target_rate.to(firing_rates.device)
        else:
            target_rates = self._compute_rates(target_spikes)

        # Compute per-trial per-neuron squared error normalized by target rate
        squared_errors = (firing_rates - target_rates) ** 2
        normalized_errors = squared_errors / (target_rates + self.epsilon)

        return normalized_errors.mean()


class SilentNeuronLoss(nn.Module):
    required_inputs = ["output_spikes"]
    requires_target = False

    def __init__(self, alpha: float = 1.0, dt: float = 1.0):
        """
        Exponential penalty for silent neurons.

        Applies an exponential penalty to neurons with very low or zero firing rates
        to encourage all neurons to be active. The penalty is: exp(-alpha * firing_rate).

        Expects input shape: (batch, time, n_neurons).

        Args:
            alpha (float or torch.Tensor): Exponential decay parameter(s). Higher values create stronger
                          penalties for silent neurons. Can be scalar or tensor of shape (n_neurons,). Default: 1.0.
            dt (float): Simulation time step (ms). Default: 1.0.
        """
        super().__init__()
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha
        self.dt = dt

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute silent neuron penalty.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean exponential penalty across all batches and neurons.
        """
        # Silent neurons (rate ~ 0) cost ~1, active ones less.
        return torch.exp(-self.alpha * firing_rates(output_spikes, self.dt)).mean()


class PoissonKLLoss(VanRossumLoss):
    """Poisson KL divergence loss with double-exponential temporal filtering.

    Drop-in replacement for :class:`VanRossumLoss`.  Both losses smooth spike
    trains with the same double-exponential kernel (rise + decay time
    constants), but this loss computes Poisson KL divergence between the
    filtered teacher and student rates instead of squared error::

        KL = λ_T * log(λ_T / λ_S) - λ_T + λ_S

    The KL formulation penalises under-firing proportionally to the teacher
    rate, eliminating the systematic bias toward silence that squared-error
    losses exhibit.  Maintains internal state across chunks for seamless
    streaming during chunked training.

    Compare with :class:`BernoulliNLLLoss`, which operates on raw per-timestep
    probabilities without temporal filtering.
    """

    def __init__(
        self,
        tau_rise: float,
        tau_decay: float,
        dt: float,
        window_size: int,
        device: str = "cpu",
        debug: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(tau_rise, tau_decay, dt, window_size, device, debug)
        self.eps = eps

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poisson KL divergence of the filtered output from the filtered target."""
        output_smooth, target_smooth = self.smooth(output_spikes, target_spikes)
        # Poisson KL divergence: λ_T * log(λ_T / (λ_S + ε)) - λ_T + λ_S
        lambda_T = target_smooth.detach()
        lambda_S = output_smooth
        kl = (
            lambda_T * torch.log(lambda_T / (lambda_S + self.eps) + self.eps)
            - lambda_T
            + lambda_S
        )
        loss = kl.mean() * self.dt
        if self.debug:
            return loss, _per_neuron(output_smooth), _per_neuron(target_smooth)
        return loss


class CalciumMSELoss(nn.Module):
    """MSE between a rendered dF/F prediction and a measured dF/F target.

    Runs the network's spikes through a :class:`CalciumObservationModel` to get a
    predicted dF/F trace at the simulation timestep, frame-averages it down to the
    measured imaging frame rate, and compares it to the measured dF/F with mean
    squared error — i.e. the network is fit in *observable* space, with no
    spike-deconvolution of the target.

    The target is a **first-class calcium target**, supplied by the dataloader as
    ``SpikeData.target_calcium`` and routed by the trainer via ``required_inputs``
    (no overloading of the spike-target slot).

    Resolution: ``output_spikes`` is ``(batch, time, n_cells)`` at the simulation
    dt; ``target_calcium`` is ``(batch, n_frames, n_cells)`` on the (coarser)
    imaging grid, where ``n_frames`` divides ``time``. The rendered prediction is
    averaged within each frame (matching how an imaging frame integrates the
    signal) before the MSE. The simulation steps per frame are inferred from the
    two lengths, so the loss adapts to any dt / frame-rate combination. NaN
    entries in the target (invalid frames) are masked out of the MSE.

    The calcium model is stateful (it carries latent calcium across chunks), so
    this loss is too: :meth:`reset_state` forwards to the model and must be
    called at the start of each sequence. The trainer does this automatically for
    any loss exposing ``reset_state``.

    Args:
        calcium_model: A configured :class:`CalciumObservationModel` carrying the
            per-cell gain / baseline / kinetics. Owned as a submodule so it moves
            with ``.to(device)`` and its state resets with this loss.
    """

    required_inputs = ["output_spikes", "target_calcium"]
    requires_target = False

    def __init__(self, calcium_model: CalciumObservationModel):
        super().__init__()
        self.calcium_model = calcium_model
        # The exact frame-averaged prediction from the last forward, cached
        # (detached) so the trainer/plotter can show the real computed dF/F — with
        # the correct carried calcium state — instead of cold-re-rendering it.
        self.last_prediction = None

    def forward(
        self, output_spikes: torch.Tensor, target_calcium: torch.Tensor
    ) -> torch.Tensor:
        """Render spikes to dF/F, frame-average to the target grid, masked MSE.

        Args:
            output_spikes: Network spikes, shape ``(batch, time, n_cells)``.
            target_calcium: Measured dF/F, shape ``(batch, n_frames, n_cells)``
                with ``n_frames`` dividing ``time``. NaN marks invalid entries.
        """
        pred_dff = self.calcium_model(output_spikes)  # (batch, time, n_cells)
        batch, n_time, n_cells = pred_dff.shape
        n_frames = target_calcium.shape[1]
        if n_time % n_frames != 0:
            raise ValueError(
                f"prediction length {n_time} is not divisible by the number of "
                f"target frames {n_frames}; cannot frame-average to the imaging grid."
            )
        samples_per_frame = n_time // n_frames
        if samples_per_frame > 1:  # frame-average sim dt -> imaging grid
            pred_dff = pred_dff.reshape(
                batch, n_frames, samples_per_frame, n_cells
            ).mean(dim=2)

        # Cache the exact prediction (on the imaging grid) for plotting.
        self.last_prediction = pred_dff.detach()

        target = target_calcium.float()
        mask = torch.isfinite(target).float()  # invalid frames -> 0 weight
        diff = (pred_dff - torch.nan_to_num(target)) * mask
        return (diff**2).sum() / mask.sum().clamp(min=1.0)

    def reset_state(self):
        """Reset the underlying calcium model's carried latent state."""
        self.calcium_model.reset_state()
