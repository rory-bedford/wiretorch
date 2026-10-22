"""
Time-varying rate processes for inhomogeneous Poisson spike generation.

This module provides dataset classes that generate temporal firing rate dynamics,
which can be used as input to inhomogeneous Poisson spike generators. Each process
generates firing rates that evolve over time according to different stochastic or
deterministic dynamics.

Rate processes maintain continuous state across iterations, allowing for seamless
chunked generation of long time series.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from wiretorch.utils.recurrences import ar1_trajectory
from typing import Union, Tuple


class OrnsteinUhlenbeckRateProcess(Dataset):
    """
    Continuous Ornstein-Uhlenbeck process operating in pattern space.

    Generates smooth, mean-reverting rate trajectories by running independent OU
    processes for each pattern, then combining them as a weighted sum. The process
    maintains continuous state across iterations, yielding chunks of specified size
    from an ongoing trajectory.

    Each pattern undergoes its own OU dynamics (mean-reverting to zero)::

        da_i = -(a_i / tau) * dt + sigma_i * sqrt(2 * dt / tau) * dW_i

    where a_i is the activation of pattern i. The final firing rate is computed
    by applying softmax normalization to the activations, then taking a weighted
    sum of the patterns::

        weights_i(t) = softmax(a_i(t) / temperature)
        r(t) = sum_i [weights_i(t) * pattern_i]

    The softmax temperature controls the sharpness of the pattern mixture: lower
    values make the distribution spikier (one pattern dominates), higher values
    make it softer (patterns blend more uniformly).

    Args:
        patterns: Array of firing rate patterns, shape (n_patterns, n_neurons) in Hz.
            Each row is a spatial pattern that will be modulated over time.
        chunk_size: Number of timesteps per chunk/sample.
        dt: Timestep in milliseconds.
        tau: Timescale of mean reversion in milliseconds (shared across all patterns).
        temperature: Softmax temperature for normalizing pattern activations. Higher values
            make the distribution softer (more uniform), lower values sharpen it. Default: 1.0.
        sigma: Noise amplitude for each pattern. Shape (n_patterns,) or scalar. Default: 0.5.
        a_init: Initial activation for each pattern. Shape (n_patterns,) or scalar.
            If None, uses value 1.0 for all patterns. Default: None.
        return_rates: If True, __getitem__ returns tuple (rates, weights)
            for diagnostic/visualization purposes. If False, returns only rates. Default: False.
        batch_size: Number of independent trials. Default: 1.
        device: Device for all tensors ('cpu' or 'cuda'). Default: 'cpu'.

    Attributes:
        n_neurons: Number of neurons in each pattern.
        n_patterns: Number of patterns.
        chunk_size: Number of timesteps per chunk.
        dt: Timestep in milliseconds.
        activations: Current state of pattern activations, shape (n_patterns,).

    Example:
        >>> from wiretorch.dataloaders.input_patterns import (
        ...     generate_assembly_firing_rates,
        ... )
        >>> # One pattern per assembly: 20 patterns for 5000 inputs
        >>> patterns = generate_assembly_firing_rates(...)  # Shape: (20, 5000)
        >>>
        >>> # Continuous OU process in pattern space with softmax normalization
        >>> rate_process = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=100,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     temperature=1.0,
        ...     sigma=0.5,
        ...     device='cuda',
        ... )
        >>>
        >>> # Each call returns the next chunk from continuous trajectory
        >>> chunk1 = rate_process[0]  # Shape: (100, 5000)
        >>> chunk2 = rate_process[1]  # Shape: (100, 5000) - continues from chunk1
        >>>
        >>> # For diagnostic/visualization purposes, enable return_rates
        >>> rate_process_diag = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=100,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     temperature=1.0,
        ...     return_rates=True,
        ...     device='cuda',
        ... )
        >>> rates, weights = rate_process_diag[0]
        >>> # rates: (100, 5000), weights: (100, 20)
    """

    def __init__(
        self,
        patterns: Union[np.ndarray, torch.Tensor],
        chunk_size: int,
        dt: float,
        tau: float,
        temperature: float,
        sigma: Union[float, np.ndarray, torch.Tensor],
        a_init: Union[float, np.ndarray, torch.Tensor, None],
        return_rates: bool,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)

        # Convert patterns to tensor on target device
        if isinstance(patterns, np.ndarray):
            patterns = torch.from_numpy(patterns).float()
        self.patterns = patterns.to(self.device).float()  # (n_patterns, n_neurons)

        self.n_patterns, self.n_neurons = self.patterns.shape
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dt = dt
        self.tau = tau
        self.temperature = temperature

        # Convert sigma to tensor with shape (n_patterns,)
        self.sigma = self._to_pattern_tensor(sigma, self.n_patterns).to(self.device)

        # Store return_rates flag for diagnostic outputs
        self.return_rates = return_rates

        self.reset(a_init)  # activations, (batch_size, n_patterns)

        # Precompute OU parameters
        self.alpha = 1.0 - dt / tau  # AR(1) decay coefficient
        self.diffusion = (self.sigma * np.sqrt(2.0 * dt / tau)).to(
            self.device
        )  # (n_patterns,)

    def _to_pattern_tensor(
        self, value: Union[float, np.ndarray, torch.Tensor], n_patterns: int
    ) -> torch.Tensor:
        """Convert parameter to tensor of shape (n_patterns,)."""
        if isinstance(value, (int, float)):
            return torch.full((n_patterns,), float(value), dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()
        elif isinstance(value, torch.Tensor):
            value = value.float()

        if value.numel() == 1:
            return value.expand(n_patterns)
        elif value.shape[0] != n_patterns:
            raise ValueError(
                f"Parameter shape {value.shape} incompatible with "
                f"n_patterns={n_patterns}"
            )

        return value

    def __len__(self) -> int:
        """Return arbitrary large number for infinite generation."""
        return int(1e9)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate next chunk of OU rate trajectory in pattern space.

        The activations follow ``s[t+1] = alpha s[t] + diffusion dW[t]``, evaluated in
        closed form (see ``ar1_trajectory``).

        Args:
            idx: Index (ignored, process continues from current state).

        Returns:
            If return_rates=False:
                Rate trajectory chunk of shape (batch_size, chunk_size, n_neurons) in Hz.
            If return_rates=True:
                Tuple of (rates, weights) where:
                - rates: shape (batch_size, chunk_size, n_neurons) in Hz
                - weights: shape (batch_size, chunk_size, n_patterns) - normalized mixing weights
        """
        B, T, P = self.batch_size, self.chunk_size, self.n_patterns
        noise = self.diffusion * torch.randn(B, T, P, device=self.device)
        activation_chunk, self.activations = ar1_trajectory(
            self.activations, noise, self.alpha
        )

        # Combine patterns using softmax normalization of activations
        normalized_activations = torch.softmax(
            activation_chunk / self.temperature, dim=2
        )  # (B, T, P)

        # Weighted sum of patterns: (B, T, P) @ (P, N) -> (B, T, N)
        rates = normalized_activations @ self.patterns

        if self.return_rates:
            return rates, normalized_activations
        else:
            return rates

    def reset(self, a_init: Union[float, np.ndarray, torch.Tensor, None] = None):
        """
        Reset the process state to initial conditions.

        Args:
            a_init: New initial activation. If None, resets to ones.
        """
        if a_init is None:
            self.activations = torch.ones(
                (self.batch_size, self.n_patterns),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            base = self._to_pattern_tensor(a_init, self.n_patterns).to(self.device)
            self.activations = base.unsqueeze(0).expand(self.batch_size, -1).clone()
