"""Linear recurrences evaluated in closed form, without a Python loop over time."""

import numpy as np
import torch


def ar1_trajectory(
    s0: torch.Tensor,
    noise: torch.Tensor,
    alpha: float | torch.Tensor,
    min_power: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The AR(1) process ``s[t+1] = alpha s[t] + noise[t]`` from ``s[0] = s0``.

    Args:
        s0: Initial state, ``(B, P)``.
        noise: Input per step, ``(B, T, P)``.
        alpha: Decay per step: a scalar, or one per state ``(P,)``.

    Returns the states ``s[0..T-1]`` ``(B, T, P)`` and ``s[T]`` ``(B, P)``. Each block of
    steps uses the closed form ``s[t] = alpha^t s[0] + alpha^(t-1) cumsum(noise[k] /
    alpha^k)``, with blocks short enough that ``alpha^k >= min_power``, so ``1/alpha^k``
    neither loses precision nor overflows however long the input.
    """
    B, T, P = noise.shape
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=noise.device)
    smallest = float(alpha.min())
    block = max(1, int(np.log(min_power) / np.log(smallest))) if 0 < smallest < 1 else T
    states, s = [], s0
    for start in range(0, T, block):
        piece = noise[:, start : start + block]
        L = piece.shape[1]
        steps = torch.arange(L + 1, device=noise.device, dtype=torch.float32)
        powers = alpha ** steps.view(-1, *([1] * alpha.ndim))  # (L + 1[, P])
        powers = powers.view(L + 1, -1)  # (L + 1, 1 or P)
        cumsum = torch.cumsum(piece * (1.0 / powers[:L]), dim=1)  # noise / alpha^k
        out = powers[:L] * s[:, None, :]
        out = torch.cat(
            [out[:, :1], out[:, 1:] + powers[: L - 1] * cumsum[:, : L - 1]], 1
        )
        states.append(out)
        s = powers[L] * s + powers[L - 1] * cumsum[:, -1]
    return torch.cat(states, dim=1), s
