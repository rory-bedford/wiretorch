"""Model comparison metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def r_squared(true: NDArray, predicted: NDArray) -> float:
    """R² against the identity line (true == predicted).

    Args:
        true: Ground-truth values.
        predicted: Predicted/estimated values.

    Returns:
        R² value, or NaN when *true* has zero variance.
    """
    ss_res = np.sum((predicted - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fluctuation_r_squared(
    teacher_spikes: NDArray,
    student_spikes: NDArray,
    tau_ms: float,
    dt: float,
) -> float:
    """R² between Gaussian-smoothed spike trains (fluctuation match).

    Both arrays are smoothed along the time axis, then flattened and
    compared with :func:`r_squared`.

    Args:
        teacher_spikes: Shape ``(time, neurons)``.
        student_spikes: Shape ``(time, neurons)``.
        tau_ms: Gaussian kernel standard deviation in milliseconds.
        dt: Simulation timestep in milliseconds.
    """
    sigma = tau_ms / dt
    teacher_conv = gaussian_filter1d(teacher_spikes, sigma=sigma, axis=0)
    student_conv = gaussian_filter1d(student_spikes, sigma=sigma, axis=0)
    return r_squared(teacher_conv.ravel(), student_conv.ravel())
