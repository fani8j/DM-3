"""Stretch features: gravity decomposition and wrist-placement prior.

Implements optional domain signals (R15) that can be enabled independently
to explore improvements beyond the baseline pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, filtfilt


def gravity_decompose(
    data: torch.Tensor,
    cutoff_hz: float = 0.3,
    fs: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose mean acceleration channels into gravity and body-acceleration.

    Applies a 4th-order Butterworth low-pass filter at cutoff_hz to each of
    mean_x, mean_y, mean_z (first 3 channels of data).

    Args:
        data: [300, 6] tensor (base channels)
        cutoff_hz: Low-pass cutoff frequency (default 0.3 Hz)
        fs: Sampling frequency (default 1.0 Hz since data is 1-second aggregates)

    Returns:
        (gravity, body_acc) where:
        - gravity: [300, 3] — low-frequency component of mean_x, mean_y, mean_z
        - body_acc: [300, 3] — residual = original - gravity

    Raises:
        ValueError: If output contains non-finite values (R15.4)

    R15.1: gravity + body_acc == original (within floating-point tolerance)
    """
    mean_channels = data[:, :3].numpy()  # [300, 3]

    # Design 4th-order Butterworth low-pass filter
    nyquist = fs / 2.0
    # Handle edge case where cutoff >= nyquist
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(4, normalized_cutoff, btype="low")

    # Apply filter to each channel
    gravity_np = np.zeros_like(mean_channels)
    for ch in range(3):
        gravity_np[:, ch] = filtfilt(b, a, mean_channels[:, ch])

    body_acc_np = mean_channels - gravity_np

    # Check for non-finite values (R15.4)
    if not np.all(np.isfinite(gravity_np)):
        raise ValueError("Gravity decomposition produced non-finite values")
    if not np.all(np.isfinite(body_acc_np)):
        raise ValueError("Body acceleration computation produced non-finite values")

    gravity = torch.from_numpy(gravity_np.astype(np.float32))
    body_acc = torch.from_numpy(body_acc_np.astype(np.float32))

    return gravity, body_acc


def validate_wrist_prior(
    prior_vector: list[float], expected_size: int
) -> torch.Tensor:
    """Validate and convert wrist prior vector.

    Args:
        prior_vector: List of float values from config
        expected_size: Expected length (must match Model's auxiliary input size)

    Returns:
        Tensor of shape [expected_size]

    Raises:
        ValueError: If prior_vector is empty or length doesn't match
                    expected_size (R15.5)
    """
    if not prior_vector:
        raise ValueError("wrist_prior_vector is empty but wrist_prior is enabled")
    if len(prior_vector) != expected_size:
        raise ValueError(
            f"wrist_prior_vector length ({len(prior_vector)}) does not match "
            f"Model auxiliary input size ({expected_size})"
        )
    return torch.tensor(prior_vector, dtype=torch.float32)
