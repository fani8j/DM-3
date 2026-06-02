"""Temporal feature groups for the HAR pipeline.

Computes derived temporal features from the 6 base accelerometer channels:
- Magnitude: Euclidean norm of (mean_x, mean_y, mean_z) → +1 channel (R4.2)
- Std-magnitude: Euclidean norm of (std_x, std_y, std_z) → +1 channel (R4.3)
- First-difference: Discrete first difference of all 6 channels → +6 channels (R4.4)
- Rolling statistics: Rolling mean + rolling std for all 6 channels → +12 channels (R4.5)

All functions validate input shape [300, 6] (R4.10) and rolling window range [2, 300] (R4.9).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

_EXPECTED_ROWS = 300
_EXPECTED_COLS = 6


def _validate_input_shape(data: torch.Tensor) -> None:
    """Validate that input tensor has shape [300, 6].

    Raises:
        ValueError: If shape does not match [300, 6] (R4.10).
    """
    if data.ndim != 2 or data.shape[0] != _EXPECTED_ROWS or data.shape[1] != _EXPECTED_COLS:
        raise ValueError(
            f"Expected input shape [{_EXPECTED_ROWS}, {_EXPECTED_COLS}], "
            f"got {list(data.shape)}"
        )


def _validate_window(window: int) -> None:
    """Validate that rolling window is in [2, 300].

    Raises:
        ValueError: If window is outside [2, 300] (R4.9).
    """
    if window < 2 or window > 300:
        raise ValueError(
            f"rolling_window must be in [2, 300], got {window}"
        )


# ---------------------------------------------------------------------------
# Temporal feature functions
# ---------------------------------------------------------------------------


def compute_magnitude(data: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean norm of (mean_x, mean_y, mean_z) per row.

    Input: [300, 6] (base channels)
    Output: [300, 1] — the magnitude channel
    R4.2: sqrt(mean_x^2 + mean_y^2 + mean_z^2)
    """
    _validate_input_shape(data)
    return torch.norm(data[:, :3], dim=1, keepdim=True)


def compute_std_magnitude(data: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean norm of (std_x, std_y, std_z) per row.

    Input: [300, 6] (base channels)
    Output: [300, 1] — the std-magnitude channel
    R4.3: sqrt(std_x^2 + std_y^2 + std_z^2)
    """
    _validate_input_shape(data)
    return torch.norm(data[:, 3:6], dim=1, keepdim=True)


def compute_first_difference(data: torch.Tensor) -> torch.Tensor:
    """Compute discrete first difference of each of the 6 base channels along time.

    Input: [300, 6]
    Output: [300, 6] — first row padded with 0.0
    R4.4: diff[t] = data[t] - data[t-1], diff[0] = 0.0
    """
    _validate_input_shape(data)
    diff = torch.zeros_like(data)
    diff[1:] = data[1:] - data[:-1]
    return diff


def compute_rolling_stats(data: torch.Tensor, window: int) -> torch.Tensor:
    """Compute rolling mean and rolling std for each of 6 base channels.

    Input: [300, 6], window W where 2 <= W <= 300
    Output: [300, 12] — 6 rolling-mean channels followed by 6 rolling-std channels
    Leading W-1 rows padded with 0.0

    R4.5: For each channel, rolling_mean[t] = mean(data[t-W+1:t+1]) for t >= W-1,
           rolling_std[t] = std(data[t-W+1:t+1]) for t >= W-1, else 0.0.
    Uses unbiased std (Bessel's correction, ddof=1) for rolling_std.
    """
    _validate_input_shape(data)
    _validate_window(window)

    rows, cols = data.shape  # 300, 6
    rolling_mean = torch.zeros(rows, cols, dtype=data.dtype, device=data.device)
    rolling_std = torch.zeros(rows, cols, dtype=data.dtype, device=data.device)

    # Use unfold to create sliding windows: [300, 6] -> transpose to [6, 300]
    # then unfold along dim=1 with size=window -> [6, 300-window+1, window]
    data_t = data.t()  # [6, 300]
    # unfold(dimension, size, step) -> [6, num_windows, window]
    unfolded = data_t.unfold(1, window, 1)  # [6, 300-window+1, window]

    # Compute mean and std along the window dimension
    means = unfolded.mean(dim=2)  # [6, 300-window+1]
    stds = unfolded.std(dim=2, correction=1)  # [6, 300-window+1] with Bessel's correction

    # Place results starting at index window-1 (leading W-1 rows stay 0)
    rolling_mean[window - 1:, :] = means.t()
    rolling_std[window - 1:, :] = stds.t()

    # Concatenate: 6 rolling-mean channels followed by 6 rolling-std channels
    return torch.cat([rolling_mean, rolling_std], dim=1)  # [300, 12]
