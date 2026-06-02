"""FFT-derived frequency-domain features for HAR pipeline.

Computes 3 statistics per base channel (6 channels) = 18 values per window:
1. Dominant frequency magnitude: magnitude of the largest FFT coefficient (excluding DC)
2. Spectral energy: sum of squared magnitudes of all FFT coefficients
3. Spectral entropy: entropy of the normalized power spectrum

Requirements: R4.6, R4.7
"""

import torch
import numpy as np


def compute_frequency_features(
    data: torch.Tensor,
    mode: str = "static",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute FFT-derived statistics for each of 6 base channels.

    For each channel, compute 3 statistics:
    1. Dominant frequency magnitude: magnitude of the largest FFT coefficient (excluding DC)
    2. Spectral energy: sum of squared magnitudes of all FFT coefficients
    3. Spectral entropy: entropy of the normalized power spectrum

    Total: 3 stats × 6 channels = 18 values per window.

    Args:
        data: [300, 6] tensor (base channels)
        mode: "broadcast" or "static"
            - "broadcast": tile 18 values across 300 timesteps → return (channels [300, 18], None)
            - "static": return (None, static_vector [18])

    Returns:
        (broadcast_channels, static_vector) — one is None depending on mode.

    Raises:
        ValueError: If input shape is not [300, 6] or mode is not "broadcast"/"static".

    R4.6
    """
    # Validate input shape
    if data.ndim != 2 or data.shape[0] != 300 or data.shape[1] != 6:
        raise ValueError(
            f"Expected input shape [300, 6], got {list(data.shape)}"
        )

    # Validate mode
    if mode not in ("broadcast", "static"):
        raise ValueError(
            f"Mode must be 'broadcast' or 'static', got '{mode}'"
        )

    # Convert to numpy for FFT computation
    data_np = data.numpy()

    # Compute 3 statistics for each of the 6 channels
    features = []
    for ch in range(6):
        signal = data_np[:, ch]

        # Compute real FFT
        fft_vals = np.fft.rfft(signal)
        magnitudes = np.abs(fft_vals)

        # 1. Dominant frequency magnitude (exclude DC component at index 0)
        dom_freq_mag = magnitudes[1:].max()

        # 2. Spectral energy: sum of squared magnitudes
        spectral_energy = (magnitudes ** 2).sum()

        # 3. Spectral entropy
        power = magnitudes ** 2
        power_norm = power / (power.sum() + 1e-12)  # avoid div by zero
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-12))

        features.extend([dom_freq_mag, spectral_energy, entropy])

    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Return based on mode
    if mode == "broadcast":
        # Tile to [300, 18]
        broadcast_tensor = features_tensor.unsqueeze(0).expand(300, -1).contiguous()
        return (broadcast_tensor, None)
    else:
        # Static mode: return [18] vector
        return (None, features_tensor)
