"""Feature_Engineer orchestrator for the HAR pipeline.

Calls temporal, frequency, and stretch feature functions and concatenates
results in the documented channel order (design.md §3 Channel Layout).

Requirements: R4.1, R4.7, R4.8, R5.5
"""

from __future__ import annotations

import torch

from har.features.temporal import (
    compute_magnitude,
    compute_std_magnitude,
    compute_first_difference,
    compute_rolling_stats,
)
from har.features.frequency import compute_frequency_features
from har.features.stretch import gravity_decompose


class FeatureEngineer:
    """Orchestrates feature engineering for a single Window.

    Concatenates derived channels in fixed documented order:
    1. Base channels [300, 6]
    2. Magnitude [300, 1] (if enabled)
    3. Std-magnitude [300, 1] (if enabled)
    4. First-difference [300, 6] (if enabled)
    5. Rolling-mean [300, 6] (if enabled)
    6. Rolling-std [300, 6] (if enabled)
    7. Frequency-domain broadcast [300, 18] (if enabled AND mode="broadcast")
    8. Gravity [300, 3] + Body-acc [300, 3] (if stretch.gravity enabled)

    Static vector (returned separately):
    - Frequency-domain static [18] (if enabled AND mode="static")
    - Wrist prior vector [F] (if stretch.wrist_prior enabled) — appended after FFT static

    Returns (seq_tensor, static_vector | None)
    """

    def __init__(self, config: dict) -> None:
        """Initialize from the [features] section of PipelineConfig.

        Args:
            config: Dictionary with feature toggle flags and parameters.
        """
        self.magnitude = config.get("magnitude", True)
        self.std_magnitude = config.get("std_magnitude", True)
        self.first_difference = config.get("first_difference", True)
        self.rolling_stats = config.get("rolling_stats", True)
        self.rolling_window = config.get("rolling_window", 5)
        self.frequency_domain = config.get("frequency_domain", True)
        self.freq_mode = config.get("freq_mode", "static")

        # Stretch features
        stretch = config.get("stretch", {})
        self.gravity_enabled = stretch.get("gravity", False)
        self.gravity_cutoff_hz = stretch.get("gravity_cutoff_hz", 0.3)
        self.wrist_prior_enabled = stretch.get("wrist_prior", False)
        self.wrist_prior_vector: list[float] = stretch.get("wrist_prior_vector", [])

    def transform(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform a single [300, 6] base-channel tensor.

        Args:
            data: [300, 6] float32 tensor (base channels)

        Returns:
            (seq_tensor, static_vector):
            - seq_tensor: [300, C] where C depends on enabled features
            - static_vector: [F] tensor or None (FFT static + optional wrist prior)

        Raises:
            ValueError: If input shape is not [300, 6] (R4.10).
        """
        if data.ndim != 2 or data.shape[0] != 300 or data.shape[1] != 6:
            raise ValueError(f"Expected input shape [300, 6], got {list(data.shape)}")

        channels: list[torch.Tensor] = [data]  # Start with base [300, 6]
        static_parts: list[torch.Tensor] = []

        # 1. Magnitude
        if self.magnitude:
            channels.append(compute_magnitude(data))  # [300, 1]

        # 2. Std-magnitude
        if self.std_magnitude:
            channels.append(compute_std_magnitude(data))  # [300, 1]

        # 3. First-difference
        if self.first_difference:
            channels.append(compute_first_difference(data))  # [300, 6]

        # 4 & 5. Rolling stats (mean + std concatenated as [300, 12])
        if self.rolling_stats:
            channels.append(compute_rolling_stats(data, self.rolling_window))  # [300, 12]

        # 6. Frequency-domain
        if self.frequency_domain:
            broadcast, static = compute_frequency_features(data, mode=self.freq_mode)
            if broadcast is not None:
                channels.append(broadcast)  # [300, 18]
            if static is not None:
                static_parts.append(static)  # [18]

        # 7. Gravity decomposition (stretch)
        if self.gravity_enabled:
            gravity, body_acc = gravity_decompose(data, cutoff_hz=self.gravity_cutoff_hz)
            channels.append(gravity)    # [300, 3]
            channels.append(body_acc)   # [300, 3]

        # 8. Wrist prior (stretch) — goes into static vector
        if self.wrist_prior_enabled and self.wrist_prior_vector:
            wrist_tensor = torch.tensor(self.wrist_prior_vector, dtype=torch.float32)
            static_parts.append(wrist_tensor)

        # Concatenate sequence channels along feature dimension
        seq_tensor = torch.cat(channels, dim=1)  # [300, C]

        # Concatenate static parts (or None if empty)
        static_vector = torch.cat(static_parts, dim=0) if static_parts else None

        return seq_tensor, static_vector

    def get_output_channels(self) -> int:
        """Compute the number of output sequence channels based on config.

        Returns:
            Integer count of channels in the seq_tensor output.
        """
        c = 6  # base
        if self.magnitude:
            c += 1
        if self.std_magnitude:
            c += 1
        if self.first_difference:
            c += 6
        if self.rolling_stats:
            c += 12
        if self.frequency_domain and self.freq_mode == "broadcast":
            c += 18
        if self.gravity_enabled:
            c += 6  # gravity [300, 3] + body_acc [300, 3]
        return c

    def get_static_dim(self) -> int:
        """Compute the dimension of the static feature vector.

        Returns:
            Integer dimension of the static vector, or 0 if no static features.
        """
        f = 0
        if self.frequency_domain and self.freq_mode == "static":
            f += 18
        if self.wrist_prior_enabled and self.wrist_prior_vector:
            f += len(self.wrist_prior_vector)
        return f
