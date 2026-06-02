"""Feature engineering configuration.

Re-exports FeaturesConfig from the main config module for convenience,
and provides feature-specific constants.
"""

from __future__ import annotations

from har.config import FeaturesConfig

__all__ = ["FeaturesConfig", "TEMPORAL_FEATURE_CHANNELS"]

# Channel counts added by each temporal feature group
TEMPORAL_FEATURE_CHANNELS = {
    "magnitude": 1,       # R4.2: sqrt(mean_x^2 + mean_y^2 + mean_z^2)
    "std_magnitude": 1,   # R4.3: sqrt(std_x^2 + std_y^2 + std_z^2)
    "first_difference": 6,  # R4.4: diff of 6 base channels
    "rolling_stats": 12,  # R4.5: 6 rolling-mean + 6 rolling-std
}
