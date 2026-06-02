"""Window-level feature extraction for gradient-boosted tree models.

Extracts a rich set of statistical, frequency, and shape features from each
5-minute window (300 one-second aggregates of 6 base channels).

CRITICAL: Does NOT remove gravity. The mean_x/y/z channels encode wrist
orientation (gravity direction), which is one of the strongest activity
discriminators (sitting/standing/lying have distinct gravity vectors).
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.fft import rfft


BASE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _stat_features(x, prefix):
    """Basic distributional statistics for a 1D signal."""
    feats = {}
    feats[f"{prefix}_mean"] = np.mean(x)
    feats[f"{prefix}_std"] = np.std(x)
    feats[f"{prefix}_min"] = np.min(x)
    feats[f"{prefix}_max"] = np.max(x)
    feats[f"{prefix}_range"] = np.max(x) - np.min(x)
    feats[f"{prefix}_median"] = np.median(x)
    feats[f"{prefix}_q10"] = np.percentile(x, 10)
    feats[f"{prefix}_q25"] = np.percentile(x, 25)
    feats[f"{prefix}_q75"] = np.percentile(x, 75)
    feats[f"{prefix}_q90"] = np.percentile(x, 90)
    feats[f"{prefix}_iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
    feats[f"{prefix}_mad"] = np.mean(np.abs(x - np.mean(x)))  # mean abs deviation
    feats[f"{prefix}_rms"] = np.sqrt(np.mean(x ** 2))
    feats[f"{prefix}_energy"] = np.sum(x ** 2) / len(x)
    # Robust skew/kurtosis
    std = np.std(x)
    if std > 1e-8:
        feats[f"{prefix}_skew"] = float(scipy_stats.skew(x))
        feats[f"{prefix}_kurt"] = float(scipy_stats.kurtosis(x))
    else:
        feats[f"{prefix}_skew"] = 0.0
        feats[f"{prefix}_kurt"] = 0.0
    # Zero crossing rate (around mean)
    centered = x - np.mean(x)
    feats[f"{prefix}_zcr"] = np.mean(np.abs(np.diff(np.sign(centered)))) / 2.0
    return feats


def _fft_features(x, prefix, top_k=3):
    """Frequency-domain features."""
    feats = {}
    n = len(x)
    x_centered = x - np.mean(x)
    fft_vals = np.abs(rfft(x_centered))
    freqs = np.fft.rfftfreq(n, d=1.0)  # 1 Hz sampling
    
    power = fft_vals ** 2
    total_power = power.sum() + 1e-12
    
    feats[f"{prefix}_spec_energy"] = total_power
    # Dominant frequency
    if len(power) > 1:
        dom_idx = np.argmax(power[1:]) + 1
        feats[f"{prefix}_dom_freq"] = freqs[dom_idx]
        feats[f"{prefix}_dom_mag"] = fft_vals[dom_idx]
        # 2nd and 3rd dominant frequencies
        order = np.argsort(power[1:])[::-1] + 1
        for r in range(1, min(top_k, len(order))):
            feats[f"{prefix}_dom_freq_{r}"] = freqs[order[r]]
            feats[f"{prefix}_dom_mag_{r}"] = fft_vals[order[r]]
    else:
        feats[f"{prefix}_dom_freq"] = 0.0
        feats[f"{prefix}_dom_mag"] = 0.0
    # Spectral entropy
    p_norm = power / total_power
    feats[f"{prefix}_spec_entropy"] = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
    # Spectral centroid
    feats[f"{prefix}_spec_centroid"] = float(np.sum(freqs * power) / total_power)
    # Spectral spread / bandwidth
    centroid = np.sum(freqs * power) / total_power
    feats[f"{prefix}_spec_spread"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    # Spectral rolloff (85% of energy)
    cumpow = np.cumsum(power) / total_power
    rolloff_idx = np.searchsorted(cumpow, 0.85)
    feats[f"{prefix}_spec_rolloff"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    # Finer-grained energy bands
    bands = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"{prefix}_pow_{lo}_{hi}"] = power[mask].sum() / total_power
    return feats


def _autocorr_features(x, prefix, lags=(1, 2, 5, 10)):
    """Autocorrelation at selected lags."""
    feats = {}
    x_centered = x - np.mean(x)
    var = np.sum(x_centered ** 2) + 1e-12
    for lag in lags:
        if lag < len(x):
            ac = np.sum(x_centered[:-lag] * x_centered[lag:]) / var
            feats[f"{prefix}_autocorr_{lag}"] = float(ac)
        else:
            feats[f"{prefix}_autocorr_{lag}"] = 0.0
    return feats


def extract_window_features(data):
    """Extract all features from a [300, 6] window.

    Args:
        data: numpy array [300, 6] with columns mean_x,mean_y,mean_z,std_x,std_y,std_z

    Returns:
        dict of feature_name -> value
    """
    data = np.asarray(data, dtype=np.float64)
    feats = {}

    # Per-channel features (6 base channels)
    for i, col in enumerate(BASE_COLS):
        x = data[:, i]
        feats.update(_stat_features(x, col))
        feats.update(_fft_features(x, col))
        feats.update(_autocorr_features(x, col))

    # --- Derived signals ---
    mean_xyz = data[:, :3]  # [300, 3]
    std_xyz = data[:, 3:6]

    # Magnitude of mean acceleration (orientation-aware, gravity intact)
    mag = np.linalg.norm(mean_xyz, axis=1)  # [300]
    feats.update(_stat_features(mag, "mag"))
    feats.update(_fft_features(mag, "mag"))
    feats.update(_autocorr_features(mag, "mag"))

    # Magnitude of std (movement intensity)
    std_mag = np.linalg.norm(std_xyz, axis=1)
    feats.update(_stat_features(std_mag, "stdmag"))

    # Signal magnitude area (SMA)
    feats["sma_mean"] = np.mean(np.sum(np.abs(mean_xyz), axis=1))
    feats["sma_std"] = np.mean(np.sum(np.abs(std_xyz), axis=1))

    # Tilt angles (orientation)
    tilt_xy = np.arctan2(mean_xyz[:, 0], mean_xyz[:, 1] + 1e-8)
    tilt_z = np.arctan2(np.sqrt(mean_xyz[:, 0]**2 + mean_xyz[:, 1]**2), mean_xyz[:, 2] + 1e-8)
    feats.update(_stat_features(tilt_xy, "tilt_xy"))
    feats.update(_stat_features(tilt_z, "tilt_z"))

    # Jerk (first difference of mean) — movement dynamics
    jerk = np.diff(mean_xyz, axis=0)  # [299, 3]
    jerk_mag = np.linalg.norm(jerk, axis=1)
    feats.update(_stat_features(jerk_mag, "jerk"))

    # Cross-axis correlations (posture/movement coupling)
    for (a, b, name) in [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]:
        xa, xb = mean_xyz[:, a], mean_xyz[:, b]
        if np.std(xa) > 1e-8 and np.std(xb) > 1e-8:
            feats[f"corr_mean_{name}"] = float(np.corrcoef(xa, xb)[0, 1])
        else:
            feats[f"corr_mean_{name}"] = 0.0

    # Overall window-level movement summary
    feats["total_movement"] = np.sum(jerk_mag)
    feats["mean_std_overall"] = np.mean(std_xyz)
    feats["max_std_overall"] = np.max(std_xyz)

    # Fraction of time "active" (std above threshold) — captures rest vs motion
    activity = std_mag > np.median(std_mag)
    feats["active_fraction"] = np.mean(activity)
    # Number of activity transitions
    feats["activity_transitions"] = np.sum(np.abs(np.diff(activity.astype(int))))

    # --- Extra discriminative features ---
    # Ratio of std energy to mean energy (movement vs posture)
    mean_energy = np.mean(np.sum(mean_xyz ** 2, axis=1)) + 1e-8
    std_energy = np.mean(np.sum(std_xyz ** 2, axis=1)) + 1e-8
    feats["std_to_mean_energy"] = std_energy / mean_energy

    # Dominant axis of orientation (which axis carries gravity)
    abs_mean = np.abs(mean_xyz).mean(axis=0)
    feats["dom_axis_0"] = abs_mean[0]
    feats["dom_axis_1"] = abs_mean[1]
    feats["dom_axis_2"] = abs_mean[2]
    feats["dom_axis_argmax"] = float(np.argmax(abs_mean))
    feats["orient_concentration"] = float(abs_mean.max() / (abs_mean.sum() + 1e-8))

    # Stability of orientation over time (low = stable posture, high = changing)
    feats["orient_drift"] = float(np.mean(np.std(mean_xyz, axis=0)))

    # Per-axis movement intensity percentiles (capture bursty vs steady motion)
    for i, col in enumerate(["sx", "sy", "sz"]):
        s = std_xyz[:, i]
        feats[f"{col}_p95"] = np.percentile(s, 95)
        feats[f"{col}_p05"] = np.percentile(s, 5)
        feats[f"{col}_active_frac"] = np.mean(s > np.median(s) + 1e-8)

    # Peak count in magnitude signal (rhythmic activities like walking)
    mag_centered = mag - np.mean(mag)
    mag_thresh = np.std(mag_centered)
    peaks = np.sum((mag_centered[1:-1] > mag_centered[:-2]) &
                   (mag_centered[1:-1] > mag_centered[2:]) &
                   (mag_centered[1:-1] > mag_thresh))
    feats["mag_peak_count"] = float(peaks)

    # Entropy of the magnitude histogram (signal complexity)
    hist, _ = np.histogram(mag, bins=20)
    hist = hist / (hist.sum() + 1e-12)
    feats["mag_hist_entropy"] = float(-np.sum(hist * np.log(hist + 1e-12)))

    return feats
