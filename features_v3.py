"""Enhanced feature extraction v3 — builds on v2 with features targeting the
intensity-continuum classes (1/2/3/5) that the v2 features couldn't separate.

New feature groups:
- Cross-correlation between axes at multiple lags (movement coupling)
- Posture-transition features (how orientation changes over the window)
- Per-segment statistics (split window into 4 segments, capture temporal evolution)
- Spectral features on jerk and derived magnitude signals
- Movement-burst statistics (run-length of active periods)
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.fft import rfft

from features_lgbm import extract_window_features, BASE_COLS


def _segment_features(data, n_seg=4):
    """Split window into segments and capture temporal evolution of key stats."""
    feats = {}
    mean_xyz = data[:, :3]
    std_xyz = data[:, 3:6]
    seg_len = len(data) // n_seg

    seg_movement = []
    seg_orient = []
    for s in range(n_seg):
        lo = s * seg_len
        hi = (s + 1) * seg_len if s < n_seg - 1 else len(data)
        seg = data[lo:hi]
        mv = np.mean(np.linalg.norm(seg[:, 3:6], axis=1))
        seg_movement.append(mv)
        seg_orient.append(seg[:, :3].mean(axis=0))

    seg_movement = np.array(seg_movement)
    # Temporal trend of movement (increasing/decreasing activity)
    feats["seg_mv_trend"] = float(np.polyfit(range(n_seg), seg_movement, 1)[0])
    feats["seg_mv_std"] = float(np.std(seg_movement))
    feats["seg_mv_range"] = float(seg_movement.max() - seg_movement.min())
    feats["seg_mv_first"] = float(seg_movement[0])
    feats["seg_mv_last"] = float(seg_movement[-1])

    # Orientation change across segments (posture transitions)
    seg_orient = np.array(seg_orient)  # [n_seg, 3]
    orient_changes = np.linalg.norm(np.diff(seg_orient, axis=0), axis=1)
    feats["seg_orient_total_change"] = float(orient_changes.sum())
    feats["seg_orient_max_change"] = float(orient_changes.max())
    return feats


def _crosscorr_features(data):
    """Cross-correlation between axes at multiple lags."""
    feats = {}
    mean_xyz = data[:, :3]
    for (a, b, name) in [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]:
        xa = mean_xyz[:, a] - mean_xyz[:, a].mean()
        xb = mean_xyz[:, b] - mean_xyz[:, b].mean()
        denom = np.sqrt(np.sum(xa**2) * np.sum(xb**2)) + 1e-12
        for lag in [1, 5, 10]:
            if lag < len(xa):
                cc = np.sum(xa[:-lag] * xb[lag:]) / denom
                feats[f"xcorr_{name}_{lag}"] = float(cc)
            else:
                feats[f"xcorr_{name}_{lag}"] = 0.0
    return feats


def _burst_features(data):
    """Movement-burst run-length statistics."""
    feats = {}
    std_mag = np.linalg.norm(data[:, 3:6], axis=1)
    thresh = np.median(std_mag)
    active = (std_mag > thresh).astype(int)

    # Run lengths of active periods
    runs = []
    cur = 0
    for a in active:
        if a == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)

    if runs:
        feats["burst_count"] = len(runs)
        feats["burst_mean_len"] = float(np.mean(runs))
        feats["burst_max_len"] = float(np.max(runs))
        feats["burst_std_len"] = float(np.std(runs))
    else:
        feats["burst_count"] = 0
        feats["burst_mean_len"] = 0.0
        feats["burst_max_len"] = 0.0
        feats["burst_std_len"] = 0.0
    return feats


def _jerk_spectral(data):
    """Spectral features on the jerk-magnitude signal (movement smoothness)."""
    feats = {}
    mean_xyz = data[:, :3]
    jerk = np.diff(mean_xyz, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    jm = jerk_mag - jerk_mag.mean()
    fft_vals = np.abs(rfft(jm))
    freqs = np.fft.rfftfreq(len(jm), d=1.0)
    power = fft_vals ** 2
    total = power.sum() + 1e-12
    feats["jerk_spec_centroid"] = float(np.sum(freqs * power) / total)
    feats["jerk_spec_entropy"] = float(-np.sum((power/total) * np.log(power/total + 1e-12)))
    if len(power) > 1:
        feats["jerk_dom_freq"] = float(freqs[np.argmax(power[1:]) + 1])
    else:
        feats["jerk_dom_freq"] = 0.0
    # Smoothness ratio (low-freq vs high-freq jerk energy)
    lowf = power[freqs < 0.1].sum()
    highf = power[freqs >= 0.1].sum() + 1e-12
    feats["jerk_lf_hf_ratio"] = float(lowf / highf)
    return feats


def extract_window_features_v3(data):
    """v2 features + v3 enhancements."""
    data = np.asarray(data, dtype=np.float64)
    feats = extract_window_features(data)  # all v2 features
    feats.update(_segment_features(data))
    feats.update(_crosscorr_features(data))
    feats.update(_burst_features(data))
    feats.update(_jerk_spectral(data))
    return feats
