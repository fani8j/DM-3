"""Preprocessor with fit-on-train-only semantics.

Applies boolean-flagged preprocessing steps in a fixed order:
1. NaN/Inf → 0.0 (if nan_inf_to_zero enabled)
2. Per-window per-channel demean (if per_window_demean enabled)
3. Global standardization using training-fold statistics (if standardize enabled)

All operations preserve [300, C] shape and float32 dtype.
Deterministic: same input + same config → bitwise-identical output.
"""

from __future__ import annotations

import torch

from har.data.types import Window


class Preprocessor:
    """Applies boolean-flagged preprocessing steps with fit-on-train-only semantics."""

    def __init__(self, config: dict) -> None:
        """Initialize preprocessor from a config dict.

        Config keys:
            nan_inf_to_zero (bool): Replace NaN/Inf with 0.0. Default True.
            standardize (bool): Apply per-channel z-score normalization. Default True.
            per_window_demean (bool): Subtract per-window per-channel mean. Default False.
        """
        self.nan_inf_to_zero: bool = config.get("nan_inf_to_zero", True)
        self.standardize: bool = config.get("standardize", True)
        self.per_window_demean: bool = config.get("per_window_demean", False)
        self._mean: torch.Tensor | None = None  # [C]
        self._std: torch.Tensor | None = None  # [C]

    def fit(self, windows: list[Window]) -> None:
        """Compute per-channel mean/std from training windows only.

        Stacks all window data tensors into [N, 300, C], computes mean and std
        across all timesteps and all windows (dim=[0,1]) → shape [C].
        Floors std: where std <= 1e-8, substitute 1.0 to avoid division by zero (R3.4).
        """
        if not windows:
            return

        # Stack all window data: [N, 300, C]
        data = torch.stack([w.data for w in windows], dim=0)

        # Per-channel mean across all windows and timesteps: [C]
        self._mean = data.mean(dim=[0, 1])

        # Per-channel std across all windows and timesteps: [C]
        self._std = data.std(dim=[0, 1])

        # Floor: where std <= 1e-8, set std = 1.0 (R3.4)
        self._std = torch.where(self._std <= 1e-8, torch.ones_like(self._std), self._std)

    def transform(self, windows: list[Window]) -> list[Window]:
        """Apply preprocessing to a list of windows.

        Returns new Window objects with transformed data tensors.
        If all flags are disabled, returns input unchanged (R3.8).
        """
        # All-disabled fast path (R3.8): return input unchanged (bitwise identical)
        if not self.nan_inf_to_zero and not self.standardize and not self.per_window_demean:
            return windows

        result = []
        for w in windows:
            transformed_data = self.transform_tensor(w.data)
            # Create new Window with transformed data
            result.append(
                Window(
                    file_id=w.file_id,
                    user_id=w.user_id,
                    data=transformed_data,
                    label=w.label,
                )
            )
        return result

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform a single [300, C] tensor.

        Applies preprocessing in order:
        1. NaN/Inf → 0.0 (if enabled)
        2. Per-window demean (if enabled)
        3. Standardize using fitted stats (if enabled and fit has been called)

        Returns tensor with float32 dtype and [300, C] shape preserved (R3.1, R3.9).
        """
        # All-disabled fast path (R3.8)
        if not self.nan_inf_to_zero and not self.standardize and not self.per_window_demean:
            return tensor

        # Ensure float32 (R3.1)
        t = tensor.to(dtype=torch.float32)

        # Step 1: NaN/Inf → 0.0 (R3.5, R3.6)
        if self.nan_inf_to_zero:
            # Only modify if there are NaN or Inf values (R3.6: leave unchanged if none)
            has_nan = torch.isnan(t).any()
            has_inf = torch.isinf(t).any()
            if has_nan or has_inf:
                t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))

        # Step 2: Per-window demean (R3.7)
        if self.per_window_demean:
            # Subtract per-channel mean of this window
            t = t - t.mean(dim=0, keepdim=True)

        # Step 3: Standardize using fitted stats (R3.3)
        if self.standardize and self._mean is not None:
            t = (t - self._mean) / self._std

        return t

    def state_dict(self) -> dict:
        """Return serializable state for manifest persistence (R10.6).

        Returns a dict with:
            mean: list of per-channel means or None
            std: list of per-channel stds or None
            config: dict of preprocessing flags
        """
        return {
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
            "config": {
                "nan_inf_to_zero": self.nan_inf_to_zero,
                "standardize": self.standardize,
                "per_window_demean": self.per_window_demean,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state from a previously saved state_dict.

        Restores mean/std tensors and config flags from the dict.
        """
        if sd.get("mean") is not None:
            self._mean = torch.tensor(sd["mean"], dtype=torch.float32)
        else:
            self._mean = None

        if sd.get("std") is not None:
            self._std = torch.tensor(sd["std"], dtype=torch.float32)
        else:
            self._std = None

        # Restore config flags if present
        if "config" in sd:
            cfg = sd["config"]
            self.nan_inf_to_zero = cfg.get("nan_inf_to_zero", self.nan_inf_to_zero)
            self.standardize = cfg.get("standardize", self.standardize)
            self.per_window_demean = cfg.get("per_window_demean", self.per_window_demean)
