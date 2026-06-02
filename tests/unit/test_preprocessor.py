"""Unit tests for the Preprocessor class."""

import torch
import pytest

from har.data.types import Window
from har.preprocess.preprocessor import Preprocessor


def _make_window(data: torch.Tensor, file_id: int = 1, user_id: int = 1, label: int = 0) -> Window:
    """Helper to create a Window with given data tensor."""
    return Window(file_id=file_id, user_id=user_id, data=data, label=label)


class TestPreprocessorInit:
    def test_default_config(self):
        pp = Preprocessor({})
        assert pp.nan_inf_to_zero is True
        assert pp.standardize is True
        assert pp.per_window_demean is False
        assert pp._mean is None
        assert pp._std is None

    def test_custom_config(self):
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": False, "per_window_demean": True})
        assert pp.nan_inf_to_zero is False
        assert pp.standardize is False
        assert pp.per_window_demean is True


class TestPreprocessorFit:
    def test_fit_computes_mean_std(self):
        """fit() should compute per-channel mean and std from training windows."""
        # Create 2 windows with known data
        data1 = torch.ones(300, 6) * 2.0
        data2 = torch.ones(300, 6) * 4.0
        windows = [_make_window(data1), _make_window(data2, file_id=2)]

        pp = Preprocessor({"standardize": True})
        pp.fit(windows)

        assert pp._mean is not None
        assert pp._std is not None
        assert pp._mean.shape == (6,)
        assert pp._std.shape == (6,)
        # Mean of [2, 4] = 3.0
        assert torch.allclose(pp._mean, torch.tensor([3.0] * 6))

    def test_fit_floors_std_at_1e8(self):
        """fit() should substitute 1.0 for std <= 1e-8 (R3.4)."""
        # All identical values → std = 0 → should become 1.0
        data = torch.ones(300, 6) * 5.0
        windows = [_make_window(data)]

        pp = Preprocessor({"standardize": True})
        pp.fit(windows)

        assert torch.allclose(pp._std, torch.tensor([1.0] * 6))

    def test_fit_empty_windows(self):
        """fit() with empty list should not crash."""
        pp = Preprocessor({"standardize": True})
        pp.fit([])
        assert pp._mean is None
        assert pp._std is None


class TestPreprocessorTransformTensor:
    def test_nan_inf_to_zero(self):
        """NaN and Inf values should be replaced with 0.0 (R3.5)."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": False, "per_window_demean": False})
        t = torch.ones(300, 6)
        t[0, 0] = float("nan")
        t[1, 1] = float("inf")
        t[2, 2] = float("-inf")

        result = pp.transform_tensor(t)

        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0
        assert result[2, 2] == 0.0
        # Other values unchanged
        assert result[3, 0] == 1.0

    def test_nan_inf_no_change_when_finite(self):
        """If no NaN/Inf present, tensor should be unchanged (R3.6)."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": False, "per_window_demean": False})
        t = torch.ones(300, 6) * 3.14

        result = pp.transform_tensor(t)

        assert torch.equal(result, t)

    def test_per_window_demean(self):
        """Per-window demean should subtract per-channel mean (R3.7)."""
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": False, "per_window_demean": True})
        t = torch.randn(300, 6) + 5.0  # offset from zero

        result = pp.transform_tensor(t)

        # Per-channel mean should be ~0
        channel_means = result.mean(dim=0)
        assert torch.allclose(channel_means, torch.zeros(6), atol=1e-5)

    def test_standardize(self):
        """Standardization should apply (x - mean) / std (R3.3)."""
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": True, "per_window_demean": False})
        pp._mean = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pp._std = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        t = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 300)
        result = pp.transform_tensor(t)

        # (x - mean) / std = 0 for all
        assert torch.allclose(result, torch.zeros(300, 6))

    def test_all_disabled_passthrough(self):
        """All flags disabled should return input unchanged (R3.8)."""
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": False, "per_window_demean": False})
        t = torch.randn(300, 6)

        result = pp.transform_tensor(t)

        # Should be the exact same object (bitwise identical)
        assert result is t

    def test_output_dtype_float32(self):
        """Output should always be float32 (R3.1)."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": False, "per_window_demean": False})
        t = torch.ones(300, 6, dtype=torch.float32)
        t[0, 0] = float("nan")

        result = pp.transform_tensor(t)

        assert result.dtype == torch.float32

    def test_output_shape_preserved(self):
        """Output shape should be [300, C] (R3.9)."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": True, "per_window_demean": True})
        pp._mean = torch.zeros(6)
        pp._std = torch.ones(6)
        t = torch.randn(300, 6)

        result = pp.transform_tensor(t)

        assert result.shape == (300, 6)

    def test_transform_order(self):
        """Transform should apply in order: NaN→0, demean, standardize."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": True, "per_window_demean": True})
        pp._mean = torch.zeros(6)
        pp._std = torch.tensor([2.0] * 6)

        t = torch.ones(300, 6) * 4.0
        t[0, 0] = float("nan")

        result = pp.transform_tensor(t)

        # After NaN→0: t[0,0] = 0, rest = 4.0
        # After demean: mean of col 0 = (0 + 4*299)/300 ≈ 3.9867, subtract that
        # After standardize: divide by 2.0
        assert result.shape == (300, 6)
        assert torch.isfinite(result).all()


class TestPreprocessorTransform:
    def test_transform_returns_new_windows(self):
        """transform() should return new Window objects with transformed data."""
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": False, "per_window_demean": True})
        data = torch.randn(300, 6) + 10.0
        windows = [_make_window(data, file_id=42, user_id=7, label=3)]

        result = pp.transform(windows)

        assert len(result) == 1
        assert result[0].file_id == 42
        assert result[0].user_id == 7
        assert result[0].label == 3
        # Data should be different (demeaned)
        assert not torch.equal(result[0].data, data)

    def test_transform_all_disabled_returns_same(self):
        """All flags disabled should return input list unchanged (R3.8)."""
        pp = Preprocessor({"nan_inf_to_zero": False, "standardize": False, "per_window_demean": False})
        data = torch.randn(300, 6)
        windows = [_make_window(data)]

        result = pp.transform(windows)

        assert result is windows


class TestPreprocessorStateDict:
    def test_state_dict_with_fitted_stats(self):
        """state_dict() should serialize mean/std and config."""
        pp = Preprocessor({"nan_inf_to_zero": True, "standardize": True, "per_window_demean": False})
        pp._mean = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pp._std = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        sd = pp.state_dict()

        assert sd["mean"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert sd["std"] == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert sd["config"]["nan_inf_to_zero"] is True
        assert sd["config"]["standardize"] is True
        assert sd["config"]["per_window_demean"] is False

    def test_state_dict_without_fit(self):
        """state_dict() before fit should have None for mean/std."""
        pp = Preprocessor({})
        sd = pp.state_dict()

        assert sd["mean"] is None
        assert sd["std"] is None

    def test_load_state_dict_restores(self):
        """load_state_dict() should restore mean/std from saved state."""
        pp = Preprocessor({})
        sd = {
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "std": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "config": {
                "nan_inf_to_zero": False,
                "standardize": True,
                "per_window_demean": True,
            },
        }

        pp.load_state_dict(sd)

        assert torch.allclose(pp._mean, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert torch.allclose(pp._std, torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        assert pp.nan_inf_to_zero is False
        assert pp.standardize is True
        assert pp.per_window_demean is True

    def test_roundtrip_state_dict(self):
        """state_dict → load_state_dict should produce equivalent preprocessor."""
        pp1 = Preprocessor({"nan_inf_to_zero": True, "standardize": True, "per_window_demean": True})
        data = torch.randn(300, 6)
        pp1.fit([_make_window(data)])

        sd = pp1.state_dict()

        pp2 = Preprocessor({})
        pp2.load_state_dict(sd)

        # Both should produce same output
        test_tensor = torch.randn(300, 6)
        out1 = pp1.transform_tensor(test_tensor.clone())
        out2 = pp2.transform_tensor(test_tensor.clone())
        assert torch.allclose(out1, out2)
