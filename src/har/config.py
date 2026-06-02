"""Pipeline configuration schema and TOML loader.

Provides a pydantic-based configuration schema that mirrors configs/default.toml,
with range validators for all hyperparameters and a load_config() function that
reads TOML, applies dot-notation overrides, and returns a validated PipelineConfig.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic import StrictBool


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration validation fails.

    Attributes:
        field: The configuration field that failed validation.
        value: The invalid value that was provided.
        message: A human-readable description of the error.
    """

    def __init__(self, field: str, value: object, message: str) -> None:
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"ConfigError [{field}={value!r}]: {message}")


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_dir: str = "train/train"
    test_dir: str = "test/test"
    sample_submission: str = "sample_submission.csv"
    experiments_dir: str = "experiments"
    checkpoints_dir: str = "checkpoints"
    submissions_dir: str = "submissions"


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nan_inf_to_zero: StrictBool = True
    standardize: StrictBool = True
    per_window_demean: StrictBool = False


class StretchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gravity: StrictBool = False
    gravity_cutoff_hz: float = 0.3
    wrist_prior: StrictBool = False
    wrist_prior_vector: list[float] = []


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    magnitude: StrictBool = True
    std_magnitude: StrictBool = True
    first_difference: StrictBool = True
    rolling_stats: StrictBool = True
    rolling_window: int = 5
    frequency_domain: StrictBool = True
    freq_mode: Literal["broadcast", "static"] = "static"
    stretch: StretchConfig = StretchConfig()

    @field_validator("rolling_window")
    @classmethod
    def _validate_rolling_window(cls, v: int) -> int:
        if v < 2 or v > 300:
            raise ValueError("rolling_window must be in [2, 300]")
        return v

    @field_validator("freq_mode")
    @classmethod
    def _validate_freq_mode(cls, v: str) -> str:
        if v not in ("broadcast", "static"):
            raise ValueError('freq_mode must be "broadcast" or "static"')
        return v


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arch: Literal["tcn", "bigru", "transformer"] = "tcn"
    hidden_dim: int = 128
    depth: int = 6
    kernel_size: int = 5
    dropout: float = 0.2
    nhead: int = 4

    @field_validator("hidden_dim")
    @classmethod
    def _validate_hidden_dim(cls, v: int) -> int:
        if v < 8 or v > 2048:
            raise ValueError("hidden_dim must be in [8, 2048]")
        return v

    @field_validator("depth")
    @classmethod
    def _validate_depth(cls, v: int) -> int:
        if v < 1 or v > 12:
            raise ValueError("depth must be in [1, 12]")
        return v

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, v: float) -> float:
        if v < 0.0 or v > 0.9:
            raise ValueError("dropout must be in [0.0, 0.9]")
        return v

    @field_validator("kernel_size")
    @classmethod
    def _validate_kernel_size(cls, v: int) -> int:
        if v < 1 or v > 31:
            raise ValueError("kernel_size must be in [1, 31]")
        if v % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return v

    @field_validator("nhead")
    @classmethod
    def _validate_nhead(cls, v: int) -> int:
        if v < 1 or v > 32:
            raise ValueError("nhead must be in [1, 32]")
        return v


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 64
    epochs: int = 60
    optimizer: Literal["sgd", "adam", "adamw"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: Literal["constant", "step", "cosine"] = "cosine"
    class_weighted_loss: StrictBool = True

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        if v < 1 or v > 4096:
            raise ValueError("batch_size must be in [1, 4096]")
        return v

    @field_validator("lr")
    @classmethod
    def _validate_lr(cls, v: float) -> float:
        if v < 1e-6 or v > 1.0:
            raise ValueError("lr must be in [1e-6, 1.0]")
        return v

    @field_validator("weight_decay")
    @classmethod
    def _validate_weight_decay(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("weight_decay must be in [0.0, 1.0]")
        return v

    @field_validator("epochs")
    @classmethod
    def _validate_epochs(cls, v: int) -> int:
        if v < 1 or v > 1000:
            raise ValueError("epochs must be in [1, 1000]")
        return v


class GroupKFoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_splits: int = 5

    @field_validator("n_splits")
    @classmethod
    def _validate_n_splits(cls, v: int) -> int:
        if v < 2 or v > 60:
            raise ValueError("n_splits must be in [2, 60]")
        return v


class RandomSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_fraction: float = 0.8
    seed: int = 42

    @field_validator("train_fraction")
    @classmethod
    def _validate_train_fraction(cls, v: float) -> float:
        if v < 0.5 or v > 0.95:
            raise ValueError("train_fraction must be in [0.5, 0.95]")
        return v


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: list[str] = ["group_kfold", "random_split"]
    group_kfold: GroupKFoldConfig = GroupKFoldConfig()
    random_split: RandomSplitConfig = RandomSplitConfig()


class PredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ensemble_checkpoints: list[str] = []
    ensemble_weights: list[float] = []


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    reproducibility_mode: StrictBool = True
    device: str = "cuda"

    paths: PathsConfig = PathsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    validation: ValidationConfig = ValidationConfig()
    predict: PredictConfig = PredictConfig()

    @model_validator(mode="after")
    def _validate_ensemble_weights(self) -> "PipelineConfig":
        weights = self.predict.ensemble_weights
        checkpoints = self.predict.ensemble_checkpoints
        if weights and checkpoints and len(weights) != len(checkpoints):
            raise ValueError(
                f"ensemble_weights length ({len(weights)}) must match "
                f"ensemble_checkpoints length ({len(checkpoints)})"
            )
        return self


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict:
    """Read a TOML file and return the parsed dict."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError(
                "tomli is required on Python < 3.11. Install it with: pip install tomli"
            ) from e

    with open(path, "rb") as f:
        return tomllib.load(f)


def _apply_overrides(data: dict, overrides: dict) -> dict:
    """Apply dot-notation key=value overrides to a nested dict.

    For example, overrides={"model.arch": "bigru"} sets data["model"]["arch"] = "bigru".
    Values are coerced: "true"/"false" → bool, numeric strings → int/float.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Coerce string values to appropriate types
        if isinstance(value, str):
            lower = value.lower()
            if lower == "true":
                value = True
            elif lower == "false":
                value = False
            else:
                # Try int first, then float
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
        target[parts[-1]] = value
    return data


def load_config(path: Path, overrides: dict | None = None) -> PipelineConfig:
    """Load and validate a pipeline configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.
        overrides: Optional dict of dot-notation key=value overrides
                   (e.g., {"model.arch": "bigru", "train.lr": "0.01"}).

    Returns:
        A validated PipelineConfig instance.

    Raises:
        ConfigError: If the file cannot be read or validation fails.
    """
    try:
        data = _read_toml(path)
    except FileNotFoundError:
        raise ConfigError(
            field="path",
            value=str(path),
            message=f"Configuration file not found: {path}",
        )
    except Exception as e:
        raise ConfigError(
            field="path",
            value=str(path),
            message=f"Failed to parse TOML file: {e}",
        )

    if overrides:
        data = _apply_overrides(data, overrides)

    try:
        return PipelineConfig(**data)
    except Exception as e:
        # Extract field info from pydantic validation errors if possible
        error_msg = str(e)
        field = "unknown"
        value = None

        # Try to extract the first field from pydantic ValidationError
        if hasattr(e, "errors"):
            errors = e.errors()  # type: ignore[union-attr]
            if errors:
                first = errors[0]
                loc = first.get("loc", ())
                field = ".".join(str(part) for part in loc) if loc else "unknown"
                value = first.get("input")
                error_msg = first.get("msg", error_msg)

        raise ConfigError(
            field=field,
            value=value,
            message=error_msg,
        ) from e
