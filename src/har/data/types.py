"""Core data types for the HAR pipeline.

Frozen dataclasses for immutable data records and mutable dataclasses
for tracking state that evolves during a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Window:
    """One 5-minute recording (300 rows × 6 channels).

    Each Window corresponds to a single CSV file containing 1-second
    aggregates of wrist-worn accelerometer data.

    Attributes:
        file_id: Integer identifier parsed from the CSV filename stem.
        user_id: Integer parsed from the parent folder name (e.g., User_061 → 61).
        data: Tensor of shape [300, 6] with float32 dtype containing columns
              mean_x, mean_y, mean_z, std_x, std_y, std_z in that order.
        label: Integer class label in {0, 1, 2, 3, 4, 5} for train windows,
               or None for test windows.
    """

    file_id: int
    user_id: int
    data: torch.Tensor  # [300, 6] float32
    label: int | None  # None for test windows


@dataclass(frozen=True)
class FoldSplit:
    """A single train/val partition.

    Attributes:
        fold_index: 0..K-1 for KFold; 0 for RandomSplit.
        train_indices: Indices into the full Window list for training.
        val_indices: Indices into the full Window list for validation.
        strategy: One of "group_kfold" or "random_split".
        user_ids_train: Set of user IDs in the training partition (for invariant checks).
        user_ids_val: Set of user IDs in the validation partition (for invariant checks).
    """

    fold_index: int  # 0..K-1 for KFold; 0 for RandomSplit
    train_indices: tuple[int, ...]  # indices into the full Window list
    val_indices: tuple[int, ...]
    strategy: str  # "group_kfold" | "random_split"
    user_ids_train: frozenset[int]  # for invariant checks
    user_ids_val: frozenset[int]


@dataclass
class FoldMetrics:
    """Metrics computed on one validation fold.

    Attributes:
        macro_f1: Unweighted mean of per-class F1 scores across the six labels.
        per_class_f1: Per-class F1 scores (length 6).
        per_class_precision: Per-class precision scores (length 6).
        per_class_recall: Per-class recall scores (length 6).
        confusion_matrix: 6×6 confusion matrix as nested lists.
    """

    macro_f1: float
    per_class_f1: list[float]  # length 6
    per_class_precision: list[float]  # length 6
    per_class_recall: list[float]  # length 6
    confusion_matrix: list[list[int]]  # 6 × 6


@dataclass
class RunRecord:
    """Metadata for one experiment run.

    Attributes:
        run_id: UUID v4 string identifying this run.
        start_ts: ISO 8601 UTC timestamp when the run started.
        end_ts: ISO 8601 UTC timestamp when the run ended, or None if still running.
        status: One of "running", "success", "failed", "interrupted".
        seed: Integer seed used for reproducibility.
        code_version: Git SHA of the code at run time.
        config: Full PipelineConfig serialized as a dict.
        metrics: FoldMetrics serialized as a dict, or None if not yet computed.
        error: Error description string, or None if no error occurred.
    """

    run_id: str  # UUID v4
    start_ts: str  # ISO 8601 UTC
    end_ts: str | None
    status: str  # "running" | "success" | "failed" | "interrupted"
    seed: int
    code_version: str  # git SHA
    config: dict  # full PipelineConfig as dict
    metrics: dict | None  # FoldMetrics as dict, or None
    error: str | None


@dataclass
class CheckpointManifest:
    """Persisted alongside each checkpoint for reproducibility.

    Attributes:
        run_id: UUID v4 string identifying the run that produced this checkpoint.
        seed: Integer seed used during training.
        code_version: Output of ``git rev-parse HEAD`` at training time.
        config: Full PipelineConfig serialized as a dict.
        preprocessor_state: Standardization statistics (mean/std per channel).
        feature_config: Feature engineering configuration for predict-time mismatch check.
        epoch: Epoch number this checkpoint corresponds to.
        macro_f1: Macro F1 score achieved at this checkpoint's epoch.
    """

    run_id: str
    seed: int
    code_version: str  # git rev-parse HEAD
    config: dict  # full PipelineConfig as dict
    preprocessor_state: dict  # standardization stats
    feature_config: dict  # for predict-time mismatch check
    epoch: int  # epoch this checkpoint corresponds to
    macro_f1: float


@dataclass(frozen=True)
class DatasetSummary:
    """Summary statistics from data ingestion.

    Attributes:
        n_windows: Total number of windows loaded.
        n_users: Number of distinct users.
        user_ids: Sorted tuple of all user IDs.
        file_ids: Sorted tuple of all file IDs.
        class_distribution: Mapping from label to count (includes zeros for all 6 classes).
        per_user_counts: Mapping from user_id to number of windows for that user.
    """

    n_windows: int
    n_users: int
    user_ids: tuple[int, ...]
    file_ids: tuple[int, ...]
    class_distribution: dict[int, int]  # label → count, includes zeros
    per_user_counts: dict[int, int]  # user_id → count
