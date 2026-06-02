"""Data loader for HAR pipeline.

Walks a dataset root directory, reads per-user CSV files into Window objects,
validates per-file integrity, and returns a deterministic ordered list of Windows
along with a DatasetSummary.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from har.data.types import DatasetSummary, Window

logger = logging.getLogger(__name__)

# Column names for the 6 base sensor channels, in canonical order.
BASE_COLUMNS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

# Pattern to match User_NNN folder names.
_USER_DIR_RE = re.compile(r"^User_(\d+)$")


class DataIntegrityError(Exception):
    """Raised when dataset validation fails.

    Attributes:
        file_id: The file_id of the offending file, or None for directory-level errors.
        message: A human-readable description of the error.
    """

    def __init__(self, file_id: int | None, message: str) -> None:
        self.file_id = file_id
        self.message = message
        super().__init__(message)


def load_dataset(
    root: Path,
    *,
    expect_label: bool,
    sample_submission_path: Path | None = None,
) -> tuple[list[Window], DatasetSummary]:
    """Load all Windows from a dataset root directory.

    Args:
        root: Path to the dataset root (e.g. train/train or test/test).
        expect_label: If True, expect and validate a 'label' column (train mode).
        sample_submission_path: Path to sample_submission.csv (unused here but
            accepted for interface compatibility).

    Returns:
        A tuple of (list of Window objects sorted by (user_id, file_id),
        DatasetSummary with statistics).

    Raises:
        DataIntegrityError: On any validation failure.
    """
    # Step 1: Verify root exists and contains valid User_NNN subdirectories.
    if not root.exists() or not root.is_dir():
        raise DataIntegrityError(
            file_id=None,
            message=f"Dataset root does not exist or is not a directory: {root}",
        )

    # Discover User_NNN directories.
    user_dirs: list[tuple[int, Path]] = []
    for entry in root.iterdir():
        if entry.is_dir():
            match = _USER_DIR_RE.match(entry.name)
            if match:
                user_id = int(match.group(1))
                # Check that the directory contains at least one CSV file.
                csv_files = list(entry.glob("*.csv"))
                if csv_files:
                    user_dirs.append((user_id, entry))

    if not user_dirs:
        raise DataIntegrityError(
            file_id=None,
            message=(
                f"Dataset root '{root}' contains no User_NNN subdirectories "
                "with at least one .csv file."
            ),
        )

    # Sort by user_id ascending for deterministic ordering (R1.9).
    user_dirs.sort(key=lambda x: x[0])

    # Step 2 & 3: Walk directories and load each CSV file.
    windows: list[Window] = []
    per_user_counts: dict[int, int] = {}

    for user_id, user_path in user_dirs:
        # Collect CSV files and sort by integer filename stem ascending.
        csv_files = list(user_path.glob("*.csv"))
        csv_with_ids: list[tuple[int, Path]] = []
        for csv_path in csv_files:
            try:
                file_id = int(csv_path.stem)
            except ValueError:
                # Skip non-integer-named CSV files silently.
                continue
            csv_with_ids.append((file_id, csv_path))

        csv_with_ids.sort(key=lambda x: x[0])
        user_window_count = 0

        for file_id, csv_path in csv_with_ids:
            window = _load_single_csv(
                csv_path, file_id=file_id, user_id=user_id, expect_label=expect_label
            )
            windows.append(window)
            user_window_count += 1

        per_user_counts[user_id] = user_window_count

    # Step 4: Build DatasetSummary.
    all_user_ids = sorted(per_user_counts.keys())
    all_file_ids = sorted(w.file_id for w in windows)

    # Class distribution: count each label 0-5 (include zeros for missing labels).
    class_distribution: dict[int, int] = {i: 0 for i in range(6)}
    if expect_label:
        for w in windows:
            if w.label is not None:
                class_distribution[w.label] += 1

    summary = DatasetSummary(
        n_windows=len(windows),
        n_users=len(all_user_ids),
        user_ids=tuple(all_user_ids),
        file_ids=tuple(all_file_ids),
        class_distribution=class_distribution,
        per_user_counts=per_user_counts,
    )

    # Step 5: Log class distribution at INFO level (R2.3).
    if expect_label:
        dist_str = ", ".join(
            f"label {k}: {v}" for k, v in sorted(class_distribution.items())
        )
        logger.info("Class distribution: %s", dist_str)

    return windows, summary


def _load_single_csv(
    csv_path: Path,
    *,
    file_id: int,
    user_id: int,
    expect_label: bool,
) -> Window:
    """Load and validate a single CSV file into a Window.

    Raises:
        DataIntegrityError: On any validation failure.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns exist (R1.8).
    required_cols = list(BASE_COLUMNS)
    if expect_label:
        required_cols.extend(["label", "file_id"])

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataIntegrityError(
            file_id=file_id,
            message=(
                f"File {file_id} (user {user_id}) is missing required columns: "
                f"{missing_cols}"
            ),
        )

    # Validate exactly 300 rows (R1.7).
    if len(df) != 300:
        raise DataIntegrityError(
            file_id=file_id,
            message=(
                f"File {file_id} (user {user_id}) has {len(df)} rows, expected 300."
            ),
        )

    # Validate all numeric values are finite in the 6 base columns (R1.10).
    for col in BASE_COLUMNS:
        col_values = pd.to_numeric(df[col], errors="coerce")
        # Check for values that couldn't be parsed as numeric (became NaN after coerce).
        non_numeric_mask = col_values.isna() & df[col].notna()
        if non_numeric_mask.any():
            row_idx = int(non_numeric_mask.idxmax())
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}), column '{col}', "
                    f"row {row_idx}: value cannot be parsed as numeric."
                ),
            )

        # Check for NaN values (original NaN in data).
        nan_mask = np.isnan(col_values.values)
        if nan_mask.any():
            row_idx = int(np.argmax(nan_mask))
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}), column '{col}', "
                    f"row {row_idx}: value is NaN."
                ),
            )

        # Check for Inf values.
        inf_mask = np.isinf(col_values.values)
        if inf_mask.any():
            row_idx = int(np.argmax(inf_mask))
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}), column '{col}', "
                    f"row {row_idx}: value is infinite."
                ),
            )

    # Extract label if expected (R1.5, R1.6, R2.1, R2.2).
    label: int | None = None
    if expect_label:
        label_col = df["label"]

        # Validate single-valued label column (R1.6).
        unique_labels = label_col.unique()
        if len(unique_labels) != 1:
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}) has multiple distinct label "
                    f"values: {unique_labels.tolist()}"
                ),
            )

        label_value = label_col.iloc[0]

        # Validate label is integer and in {0, 1, 2, 3, 4, 5} (R2.1, R2.2).
        try:
            label = int(label_value)
        except (ValueError, TypeError):
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}) has non-integer label: "
                    f"{label_value!r}"
                ),
            )

        if label not in {0, 1, 2, 3, 4, 5}:
            raise DataIntegrityError(
                file_id=file_id,
                message=(
                    f"File {file_id} (user {user_id}) has label {label} outside "
                    f"valid set {{0, 1, 2, 3, 4, 5}}."
                ),
            )

    # Cast 6 base columns to a torch.float32 tensor of shape [300, 6] (R1.3).
    data_array = df[BASE_COLUMNS].values.astype(np.float32)
    data_tensor = torch.from_numpy(data_array)  # [300, 6], float32

    return Window(file_id=file_id, user_id=user_id, data=data_tensor, label=label)
