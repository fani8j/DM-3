"""Dataset integrity validation checks.

Verifies user-set partitioning (train vs test) and submission alignment
(test File_Ids vs sample_submission Ids) before training or inference begins.

Requirements: R2.4, R2.5, R2.6, R2.7, R2.8
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataIntegrityError(Exception):
    """Raised when dataset integrity validation fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def check_user_sets(
    train_user_ids: set[int],
    test_user_ids: set[int],
) -> None:
    """Verify train users are {1..60} and test users are {61..100}, disjoint.

    Raises DataIntegrityError if:
    - Train user set is not exactly {1, 2, ..., 60}
    - Test user set is not exactly {61, 62, ..., 100}
    - Any user appears in both sets

    Requirements: R2.4, R2.5
    """
    expected_train = set(range(1, 61))
    expected_test = set(range(61, 101))

    errors: list[str] = []

    # Check train set
    missing_train = expected_train - train_user_ids
    unexpected_train = train_user_ids - expected_train
    if missing_train:
        errors.append(
            f"Train set missing user IDs: {sorted(missing_train)}"
        )
    if unexpected_train:
        errors.append(
            f"Train set has unexpected user IDs: {sorted(unexpected_train)}"
        )

    # Check test set
    missing_test = expected_test - test_user_ids
    unexpected_test = test_user_ids - expected_test
    if missing_test:
        errors.append(
            f"Test set missing user IDs: {sorted(missing_test)}"
        )
    if unexpected_test:
        errors.append(
            f"Test set has unexpected user IDs: {sorted(unexpected_test)}"
        )

    # Check disjoint
    overlap = train_user_ids & test_user_ids
    if overlap:
        errors.append(
            f"Overlapping user IDs in train and test: {sorted(overlap)}"
        )

    if errors:
        raise DataIntegrityError("; ".join(errors))


def check_submission_alignment(
    test_file_ids: set[int],
    submission_ids: set[int],
) -> None:
    """Verify test File_Ids equal Sample_Submission Id set.

    Raises DataIntegrityError with named lists:
    - missing_in_submission: File_Ids in test but not in submission
    - missing_in_test: Ids in submission but not in test

    Requirements: R2.6, R2.7
    """
    missing_in_submission = test_file_ids - submission_ids
    missing_in_test = submission_ids - test_file_ids

    if missing_in_submission or missing_in_test:
        parts: list[str] = []
        parts.append(
            "Test File_Id set does not match Sample_Submission Id set"
        )
        if missing_in_submission:
            parts.append(
                f"missing_in_submission (in test but not submission): "
                f"{sorted(missing_in_submission)}"
            )
        if missing_in_test:
            parts.append(
                f"missing_in_test (in submission but not test): "
                f"{sorted(missing_in_test)}"
            )
        raise DataIntegrityError("; ".join(parts))


def read_sample_submission(path: Path) -> list[int]:
    """Read sample_submission.csv and return the list of Id values in order.

    Validates:
    - File exists and is readable
    - Has an 'Id' column
    - All Id values are integers

    Returns the list of integer Ids in the order they appear in the file.

    Raises DataIntegrityError if any validation fails.

    Requirements: R2.8
    """
    if not path.exists():
        raise DataIntegrityError(
            f"Sample submission file does not exist: {path}"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataIntegrityError(
            f"Sample submission file is unreadable: {path} ({exc})"
        ) from exc

    if "Id" not in df.columns:
        raise DataIntegrityError(
            f"Sample submission file is missing 'Id' column: {path}. "
            f"Found columns: {list(df.columns)}"
        )

    id_series = df["Id"]

    # Check all values are integers (no NaN, no float with fractional part)
    if id_series.isna().any():
        raise DataIntegrityError(
            f"Sample submission 'Id' column contains non-integer (NaN) values: {path}"
        )

    try:
        id_values = id_series.astype(int).tolist()
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(
            f"Sample submission 'Id' column contains non-integer values: {path} ({exc})"
        ) from exc

    # Verify no precision loss from float conversion
    if not (id_series == pd.Series(id_values)).all():
        raise DataIntegrityError(
            f"Sample submission 'Id' column contains non-integer values: {path}"
        )

    return id_values
