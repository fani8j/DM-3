"""CLI entry point for har-validate-data: run data ingestion + integrity checks.

Usage:
    har-validate-data --config configs/default.toml

Loads train and test datasets, validates data integrity, checks user set
partitioning, and verifies submission alignment.

Requirements: R1.1, R1.2, R2.1, R2.4, R2.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from har.config import load_config
from har.data.integrity import (
    check_submission_alignment,
    check_user_sets,
    read_sample_submission,
)
from har.data.loader import load_dataset


def main() -> None:
    """Entry point for the har-validate-data CLI command."""
    parser = argparse.ArgumentParser(
        prog="har-validate-data",
        description="Run data ingestion and integrity checks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline configuration TOML file.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)

        # Load train dataset
        train_windows, train_summary = load_dataset(
            Path(config.paths.train_dir), expect_label=True
        )
        print(f"Train: {train_summary.n_windows} windows, {train_summary.n_users} users")
        print(f"Class distribution: {train_summary.class_distribution}")

        # Load test dataset
        test_windows, test_summary = load_dataset(
            Path(config.paths.test_dir), expect_label=False
        )
        print(f"Test: {test_summary.n_windows} windows, {test_summary.n_users} users")

        # Check user sets (R2.4, R2.5)
        check_user_sets(
            set(train_summary.user_ids), set(test_summary.user_ids)
        )
        print("User set check: PASSED")

        # Check submission alignment (R2.6, R2.7, R2.8)
        submission_ids = read_sample_submission(
            Path(config.paths.sample_submission)
        )
        check_submission_alignment(
            set(test_summary.file_ids), set(submission_ids)
        )
        print("Submission alignment check: PASSED")

        print("\nAll data validation checks passed!")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
