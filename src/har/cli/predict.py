"""CLI entry point for har-predict: generate submission from checkpoint(s).

Usage:
    har-predict --config configs/default.toml --checkpoints checkpoints/run_id/best.pt --output submissions/submission.csv

Loads one or more trained checkpoints and produces a Kaggle-compatible
submission CSV with ensemble averaging.

Requirements: R8.1, R16.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from har.config import load_config
from har.predict.predictor import predict_to_submission


def main() -> None:
    """Entry point for the har-predict CLI command."""
    parser = argparse.ArgumentParser(
        prog="har-predict",
        description="Generate a Kaggle submission CSV from trained checkpoint(s).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline configuration TOML file.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to checkpoint .pt file(s) for ensemble prediction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/submission.csv"),
        help="Output path for the submission CSV (default: submissions/submission.csv).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)

        predict_to_submission(
            checkpoint_paths=args.checkpoints,
            test_root=Path(config.paths.test_dir),
            sample_submission_path=Path(config.paths.sample_submission),
            output_path=args.output,
            config=config.model_dump(),
            device=config.device,
        )
        print(f"Submission written to: {args.output}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
