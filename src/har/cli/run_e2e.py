"""CLI entry point for har-run: end-to-end train + predict.

Usage:
    har-run --config configs/default.toml
    har-run --config configs/default.toml --output submissions/final.csv --override train.epochs=30

Sequences validate-data → train → predict in a single command.
Exits with code 0 on success (R10.1), non-zero on any failure with no
partial submission written (R10.2).

Requirements: R10.1, R10.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the har-run CLI command."""
    parser = argparse.ArgumentParser(
        prog="har-run",
        description="End-to-end train + predict pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline configuration TOML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/final.csv"),
        help="Output path for the submission CSV (default: submissions/final.csv).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values as KEY=VALUE pairs (dot notation).",
    )
    args = parser.parse_args()

    # Parse overrides into a dict
    overrides: dict[str, str] = {}
    for ov in args.override:
        if "=" in ov:
            key, value = ov.split("=", 1)
            overrides[key] = value

    try:
        from har.config import load_config
        from har.data.integrity import (
            check_submission_alignment,
            check_user_sets,
            read_sample_submission,
        )
        from har.data.loader import load_dataset
        from har.features.engineer import FeatureEngineer
        from har.models import build_model
        from har.predict.predictor import predict_to_submission
        from har.preprocess.preprocessor import Preprocessor
        from har.tracking.experiment import ExperimentTracker
        from har.train.manifest import get_code_version
        from har.train.splitter import RandomSplitSplitter
        from har.train.trainer import Trainer

        config = load_config(args.config, overrides=overrides if overrides else None)

        # Step 1: Validate data
        print("Step 1: Validating data...")
        train_windows, train_summary = load_dataset(
            Path(config.paths.train_dir), expect_label=True
        )
        test_windows, test_summary = load_dataset(
            Path(config.paths.test_dir), expect_label=False
        )
        check_user_sets(
            set(train_summary.user_ids), set(test_summary.user_ids)
        )
        submission_ids = read_sample_submission(
            Path(config.paths.sample_submission)
        )
        check_submission_alignment(
            set(test_summary.file_ids), set(submission_ids)
        )
        print(
            f"  Train: {train_summary.n_windows} windows, "
            f"Test: {test_summary.n_windows} windows"
        )

        # Step 2: Train (using random_split for speed in e2e mode)
        print("\nStep 2: Training...")
        tracker = ExperimentTracker(config.paths.experiments_dir)
        run_id = tracker.start_run(
            config=config.model_dump(),
            seed=config.seed,
            code_version=get_code_version(),
        )

        preprocess_config = config.preprocess.model_dump()
        features_config = config.features.model_dump()
        feature_engineer = FeatureEngineer(features_config)
        input_channels = feature_engineer.get_output_channels()
        static_dim = feature_engineer.get_static_dim()

        # Use random split for e2e
        splitter = RandomSplitSplitter(
            train_fraction=config.validation.random_split.train_fraction,
            seed=config.validation.random_split.seed,
        )
        fold_split = splitter.split(train_windows)

        train_fold = [train_windows[i] for i in fold_split.train_indices]
        val_fold = [train_windows[i] for i in fold_split.val_indices]

        # Fit preprocessor on train fold only
        preprocessor = Preprocessor(preprocess_config)
        preprocessor.fit(train_fold)

        # Transform
        train_transformed = preprocessor.transform(train_fold)
        val_transformed = preprocessor.transform(val_fold)

        # Feature engineering
        train_samples = []
        for w in train_transformed:
            seq, static = feature_engineer.transform(w.data)
            train_samples.append((seq, static, w.label))

        val_samples = []
        for w in val_transformed:
            seq, static = feature_engineer.transform(w.data)
            val_samples.append((seq, static, w.label))

        # Build model and train
        model = build_model(config.model.model_dump(), input_channels, static_dim)
        trainer = Trainer(model, config, config.device, experiment_tracker=tracker)
        best_f1 = trainer.train_fold(
            train_samples,
            val_samples,
            fold_split,
            run_id,
            preprocessor_state=preprocessor.state_dict(),
            feature_config=features_config,
        )
        print(f"  Best Macro F1: {best_f1:.4f}")

        # Step 3: Predict
        print("\nStep 3: Generating submission...")
        checkpoint_path = (
            Path(config.paths.checkpoints_dir) / run_id / "best.pt"
        )
        predict_to_submission(
            checkpoint_paths=[checkpoint_path],
            test_root=Path(config.paths.test_dir),
            sample_submission_path=Path(config.paths.sample_submission),
            output_path=args.output,
            config=config.model_dump(),
            device=config.device,
        )
        print(f"  Submission written to: {args.output}")
        print(f"\nDone! Run ID: {run_id}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
