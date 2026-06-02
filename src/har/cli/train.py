"""CLI entry point for har-train: train model with configured validation strategies.

Usage:
    har-train --config configs/default.toml
    har-train --config configs/default.toml --override model.arch=bigru train.lr=0.01

Drives Splitter → Preprocessor.fit → Trainer per fold for each configured
validation strategy. Supports --override KEY=VALUE for ablation runs.

Requirements: R6.1, R7.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from har.config import load_config
from har.data.loader import load_dataset
from har.features.engineer import FeatureEngineer
from har.models import build_model
from har.preprocess.preprocessor import Preprocessor
from har.tracking.experiment import ExperimentTracker
from har.tracking.logger import setup_run_logger
from har.train.manifest import get_code_version
from har.train.splitter import GroupKFoldSplitter, RandomSplitSplitter
from har.train.trainer import Trainer


def main() -> None:
    """Entry point for the har-train CLI command."""
    parser = argparse.ArgumentParser(
        prog="har-train",
        description="Train model with configured validation strategies.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline configuration TOML file.",
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
        config = load_config(args.config, overrides=overrides if overrides else None)

        # Load training data
        train_windows, train_summary = load_dataset(
            Path(config.paths.train_dir), expect_label=True
        )
        print(f"Loaded {train_summary.n_windows} training windows from {train_summary.n_users} users")

        # Setup tracking
        tracker = ExperimentTracker(config.paths.experiments_dir)

        # Setup feature engineer to determine model dimensions
        features_config = config.features.model_dump()
        preprocess_config = config.preprocess.model_dump()
        feature_engineer = FeatureEngineer(features_config)
        input_channels = feature_engineer.get_output_channels()
        static_dim = feature_engineer.get_static_dim()

        # Run each configured validation strategy (R7.5)
        strategies = config.validation.strategies

        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Training with strategy: {strategy}")
            print(f"{'='*60}")

            # Start run
            run_id = tracker.start_run(
                config=config.model_dump(),
                seed=config.seed,
                code_version=get_code_version(),
            )
            run_logger = setup_run_logger(run_id, Path(config.paths.experiments_dir))
            run_logger.info("Starting training run with strategy: %s", strategy)

            try:
                if strategy == "group_kfold":
                    _run_group_kfold(
                        config=config,
                        train_windows=train_windows,
                        preprocess_config=preprocess_config,
                        features_config=features_config,
                        feature_engineer=feature_engineer,
                        input_channels=input_channels,
                        static_dim=static_dim,
                        tracker=tracker,
                        run_id=run_id,
                        run_logger=run_logger,
                    )
                elif strategy == "random_split":
                    _run_random_split(
                        config=config,
                        train_windows=train_windows,
                        preprocess_config=preprocess_config,
                        features_config=features_config,
                        feature_engineer=feature_engineer,
                        input_channels=input_channels,
                        static_dim=static_dim,
                        tracker=tracker,
                        run_id=run_id,
                        run_logger=run_logger,
                    )
                else:
                    run_logger.warning("Unknown strategy: %s, skipping", strategy)
                    print(f"  WARNING: Unknown strategy '{strategy}', skipping.")
                    continue

                run_logger.info("Training completed successfully")
                print(f"\nRun ID: {run_id}")

            except Exception as e:
                tracker.end_run_failure(run_id, str(e))
                run_logger.error("Training failed: %s", e)
                raise

        print("\nTraining complete!")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _run_group_kfold(
    config,
    train_windows,
    preprocess_config,
    features_config,
    feature_engineer,
    input_channels,
    static_dim,
    tracker,
    run_id,
    run_logger,
) -> None:
    """Run training with GroupKFold validation strategy."""
    splitter = GroupKFoldSplitter(config.validation.group_kfold.n_splits)

    for fold_split in splitter.split(train_windows):
        print(f"\n  Fold {fold_split.fold_index + 1}/{splitter.n_splits}")

        # Prepare data for this fold
        train_fold_windows = [train_windows[i] for i in fold_split.train_indices]
        val_fold_windows = [train_windows[i] for i in fold_split.val_indices]

        # Fit preprocessor on train fold only (R3.3)
        preprocessor = Preprocessor(preprocess_config)
        preprocessor.fit(train_fold_windows)

        # Transform
        train_transformed = preprocessor.transform(train_fold_windows)
        val_transformed = preprocessor.transform(val_fold_windows)

        # Feature engineering
        train_samples = _extract_samples(train_transformed, feature_engineer)
        val_samples = _extract_samples(val_transformed, feature_engineer)

        # Build model
        model = build_model(config.model.model_dump(), input_channels, static_dim)

        # Train
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
        run_logger.info(
            "Fold %d complete — best_macro_f1=%.4f",
            fold_split.fold_index + 1,
            best_f1,
        )


def _run_random_split(
    config,
    train_windows,
    preprocess_config,
    features_config,
    feature_engineer,
    input_channels,
    static_dim,
    tracker,
    run_id,
    run_logger,
) -> None:
    """Run training with RandomSplit validation strategy."""
    splitter = RandomSplitSplitter(
        train_fraction=config.validation.random_split.train_fraction,
        seed=config.validation.random_split.seed,
    )
    fold_split = splitter.split(train_windows)

    train_fold_windows = [train_windows[i] for i in fold_split.train_indices]
    val_fold_windows = [train_windows[i] for i in fold_split.val_indices]

    # Fit preprocessor on train fold only (R3.3)
    preprocessor = Preprocessor(preprocess_config)
    preprocessor.fit(train_fold_windows)

    # Transform
    train_transformed = preprocessor.transform(train_fold_windows)
    val_transformed = preprocessor.transform(val_fold_windows)

    # Feature engineering
    train_samples = _extract_samples(train_transformed, feature_engineer)
    val_samples = _extract_samples(val_transformed, feature_engineer)

    # Build model
    model = build_model(config.model.model_dump(), input_channels, static_dim)

    # Train
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
    run_logger.info("Random split complete — best_macro_f1=%.4f", best_f1)


def _extract_samples(
    windows, feature_engineer
) -> list[tuple]:
    """Transform windows into (seq_tensor, static_vector, label) tuples."""
    samples = []
    for w in windows:
        seq, static = feature_engineer.transform(w.data)
        samples.append((seq, static, w.label))
    return samples
