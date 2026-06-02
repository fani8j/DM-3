"""Predictor with submission writer and ensemble averaging.

Produces a Kaggle-compatible submission CSV whose row order and Id set
match sample_submission.csv exactly. Supports ensemble prediction via
uniform softmax averaging across 1..50 checkpoints.

Requirements: R8.1-R8.13, R16.1-R16.5
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from har.data.integrity import read_sample_submission, check_submission_alignment
from har.data.loader import load_dataset
from har.features.engineer import FeatureEngineer
from har.models import build_model
from har.preprocess.preprocessor import Preprocessor
from har.train.manifest import load_manifest


class PredictionError(Exception):
    """Raised when prediction fails due to validation or runtime errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def predict_to_submission(
    checkpoint_paths: list[Path],
    test_root: Path,
    sample_submission_path: Path,
    output_path: Path,
    config: dict,
    device: str = "cpu",
) -> None:
    """Generate a Kaggle submission CSV from one or more checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoint .pt files (1..50).
        test_root: Path to the test dataset root directory.
        sample_submission_path: Path to sample_submission.csv for canonical ordering.
        output_path: Path where the submission CSV will be written.
        config: Full PipelineConfig as a dict.
        device: Device for inference (default: "cpu").

    Raises:
        PredictionError: On any validation failure (no submission written).
    """
    # --- Validation before any inference (R8.8) ---
    n_checkpoints = len(checkpoint_paths)
    if n_checkpoints == 0 or n_checkpoints > 50:
        raise PredictionError(
            f"Invalid ensemble size: {n_checkpoints}. Must be between 1 and 50."
        )

    # Verify all checkpoint files exist (R8.9)
    for cp_path in checkpoint_paths:
        if not cp_path.exists():
            raise PredictionError(
                f"Checkpoint file does not exist: {cp_path}"
            )

    # Step 1: Load Sample_Submission for canonical Id ordering (R8.4)
    submission_ids = read_sample_submission(sample_submission_path)

    # Step 2: Load test windows (R8.10)
    test_windows, test_summary = load_dataset(test_root, expect_label=False)

    # Step 3: Verify test File_Ids match Sample_Submission Ids (R8.10)
    test_file_ids = set(w.file_id for w in test_windows)
    submission_id_set = set(submission_ids)
    check_submission_alignment(test_file_ids, submission_id_set)

    # Build a lookup from file_id to window for ordered access
    file_id_to_window = {w.file_id: w for w in test_windows}

    # Verify every submission Id has a corresponding test window
    for sid in submission_ids:
        if sid not in file_id_to_window:
            raise PredictionError(
                f"Missing test file for Sample_Submission Id: {sid}"
            )

    # Extract config sections
    preprocess_config = config.get("preprocess", {})
    features_config = config.get("features", {})
    model_config = config.get("model", {})

    # Step 4: Per-checkpoint inference
    # Collect softmax probabilities [N, 6] per checkpoint
    n_samples = len(submission_ids)
    all_probs: list[np.ndarray] = []

    for cp_path in checkpoint_paths:
        # Load manifest and verify config match (R8.11)
        manifest = load_manifest(cp_path.parent / "manifest.json")

        # Verify preprocessor config matches (R8.11)
        manifest_preprocess = manifest.preprocessor_state.get("config", {})
        _verify_config_match(
            manifest_preprocess, preprocess_config, "preprocess"
        )

        # Verify feature config matches (R8.11)
        _verify_config_match(
            manifest.feature_config, features_config, "features"
        )

        # Restore Preprocessor state_dict (no test-time refit) (R8.6)
        preprocessor = Preprocessor(preprocess_config)
        preprocessor.load_state_dict(manifest.preprocessor_state)

        # Build feature engineer
        feature_engineer = FeatureEngineer(features_config)

        # Build model and load weights
        input_channels = feature_engineer.get_output_channels()
        static_dim = feature_engineer.get_static_dim()
        model = build_model(model_config, input_channels, static_dim)
        state_dict = torch.load(cp_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Run forward pass on all test windows in submission order
        probs = _run_inference(
            model=model,
            preprocessor=preprocessor,
            feature_engineer=feature_engineer,
            file_id_to_window=file_id_to_window,
            submission_ids=submission_ids,
            device=device,
        )
        all_probs.append(probs)

    # Step 5: Average probabilities across checkpoints (uniform 1/N) (R8.7, R16.5)
    # Promote to float64 for tie-break stability
    prob_stack = np.stack(all_probs, axis=0).astype(np.float64)  # [K, N, 6]
    avg_probs = prob_stack.mean(axis=0)  # [N, 6]

    # Step 6: Argmax with tie-break = lowest class index (R8.12)
    # np.argmax returns the first occurrence (lowest index) on ties
    predictions = np.argmax(avg_probs, axis=1)  # [N]

    # Step 7: Write submission CSV (R8.1, R8.2, R8.3, R8.4, R8.5)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="ascii") as f:
        # Header: no BOM, no whitespace (R8.2)
        f.write("Id,Label\n")
        # Rows in Sample_Submission order (R8.3, R8.4, R8.5)
        for i, id_ in enumerate(submission_ids):
            label = int(predictions[i])
            f.write(f"{id_},{label}\n")


def _verify_config_match(
    manifest_config: dict, invocation_config: dict, section_name: str
) -> None:
    """Verify that manifest config matches invocation config field-by-field.

    Args:
        manifest_config: Config dict from the checkpoint manifest.
        invocation_config: Config dict from the current invocation.
        section_name: Name of the config section (for error messages).

    Raises:
        PredictionError: If any field differs between manifest and invocation.
    """
    differences: list[str] = []

    for key, manifest_value in manifest_config.items():
        if key in invocation_config:
            invocation_value = invocation_config[key]
            if manifest_value != invocation_value:
                differences.append(
                    f"{section_name}.{key}: manifest={manifest_value!r}, "
                    f"invocation={invocation_value!r}"
                )

    for key in invocation_config:
        if key not in manifest_config:
            differences.append(
                f"{section_name}.{key}: missing in manifest, "
                f"invocation={invocation_config[key]!r}"
            )

    if differences:
        raise PredictionError(
            f"Config mismatch between checkpoint manifest and invocation: "
            f"{'; '.join(differences)}"
        )


def _run_inference(
    model: torch.nn.Module,
    preprocessor: Preprocessor,
    feature_engineer: FeatureEngineer,
    file_id_to_window: dict,
    submission_ids: list[int],
    device: str,
) -> np.ndarray:
    """Run inference on all test windows and return softmax probabilities.

    Args:
        model: Loaded model in eval mode.
        preprocessor: Preprocessor with restored state.
        feature_engineer: Feature engineer instance.
        file_id_to_window: Mapping from file_id to Window.
        submission_ids: Ordered list of submission Ids.
        device: Device for inference.

    Returns:
        numpy array of shape [N, 6] with softmax probabilities.
    """
    from har.data.types import Window

    n_samples = len(submission_ids)
    all_probs = np.zeros((n_samples, 6), dtype=np.float64)

    with torch.no_grad():
        for i, file_id in enumerate(submission_ids):
            window = file_id_to_window[file_id]

            # Apply preprocessor transform (single window)
            transformed_windows = preprocessor.transform([window])
            transformed_data = transformed_windows[0].data

            # Apply feature engineering
            seq_tensor, static_vector = feature_engineer.transform(transformed_data)

            # Prepare batch dimension [1, 300, C]
            seq_batch = seq_tensor.unsqueeze(0).to(device)
            static_batch = None
            if static_vector is not None:
                static_batch = static_vector.unsqueeze(0).to(device)

            # Forward pass
            logits = model(seq_batch, static_batch)  # [1, 6]

            # Softmax probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
            all_probs[i] = probs[0]

    return all_probs
