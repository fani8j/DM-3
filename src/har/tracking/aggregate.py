"""Ablation table builder for the HAR experiment tracker.

Reads runs.jsonl and produces ablation CSV tables for the report:
- ablation_preprocess.csv: pivots over preprocessing toggles (R14.1)
- ablation_features.csv: pivots over feature-group toggles (R14.2)
- ablation_arch.csv: pivots over model.arch (R14.3)
- ablation_splitter.csv: 2 rows (GroupKFold vs RandomSplit) with Macro F1 gap (R14.4)
- preliminary_analysis.csv: class distribution + per-user counts (R14.5)

Missing/incomplete runs are marked as N/A with a WARNING (R14.6).
All metrics reported to 4 decimal places.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def aggregate_ablation_tables(experiments_dir: Path, output_dir: Path) -> None:
    """Read runs.jsonl and produce ablation CSV tables.

    Produces:
    - ablation_preprocess.csv: pivots over preprocessing toggles (R14.1)
    - ablation_features.csv: pivots over feature-group toggles (R14.2)
    - ablation_arch.csv: pivots over model.arch (R14.3)
    - ablation_splitter.csv: 2 rows (GroupKFold vs RandomSplit) with Macro F1 gap (R14.4)
    - preliminary_analysis.csv: class distribution + per-user counts (R14.5)

    Missing/incomplete runs are marked as N/A with a WARNING (R14.6).
    All metrics reported to 4 decimal places.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = experiments_dir / "runs.jsonl"
    completed_runs = _load_completed_runs(runs_path)

    _write_ablation_preprocess(completed_runs, output_dir)
    _write_ablation_features(completed_runs, output_dir)
    _write_ablation_arch(completed_runs, output_dir)
    _write_ablation_splitter(completed_runs, output_dir)
    _write_preliminary_analysis(completed_runs, runs_path, output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_completed_runs(runs_path: Path) -> list[dict[str, Any]]:
    """Load and join run_start + run_end_success events from runs.jsonl.

    Returns a list of dicts, each containing merged start + end fields for
    successfully completed runs. Incomplete/failed runs are excluded from
    the main list but logged as warnings.
    """
    if not runs_path.exists():
        logger.warning("runs.jsonl not found at %s; producing empty tables.", runs_path)
        return []

    start_events: dict[str, dict[str, Any]] = {}
    end_events: dict[str, dict[str, Any]] = {}

    try:
        with open(runs_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at line %d in runs.jsonl", line_num
                    )
                    continue

                event_type = record.get("type", "")
                run_id = record.get("run_id", "")

                if event_type == "run_start":
                    start_events[run_id] = record
                elif event_type == "run_end_success":
                    end_events[run_id] = record
                elif event_type == "run_end_failure":
                    logger.warning(
                        "Run %s ended with failure/interruption; marking as N/A.",
                        run_id,
                    )
    except OSError as e:
        logger.warning("Could not read runs.jsonl: %s; producing empty tables.", e)
        return []

    # Join start + end events by run_id
    completed: list[dict[str, Any]] = []
    for run_id, start in start_events.items():
        end = end_events.get(run_id)
        if end is None:
            logger.warning(
                "Run %s has no successful completion record; marking as incomplete.",
                run_id,
            )
            # Still include it with N/A metrics for table generation
            completed.append({
                "run_id": run_id,
                "config": start.get("config", {}),
                "seed": start.get("seed"),
                "macro_f1": None,  # N/A
                "per_class_f1": None,
                "per_class_precision": None,
                "per_class_recall": None,
                "confusion_matrix": None,
            })
        else:
            completed.append({
                "run_id": run_id,
                "config": start.get("config", {}),
                "seed": start.get("seed"),
                "macro_f1": end.get("macro_f1"),
                "per_class_f1": end.get("per_class_f1"),
                "per_class_precision": end.get("per_class_precision"),
                "per_class_recall": end.get("per_class_recall"),
                "confusion_matrix": end.get("confusion_matrix"),
            })

    return completed


def _format_metric(value: float | None) -> str:
    """Format a metric value to 4 decimal places, or 'N/A' if missing."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _get_config_field(run: dict[str, Any], *keys: str) -> Any:
    """Safely traverse nested config dict."""
    obj = run.get("config", {})
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------


def _write_ablation_preprocess(
    runs: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write ablation_preprocess.csv: one row per preprocessing toggle state.

    Columns: config_label, toggle_state, macro_f1
    """
    output_path = output_dir / "ablation_preprocess.csv"
    preprocess_toggles = ["nan_inf_to_zero", "standardize", "per_window_demean"]

    rows: list[dict[str, str]] = []

    for toggle in preprocess_toggles:
        # Find runs where this toggle is enabled
        enabled_runs = [
            r for r in runs
            if _get_config_field(r, "preprocess", toggle) is True
        ]
        disabled_runs = [
            r for r in runs
            if _get_config_field(r, "preprocess", toggle) is False
        ]

        # Pick the best (or first) run for each state
        enabled_f1 = _best_macro_f1(enabled_runs)
        disabled_f1 = _best_macro_f1(disabled_runs)

        rows.append({
            "config_label": toggle,
            "toggle_state": "enabled",
            "macro_f1": _format_metric(enabled_f1),
        })
        rows.append({
            "config_label": toggle,
            "toggle_state": "disabled",
            "macro_f1": _format_metric(disabled_f1),
        })

    _write_csv(output_path, ["config_label", "toggle_state", "macro_f1"], rows)


def _write_ablation_features(
    runs: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write ablation_features.csv: one row per feature-group toggle state.

    Columns: config_label, toggle_state, macro_f1
    """
    output_path = output_dir / "ablation_features.csv"
    feature_toggles = [
        "magnitude",
        "std_magnitude",
        "first_difference",
        "rolling_stats",
        "frequency_domain",
    ]

    rows: list[dict[str, str]] = []

    for toggle in feature_toggles:
        enabled_runs = [
            r for r in runs
            if _get_config_field(r, "features", toggle) is True
        ]
        disabled_runs = [
            r for r in runs
            if _get_config_field(r, "features", toggle) is False
        ]

        enabled_f1 = _best_macro_f1(enabled_runs)
        disabled_f1 = _best_macro_f1(disabled_runs)

        rows.append({
            "config_label": toggle,
            "toggle_state": "enabled",
            "macro_f1": _format_metric(enabled_f1),
        })
        rows.append({
            "config_label": toggle,
            "toggle_state": "disabled",
            "macro_f1": _format_metric(disabled_f1),
        })

    _write_csv(output_path, ["config_label", "toggle_state", "macro_f1"], rows)


def _write_ablation_arch(
    runs: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write ablation_arch.csv: one row per architecture.

    Columns: architecture, macro_f1
    """
    output_path = output_dir / "ablation_arch.csv"
    architectures = ["tcn", "bigru", "transformer"]

    rows: list[dict[str, str]] = []

    for arch in architectures:
        arch_runs = [
            r for r in runs
            if _get_config_field(r, "model", "arch") == arch
        ]
        arch_f1 = _best_macro_f1(arch_runs)

        if not arch_runs:
            logger.warning(
                "No runs found for architecture '%s'; marking as N/A.", arch
            )

        rows.append({
            "architecture": arch,
            "macro_f1": _format_metric(arch_f1),
        })

    _write_csv(output_path, ["architecture", "macro_f1"], rows)


def _write_ablation_splitter(
    runs: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write ablation_splitter.csv: exactly 2 rows with Macro F1 gap.

    Columns: splitter, macro_f1, gap
    """
    output_path = output_dir / "ablation_splitter.csv"

    # Find runs using group_kfold vs random_split
    group_kfold_runs = [
        r for r in runs
        if _has_strategy(r, "group_kfold")
    ]
    random_split_runs = [
        r for r in runs
        if _has_strategy(r, "random_split")
    ]

    gk_f1 = _best_macro_f1(group_kfold_runs)
    rs_f1 = _best_macro_f1(random_split_runs)

    # Compute absolute gap
    if gk_f1 is not None and rs_f1 is not None:
        gap = abs(gk_f1 - rs_f1)
        gap_str = _format_metric(gap)
    else:
        gap_str = "N/A"
        if gk_f1 is None:
            logger.warning(
                "No completed runs found for group_kfold strategy; gap is N/A."
            )
        if rs_f1 is None:
            logger.warning(
                "No completed runs found for random_split strategy; gap is N/A."
            )

    rows = [
        {
            "splitter": "group_kfold",
            "macro_f1": _format_metric(gk_f1),
            "gap": gap_str,
        },
        {
            "splitter": "random_split",
            "macro_f1": _format_metric(rs_f1),
            "gap": gap_str,
        },
    ]

    _write_csv(output_path, ["splitter", "macro_f1", "gap"], rows)


def _write_preliminary_analysis(
    runs: list[dict[str, Any]], runs_path: Path, output_dir: Path
) -> None:
    """Write preliminary_analysis.csv: class distribution + per-user counts.

    Columns for class distribution section: label, count, percentage
    Columns for per-user section: user_id, count

    Data is extracted from the first run's config or from ingestion metadata
    stored in the run_start event.
    """
    output_path = output_dir / "preliminary_analysis.csv"

    # Try to extract class distribution and per-user counts from runs.jsonl
    class_distribution: dict[str, int] | None = None
    per_user_counts: dict[str, int] | None = None

    if runs_path.exists():
        try:
            with open(runs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for class_distribution and per_user_counts in run_start events
                    if record.get("type") == "run_start":
                        if "class_distribution" in record:
                            class_distribution = record["class_distribution"]
                        if "per_user_counts" in record:
                            per_user_counts = record["per_user_counts"]
                        # Also check nested in config or dataset_summary
                        if class_distribution is None:
                            ds = record.get("dataset_summary", {})
                            if "class_distribution" in ds:
                                class_distribution = ds["class_distribution"]
                            if "per_user_counts" in ds:
                                per_user_counts = ds["per_user_counts"]
                        # Use the first run that has this data
                        if class_distribution is not None:
                            break
        except OSError:
            pass

    # Write the CSV with both sections
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Section 1: Class distribution
        writer.writerow(["label", "count", "percentage"])
        if class_distribution:
            total = sum(class_distribution.values())
            for label in sorted(class_distribution.keys(), key=lambda x: int(x)):
                count = class_distribution[label]
                percentage = (count / total * 100) if total > 0 else 0.0
                writer.writerow([label, count, f"{percentage:.4f}"])
        else:
            # Write 6 empty rows for labels 0-5 with N/A
            for label in range(6):
                writer.writerow([label, "N/A", "N/A"])
            logger.warning(
                "No class distribution data found in runs.jsonl; "
                "preliminary_analysis.csv contains N/A values."
            )

        # Separator row
        writer.writerow([])

        # Section 2: Per-user counts
        writer.writerow(["user_id", "count"])
        if per_user_counts:
            for user_id in sorted(per_user_counts.keys(), key=lambda x: int(x)):
                writer.writerow([user_id, per_user_counts[user_id]])
        else:
            logger.warning(
                "No per-user count data found in runs.jsonl; "
                "preliminary_analysis.csv per-user section is empty."
            )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _has_strategy(run: dict[str, Any], strategy: str) -> bool:
    """Check if a run used a specific validation strategy."""
    strategies = _get_config_field(run, "validation", "strategies")
    if isinstance(strategies, list):
        return strategy in strategies
    # Also check if the run has a 'strategy' field directly (single-strategy runs)
    run_strategy = _get_config_field(run, "validation", "strategy")
    if run_strategy == strategy:
        return True
    # Check top-level strategy field (some tracker implementations store it here)
    return run.get("strategy") == strategy


def _best_macro_f1(runs: list[dict[str, Any]]) -> float | None:
    """Return the best (highest) macro_f1 from a list of runs, or None if all are N/A."""
    valid_f1s = [
        r["macro_f1"] for r in runs
        if r.get("macro_f1") is not None
    ]
    if not valid_f1s:
        return None
    return max(valid_f1s)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Write a list of row dicts to a CSV file with the given fieldnames."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
