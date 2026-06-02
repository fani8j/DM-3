"""Append-only JSONL experiment tracker.

Stores run records in experiments/runs.jsonl with event types:
- run_start: UUID v4, ISO 8601 UTC timestamp, config, seed, code_version
- epoch: run_id, epoch number, train_loss, val_macro_f1
- run_end_success: run_id, end timestamp, all metrics (4 decimal places)
- run_end_failure: run_id, end timestamp, status, error description

Requirements: R9.1, R9.2, R9.3, R9.4, R9.7
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from har.data.types import FoldMetrics


class ExperimentTracker:
    """Append-only JSONL experiment tracker.

    Each event is appended as a single JSON line to ``runs.jsonl`` under the
    configured experiments directory. The file is never overwritten or
    truncated, ensuring that failed/interrupted runs preserve their start
    records (R9.4) and identical-config runs each receive a distinct UUID (R9.7).
    """

    def __init__(self, experiments_dir: Path | str = "experiments") -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.experiments_dir / "runs.jsonl"

    def start_run(self, config: dict, seed: int, code_version: str) -> str:
        """Record run start. Returns the generated UUID v4 run_id.

        R9.1: Unique run identifier (UUID v4), ISO 8601 UTC start timestamp,
        full configuration object, seed, and code version.
        R9.7: Fresh uuid4() regardless of config equivalence with prior runs.
        """
        run_id = str(uuid.uuid4())
        record = {
            "type": "run_start",
            "run_id": run_id,
            "start_ts": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "seed": seed,
            "code_version": code_version,
        }
        self._append(record)
        return run_id

    def log_epoch(
        self, run_id: str, epoch: int, train_loss: float, val_macro_f1: float
    ) -> None:
        """Record per-epoch metrics (R6.3)."""
        record = {
            "type": "epoch",
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
        }
        self._append(record)

    def end_run_success(self, run_id: str, metrics: FoldMetrics) -> None:
        """Record successful run completion with metrics rounded to 4 decimals.

        R9.2: Completion timestamp (ISO 8601 UTC), final Macro_F1, per-class F1,
        per-class precision, per-class recall, and confusion matrix — all metric
        values rounded to 4 decimal places.
        """
        record = {
            "type": "run_end_success",
            "run_id": run_id,
            "end_ts": datetime.now(timezone.utc).isoformat(),
            "macro_f1": round(metrics.macro_f1, 4),
            "per_class_f1": [round(x, 4) for x in metrics.per_class_f1],
            "per_class_precision": [round(x, 4) for x in metrics.per_class_precision],
            "per_class_recall": [round(x, 4) for x in metrics.per_class_recall],
            "confusion_matrix": metrics.confusion_matrix,
        }
        self._append(record)

    def end_run_failure(
        self, run_id: str, error: str, status: str = "failed"
    ) -> None:
        """Record failed/interrupted run. Preserves original start record (R9.4).

        The append-only nature of the JSONL file guarantees the run_start event
        is never overwritten.
        """
        record = {
            "type": "run_end_failure",
            "run_id": run_id,
            "end_ts": datetime.now(timezone.utc).isoformat(),
            "status": status,  # "failed" or "interrupted"
            "error": error,
        }
        self._append(record)

    def _append(self, record: dict) -> None:
        """Append a JSON record as a single line to runs.jsonl."""
        with open(self.runs_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all_records(self) -> list[dict]:
        """Read all records from runs.jsonl.

        Returns an empty list if the file does not yet exist.
        """
        if not self.runs_file.exists():
            return []
        records = []
        with open(self.runs_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
