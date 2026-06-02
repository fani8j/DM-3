"""Structured per-run logging for the HAR pipeline.

Provides a configured logger per training run with:
- R12.1: INFO for lifecycle events
- R12.2: WARNING for anomalies
- R12.3: ERROR for aborts
- R12.4: Per-run log file under experiments/{run_id}.log
- R12.5: Stderr fallback on file write failure
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_run_logger(run_id: str, experiments_dir: Path) -> logging.Logger:
    """Set up a per-run logger with file handler and stderr fallback.

    Creates a logger named ``har.run.{run_id}`` with a file handler writing to
    ``experiments_dir/{run_id}.log``. If the file handler cannot be created
    (e.g., permission error, read-only filesystem), falls back to a stderr
    handler so that log records are never silently lost.

    Args:
        run_id: UUID v4 string identifying the current run.
        experiments_dir: Directory where log files are stored.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(f"har.run.{run_id}")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times for the same run
    if logger.handlers:
        return logger

    # File handler
    log_dir = experiments_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    try:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError:
        # R12.5: Fallback to stderr when file write fails
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
