"""CLI entry point for har-aggregate: build ablation tables from experiment runs.

Usage:
    har-aggregate --experiments-dir experiments --out experiments/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from har.tracking.aggregate import aggregate_ablation_tables


def main() -> None:
    """Entry point for the har-aggregate CLI command."""
    parser = argparse.ArgumentParser(
        prog="har-aggregate",
        description="Build ablation CSV tables from experiment run records.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing runs.jsonl (default: experiments)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for ablation CSV tables (default: same as experiments-dir)",
    )

    args = parser.parse_args()
    experiments_dir: Path = args.experiments_dir
    output_dir: Path = args.out if args.out is not None else experiments_dir

    try:
        aggregate_ablation_tables(experiments_dir, output_dir)
        print(f"Ablation tables written to {output_dir}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
