"""Download the Assignment 3 Kaggle dataset with kagglehub.

By default this script also creates local symlinks named train/, test/, and
sample_submission.csv so the existing config paths continue to work.
"""

from __future__ import annotations

import argparse
from pathlib import Path


COMPETITION = "nycu-data-mining-assignment-3"


def _link_or_report(source: Path, target: Path) -> None:
    if not source.exists():
        print(f"skip: {source} was not found in downloaded data")
        return
    if target.exists() or target.is_symlink():
        print(f"exists: {target}")
        return
    target.symlink_to(source, target_is_directory=source.is_dir())
    print(f"linked: {target} -> {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle competition data.")
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Only download and print the kagglehub path; do not create local symlinks.",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run:\n"
            "  uv run --with kagglehub python scripts/download_data.py"
        ) from exc

    path = Path(kagglehub.competition_download(COMPETITION)).resolve()
    print("Path to competition files:", path)

    if args.no_link:
        return

    _link_or_report(path / "train", Path("train"))
    _link_or_report(path / "test", Path("test"))
    _link_or_report(path / "sample_submission.csv", Path("sample_submission.csv"))


if __name__ == "__main__":
    main()
