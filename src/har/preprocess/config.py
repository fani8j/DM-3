"""Preprocessing configuration re-export.

The canonical PreprocessConfig lives in har.config to keep the full pipeline
schema in one place. This module re-exports it for convenience.
"""

from __future__ import annotations

from har.config import PreprocessConfig

__all__ = ["PreprocessConfig"]
