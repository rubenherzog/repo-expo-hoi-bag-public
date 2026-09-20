"""Shared paths for publication-facing aggregation stages."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def model_comparison_output_dir() -> Path:
    """Return the established output directory or an isolated smoke directory."""
    override = os.environ.get("PUBLICATION_STATS_OUTDIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return REPOSITORY_ROOT / "outputs" / "dedup" / "model_comparison"


def required_dedup_root() -> Path:
    """Return the deduplicated bundle without personal-path fallbacks."""
    value = os.environ.get("DEDUP_DATA_ROOT", "").strip()
    if not value:
        raise RuntimeError("DEDUP_DATA_ROOT must identify the deduplicated analysis bundle")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DEDUP_DATA_ROOT is not a directory: {path}")
    return path
