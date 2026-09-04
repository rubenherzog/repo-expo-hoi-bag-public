"""Deterministic writers and scientific selection rules."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd


class ResultContractError(ValueError):
    """Raised when a canonical result cannot support a published comparison."""


def select_top_within_rung(
    metrics: pd.DataFrame,
    *,
    objective: str,
    top_k: int,
    score_column: str = "full_r2",
) -> pd.DataFrame:
    """Select top models independently in every rung, never from a reference rung."""
    required = {"rung_id", "objective", score_column}
    missing = required.difference(metrics.columns)
    if missing:
        raise ResultContractError(f"Metrics missing required columns: {sorted(missing)}")
    selected = metrics.loc[metrics["objective"].eq(objective)].copy()
    if selected.empty:
        raise ResultContractError(f"No metrics available for objective {objective!r}")
    return (
        selected.sort_values(["rung_id", score_column], ascending=[True, False], kind="stable")
        .groupby("rung_id", sort=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def write_canonical_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write a stable CSV and return its content checksum."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n", float_format="%.12g")
    return sha256(destination.read_bytes()).hexdigest()


def verify_sha256(path: Path, expected: str) -> bool:
    return sha256(Path(path).read_bytes()).hexdigest() == expected
