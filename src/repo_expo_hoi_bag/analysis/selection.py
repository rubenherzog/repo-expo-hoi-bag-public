"""Shared scientific selection rules for candidate and rung comparisons."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from repo_expo_hoi_bag.analysis.contracts import ResultContractError


SortDirection = Literal["max", "min"]


def xgb_depth_for_rung(rung_id: str) -> int:
    """Return the tree depth encoded by an XGBoost ladder rung."""
    depths = {"xgb_tree_d1": 1, "xgb_tree_d2": 2, "xgb_tree_d3": 3}
    try:
        return depths[rung_id]
    except KeyError as exc:
        raise ResultContractError(f"Unsupported XGBoost rung {rung_id!r}") from exc


def require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ResultContractError(f"{label} missing required columns: {sorted(missing)}")


def select_best_per_rung(
    metrics: pd.DataFrame,
    *,
    objective: str,
    score_column: str = "full_r2",
    direction: SortDirection = "max",
) -> pd.DataFrame:
    """Select the best row independently inside every rung."""
    selected = select_top_per_rung(
        metrics,
        objective=objective,
        top_k=1,
        score_column=score_column,
        direction=direction,
    )
    return selected.reset_index(drop=True)


def select_top_per_rung(
    metrics: pd.DataFrame,
    *,
    objective: str,
    top_k: int,
    score_column: str = "full_r2",
    direction: SortDirection = "max",
) -> pd.DataFrame:
    """Select top-k models independently per rung.

    This is the canonical intra-rung rule used for top-20, best exhaustive and
    baseline comparisons. It never carries candidate IDs from one reference rung
    into another rung.
    """
    if top_k < 1:
        raise ResultContractError("top_k must be >= 1")
    require_columns(metrics, {"rung_id", "objective", score_column}, label="Metrics")
    selected = metrics.loc[metrics["objective"].astype(str).eq(objective)].copy()
    if selected.empty:
        raise ResultContractError(f"No metrics available for objective {objective!r}")
    ascending = direction == "min"
    return (
        selected.sort_values(["rung_id", score_column], ascending=[True, ascending], kind="stable")
        .groupby("rung_id", sort=False, group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def select_baseline_per_rung(
    metrics: pd.DataFrame,
    *,
    baseline_label: str,
    label_column: str = "candidate_family",
    score_column: str = "full_r2",
) -> pd.DataFrame:
    """Resolve one baseline row per rung using the same explicit contract."""
    require_columns(metrics, {"rung_id", label_column, score_column}, label="Baseline metrics")
    selected = metrics.loc[metrics[label_column].astype(str).eq(baseline_label)].copy()
    if selected.empty:
        raise ResultContractError(f"No baseline rows found for {baseline_label!r}")
    return (
        selected.sort_values(["rung_id", score_column], ascending=[True, False], kind="stable")
        .groupby("rung_id", sort=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
