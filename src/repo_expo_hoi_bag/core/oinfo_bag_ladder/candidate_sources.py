from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from loco_fusion_matrix_engine import parse_literal_list
from oinfo_bag_ladder.config import DEFAULT_ANALYSIS_FAMILY, REFERENCE_SNAPSHOT_CANONICAL_ROOT
from oinfo_bag_ladder.io_utils import read_parquet


def _predictors_identity_from_vars(nplet_vars: object, exposome_cols: list[str]) -> str:
    expo_idx = {c: i for i, c in enumerate(exposome_cols)}
    raw_vars = parse_literal_list(nplet_vars)
    pred_names = [str(x) for x in raw_vars if str(x) in expo_idx]
    pred_idx = sorted(set(int(expo_idx[x]) for x in pred_names))
    return "|".join(exposome_cols[i] for i in pred_idx)


def build_greedy_o_only_candidates(candidate_df: pd.DataFrame, exposome_cols: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = candidate_df.copy().reset_index(drop=True)
    out["feature_id"] = out["feature_id"].astype(str)
    out["objective"] = out["objective"].astype(str).str.lower()
    out = out[out["objective"].isin({"o_max", "o_min"})].copy().reset_index(drop=True)
    out["predictors_identity"] = out["nplet_vars"].apply(lambda x: _predictors_identity_from_vars(x, exposome_cols))
    out["predictors_identity_n"] = out["predictors_identity"].astype(str).apply(
        lambda s: len([part for part in s.split("|") if part.strip()])
    )
    out["candidate_id"] = out["feature_id"].astype(str)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce")
    out["raw_thoi_o"] = pd.to_numeric(out.get("thoi_o"), errors="coerce")
    out["raw_thoi_s"] = pd.to_numeric(out.get("thoi_s"), errors="coerce")
    out["thoi_o"] = pd.to_numeric(out.get("thoi_o", out.get("score")), errors="coerce")
    out["score_mode"] = out.get("score_mode", pd.Series(["oinfo"] * len(out))).astype(str).str.lower()
    out["rank"] = pd.to_numeric(out.get("rank"), errors="coerce")
    out["order"] = pd.to_numeric(out.get("order"), errors="coerce").astype("Int64")
    out["nplet_vars_str"] = out["nplet_vars"].apply(lambda x: "|".join(str(v) for v in parse_literal_list(x)))

    candidate_registry = out[
        [
            "candidate_id",
            "feature_id",
            "predictors_identity",
            "predictors_identity_n",
            "nplet_vars_str",
            "objective",
            "order",
            "score",
            "rank",
            "thoi_o",
            "raw_thoi_o",
            "raw_thoi_s",
            "score_mode",
        ]
    ].drop_duplicates(subset=["candidate_id"], keep="first").reset_index(drop=True)
    candidate_registry["candidate_source"] = "greedy_o_only"
    candidate_registry["source_experiment_id"] = pd.NA
    candidate_registry["source_analysis_family"] = pd.NA
    candidate_registry["source_test_diagnosis_group"] = pd.NA
    return out.reset_index(drop=True), candidate_registry


def _load_top_tail(canonical_root: Path) -> pd.DataFrame:
    path = Path(canonical_root) / "top_tail_summary_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing top-tail summary: {path}")
    return read_parquet(path)


def _select_top_tail_source(
    candidate_pool_df: pd.DataFrame,
    candidate_registry_all: pd.DataFrame,
    *,
    canonical_root: Path,
    source_family: str,
    bag_target: str,
    selection_diagnosis_group: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_tail = _load_top_tail(canonical_root)
    top_tail["analysis_family"] = top_tail.get("analysis_family", pd.Series(dtype=object)).astype(str)
    top_tail["bag_target"] = top_tail["bag_target"].astype(str)
    top_tail["candidate_id"] = top_tail["candidate_id"].astype(str)
    top_tail["objective"] = top_tail["objective"].astype(str).str.lower()

    sel = top_tail[top_tail["bag_target"] == str(bag_target)].copy()
    if source_family:
        sel = sel[sel["analysis_family"].astype(str) == str(source_family)].copy()
    if selection_diagnosis_group is not None and "test_diagnosis_group" in sel.columns:
        sel = sel[sel["test_diagnosis_group"].astype(str) == str(selection_diagnosis_group)].copy()
    if sel.empty:
        raise ValueError(
            f"No top-tail candidates found for bag={bag_target}, source_family={source_family}, "
            f"selection_diagnosis_group={selection_diagnosis_group} in {canonical_root}"
        )

    key_cols = [
        "candidate_id",
        "rung_id",
        "objective",
        "experiment_id",
        "analysis_family",
        "test_diagnosis_group",
        "tail_rank",
        "tail_n_selected",
        "tail_n_total",
        "tail_fraction",
        "tail_threshold_full_r2",
        "full_r2",
        "delta_r2_vs_base",
        "order",
        "thoi_o",
        "predictors_identity",
    ]
    key_cols = [c for c in key_cols if c in sel.columns]
    sel = sel[key_cols].drop_duplicates().reset_index(drop=True)
    sel = sel.rename(columns={"rung_id": "source_rung_id"})

    keep_ids = set(sel["candidate_id"].astype(str).tolist())
    selected = candidate_pool_df[candidate_pool_df["candidate_id"].astype(str).isin(keep_ids)].copy()
    if selected.empty:
        raise ValueError(f"Top-tail candidate ids could not be mapped back to the greedy pool for bag={bag_target}.")

    sel_meta = sel.rename(
        columns={
            "experiment_id": "source_experiment_id",
            "analysis_family": "source_analysis_family",
            "test_diagnosis_group": "source_test_diagnosis_group",
            "full_r2": "source_full_r2",
            "delta_r2_vs_base": "source_delta_r2_vs_base",
            "tail_rank": "source_tail_rank",
            "tail_n_selected": "source_tail_n_selected",
            "tail_n_total": "source_tail_n_total",
            "tail_fraction": "source_tail_fraction",
            "tail_threshold_full_r2": "source_tail_threshold_full_r2",
        }
    )
    sel_meta = _ensure_selection_columns(
        sel_meta,
        [
            "candidate_id",
            "source_rung_id",
            "objective",
            "predictors_identity",
            "order",
            "thoi_o",
            "source_experiment_id",
            "source_analysis_family",
            "source_test_diagnosis_group",
            "source_full_r2",
            "source_delta_r2_vs_base",
            "source_tail_rank",
            "source_tail_n_selected",
            "source_tail_n_total",
            "source_tail_fraction",
            "source_tail_threshold_full_r2",
        ],
    )

    registry = candidate_registry_all[candidate_registry_all["candidate_id"].astype(str).isin(keep_ids)].copy()
    meta_agg = (
        sel_meta.groupby("candidate_id", dropna=False)
        .agg(
            source_experiment_id=("source_experiment_id", "first"),
            source_analysis_family=("source_analysis_family", "first"),
            source_test_diagnosis_group=("source_test_diagnosis_group", "first"),
            source_rung_ids=("source_rung_id", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )
    registry = registry.merge(meta_agg, on="candidate_id", how="left")
    registry["candidate_source"] = (
        "cn_top_tail_per_rung" if str(source_family) == "single_dx_oinfo_ladder" else "pooled_top_tail_per_rung"
    )
    selection_df = sel_meta[
        [
            "candidate_id",
            "source_rung_id",
            "objective",
            "predictors_identity",
            "order",
            "thoi_o",
            "source_experiment_id",
            "source_analysis_family",
            "source_test_diagnosis_group",
            "source_full_r2",
            "source_delta_r2_vs_base",
            "source_tail_rank",
            "source_tail_n_selected",
            "source_tail_n_total",
            "source_tail_fraction",
            "source_tail_threshold_full_r2",
        ]
    ].drop_duplicates().reset_index(drop=True)
    return selected.reset_index(drop=True), registry.reset_index(drop=True), selection_df.reset_index(drop=True)


def _ensure_selection_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def build_pooled_top_tail_candidates(
    candidate_pool_df: pd.DataFrame,
    candidate_registry_all: pd.DataFrame,
    bag_target: str,
    canonical_root: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = REFERENCE_SNAPSHOT_CANONICAL_ROOT if canonical_root is None else Path(canonical_root)
    return _select_top_tail_source(
        candidate_pool_df,
        candidate_registry_all,
        canonical_root=root,
        source_family=DEFAULT_ANALYSIS_FAMILY,
        bag_target=bag_target,
        selection_diagnosis_group=None,
    )


def build_cn_top_tail_candidates(
    candidate_pool_df: pd.DataFrame,
    candidate_registry_all: pd.DataFrame,
    bag_target: str,
    canonical_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _select_top_tail_source(
        candidate_pool_df,
        candidate_registry_all,
        canonical_root=Path(canonical_root),
        source_family="single_dx_oinfo_ladder",
        bag_target=bag_target,
        selection_diagnosis_group="CN",
    )
