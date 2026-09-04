from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from oinfo_bag_ladder.config import FRONTIER_MIN_BIN_SIZE, FRONTIER_N_BINS, TOP_TAIL_FRAC, TOP_TAIL_MIN_N
from oinfo_bag_ladder.rungs import RUNG_LABELS, RUNG_ORDER

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    _spearmanr = None


META_COLUMNS = [
    "experiment_id",
    "analysis_family",
    "candidate_source",
    "population_mode",
    "objective_scope",
    "cv_scheme",
    "bag_target",
    "bag_target_column",
    "train_diagnosis_group",
    "test_diagnosis_group",
    "selection_family",
    "selection_diagnosis_group",
    "evaluation_mode",
]


GLOBAL_COLUMNS = [
    *META_COLUMNS,
    "rung_id",
    "rung_label",
    "candidate_id",
    "feature_id",
    "objective",
    "metric",
    "direction",
    "order",
    "score",
    "rank",
    "global_rank",
    "representative_id",
    "predictors_identity",
    "predictors_identity_n",
    "thoi_o",
    "full_r2",
    "full_rmse",
    "full_mae",
    "full_corr2",
    "base_r2",
    "base_rmse",
    "base_mae",
    "base_corr2",
    "delta_r2_vs_base",
    "delta_rmse_vs_base",
    "delta_mae_vs_base",
    "delta_corr2_vs_base",
    "calib_slope",
    "calib_intercept",
    "bias_mean",
    "calib_slope_base",
    "calib_intercept_base",
    "bias_mean_base",
    "n_total_with_y",
    "n_scored",
    "n_scored_base",
    "coverage_pct",
    "coverage_pct_base",
]

COUNTRY_COLUMNS = [
    *META_COLUMNS,
    "rung_id",
    "rung_label",
    "candidate_id",
    "feature_id",
    "objective",
    "metric",
    "direction",
    "order",
    "score",
    "rank",
    "predictors_identity",
    "predictors_identity_n",
    "thoi_o",
    "fold_country",
    "country_full_r2",
    "country_full_rmse",
    "country_full_mae",
    "country_full_corr2",
    "country_base_r2",
    "country_base_rmse",
    "country_base_mae",
    "country_base_corr2",
    "country_delta_r2_vs_base",
    "country_delta_rmse_vs_base",
    "country_delta_mae_vs_base",
    "country_delta_corr2_vs_base",
    "calib_slope",
    "calib_intercept",
    "bias_mean",
    "calib_slope_base",
    "calib_intercept_base",
    "bias_mean_base",
    "n_test_total",
    "n_test_scored",
    "n_test_total_base",
    "n_test_scored_base",
    "coverage_pct",
    "coverage_pct_base",
]

CONTRAST_DEFS = [
    ("d1_minus_ols", "xgb_tree_d1", "ols"),
    ("d2_minus_d1", "xgb_tree_d2", "xgb_tree_d1"),
    ("d3_minus_d2", "xgb_tree_d3", "xgb_tree_d2"),
    ("d3_minus_d1", "xgb_tree_d3", "xgb_tree_d1"),
    ("d3_minus_ols", "xgb_tree_d3", "ols"),
]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_str(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"nan": ""})


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "score",
        "rank",
        "order",
        "global_rank",
        "predictors_identity_n",
        "global_oof_r2",
        "global_oof_rmse",
        "global_oof_mae",
        "global_oof_corr2",
        "global_oof_r2_base",
        "global_oof_rmse_base",
        "global_oof_mae_base",
        "global_oof_corr2_base",
        "delta_r2_vs_base",
        "delta_rmse_vs_base",
        "delta_mae_vs_base",
        "delta_corr2_vs_base",
        "calib_slope",
        "calib_intercept",
        "bias_mean",
        "calib_slope_base",
        "calib_intercept_base",
        "bias_mean_base",
        "n_total_with_y",
        "n_scored",
        "n_scored_base",
        "coverage_pct",
        "coverage_pct_base",
        "r2",
        "rmse",
        "mae",
        "corr2",
        "r2_base",
        "rmse_base",
        "mae_base",
        "corr2_base",
        "n_test_total",
        "n_test_scored",
        "n_test_total_base",
        "n_test_scored_base",
        "thoi_o",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = _safe_numeric(out[col])
    return out


def canonicalize_global_metrics(
    global_df: pd.DataFrame,
    experiment_row: dict,
    rung_spec: dict,
    candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    if global_df is None or global_df.empty:
        return _empty_frame(GLOBAL_COLUMNS)

    df = _normalize_metric_columns(global_df)
    df["candidate_id"] = _safe_str(df["model_id"])
    merged = df.merge(
        candidate_registry,
        on="candidate_id",
        how="left",
        suffixes=("", "_reg"),
    )
    merged["feature_id"] = merged.get("feature_id", merged["candidate_id"])
    merged["objective"] = merged.get("objective_reg", merged.get("objective")).fillna(merged.get("objective"))
    merged["score"] = merged.get("score_reg", merged.get("score"))
    merged["rank"] = merged.get("rank_reg", merged.get("rank"))
    merged["order"] = merged.get("order_reg", merged.get("order"))
    merged["predictors_identity"] = merged.get("predictors_identity_reg", merged.get("predictors_identity"))
    merged["predictors_identity_n"] = merged.get("predictors_identity_n_reg", merged.get("predictors_identity_n"))
    merged["thoi_o"] = merged.get("thoi_o_reg", merged.get("thoi_o", merged.get("score")))

    for key, value in experiment_row.items():
        merged[key] = value
    merged["rung_id"] = rung_spec["rung_id"]
    merged["rung_label"] = rung_spec["display_label"]

    merged = merged.rename(
        columns={
            "global_oof_r2": "full_r2",
            "global_oof_rmse": "full_rmse",
            "global_oof_mae": "full_mae",
            "global_oof_corr2": "full_corr2",
            "global_oof_r2_base": "base_r2",
            "global_oof_rmse_base": "base_rmse",
            "global_oof_mae_base": "base_mae",
            "global_oof_corr2_base": "base_corr2",
        }
    )
    merged["objective"] = merged["objective"].astype(str).str.lower()
    merged = _ensure_columns(merged, GLOBAL_COLUMNS)
    return merged[GLOBAL_COLUMNS].copy()


def canonicalize_country_metrics(
    country_df: pd.DataFrame,
    experiment_row: dict,
    rung_spec: dict,
    candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    if country_df is None or country_df.empty:
        return _empty_frame(COUNTRY_COLUMNS)

    df = _normalize_metric_columns(country_df)
    df["candidate_id"] = _safe_str(df["model_id"])
    merged = df.merge(
        candidate_registry,
        on="candidate_id",
        how="left",
        suffixes=("", "_reg"),
    )
    merged["feature_id"] = merged.get("feature_id", merged["candidate_id"])
    merged["objective"] = merged.get("objective_reg", merged.get("objective")).fillna(merged.get("objective"))
    merged["score"] = merged.get("score_reg", merged.get("score"))
    merged["rank"] = merged.get("rank_reg", merged.get("rank"))
    merged["order"] = merged.get("order_reg", merged.get("order"))
    merged["predictors_identity"] = merged.get("predictors_identity_reg", merged.get("predictors_identity"))
    merged["predictors_identity_n"] = merged.get("predictors_identity_n_reg", merged.get("predictors_identity_n"))
    merged["thoi_o"] = merged.get("thoi_o_reg", merged.get("thoi_o", merged.get("score")))

    for key, value in experiment_row.items():
        merged[key] = value
    merged["rung_id"] = rung_spec["rung_id"]
    merged["rung_label"] = rung_spec["display_label"]

    merged = merged.rename(
        columns={
            "r2": "country_full_r2",
            "rmse": "country_full_rmse",
            "mae": "country_full_mae",
            "corr2": "country_full_corr2",
            "r2_base": "country_base_r2",
            "rmse_base": "country_base_rmse",
            "mae_base": "country_base_mae",
            "corr2_base": "country_base_corr2",
            "delta_r2_vs_base": "country_delta_r2_vs_base",
            "delta_rmse_vs_base": "country_delta_rmse_vs_base",
            "delta_mae_vs_base": "country_delta_mae_vs_base",
            "delta_corr2_vs_base": "country_delta_corr2_vs_base",
        }
    )
    merged["objective"] = merged["objective"].astype(str).str.lower()
    merged = _ensure_columns(merged, COUNTRY_COLUMNS)
    return merged[COUNTRY_COLUMNS].copy()


def compute_ladder_contrasts(metrics_global_long: pd.DataFrame) -> pd.DataFrame:
    if metrics_global_long is None or metrics_global_long.empty:
        return pd.DataFrame(
            columns=[
                *META_COLUMNS,
                "candidate_id",
                "feature_id",
                "objective",
                "order",
                "score",
                "rank",
                "predictors_identity",
                "predictors_identity_n",
                "thoi_o",
                "contrast_name",
                "lhs_rung_id",
                "rhs_rung_id",
                "lhs_rung_label",
                "rhs_rung_label",
                "full_r2_diff",
                "delta_r2_diff",
                "full_mae_diff",
                "delta_mae_diff",
            ]
        )

    keys = [
        *META_COLUMNS,
        "candidate_id",
        "feature_id",
        "objective",
        "order",
        "score",
        "rank",
        "predictors_identity",
        "predictors_identity_n",
        "thoi_o",
    ]
    cols = keys + ["rung_id", "full_r2", "delta_r2_vs_base", "full_mae", "delta_mae_vs_base"]
    df = _ensure_columns(metrics_global_long, cols)[cols].copy()
    out = []
    for contrast_name, lhs, rhs in CONTRAST_DEFS:
        lhs_df = df[df["rung_id"] == lhs].copy()
        rhs_df = df[df["rung_id"] == rhs].copy()
        if lhs_df.empty or rhs_df.empty:
            continue
        lhs_df = lhs_df.rename(
            columns={
                "full_r2": "lhs_full_r2",
                "delta_r2_vs_base": "lhs_delta_r2_vs_base",
                "full_mae": "lhs_full_mae",
                "delta_mae_vs_base": "lhs_delta_mae_vs_base",
            }
        )
        rhs_df = rhs_df.rename(
            columns={
                "full_r2": "rhs_full_r2",
                "delta_r2_vs_base": "rhs_delta_r2_vs_base",
                "full_mae": "rhs_full_mae",
                "delta_mae_vs_base": "rhs_delta_mae_vs_base",
            }
        )
        merged = lhs_df.merge(rhs_df[keys + ["rhs_full_r2", "rhs_delta_r2_vs_base", "rhs_full_mae", "rhs_delta_mae_vs_base"]], on=keys, how="inner")
        if merged.empty:
            continue
        merged["contrast_name"] = contrast_name
        merged["lhs_rung_id"] = lhs
        merged["rhs_rung_id"] = rhs
        merged["lhs_rung_label"] = RUNG_LABELS.get(lhs, lhs)
        merged["rhs_rung_label"] = RUNG_LABELS.get(rhs, rhs)
        merged["full_r2_diff"] = merged["lhs_full_r2"] - merged["rhs_full_r2"]
        merged["delta_r2_diff"] = merged["lhs_delta_r2_vs_base"] - merged["rhs_delta_r2_vs_base"]
        merged["full_mae_diff"] = merged["lhs_full_mae"] - merged["rhs_full_mae"]
        merged["delta_mae_diff"] = merged["lhs_delta_mae_vs_base"] - merged["rhs_delta_mae_vs_base"]
        out.append(
            merged[
                keys
                + [
                    "contrast_name",
                    "lhs_rung_id",
                    "rhs_rung_id",
                    "lhs_rung_label",
                    "rhs_rung_label",
                    "full_r2_diff",
                    "delta_r2_diff",
                    "full_mae_diff",
                    "delta_mae_diff",
                ]
            ].copy()
        )
    return pd.concat(out, ignore_index=True) if out else _empty_frame(
        [
            *META_COLUMNS,
            "candidate_id",
            "feature_id",
            "objective",
            "order",
            "score",
            "rank",
            "predictors_identity",
            "predictors_identity_n",
            "thoi_o",
            "contrast_name",
            "lhs_rung_id",
            "rhs_rung_id",
            "lhs_rung_label",
            "rhs_rung_label",
            "full_r2_diff",
            "delta_r2_diff",
            "full_mae_diff",
            "delta_mae_diff",
        ]
    )


def _spearman_stats(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    x2 = x[mask]
    y2 = y[mask]
    if _spearmanr is None:
        return float(x2.corr(y2, method="spearman")), np.nan
    rho, pvalue = _spearmanr(x2.to_numpy(), y2.to_numpy())
    return float(rho), float(pvalue)


def _fit_slope(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2 or x[mask].nunique() < 2:
        return np.nan
    coeff = np.polyfit(x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float), deg=1)
    return float(coeff[0])


def compute_frontier_summary(metrics_global_long: pd.DataFrame) -> pd.DataFrame:
    if metrics_global_long is None or metrics_global_long.empty:
        return pd.DataFrame(
            columns=[
                *META_COLUMNS,
                "objective",
                "rung_id",
                "rung_label",
                "bin_idx",
                "bin_n",
                "thoi_o_median",
                "full_r2_q50",
                "full_r2_q75",
                "full_r2_q90",
                "full_r2_q95",
                "candidate_n",
                "q95_slope",
                "spearman_rho",
                "spearman_pvalue",
            ]
        )

    base = metrics_global_long.copy()
    base = base[base["objective"].isin(["o_max", "o_min"])].copy()
    metric_col = "thoi_o"
    base[metric_col] = _safe_numeric(base[metric_col])
    base["full_r2"] = _safe_numeric(base["full_r2"])
    out = []
    group_cols = [*META_COLUMNS, "objective", "rung_id", "rung_label"]
    for group_key, group in base.groupby(group_cols, dropna=False, observed=True):
        group = group.dropna(subset=[metric_col, "full_r2"]).sort_values(metric_col).reset_index(drop=True)
        if group.empty:
            continue
        n = len(group)
        n_bins = max(1, min(FRONTIER_N_BINS, n // FRONTIER_MIN_BIN_SIZE if n >= FRONTIER_MIN_BIN_SIZE else 1))
        if n_bins <= 1:
            group["bin_idx"] = 0
        else:
            ranked = group[metric_col].rank(method="first")
            group["bin_idx"] = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
            group["bin_idx"] = _safe_numeric(group["bin_idx"]).fillna(0).astype(int)
        binned = (
            group.groupby("bin_idx", dropna=False)
            .agg(
                bin_n=("candidate_id", "size"),
                thoi_o_median=("thoi_o", "median"),
                full_r2_q50=("full_r2", lambda s: s.quantile(0.50)),
                full_r2_q75=("full_r2", lambda s: s.quantile(0.75)),
                full_r2_q90=("full_r2", lambda s: s.quantile(0.90)),
                full_r2_q95=("full_r2", lambda s: s.quantile(0.95)),
            )
            .reset_index()
        )
        rho, pvalue = _spearman_stats(group[metric_col], group["full_r2"])
        q95_slope = _fit_slope(binned["thoi_o_median"], binned["full_r2_q95"])
        for col_name, value in zip(group_cols, group_key):
            binned[col_name] = value
        binned["candidate_n"] = n
        binned["q95_slope"] = q95_slope
        binned["spearman_rho"] = rho
        binned["spearman_pvalue"] = pvalue
        out.append(
            binned[
                [
                    *META_COLUMNS,
                    "objective",
                    "rung_id",
                    "rung_label",
                    "bin_idx",
                    "bin_n",
                    "thoi_o_median",
                    "full_r2_q50",
                    "full_r2_q75",
                    "full_r2_q90",
                    "full_r2_q95",
                    "candidate_n",
                    "q95_slope",
                    "spearman_rho",
                    "spearman_pvalue",
                ]
            ].copy()
        )
    return pd.concat(out, ignore_index=True) if out else _empty_frame(
        [
            *META_COLUMNS,
            "objective",
            "rung_id",
            "rung_label",
            "bin_idx",
            "bin_n",
            "thoi_o_median",
            "full_r2_q50",
            "full_r2_q75",
            "full_r2_q90",
            "full_r2_q95",
            "candidate_n",
            "q95_slope",
            "spearman_rho",
            "spearman_pvalue",
        ]
    )


def compute_top_tail_summary(metrics_global_long: pd.DataFrame) -> pd.DataFrame:
    if metrics_global_long is None or metrics_global_long.empty:
        return pd.DataFrame(
            columns=[
                *META_COLUMNS,
                "rung_id",
                "rung_label",
                "candidate_id",
                "objective",
                "order",
                "predictors_identity",
                "predictors_identity_n",
                "thoi_o",
                "full_r2",
                "delta_r2_vs_base",
                "tail_rank",
                "tail_n_total",
                "tail_n_selected",
                "tail_fraction",
                "tail_threshold_full_r2",
            ]
        )

    base = metrics_global_long.copy()
    base["full_r2"] = _safe_numeric(base["full_r2"])
    base["delta_r2_vs_base"] = _safe_numeric(base["delta_r2_vs_base"])
    metric_col = "thoi_o"
    base[metric_col] = _safe_numeric(base[metric_col])
    out = []
    group_cols = [*META_COLUMNS, "objective", "rung_id", "rung_label"]
    for group_key, group in base.groupby(group_cols, dropna=False, observed=True):
        group = group.sort_values(
            by=["full_r2", "delta_r2_vs_base", "thoi_o", "candidate_id"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        n_total = len(group)
        if n_total == 0:
            continue
        n_selected = min(n_total, max(TOP_TAIL_MIN_N, int(math.ceil(TOP_TAIL_FRAC * n_total))))
        tail = group.head(n_selected).copy()
        tail["tail_rank"] = np.arange(1, len(tail) + 1)
        tail["tail_n_total"] = n_total
        tail["tail_n_selected"] = n_selected
        tail["tail_fraction"] = TOP_TAIL_FRAC
        tail["tail_threshold_full_r2"] = _safe_numeric(tail["full_r2"]).min()
        out.append(
            tail[
                [
                    *META_COLUMNS,
                    "rung_id",
                    "rung_label",
                    "candidate_id",
                    "objective",
                    "order",
                    "predictors_identity",
                    "predictors_identity_n",
                    "thoi_o",
                    "full_r2",
                    "delta_r2_vs_base",
                    "tail_rank",
                    "tail_n_total",
                    "tail_n_selected",
                    "tail_fraction",
                    "tail_threshold_full_r2",
                ]
            ].copy()
        )
    return pd.concat(out, ignore_index=True) if out else _empty_frame(
        [
            *META_COLUMNS,
            "rung_id",
            "rung_label",
            "candidate_id",
            "objective",
            "order",
            "predictors_identity",
            "predictors_identity_n",
            "thoi_o",
            "full_r2",
            "delta_r2_vs_base",
            "tail_rank",
            "tail_n_total",
            "tail_n_selected",
            "tail_fraction",
            "tail_threshold_full_r2",
        ]
    )


def compute_portability_comparison_global(
    pooled_reference_global: pd.DataFrame,
    portability_global: pd.DataFrame,
) -> pd.DataFrame:
    if pooled_reference_global is None or pooled_reference_global.empty or portability_global is None or portability_global.empty:
        return pd.DataFrame()

    join_keys = ["bag_target", "objective", "rung_id", "predictors_identity"]
    ref_cols = [
        "candidate_id",
        "feature_id",
        "bag_target",
        "objective",
        "rung_id",
        "rung_label",
        "predictors_identity",
        "predictors_identity_n",
        "thoi_o",
        "order",
        "full_r2",
        "delta_r2_vs_base",
        "tail_rank",
        "tail_n_selected",
        "tail_n_total",
        "source_experiment_id",
    ]
    ref = _ensure_columns(pooled_reference_global, ref_cols)[ref_cols].copy().rename(
        columns={
            "candidate_id": "reference_candidate_id",
            "feature_id": "reference_feature_id",
            "full_r2": "pooled_full_r2",
            "delta_r2_vs_base": "pooled_delta_r2_vs_base",
            "tail_rank": "pooled_tail_rank",
            "tail_n_selected": "pooled_tail_n_selected",
            "tail_n_total": "pooled_tail_n_total",
            "source_experiment_id": "pooled_source_experiment_id",
        }
    )

    tgt_cols = META_COLUMNS + [
        "candidate_id",
        "feature_id",
        "objective",
        "rung_id",
        "rung_label",
        "predictors_identity",
        "predictors_identity_n",
        "thoi_o",
        "order",
        "full_r2",
        "delta_r2_vs_base",
        "n_scored",
        "coverage_pct",
    ]
    tgt = _ensure_columns(portability_global, tgt_cols)[tgt_cols].copy().rename(
        columns={
            "candidate_id": "target_candidate_id",
            "feature_id": "target_feature_id",
            "full_r2": "target_full_r2",
            "delta_r2_vs_base": "target_delta_r2_vs_base",
            "n_scored": "target_n_scored",
            "coverage_pct": "target_coverage_pct",
        }
    )

    out = tgt.merge(ref, on=join_keys, how="left")
    out["shift_full_r2_vs_pooled"] = out["target_full_r2"] - out["pooled_full_r2"]
    out["shift_delta_r2_vs_pooled"] = out["target_delta_r2_vs_base"] - out["pooled_delta_r2_vs_base"]
    return out


def compute_portability_comparison_country(
    pooled_reference_country: pd.DataFrame,
    portability_country: pd.DataFrame,
) -> pd.DataFrame:
    if pooled_reference_country is None or pooled_reference_country.empty or portability_country is None or portability_country.empty:
        return pd.DataFrame()

    join_keys = ["bag_target", "objective", "rung_id", "predictors_identity", "fold_country"]
    ref_cols = [
        "bag_target",
        "objective",
        "rung_id",
        "predictors_identity",
        "fold_country",
        "country_full_r2",
        "country_delta_r2_vs_base",
        "n_test_scored",
        "coverage_pct",
    ]
    ref = _ensure_columns(pooled_reference_country, ref_cols)[ref_cols].copy().rename(
        columns={
            "country_full_r2": "pooled_country_full_r2",
            "country_delta_r2_vs_base": "pooled_country_delta_r2_vs_base",
            "n_test_scored": "pooled_n_test_scored",
            "coverage_pct": "pooled_coverage_pct",
        }
    )

    tgt_cols = META_COLUMNS + [
        "objective",
        "rung_id",
        "predictors_identity",
        "fold_country",
        "country_full_r2",
        "country_delta_r2_vs_base",
        "n_test_scored",
        "coverage_pct",
    ]
    tgt = _ensure_columns(portability_country, tgt_cols)[tgt_cols].copy().rename(
        columns={
            "country_full_r2": "target_country_full_r2",
            "country_delta_r2_vs_base": "target_country_delta_r2_vs_base",
            "n_test_scored": "target_n_test_scored",
            "coverage_pct": "target_coverage_pct",
        }
    )

    out = tgt.merge(ref, on=join_keys, how="left")
    out["shift_country_full_r2_vs_pooled"] = out["target_country_full_r2"] - out["pooled_country_full_r2"]
    out["shift_country_delta_r2_vs_pooled"] = out["target_country_delta_r2_vs_base"] - out["pooled_country_delta_r2_vs_base"]
    return out


def compute_transfer_comparison_global(
    transfer_global: pd.DataFrame,
    dx_refit_global: pd.DataFrame,
) -> pd.DataFrame:
    if transfer_global is None or transfer_global.empty or dx_refit_global is None or dx_refit_global.empty:
        return pd.DataFrame()

    join_keys = ["bag_target", "objective", "rung_id", "predictors_identity", "test_diagnosis_group"]
    lhs_cols = META_COLUMNS + [
        "candidate_id",
        "feature_id",
        "objective",
        "rung_id",
        "rung_label",
        "predictors_identity",
        "predictors_identity_n",
        "thoi_o",
        "order",
        "full_r2",
        "delta_r2_vs_base",
        "n_scored",
        "coverage_pct",
    ]
    lhs = _ensure_columns(transfer_global, lhs_cols)[lhs_cols].copy().rename(
        columns={
            "candidate_id": "transfer_candidate_id",
            "feature_id": "transfer_feature_id",
            "full_r2": "transfer_full_r2",
            "delta_r2_vs_base": "transfer_delta_r2_vs_base",
            "n_scored": "transfer_n_scored",
            "coverage_pct": "transfer_coverage_pct",
        }
    )

    rhs_cols = META_COLUMNS + [
        "candidate_id",
        "feature_id",
        "objective",
        "rung_id",
        "predictors_identity",
        "full_r2",
        "delta_r2_vs_base",
        "n_scored",
        "coverage_pct",
    ]
    rhs = _ensure_columns(dx_refit_global, rhs_cols)[rhs_cols].copy().rename(
        columns={
            "candidate_id": "dx_refit_candidate_id",
            "feature_id": "dx_refit_feature_id",
            "full_r2": "dx_refit_full_r2",
            "delta_r2_vs_base": "dx_refit_delta_r2_vs_base",
            "n_scored": "dx_refit_n_scored",
            "coverage_pct": "dx_refit_coverage_pct",
        }
    )

    out = lhs.merge(rhs[join_keys + ["dx_refit_candidate_id", "dx_refit_feature_id", "dx_refit_full_r2", "dx_refit_delta_r2_vs_base", "dx_refit_n_scored", "dx_refit_coverage_pct"]], on=join_keys, how="left")
    out["shift_full_r2_transfer_minus_refit"] = out["transfer_full_r2"] - out["dx_refit_full_r2"]
    out["shift_delta_r2_transfer_minus_refit"] = out["transfer_delta_r2_vs_base"] - out["dx_refit_delta_r2_vs_base"]
    return out


def compute_transfer_comparison_country(
    transfer_country: pd.DataFrame,
    dx_refit_country: pd.DataFrame,
) -> pd.DataFrame:
    if transfer_country is None or transfer_country.empty or dx_refit_country is None or dx_refit_country.empty:
        return pd.DataFrame()

    join_keys = ["bag_target", "objective", "rung_id", "predictors_identity", "test_diagnosis_group", "fold_country"]
    lhs_cols = META_COLUMNS + [
        "objective",
        "rung_id",
        "predictors_identity",
        "fold_country",
        "country_full_r2",
        "country_delta_r2_vs_base",
        "n_test_scored",
        "coverage_pct",
    ]
    lhs = _ensure_columns(transfer_country, lhs_cols)[lhs_cols].copy().rename(
        columns={
            "country_full_r2": "transfer_country_full_r2",
            "country_delta_r2_vs_base": "transfer_country_delta_r2_vs_base",
            "n_test_scored": "transfer_n_test_scored",
            "coverage_pct": "transfer_coverage_pct",
        }
    )

    rhs_cols = META_COLUMNS + [
        "objective",
        "rung_id",
        "predictors_identity",
        "fold_country",
        "country_full_r2",
        "country_delta_r2_vs_base",
        "n_test_scored",
        "coverage_pct",
    ]
    rhs = _ensure_columns(dx_refit_country, rhs_cols)[rhs_cols].copy().rename(
        columns={
            "country_full_r2": "dx_refit_country_full_r2",
            "country_delta_r2_vs_base": "dx_refit_country_delta_r2_vs_base",
            "n_test_scored": "dx_refit_n_test_scored",
            "coverage_pct": "dx_refit_coverage_pct",
        }
    )

    out = lhs.merge(
        rhs[join_keys + ["dx_refit_country_full_r2", "dx_refit_country_delta_r2_vs_base", "dx_refit_n_test_scored", "dx_refit_coverage_pct"]],
        on=join_keys,
        how="left",
    )
    out["shift_country_full_r2_transfer_minus_refit"] = out["transfer_country_full_r2"] - out["dx_refit_country_full_r2"]
    out["shift_country_delta_r2_transfer_minus_refit"] = out["transfer_country_delta_r2_vs_base"] - out["dx_refit_country_delta_r2_vs_base"]
    return out


def sort_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    if "objective" in out.columns:
        out["objective"] = pd.Categorical(out["objective"], categories=["o_max", "o_min"], ordered=True)
    if "rung_id" in out.columns:
        out["rung_id"] = pd.Categorical(out["rung_id"], categories=RUNG_ORDER, ordered=True)
    sort_cols = [c for c in ["experiment_id", "objective", "rung_id", "tail_rank", "candidate_id", "fold_country"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)
