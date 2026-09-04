#!/usr/bin/env python3
"""
run_single_exposure_eval.py
===========================
Single-exposure LOCO XGB d2 evaluation: every exposome feature individually.

Scientific purpose
------------------
Establishes the per-feature performance ceiling so that the fold-gain of the
best synergistic multi-domain combination over the best individual exposure
can be quantified.  This is the direct analogue of Legaz et al.'s "aggregated
exposome explained X-fold more variance than individual exposures."

Design
------
For each of the 63 exposome features:
  1. Fit XGB d2 with covariates (age, sex, diagnosis, year) + that ONE feature.
  2. Same LOCO-country CV, identical hyperparameters to the main pipeline.
  3. Record global OOF R² and per-country R².

Additionally, a covariate-only baseline (0 exposome features) is evaluated
for exact ΔR² computation.

Usage
-----
    # All three BAGs in one run:
    python run_single_exposure_eval.py

    # Specific BAG:
    BAG_TARGET_MODE=functional python run_single_exposure_eval.py

    # Smoke test:
    SMOKE_TEST=1 python run_single_exposure_eval.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from scripts.pipeline_utils import (
    build_analysis_cfg,
    default_perf_cfg,
    default_paths,
    default_xgb_cfg,
    env_list,
    load_stage_config,
)

OLS_MODE = os.environ.get("SINGLE_EXPOSURE_OLS_MODE", "0").strip() in ("1", "true", "yes")
from scripts.exposome_domains import load_domain_map

from loco_fusion_matrix_engine import (
    build_model_df,
    fit_predict_baseline_across_folds,
    fit_predict_one_rep_across_folds,
    load_artifacts_greedy_only,
    pin_blas_threads,
    prepare_bag_context,
    regression_metrics_extended,
    target_map,
)
from xgb_loco_engine import (
    _build_base_fold_mats,
    _fit_xgb_fold,
    _append_predictors,
    require_xgboost,
)

# ── config ────────────────────────────────────────────────────────────────────
BAG_TARGET_MODE = os.environ.get("BAG_TARGET_MODE", "all").strip().lower()
SMOKE_TEST      = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
STAGE_CFG = load_stage_config("single_exposure", smoke=SMOKE_TEST)

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", "outputs/variant_a/single_exposure_eval"),
))
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", STAGE_CFG.get("greedy_root", "outputs/greedy")))

PATHS = default_paths(STAGE_CFG)

ANALYSIS_CFG = build_analysis_cfg(STAGE_CFG)
ANALYSIS_CFG["exclude_diagnosis"] = env_list(
    "V3_EXCLUDE_DIAGNOSIS",
    ANALYSIS_CFG.get("exclude_diagnosis", "Other,AFM"),
)
ANALYSIS_CFG["exclude_countries"] = env_list(
    "V3_EXCLUDE_COUNTRIES",
    ANALYSIS_CFG.get("exclude_countries", "North Macedonia,New Zeland,Belgium"),
)

CV_CFG = {
    "split_col": "country_clean",
    "min_country_size_test": 0,
    "unseen_diag_policy": "drop_test_rows",
}

XGB_CFG = {**default_xgb_cfg(), **STAGE_CFG.get("xgb_cfg", {})}

EARLY_STOP_CFG = {"val_country_frac": 0.20, "val_country_min": 1}

PERF_CFG = default_perf_cfg(STAGE_CFG)

MATRIX_CFG = {"rcond": None}  # identical to oinfo_bag_ladder/config.py

# ── canonical domain map ─────────────────────────────────────────────────────
DOMAIN_LABELS_CSV = Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv",
))
_DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)


# ── OLS fitting helpers (np.linalg.lstsq, identical to main-ladder OLS rung) ──

def _ols_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_test @ beta


# ── core fitting function ─────────────────────────────────────────────────────

def _fit_single_feature(
    feat_idx: int,
    feat_name: str,
    context: dict,
    fold_designs: dict,
    xgb_cfg: dict,
    min_n: int,
) -> tuple[dict, list[dict]]:
    """
    Fit one feature + covariates across all LOCO folds.
    Returns (global_row_dict, list_of_country_row_dicts).
    """
    if not OLS_MODE:
        xgb       = require_xgboost()
    y         = np.asarray(context["y"],     dtype=float)
    X_exp     = np.asarray(context["X_exp"], dtype=np.float32)
    base_seed = int(xgb_cfg.get("random_state", 20260304))

    country_rows: list[dict] = []
    y_true_all:   list[np.ndarray] = []
    y_pred_all:   list[np.ndarray] = []

    pred_all = [feat_idx]

    for fold_i, country in enumerate(context["countries"]):
        fd        = fold_designs[country]
        tr_inner  = fd["tr_inner_idx"]
        val_idx   = fd["val_idx"]
        train_idx = fd["train_idx"]
        test_idx  = fd["test_idx"]

        # Drop zero-variance predictors
        pred_used = pred_all
        if pred_all:
            xtr  = X_exp[np.ix_(train_idx, pred_all)]
            keep = np.nanvar(xtr, axis=0) > 0
            pred_used = [i for i, k in zip(pred_all, keep) if k]

        X_tr_inner = _append_predictors(fd["Xb_train_inner"], X_exp, tr_inner,  pred_used)
        X_val_mat  = _append_predictors(fd["Xb_val"],         X_exp, val_idx,   pred_used)
        X_test_mat = _append_predictors(fd["Xb_test"],        X_exp, test_idx,  pred_used)

        try:
            if OLS_MODE:
                # Use full train set (inner + val) for OLS — no early stopping needed
                X_train_full = _append_predictors(fd["Xb_train_full"], X_exp, train_idx, pred_used)
                y_pred = _ols_predict(
                    X_train_full.astype(float), y[train_idx],
                    X_test_mat.astype(float),
                )
            else:
                xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}
                reg    = _fit_xgb_fold(xgb, xgb_params, X_tr_inner, y[tr_inner],
                                       X_val_mat, y[val_idx])
                y_pred = reg.predict(X_test_mat)
        except Exception:
            y_pred = np.full(len(test_idx), np.nan)

        y_true = y[test_idx]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        n  = int(ok.sum())
        if n >= min_n:
            met = regression_metrics_extended(y_true[ok], y_pred[ok], min_n=min_n)
            country_rows.append({
                "feature_name":  feat_name,
                "feature_idx":   feat_idx,
                "fold_country":  country,
                "n_test":        n,
                "r2":            met["r2"],
                "rmse":          met["rmse"],
                "mae":           met["mae"],
                "corr2":         met["corr2"],
            })

    # Global OOF R²
    yt   = np.concatenate(y_true_all)
    yp   = np.concatenate(y_pred_all)
    ok_g = np.isfinite(yt) & np.isfinite(yp)

    global_row = {
        "feature_name":  feat_name,
        "feature_idx":   feat_idx,
        "domain":        _DOMAIN_MAP.get(feat_name, "Other"),
    }
    if ok_g.sum() >= min_n:
        met_g = regression_metrics_extended(yt[ok_g], yp[ok_g], min_n=min_n)
        global_row.update({
            "global_oof_r2":  met_g["r2"],
            "global_rmse":    met_g["rmse"],
            "global_mae":     met_g["mae"],
            "global_corr2":   met_g["corr2"],
            "n_total":        int(ok_g.sum()),
        })
    else:
        global_row.update({
            "global_oof_r2": np.nan,
            "global_rmse": np.nan,
            "global_mae": np.nan,
            "global_corr2": np.nan,
            "n_total": int(ok_g.sum()),
        })

    return global_row, country_rows


def _fit_baseline_only(
    context: dict,
    fold_designs: dict,
    xgb_cfg: dict,
    min_n: int,
) -> tuple[dict, list[dict]]:
    """
    Fit covariate-only model (no exposome features) across all LOCO folds.
    """
    if not OLS_MODE:
        xgb       = require_xgboost()
    y         = np.asarray(context["y"], dtype=float)
    base_seed = int(xgb_cfg.get("random_state", 20260304))

    country_rows: list[dict] = []
    y_true_all:   list[np.ndarray] = []
    y_pred_all:   list[np.ndarray] = []

    for fold_i, country in enumerate(context["countries"]):
        fd        = fold_designs[country]
        tr_inner  = fd["tr_inner_idx"]
        val_idx   = fd["val_idx"]
        train_idx = fd["train_idx"]
        test_idx  = fd["test_idx"]

        X_tr_inner = fd["Xb_train_inner"]
        X_val_mat  = fd["Xb_val"]
        X_test_mat = fd["Xb_test"]

        try:
            if OLS_MODE:
                y_pred = _ols_predict(
                    fd["Xb_train_full"].astype(float), y[train_idx],
                    X_test_mat.astype(float),
                )
            else:
                xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}
                reg    = _fit_xgb_fold(xgb, xgb_params, X_tr_inner, y[tr_inner],
                                       X_val_mat, y[val_idx])
                y_pred = reg.predict(X_test_mat)
        except Exception:
            y_pred = np.full(len(test_idx), np.nan)

        y_true = y[test_idx]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        n  = int(ok.sum())
        if n >= min_n:
            met = regression_metrics_extended(y_true[ok], y_pred[ok], min_n=min_n)
            country_rows.append({
                "feature_name":  "__baseline__",
                "feature_idx":   -1,
                "fold_country":  country,
                "n_test":        n,
                "r2":            met["r2"],
                "rmse":          met["rmse"],
                "mae":           met["mae"],
                "corr2":         met["corr2"],
            })

    yt   = np.concatenate(y_true_all)
    yp   = np.concatenate(y_pred_all)
    ok_g = np.isfinite(yt) & np.isfinite(yp)

    global_row = {
        "feature_name":  "__baseline__",
        "feature_idx":   -1,
        "domain":        "baseline",
    }
    if ok_g.sum() >= min_n:
        met_g = regression_metrics_extended(yt[ok_g], yp[ok_g], min_n=min_n)
        global_row.update({
            "global_oof_r2": met_g["r2"],
            "global_rmse":   met_g["rmse"],
            "global_mae":    met_g["mae"],
            "global_corr2":  met_g["corr2"],
            "n_total":       int(ok_g.sum()),
        })
    else:
        global_row.update({
            "global_oof_r2": np.nan,
            "global_rmse": np.nan,
            "global_mae": np.nan,
            "global_corr2": np.nan,
            "n_total": int(ok_g.sum()),
        })

    return global_row, country_rows


# ── OLS per-BAG runner (uses exact same functions as the main ladder) ─────────

def run_bag_ols(
    bag_name: str,
    y_col: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    outdir_base: Path,
    smoke: bool,
) -> None:
    n_features = len(feature_names)
    outdir = outdir_base / bag_name
    if smoke:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"BAG: {bag_name}  |  n_features={n_features}  |  smoke={smoke}")
    print(f"{'─'*60}")

    context = prepare_bag_context(
        model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, feature_names,
    )
    countries = context["countries"]

    if smoke:
        countries = countries[:3]
        context["countries"] = countries
        context["train_idx_by_country"] = {c: context["train_idx_by_country"][c] for c in countries}
        context["test_idx_by_country"]  = {c: context["test_idx_by_country"][c]  for c in countries}
        context["year_basis_by_country"] = {c: context["year_basis_by_country"][c] for c in countries}

    print(f"  {len(countries)} LOCO folds, {len(context['y'])} subjects with valid BAG")

    min_n = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    # ── baseline: exact same call as main ladder ──────────────────────────
    print("  Fitting covariate-only baseline...")
    t0 = time.time()
    _, base_pred_df, _ = fit_predict_baseline_across_folds(context, MATRIX_CFG)
    base_scored = base_pred_df[base_pred_df["scored"]] if "scored" in base_pred_df.columns \
        else base_pred_df[base_pred_df["y_pred"].notna()]
    base_met = regression_metrics_extended(
        base_scored["y_true"].values, base_scored["y_pred"].values, min_n=min_n,
    )
    base_r2 = base_met["r2"]
    print(f"    baseline R² = {base_r2:.6f}  ({time.time() - t0:.1f}s)")

    base_country_rows = []
    for country, grp in base_scored.groupby("fold_country"):
        ok = np.isfinite(grp["y_true"].values) & np.isfinite(grp["y_pred"].values)
        if ok.sum() >= min_n:
            m = regression_metrics_extended(grp["y_true"].values[ok], grp["y_pred"].values[ok], min_n=min_n)
            base_country_rows.append({
                "feature_name": "__baseline__", "feature_idx": -1,
                "fold_country": country, "n_test": int(ok.sum()),
                "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "corr2": m["corr2"],
            })
    base_global = {
        "feature_name": "__baseline__", "feature_idx": -1, "domain": "baseline",
        "global_oof_r2": base_r2, "global_rmse": base_met["rmse"],
        "global_mae": base_met["mae"], "global_corr2": base_met["corr2"],
        "n_total": int(base_met["n_scored"]),
    }

    # ── single-feature models ─────────────────────────────────────────────
    if smoke:
        feat_indices = list(range(min(3, n_features)))
    else:
        feat_indices = list(range(n_features))

    n_jobs = int(PERF_CFG["n_jobs"])
    print(f"  Fitting {len(feat_indices)} single-feature models (n_jobs={n_jobs})...")
    t0 = time.time()

    def _fit_ols_feature(feat_idx, feat_name):
        rep_row = {
            "model_id": f"single__{feat_name}",
            "predictor_idx_list": [feat_idx],
            "predictors_identity": feat_name,
            "predictors_identity_n": 1,
        }
        summary, pred_df, _ = fit_predict_one_rep_across_folds(rep_row, context, MATRIX_CFG)
        scored = pred_df[pred_df["scored"]] if "scored" in pred_df.columns \
            else pred_df[pred_df["y_pred"].notna()]
        global_row = {
            "feature_name": feat_name, "feature_idx": feat_idx,
            "domain": _DOMAIN_MAP.get(feat_name, "Other"),
            "global_oof_r2": summary.get("global_oof_r2", np.nan),
            "global_rmse": summary.get("global_oof_rmse", np.nan),
            "global_mae": summary.get("global_oof_mae", np.nan),
            "global_corr2": summary.get("global_oof_corr2", np.nan),
            "n_total": int(summary.get("n_scored", 0)),
        }
        country_rows = []
        for country, grp in scored.groupby("fold_country"):
            ok = np.isfinite(grp["y_true"].values) & np.isfinite(grp["y_pred"].values)
            if ok.sum() >= min_n:
                m = regression_metrics_extended(grp["y_true"].values[ok], grp["y_pred"].values[ok], min_n=min_n)
                country_rows.append({
                    "feature_name": feat_name, "feature_idx": feat_idx,
                    "fold_country": country, "n_test": int(ok.sum()),
                    "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "corr2": m["corr2"],
                })
        return global_row, country_rows

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_fit_ols_feature)(fi, feature_names[fi])
        for fi in tqdm(feat_indices, desc=f"features ({bag_name})")
    )

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  ({elapsed / max(len(feat_indices), 1):.1f}s per feature)")

    global_rows  = [base_global] + [r[0] for r in results]
    country_rows = base_country_rows + [row for r in results for row in r[1]]

    global_df  = pd.DataFrame(global_rows)
    country_df = pd.DataFrame(country_rows)

    if np.isfinite(base_r2):
        global_df["delta_r2_vs_base"] = global_df["global_oof_r2"] - base_r2
    else:
        global_df["delta_r2_vs_base"] = np.nan

    global_df = global_df.sort_values("global_oof_r2", ascending=False).reset_index(drop=True)

    gp = outdir / "single_exposure_global.csv"
    cp = outdir / "single_exposure_country.csv"
    global_df.to_csv(gp, index=False)
    country_df.to_csv(cp, index=False)
    print(f"\n  ✓ Saved:\n    {gp}\n    {cp}")

    feats_only = global_df[global_df["feature_name"] != "__baseline__"].copy()
    if not feats_only.empty:
        best = feats_only.iloc[0]
        print(f"\n  ── Summary ({bag_name}) ──")
        print(f"  Baseline R²:       {base_r2:.6f}")
        print(f"  Best single feat:  {best['feature_name']}  "
              f"R²={best['global_oof_r2']:.6f}  ΔR²={best['delta_r2_vs_base']:.6f}  "
              f"domain={best['domain']}")


# ── per-BAG runner ────────────────────────────────────────────────────────────

def run_bag(
    bag_name: str,
    y_col: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    outdir_base: Path,
    smoke: bool,
) -> None:
    n_features = len(feature_names)
    outdir = outdir_base / bag_name
    if smoke:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"BAG: {bag_name}  |  n_features={n_features}  |  smoke={smoke}")
    print(f"{'─'*60}")

    context = prepare_bag_context(
        model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, feature_names,
    )
    countries = context["countries"]

    if smoke:
        countries = countries[:3]
        context["countries"] = countries
        context["train_idx_by_country"] = {c: context["train_idx_by_country"][c] for c in countries}
        context["test_idx_by_country"]  = {c: context["test_idx_by_country"][c]  for c in countries}

    print(f"  {len(countries)} LOCO folds, {len(context['y'])} subjects with valid BAG")

    print("  Building fold matrices...")
    base_seed = int(XGB_CFG["random_state"])
    fold_designs = {}
    for i, c in enumerate(tqdm(countries, desc="fold mats", leave=False)):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG,
            seed=base_seed + i,
        )

    n_jobs = int(PERF_CFG["n_jobs"])
    min_n  = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    # ── feature list ──────────────────────────────────────────────────────
    if smoke:
        feat_indices = list(range(min(3, n_features)))
    else:
        feat_indices = list(range(n_features))

    # ── baseline (covariate-only) ─────────────────────────────────────────
    print("  Fitting covariate-only baseline...")
    t0 = time.time()
    base_global, base_country = _fit_baseline_only(context, fold_designs, XGB_CFG, min_n)
    print(f"    baseline R² = {base_global.get('global_oof_r2', float('nan')):.6f}"
          f"  ({time.time() - t0:.1f}s)")

    # ── single-feature models (parallelized) ──────────────────────────────
    print(f"  Fitting {len(feat_indices)} single-feature models (n_jobs={n_jobs})...")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_fit_single_feature)(
            fi, feature_names[fi], context, fold_designs, XGB_CFG, min_n,
        )
        for fi in tqdm(feat_indices, desc=f"features ({bag_name})")
    )

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  ({elapsed / max(len(feat_indices), 1):.1f}s per feature)")

    # ── assemble results ──────────────────────────────────────────────────
    global_rows  = [base_global] + [r[0] for r in results]
    country_rows = base_country  + [row for r in results for row in r[1]]

    global_df  = pd.DataFrame(global_rows)
    country_df = pd.DataFrame(country_rows)

    # Add ΔR² over baseline
    base_r2 = base_global.get("global_oof_r2", np.nan)
    if np.isfinite(base_r2):
        global_df["delta_r2_vs_base"] = global_df["global_oof_r2"] - base_r2
    else:
        global_df["delta_r2_vs_base"] = np.nan

    # Sort by R²
    global_df = global_df.sort_values("global_oof_r2", ascending=False).reset_index(drop=True)

    # ── save ──────────────────────────────────────────────────────────────
    gp = outdir / "single_exposure_global.csv"
    cp = outdir / "single_exposure_country.csv"
    global_df.to_csv(gp, index=False)
    country_df.to_csv(cp, index=False)
    print(f"\n  ✓ Saved:\n    {gp}\n    {cp}")

    # ── summary ───────────────────────────────────────────────────────────
    feats_only = global_df[global_df["feature_name"] != "__baseline__"].copy()
    if not feats_only.empty:
        best = feats_only.iloc[0]  # already sorted desc
        print(f"\n  ── Summary ({bag_name}) ──")
        print(f"  Baseline R²:       {base_r2:.6f}")
        print(f"  Best single feat:  {best['feature_name']}  "
              f"R²={best['global_oof_r2']:.6f}  "
              f"ΔR²={best['delta_r2_vs_base']:.6f}  "
              f"domain={best['domain']}")
        print(f"  Top 5:")
        for _, row in feats_only.head(5).iterrows():
            print(f"    {row['feature_name']:50s}  R²={row['global_oof_r2']:.6f}  "
                  f"ΔR²={row['delta_r2_vs_base']:.6f}  [{row['domain']}]")



# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pin_blas_threads(int(PERF_CFG["blas_threads"]))

    print(f"\n{'='*70}")
    print(f"Single-exposure evaluation")
    print(f"  BAG_TARGET_MODE={BAG_TARGET_MODE}  SMOKE_TEST={SMOKE_TEST}")
    print(f"  n_jobs={PERF_CFG['n_jobs']}")
    print(f"{'='*70}\n")

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = load_artifacts_greedy_only(PATHS, ANALYSIS_CFG)
    model_df = build_model_df(raw_df, expo_df, ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()
    print(f"  model_df: {model_df.shape}  |  n_features={len(feature_names)}")

    # Determine which BAGs to run
    if BAG_TARGET_MODE == "all":
        bags = target_map("combined")
        bags.update(target_map("functional"))
        bags.update(target_map("structural"))
    else:
        bags = target_map(BAG_TARGET_MODE)

    _runner = run_bag_ols if OLS_MODE else run_bag
    for bag_name, y_col in bags.items():
        _runner(bag_name, y_col, model_df, feature_names, OUTDIR_BASE, SMOKE_TEST)

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
