#!/usr/bin/env python3
"""
run_dx_stratified_eval.py
=========================
Pooled-train / diagnosis-stratified-test LOCO evaluation.

Scientific question
-------------------
Does the exposome-BAG relationship attenuate as a function of disease?
  H: R²(CN) > R²(MCI) > R²(AD) ≈ R²(FTD) ≈ 0
  because in healthy controls brain aging is environmentally governed,
  whereas in disease it is governed by pathological processes.

Design
------
For each LOCO fold (held-out country C):
  - Train  : XGB d2 on ALL subjects (CN+AD+MCI+FTD) from countries ≠ C
             — identical to the main pooled analysis
  - Test   : subjects from country C, R² computed separately per Diagnosis
             {CN, AD, MCI, FTD} — whichever have ≥ MIN_N_TEST_DX subjects

Models evaluated : top-TOP_K_SYN synergistic (o_min discovery objective) +
                   top-TOP_K_RED redundant  (o_max)
                   ranked by global_oof_r2 from the pooled xgb_tree_d2 run.

Usage
-----
    BAG_TARGET_MODE=combined python run_dx_stratified_eval.py
    SMOKE_TEST=1 BAG_TARGET_MODE=combined python run_dx_stratified_eval.py

Output
------
    <OUTDIR_BASE>/<bag>/dx_stratified_country_dx.parquet
    <OUTDIR_BASE>/<bag>/dx_stratified_global.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from scripts.pipeline_utils import (
    build_analysis_cfg,
    default_cv_cfg,
    default_paths,
    default_perf_cfg,
    default_xgb_cfg,
    load_stage_config,
)

from loco_fusion_matrix_engine import (
    build_model_df,
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
BAG_TARGET_MODE = os.environ.get("BAG_TARGET_MODE", "combined").strip().lower()
SMOKE_TEST      = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
STAGE_CFG = load_stage_config("dx_stratified", smoke=SMOKE_TEST)

TOP_K_SYN = int(STAGE_CFG.get("top_k_syn", 20))
TOP_K_RED = int(STAGE_CFG.get("top_k_red", 20))


def _filter_syn(df: pd.DataFrame) -> pd.DataFrame:
    """Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): keep
    only o_min candidates with evaluated O-info (score) < 0. Caller has already
    restricted df to o_min rows. No-op when the flag is off."""
    if os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() not in (
        "1", "true", "yes", "y", "on",
    ):
        return df
    if "score" not in df.columns:
        raise KeyError("SYN_OINFO_NEGATIVE set but 'score' absent from candidate summary.")
    return df[pd.to_numeric(df["score"], errors="coerce") < 0]
MIN_N_TEST_DX = int(STAGE_CFG.get("min_n_test_dx", 5))

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", "outputs/variant_a/dx_stratified_eval"),
))
EXPERIMENTS_ROOT = Path(os.environ.get(
    "V3_EXPERIMENTS_ROOT",
    STAGE_CFG.get("experiments_root", "outputs/variant_a/experiments"),
))
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", STAGE_CFG.get("greedy_root", "outputs/greedy")))

PATHS = default_paths(STAGE_CFG)

# Identical to the main pooled run — diagnosis IS a covariate, all dx in train
ANALYSIS_CFG = build_analysis_cfg(STAGE_CFG)
ANALYSIS_CFG["min_n_obs_for_metrics"] = MIN_N_TEST_DX

# Standard pooled LOCO CV — no diagnosis filtering
CV_CFG = default_cv_cfg()

XGB_CFG = {**default_xgb_cfg(), **STAGE_CFG.get("xgb_cfg", {})}

EARLY_STOP_CFG = {"val_country_frac": 0.20, "val_country_min": 1}

PERF_CFG = default_perf_cfg(STAGE_CFG)


# ── helpers ───────────────────────────────────────────────────────────────────

def _best_primary_rung(bag: str) -> str:
    """Return the rung_id with highest best-syn R² across d1/d2/d3 stage CSVs."""
    best_rung, best_r2 = "xgb_tree_d2", -999.0
    for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
        p = EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / rung / "xgb_summary_with_calibration_complexity.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["global_oof_r2"] = pd.to_numeric(df["global_oof_r2"], errors="coerce")
        # "synergistic" = the greedy discovery objective (o_min), not the sign of the
        # evaluated O-info: a candidate found via o_min can evaluate to score > 0 and
        # still IS the synergistic-route model (its sub-tuples carry the synergy the
        # model exploits). So we do NOT filter on score < 0.
        syn = _filter_syn(df[df["objective"] == "o_min"])
        r2 = syn["global_oof_r2"].max() if not syn.empty else -999.0
        if r2 > best_r2:
            best_r2, best_rung = r2, rung
    print(f"  Primary rung for {bag}: {best_rung} (best syn R²={best_r2:.4f})")
    return best_rung


def _select_top_candidates(bag: str, top_k_syn: int, top_k_red: int) -> pd.DataFrame:
    p = (
        EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}"
        / _best_primary_rung(bag) / "xgb_summary_with_calibration_complexity.csv"
    )
    if not p.exists():
        raise FileNotFoundError(f"Pooled xgb summary not found: {p}")
    df = pd.read_csv(p)
    df["is_syn"] = df["objective"].astype(str) == "o_min"
    df["global_oof_r2"] = pd.to_numeric(df["global_oof_r2"], errors="coerce")
    syn = _filter_syn(df[df["is_syn"]]).nlargest(top_k_syn, "global_oof_r2").copy()
    red = df[df["objective"].astype(str) == "o_max"].nlargest(top_k_red, "global_oof_r2").copy()
    syn["transfer_group"] = "synergistic"
    red["transfer_group"] = "redundant"
    out = pd.concat([syn, red], ignore_index=True)
    print(
        f"  Candidates: {len(syn)} syn (best R²={syn['global_oof_r2'].max():.4f}), "
        f"{len(red)} red (best R²={red['global_oof_r2'].max():.4f})"
    )
    return out


def _build_reps_df(candidates: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    rows = []
    for _, row in candidates.iterrows():
        var_names = [v.strip() for v in str(row.get("predictors_identity", "")).split("|") if v.strip()]
        pred_idx = [name_to_idx[v] for v in var_names if v in name_to_idx]
        if not pred_idx:
            continue
        rows.append({
            "model_id":            str(row["model_id"]),
            "objective":           str(row["objective"]),
            "transfer_group":      str(row.get("transfer_group", "")),
            "pooled_r2":           float(row.get("global_oof_r2", np.nan)),
            "thoi_score":          float(row.get("score", np.nan)),
            "order":               int(row.get("order", len(pred_idx))),
            "predictors_identity": str(row.get("predictors_identity", "")),
            "predictor_idx_list":  pred_idx,
        })
    return pd.DataFrame(rows)


def _fit_one_model(
    rep_row: dict,
    context: dict,
    fold_designs: dict,
    xgb_cfg: dict,
    min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one model with FULL pooled training (all diagnoses), score test fold
    per diagnosis.  This is the correct test of the attenuation hypothesis.
    """
    xgb      = require_xgboost()
    rep_id   = str(rep_row["model_id"])
    pred_all = list(rep_row["predictor_idx_list"])
    y        = np.asarray(context["y"],     dtype=float)
    X_exp    = np.asarray(context["X_exp"], dtype=np.float32)
    diag_ctx = np.asarray(context["diag"],  dtype=str)
    base_seed = int(xgb_cfg.get("random_state", 20260304))

    global_rows:  list[dict] = []
    country_rows: list[dict] = []

    for fold_i, country in enumerate(context["countries"]):
        fd        = fold_designs[country]
        tr_inner  = fd["tr_inner_idx"]
        val_idx   = fd["val_idx"]
        train_idx = fd["train_idx"]
        test_idx  = fd["test_idx"]

        # Drop zero-variance predictors on full training set
        pred_used = pred_all
        if pred_all:
            xtr = X_exp[np.ix_(train_idx, pred_all)]
            keep = np.nanvar(xtr, axis=0) > 0
            pred_used = [i for i, k in zip(pred_all, keep) if k]

        X_tr_inner = _append_predictors(fd["Xb_train_inner"], X_exp, tr_inner,   pred_used)
        X_val_mat  = _append_predictors(fd["Xb_val"],         X_exp, val_idx,    pred_used)
        X_test_mat = _append_predictors(fd["Xb_test"],        X_exp, test_idx,   pred_used)

        xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}

        try:
            reg    = _fit_xgb_fold(xgb, xgb_params, X_tr_inner, y[tr_inner], X_val_mat, y[val_idx])
            y_pred = reg.predict(X_test_mat)
        except Exception:
            y_pred = np.full(len(test_idx), np.nan)

        y_true = y[test_idx]
        test_d = diag_ctx[test_idx]

        # ── per-diagnosis R² ─────────────────────────────────────────────────
        for dx in ["CN", "AD", "MCI", "FTD"]:
            mask  = test_d == dx
            yt, yp = y_true[mask], y_pred[mask]
            ok    = np.isfinite(yt) & np.isfinite(yp)
            n     = int(ok.sum())
            if n < min_n:
                continue
            met = regression_metrics_extended(yt[ok], yp[ok], min_n=min_n)
            country_rows.append({
                "model_id":       rep_id,
                "transfer_group": rep_row["transfer_group"],
                "objective":      rep_row["objective"],
                "pooled_r2":      rep_row["pooled_r2"],
                "thoi_score":     rep_row["thoi_score"],
                "order":          rep_row["order"],
                "fold_country":   country,
                "eval_dx":        dx,
                "n_test":         n,
                "r2":             met["r2"],
                "rmse":           met["rmse"],
                "mae":            met["mae"],
                "corr2":          met["corr2"],
            })

        # ── all-dx R² (sanity check: should ≈ pooled country R²) ─────────────
        ok_all = np.isfinite(y_true) & np.isfinite(y_pred)
        n_all  = int(ok_all.sum())
        if n_all >= min_n:
            met_all = regression_metrics_extended(y_true[ok_all], y_pred[ok_all], min_n=min_n)
            global_rows.append({
                "model_id":       rep_id,
                "transfer_group": rep_row["transfer_group"],
                "objective":      rep_row["objective"],
                "pooled_r2":      rep_row["pooled_r2"],
                "thoi_score":     rep_row["thoi_score"],
                "fold_country":   country,
                "eval_dx":        "ALL",
                "n_test":         n_all,
                "r2":             met_all["r2"],
            })

    return pd.DataFrame(global_rows), pd.DataFrame(country_rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if BAG_TARGET_MODE not in {"structural", "functional", "combined"}:
        raise ValueError(f"BAG_TARGET_MODE must be structural/functional/combined, got {BAG_TARGET_MODE!r}")

    pin_blas_threads(int(PERF_CFG["blas_threads"]))

    top_k_syn = 2 if SMOKE_TEST else TOP_K_SYN
    top_k_red = 2 if SMOKE_TEST else TOP_K_RED

    print(f"\n{'='*70}")
    print(f"DX-stratified eval | bag={BAG_TARGET_MODE} | smoke={SMOKE_TEST}")
    print(f"  top_k_syn={top_k_syn}  top_k_red={top_k_red}  n_jobs={PERF_CFG['n_jobs']}")
    print(f"  Train: ALL diagnoses (pooled)   Test: per-diagnosis stratified")
    print(f"{'='*70}\n")

    outdir = OUTDIR_BASE / BAG_TARGET_MODE
    if SMOKE_TEST:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = load_artifacts_greedy_only(PATHS, ANALYSIS_CFG)
    model_df = build_model_df(raw_df, expo_df, ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()

    print(f"  model_df: {model_df.shape}  |  dx counts:")
    print("  " + model_df["Diagnosis"].value_counts().to_string().replace("\n", "\n  "))

    print("\nSelecting top candidates...")
    candidates = _select_top_candidates(BAG_TARGET_MODE, top_k_syn, top_k_red)
    reps_df    = _build_reps_df(candidates, feature_names)
    print(f"  {len(reps_df)} models resolved")

    if reps_df.empty:
        raise RuntimeError("No models resolved — check predictors_identity alignment.")

    bag_targets          = target_map(BAG_TARGET_MODE)
    bag_name, y_col      = next(iter(bag_targets.items()))

    print(f"\nBuilding LOCO context (bag={bag_name}, all dx)...")
    context  = prepare_bag_context(model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, feature_names)
    countries = context["countries"]

    if SMOKE_TEST:
        countries = countries[:3]
        context["countries"] = countries
        context["train_idx_by_country"] = {c: context["train_idx_by_country"][c] for c in countries}
        context["test_idx_by_country"]  = {c: context["test_idx_by_country"][c]  for c in countries}

    print(f"  {len(countries)} LOCO folds")

    print("Building fold matrices...")
    fold_designs = {}
    for c in tqdm(countries, desc="fold mats"):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG,
            seed=int(XGB_CFG["random_state"]),
        )

    records = reps_df.to_dict("records")
    n_jobs  = int(PERF_CFG["n_jobs"])
    min_n   = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    print(f"\nRunning {len(records)} models × {len(countries)} folds (n_jobs={n_jobs})...")
    if n_jobs <= 1:
        results = [
            _fit_one_model(r, context, fold_designs, XGB_CFG, min_n)
            for r in tqdm(records, desc="models")
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_one_model)(r, context, fold_designs, XGB_CFG, min_n)
            for r in tqdm(records, desc="models")
        )

    global_frames = [r[0] for r in results if not r[0].empty]
    country_frames = [r[1] for r in results if not r[1].empty]

    if global_frames:
        global_df = pd.concat(global_frames, ignore_index=True)
    else:
        global_df = pd.DataFrame(
            columns=[
                'model_id', 'transfer_group', 'objective', 'pooled_r2',
                'thoi_score', 'fold_country', 'eval_dx', 'n_test',
                'r2', 'rmse', 'mae', 'corr2',
            ]
        )
        print("WARNING: dx_stratified produced no global rows; saving empty global output.")

    if country_frames:
        country_dx_df = pd.concat(country_frames, ignore_index=True)
    else:
        country_dx_df = pd.DataFrame(
            columns=[
                'model_id', 'transfer_group', 'objective', 'pooled_r2',
                'thoi_score', 'order', 'fold_country', 'eval_dx', 'n_test',
                'r2', 'rmse', 'mae', 'corr2',
            ]
        )
        print("WARNING: dx_stratified produced no country-dx rows; saving empty country output.")

    print(f"  dx_stratified rows: global={len(global_df)} country={len(country_dx_df)}")

    # ── save ─────────────────────────────────────────────────────────────────
    gp  = outdir / "dx_stratified_global.parquet"
    cdp = outdir / "dx_stratified_country_dx.parquet"
    global_df.to_parquet(gp,   index=False)
    country_dx_df.to_parquet(cdp, index=False)
    print(f"\n✓ Saved:\n  {gp}\n  {cdp}")

    # ── summary table ────────────────────────────────────────────────────────
    if not country_dx_df.empty:
        print("\n── Median R² by group × diagnosis (attenuation gradient) ──")
        tbl = (
            country_dx_df
            .groupby(["transfer_group", "eval_dx"])["r2"]
            .agg(median="median", count="count")
            .round(4)
        )
        # reorder diagnoses so gradient is visible
        tbl = tbl.reindex(
            pd.MultiIndex.from_product(
                [["synergistic", "redundant"], ["CN", "MCI", "AD", "FTD"]],
                names=["transfer_group", "eval_dx"],
            ),
            fill_value=np.nan,
        )
        print(tbl.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
