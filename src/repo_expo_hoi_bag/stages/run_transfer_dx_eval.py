#!/usr/bin/env python3
"""
run_transfer_dx_eval.py
=======================
CN-train / transfer-test LOCO evaluation.

Design
------
For each LOCO fold (held-out country C):
  - Train  : XGB d2 on CN-only subjects from all countries ≠ C
  - Test   : subjects from country C stratified by Diagnosis
             {CN, AD, MCI, FTD} — whichever have ≥ MIN_N_TEST_DX subjects

Models evaluated : top-TOP_K_SYN synergistic (o_min discovery objective) +
                   top-TOP_K_RED redundant  (o_max)
                   ranked by global_oof_r2 from the pooled xgb_tree_d2 run.

Usage (env-vars)
----------------
    BAG_TARGET_MODE=combined python run_transfer_dx_eval.py

Smoke test (3 countries, 4 models):
    SMOKE_TEST=1 BAG_TARGET_MODE=combined python run_transfer_dx_eval.py
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
    _split_train_val_by_country,
    require_xgboost,
)

# ── config ────────────────────────────────────────────────────────────────────
BAG_TARGET_MODE = os.environ.get("BAG_TARGET_MODE", "combined").strip().lower()
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
STAGE_CFG = load_stage_config("transfer", smoke=SMOKE_TEST)

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
MIN_TRAIN_CN_SUBJECTS = int(STAGE_CFG.get("min_train_cn_subjects", 30))

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", "outputs/variant_a/transfer_dx_eval"),
))
EXPERIMENTS_ROOT = Path(os.environ.get(
    "V3_EXPERIMENTS_ROOT",
    STAGE_CFG.get("experiments_root", "outputs/variant_a/experiments"),
))
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", STAGE_CFG.get("greedy_root", "outputs/greedy")))

PATHS = default_paths(STAGE_CFG)

ANALYSIS_CFG = build_analysis_cfg(STAGE_CFG)
ANALYSIS_CFG["min_n_obs_for_metrics"] = MIN_N_TEST_DX

# LOCO CV: NO diagnosis filtering here — keep ALL diagnoses in context so that
# test folds contain CN + AD + MCI + FTD subjects. Train-CN restriction is
# applied manually inside _fit_one_model_transfer by masking train_idx.
CV_CFG = default_cv_cfg()

XGB_CFG = {**default_xgb_cfg(), **STAGE_CFG.get("xgb_cfg", {})}

EARLY_STOP_CFG = {"val_country_frac": 0.20, "val_country_min": 1}

PERF_CFG = default_perf_cfg(STAGE_CFG)
PERF_CFG.setdefault("chunk_size_models", 8)


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
        # "synergistic" = the greedy discovery objective (o_min), NOT the sign of the
        # evaluated O-info: a candidate found via the o_min route can still evaluate to
        # score > 0 (its sub-tuples may carry the synergy the model exploits). So we do
        # NOT filter on score < 0.
        syn = _filter_syn(df[df["objective"] == "o_min"])
        r2 = syn["global_oof_r2"].max() if not syn.empty else -999.0
        if r2 > best_r2:
            best_r2, best_rung = r2, rung
    print(f"  Primary rung for {bag}: {best_rung} (best syn R²={best_r2:.4f})")
    return best_rung


def _select_top_candidates(bag: str, top_k_syn: int, top_k_red: int) -> pd.DataFrame:
    """Load best-rung pooled summary and pick top-K syn + top-K red."""
    p = (
        EXPERIMENTS_ROOT
        / f"pooled_oinfo_ladder_{bag}"
        / _best_primary_rung(bag)
        / "xgb_summary_with_calibration_complexity.csv"
    )
    if not p.exists():
        raise FileNotFoundError(f"Pooled xgb summary not found: {p}")

    df = pd.read_csv(p)
    # "synergistic" = the greedy discovery objective (o_min), not the sign of the
    # evaluated O-info (see _best_primary_rung): keep all o_min candidates.
    df["is_syn"] = df["objective"].astype(str) == "o_min"
    df["global_oof_r2"] = pd.to_numeric(df["global_oof_r2"], errors="coerce")

    syn = (
        _filter_syn(df[df["is_syn"]])
        .sort_values("global_oof_r2", ascending=False)
        .head(top_k_syn)
        .copy()
    )
    red = (
        df[df["objective"].astype(str) == "o_max"]
        .sort_values("global_oof_r2", ascending=False)
        .head(top_k_red)
        .copy()
    )
    syn["transfer_group"] = "synergistic"
    red["transfer_group"] = "redundant"

    out = pd.concat([syn, red], ignore_index=True)
    print(
        f"  Candidates selected: {len(syn)} synergistic (best R²={syn['global_oof_r2'].max():.4f}), "
        f"{len(red)} redundant (best R²={red['global_oof_r2'].max():.4f})"
    )
    return out


def _build_reps_df(
    candidates: pd.DataFrame,
    exposome_feature_names: list[str],
) -> pd.DataFrame:
    """Convert predictors_identity (pipe-separated names) → integer indices."""
    name_to_idx = {n: i for i, n in enumerate(exposome_feature_names)}
    rows = []
    for _, row in candidates.iterrows():
        identity = str(row.get("predictors_identity", ""))
        var_names = [v.strip() for v in identity.split("|") if v.strip()]
        pred_idx = [name_to_idx[v] for v in var_names if v in name_to_idx]
        if not pred_idx:
            continue
        rows.append(
            {
                "model_id": str(row["model_id"]),
                "objective": str(row["objective"]),
                "transfer_group": str(row.get("transfer_group", "")),
                "pooled_r2": float(row.get("global_oof_r2", np.nan)),
                "thoi_score": float(row.get("score", np.nan)),
                "order": int(row.get("order", len(pred_idx))),
                "predictors_identity": identity,
                "predictors_identity_n": len(pred_idx),
                "predictor_idx_list": pred_idx,
            }
        )
    return pd.DataFrame(rows)


def _build_cov_mat(context: dict, idx: np.ndarray, analysis_cfg: dict) -> np.ndarray:
    """Build base covariate matrix (age, year, sex) for a subset of rows.

    Diagnosis is deliberately excluded: training is CN-only (constant) and test
    subjects include multiple diagnoses whose OHE levels must not leak into the
    feature space.
    """
    age  = np.asarray(context["age"],  dtype=float)[idx].reshape(-1, 1).astype(np.float32)
    year = np.asarray(context["year"], dtype=float)[idx].reshape(-1, 1).astype(np.float32)
    parts = [age]
    if analysis_cfg.get("include_year", True):
        parts.append(year)
    # sex OHE — levels from the training set are pre-baked in fold_designs;
    # we skip it here to keep the helper simple and avoid level mismatch across
    # CN-train / all-dx test.  Sex is a minor covariate; omitting it is safe.
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def _fit_one_model_transfer(
    rep_row: dict,
    context: dict,
    fold_designs: dict,
    xgb_cfg: dict,
    min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one XGB model across all LOCO folds with CN-only training and return:
      - global_rows   : one summary row per (model_id, transfer_group, fold)
      - country_dx_rows : R² per (model_id, fold_country, eval_dx)

    Train restriction: train_idx is filtered to CN subjects only.
    Test evaluation:   all diagnoses from the held-out country are scored.
    """
    xgb = require_xgboost()
    rep_id = str(rep_row["model_id"])
    pred_idx_all = list(rep_row["predictor_idx_list"])
    y = np.asarray(context["y"], dtype=float)
    X_exp = np.asarray(context["X_exp"], dtype=np.float32)
    # diag comes from context — aligned with work table, covers all diagnoses
    diag_ctx = np.asarray(context["diag"], dtype=str)
    base_seed = int(xgb_cfg.get("random_state", 20260304))

    global_preds: list[dict] = []
    country_dx_rows: list[dict] = []

    if "diag" not in context:
        raise ValueError("The 'diag' array is missing from the context. Ensure it is populated correctly.")

    for fold_i, country in enumerate(context["countries"]):
        fd = fold_designs[country]
        train_idx_all = fd["train_idx"]   # all non-held-out subjects
        test_idx      = fd["test_idx"]    # held-out country, all diagnoses

        # ── restrict training to CN subjects only ────────────────────────────
        cn_mask_train = diag_ctx[train_idx_all] == "CN"
        train_idx = train_idx_all[cn_mask_train]
        if len(train_idx) < MIN_TRAIN_CN_SUBJECTS:
            continue

        # Rebuild inner/val split from CN-only train pool
        tr_inner, val_idx = _split_train_val_by_country(
            context["country"],
            train_idx,
            val_country_frac=float(EARLY_STOP_CFG.get("val_country_frac", 0.20)),
            val_country_min=int(EARLY_STOP_CFG.get("val_country_min", 1)),
            seed=int(base_seed + fold_i),
        )

        # Drop zero-variance predictors on CN training set
        pred_used_idx = pred_idx_all
        if pred_idx_all:
            xtr_full = X_exp[np.ix_(train_idx, pred_idx_all)]
            keep = np.nanvar(xtr_full, axis=0) > 0
            pred_used_idx = [i for i, k in zip(pred_idx_all, keep) if k]

        y_tr_inner = y[tr_inner]
        y_val      = y[val_idx]

        # Build base covariate matrices (age, year — no diagnosis, see _build_cov_mat)
        Xb_tr_inner = _build_cov_mat(context, tr_inner, ANALYSIS_CFG)
        Xb_val      = _build_cov_mat(context, val_idx,   ANALYSIS_CFG)
        Xb_tr_full  = _build_cov_mat(context, train_idx, ANALYSIS_CFG)
        Xb_test     = _build_cov_mat(context, test_idx,  ANALYSIS_CFG)

        X_tr_inner = _append_predictors(Xb_tr_inner, X_exp, tr_inner,   pred_used_idx)
        X_val_mat  = _append_predictors(Xb_val,      X_exp, val_idx,    pred_used_idx)
        X_tr_full  = _append_predictors(Xb_tr_full,  X_exp, train_idx,  pred_used_idx)
        X_test_mat = _append_predictors(Xb_test,     X_exp, test_idx,   pred_used_idx)

        xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}

        try:
            reg = _fit_xgb_fold(xgb, xgb_params, X_tr_inner, y_tr_inner, X_val_mat, y_val)
            y_pred = reg.predict(X_test_mat)
        except Exception:
            y_pred = np.full(len(test_idx), np.nan)

        # ── per-diagnosis R² in this test fold ──────────────────────────────
        y_true_test = y[test_idx]
        test_diag   = diag_ctx[test_idx]   # correct: from context, not model_df

        for dx in ["CN", "AD", "MCI", "FTD"]:
            mask_dx = test_diag == dx
            yt = y_true_test[mask_dx]
            yp = y_pred[mask_dx]
            valid = np.isfinite(yt) & np.isfinite(yp)
            n = int(valid.sum())
            if n < min_n:
                continue
            met = regression_metrics_extended(yt[valid], yp[valid], min_n=min_n)
            country_dx_rows.append(
                {
                    "model_id": rep_id,
                    "transfer_group": rep_row["transfer_group"],
                    "objective": rep_row["objective"],
                    "pooled_r2": rep_row["pooled_r2"],
                    "thoi_score": rep_row["thoi_score"],
                    "order": rep_row["order"],
                    "fold_country": country,
                    "eval_dx": dx,
                    "n_test": n,
                    "r2": met["r2"],
                    "rmse": met["rmse"],
                    "mae": met["mae"],
                    "corr2": met["corr2"],
                }
            )

        # ── overall (all-dx) R² for this fold ───────────────────────────────
        valid_all = np.isfinite(y_true_test) & np.isfinite(y_pred)
        n_all = int(valid_all.sum())
        if n_all >= min_n:
            met_all = regression_metrics_extended(
                y_true_test[valid_all], y_pred[valid_all], min_n=min_n
            )
            global_preds.append(
                {
                    "model_id": rep_id,
                    "transfer_group": rep_row["transfer_group"],
                    "objective": rep_row["objective"],
                    "pooled_r2": rep_row["pooled_r2"],
                    "thoi_score": rep_row["thoi_score"],
                    "fold_country": country,
                    "eval_dx": "ALL",
                    "n_test": n_all,
                    "r2": met_all["r2"],
                }
            )

    return pd.DataFrame(global_preds), pd.DataFrame(country_dx_rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if BAG_TARGET_MODE not in {"structural", "functional", "combined"}:
        raise ValueError(f"BAG_TARGET_MODE must be structural/functional/combined, got {BAG_TARGET_MODE!r}")

    pin_blas_threads(int(PERF_CFG["blas_threads"]))

    top_k_syn = 2 if SMOKE_TEST else TOP_K_SYN
    top_k_red = 2 if SMOKE_TEST else TOP_K_RED

    print(f"\n{'='*70}")
    print(f"Transfer-dx eval | bag={BAG_TARGET_MODE} | smoke={SMOKE_TEST}")
    print(f"  top_k_syn={top_k_syn}  top_k_red={top_k_red}  n_jobs={PERF_CFG['n_jobs']}")
    print(f"{'='*70}\n")

    outdir = OUTDIR_BASE / BAG_TARGET_MODE
    if SMOKE_TEST:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load artifacts ────────────────────────────────────────────────────────
    print("Loading artifacts...")
    raw_df, exposome_df, _candidate_df, _candidate_all_df = load_artifacts_greedy_only(
        PATHS, ANALYSIS_CFG
    )
    model_df = build_model_df(raw_df, exposome_df, ANALYSIS_CFG)
    exposome_feature_names = exposome_df.columns.tolist()

    # ── select top-K candidates from pooled run ───────────────────────────────
    print("Selecting top candidates from pooled xgb_tree_d2 summary...")
    candidates = _select_top_candidates(BAG_TARGET_MODE, top_k_syn, top_k_red)
    reps_df = _build_reps_df(candidates, exposome_feature_names)
    print(f"  {len(reps_df)} models to evaluate (after index resolution)")

    if reps_df.empty:
        raise RuntimeError("No models resolved from predictors_identity. Check feature name alignment.")

    # ── build bag context (CN-train, all-test) ────────────────────────────────
    bag_targets = target_map(BAG_TARGET_MODE)
    bag_name, y_col = next(iter(bag_targets.items()))

    print(f"Building LOCO context (train=CN, test=ALL, bag={bag_name})...")
    context = prepare_bag_context(
        model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, exposome_feature_names
    )

    countries = context["countries"]
    if SMOKE_TEST:
        countries = countries[:3]
        context["countries"] = countries
        # Restrict train/test idx dicts to smoke countries
        context["train_idx_by_country"] = {c: context["train_idx_by_country"][c] for c in countries}
        context["test_idx_by_country"] = {c: context["test_idx_by_country"][c] for c in countries}

    print(f"  LOCO folds: {len(countries)} countries")

    # ── build fold designs ────────────────────────────────────────────────────
    print("Building fold feature matrices...")
    fold_designs = {}
    for c in tqdm(countries, desc="fold mats"):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG, seed=int(XGB_CFG["random_state"])
        )

    # ── evaluate models in parallel ───────────────────────────────────────────
    records = reps_df.to_dict("records")
    n_jobs = int(PERF_CFG["n_jobs"])
    min_n = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    print(f"Running {len(records)} models × {len(countries)} folds ...")
    if n_jobs <= 1:
        results = [
            _fit_one_model_transfer(r, context, fold_designs, XGB_CFG, min_n)
            for r in tqdm(records, desc="models")
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_one_model_transfer)(r, context, fold_designs, XGB_CFG, min_n)
            for r in tqdm(records, desc="models")
        )

    global_dfs = [r[0] for r in results if not r[0].empty]
    country_dx_dfs = [r[1] for r in results if not r[1].empty]

    global_df = pd.concat(global_dfs, ignore_index=True) if global_dfs else pd.DataFrame()
    country_dx_df = pd.concat(country_dx_dfs, ignore_index=True) if country_dx_dfs else pd.DataFrame()

    # ── aggregate global oof R² per model ────────────────────────────────────
    if not global_df.empty:
        agg = (
            global_df[global_df["eval_dx"] == "ALL"]
            .groupby(["model_id", "transfer_group", "objective", "pooled_r2", "thoi_score"])
            .apply(
                lambda g: pd.Series(
                    {
                        "transfer_r2_mean": float(
                            np.nanmean(pd.to_numeric(g["r2"], errors="coerce"))
                        ),
                        "n_folds": len(g),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        print("\n── Global transfer R² summary (mean across folds) ──")
        print(
            agg.sort_values("transfer_r2_mean", ascending=False)
            .groupby("transfer_group")[["transfer_r2_mean", "pooled_r2"]]
            .describe()
            .round(4)
        )

    # ── save ──────────────────────────────────────────────────────────────────
    gp = outdir / "transfer_global.parquet"
    cdp = outdir / "transfer_country_dx.parquet"
    global_df.to_parquet(gp, index=False)
    country_dx_df.to_parquet(cdp, index=False)
    print(f"\n✓ Saved:\n  {gp}\n  {cdp}")

    # Quick per-dx summary
    if not country_dx_df.empty:
        print("\n── Per-dx transfer R² (median across countries & models, top-syn vs top-red) ──")
        summary = (
            country_dx_df.groupby(["transfer_group", "eval_dx"])["r2"]
            .agg(["median", "count"])
            .round(4)
        )
        print(summary)


if __name__ == "__main__":
    main()
