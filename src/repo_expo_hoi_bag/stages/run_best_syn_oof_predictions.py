#!/usr/bin/env python3
"""
run_best_syn_oof_predictions.py
================================
Re-run the best synergistic (o_min) or redundant (o_max) model per BAG
through the exact same XGB d2 LOCO pipeline and save subject-level
out-of-fold predictions.

Saves both y_pred_full (covariates + exposome) and y_pred_base (covariates
only) so that per-subject exposure contribution can be computed downstream
as:  exposure_contribution_i = y_pred_full_i − y_pred_base_i

Usage
-----
    python run_best_syn_oof_predictions.py              # all 3 BAGs, best syn
    BEST_OBJECTIVE_MODE=o_max python run_best_syn_oof_predictions.py
    BAG_TARGET_MODE=combined python run_best_syn_oof_predictions.py
    SMOKE_TEST=1 python run_best_syn_oof_predictions.py  # 3 folds only
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scripts.pipeline_utils import (
    build_analysis_cfg,
    default_paths,
    default_xgb_cfg,
    env_list,
    load_stage_config,
)
from repo_expo_hoi_bag.analysis.selection import xgb_depth_for_rung

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
def _syn_oinfo_negative() -> bool:
    """SYN_OINFO_NEGATIVE: require evaluated O-info<0 for o_min to count as syn."""
    return os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


BAG_TARGET_MODE = os.environ.get("BAG_TARGET_MODE", "all").strip().lower()
SMOKE_TEST      = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
BEST_OBJECTIVE_MODE = os.environ.get("BEST_OBJECTIVE_MODE", "o_min").strip().lower()
RUNG_ID_OVERRIDE = os.environ.get("V3_RUNG_ID", "").strip()
STAGE_CFG = load_stage_config("oof_predictions", smoke=SMOKE_TEST)

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", "outputs/variant_a/subject_level_oof"),
))
# When set, best synergistic model is found dynamically from v3 canonical metrics
# instead of the hardcoded BEST_SYN_MODELS dict below.
V3_CANONICAL_ROOT = os.environ.get("V3_CANONICAL_ROOT", "").strip()
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", STAGE_CFG.get("greedy_root", "outputs/greedy")))
LOCAL_STATS_DIR = OUTDIR_BASE.parent / "stats"

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

# ── model spec lookup ────────────────────────────────────────────────────────

def _objective_suffix(objective: str) -> str:
    if objective == "o_min":
        return "syn"
    if objective == "o_max":
        return "red"
    raise ValueError(f"Unsupported objective: {objective!r}")


def _objective_label(objective: str) -> str:
    if objective == "o_min":
        return "best synergistic"
    if objective == "o_max":
        return "best redundant"
    raise ValueError(f"Unsupported objective: {objective!r}")


def _best_rung_from_selection(bag_name: str, objective: str) -> str | None:
    p = LOCAL_STATS_DIR / "best_rung_selection.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[
        (df["analysis"] == "A_top_k_per_rung")
        & (df["bag"] == bag_name)
        & (df["objective"] == objective)
    ]
    if sub.empty:
        return None
    return str(sub["best_rung"].iloc[0])


def _get_best_model_spec(bag_name: str, objective: str) -> dict:
    """Return best model spec for the requested objective from canonical metrics.

    Reads dynamically from the v3 canonical metrics parquet produced by the
    main LOCO pipeline (step 'main_pooled').  V3_CANONICAL_ROOT must be set.
    """
    if not V3_CANONICAL_ROOT:
        raise EnvironmentError(
            "V3_CANONICAL_ROOT is not set.\n"
            "This variable must point to the canonical/per_experiment output directory "
            "produced by the main LOCO pipeline (step 'main_pooled').\n"
            "Example: export V3_CANONICAL_ROOT=outputs/variant_a/families/"
            "pooled_oinfo_ladder/canonical/per_experiment"
        )
    path = (
        Path(V3_CANONICAL_ROOT)
        / f"pooled_oinfo_ladder_{bag_name}"
        / "metrics_global_long.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical metrics not found: {path}\n"
            "Run the main LOCO pipeline (step 'main_pooled') first."
        )
    g = pd.read_parquet(path)

    # Honour the set-size cap (PAPER_FIG_ORDER_MAX) so the best-model pick never
    # selects a candidate above the chosen interaction order. Unset -> full pool.
    #
    # CRITICAL: when a cap is in force the rung itself must be (re)chosen *under the
    # cap*, jointly with the candidate. The pre-computed best_rung_selection.csv (and
    # the median-R² fallback) rank rungs on the *uncapped* 3-30 pool, so locking the
    # rung first and only then applying the cap can freeze the pick onto a low-rung,
    # low-order candidate and make capped variants (e.g. cap21 vs cap30) collapse to
    # the same model even when the true capped optimum lives in a different rung.
    # So: cap first, then pick (rung, candidate) jointly by full_r2 — matching the
    # order_cap analysis. Only fall back to the precomputed rung when uncapped.
    _order_cap = os.environ.get("PAPER_FIG_ORDER_MAX", "").strip()
    pool = g[g["objective"] == objective].copy()
    # NOTE: "synergistic" (o_min) is defined by the *discovery path* — the candidate
    # was selected by the greedy under the o_min objective — NOT by the sign of the
    # evaluated O-information. A candidate found via the synergistic route can still
    # evaluate to thoi_o > 0; it remains the synergistic-route model. So we do NOT
    # filter o_min on thoi_o < 0 (that previously dropped the genuine best o_min model,
    # e.g. functional order-26 full_r2=0.479, leaving order-5 0.468). This also aligns
    # the OOF pick with stage_fig4_dedup.best_rung, which never applied that filter.
    pool = pool[pool["rung_id"].isin({"xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"})]
    # Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): an o_min
    # candidate counts as synergistic only if its evaluated O-info (thoi_o) < 0.
    # Off by default (path-only criterion); never filters o_max. See the parallel
    # gate in stage_fig4_dedup.best_rung so best_rung_selection.csv stays in sync.
    if _syn_oinfo_negative() and objective == "o_min":
        if "thoi_o" not in pool.columns:
            raise KeyError(
                "SYN_OINFO_NEGATIVE is set but 'thoi_o' is absent from "
                f"{path}; cannot apply the synergy sign filter."
            )
        pool = pool[pd.to_numeric(pool["thoi_o"], errors="coerce") < 0].copy()
    if _order_cap and "order" in pool.columns:
        pool = pool[pd.to_numeric(pool["order"], errors="coerce") <= int(_order_cap)]

    if _order_cap:
        # cap in force: choose rung + candidate jointly on the capped pool
        if pool.empty:
            raise ValueError(
                f"No models for bag={bag_name!r}, objective={objective!r} "
                f"within order<={_order_cap} found in {path}"
            )
        best = pool.nlargest(1, "full_r2").iloc[0]
        best_rung_id = str(best["rung_id"])
        sub = pool
    else:
        # uncapped: preserve the canonical precomputed-rung behaviour
        best_rung_id = RUNG_ID_OVERRIDE or _best_rung_from_selection(bag_name, objective)
        if best_rung_id is None:
            best_rung_id = max(
                ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"),
                key=lambda r: pool[pool["rung_id"] == r]["full_r2"].max()
                if not pool[pool["rung_id"] == r].empty else -999.0,
            )
        if best_rung_id not in {"xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"}:
            raise ValueError(f"OOF XGBoost rerun requires an XGBoost rung; got {best_rung_id!r}")
        sub = pool[pool["rung_id"] == best_rung_id]
        if sub.empty:
            raise ValueError(f"No models for bag={bag_name!r}, objective={objective!r} found in {path}")
        best = sub.nlargest(1, "full_r2").iloc[0]
    predictors = [p for p in str(best["predictors_identity"]).split("|") if p.strip()]
    print(
        f"  [v3] {_objective_label(objective)} model for {bag_name}: "
        f"{best['candidate_id']}  order={len(predictors)}  R²={best['full_r2']:.4f}"
    )
    return {
        "candidate_id": str(best["candidate_id"]),
        "predictors": predictors,
        "expected_r2": float(best["full_r2"]),
        "best_rung": str(best_rung_id),
        "objective": objective,
    }


# ── core: run one BAG ────────────────────────────────────────────────────────

def run_bag(
    bag_name: str,
    y_col: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    outdir_base: Path,
    smoke: bool,
) -> None:
    spec = _get_best_model_spec(bag_name, BEST_OBJECTIVE_MODE)
    xgb_cfg = {**XGB_CFG, "max_depth": xgb_depth_for_rung(spec["best_rung"])}
    pred_names = spec["predictors"]
    pred_idx = [feature_names.index(n) for n in pred_names]
    suffix = _objective_suffix(BEST_OBJECTIVE_MODE)

    outdir = outdir_base / bag_name
    if smoke:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(
        f"BAG: {bag_name}  |  model: {spec['candidate_id']}  |  "
        f"objective={BEST_OBJECTIVE_MODE}  |  order={len(pred_names)}"
    )
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

    print(f"  {len(countries)} LOCO folds, {len(context['y'])} subjects")

    # ── Build fold designs (SAME seed strategy as main pipeline) ──────────
    print("  Building fold matrices...")
    base_seed = int(xgb_cfg["random_state"])
    fold_designs = {}
    for i, c in enumerate(tqdm(countries, desc="fold mats", leave=False)):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG,
            seed=base_seed + i,
        )

    xgb = require_xgboost()
    y   = np.asarray(context["y"], dtype=float)
    X_exp = np.asarray(context["X_exp"], dtype=np.float32)
    min_n = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    # ── Subject-level arrays ──────────────────────────────────────────────
    n_total = len(y)
    y_pred_full_arr = np.full(n_total, np.nan, dtype=float)
    y_pred_base_arr = np.full(n_total, np.nan, dtype=float)

    t0 = time.time()
    for fold_i, country in enumerate(tqdm(countries, desc=f"folds ({bag_name})")):
        fd        = fold_designs[country]
        tr_inner  = fd["tr_inner_idx"]
        val_idx   = fd["val_idx"]
        train_idx = fd["train_idx"]
        test_idx  = fd["test_idx"]

        # ── Zero-variance filter (same as main pipeline) ─────────────────
        pred_used = list(pred_idx)
        if pred_used:
            xtr  = X_exp[np.ix_(train_idx, pred_used)]
            keep = np.nanvar(xtr, axis=0) > 0
            pred_used = [j for j, k in zip(pred_used, keep) if k]

        # ── FULL model (covariates + exposome) ───────────────────────────
        X_tr_inner_full = _append_predictors(fd["Xb_train_inner"], X_exp, tr_inner, pred_used)
        X_val_full      = _append_predictors(fd["Xb_val"],         X_exp, val_idx,  pred_used)
        X_test_full     = _append_predictors(fd["Xb_test"],        X_exp, test_idx, pred_used)

        xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}
        try:
            reg_full = _fit_xgb_fold(
                xgb, xgb_params,
                X_tr_inner_full, y[tr_inner],
                X_val_full, y[val_idx],
            )
            yp_full = reg_full.predict(X_test_full)
        except Exception:
            yp_full = np.full(len(test_idx), np.nan)

        # ── BASELINE model (covariates only) ─────────────────────────────
        try:
            reg_base = _fit_xgb_fold(
                xgb, xgb_params,
                fd["Xb_train_inner"], y[tr_inner],
                fd["Xb_val"], y[val_idx],
            )
            yp_base = reg_base.predict(fd["Xb_test"])
        except Exception:
            yp_base = np.full(len(test_idx), np.nan)

        y_pred_full_arr[test_idx] = yp_full
        y_pred_base_arr[test_idx] = yp_base

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Build output DataFrame ────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "row_id":        context["row_id"],
        "N_MEGA":        context["N_MEGA"],
        "country":       context["country"],
        "diagnosis":     context["diag"],
        "age":           context["age"],
        "sex":           context["sex"],
        "y_true":        y,
        "y_pred_full":   y_pred_full_arr,
        "y_pred_base":   y_pred_base_arr,
    })
    oof_df["exposure_contribution"] = oof_df["y_pred_full"] - oof_df["y_pred_base"]
    oof_df["bag_target"] = bag_name
    oof_df["candidate_id"] = spec["candidate_id"]
    oof_df["best_rung"] = spec["best_rung"]
    oof_df["objective"] = spec["objective"]

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = outdir / f"oof_predictions_best_{suffix}.parquet"
    oof_df.to_parquet(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    shape: {oof_df.shape}")

    # ── Verify R² matches canonical ──────────────────────────────────────
    ok = np.isfinite(oof_df["y_true"]) & np.isfinite(oof_df["y_pred_full"])
    met_full = regression_metrics_extended(
        oof_df.loc[ok, "y_true"].values,
        oof_df.loc[ok, "y_pred_full"].values,
        min_n=min_n,
    )
    ok_b = np.isfinite(oof_df["y_true"]) & np.isfinite(oof_df["y_pred_base"])
    met_base = regression_metrics_extended(
        oof_df.loc[ok_b, "y_true"].values,
        oof_df.loc[ok_b, "y_pred_base"].values,
        min_n=min_n,
    )

    full_r2 = met_full["r2"]
    base_r2 = met_base["r2"]
    exp_r2  = spec["expected_r2"]

    print(f"\n  ── Verification ({bag_name}) ──")
    print(f"  Full model R²:     {full_r2:.6f}  (expected {exp_r2:.6f})")
    print(f"  Baseline R²:       {base_r2:.6f}")
    print(f"  ΔR²:               {full_r2 - base_r2:.6f}")

    # Only warn on large divergence (>1%); small gaps are expected between the
    # canonical training-set full_r2 and this re-run OOF R².
    if abs(full_r2 - exp_r2) > 0.01 and not V3_CANONICAL_ROOT:
        print(f"  ⚠ WARNING: R² mismatch for {bag_name}! Check BEST_SYN_MODELS dict.")

    # ── Quick exposure contribution summary ──────────────────────────────
    ec = oof_df["exposure_contribution"]
    print(f"\n  Exposure contribution stats:")
    print(f"    mean={ec.mean():.4f}  std={ec.std():.4f}  "
          f"min={ec.min():.4f}  max={ec.max():.4f}")
    print(f"    subjects with positive contribution: "
          f"{(ec > 0).sum()} / {len(ec)} ({100*(ec > 0).mean():.1f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pin_blas_threads(1)

    print(f"\n{'='*70}")
    print(f"Subject-level OOF predictions — {_objective_label(BEST_OBJECTIVE_MODE)} model")
    print(
        f"  BAG_TARGET_MODE={BAG_TARGET_MODE}  SMOKE_TEST={SMOKE_TEST}  "
        f"BEST_OBJECTIVE_MODE={BEST_OBJECTIVE_MODE}"
    )
    print(f"{'='*70}\n")

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = load_artifacts_greedy_only(PATHS, ANALYSIS_CFG)
    model_df = build_model_df(raw_df, expo_df, ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()
    print(f"  model_df: {model_df.shape}  |  n_features={len(feature_names)}")

    if BAG_TARGET_MODE == "all":
        bags = target_map("combined")
        bags.update(target_map("functional"))
        bags.update(target_map("structural"))
    else:
        bags = target_map(BAG_TARGET_MODE)

    for bag_name, y_col in bags.items():
        run_bag(bag_name, y_col, model_df, feature_names, OUTDIR_BASE, SMOKE_TEST)

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
