#!/usr/bin/env python3
"""
run_shap_oof.py
===============
Compute out-of-fold SHAP values (and SHAP interaction matrices) for the best
synergistic XGBoost d2 model per BAG under LOCO-country cross-validation.

For each fold the model is trained identically to run_best_syn_oof_predictions.py
(same data split, same seed, same zero-variance filter).  After fitting, XGBoost's
native tree-SHAP is used to decompose each test-set prediction into per-feature
contributions.  This is done for BOTH the full model (covariates + exposome) and
the baseline model (covariates only).

No external `shap` library is required — XGBoost's built-in
  Booster.predict(pred_contribs=True)
  Booster.predict(pred_interactions=True)
provide exact tree-SHAP values.

Outputs (per BAG, in {OUTDIR_BASE}/{bag}/):
  shap_values_oof.parquet       — one row per subject, all SHAP columns
  shap_interaction_oof.npz      — compressed 3-D interaction arrays
  feature_names.json            — union feature name lists + metadata
  shap_summary.parquet          — mean |SHAP| and mean SHAP per feature/scope

Usage
-----
    conda run -n tvb python run_shap_oof.py --bags combined
    conda run -n tvb python run_shap_oof.py --bags structural functional combined
    SMOKE_TEST=1 conda run -n tvb python run_shap_oof.py --bags structural
    V3_CANONICAL_ROOT=outputs/variant_a python run_shap_oof.py --bags all
"""
from __future__ import annotations

import argparse
import json
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
    _append_predictors,
    _as_clean_str,
    _build_base_fold_mats,
    _fit_xgb_fold,
    require_xgboost,
)

# ── config ────────────────────────────────────────────────────────────────────
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
STAGE_CFG = load_stage_config("shap_oof", smoke=SMOKE_TEST)

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", ""),
))
if not OUTDIR_BASE:
    base_root = Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
    OUTDIR_BASE = base_root / "shap_oof"

# When set, best synergistic model is found dynamically from v3 canonical metrics.
V3_CANONICAL_ROOT = os.environ.get("V3_CANONICAL_ROOT", "").strip()
RUNG_ID_OVERRIDE = os.environ.get("V3_RUNG_ID", "").strip()
SHAP_OBJECTIVE = os.environ.get("SHAP_OBJECTIVE", "o_min").strip()

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

# ── model spec lookup ─────────────────────────────────────────────────────────

def _get_best_model_spec(bag_name: str) -> dict:
    """Return best synergistic (o_min discovery objective) model spec from canonical metrics.

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
    root = Path(V3_CANONICAL_ROOT)
    candidates = [
        root / f"pooled_oinfo_ladder_{bag_name}" / "metrics_global_long.parquet",
        root / "metrics_global_long.parquet",
        root / "canonical" / "metrics_global_long.parquet",
        root / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Canonical metrics not found in any expected location. Checked:\n"
            + "\n".join(str(p) for p in candidates)
            + "\nRun the main LOCO pipeline (step 'main_pooled') first."
        )
    g = pd.read_parquet(path)
    _rung_map = {"xgb_tree_d1": "XGB d1", "xgb_tree_d2": "XGB d2", "xgb_tree_d3": "XGB d3"}
    _best_rung_id = RUNG_ID_OVERRIDE or max(
        _rung_map,
        key=lambda r: g[(g["rung_id"] == r) & (g["objective"] == SHAP_OBJECTIVE)]["full_r2"].max()
        if not g[(g["rung_id"] == r) & (g["objective"] == SHAP_OBJECTIVE)].empty else -999.0,
    )
    _best_rung_label = _rung_map[_best_rung_id]
    if _best_rung_id not in {"xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"}:
        raise ValueError(f"SHAP rerun requires an XGBoost rung; got {_best_rung_id!r}")
    sub = g[
        (g["rung_id"] == _best_rung_id)
        & (g["objective"] == SHAP_OBJECTIVE)
    ]
    if sub.empty:
        raise ValueError(
            f"No {SHAP_OBJECTIVE} {_best_rung_label} models found in {path}"
        )
    best = sub.nlargest(1, "full_r2").iloc[0]
    predictors = [p for p in str(best["predictors_identity"]).split("|") if p.strip()]
    print(
        f"  [v3] Best synergistic model for {bag_name}: "
        f"{best['candidate_id']}  order={len(predictors)}  R²={best['full_r2']:.4f}"
    )
    return {
        "candidate_id": str(best["candidate_id"]),
        "predictors": predictors,
        "expected_r2": float(best["full_r2"]),
        "best_rung": _best_rung_id,
    }


# ── feature name utilities ────────────────────────────────────────────────────

def _fold_feature_names(
    analysis_cfg: dict,
    sex_levels: list[str],
    diag_levels: list[str],
    pred_names_used: list[str],
) -> list[str]:
    """Ordered feature names matching columns of the XGBoost input matrix.

    Does NOT include the bias column — the caller appends '__bias__' when
    slicing SHAP output.  Length must equal X_test_full.shape[1].
    """
    names: list[str] = ["age"]
    if analysis_cfg.get("include_year", True):
        names.append("year")
    if analysis_cfg.get("include_sex", True):
        names.extend(f"sex_{lv}" for lv in sex_levels)
    if analysis_cfg.get("include_diagnosis", True):
        names.extend(f"diag_{lv}" for lv in diag_levels)
    names.extend(pred_names_used)
    return names


# ── SHAP computation ──────────────────────────────────────────────────────────

def _compute_shap_fold(
    booster,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values and interaction values for one fold test set.

    Uses XGBoost's native tree-SHAP — no external `shap` library needed.

    Parameters
    ----------
    booster : xgb.Booster
        Fitted booster (obtained via reg.get_booster()).
    X : np.ndarray, shape (n_test, n_features)
        Feature matrix for the held-out test fold.

    Returns
    -------
    shap_vals : np.ndarray, shape (n_test, n_features + 1)
        SHAP values.  Last column is the global bias (intercept) term.
    interact_vals : np.ndarray, shape (n_test, n_features + 1, n_features + 1)
        SHAP interaction values.  Last row/column is the bias.
        Diagonal entries are the "main" SHAP contributions (self-interaction);
        off-diagonal entries are pairwise interactions (symmetric: [i,j]==[j,i]).
    """
    import xgboost as xgb  # already available in tvb env

    dmat = xgb.DMatrix(X)
    shap_vals = booster.predict(dmat, pred_contribs=True)       # (n, n_feat+1)
    interact_vals = booster.predict(dmat, pred_interactions=True)  # (n, n_feat+1, n_feat+1)
    return shap_vals, interact_vals


# ── fold result alignment ─────────────────────────────────────────────────────

def _align_folds_to_union(
    fold_results: list[dict],
    n_total: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align per-fold SHAP arrays to a common union feature set.

    Parameters
    ----------
    fold_results : list of dicts with keys:
        test_idx      : np.ndarray of subject indices (into the context arrays)
        feature_names : list[str] including '__bias__' as last entry
        shap_vals     : np.ndarray (n_test, n_fold_features) or None
        interact_vals : np.ndarray (n_test, n_fold_features, n_fold_features) or None
    n_total : int
        Total number of subjects across all folds.

    Returns
    -------
    shap_oof : np.ndarray, shape (n_total, n_union)
        NaN where subject was not in any fold or feature absent from fold.
    interact_oof : np.ndarray, shape (n_total, n_union, n_union)
        NaN same convention.
    union_names : list[str]
        Union of all feature names, first-seen ordering.
    """
    # Build union ordering (first-seen across folds)
    seen: dict[str, int] = {}
    for fr in fold_results:
        if fr["shap_vals"] is None:
            continue
        for name in fr["feature_names"]:
            if name not in seen:
                seen[name] = len(seen)
    union_names = sorted(seen, key=lambda n: seen[n])
    n_union = len(union_names)
    name_to_idx = {n: i for i, n in enumerate(union_names)}

    shap_oof = np.full((n_total, n_union), np.nan, dtype=np.float32)
    interact_oof = np.full((n_total, n_union, n_union), np.nan, dtype=np.float32)

    for fr in fold_results:
        if fr["shap_vals"] is None:
            continue
        tidx = fr["test_idx"]
        fn = fr["feature_names"]
        col_map = [name_to_idx[n] for n in fn]  # local col index → union col index
        sv = fr["shap_vals"]      # (n_test, n_local)
        iv = fr["interact_vals"]  # (n_test, n_local, n_local)

        for local_col, union_col in enumerate(col_map):
            shap_oof[tidx, union_col] = sv[:, local_col]

        for lc_i, uc_i in enumerate(col_map):
            for lc_j, uc_j in enumerate(col_map):
                interact_oof[np.ix_(tidx, [uc_i], [uc_j])] = (
                    iv[:, lc_i, lc_j].reshape(-1, 1, 1)
                )

    return shap_oof, interact_oof, union_names


# ── SHAP summary ──────────────────────────────────────────────────────────────

def _build_shap_summary(
    shap_oof: np.ndarray,
    union_names: list[str],
    model_label: str,
    bag_target: str,
    candidate_id: str,
    context: dict,
) -> pd.DataFrame:
    """Mean |SHAP| and mean signed SHAP per feature, globally and per country.

    Parameters
    ----------
    shap_oof : (n_subjects, n_union) float32 array — NaN where absent.
    union_names : list of feature names (length n_union).
    model_label : "full" or "base".
    context : bag context dict (used for country assignments).
    """
    country_arr = np.asarray(context["country"])
    row_id_arr = np.asarray(context["row_id"])

    rows = []

    def _scope_rows(mask: np.ndarray, scope: str) -> None:
        sub = shap_oof[mask]  # (n_scope, n_union)
        n_subj = int(np.sum(mask))
        for col_i, feat in enumerate(union_names):
            vals = sub[:, col_i]
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                continue
            rows.append({
                "bag_target": bag_target,
                "candidate_id": candidate_id,
                "model": model_label,
                "scope": scope,
                "feature": feat,
                "mean_abs_shap": float(np.mean(np.abs(finite))),
                "mean_shap": float(np.mean(finite)),
                "n_subjects": n_subj,
            })

    # Global
    _scope_rows(np.ones(len(shap_oof), dtype=bool), "global")

    # Per country
    for c in sorted(set(country_arr)):
        mask = country_arr == c
        _scope_rows(mask, str(c))

    return pd.DataFrame(rows)


# ── output saving ─────────────────────────────────────────────────────────────

def _save_outputs(
    bag_name: str,
    outdir: Path,
    context: dict,
    y: np.ndarray,
    y_pred_full_arr: np.ndarray,
    y_pred_base_arr: np.ndarray,
    shap_oof_full: np.ndarray,
    interact_oof_full: np.ndarray,
    union_names_full: list[str],
    shap_oof_base: np.ndarray,
    interact_oof_base: np.ndarray,
    union_names_base: list[str],
    spec: dict,
    fold_country_arr: np.ndarray,
) -> None:
    """Write all four output files for one BAG."""
    candidate_id = spec["candidate_id"]
    min_n = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    # ── 1. shap_values_oof.parquet ────────────────────────────────────────
    meta_df = pd.DataFrame({
        "row_id":       context["row_id"],
        "N_MEGA":       context["N_MEGA"],
        "country":      context["country"],
        "fold_country": fold_country_arr,
        "diagnosis":    context["diag"],
        "age":          context["age"],
        "sex":          context["sex"],
        "y_true":       y,
        "y_pred_full":  y_pred_full_arr,
        "y_pred_base":  y_pred_base_arr,
        "bag_target":   bag_name,
        "candidate_id": candidate_id,
    })

    shap_full_df = pd.DataFrame(
        shap_oof_full,
        columns=[f"shap_full__{n}" for n in union_names_full],
    ).astype(np.float32)

    shap_base_df = pd.DataFrame(
        shap_oof_base,
        columns=[f"shap_base__{n}" for n in union_names_base],
    ).astype(np.float32)

    oof_df = pd.concat([meta_df, shap_full_df, shap_base_df], axis=1)
    out_parquet = outdir / "shap_values_oof.parquet"
    oof_df.to_parquet(out_parquet, index=False)
    print(f"\n  ✓ Saved: {out_parquet}  shape={oof_df.shape}")

    # ── 2. shap_interaction_oof.npz ──────────────────────────────────────
    out_npz = outdir / "shap_interaction_oof.npz"
    np.savez_compressed(
        out_npz,
        interactions_full=interact_oof_full.astype(np.float32),
        interactions_base=interact_oof_base.astype(np.float32),
        feature_names_full=np.array(union_names_full, dtype=object),
        feature_names_base=np.array(union_names_base, dtype=object),
        row_ids=np.asarray(context["row_id"]),
    )
    print(f"  ✓ Saved: {out_npz}"
          f"  full={interact_oof_full.shape}  base={interact_oof_base.shape}")

    # ── 3. feature_names.json ─────────────────────────────────────────────
    json_meta = {
        "bag_target": bag_name,
        "candidate_id": candidate_id,
        "union_feature_names_full": union_names_full,
        "union_feature_names_base": union_names_base,
        "n_features_full_with_bias": len(union_names_full),
        "n_features_base_with_bias": len(union_names_base),
        "exposome_predictor_names": spec["predictors"],
    }
    out_json = outdir / "feature_names.json"
    out_json.write_text(json.dumps(json_meta, indent=2, ensure_ascii=False))
    print(f"  ✓ Saved: {out_json}")

    # ── 4. shap_summary.parquet ──────────────────────────────────────────
    summary_full = _build_shap_summary(
        shap_oof_full, union_names_full, "full", bag_name, candidate_id, context
    )
    summary_base = _build_shap_summary(
        shap_oof_base, union_names_base, "base", bag_name, candidate_id, context
    )
    summary_df = pd.concat([summary_full, summary_base], ignore_index=True)
    out_summary = outdir / "shap_summary.parquet"
    summary_df.to_parquet(out_summary, index=False)
    print(f"  ✓ Saved: {out_summary}  shape={summary_df.shape}")

    # ── Verification ──────────────────────────────────────────────────────
    # R² check (full model)
    ok = np.isfinite(y) & np.isfinite(y_pred_full_arr)
    met = regression_metrics_extended(y[ok], y_pred_full_arr[ok], min_n=min_n)
    full_r2 = met["r2"]
    exp_r2  = spec["expected_r2"]
    print(f"\n  ── Verification ({bag_name}) ──")
    print(f"  Full model OOF R²: {full_r2:.6f}  (expected {exp_r2:.6f})")
    # Small gaps between canonical full_r2 and re-run OOF R² are expected.
    if abs(full_r2 - exp_r2) > 0.01:
        print(f"  ⚠ WARNING: R² mismatch — check seeds / exclusions!")

    # SHAP additivity check (subjects that were in a fold)
    shap_cols_full = [c for c in oof_df.columns if c.startswith("shap_full__")]
    in_fold_mask = oof_df["fold_country"].ne("").values
    if in_fold_mask.sum() > 0:
        shap_arr = oof_df.loc[in_fold_mask, shap_cols_full].values.astype(np.float64)
        shap_sum = np.nansum(shap_arr, axis=1)
        pred_vals = y_pred_full_arr[in_fold_mask]
        max_residual = np.nanmax(np.abs(shap_sum - pred_vals))
        print(f"  SHAP additivity max |residual|: {max_residual:.2e}"
              + ("  ✓" if max_residual < 1e-3 else "  ⚠ CHECK"))


# ── core: run one BAG ─────────────────────────────────────────────────────────

def run_bag(
    bag_name: str,
    y_col: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    outdir_base: Path,
    smoke: bool,
) -> None:
    spec = _get_best_model_spec(bag_name)
    xgb_cfg = {**XGB_CFG, "max_depth": xgb_depth_for_rung(spec["best_rung"])}
    pred_names = spec["predictors"]
    pred_idx = [feature_names.index(n) for n in pred_names]

    outdir = outdir_base / bag_name
    if smoke:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"BAG: {bag_name}  |  model: {spec['candidate_id']}  |  order={len(pred_names)}")
    print(f"{'─'*60}")

    context = prepare_bag_context(
        model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, feature_names,
    )
    countries = context["countries"]

    if smoke:
        countries = countries[:3]
        context["countries"] = countries
        context["train_idx_by_country"] = {
            c: context["train_idx_by_country"][c] for c in countries
        }
        context["test_idx_by_country"] = {
            c: context["test_idx_by_country"][c] for c in countries
        }

    print(f"  {len(countries)} LOCO folds, {len(context['y'])} subjects")

    # ── Build fold designs (identical seed strategy to run_best_syn_oof_predictions.py)
    print("  Building fold matrices...")
    base_seed = int(xgb_cfg["random_state"])
    fold_designs: dict = {}
    for i, c in enumerate(tqdm(countries, desc="fold mats", leave=False)):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG, seed=base_seed + i,
        )

    xgb_module = require_xgboost()
    y       = np.asarray(context["y"], dtype=float)
    X_exp   = np.asarray(context["X_exp"], dtype=np.float32)
    sex_arr = context["sex"]
    diag_arr = context["diag"]
    n_total = len(y)

    # Accumulation arrays
    y_pred_full_arr = np.full(n_total, np.nan, dtype=float)
    y_pred_base_arr = np.full(n_total, np.nan, dtype=float)
    fold_country_arr = np.full(n_total, "", dtype=object)

    fold_results_full: list[dict] = []
    fold_results_base: list[dict] = []

    t0 = time.time()
    for fold_i, country in enumerate(tqdm(countries, desc=f"SHAP folds ({bag_name})")):
        fd        = fold_designs[country]
        tr_inner  = fd["tr_inner_idx"]
        val_idx   = fd["val_idx"]
        train_idx = fd["train_idx"]
        test_idx  = fd["test_idx"]

        fold_country_arr[test_idx] = country

        # ── Zero-variance filter (identical to run_best_syn_oof_predictions.py) ──
        pred_used = list(pred_idx)
        if pred_used:
            xtr  = X_exp[np.ix_(train_idx, pred_used)]
            keep = np.nanvar(xtr, axis=0) > 0
            pred_used = [j for j, k in zip(pred_used, keep) if k]
        pred_names_used = [feature_names[j] for j in pred_used]

        # ── Reconstruct per-fold feature names (must match _build_base_fold_mats logic) ──
        sex_levels = (
            sorted(pd.Series(_as_clean_str(sex_arr[tr_inner])).unique().tolist())
            if ANALYSIS_CFG.get("include_sex", True)
            else []
        )
        diag_levels = (
            sorted(pd.Series(_as_clean_str(diag_arr[tr_inner])).unique().tolist())
            if ANALYSIS_CFG.get("include_diagnosis", True)
            else []
        )

        # ── FULL model matrices ───────────────────────────────────────────────
        X_tr_inner_full = _append_predictors(fd["Xb_train_inner"], X_exp, tr_inner, pred_used)
        X_val_full      = _append_predictors(fd["Xb_val"],         X_exp, val_idx,  pred_used)
        X_test_full     = _append_predictors(fd["Xb_test"],        X_exp, test_idx, pred_used)

        xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}

        # Fit full model
        try:
            reg_full = _fit_xgb_fold(
                xgb_module, xgb_params,
                X_tr_inner_full, y[tr_inner],
                X_val_full, y[val_idx],
            )
        except Exception as e:
            print(f"  WARNING: full model fit failed for {country}: {e}")
            yp_full = np.full(len(test_idx), np.nan)
            fold_results_full.append({
                "test_idx": test_idx, "fold_country": country,
                "feature_names": [], "shap_vals": None, "interact_vals": None,
            })
            fold_results_base.append({
                "test_idx": test_idx, "fold_country": country,
                "feature_names": [], "shap_vals": None, "interact_vals": None,
            })
            y_pred_full_arr[test_idx] = yp_full
            y_pred_base_arr[test_idx] = yp_full  # both NaN
            continue

        # Compute SHAP for full model (use output_margin=True so SHAP sum == y_pred)
        fold_names_full = _fold_feature_names(
            ANALYSIS_CFG, sex_levels, diag_levels, pred_names_used
        )
        assert len(fold_names_full) == X_test_full.shape[1], (
            f"Feature name count mismatch: {len(fold_names_full)} != {X_test_full.shape[1]}"
        )
        fold_names_full_with_bias = fold_names_full + ["__bias__"]

        shap_vals_full, interact_vals_full = _compute_shap_fold(
            reg_full.get_booster(), X_test_full
        )
        # y_pred in margin space so SHAP additivity holds: sum(SHAP) == y_pred
        yp_full = shap_vals_full.sum(axis=1)

        y_pred_full_arr[test_idx] = yp_full

        fold_results_full.append({
            "test_idx":      test_idx,
            "fold_country":  country,
            "feature_names": fold_names_full_with_bias,
            "shap_vals":     shap_vals_full,
            "interact_vals": interact_vals_full,
        })

        # ── BASELINE model (covariates only) ──────────────────────────────────
        try:
            reg_base = _fit_xgb_fold(
                xgb_module, xgb_params,
                fd["Xb_train_inner"], y[tr_inner],
                fd["Xb_val"], y[val_idx],
            )
        except Exception as e:
            print(f"  WARNING: baseline model fit failed for {country}: {e}")
            yp_base = np.full(len(test_idx), np.nan)
            fold_results_base.append({
                "test_idx": test_idx, "fold_country": country,
                "feature_names": [], "shap_vals": None, "interact_vals": None,
            })
            y_pred_base_arr[test_idx] = yp_base
            continue

        # Compute SHAP for baseline model (no exposome predictors)
        fold_names_base = _fold_feature_names(
            ANALYSIS_CFG, sex_levels, diag_levels, pred_names_used=[]
        )
        assert len(fold_names_base) == fd["Xb_test"].shape[1], (
            f"Base feature name count mismatch: {len(fold_names_base)} != {fd['Xb_test'].shape[1]}"
        )
        fold_names_base_with_bias = fold_names_base + ["__bias__"]

        shap_vals_base, interact_vals_base = _compute_shap_fold(
            reg_base.get_booster(), fd["Xb_test"]
        )
        # y_pred in margin space so SHAP additivity holds
        yp_base = shap_vals_base.sum(axis=1)

        y_pred_base_arr[test_idx] = yp_base

        fold_results_base.append({
            "test_idx":      test_idx,
            "fold_country":  country,
            "feature_names": fold_names_base_with_bias,
            "shap_vals":     shap_vals_base,
            "interact_vals": interact_vals_base,
        })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Align fold results to union feature sets ──────────────────────────────
    print("  Aligning SHAP arrays to union feature sets...")
    shap_oof_full, interact_oof_full, union_names_full = _align_folds_to_union(
        fold_results_full, n_total
    )
    shap_oof_base, interact_oof_base, union_names_base = _align_folds_to_union(
        fold_results_base, n_total
    )

    print(f"  Union features — full: {len(union_names_full)}, base: {len(union_names_base)}")

    # ── Save all outputs ──────────────────────────────────────────────────────
    _save_outputs(
        bag_name=bag_name,
        outdir=outdir,
        context=context,
        y=y,
        y_pred_full_arr=y_pred_full_arr,
        y_pred_base_arr=y_pred_base_arr,
        shap_oof_full=shap_oof_full,
        interact_oof_full=interact_oof_full,
        union_names_full=union_names_full,
        shap_oof_base=shap_oof_base,
        interact_oof_base=interact_oof_base,
        union_names_base=union_names_base,
        spec=spec,
        fold_country_arr=fold_country_arr,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOF SHAP values for best synergistic XGB model (LOCO CV)"
    )
    p.add_argument(
        "--bags",
        nargs="+",
        choices=["structural", "functional", "combined"],
        default=["combined", "functional", "structural"],
        metavar="BAG",
        help=(
            "BAG modalities to run. Choices: structural functional combined. "
            "Default: all three."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of threads per XGBoost model (nthread). Default: 1.",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pin_blas_threads(1)

    # Override nthread in XGB_CFG if --n-jobs given
    if args.n_jobs != 1:
        XGB_CFG["nthread"] = args.n_jobs

    print(f"\n{'='*70}")
    print("OOF SHAP values — best synergistic model per BAG")
    print(f"  SMOKE_TEST={SMOKE_TEST}  bags={args.bags}  nthread={XGB_CFG['nthread']}")
    print(f"  OUTDIR_BASE={OUTDIR_BASE}")
    print(f"{'='*70}\n")

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = load_artifacts_greedy_only(PATHS, ANALYSIS_CFG)
    model_df = build_model_df(raw_df, expo_df, ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()
    print(f"  model_df: {model_df.shape}  |  n_features={len(feature_names)}")

    for bag_name in args.bags:
        y_col = target_map(bag_name)[bag_name]
        run_bag(bag_name, y_col, model_df, feature_names, OUTDIR_BASE, SMOKE_TEST)

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
