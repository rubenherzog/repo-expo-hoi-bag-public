#!/usr/bin/env python3
"""
run_null_model.py
=================
Null model evaluation: random variable selections through the same XGB d2 pipeline.

Scientific purpose
------------------
Establishes the empirical null distribution for global OOF R².  A synergistic
or redundant variable set is only scientifically meaningful if it significantly
exceeds what a random variable set of the same size achieves.

Design
------
For each null draw k = 1 … N_NULL:
  1. Sample an "order" from the empirical order distribution of the top-40
     real models (top-20 syn + top-20 red from pooled xgb_tree_d2).
  2. Draw `order` feature indices uniformly at random (without replacement)
     from the 63-feature exposome pool.
  3. Run the same LOCO-country XGB d2 pipeline (identical hyperparameters,
     identical fold structure, ALL diagnoses pooled in training).
  4. Record the OOF R² across all LOCO folds → global_oof_r2.

Output: one row per null model in <OUTDIR>/null_global.parquet, plus
        per-country R² in <OUTDIR>/null_country.parquet.

Usage
-----
    BAG_TARGET_MODE=combined python run_null_model.py
    SMOKE_TEST=1 BAG_TARGET_MODE=combined N_NULL=5 python run_null_model.py
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
    RUNG_DEPTH,
)
from scripts.exposome_domains import load_domain_map, unweighted_domain_stats

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
STAGE_CFG = load_stage_config("null_model", smoke=SMOKE_TEST)
N_NULL = int(os.environ.get("N_NULL", str(STAGE_CFG.get("n_null", 10000))))

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

OUTDIR_BASE = Path(os.environ.get(
    "V3_OUTDIR_BASE",
    STAGE_CFG.get("outdir_base", "outputs/variant_a/null_model"),
))
EXPERIMENTS_ROOT = Path(os.environ.get(
    "V3_EXPERIMENTS_ROOT",
    STAGE_CFG.get("experiments_root", "outputs/variant_a/experiments"),
))
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", STAGE_CFG.get("greedy_root", "outputs/greedy")))

PATHS = default_paths(STAGE_CFG)

ANALYSIS_CFG = build_analysis_cfg(STAGE_CFG)

CV_CFG = default_cv_cfg()

XGB_CFG = {**default_xgb_cfg(), **STAGE_CFG.get("xgb_cfg", {})}

EARLY_STOP_CFG = {"val_country_frac": 0.20, "val_country_min": 1}

PERF_CFG = default_perf_cfg(STAGE_CFG)

# ── canonical domain map ──────────────────────────────────────────────────────
# Used to compute per-draw domain diversity so null draws can be filtered
# post-hoc (e.g. exclude draws dominated by a single domain). This is the same
# explicit input used by domain-greedy.
DOMAIN_LABELS_CSV = Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv",
))
DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)


def _domain_diversity_stats(
    feat_idx: list[int],
    feature_names: list[str],
) -> dict:
    """
    Given a list of feature indices, return domain diversity statistics:
      n_domains       : number of distinct domains represented
      max_domain_frac : fraction of features belonging to the dominant domain
      dominant_domain : name of the dominant domain
    """
    selected_features = [feature_names[i] for i in feat_idx]
    stats = unweighted_domain_stats(selected_features, domain_map=DOMAIN_MAP)
    return {
        "n_domains":        stats["n_domains"],
        "max_domain_frac":  round(float(stats["max_domain_frac"]), 4),
        "dominant_domain":  stats["dominant_domain"] or "Other",
    }


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
        # synergistic = greedy discovery objective (o_min), not the sign of evaluated
        # O-info (a candidate found via o_min can evaluate to score>0 and still IS the
        # syn-route model). No score<0 filter.
        syn = _filter_syn(df[df["objective"] == "o_min"])
        r2 = syn["global_oof_r2"].max() if not syn.empty else -999.0
        if r2 > best_r2:
            best_r2, best_rung = r2, rung
    print(f"  Primary rung for {bag}: {best_rung} (best syn R²={best_r2:.4f})")
    return best_rung


def _get_real_order_distribution(bag: str) -> list[int]:
    """Return orders of top-20-syn + top-20-red real models (empirical distribution)."""
    p = (
        EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}"
        / _best_primary_rung(bag) / "xgb_summary_with_calibration_complexity.csv"
    )
    df = pd.read_csv(p)
    df["is_syn"] = df["objective"] == "o_min"
    df["global_oof_r2"] = pd.to_numeric(df["global_oof_r2"], errors="coerce")
    syn = _filter_syn(df[df["is_syn"]]).nlargest(TOP_K_SYN, "global_oof_r2")
    red = df[df["objective"] == "o_max"].nlargest(TOP_K_RED, "global_oof_r2")
    orders = sorted(pd.concat([syn, red])["order"].dropna().astype(int).tolist())
    print(f"  Empirical order distribution (n={len(orders)}): {orders}")
    return orders


def _sample_null_draws(
    real_orders: list[int],
    n_features: int,
    n_null: int,
    rng: np.random.Generator,
    feature_names: list[str],
) -> list[dict]:
    """
    For each null draw, sample an order from real_orders (with replacement)
    and draw that many feature indices uniformly at random.
    Diversity stats (n_domains, max_domain_frac, dominant_domain) are computed
    at draw time and stored so post-hoc domain filtering requires no re-run.
    """
    draws = []
    for k in range(n_null):
        order    = int(rng.choice(real_orders))
        order    = min(order, n_features)
        feat_idx = sorted(rng.choice(n_features, size=order, replace=False).tolist())
        div      = _domain_diversity_stats(feat_idx, feature_names)
        draws.append({
            "null_id":            f"null_{k:05d}",
            "order":               order,
            "predictor_idx_list": feat_idx,
            "n_domains":           div["n_domains"],
            "max_domain_frac":     div["max_domain_frac"],
            "dominant_domain":     div["dominant_domain"],
        })
    return draws


def _fit_null_model(
    null_row: dict,
    context: dict,
    fold_designs: dict,
    xgb_cfg: dict,
    min_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit one null model (random variable set) across all LOCO folds.
    Single XGB fit per fold; global R² computed from concatenated fold predictions.
    Returns (global_rows_df, country_rows_df).
    """
    xgb       = require_xgboost()
    null_id   = str(null_row["null_id"])
    pred_all  = list(null_row["predictor_idx_list"])
    y         = np.asarray(context["y"],     dtype=float)
    X_exp     = np.asarray(context["X_exp"], dtype=np.float32)
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

        # Drop zero-variance predictors on training data
        pred_used = pred_all
        if pred_all:
            xtr  = X_exp[np.ix_(train_idx, pred_all)]
            keep = np.nanvar(xtr, axis=0) > 0
            pred_used = [i for i, k in zip(pred_all, keep) if k]

        X_tr_inner = _append_predictors(fd["Xb_train_inner"], X_exp, tr_inner,  pred_used)
        X_val_mat  = _append_predictors(fd["Xb_val"],         X_exp, val_idx,   pred_used)
        X_test_mat = _append_predictors(fd["Xb_test"],        X_exp, test_idx,  pred_used)

        xgb_params = {**xgb_cfg, "random_state": int(base_seed + fold_i)}
        try:
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
                "null_id":      null_id,
                "order":        null_row["order"],
                "fold_country": country,
                "n_test":       n,
                "r2":           met["r2"],
                "rmse":         met["rmse"],
                "mae":          met["mae"],
                "corr2":        met["corr2"],
            })

    # Global R² from all concatenated held-out predictions
    global_rows: list[dict] = []
    yt   = np.concatenate(y_true_all)
    yp   = np.concatenate(y_pred_all)
    ok_g = np.isfinite(yt) & np.isfinite(yp)
    if ok_g.sum() >= min_n:
        met_g = regression_metrics_extended(yt[ok_g], yp[ok_g], min_n=min_n)
        global_rows.append({
            "null_id":          null_id,
            "order":            null_row["order"],
            "n_domains":        null_row["n_domains"],
            "max_domain_frac":  null_row["max_domain_frac"],
            "dominant_domain":  null_row["dominant_domain"],
            "global_oof_r2":    met_g["r2"],
            "global_rmse":      met_g["rmse"],
            "global_mae":       met_g["mae"],
            "global_corr2":     met_g["corr2"],
            "n_total":          int(ok_g.sum()),
        })

    return pd.DataFrame(global_rows), pd.DataFrame(country_rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if BAG_TARGET_MODE not in {"structural", "functional", "combined"}:
        raise ValueError(f"BAG_TARGET_MODE must be structural/functional/combined, got {BAG_TARGET_MODE!r}")

    pin_blas_threads(int(PERF_CFG["blas_threads"]))

    n_null = 5 if SMOKE_TEST else N_NULL

    primary_rung = _best_primary_rung(BAG_TARGET_MODE)
    xgb_cfg = {**XGB_CFG, "max_depth": RUNG_DEPTH.get(primary_rung, XGB_CFG["max_depth"])}

    print(f"\n{'='*70}")
    print(f"Null model eval | bag={BAG_TARGET_MODE} | smoke={SMOKE_TEST}")
    print(f"  n_null={n_null}  n_jobs={PERF_CFG['n_jobs']}")
    print(f"  Random variable sets, {primary_rung} hyperparameters (max_depth={xgb_cfg['max_depth']}), pooled train")
    print(f"{'='*70}\n")

    outdir = OUTDIR_BASE / BAG_TARGET_MODE
    if SMOKE_TEST:
        outdir = outdir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = load_artifacts_greedy_only(PATHS, ANALYSIS_CFG)
    model_df = build_model_df(raw_df, expo_df, ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()
    n_features = len(feature_names)
    print(f"  model_df: {model_df.shape}  |  n_features={n_features}")
    print("  " + model_df["Diagnosis"].value_counts().to_string().replace("\n", "\n  "))

    print("\nDeriving empirical order distribution from real top-40 models...")
    real_orders = _get_real_order_distribution(BAG_TARGET_MODE)

    rng = np.random.default_rng(int(STAGE_CFG.get("rng_seed", 20260403)))
    null_draws = _sample_null_draws(real_orders, n_features, n_null, rng, feature_names)
    print(f"\n  Generated {len(null_draws)} null draws")
    orders_sampled = [d["order"] for d in null_draws]
    print(f"  Order range: {min(orders_sampled)}–{max(orders_sampled)}, "
          f"median={int(np.median(orders_sampled))}")

    bag_targets     = target_map(BAG_TARGET_MODE)
    bag_name, y_col = next(iter(bag_targets.items()))

    print(f"\nBuilding LOCO context (bag={bag_name}, all dx)...")
    context  = prepare_bag_context(model_df, y_col, bag_name, ANALYSIS_CFG, CV_CFG, feature_names)
    countries = context["countries"]

    if SMOKE_TEST:
        countries = countries[:3]
        context["countries"]               = countries
        context["train_idx_by_country"]    = {c: context["train_idx_by_country"][c] for c in countries}
        context["test_idx_by_country"]     = {c: context["test_idx_by_country"][c]  for c in countries}

    print(f"  {len(countries)} LOCO folds")

    print("Building fold matrices...")
    fold_designs = {}
    for c in tqdm(countries, desc="fold mats"):
        fold_designs[c] = _build_base_fold_mats(
            context, c, ANALYSIS_CFG, EARLY_STOP_CFG,
            seed=int(xgb_cfg["random_state"]),
        )

    n_jobs = int(PERF_CFG["n_jobs"])
    min_n  = int(ANALYSIS_CFG["min_n_obs_for_metrics"])

    print(f"\nRunning {len(null_draws)} null models × {len(countries)} folds (n_jobs={n_jobs})...")
    if n_jobs <= 1:
        results = [
            _fit_null_model(r, context, fold_designs, xgb_cfg, min_n)
            for r in tqdm(null_draws, desc="null models")
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_null_model)(r, context, fold_designs, xgb_cfg, min_n)
            for r in tqdm(null_draws, desc="null models")
        )

    global_frames = [r[0] for r in results if not r[0].empty]
    country_frames = [r[1] for r in results if not r[1].empty]

    if global_frames:
        global_df = pd.concat(global_frames, ignore_index=True)
    else:
        global_df = pd.DataFrame(
            columns=[
                "null_id", "order", "n_domains", "max_domain_frac",
                "dominant_domain", "global_oof_r2", "global_rmse",
                "global_mae", "global_corr2", "n_total",
            ]
        )
        print("WARNING: null_model produced no global rows; saving empty global output.")

    if country_frames:
        country_df = pd.concat(country_frames, ignore_index=True)
    else:
        country_df = pd.DataFrame(
            columns=[
                "null_id", "order", "fold_country", "n_test",
                "r2", "rmse", "mae", "corr2",
            ]
        )
        print("WARNING: null_model produced no country rows; saving empty country output.")

    # ── save ─────────────────────────────────────────────────────────────────
    gp  = outdir / "null_global.parquet"
    cp  = outdir / "null_country.parquet"
    global_df.to_parquet(gp,  index=False)
    country_df.to_parquet(cp, index=False)
    print(f"\n✓ Saved:\n  {gp}\n  {cp}")

    # ── summary ───────────────────────────────────────────────────────────────
    if not global_df.empty:
        _p_best = (
            EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{BAG_TARGET_MODE}"
            / primary_rung / "xgb_summary_with_calibration_complexity.csv"
        )
        _df_best = pd.read_csv(_p_best)
        _df_best["global_oof_r2"] = pd.to_numeric(_df_best["global_oof_r2"], errors="coerce")
        # synergistic = o_min discovery objective (no score<0 filter; see _best_primary_rung)
        _syn_best = _filter_syn(_df_best[_df_best["objective"] == "o_min"])
        real_best = float(_syn_best["global_oof_r2"].max()) if not _syn_best.empty else float("nan")
        r2        = global_df["global_oof_r2"]
        print("\n── Null global R² distribution (all draws) ──")
        print(f"  n={len(global_df)}  median={r2.median():.4f}"
              f"  p95={r2.quantile(0.95):.4f}  max={r2.max():.4f}")
        print(f"  Real model ceiling (syn): {real_best:.4f}")
        print(f"  Empirical p(null ≥ real best) = {(r2 >= real_best).mean():.4f}")

        # ── diversity-filtered summary ────────────────────────────────────
        div_thresh = float(STAGE_CFG.get("diversity_max_domain_frac", 0.60))
        div_mask = global_df["max_domain_frac"] <= div_thresh
        r2_div   = global_df.loc[div_mask, "global_oof_r2"]
        print(f"\n── Diversity-filtered (max_domain_frac ≤ {div_thresh:.2f}): n={div_mask.sum()} "
              f"({div_mask.mean()*100:.1f}%) ──")
        if len(r2_div):
            print(f"  median={r2_div.median():.4f}"
                  f"  p95={r2_div.quantile(0.95):.4f}  max={r2_div.max():.4f}")
            print(f"  Empirical p(null_div ≥ real best) = {(r2_div >= real_best).mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
