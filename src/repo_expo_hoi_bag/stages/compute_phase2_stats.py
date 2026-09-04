#!/usr/bin/env python3
"""
compute_phase2_stats.py
========================
Phase 2 — Stats & Thin Wrappers.  Computes all derived tables from existing
pipeline outputs + newly generated Phase 1 outputs (single-exposure eval,
subject-level OOF predictions).

Outputs (all saved to STATS_DIR):
  2.1  diagnosis_odds_ratios.csv
  2.2  aggregation_hierarchy.csv
  2.3  dx_stratified_summary.csv
  2.4  transfer_summary.csv
  2.6  country_syn_vs_red.csv
  2.7  enrichment_summary.csv
  2.8  effect_sizes.csv

Usage:
    python compute_phase2_stats.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from scripts.pipeline_utils import load_stage_config

warnings.filterwarnings("ignore", category=FutureWarning)

_STAGE_CFG  = load_stage_config("compute_phase2_stats")
_COMMON_CFG = load_stage_config("common")

# ── analysis parameters (all values defined in config/pipeline.yaml) ──────────
_AGE_BINS       = _STAGE_CFG.get("age_bins", [0, 50, 60, 70, 80, 200])
_AGE_LABELS     = _STAGE_CFG.get("age_bin_labels", ["<50", "50-60", "60-70", "70-80", "80+"])
_CONTRAST_DX    = _STAGE_CFG.get("contrast_diagnoses", ["AD", "FTD", "MCI"])
_MIN_N_TEST     = int(_STAGE_CFG.get("min_n_test", 20))
_GRAD_AMPL      = float(_STAGE_CFG.get("gradient_amplification_threshold", 1.1))
_GRAD_PERS      = float(_STAGE_CFG.get("gradient_persistence_threshold", 0.8))
_ENRICH_THRESH  = _STAGE_CFG.get("enrichment_thresholds", [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.50])
_BOOT_SEED      = int(_COMMON_CFG.get("bootstrap_rng_seed", 42))
_N_BOOT         = int(_COMMON_CFG.get("n_bootstrap", 2000))

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
STATS_DIR = ROOT / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

def _canonical_root(root: Path) -> Path:
    v3_root = root / "families" / "pooled_oinfo_ladder" / "canonical"
    return v3_root if v3_root.exists() else root / "canonical"

CANONICAL = _canonical_root(ROOT)
BAGS = ["combined", "functional", "structural"]


def _order_from_candidate_id(candidate_id: str) -> float:
    if not isinstance(candidate_id, str):
        return np.nan
    match = re.search(r"ord(\d+)", candidate_id)
    return int(match.group(1)) if match else np.nan


def _best_rung_df(mg: pd.DataFrame, objective: str = "o_min") -> pd.DataFrame:
    """Return rows of mg filtered to the best rung for the given objective.

    Best rung = the rung_id whose max full_r2 for `objective` is highest.
    Falls back to xgb_tree_d2 if no tree rung is found.
    """
    tree_rungs = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
    best = max(
        tree_rungs,
        key=lambda r: mg[(mg["rung_id"] == r) & (mg["objective"] == objective)]["full_r2"].max()
        if not mg[(mg["rung_id"] == r) & (mg["objective"] == objective)].empty else -999.0,
    )
    return mg[mg["rung_id"] == best]


# ── reference values (computed dynamically from canonical metrics) ─────────────
# These are derived at runtime from metrics_global_long.parquet so they remain
# correct after any rerun with different data, exclusions, or model seeds.

def _load_ref_r2(bag: str) -> dict[str, float]:
    """Return base_r2, best_syn_r2, best_red_r2 for one BAG from canonical metrics."""
    p = CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not p.exists():
        return {"base_r2": float("nan"), "best_syn_r2": float("nan"), "best_red_r2": float("nan")}
    mg = pd.read_parquet(p)
    # Use best rung (highest best-syn full_r2 across d1/d2/d3)
    rung_map = {"xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"}
    best_rung = max(
        rung_map,
        key=lambda r: mg[(mg["rung_id"] == r) & (mg["objective"] == "o_min")]["full_r2"].max()
        if not mg[(mg["rung_id"] == r) & (mg["objective"] == "o_min")].empty else -999.0,
    )
    d = mg[mg["rung_id"] == best_rung]
    base_r2    = d["base_r2"].max() if "base_r2" in d.columns else float("nan")
    best_syn_r2 = d[d["objective"] == "o_min"]["full_r2"].max()
    best_red_r2 = d[d["objective"] == "o_max"]["full_r2"].max()
    return {"base_r2": float(base_r2), "best_syn_r2": float(best_syn_r2), "best_red_r2": float(best_red_r2)}

_REF_CACHE: dict[str, dict[str, float]] = {}

def _ref(bag: str, key: str) -> float:
    if bag not in _REF_CACHE:
        _REF_CACHE[bag] = _load_ref_r2(bag)
    return _REF_CACHE[bag][key]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  Diagnosis odds-ratio analysis
# ═══════════════════════════════════════════════════════════════════════════════
def compute_diagnosis_odds_ratios() -> pd.DataFrame:
    """
    Logistic regression: accelerated aging (BAG > 0) ~ diagnosis + age + sex.

    """
    print("\n── 2.1 Diagnosis odds-ratio analysis ──")

    rows = []
    for bag in BAGS:
        p = ROOT / "subject_level_oof" / bag / "oof_predictions_best_syn.parquet"
        if not p.exists():
            print(f"  ⚠ Missing OOF for {bag}, skipping")
            continue
        df = pd.read_parquet(p)

        # Binary outcome
        df["accelerated"] = (df["y_true"] > 0).astype(int)

        # Age group (decade bins)
        df["age_group"] = pd.cut(df["age"], bins=_AGE_BINS, labels=_AGE_LABELS)

        # Encode for logistic regression
        import statsmodels.api as sm
        from statsmodels.formula.api import logit as sm_logit

        df_model = df[["accelerated", "age_group", "sex", "diagnosis"]].dropna().copy()
        df_model["age_group"] = df_model["age_group"].astype(str)
        df_model["sex"] = df_model["sex"].astype(str)
        df_model["diagnosis"] = df_model["diagnosis"].astype(str)

        n_accel = df_model["accelerated"].sum()
        n_total = len(df_model)
        print(f"\n  {bag}: n={n_total}, accelerated={n_accel} ({100*n_accel/n_total:.1f}%)")

        # ── Diagnosis contrasts ──────────────────────────────────────────
        try:
            if "CN" not in df_model["diagnosis"].unique():
                print(f"    ⚠ Diagnosis model skipped: CN not present in {bag} OOF data (excluded in this variant)")
                raise ValueError("CN not in data")
            fit_dx = sm_logit(
                "accelerated ~ C(diagnosis, Treatment(reference='CN')) + C(age_group) + C(sex)",
                data=df_model,
            ).fit(disp=0, maxiter=100)

            params_d = fit_dx.params
            conf_d   = fit_dx.conf_int()
            pvals_d  = fit_dx.pvalues

            for dx in _CONTRAST_DX:
                keys = [k for k in params_d.index if dx in k]
                if keys:
                    k = keys[0]
                    or_dx = np.exp(params_d[k])
                    ci_lo = np.exp(conf_d.loc[k, 0])
                    ci_hi = np.exp(conf_d.loc[k, 1])
                    p_dx  = pvals_d[k]
                    print(f"    Diagnosis {dx} vs CN: OR={or_dx:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] p={p_dx:.2e}")
                    rows.append({"bag": bag, "model": "diagnosis", "contrast": f"{dx}_vs_CN",
                                 "OR": or_dx, "CI_lo": ci_lo, "CI_hi": ci_hi, "p": p_dx,
                                 "n": n_total, "n_accelerated": n_accel})
        except Exception as e:
            print(f"    ⚠ Diagnosis model failed: {e}")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "diagnosis_odds_ratios.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2  Aggregation hierarchy table
# ═══════════════════════════════════════════════════════════════════════════════
def compute_aggregation_hierarchy() -> pd.DataFrame:
    print("\n── 2.2 Aggregation hierarchy table ──")
    rows = []

    for bag in BAGS:
        se_path = ROOT / "single_exposure_eval" / bag / "single_exposure_global.csv"
        mg_path = CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        if not se_path.exists() or not mg_path.exists():
            print(f"  ⚠ {bag}: missing single_exposure or canonical metrics — skipping")
            continue

        base_r2 = _ref(bag, "base_r2")

        # Best single exposure (from Phase 1.1)
        se = pd.read_csv(se_path)
        feats = se[se["feature_name"] != "__baseline__"]
        best_single = feats.iloc[0]  # sorted desc by R²
        single_r2 = best_single["global_oof_r2"]
        single_name = best_single["feature_name"]
        single_domain = best_single["domain"]
        delta_single = single_r2 - base_r2

        # Best order-3, best redundant, best synergistic — each at its own best rung
        mg = pd.read_parquet(CANONICAL / "per_experiment" /
                             f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet")

        # Best order-3: use best rung for o_min (synergistic context)
        best_rung_mg = _best_rung_df(mg, objective="o_min")
        o3 = best_rung_mg[best_rung_mg["order"] == 3]
        best_o3_r2 = o3["full_r2"].max()
        delta_o3 = best_o3_r2 - base_r2

        # Best redundant (o_max) — at best rung for o_max
        red_mg = _best_rung_df(mg, objective="o_max")
        red = red_mg[red_mg["objective"] == "o_max"]
        best_red_r2 = red["full_r2"].max()
        delta_red = best_red_r2 - base_r2

        # Best synergistic (o_min) — at best rung for o_min
        syn = best_rung_mg[best_rung_mg["objective"] == "o_min"]
        best_syn_r2 = syn["full_r2"].max()
        delta_syn = best_syn_r2 - base_r2

        # Best order of syn/red
        best_syn_row = syn.sort_values("full_r2", ascending=False).iloc[0]
        best_red_row = red.sort_values("full_r2", ascending=False).iloc[0]

        levels = [
            {"bag": bag, "level": "baseline", "description": "age + sex + dx + year",
             "r2": base_r2, "delta_r2": 0.0, "fold_gain_vs_single": np.nan,
             "order": 0, "best_feature_or_model": "covariates only"},
            {"bag": bag, "level": "best_single_exposure", "description": f"1 feature + covariates",
             "r2": single_r2, "delta_r2": delta_single, "fold_gain_vs_single": 1.0,
             "order": 1, "best_feature_or_model": f"{single_name} [{single_domain}]"},
            {"bag": bag, "level": "best_order3", "description": "best 3-var set (any objective)",
             "r2": best_o3_r2, "delta_r2": delta_o3,
             "fold_gain_vs_single": delta_o3 / delta_single if delta_single > 0 else np.nan,
             "order": 3, "best_feature_or_model": ""},
            {"bag": bag, "level": "best_redundant", "description": "best o_max model",
             "r2": best_red_r2, "delta_r2": delta_red,
             "fold_gain_vs_single": delta_red / delta_single if delta_single > 0 else np.nan,
             "order": int(best_red_row["order"]) if pd.notna(best_red_row["order"]) else _order_from_candidate_id(best_red_row["candidate_id"]),
             "best_feature_or_model": best_red_row["candidate_id"]},
            {"bag": bag, "level": "best_synergistic", "description": "best o_min model",
             "r2": best_syn_r2, "delta_r2": delta_syn,
             "fold_gain_vs_single": delta_syn / delta_single if delta_single > 0 else np.nan,
             "order": int(best_syn_row["order"]) if pd.notna(best_syn_row["order"]) else _order_from_candidate_id(best_syn_row["candidate_id"]),
             "best_feature_or_model": best_syn_row["candidate_id"]},
        ]
        rows.extend(levels)

        print(f"\n  {bag}:")
        for lv in levels:
            fg = f"{lv['fold_gain_vs_single']:.2f}×" if np.isfinite(lv['fold_gain_vs_single']) else "—"
            print(f"    {lv['level']:25s}  R²={lv['r2']:.6f}  ΔR²={lv['delta_r2']:+.6f}  fold-gain={fg}")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "aggregation_hierarchy.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3  Dx-stratified summary rebuild
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dx_stratified_summary() -> pd.DataFrame:
    print("\n── 2.3 Dx-stratified summary rebuild ──")
    rows = []

    for bag in BAGS:
        p = ROOT / "dx_stratified_eval" / bag / "dx_stratified_country_dx.parquet"
        if not p.exists():
            print(f"  ⚠ Missing {p}")
            continue
        df = pd.read_parquet(p)

        # Filter: n_test >= 20
        df = df[df["n_test"] >= _MIN_N_TEST].copy()

        for obj in ["o_min", "o_max"]:
            sub = df[df["objective"] == obj] if "objective" in df.columns else df
            if sub.empty:
                continue

            for dx in ["CN", "MCI", "AD", "FTD"]:
                dx_sub = sub[sub["eval_dx"] == dx]
                if dx_sub.empty:
                    continue

                n_folds = dx_sub["fold_country"].nunique()

                # Per-model: median country R²
                model_medians = dx_sub.groupby("model_id")["r2"].median()
                n_models = len(model_medians)
                grand_median = model_medians.median()

                # Bootstrap 95% CI on grand median
                rng = np.random.default_rng(_BOOT_SEED)
                boot_medians = []
                for _ in range(_N_BOOT):
                    samp = rng.choice(model_medians.values, size=n_models, replace=True)
                    boot_medians.append(np.median(samp))
                boot_medians = np.array(boot_medians)
                ci_lo = np.percentile(boot_medians, 2.5)
                ci_hi = np.percentile(boot_medians, 97.5)

                # Fraction of country-model pairs with R² > 0
                pct_positive = (dx_sub["r2"] > 0).mean()

                rows.append({
                    "bag": bag, "objective": obj, "eval_dx": dx,
                    "n_folds": n_folds, "n_models": n_models,
                    "median_r2": grand_median, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "pct_country_positive": pct_positive,
                })

        print(f"  {bag}: processed")

    result = pd.DataFrame(rows)

    # Gradient label + p_vs_CN (Wilcoxon)
    gradient_labels = []
    p_vs_cn_vals = []
    for _, row in result.iterrows():
        bag, obj, dx = row["bag"], row["objective"], row["eval_dx"]
        if dx == "CN":
            gradient_labels.append("reference")
            p_vs_cn_vals.append(np.nan)
            continue

        # Get CN median for comparison
        cn_row = result[(result["bag"] == bag) & (result["objective"] == obj) &
                        (result["eval_dx"] == "CN")]
        if cn_row.empty:
            gradient_labels.append("unknown")
            p_vs_cn_vals.append(np.nan)
            continue

        cn_median = cn_row.iloc[0]["median_r2"]
        dx_median = row["median_r2"]

        # Compare per-model medians via Mann-Whitney
        p_dx = ROOT / "dx_stratified_eval" / bag / "dx_stratified_country_dx.parquet"
        df_all = pd.read_parquet(p_dx)
        df_all = df_all[df_all["n_test"] >= _MIN_N_TEST]
        if "objective" in df_all.columns:
            df_all = df_all[df_all["objective"] == obj]

        cn_meds = df_all[df_all["eval_dx"] == "CN"].groupby("model_id")["r2"].median()
        dx_meds = df_all[df_all["eval_dx"] == dx].groupby("model_id")["r2"].median()

        common = cn_meds.index.intersection(dx_meds.index)
        if len(common) >= 5:
            stat, p_val = sp_stats.wilcoxon(cn_meds[common].values, dx_meds[common].values)
            p_vs_cn_vals.append(p_val)
        else:
            p_vs_cn_vals.append(np.nan)

        if dx_median > cn_median * _GRAD_AMPL:
            gradient_labels.append("amplification")
        elif dx_median > cn_median * _GRAD_PERS:
            gradient_labels.append("persistence")
        elif dx_median > 0:
            gradient_labels.append("attenuation")
        else:
            gradient_labels.append("collapse")

    result["gradient_label"] = gradient_labels
    result["p_vs_CN"] = p_vs_cn_vals

    out = STATS_DIR / "dx_stratified_summary.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")

    # Print summary table
    for bag in BAGS:
        sub = result[(result["bag"] == bag) & (result["objective"] == "o_min")]
        if sub.empty:
            continue
        print(f"\n  {bag} (o_min):")
        for _, r in sub.iterrows():
            p_str = f"p={r['p_vs_CN']:.2e}" if np.isfinite(r.get("p_vs_CN", np.nan)) else ""
            print(f"    {r['eval_dx']:4s}  R²={r['median_r2']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
                  f"folds={r['n_folds']}  +%={100*r['pct_country_positive']:.0f}%  "
                  f"{r['gradient_label']:15s} {p_str}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.4  CN-transfer summary table
# ═══════════════════════════════════════════════════════════════════════════════
def compute_transfer_summary() -> pd.DataFrame:
    print("\n── 2.4 CN-transfer summary table ──")
    rows = []

    for bag in BAGS:
        p = ROOT / "transfer_dx_eval" / bag / "transfer_country_dx.parquet"
        if not p.exists():
            print(f"  ⚠ Missing {p}")
            continue
        df = pd.read_parquet(p)
        df = df[df["n_test"] >= _MIN_N_TEST].copy()

        # Also load dx-stratified pooled for comparison
        pooled_p = ROOT / "dx_stratified_eval" / bag / "dx_stratified_country_dx.parquet"
        pooled = pd.read_parquet(pooled_p) if pooled_p.exists() else pd.DataFrame()
        if not pooled.empty:
            pooled = pooled[pooled["n_test"] >= _MIN_N_TEST]

        for obj in df["objective"].unique() if "objective" in df.columns else [None]:
            sub = df[df["objective"] == obj] if obj is not None else df

            for dx in sub["eval_dx"].unique():
                dx_sub = sub[sub["eval_dx"] == dx]
                if dx_sub.empty:
                    continue

                n_folds = dx_sub["fold_country"].nunique()
                model_medians = dx_sub.groupby("model_id")["r2"].median()
                n_models = len(model_medians)
                med = model_medians.median()

                # Bootstrap CI
                rng = np.random.default_rng(_BOOT_SEED)
                boot = [np.median(rng.choice(model_medians.values, size=n_models, replace=True))
                        for _ in range(_N_BOOT)]
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

                # Transfer decay vs pooled
                decay = np.nan
                if not pooled.empty:
                    if "objective" in pooled.columns and obj is not None:
                        p_sub = pooled[(pooled["objective"] == obj) & (pooled["eval_dx"] == dx)]
                    else:
                        p_sub = pooled[pooled["eval_dx"] == dx]
                    if not p_sub.empty:
                        p_model_medians = p_sub.groupby("model_id")["r2"].median()
                        decay = med - p_model_medians.median()

                rows.append({
                    "bag": bag, "objective": obj or "all", "eval_dx": dx,
                    "n_folds": n_folds, "n_models": n_models,
                    "median_r2_transfer": med, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "transfer_decay": decay,
                })

        print(f"  {bag}: processed")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "transfer_summary.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")

    if result.empty or "bag" not in result.columns:
        print("  ⚠ No transfer data available (skipped in this variant)")
        return result

    for bag in BAGS:
        sub = result[(result["bag"] == bag) & (result["objective"] == "o_min")]
        if sub.empty:
            sub = result[result["bag"] == bag]
        if sub.empty:
            continue
        print(f"\n  {bag} (transfer, o_min):")
        for _, r in sub.iterrows():
            print(f"    {r['eval_dx']:4s}  R²={r['median_r2_transfer']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
                  f"decay={r['transfer_decay']:+.4f}  folds={r['n_folds']}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.6  Country syn-vs-red proportions
# ═══════════════════════════════════════════════════════════════════════════════
def compute_country_syn_vs_red() -> pd.DataFrame:
    print("\n── 2.6 Country syn vs red proportions ──")
    rows = []

    for bag in BAGS:
        p = CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
        if not p.exists():
            print(f"  ⚠ No country metrics for {bag}")
            continue
        sub = pd.read_parquet(p)
        # Use best rung for o_min (same logic as aggregation hierarchy)
        mg_global = pd.read_parquet(
            CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        )
        best_rung_id = _best_rung_df(mg_global, objective="o_min")["rung_id"].iloc[0]
        sub = sub[sub["rung_id"] == best_rung_id] if "rung_id" in sub.columns else \
              sub[sub["rung_label"] == {"xgb_tree_d1": "XGB d1", "xgb_tree_d2": "XGB d2",
                                        "xgb_tree_d3": "XGB d3"}.get(best_rung_id, "XGB d2")]

        if sub.empty:
            print(f"  ⚠ No {best_rung_id} country metrics for {bag}")
            continue

        syn = sub[sub["objective"] == "o_min"]
        red = sub[sub["objective"] == "o_max"]

        countries = sorted(set(syn["fold_country"].unique()) & set(red["fold_country"].unique()))

        n_syn_wins = 0
        delta_list = []
        for c in countries:
            syn_best = syn[syn["fold_country"] == c]["country_full_r2"].max()
            red_best = red[red["fold_country"] == c]["country_full_r2"].max()
            delta = syn_best - red_best
            delta_list.append(delta)
            if syn_best > red_best:
                n_syn_wins += 1

        n_countries = len(countries)
        pct_syn_wins = n_syn_wins / n_countries if n_countries > 0 else np.nan
        median_delta = np.median(delta_list) if delta_list else np.nan
        mean_delta = np.mean(delta_list) if delta_list else np.nan

        # Sign test
        if delta_list:
            n_pos = sum(1 for d in delta_list if d > 0)
            p_sign = sp_stats.binomtest(n_pos, n_countries, 0.5).pvalue
        else:
            p_sign = np.nan

        rows.append({
            "bag": bag, "n_countries": n_countries,
            "n_syn_wins": n_syn_wins, "pct_syn_wins": pct_syn_wins,
            "median_delta_r2": median_delta, "mean_delta_r2": mean_delta,
            "p_sign_test": p_sign,
        })

        print(f"  {bag}: syn wins {n_syn_wins}/{n_countries} ({100*pct_syn_wins:.0f}%) "
              f"median Δ={median_delta:+.4f}  p={p_sign:.4f}")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "country_syn_vs_red.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.7  Enrichment stats persistence
# ═══════════════════════════════════════════════════════════════════════════════
def compute_enrichment_summary() -> pd.DataFrame:
    print("\n── 2.7 Enrichment stats ──")
    rows = []

    thresholds = _ENRICH_THRESH
    rng = np.random.default_rng(_BOOT_SEED)

    for bag in BAGS:
        p = CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        if not p.exists():
            print(f"  ⚠ Missing {p}")
            continue
        mg = pd.read_parquet(p)
        best_rung = _best_rung_df(mg, objective="o_min")
        best_rung = best_rung.sort_values("full_r2", ascending=False).reset_index(drop=True)
        n_total = len(best_rung)

        best_rung["is_syn"] = best_rung["objective"] == "o_min"
        n_syn_total = best_rung["is_syn"].sum()
        frac_syn_overall = n_syn_total / n_total

        for thr in thresholds:
            k = max(1, int(np.ceil(n_total * thr)))
            top_k = best_rung.head(k)
            n_syn_topk = top_k["is_syn"].sum()
            frac_syn_topk = n_syn_topk / k

            # Enrichment ratio
            enrichment = frac_syn_topk / frac_syn_overall if frac_syn_overall > 0 else np.nan

            # Bootstrap CI on fraction
            boot_fracs = []
            for _ in range(_N_BOOT):
                samp = rng.choice(top_k["is_syn"].values, size=k, replace=True)
                boot_fracs.append(samp.mean())
            ci_lo, ci_hi = np.percentile(boot_fracs, [2.5, 97.5])

            # Fisher exact test
            a = n_syn_topk
            b = k - n_syn_topk
            c = n_syn_total - n_syn_topk
            d = (n_total - n_syn_total) - b
            if d >= 0:
                _, p_fisher = sp_stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            else:
                p_fisher = np.nan

            rows.append({
                "bag": bag, "threshold": thr, "top_k": k, "n_total": n_total,
                "n_syn_topk": n_syn_topk, "frac_syn_topk": frac_syn_topk,
                "frac_syn_overall": frac_syn_overall, "enrichment_ratio": enrichment,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "p_fisher": p_fisher,
            })

        print(f"  {bag}: enrichment at 5%: "
              f"{[r for r in rows if r['bag']==bag and r['threshold']==0.05][0]['enrichment_ratio']:.2f}×  "
              f"p={[r for r in rows if r['bag']==bag and r['threshold']==0.05][0]['p_fisher']:.2e}")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "enrichment_summary.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2.8  Cohen's f² effect sizes
# ═══════════════════════════════════════════════════════════════════════════════
def compute_effect_sizes() -> pd.DataFrame:
    print("\n── 2.8 Effect sizes ──")
    rows = []

    for bag in BAGS:
        se_path = ROOT / "single_exposure_eval" / bag / "single_exposure_global.csv"
        if not se_path.exists():
            print(f"  ⚠ {bag}: missing single-exposure outputs — skipping")
            continue
        base_r2 = _ref(bag, "base_r2")
        syn_r2  = _ref(bag, "best_syn_r2")
        red_r2  = _ref(bag, "best_red_r2")

        # Best single exposure
        se = pd.read_csv(se_path)
        feats = se[se["feature_name"] != "__baseline__"]
        single_r2 = feats.iloc[0]["global_oof_r2"]

        for label, full_r2 in [("best_synergistic", syn_r2),
                                ("best_redundant", red_r2),
                                ("best_single_exposure", single_r2)]:
            delta_r2 = full_r2 - base_r2
            cohens_f2 = delta_r2 / (1 - full_r2) if full_r2 < 1 else np.nan
            partial_eta2 = delta_r2

            if cohens_f2 >= 0.35:
                interpretation = "large"
            elif cohens_f2 >= 0.15:
                interpretation = "medium"
            elif cohens_f2 >= 0.02:
                interpretation = "small"
            else:
                interpretation = "negligible"

            rows.append({
                "bag": bag, "model": label,
                "base_r2": base_r2, "full_r2": full_r2,
                "delta_r2": delta_r2, "cohens_f2": cohens_f2,
                "partial_eta2": partial_eta2, "interpretation": interpretation,
            })

        print(f"  {bag}: syn f²={rows[-3]['cohens_f2']:.4f} ({rows[-3]['interpretation']}), "
              f"red f²={rows[-2]['cohens_f2']:.4f} ({rows[-2]['interpretation']}), "
              f"single f²={rows[-1]['cohens_f2']:.4f} ({rows[-1]['interpretation']})")

    result = pd.DataFrame(rows)
    out = STATS_DIR / "effect_sizes.csv"
    result.to_csv(out, index=False)
    print(f"\n  ✓ Saved {out}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Phase 2 — Stats & Thin Wrappers")
    print("=" * 70)

    or_df   = compute_diagnosis_odds_ratios()
    hier_df = compute_aggregation_hierarchy()
    dx_df   = compute_dx_stratified_summary()
    tr_df   = compute_transfer_summary()
    svr_df  = compute_country_syn_vs_red()
    enr_df  = compute_enrichment_summary()
    eff_df  = compute_effect_sizes()

    print("\n" + "=" * 70)
    print("All Phase 2 stats completed. Files saved to:")
    print(f"  {STATS_DIR}")
    for f in sorted(STATS_DIR.glob("*.csv")):
        print(f"    {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
