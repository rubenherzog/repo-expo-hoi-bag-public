"""
compute_regression_residuals.py
================================
Subject- and country-level residual analysis for the regression pipeline.

The signed BAG bias is defined project-wide as

    bias = y_pred_full − y_true

so positive values denote BAG overestimation and negative values denote BAG
underestimation. No arbitrary threshold is needed.

Two complementary data sources are used:

1. Subject-level OOF (local, best-syn / best-red model per BAG):
       outputs/variant_a/subject_level_oof/{bag}/oof_predictions_best_syn.parquet
       outputs/variant_a/subject_level_oof/{bag}/oof_predictions_best_red.parquet
   Columns: row_id, N_MEGA, country, diagnosis, age, sex, y_true,
            y_pred_full, y_pred_base, exposure_contribution

2. Country-level bias from the cluster canonical parquet (top-20 per objective):
       /data/.../output_v4/variant_a/families/pooled_oinfo_ladder/canonical/
           per_experiment/pooled_oinfo_ladder_{bag}/metrics_country_long.parquet
   Key columns: fold_country, bias_mean (= mean signed residual),
                country_full_rmse, country_full_r2, candidate_id, objective, rung_id

Outputs (written to outputs/variant_a/stats/residuals/):
  residuals_subject_{bag}.csv          ← per-subject residuals from best-syn OOF
  residuals_subject_{bag}_red.csv      ← per-subject residuals from best-red OOF
  residuals_country_{bag}_{obj}.csv    ← country-level bias, averaged top-20
  residuals_dx_{bag}_{obj}.csv         ← diagnosis-level bias (averaged over countries)
  residuals_age_{bag}_{obj}.csv        ← age-group bias
  residuals_overall_{bag}_{obj}.csv    ← global summary stats
  residuals_country_best_{bag}_{obj}.csv ← country summary for one fixed best model
  residual_bias_subject_summary.csv    ← released global/diagnosis metrics
  residual_bias_diagnosis_tests.csv    ← released Kruskal-Wallis tests
  residual_bias_provenance.csv         ← exact OOF source, candidate, rung, cap
  residuals_by_rung_{bag}.csv          ← rung ladder comparison (combined, o_min)

The released outputs use HC/AD/FTLD only and may be routed separately
with ``V3_PAPER_RESULTS_DIR`` so the heavy subject tables remain in the bundle.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scripts.sensitivity_common import (
    load_sensitivity_config,
    paper_analysis_config,
)

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
# INPUT_ROOT is the canonical variant_a input root (overridable via
# V3_INPUT_ROOT so this script can read the complete external runtime layout
# instead of a fresh-run outputs/variant_a
# tree). Residual outputs always land under REPO_ROOT/outputs/variant_a/stats
# regardless of INPUT_ROOT, since that is the fixed location for residual confounds
# reads them back from.
INPUT_ROOT = Path(os.environ.get(
    "V3_INPUT_ROOT",
    str(REPO_ROOT / "outputs" / "variant_a"),
))
LOCAL_OOF  = INPUT_ROOT / "subject_level_oof"
INPUT_STATS_DIR = INPUT_ROOT / "stats"
STATS_DIR  = Path(os.environ.get("V3_FIG_STATS_DIR", str(REPO_ROOT / "outputs" / "variant_a" / "stats")))
OUT_DIR    = STATS_DIR / "residuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_RESULTS_DIR = Path(os.environ.get("V3_PAPER_RESULTS_DIR", str(OUT_DIR)))
PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PER_EXP    = INPUT_ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"

_SENSITIVITY_CFG = load_sensitivity_config()
_PAPER_CFG = paper_analysis_config(_SENSITIVITY_CFG)
_RESIDUAL_CFG = _PAPER_CFG.residual_bias
BAGS = [
    value.strip()
    for value in os.environ.get(
        "PAPER_FIG_BAGS", ",".join(_PAPER_CFG.residual_bias_bags)
    ).split(",")
    if value.strip()
]
OBJECTIVES = list(_PAPER_CFG.objectives)
ALL_RUNGS = [str(value) for value in _SENSITIVITY_CFG["defaults"]["active_rungs"]]
ANALYSIS = str(_RESIDUAL_CFG["best_rung_analysis"])
TOP_K = int(_RESIDUAL_CFG["top_k_candidates"])
OBJ_SUFFIX = {
    objective: values["file_suffix"]
    for objective, values in _PAPER_CFG.objective_metadata.items()
}
# Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): an o_min
# candidate counts as synergistic only if its evaluated O-info (thoi_o) < 0.
_SYN_OINFO_NEGATIVE = os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

AGE_BINS = list(_RESIDUAL_CFG["age_bins"])
AGE_LABELS = [str(value) for value in _RESIDUAL_CFG["age_labels"]]
PRIMARY_DIAGNOSES = list(_PAPER_CFG.primary_diagnoses)
CONTROL_DIAGNOSIS = _PAPER_CFG.control_diagnosis
RUNG_LADDER_BAG = str(_RESIDUAL_CFG["rung_ladder_bag"])
RUNG_LADDER_OBJECTIVE = str(_RESIDUAL_CFG["rung_ladder_objective"])
COUNTRY_PAIRED_DRAWS = int(_RESIDUAL_CFG["country_paired_draws"])
COUNTRY_PAIRED_SEED = int(_RESIDUAL_CFG["country_paired_seed"])

if len(AGE_LABELS) != len(AGE_BINS) - 1:
    raise ValueError("paper_analysis.residual_bias age labels/bins do not align")
if TOP_K < 1:
    raise ValueError("paper_analysis.residual_bias.top_k_candidates must be positive")
if RUNG_LADDER_OBJECTIVE not in OBJECTIVES:
    raise ValueError("Configured rung-ladder objective is not a paper objective")
if COUNTRY_PAIRED_DRAWS < 1:
    raise ValueError("residual_bias.country_paired_draws must be positive")


# ── helpers ────────────────────────────────────────────────────────────────

def load_best_rungs() -> pd.DataFrame:
    p = INPUT_STATS_DIR / "best_rung_selection.csv"
    df = pd.read_csv(p)
    return df[df["analysis"] == ANALYSIS][["bag", "objective", "best_rung"]].copy()


def _subject_oof_path(bag: str, objective: str) -> Path:
    return LOCAL_OOF / bag / f"oof_predictions_best_{OBJ_SUFFIX[objective]}.parquet"


def _subject_out_path(bag: str, objective: str) -> Path:
    return OUT_DIR / f"residuals_subject_{bag}_{OBJ_SUFFIX[objective]}.csv"


def _country_out_path(bag: str, objective: str) -> Path:
    return OUT_DIR / f"residuals_country_{bag}_{objective}.csv"


def _dx_out_path(bag: str, objective: str) -> Path:
    return OUT_DIR / f"residuals_dx_{bag}_{OBJ_SUFFIX[objective]}.csv"


def _age_out_path(bag: str, objective: str) -> Path:
    return OUT_DIR / f"residuals_age_{bag}_{OBJ_SUFFIX[objective]}.csv"


def _overall_out_path(bag: str, objective: str) -> Path:
    return OUT_DIR / f"residuals_overall_{bag}_{objective}.csv"


def get_top_candidate_ids(bag: str, rung_id: str, objective: str) -> set[str]:
    gpath = PER_EXP / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    gdf = pd.read_parquet(gpath)
    sub = gdf[(gdf["rung_id"] == rung_id) & (gdf["objective"] == objective)]
    # Optional negative-O-info synergy criterion: keep only o_min candidates with
    # evaluated O-info (thoi_o) < 0. o_max untouched. No-op when the flag is off.
    if _SYN_OINFO_NEGATIVE and objective == "o_min":
        if "thoi_o" not in sub.columns:
            raise KeyError("SYN_OINFO_NEGATIVE set but 'thoi_o' absent from metrics.")
        sub = sub[pd.to_numeric(sub["thoi_o"], errors="coerce") < 0]
    top = sub.sort_values("full_r2", ascending=False).head(TOP_K)
    return set(top["candidate_id"].tolist())


def load_country_bias(bag: str, rung_id: str, objective: str,
                      top_ids: set[str]) -> pd.DataFrame:
    """
    Load country-level bias_mean from cluster parquet, filtered to top_ids.
    Returns DataFrame: fold_country, bias_mean, rmse, r2, mae (averaged over top-K).
    bias_mean = mean(y_pred - y_true) per country fold (signed: + = overestimate).
    """
    cpath = PER_EXP / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
    df = pd.read_parquet(cpath)
    sub = df[
        (df["rung_id"] == rung_id) &
        (df["objective"] == objective) &
        (df["candidate_id"].isin(top_ids))
    ].copy()
    agg = (
        sub.groupby("fold_country")
        .agg(
            bias_mean=("bias_mean", "mean"),
            rmse=("country_full_rmse", "mean"),
            r2=("country_full_r2", "mean"),
            mae=("country_full_mae", "mean"),
            bias_mean_base=("bias_mean_base", "mean"),
            n_subjects=("n_test_scored", "mean"),
        )
        .reset_index()
        .rename(columns={"fold_country": "country"})
    )
    return agg


def load_subject_oof(bag: str, objective: str) -> pd.DataFrame:
    """Load best objective-specific subject OOF, compute residuals and age groups."""
    df = pd.read_parquet(_subject_oof_path(bag, objective))
    df["residual"] = df["y_pred_full"] - df["y_true"]
    df["abs_residual"] = df["residual"].abs()
    df["residual_base"] = df["y_pred_base"] - df["y_true"]
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["objective"] = objective
    return df


def country_dx_bias_from_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute country × diagnosis bias stats from subject OOF.
    Returns: country, diagnosis, bias_mean, bias_std, rmse, mae, n_subjects
    """
    records = []
    for (country, dx), grp in df.groupby(["country", "diagnosis"]):
        n = len(grp)
        if n == 0:
            continue
        records.append({
            "country": country,
            "diagnosis": dx,
            "bias_mean": grp["residual"].mean(),
            "bias_std": grp["residual"].std(),
            "abs_bias": grp["residual"].abs().mean(),
            "rmse": np.sqrt((grp["residual"] ** 2).mean()),
            "mae": grp["residual"].abs().mean(),
            "pct_under": (grp["residual"] < 0).mean(),   # fraction under-predicted
            "pct_over": (grp["residual"] > 0).mean(),    # fraction over-predicted
            "n_subjects": n,
        })
    return pd.DataFrame(records)


def country_bias_from_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Country summaries for the single fixed best model represented by ``df``."""
    records = []
    for country, grp in df.groupby("country"):
        error = grp["residual"].to_numpy(dtype=float)
        y_true = pd.to_numeric(grp["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(grp["y_pred_full"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(error) & np.isfinite(y_true) & np.isfinite(y_pred)
        if not ok.any():
            continue
        centered = y_true[ok] - np.mean(y_true[ok])
        denominator = float(np.sum(centered**2))
        r2 = 1.0 - float(np.sum(error[ok] ** 2)) / denominator if denominator > 0 else np.nan
        records.append(
            {
                "country": country,
                "bias_mean": float(np.mean(error[ok])),
                "rmse": float(np.sqrt(np.mean(error[ok] ** 2))),
                "mae": float(np.mean(np.abs(error[ok]))),
                "r2": r2,
                "n_subjects": int(ok.sum()),
            }
        )
    return pd.DataFrame(records)


def _subject_summary_rows(df: pd.DataFrame, bag: str, objective: str) -> list[dict]:
    rows = []
    primary = df[df["diagnosis"].astype(str).isin(PRIMARY_DIAGNOSES)].copy()
    groups = [("all", primary)] + [
        (diagnosis, primary[primary["diagnosis"].astype(str) == diagnosis])
        for diagnosis in PRIMARY_DIAGNOSES
    ]
    for diagnosis, grp in groups:
        error = pd.to_numeric(grp["residual"], errors="coerce").dropna().to_numpy(dtype=float)
        if error.size == 0:
            continue
        rows.append(
            {
                "bag": bag,
                "objective": objective,
                "diagnosis": diagnosis,
                "n_subjects": int(error.size),
                "bias_mean": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "candidate_id": str(df["candidate_id"].iloc[0]),
                "rung_id": str(df["best_rung"].iloc[0]),
                "bias_definition": "predicted_minus_observed",
            }
        )
    return rows


def _diagnosis_test_rows(df: pd.DataFrame, bag: str, objective: str) -> list[dict]:
    groups = [
        pd.to_numeric(
            df.loc[df["diagnosis"].astype(str) == diagnosis, "residual"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        for diagnosis in PRIMARY_DIAGNOSES
    ]
    if any(values.size == 0 for values in groups):
        raise ValueError(f"Missing a primary diagnosis for {bag}/{objective}")
    rows = []
    outcomes = (
        ("signed_bias", groups),
        ("absolute_error", [np.abs(values) for values in groups]),
    )
    for outcome, values in outcomes:
        result = stats.kruskal(*values)
        rows.append(
            {
                "bag": bag,
                "objective": objective,
                "outcome": outcome,
                "test": "Kruskal-Wallis",
                "groups": "|".join(PRIMARY_DIAGNOSES),
                "statistic_h": float(result.statistic),
                "p_value": float(result.pvalue),
                "n_subjects": int(sum(len(group) for group in groups)),
                "candidate_id": str(df["candidate_id"].iloc[0]),
                "rung_id": str(df["best_rung"].iloc[0]),
            }
        )
    return rows


def _country_paired_diagnosis_rows(
    df: pd.DataFrame,
    bag: str,
    objective: str,
) -> list[dict]:
    """Compare disease and control errors within countries, weighting countries equally."""
    primary = df[df["diagnosis"].astype(str).isin(PRIMARY_DIAGNOSES)].copy()
    primary["absolute_error"] = pd.to_numeric(
        primary["residual"], errors="coerce"
    ).abs()
    rows: list[dict] = []
    for outcome, value_col in (
        ("signed_bias", "residual"),
        ("absolute_error", "absolute_error"),
    ):
        country_diagnosis = (
            primary.groupby(["country", "diagnosis"], observed=True)[value_col]
            .mean()
            .unstack("diagnosis")
        )
        for diagnosis in PRIMARY_DIAGNOSES:
            if diagnosis == CONTROL_DIAGNOSIS:
                continue
            paired = country_diagnosis[[CONTROL_DIAGNOSIS, diagnosis]].dropna()
            difference = (
                paired[diagnosis] - paired[CONTROL_DIAGNOSIS]
            ).to_numpy(dtype=float)
            if difference.size < 2:
                raise ValueError(
                    f"Fewer than two paired countries for {bag}/{objective}/"
                    f"{diagnosis}/{outcome}"
                )

            seed_offset = sum(
                ord(char)
                for char in f"{bag}|{objective}|{diagnosis}|{outcome}"
            )
            rng = np.random.default_rng(COUNTRY_PAIRED_SEED + seed_offset)
            bootstrap_indices = rng.integers(
                0,
                difference.size,
                size=(COUNTRY_PAIRED_DRAWS, difference.size),
            )
            bootstrap_means = difference[bootstrap_indices].mean(axis=1)
            signs = rng.choice(
                (-1.0, 1.0),
                size=(COUNTRY_PAIRED_DRAWS, difference.size),
            )
            null_statistics = np.abs((signs * difference).mean(axis=1))
            observed_statistic = abs(float(difference.mean()))
            exceedances = int(
                np.count_nonzero(null_statistics >= observed_statistic)
            )
            rows.append(
                {
                    "bag": bag,
                    "objective": objective,
                    "outcome": outcome,
                    "contrast": f"{diagnosis}-{CONTROL_DIAGNOSIS}",
                    "diagnosis": diagnosis,
                    "reference_diagnosis": CONTROL_DIAGNOSIS,
                    "n_paired_countries": int(difference.size),
                    "mean_country_difference": float(difference.mean()),
                    "median_country_difference": float(np.median(difference)),
                    "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci_high": float(np.quantile(bootstrap_means, 0.975)),
                    "permutation_p": (exceedances + 1)
                    / (COUNTRY_PAIRED_DRAWS + 1),
                    "permutation_exceedances": exceedances,
                    "n_draws": COUNTRY_PAIRED_DRAWS,
                    "country_weighting": "equal",
                    "difference_definition": "diagnosis_minus_control",
                    "bias_definition": "predicted_minus_observed",
                    "candidate_id": str(df["candidate_id"].iloc[0]),
                    "rung_id": str(df["best_rung"].iloc[0]),
                }
            )
    return rows


def age_bias_from_subject(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for age_grp, grp in df.groupby("age_group", observed=True):
        records.append({
            "age_group": str(age_grp),
            "bias_mean": grp["residual"].mean(),
            "bias_std": grp["residual"].std(),
            "rmse": np.sqrt((grp["residual"] ** 2).mean()),
            "pct_under": (grp["residual"] < 0).mean(),
            "pct_over": (grp["residual"] > 0).mean(),
            "n_subjects": len(grp),
        })
    return pd.DataFrame(records)


def dx_bias_from_subject(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dx, grp in df.groupby("diagnosis"):
        records.append({
            "diagnosis": dx,
            "bias_mean": grp["residual"].mean(),
            "bias_std": grp["residual"].std(),
            "rmse": np.sqrt((grp["residual"] ** 2).mean()),
            "pct_under": (grp["residual"] < 0).mean(),
            "pct_over": (grp["residual"] > 0).mean(),
            "n_subjects": len(grp),
        })
    return pd.DataFrame(records)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    best_rungs_df = load_best_rungs()
    subject_summary_rows = []
    diagnosis_test_rows = []
    country_paired_diagnosis_rows = []
    provenance_rows = []
    print("Best rungs (A_top_k_per_rung):")
    print(best_rungs_df.to_string(index=False))

    for bag in BAGS:
        print(f"\n{'='*60}\nBAG: {bag}\n{'='*60}")

        for objective in OBJECTIVES:
            suffix = OBJ_SUFFIX[objective]

            # ── subject-level from local objective-specific OOF ───────────
            subj = load_subject_oof(bag, objective)
            subject_summary_rows.extend(_subject_summary_rows(subj, bag, objective))
            diagnosis_test_rows.extend(_diagnosis_test_rows(subj, bag, objective))
            country_paired_diagnosis_rows.extend(
                _country_paired_diagnosis_rows(subj, bag, objective)
            )
            provenance_rows.append(
                {
                    "bag": bag,
                    "objective": objective,
                    "input_oof": str(_subject_oof_path(bag, objective).resolve()),
                    "candidate_id": str(subj["candidate_id"].iloc[0]),
                    "rung_id": str(subj["best_rung"].iloc[0]),
                    "set_size_cap": _PAPER_CFG.order_max,
                }
            )
            print(f"  [{suffix}] Subject OOF: {len(subj)} subjects")
            print(f"  [{suffix}] Residual: mean={subj['residual'].mean():.3f}, "
                  f"std={subj['residual'].std():.3f}, "
                  f"RMSE={np.sqrt((subj['residual']**2).mean()):.3f}")
            print(f"  [{suffix}] Under-predicted (bias<0): {(subj['residual']<0).mean():.3f}  "
                  f"Over-predicted (bias>0): {(subj['residual']>0).mean():.3f}")

            subj_cols = [
                "row_id", "N_MEGA", "country", "diagnosis", "age", "sex",
                "y_true", "y_pred_full", "y_pred_base", "exposure_contribution",
                "residual", "abs_residual", "residual_base", "age_group",
            ]
            for extra in ["bag_target", "candidate_id", "best_rung", "objective"]:
                if extra in subj.columns:
                    subj_cols.append(extra)
            subj_out = _subject_out_path(bag, objective)
            subj[subj_cols].to_csv(subj_out, index=False)
            print(f"  Saved: {subj_out}")
            if objective == "o_min":
                legacy_subj_out = OUT_DIR / f"residuals_subject_{bag}.csv"
                subj[subj_cols].to_csv(legacy_subj_out, index=False)
                print(f"  Saved: {legacy_subj_out}")

            # ── country × diagnosis from subject OOF ──────────────────────
            cdx = country_dx_bias_from_subject(subj)
            cdx["bag"] = bag
            cdx["objective"] = objective
            cdx_out = OUT_DIR / f"residuals_ctry_dx_{bag}_{suffix}.csv"
            cdx.to_csv(cdx_out, index=False)
            print(f"  Saved: {cdx_out}  ({len(cdx)} rows)")
            if objective == "o_min":
                legacy_cdx_out = OUT_DIR / f"residuals_ctry_dx_{bag}.csv"
                cdx.to_csv(legacy_cdx_out, index=False)
                print(f"  Saved: {legacy_cdx_out}")

            # ── country summaries from this single fixed best model ─────
            primary_subj = subj[subj["diagnosis"].astype(str).isin(PRIMARY_DIAGNOSES)].copy()
            country_best = country_bias_from_subject(primary_subj)
            country_best["bag"] = bag
            country_best["objective"] = objective
            country_best["candidate_id"] = str(subj["candidate_id"].iloc[0])
            country_best["rung_id"] = str(subj["best_rung"].iloc[0])
            country_best["aggregation_pool"] = "best_single_model"
            country_best["diagnosis_pool"] = "|".join(PRIMARY_DIAGNOSES)
            country_best_out = PAPER_RESULTS_DIR / f"residuals_country_best_{bag}_{suffix}.csv"
            country_best.to_csv(country_best_out, index=False)
            print(f"  Saved: {country_best_out}  countries={len(country_best)}")

            # ── diagnosis bias ────────────────────────────────────────────
            dx_bias = dx_bias_from_subject(subj)
            dx_bias["bag"] = bag
            dx_bias["objective"] = objective
            dx_out = _dx_out_path(bag, objective)
            dx_bias.to_csv(dx_out, index=False)
            print(f"  Saved: {dx_out}")
            if objective == "o_min":
                legacy_dx_out = OUT_DIR / f"residuals_dx_{bag}.csv"
                dx_bias.to_csv(legacy_dx_out, index=False)
                print(f"  Saved: {legacy_dx_out}")

            # ── age stratification ────────────────────────────────────────
            age_bias = age_bias_from_subject(subj)
            age_bias["bag"] = bag
            age_bias["objective"] = objective
            age_out = _age_out_path(bag, objective)
            age_bias.to_csv(age_out, index=False)
            print(f"  Saved: {age_out}")
            if objective == "o_min":
                legacy_age_out = OUT_DIR / f"residuals_age_{bag}.csv"
                age_bias.to_csv(legacy_age_out, index=False)
                print(f"  Saved: {legacy_age_out}")

        # ── country-level from cluster parquet (top-20, per objective) ────
        # Optional: needs per-experiment metrics_country_long.parquet. Capped
        # staging trees may not carry it; skip gracefully (fig4 reads only the
        # subject-derived residuals_subject_* / residuals_ctry_dx_* above).
        for objective in OBJECTIVES:
            row = best_rungs_df[(best_rungs_df["bag"] == bag) &
                                (best_rungs_df["objective"] == objective)]
            if row.empty:
                continue
            rung_id = row.iloc[0]["best_rung"]
            print(f"\n  {objective.upper()} → best rung: {rung_id}")
            if not (PER_EXP / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet").is_file():
                print(f"  ⚠ Skipping country-level bias for {bag}/{objective}: metrics_country_long.parquet absent")
                continue

            top_ids = get_top_candidate_ids(bag, rung_id, objective)
            print(f"  Top-{TOP_K} candidate_ids: {len(top_ids)}")

            ctry_bias = load_country_bias(bag, rung_id, objective, top_ids)
            ctry_bias["bag"] = bag
            ctry_bias["objective"] = objective
            ctry_bias["rung_id"] = rung_id
            ctry_bias["aggregation_pool"] = "top_20_candidates"
            ctry_bias["n_models"] = TOP_K
            ctry_out = _country_out_path(bag, objective)
            ctry_bias.to_csv(ctry_out, index=False)
            print(f"  Saved: {ctry_out}  countries={len(ctry_bias)}")
            print(f"  Country bias range: {ctry_bias['bias_mean'].min():.3f} "
                  f"to {ctry_bias['bias_mean'].max():.3f}")

            # overall summary
            overall = {
                "bag": bag, "objective": objective, "rung_id": rung_id,
                "global_bias_mean": ctry_bias["bias_mean"].mean(),
                "global_rmse_mean": ctry_bias["rmse"].mean(),
                "global_r2_mean": ctry_bias["r2"].mean(),
                "n_countries": len(ctry_bias),
                "aggregation_pool": "top_20_candidates",
                "n_models": TOP_K,
            }
            pd.DataFrame([overall]).to_csv(_overall_out_path(bag, objective), index=False)

        # ── rung ladder comparison (combined BAG, o_min only) ─────────────
        if bag == RUNG_LADDER_BAG:
            print("\n  --- Rung ladder comparison ---")
            rung_rows = []
            for rung_id in ALL_RUNGS:
                top_ids = get_top_candidate_ids(
                    bag, rung_id, RUNG_LADDER_OBJECTIVE
                )
                ctry_bias = load_country_bias(
                    bag, rung_id, RUNG_LADDER_OBJECTIVE, top_ids
                )
                rung_rows.append({
                    "rung_id": rung_id,
                    "bias_mean": ctry_bias["bias_mean"].mean(),
                    "rmse": ctry_bias["rmse"].mean(),
                    "r2": ctry_bias["r2"].mean(),
                    "abs_bias": ctry_bias["bias_mean"].abs().mean(),
                })
            rung_df = pd.DataFrame(rung_rows)
            rung_df.to_csv(
                OUT_DIR / f"residuals_by_rung_{RUNG_LADDER_BAG}.csv", index=False
            )
            print(rung_df.to_string(index=False))

    pd.DataFrame(subject_summary_rows).to_csv(
        PAPER_RESULTS_DIR / "residual_bias_subject_summary.csv", index=False
    )
    pd.DataFrame(diagnosis_test_rows).to_csv(
        PAPER_RESULTS_DIR / "residual_bias_diagnosis_tests.csv", index=False
    )
    pd.DataFrame(country_paired_diagnosis_rows).to_csv(
        PAPER_RESULTS_DIR / "residual_bias_country_paired_diagnosis.csv",
        index=False,
    )
    pd.DataFrame(provenance_rows).to_csv(
        PAPER_RESULTS_DIR / "residual_bias_provenance.csv", index=False
    )
    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
