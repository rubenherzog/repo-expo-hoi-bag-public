#!/usr/bin/env python3
"""BAG calibration by country / age / sex / diagnosis.

Reviewer question: could systematic age/diagnosis/site/geography bias in the
Brain-Age-Gap (BAG) model propagate into the exposome analysis? A well-calibrated
BAG should have ~zero mean in healthy controls and ~zero slope vs age in controls.

Reads only the cohort CSV (no model outputs needed) and, per BAG target
(BAG_comb, BAG_func, BAG_struc), reports:
  - mean/SD BAG overall and in CN only (calibration offset);
  - OLS slope of BAG vs Age in CN (age-bias) with CI/p;
  - mean BAG by diagnosis, sex, country, and continent (group offsets);
  - variance of BAG explained by covariates (age+sex+dx+country) via OLS R²
    (how much "BAG" is structured by demographics/geography rather than biology).

Outputs lightweight CSVs to ``outputs/sensitivity/bag_calibration/`` (repo).

Usage:
    python -m scripts.compute_bag_calibration            # uses data/ cohort CSV
    V3_RAW_PATH=/path/to/cohort.csv python -m scripts.compute_bag_calibration
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

RAW = Path(os.environ.get(
    "V3_RAW_PATH",
    "data/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
))
OUT_DIR = Path(os.environ.get("V3_SENS_OUT", "outputs/sensitivity")) / "bag_calibration"
BAGS = {"combined": "BAG_comb", "functional": "BAG_func", "structural": "BAG_struc"}
CN = "CN"
MIN_GROUP = 20


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² of y ~ [1, X] via least squares."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _design(df: pd.DataFrame) -> np.ndarray:
    """Covariate design (age + sex dummies + diagnosis dummies + country dummies)."""
    parts = [df["Age"].to_numpy(dtype=float)[:, None]]
    for col in ("Sex", "Diagnosis", "country_clean"):
        if col in df.columns:
            parts.append(pd.get_dummies(df[col].astype(str), drop_first=True).to_numpy(dtype=float))
    return np.column_stack(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"BAG calibration  (cohort: {RAW})")
    print("=" * 70)
    df = pd.read_csv(RAW)

    overall_rows, group_rows = [], []
    for bag, col in BAGS.items():
        if col not in df.columns:
            print(f"  ⚠ {bag}: column {col} missing — skipping"); continue
        d = df[[col, "Age", "Sex", "Diagnosis", "country_clean"]].dropna(subset=[col, "Age"]).copy()
        d = d.rename(columns={col: "bag"})
        cn = d[d["Diagnosis"] == CN]

        # age-bias slope in CN
        if len(cn) >= 10:
            lr = scipy_stats.linregress(cn["Age"], cn["bag"])
            age_slope, age_p, age_r = lr.slope, lr.pvalue, lr.rvalue
        else:
            age_slope = age_p = age_r = np.nan

        # variance explained by covariates
        d_cov = d.dropna(subset=["Age"]).copy()
        r2_cov = _ols_r2(d_cov["bag"].to_numpy(float), _design(d_cov)) if len(d_cov) > 50 else np.nan

        overall_rows.append({
            "bag": bag, "n": len(d), "n_CN": len(cn),
            "mean_bag_all": float(d["bag"].mean()), "sd_bag_all": float(d["bag"].std()),
            "mean_bag_CN": float(cn["bag"].mean()) if len(cn) else np.nan,
            "sd_bag_CN": float(cn["bag"].std()) if len(cn) else np.nan,
            "age_slope_CN": float(age_slope), "age_slope_p_CN": float(age_p),
            "age_corr_CN": float(age_r),
            "r2_explained_by_covariates": float(r2_cov),
        })

        # group offsets
        for level, gcol in (("diagnosis", "Diagnosis"), ("sex", "Sex"),
                            ("country", "country_clean")):
            for g, sub in d.groupby(gcol):
                if len(sub) < (MIN_GROUP if level == "country" else 1):
                    continue
                group_rows.append({
                    "bag": bag, "group_level": level, "group": str(g),
                    "n": len(sub), "mean_bag": float(sub["bag"].mean()),
                    "median_bag": float(sub["bag"].median()), "sd_bag": float(sub["bag"].std()),
                })

    ov = pd.DataFrame(overall_rows)
    gp = pd.DataFrame(group_rows)
    ov.to_csv(OUT_DIR / "bag_calibration_overall.csv", index=False)
    gp.to_csv(OUT_DIR / "bag_calibration_by_group.csv", index=False)
    print(f"\n  ✓ Saved {OUT_DIR/'bag_calibration_overall.csv'}  ({len(ov)} rows)")
    print(f"  ✓ Saved {OUT_DIR/'bag_calibration_by_group.csv'}  ({len(gp)} rows)")
    print("\n  Calibration summary (CN offset, age-bias slope, covariate R²):")
    for _, r in ov.iterrows():
        print(f"    {r['bag']:<11} mean_CN={r['mean_bag_CN']:+.3f}  "
              f"age_slope_CN={r['age_slope_CN']:+.4f} (p={r['age_slope_p_CN']:.1e})  "
              f"R²_covariates={r['r2_explained_by_covariates']:.3f}")


if __name__ == "__main__":
    main()
