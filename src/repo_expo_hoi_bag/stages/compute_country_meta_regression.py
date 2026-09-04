"""
Country meta-regression — explains country-level R² variation.

Model: lm(country_R² ~ log(n_test) + BAG_sd)
  for the fixed best model selected by the shared paper-analysis configuration.

This is a descriptive analysis of country-level predictive performance. It does
not test whether country structures signed BAG bias; that question is handled by
the country-random-intercept residual analysis.

Also produces a diagnostic scatter plot.

Outputs:
  stats/country_meta_regression.csv
  outputs/fig_country_meta_regression.{png,pdf}
"""

from __future__ import annotations
import os
import pathlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scripts.sensitivity_common import (
    load_sensitivity_config,
    paper_analysis_config,
)

# ── paths ───────────────────────────────────────────────────────────────
ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
RAW_PATH = pathlib.Path(os.environ.get(
    "V3_RAW_PATH",
    "data/"
    "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
))
CANONICAL = ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
if not CANONICAL.exists():
    CANONICAL = ROOT / "canonical" / "per_experiment"
FIG_DIR = pathlib.Path(os.environ.get("V3_FIG_OUTPUT_DIR", str(ROOT / "figures")))
STATS_DIR = pathlib.Path(os.environ.get("V3_FIG_STATS_DIR", str(ROOT / "stats")))
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

_SENSITIVITY_CFG = load_sensitivity_config()
_PAPER_CFG = paper_analysis_config(_SENSITIVITY_CFG)
_DEFAULTS_CFG = _SENSITIVITY_CFG["defaults"]
BAGS = [
    bag
    for bag in os.environ.get(
        "PAPER_FIG_BAGS", ",".join(_PAPER_CFG.primary_bags)
    ).split(",")
    if bag
]
BAG_COL  = {"combined": "BAG_comb", "functional": "BAG_func", "structural": "BAG_struc"}
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}
RUNG_ID = _PAPER_CFG.deployed_rung
ORDER_MAX = _PAPER_CFG.order_max
OBJECTIVE = _PAPER_CFG.country_meta_regression_objective
EXCLUDE_COUNTRIES = [
    value.strip()
    for value in os.environ.get(
        "V3_EXCLUDE_COUNTRIES", ",".join(_DEFAULTS_CFG["exclude_countries"])
    ).split(",")
    if value.strip()
]
EXCLUDE_DIAGNOSES = [
    value.strip()
    for value in os.environ.get(
        "V3_EXCLUDE_DIAGNOSIS", ",".join(_DEFAULTS_CFG["exclude_diagnosis"])
    ).split(",")
    if value.strip()
]

# ── Load raw data for country-level BAG sd ───────────────────────────────
raw = pd.read_csv(RAW_PATH, low_memory=False)
raw = raw[
    ~raw["country_clean"].astype(str).isin(EXCLUDE_COUNTRIES)
    & ~raw["Diagnosis"].astype(str).isin(EXCLUDE_DIAGNOSES)
].copy()

bag_sd = {}
for bag in BAGS:
    col = BAG_COL[bag]
    sub = raw[raw[col].notna()]
    stats = sub.groupby("country_clean")[col].agg(["std", "count"]).rename(
        columns={"std": "bag_sd", "count": "bag_n_total"}
    )
    bag_sd[bag] = stats

# ── Run meta-regression per BAG ──────────────────────────────────────────
fig, axes = plt.subplots(1, len(BAGS), figsize=(6.7 * len(BAGS), 6), squeeze=False)
axes = axes.ravel()
all_reg_rows = []

for i, bag in enumerate(BAGS):
    canon_path = CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
    if not canon_path.exists():
        print(f"  ⚠ {bag}: missing canonical metrics — skipping")
        continue
    country_metrics = pd.read_parquet(canon_path)
    global_path = CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    global_metrics = pd.read_parquet(global_path)
    selection_pool = global_metrics[
        (global_metrics["rung_id"].astype(str) == RUNG_ID)
        & (global_metrics["objective"].astype(str) == OBJECTIVE)
        & (pd.to_numeric(global_metrics["order"], errors="coerce") <= ORDER_MAX)
    ].copy()
    if selection_pool.empty:
        raise ValueError(
            f"No {bag} {OBJECTIVE} models at {RUNG_ID}, order<={ORDER_MAX}"
        )
    selected = selection_pool.nlargest(1, "full_r2").iloc[0]
    best_id = str(selected["candidate_id"])
    best = country_metrics[
        (country_metrics["candidate_id"].astype(str) == best_id)
        & (country_metrics["rung_id"].astype(str) == RUNG_ID)
    ][
        ["fold_country", "country_full_r2", "n_test_scored"]
    ].copy()
    best = best.rename(columns={"country_full_r2": "r2", "n_test_scored": "n_test"})

    # Merge BAG sd
    sd_df = bag_sd[bag].reset_index().rename(columns={"country_clean": "fold_country"})
    best = best.merge(sd_df, on="fold_country", how="left")
    best["log_n_test"] = np.log(best["n_test"])
    best = best.dropna(subset=["bag_sd"])

    print(f"\n{'='*60}")
    print(f"  {bag.upper()} BAG — best synergistic model: {best_id}")
    print(f"  n_countries = {len(best)}")
    print(f"{'='*60}")
    print(best[["fold_country", "r2", "n_test", "bag_sd"]].to_string(index=False))

    # Regression
    m = smf.ols("r2 ~ log_n_test + bag_sd", data=best).fit()
    print(f"\n--- lm(R² ~ log(n_test) + BAG_sd) ---")
    print(m.summary2().tables[1].to_string())
    print(f"Model R² = {m.rsquared:.4f}, Adj R² = {m.rsquared_adj:.4f}")

    tbl = m.summary2().tables[1].copy()
    tbl["bag"] = bag
    tbl["model_r2"] = m.rsquared
    tbl["n_obs"] = int(m.nobs)
    tbl["candidate_id"] = best_id
    tbl["rung_id"] = RUNG_ID
    tbl["order_max"] = ORDER_MAX
    tbl["objective"] = OBJECTIVE
    tbl["outcome"] = "country_oof_r2"
    all_reg_rows.append(tbl)

    # ── Scatter plot ─────────────────────────────────────────────────
    ax = axes[i]
    sc = ax.scatter(
        best["log_n_test"], best["r2"],
        c=best["bag_sd"], cmap="coolwarm", s=60, edgecolors="white",
        linewidths=0.5, zorder=3
    )
    # Label points
    for _, row in best.iterrows():
        ax.annotate(
            row["fold_country"],
            (row["log_n_test"], row["r2"]),
            fontsize=6.5, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points"
        )

    ax.axhline(0, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("log(n_test)", fontsize=10)
    ax.set_ylabel("Country OOF R²", fontsize=10)
    ax.set_title(
        f"{BAG_LABELS[bag]}\nAdj R²={m.rsquared_adj:.3f}",
        fontsize=11
    )
    plt.colorbar(sc, ax=ax, label="BAG SD (within-country)", shrink=0.85)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Country meta-regression: R² ~ log(n_test) + BAG variability",
    fontsize=13, y=1.02
)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig_country_meta_regression.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / f'fig_country_meta_regression.{ext}'}")
plt.close(fig)

out = pd.concat(all_reg_rows, ignore_index=False)
out.to_csv(STATS_DIR / "country_meta_regression.csv")
print(f"Saved {STATS_DIR / 'country_meta_regression.csv'}")
print("\nDone.")
