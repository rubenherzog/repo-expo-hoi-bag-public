#!/usr/bin/env python3
"""Domain-diversity association at the deployed depth-3 rung (dedup-complete pool).

Regenerates every number in the manuscript's domain-diversity paragraph and in
Table S5. Additive aggregation over the EXISTING dedup candidate pool
(`metrics_global_long.parquet`); no model is refit.

Why this file exists:
  - The four unadjusted correlations quoted in the paragraph existed nowhere on
    disk as a table. They were readable only from the Fig. 3a legend inset, at two
    decimals, and had to be recomputed by hand each time.
  - The set-size-adjusted regression came from
    `.../stats/domain_diversity_regression.csv`, whose stage hardcodes the
    `xgb_tree_d2` experiment directory. The rest of the subsection is depth-3, so
    the reported interaction described a different rung from the correlations
    printed two sentences earlier.

Both quantities are computed here at `xgb_tree_d3`, set size <= 30, from the same
pool the co-occurrence and composition stages use, so the whole subsection
describes one candidate set.

A. Unadjusted association, per (BAG, arm): Pearson correlation of Shannon domain
   entropy H with LOCO R^2 across the 560 candidates of that arm. Two-sided.
   These are descriptive: candidates overlap and share their evaluation sample, so
   the classical p is anti-conservative and inference belongs to B.

B. Set-size-adjusted model, one per BAG:
       R^2 ~ H * is_synergistic + order
   Point estimates by OLS; inference on the H x arm interaction by standard errors
   clustered on `order` (set size), because candidates of the same set size share
   search sub-structure. The interaction is reported as the synergy-minus-
   redundancy difference in the H slope, which is the sign convention the
   manuscript states. Per-arm slopes are obtained by refitting with the arm
   indicator reversed, so each carries its own cluster-robust CI.

   Note on clusters: set sizes 3-30 give 28 clusters, at the low end for
   cluster-robust SE. The label-permutation test in
   `.../stats/domain_diversity_interaction_robust.csv` is the assumption-free
   companion; it is reported in Table S5 and not duplicated here.

Outputs (additive, checkout):
  outputs/dedup/model_comparison/diversity_correlations_d3.csv
  outputs/dedup/model_comparison/diversity_regression_d3.csv
  outputs/dedup/model_comparison/diversity_d3.md
"""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from repo_expo_hoi_bag.publication_stats.paths import (
    REPOSITORY_ROOT,
    model_comparison_output_dir,
    required_dedup_root,
)

REPO = REPOSITORY_ROOT
OUTDIR = model_comparison_output_dir()
# Effective-sample (dedup) pool; same file as cooccurrence_overlap_stats.py.
DEDUP = required_dedup_root()
POOL = DEDUP / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet"
DOMAIN_CSV = REPO / "data" / "metadata" / "exposome_feature_domains.csv"

BAGS = ["structural", "functional"]          # paper order
ARMS = [("o_max", "redundancy"), ("o_min", "synergy")]
RUNG = "xgb_tree_d3"
MAX_ORDER = 30


def _domain_map() -> dict[str, str]:
    dm = pd.read_csv(DOMAIN_CSV)
    return dict(zip(dm.feature_name.astype(str), dm.domain.astype(str)))


def _shannon_h(predictors_identity: str, dmap: dict[str, str]) -> float:
    feats = [f.strip() for f in str(predictors_identity).split("|") if f.strip()]
    counts = Counter(dmap.get(f, "UNKNOWN") for f in feats)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _bag_frame(pool: pd.DataFrame, bag: str, dmap: dict[str, str]) -> pd.DataFrame:
    d = pool[
        (pool.bag_target == bag) & (pool.rung_id == RUNG) & (pool.order <= MAX_ORDER)
    ].copy()
    d = d[np.isfinite(d.full_r2)]
    d["shannon_h"] = [_shannon_h(p, dmap) for p in d.predictors_identity]
    d["is_synergistic"] = (d.objective == "o_min").astype(int)
    d["is_redundant"] = 1 - d["is_synergistic"]
    return d


def main() -> None:
    dmap = _domain_map()
    pool = pd.read_parquet(POOL)

    corr_rows, reg_rows = [], []
    for bag in BAGS:
        d = _bag_frame(pool, bag, dmap)

        # ── A. unadjusted correlations, per arm ──────────────────────────────
        for objective, arm in ARMS:
            s = d[d.objective == objective]
            lr = stats.linregress(s.shannon_h, s.full_r2)
            corr_rows.append(
                {
                    "bag": bag,
                    "arm": arm,
                    "objective": objective,
                    "rung_id": RUNG,
                    "max_order": MAX_ORDER,
                    "n": len(s),
                    "pearson_r": round(float(lr.rvalue), 4),
                    "p_two_sided": float(lr.pvalue),
                    "slope_r2_per_bit": round(float(lr.slope), 6),
                }
            )

        # ── B. set-size-adjusted model, cluster-robust on set size ───────────
        fits = {}
        for indicator, base_arm in [("is_synergistic", "redundancy"), ("is_redundant", "synergy")]:
            fits[base_arm] = smf.ols(
                f"full_r2 ~ shannon_h * {indicator} + order", data=d
            ).fit(cov_type="cluster", cov_kwds={"groups": d.order})
        for base_arm, fit in fits.items():
            ci = fit.conf_int().loc["shannon_h"]
            reg_rows.append(
                {
                    "bag": bag,
                    "term": f"beta_H_{base_arm}_arm",
                    "estimate": round(float(fit.params["shannon_h"]), 6),
                    "ci_lo": round(float(ci[0]), 6),
                    "ci_hi": round(float(ci[1]), 6),
                    "p_cluster_order": float(fit.pvalues["shannon_h"]),
                    "n": int(fit.nobs),
                    "n_clusters": int(d.order.nunique()),
                }
            )
        fit = fits["redundancy"]                      # synergy coded 1 -> syn minus red
        term = "shannon_h:is_synergistic"
        ci = fit.conf_int().loc[term]
        reg_rows.append(
            {
                "bag": bag,
                "term": "interaction_H_x_arm_syn_minus_red",
                "estimate": round(float(fit.params[term]), 6),
                "ci_lo": round(float(ci[0]), 6),
                "ci_hi": round(float(ci[1]), 6),
                "p_cluster_order": float(fit.pvalues[term]),
                "n": int(fit.nobs),
                "n_clusters": int(d.order.nunique()),
            }
        )
        ols = smf.ols("full_r2 ~ shannon_h * is_synergistic + order", data=d).fit()
        reg_rows.append(
            {
                "bag": bag,
                "term": "model_r2",
                "estimate": round(float(ols.rsquared), 4),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "p_cluster_order": np.nan,
                "n": int(ols.nobs),
                "n_clusters": int(d.order.nunique()),
            }
        )

    corr = pd.DataFrame(corr_rows)
    reg = pd.DataFrame(reg_rows)
    corr.to_csv(OUTDIR / "diversity_correlations_d3.csv", index=False)
    reg.to_csv(OUTDIR / "diversity_regression_d3.csv", index=False)

    lines = ["# Domain diversity and LOCO R^2 at depth-3 (dedup-complete)\n"]
    lines.append(
        f"Dedup pool, rung `{RUNG}`, set size <= {MAX_ORDER}. Correlations are two-sided "
        "Pearson within arm and are descriptive; inference on the arm difference comes "
        "from the interaction term with standard errors clustered on set size.\n"
    )
    lines.append("\n## A. Unadjusted association (Fig. 3a)\n")
    lines.append("| BAG | Arm | n | Pearson r | p (two-sided) |")
    lines.append("|---|---|---|---|---|")
    for _, r in corr.iterrows():
        lines.append(f"| {r.bag} | {r.arm} | {r.n} | {r.pearson_r} | {r.p_two_sided:.4g} |")
    lines.append("\n## B. Set-size-adjusted model (Table S5)\n")
    lines.append("| BAG | Term | Estimate | 95% CI | p (cluster on set size) | n | clusters |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in reg.iterrows():
        ci = "—" if not np.isfinite(r.ci_lo) else f"{r.ci_lo} to {r.ci_hi}"
        pv = "—" if not np.isfinite(r.p_cluster_order) else f"{r.p_cluster_order:.4g}"
        lines.append(f"| {r.bag} | {r.term} | {r.estimate} | {ci} | {pv} | {r.n} | {r.n_clusters} |")
    (OUTDIR / "diversity_d3.md").write_text("\n".join(lines) + "\n")

    print(corr.to_string(index=False))
    print()
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
