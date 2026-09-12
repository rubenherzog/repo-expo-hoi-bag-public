#!/usr/bin/env python3
"""Reproduce the paper's depth-3 domain-diversity statistics with k10 metrics.

Port of ``publication_stats/domain_diversity.py`` from the paper repository.
Only the input changes: ``country_balanced_r2`` comes from the new HPO
re-analysis, rather than ``full_r2`` from the immutable paper runtime.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
BAGS = ("structural", "functional")
ARMS = (("o_max", "redundancy"), ("o_min", "synergy"))
RUNG = "xgb_tree_d3"
MAX_ORDER = 30


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    return parser.parse_args()


def _domain_map() -> dict[str, str]:
    path = ROOT / "data" / "metadata" / "exposome_feature_domains.csv"
    domains = pd.read_csv(path)
    return dict(zip(domains.feature_name.astype(str), domains.domain.astype(str)))


def _shannon_h(predictors_identity: str, domain_map: dict[str, str]) -> float:
    features = [feature.strip() for feature in str(predictors_identity).split("|") if feature.strip()]
    counts = Counter(domain_map.get(feature, "UNKNOWN") for feature in features)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _bag_frame(metrics: pd.DataFrame, bag: str, domain_map: dict[str, str]) -> pd.DataFrame:
    frame = metrics[
        (metrics["bag"] == bag)
        & (metrics["rung"] == RUNG)
        & (pd.to_numeric(metrics["order"], errors="raise") <= MAX_ORDER)
    ].copy()
    frame["full_r2"] = pd.to_numeric(frame["country_balanced_r2"], errors="raise")
    frame = frame[np.isfinite(frame["full_r2"])].copy()
    frame["shannon_h"] = [_shannon_h(value, domain_map) for value in frame["predictors_identity"]]
    frame["is_synergistic"] = (frame["objective"] == "o_min").astype(int)
    frame["is_redundant"] = 1 - frame["is_synergistic"]
    if len(frame) != 1120:
        raise ValueError(f"Expected 1,120 d3 candidates at order <=30 for {bag}, got {len(frame)}")
    return frame


def _compute(metrics: pd.DataFrame, domain_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    for bag in BAGS:
        frame = _bag_frame(metrics, bag, domain_map)
        for objective, arm in ARMS:
            subset = frame[frame["objective"] == objective]
            fit = stats.linregress(subset.shannon_h, subset.full_r2)
            correlation_rows.append({
                "bag": bag, "arm": arm, "objective": objective, "rung_id": RUNG,
                "max_order": MAX_ORDER, "n": len(subset), "pearson_r": round(float(fit.rvalue), 4),
                "p_two_sided": float(fit.pvalue), "slope_r2_per_bit": round(float(fit.slope), 6),
            })

        fits: dict[str, object] = {}
        for indicator, base_arm in (("is_synergistic", "redundancy"), ("is_redundant", "synergy")):
            fits[base_arm] = smf.ols(
                f"full_r2 ~ shannon_h * {indicator} + order", data=frame
            ).fit(cov_type="cluster", cov_kwds={"groups": frame.order})
        for base_arm, fit in fits.items():
            confidence_interval = fit.conf_int().loc["shannon_h"]
            regression_rows.append({
                "bag": bag, "term": f"beta_H_{base_arm}_arm", "estimate": round(float(fit.params["shannon_h"]), 6),
                "ci_lo": round(float(confidence_interval[0]), 6), "ci_hi": round(float(confidence_interval[1]), 6),
                "p_cluster_order": float(fit.pvalues["shannon_h"]), "n": int(fit.nobs),
                "n_clusters": int(frame.order.nunique()),
            })
        interaction_fit = fits["redundancy"]
        interaction_term = "shannon_h:is_synergistic"
        confidence_interval = interaction_fit.conf_int().loc[interaction_term]
        regression_rows.append({
            "bag": bag, "term": "interaction_H_x_arm_syn_minus_red",
            "estimate": round(float(interaction_fit.params[interaction_term]), 6),
            "ci_lo": round(float(confidence_interval[0]), 6), "ci_hi": round(float(confidence_interval[1]), 6),
            "p_cluster_order": float(interaction_fit.pvalues[interaction_term]), "n": int(interaction_fit.nobs),
            "n_clusters": int(frame.order.nunique()),
        })
        ordinary_fit = smf.ols("full_r2 ~ shannon_h * is_synergistic + order", data=frame).fit()
        regression_rows.append({
            "bag": bag, "term": "model_r2", "estimate": round(float(ordinary_fit.rsquared), 4),
            "ci_lo": np.nan, "ci_hi": np.nan, "p_cluster_order": np.nan, "n": int(ordinary_fit.nobs),
            "n_clusters": int(frame.order.nunique()),
        })
    return pd.DataFrame(correlation_rows), pd.DataFrame(regression_rows)


def main() -> None:
    args = _args()
    stats_dir = args.repro_data_root.resolve() / "results" / "analysis_runs" / "paper_reanalysis_k10" / "main_statistics" / "fig3_diversity"
    metrics_path = stats_dir / "candidate_country_balanced_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Run plot_hpo_fig3_diversity first: {metrics_path}")
    correlations, regressions = _compute(pd.read_csv(metrics_path), _domain_map())
    correlations.to_csv(stats_dir / "diversity_correlations_d3.csv", index=False)
    regressions.to_csv(stats_dir / "diversity_regression_d3.csv", index=False)
    print(correlations.to_string(index=False))
    print(regressions.to_string(index=False))


if __name__ == "__main__":
    main()
