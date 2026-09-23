#!/usr/bin/env python3
"""Publication table for the Top-K frontier versus covariate baseline test.

Two worksheets: the primary K=20 comparisons and the K=10/20/50 sensitivity.
Uses the shared ``_write_table`` writer and never overwrites a previous
supplementary workbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from repo_expo_hoi_bag.stages.build_publication_supplementary_tables import (
    PublicationTable,
    _write_table,
)
from repo_expo_hoi_bag.stages.compute_frontier_vs_baseline import (
    DEPENDENCE_NOTE,
    K_VALUES,
    MEDIAN_EQUIVALENCE_NOTE,
    PRIMARY_K,
    ROOT,
)

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
LEVEL_LABEL = {"xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

NOTE = (
    f"For each held-out country the Top-{PRIMARY_K} candidates are selected by their mean R² "
    "across the other countries only, and each member's incremental R² over the matched "
    "canonical covariate baseline is computed in that country. The primary statistic is the "
    "country-level mean of those incremental values, so it reflects the expected gain of the "
    "high-performing region as a whole rather than of its best member. "
    f"{DEPENDENCE_NOTE} Confidence intervals come from a 10,000-draw country bootstrap and "
    "P values from a two-sided Wilcoxon signed-rank test of the country-level means against "
    "zero, Holm-adjusted within each BAG across exactly the four synergy/redundancy × d2/d3 "
    "comparisons; OLS and d1 are excluded from that family. The percentage of frontier "
    "candidates above baseline is descriptive. No model fitting was performed."
)

SENSITIVITY_NOTE = (
    "Mean frontier gain and positivity at three frontier sizes. K=20 is the primary analysis; "
    "K=10 and K=50 are sensitivity analyses that place the frontier cutoff tighter or looser. "
    f"{MEDIAN_EQUIVALENCE_NOTE}"
)


def build_primary_frame(tests: pd.DataFrame, stability: pd.DataFrame | None) -> pd.DataFrame:
    positivity_columns = [
        "mean_positive_fraction", "median_positive_fraction",
        "n_countries_majority_positive", "n_countries_all_positive",
    ]
    # The two test kinds share one long table, so the positivity columns exist
    # on the primary rows as NaN. Drop them there or the merge suffixes them.
    primary = tests[tests["test"].eq("primary_mean_frontier_gain") & tests["K"].eq(PRIMARY_K)].drop(
        columns=positivity_columns, errors="ignore"
    )
    positivity = tests[tests["test"].eq("secondary_positivity_vs_half") & tests["K"].eq(PRIMARY_K)]
    merged = primary.merge(
        positivity[["bag", "arm", "rung", *positivity_columns]],
        on=["bag", "arm", "rung"],
        validate="one_to_one",
    )
    if stability is not None:
        merged = merged.merge(
            stability[["bag", "arm", "rung", "top20_candidate_jaccard_mean"]],
            on=["bag", "arm", "rung"],
            how="left",
        )
    else:
        merged["top20_candidate_jaccard_mean"] = pd.NA

    order = (
        merged["bag"].map({"structural": 0, "functional": 1}) * 100
        + merged["rung"].map({"xgb_tree_d2": 0, "xgb_tree_d3": 10})
        + merged["arm"].map({"o_min": 0, "o_max": 1})
    )
    merged = merged.assign(_order=order).sort_values("_order", kind="stable")

    return pd.DataFrame(
        {
            "BAG": merged["bag"].map(BAG_LABEL),
            "Arm": merged["arm"].map(ARM_LABEL),
            "Level": merged["rung"].map(LEVEL_LABEL),
            "K": merged["K"],
            "Mean frontier ΔR²": merged["mean_country_frontier_delta"],
            "95% CI": merged.apply(lambda r: f"{r['ci_lo']:.3f} to {r['ci_hi']:.3f}", axis=1),
            "Wilcoxon P": merged["wilcoxon_p_raw"],
            "Holm-adjusted P": merged["holm_p"],
            "Rank-biserial": merged["rank_biserial"],
            "Mean % frontier > baseline": merged["mean_positive_fraction"] * 100,
            "Median % > baseline": merged["median_positive_fraction"] * 100,
            "Countries >50% above": merged.apply(
                lambda r: f"{int(r['n_countries_majority_positive'])}/{int(r['n_countries'])}", axis=1
            ),
            "Countries 100% above": merged.apply(
                lambda r: f"{int(r['n_countries_all_positive'])}/{int(r['n_countries'])}", axis=1
            ),
            "Top-K Jaccard, mean": merged["top20_candidate_jaccard_mean"],
        }
    ).reset_index(drop=True)


def build_sensitivity_frame(sensitivity: pd.DataFrame) -> pd.DataFrame:
    order = (
        sensitivity["bag"].map({"structural": 0, "functional": 1}) * 100
        + sensitivity["rung"].map({"xgb_tree_d2": 0, "xgb_tree_d3": 10})
        + sensitivity["arm"].map({"o_min": 0, "o_max": 1})
    )
    frame = sensitivity.assign(_order=order).sort_values("_order", kind="stable")
    out = pd.DataFrame(
        {
            "BAG": frame["bag"].map(BAG_LABEL),
            "Arm": frame["arm"].map(ARM_LABEL),
            "Level": frame["rung"].map(LEVEL_LABEL),
        }
    )
    for k in K_VALUES:
        out[f"Mean ΔR², K={k}"] = frame[f"mean_country_frontier_delta_K{k}"].to_numpy()
    for k in K_VALUES:
        out[f"Holm P, K={k}"] = frame[f"holm_p_K{k}"].to_numpy()
    for k in K_VALUES:
        out[f"Mean % > baseline, K={k}"] = frame[f"mean_positive_fraction_K{k}"].to_numpy() * 100
    for k in K_VALUES:
        out[f"Countries >50% above, K={k}"] = frame[f"n_countries_majority_positive_K{k}"].to_numpy()
    out["Sign changes across K"] = frame["sign_changes_across_k"].to_numpy()
    out["Significance changes across K"] = frame["significance_changes_across_k"].to_numpy()
    return out.reset_index(drop=True)


def build(
    tests_path: Path, sensitivity_path: Path, source_data: Path, workbook_path: Path
) -> tuple[Path, Path]:
    tests = pd.read_csv(tests_path)
    sensitivity = pd.read_csv(sensitivity_path)
    stability_path = (
        ROOT / "outputs/sensitivity/performance_frontier/main_k10/frontier_stability_summary.csv"
    )
    stability = pd.read_csv(stability_path) if stability_path.is_file() else None

    primary = build_primary_frame(tests, stability)
    sensitivity_frame = build_sensitivity_frame(sensitivity)

    source_data.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(source_data, index=False)
    sensitivity_frame.to_csv(source_data.with_name("frontier_vs_baseline_K_sensitivity.csv"), index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(
            23, f"Top-{PRIMARY_K} candidate frontier versus covariate baseline", primary, NOTE
        ),
    )
    _write_table(
        workbook,
        PublicationTable(24, "Frontier gain across frontier sizes", sensitivity_frame, SENSITIVITY_NOTE),
    )
    workbook["ST24"].title = "K sensitivity"

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    root = ROOT / "outputs/sensitivity/frontier_vs_baseline/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=root / "frontier_vs_baseline_tests.csv")
    parser.add_argument(
        "--sensitivity", type=Path, default=root / "frontier_vs_baseline_K_sensitivity.csv"
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/frontier_vs_baseline/frontier_vs_baseline_comparisons.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Frontier_vs_Baseline.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.tests, args.sensitivity, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
