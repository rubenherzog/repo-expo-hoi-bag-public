#!/usr/bin/env python3
"""Publication table for the country-cross-selected performance-frontier tests.

Uses the shared ``_write_table`` writer, so font, rules, number formats, note,
freeze panes and page setup match every other supplementary table. Writes a
standalone workbook with two worksheets -- the comparisons and a complete
frontier-stability table across K=10/20/50 -- and never touches the canonical
``Supplementary_Tables_main_k10.xlsx``.
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
from repo_expo_hoi_bag.stages.compute_performance_frontier import (
    NESTING_CAVEAT,
    PRIMARY_K,
    ROOT,
)

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {
    "o_min": "Synergy arm",
    "o_max": "Redundancy arm",
    "o_min_minus_o_max": "Both arms",
}
LEVEL_LABEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

SECTIONS = (
    ("Synergy vs redundancy frontier", "synergy_vs_redundancy", None),
    ("Model complexity — synergy arm", "model_complexity", "o_min"),
    ("Model complexity — redundancy arm", "model_complexity", "o_max"),
    ("Synergy frontier vs baseline", "vs_covariate_baseline", "o_min"),
    ("Redundancy frontier vs baseline", "vs_covariate_baseline", "o_max"),
    ("Synergy frontier vs cross-selected best single", "vs_cross_selected_single", "o_min"),
    ("Redundancy frontier vs cross-selected best single", "vs_cross_selected_single", "o_max"),
)

NOTE = (
    f"The performance frontier F[c, arm, level, K={PRIMARY_K}] is the median already-computed "
    "held-out R² in country c across the K candidates ranked highest by their unweighted mean "
    "R² over all countries except c. The evaluation country is therefore never used to rank its "
    "own frontier. The best single exposure is cross-selected the same way. Inference is paired "
    "across held-out countries: a two-sided Wilcoxon signed-rank test, 95% confidence intervals "
    "from a 10,000-draw paired country bootstrap, and rank-biserial correlation as the effect "
    "size. Holm adjustment is applied within the families stated in the Comparison family "
    "column: four model levels for the arm contrast, six pairwise level contrasts within each "
    "BAG and arm, and eight arm × level contrasts within each BAG for the baseline and "
    "single-exposure comparisons. Candidate Jaccard columns compare each leave-one-country-out "
    "frontier with the global Top-20 by candidate identity, not by exposure. "
    f"{NESTING_CAVEAT} No models were fitted for this table."
)

STABILITY_NOTE = (
    "Candidate Jaccard compares the set of candidate identities in each leave-one-country-out "
    "Top-K frontier with the global Top-K; exposure Jaccard compares the unions of exposures "
    "appearing in those candidate sets and is a separate, more permissive quantity. Summaries "
    "are taken across held-out countries. Rank-1 minus rank-K is the spread in mean held-out-"
    "country R² across the global frontier, so a small value means a broad near-equivalent "
    "plateau. Retained is the mean number of global Top-K candidates still present in a "
    "leave-one-country-out frontier."
)


def _comparison_label(row: pd.Series) -> str:
    family = str(row["comparison_family"])
    level_a = LEVEL_LABEL.get(str(row["rung_a"]), row["rung_a"])
    level_b = LEVEL_LABEL.get(str(row["rung_b"]), row["rung_b"])
    if family == "synergy_vs_redundancy":
        return f"Synergy − redundancy at {level_a}"
    if family == "model_complexity":
        return f"{level_a} − {level_b}"
    if family == "vs_covariate_baseline":
        return f"Frontier − baseline at {level_a}"
    return f"Frontier − best single at {level_a}"


def _jaccard_lookup(summary: pd.DataFrame, k: int = PRIMARY_K) -> dict[tuple[str, str, str], tuple[float, float]]:
    return {
        (str(row.bag), str(row.arm), str(row.rung)): (
            float(getattr(row, f"top{k}_candidate_jaccard_mean")),
            float(getattr(row, f"top{k}_candidate_jaccard_median")),
        )
        for row in summary.itertuples()
    }


def build_frame(tests: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """One ordered, publication-formatted row per K=20 contrast."""
    lookup = _jaccard_lookup(summary)
    blocks: list[pd.DataFrame] = []
    for bag in ("structural", "functional"):
        for label, family, arm in SECTIONS:
            selected = tests[tests["bag"].eq(bag) & tests["comparison_family"].eq(family)]
            if arm is not None:
                selected = selected[selected["arm"].eq(arm)]
            if selected.empty:
                continue
            blocks.append(selected.assign(_section=label))
    ordered = pd.concat(blocks, ignore_index=True)

    # For an arm contrast both frontiers are involved, so the stability of each
    # is reported in its own column; elsewhere the single frontier's values are
    # repeated into the synergy/redundancy column that applies.
    synergy_mean, synergy_median, redundancy_mean, redundancy_median = [], [], [], []
    for row in ordered.itertuples(index=False):
        bag, rung = str(row.bag), str(row.rung_a)
        if row.comparison_family == "synergy_vs_redundancy":
            s_mean, s_median = lookup[(bag, "o_min", rung)]
            r_mean, r_median = lookup[(bag, "o_max", rung)]
        else:
            arm = str(row.arm)
            mean, median = lookup[(bag, arm, rung)]
            s_mean, s_median = (mean, median) if arm == "o_min" else (None, None)
            r_mean, r_median = (mean, median) if arm == "o_max" else (None, None)
        synergy_mean.append(s_mean)
        synergy_median.append(s_median)
        redundancy_mean.append(r_mean)
        redundancy_median.append(r_median)

    return pd.DataFrame(
        {
            "BAG": ordered["bag"].map(BAG_LABEL),
            "Comparison family": ordered["_section"],
            "Arm": ordered["arm"].map(ARM_LABEL).fillna(ordered["arm"]),
            "Comparison": ordered.apply(_comparison_label, axis=1),
            "Countries": ordered["n_countries"],
            "R² A": ordered["estimate_a"],
            "R² B": ordered["estimate_b"],
            "ΔR²": ordered["delta_mean_r2"],
            "95% CI": ordered.apply(lambda r: f"{r['ci_lo']:.3f} to {r['ci_hi']:.3f}", axis=1),
            "Two-sided P": ordered["wilcoxon_p_raw"],
            "Holm-adjusted P": ordered["holm_p"],
            "Rank-biserial effect size": ordered["rank_biserial"],
            "Synergy Top-20 candidate Jaccard, mean": synergy_mean,
            "Synergy Top-20 candidate Jaccard, median": synergy_median,
            "Redundancy Top-20 candidate Jaccard, mean": redundancy_mean,
            "Redundancy Top-20 candidate Jaccard, median": redundancy_median,
        }
    )


def build_stability_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """The complete Jaccard table across K=10, 20 and 50."""
    frame = pd.DataFrame(
        {
            "BAG": summary["bag"].map(BAG_LABEL),
            "Arm": summary["arm"].map(ARM_LABEL),
            "Model level": summary["rung"].map(LEVEL_LABEL),
            "Candidates": summary["n_candidates"],
            "Countries": summary["n_countries"],
        }
    )
    for k in (10, 20, 50):
        frame[f"Top-{k} candidate Jaccard, mean"] = summary[f"top{k}_candidate_jaccard_mean"]
        frame[f"Top-{k} candidate Jaccard, median"] = summary[f"top{k}_candidate_jaccard_median"]
        frame[f"Top-{k} candidate Jaccard, Q1"] = summary[f"top{k}_candidate_jaccard_q1"]
        frame[f"Top-{k} candidate Jaccard, Q3"] = summary[f"top{k}_candidate_jaccard_q3"]
    for k in (10, 20, 50):
        frame[f"Top-{k} exposure Jaccard, mean"] = summary[f"top{k}_exposure_jaccard_mean"]
        frame[f"Top-{k} exposure Jaccard, median"] = summary[f"top{k}_exposure_jaccard_median"]
    for k in (10, 20, 50):
        frame[f"Top-{k} rank-1 minus rank-K R²"] = summary[f"top{k}_global_rank1_minus_rankK_r2"]
        frame[f"Top-{k} global candidates retained, mean"] = summary[f"top{k}_n_global_retained_mean"]
    order = frame["BAG"].map({"Structural": 0, "Functional": 1})
    return frame.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def build(tests_path: Path, summary_path: Path, source_data: Path, workbook_path: Path) -> tuple[Path, Path]:
    tests = pd.read_csv(tests_path)
    summary = pd.read_csv(summary_path)

    frame = build_frame(tests, summary)
    stability = build_stability_frame(summary)

    source_data.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_data, index=False)
    stability.to_csv(source_data.with_name("performance_frontier_stability.csv"), index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(20, "Country-cross-selected performance-frontier comparisons", frame, NOTE),
    )
    _write_table(
        workbook,
        PublicationTable(21, "Frontier stability across frontier sizes", stability, STABILITY_NOTE),
    )
    # The second sheet is referred to by name in the manuscript.
    workbook["ST21"].title = "Frontier stability"

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    frontier = ROOT / "outputs/sensitivity/performance_frontier/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=frontier / "frontier_comparison_tests.csv")
    parser.add_argument("--stability", type=Path, default=frontier / "frontier_stability_summary.csv")
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/performance_frontier/performance_frontier_comparisons.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Performance_Frontier_Comparisons.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.tests, args.stability, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
