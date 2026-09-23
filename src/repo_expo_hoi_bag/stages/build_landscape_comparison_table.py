#!/usr/bin/env python3
"""Publication table for the landscape-level model comparisons.

Formats the audit CSV written by ``compute_landscape_model_comparisons`` as a
supplementary table using the existing publication conventions (the shared
``_write_table`` writer, so font, rules, number formats, note, freeze panes and
page setup are identical to every other supplementary table).

It writes a standalone workbook and never touches
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
from repo_expo_hoi_bag.stages.compute_landscape_model_comparisons import (
    AGGREGATION_DEFINITION,
    ROOT,
)

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {"o_min": "Synergy arm", "o_max": "Redundancy arm", "o_domain": "Domain arm"}
LEVEL_LABEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

# The manuscript's row order. Each entry selects a block of rows and gives it
# the section label the table groups by.
SECTIONS = (
    ("Synergy vs redundancy", "synergy_vs_redundancy", None),
    ("Model complexity within synergy", "model_complexity", "o_min"),
    ("Model complexity within redundancy", "model_complexity", "o_max"),
    ("Synergy landscape vs single-exposure landscape", "vs_single_landscape", "o_min"),
    ("Redundancy landscape vs single-exposure landscape", "vs_single_landscape", "o_max"),
    ("Synergy landscape vs covariate baseline", "vs_covariate_baseline", "o_min"),
    ("Redundancy landscape vs covariate baseline", "vs_covariate_baseline", "o_max"),
)

NOTE = (
    "The landscape R² of a held-out country is the median R² across the 20 BAG-blind "
    "candidates at each set size k, averaged with equal weight over k=3–30; every set "
    "size contributes equally and no candidate is ever selected on its BAG performance. "
    "The single-exposure landscape is the median R² across all 63 single-exposure "
    "candidates in that country, not the best single exposure. The baseline is the "
    "unique covariate-only model. Inference is paired across held-out countries: a "
    "two-sided Wilcoxon signed-rank test across countries, with 95% confidence intervals "
    "from a 10,000-draw paired country bootstrap and rank-biserial correlation as the "
    "effect size. Holm adjustment is applied within the families stated in the "
    "Comparison family column: the four model levels for the arm contrast, the six "
    "pairwise level contrasts within each BAG and arm, and the eight arm × level "
    "contrasts within each BAG for the single-exposure and baseline comparisons. "
    "No models were fitted for this table; all values aggregate cached "
    "held-out-country candidate metrics."
)


def _comparison_label(row: pd.Series) -> str:
    """The human-readable contrast, always stated as A minus B."""
    family = str(row["comparison_family"])
    level_a, level_b = LEVEL_LABEL.get(str(row["rung_a"]), row["rung_a"]), LEVEL_LABEL.get(
        str(row["rung_b"]), row["rung_b"]
    )
    if family == "synergy_vs_redundancy":
        return f"Synergy − redundancy at {level_a}"
    if family == "model_complexity":
        return f"{level_a} − {level_b}"
    if family == "vs_single_landscape":
        return f"Multivariate − single-exposure at {level_a}"
    return f"Multivariate − baseline at {level_a}"


def build_frame(tests: pd.DataFrame) -> pd.DataFrame:
    """One ordered, publication-formatted row per statistical contrast."""
    blocks: list[pd.DataFrame] = []
    for bag in ("structural", "functional"):
        for label, family, objective in SECTIONS:
            selected = tests[tests["bag"].eq(bag) & tests["comparison_family"].eq(family)]
            if objective is not None:
                selected = selected[selected["objective"].eq(objective)]
            if selected.empty:
                continue
            blocks.append(selected.assign(_section=label))

    ordered = pd.concat(blocks, ignore_index=True)
    return pd.DataFrame(
        {
            "BAG": ordered["bag"].map(BAG_LABEL),
            "Comparison family": ordered["_section"],
            "Discovery arm": ordered["objective"].map(ARM_LABEL).fillna("Both arms"),
            "Comparison": ordered.apply(_comparison_label, axis=1),
            "Countries": ordered["n_countries"],
            "Landscape R² A": ordered["estimate_a"],
            "Landscape R² B": ordered["estimate_b"],
            "ΔR²": ordered["delta_mean_r2"],
            "95% CI": ordered.apply(lambda r: f"{r['ci_lo']:.3f} to {r['ci_hi']:.3f}", axis=1),
            "Wilcoxon P": ordered["wilcoxon_p_raw"],
            "Holm-adjusted P": ordered["holm_p"],
            "Rank-biserial effect size": ordered["rank_biserial"],
        }
    )


def build(tests_path: Path, source_data: Path, workbook_path: Path) -> tuple[Path, Path]:
    tests = pd.read_csv(tests_path)
    frame = build_frame(tests)

    source_data.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_data, index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(
            19,
            "Landscape-level model comparisons",
            frame,
            NOTE,
        ),
    )
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        type=Path,
        default=ROOT / "outputs/sensitivity/landscape_comparisons/main_k10/landscape_comparison_tests.csv",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/landscape_comparisons/landscape_model_comparisons.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Landscape_Model_Comparisons.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.tests, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
