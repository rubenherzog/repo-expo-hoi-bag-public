#!/usr/bin/env python3
"""Publication table for the cross-selected exposome vs covariate baseline test.

Uses the shared ``_write_table`` writer, so the visual conventions match every
other supplementary table. Writes a standalone workbook and never touches
``Supplementary_Tables_main_k10.xlsx`` or the existing Supplementary Table 2.
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
from repo_expo_hoi_bag.stages.compute_cross_selected_baseline import (
    NESTING_CAVEAT,
    PRIMARY_RUNGS,
    ROOT,
)

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {"o_min": "Synergy arm", "o_max": "Redundancy arm"}
LEVEL_LABEL = {"xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

NOTE = (
    "For each held-out country the exposome candidate is chosen by its unweighted mean "
    "R² across the other countries only, so the evaluation country never contributes to "
    "its own candidate ranking; that candidate's out-of-fold predictions for that "
    "country's participants are then compared with the canonical covariate-only "
    "baseline. R² is the unweighted mean of per-country R², the estimand used "
    "throughout these supplementary tables; the pooled out-of-fold R² is reported in "
    "the audit CSV as a secondary statistic. Confidence intervals and two-sided P "
    "values come from a 10,000-draw paired country-cluster bootstrap that resamples "
    "whole countries. Holm adjustment is applied within each BAG across the four "
    "primary d2/d3 comparisons; d1 is a nonlinear-additive control held in a separate "
    "descriptive family. Candidate stability is the fraction of countries whose "
    "cross-selected candidate equals the globally selected winner, and the Top-20 "
    "Jaccard is the mean candidate-identity overlap of the leave-one-country-out "
    f"frontier with the global Top-20. {NESTING_CAVEAT} No candidate discovery was "
    "performed; the folds whose cross-selected candidate had no cached predictions were "
    "refit with the production fold builder, frozen hyperparameters and per-fold seeds."
)


def build_frame(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[summary["rung"].isin(PRIMARY_RUNGS)].copy()
    order = (
        primary["bag"].map({"structural": 0, "functional": 1}) * 100
        + primary["rung"].map({"xgb_tree_d2": 0, "xgb_tree_d3": 10})
        + primary["arm"].map({"o_min": 0, "o_max": 1})
    )
    primary = primary.assign(_order=order).sort_values("_order", kind="stable")

    return pd.DataFrame(
        {
            "BAG": primary["bag"].map(BAG_LABEL),
            "Arm": primary["arm"].map(ARM_LABEL),
            "Model level": primary["rung"].map(LEVEL_LABEL),
            "Cross-selected R²": primary["selected_model_country_balanced_r2"],
            "Baseline R²": primary["baseline_country_balanced_r2"],
            "ΔR²": primary["delta_r2"],
            "95% CI": primary.apply(
                lambda row: f"{row['bootstrap_ci_lo']:.3f} to {row['bootstrap_ci_hi']:.3f}", axis=1
            ),
            "Two-sided P": primary["bootstrap_p_raw"],
            "Holm-adjusted P": primary["bootstrap_p_holm"],
            "Cohen's f²": primary["cohen_f2"],
            "Positive countries": primary.apply(
                lambda row: f"{int(row['n_countries_delta_positive'])}/{int(row['n_countries'])}",
                axis=1,
            ),
            "Candidate stability": primary["same_as_global_winner_fraction"],
            "Top-20 Jaccard, mean": primary["top20_candidate_jaccard_mean"],
        }
    ).reset_index(drop=True)


def build(summary_path: Path, source_data: Path, workbook_path: Path) -> tuple[Path, Path]:
    frame = build_frame(pd.read_csv(summary_path))
    source_data.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_data, index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(
            22, "Country-cross-selected exposome model versus covariate baseline", frame, NOTE
        ),
    )
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    root = ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=root / "cross_selected_baseline_summary.csv")
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/cross_selected_baseline/cross_selected_exposome_vs_baseline.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Cross_Selected_Exposome_vs_Baseline.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.summary, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
