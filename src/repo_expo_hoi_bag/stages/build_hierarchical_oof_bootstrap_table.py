#!/usr/bin/env python3
"""Consolidated publication table for the bootstrap incremental-R2 analysis.

Two worksheets: the prespecified Top20 family, and the complete K grid. Uses the
shared ``_write_table`` writer and never overwrites a previous supplementary
workbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from repo_expo_hoi_bag.figures.style import BAG_LABELS, LEVEL_LABELS
from repo_expo_hoi_bag.stages.build_publication_supplementary_tables import (
    PublicationTable,
    _write_table,
)
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_bootstrap import (
    K_GRID_NOTE,
    RESAMPLING_NOTE,
)
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import PRIMARY_REGION, ROOT

ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}

PRIMARY_NOTE = (
    "Incremental R² of the prespecified Top-20 candidate region over the matched covariate "
    f"baseline. The estimand is the unweighted mean across countries of "
    "(SSE_baseline − mean SSE_Top20)/SST, the country-balanced quantity used elsewhere in the "
    f"paper. {RESAMPLING_NOTE} P values are one-sided for H0: ΔR² ≤ 0 and are Holm-adjusted "
    "within each BAG across exactly the four synergy/redundancy × d2/d3 comparisons. No model "
    "was refitted for this analysis."
)

GRID_NOTE = (
    "The same estimand across the full region-size grid, from the five best-ranked candidates "
    f"to the complete BAG-blind pool. {K_GRID_NOTE} Only the Top-20 rows carry a Holm-adjusted "
    "P value; every other row is reported with its raw one-sided P. Candidate ranking is the "
    "original global LOCO ranking and is identical to the one used by the frontier analysis."
)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = (
        frame["bag"].map({"structural": 0, "functional": 1}) * 100
        + frame["rung"].map({"xgb_tree_d2": 0, "xgb_tree_d3": 10})
        + frame["arm"].map({"o_min": 0, "o_max": 1})
    )
    return frame.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order")


def build_primary_frame(results: pd.DataFrame) -> pd.DataFrame:
    primary = _ordered(results[results["region"].eq(PRIMARY_REGION)])
    return pd.DataFrame(
        {
            "BAG": primary["bag"].map(BAG_LABELS),
            "Arm": primary["arm"].map(ARM_LABEL),
            "Model level": primary["rung"].map(LEVEL_LABELS),
            "Candidate region": primary["region"],
            "ΔR² over baseline": primary["observed_delta_r2"],
            "95% CI": primary.apply(
                lambda row: f"{row['ci_lo']:.4f} to {row['ci_hi']:.4f}", axis=1
            ),
            "One-sided P": primary["p_one_sided"],
            "Holm P": primary["holm_p"],
            "P(ΔR² > 0)": primary["fraction_bootstrap_positive"],
            "n countries": primary["n_countries"],
            "n subjects": primary["n_subjects"],
        }
    ).reset_index(drop=True)


def build_grid_frame(results: pd.DataFrame) -> pd.DataFrame:
    frame = _ordered(results).copy()
    # ALL sorts last within a cell; the Top-K rows sort by K ascending.
    frame["_control"] = frame["region"].eq("ALL").astype(int)
    frame["_k"] = frame["region"].map(
        lambda region: 0 if region == "ALL" else int(region.removeprefix("Top"))
    )
    frame = frame.sort_values(
        ["bag", "rung", "arm", "_control", "_k"], kind="stable"
    )
    return pd.DataFrame(
        {
            "BAG": frame["bag"].map(BAG_LABELS),
            "Arm": frame["arm"].map(ARM_LABEL),
            "Model level": frame["rung"].map(LEVEL_LABELS),
            "Candidate region": frame["region"],
            "Role": frame["region"].map(
                lambda region: "Primary" if region == PRIMARY_REGION
                else "Control" if region == "ALL" else "Sensitivity"
            ),
            "ΔR² over baseline": frame["observed_delta_r2"],
            "95% CI": frame.apply(
                lambda row: f"{row['ci_lo']:.4f} to {row['ci_hi']:.4f}", axis=1
            ),
            "One-sided P": frame["p_one_sided"],
            "Holm P": frame["holm_p"],
            "P(ΔR² > 0)": frame["fraction_bootstrap_positive"],
            "n countries": frame["n_countries"],
        }
    ).reset_index(drop=True)


def build(results_path: Path, source_data: Path, workbook_path: Path) -> tuple[Path, Path]:
    results = pd.read_csv(results_path)
    primary = build_primary_frame(results)
    grid = build_grid_frame(results)

    source_data.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(source_data, index=False)
    grid.to_csv(source_data.with_name("hierarchical_oof_bootstrap_k_grid.csv"), index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(
            27,
            f"{PRIMARY_REGION} candidate region versus covariate baseline, incremental R²",
            primary,
            PRIMARY_NOTE,
        ),
    )
    _write_table(
        workbook,
        PublicationTable(
            28, "Incremental R² across the candidate region-size grid", grid, GRID_NOTE
        ),
    )
    workbook["ST27"].title = "Top20 primary"
    workbook["ST28"].title = "K-grid sensitivity"

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=root / "hierarchical_oof_bootstrap_results.csv"
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/hierarchical_oof/hierarchical_oof_bootstrap_top20.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Hierarchical_OOF_Bootstrap.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.results, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
