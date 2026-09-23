#!/usr/bin/env python3
"""Publication table for the hierarchical OOF improvement analysis.

Primary worksheet: the eight Top-20 hierarchical tests. Second worksheet: the
ALL/Top50/Top20/Top10 regions side by side. Uses the shared ``_write_table``
writer and never overwrites a previous supplementary workbook.
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
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import (
    CONDITIONAL_CAVEAT,
    DEPENDENCE_NOTE,
    PRIMARY_REGION,
    REGION_ORDER,
    ROOT,
    SCALE_NOTE,
)

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
LEVEL_LABEL = {"xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

NOTE = (
    "Individual out-of-fold prediction improvement over the covariate baseline, for the "
    f"{PRIMARY_REGION} region of the globally ranked candidate landscape. For each participant the "
    "improvement is the baseline squared prediction error minus the mean squared prediction "
    "error across the region's candidates; candidate predictions are never averaged. "
    f"{DEPENDENCE_NOTE} The estimate is the intercept of a REML random-intercept model with "
    "country as the grouping factor. Two-sided P values are Holm-adjusted within each BAG "
    "across exactly the four synergy/redundancy x d2/d3 tests; OLS and d1 are excluded and the "
    f"ALL control is not part of the family. {SCALE_NOTE} {CONDITIONAL_CAVEAT}"
)

SENSITIVITY_NOTE = (
    "The same hierarchical estimate across nested candidate regions. Top10 c Top20 c Top50 c ALL, "
    "verified explicitly. Top20 is primary; Top10 and Top50 are sensitivity analyses corrected "
    "within their own four-test family; ALL is the BAG-blind control and carries no Holm-adjusted "
    "P value. Observed mean delta-R2 is a descriptive landscape quantity on the R2 scale and is "
    f"not the hierarchical estimate. {SCALE_NOTE}"
)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = (
        frame["bag"].map({"structural": 0, "functional": 1}) * 100
        + frame["rung"].map({"xgb_tree_d2": 0, "xgb_tree_d3": 10})
        + frame["arm"].map({"o_min": 0, "o_max": 1})
    )
    return frame.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order")


def build_primary_frame(results: pd.DataFrame, landscape: pd.DataFrame) -> pd.DataFrame:
    keys = ["bag", "arm", "rung", "region"]
    merged = results[results["region"].eq(PRIMARY_REGION)].merge(
        landscape[[*keys, "n_candidates", "mean_delta_r2", "fraction_delta_positive"]],
        on=keys,
        validate="one_to_one",
    )
    merged = _ordered(merged)
    return pd.DataFrame(
        {
            "BAG": merged["bag"].map(BAG_LABEL),
            "Arm": merged["arm"].map(ARM_LABEL),
            "Model level": merged["rung"].map(LEVEL_LABEL),
            "Candidate region": merged["region"],
            "Candidate count": merged["n_candidates"],
            "Observed mean ΔR²": merged["mean_delta_r2"],
            "Fraction candidates ΔR² > 0": merged["fraction_delta_positive"],
            "Hierarchical mean improvement β0": merged["beta_intercept"],
            "95% CI": merged.apply(
                lambda row: f"{row['ci_lo']:.4f} to {row['ci_hi']:.4f}", axis=1
            ),
            "Two-sided P": merged["p_two_sided"],
            "Holm P": merged["holm_p"],
            "Country ICC": merged["icc_country"],
            "n subjects": merged["n_subjects"],
            "n countries": merged["n_countries"],
        }
    ).reset_index(drop=True)


def build_sensitivity_frame(results: pd.DataFrame, landscape: pd.DataFrame) -> pd.DataFrame:
    keys = ["bag", "arm", "rung", "region"]
    merged = results.merge(
        landscape[[*keys, "n_candidates", "mean_delta_r2", "fraction_delta_positive"]],
        on=keys,
        validate="one_to_one",
    )
    merged = _ordered(merged)
    region_rank = {name: index for index, name in enumerate(REGION_ORDER)}
    merged = merged.assign(_region=merged["region"].map(region_rank)).sort_values(
        ["bag", "rung", "arm", "_region"], kind="stable"
    )
    return pd.DataFrame(
        {
            "BAG": merged["bag"].map(BAG_LABEL),
            "Arm": merged["arm"].map(ARM_LABEL),
            "Model level": merged["rung"].map(LEVEL_LABEL),
            "Region": merged["region"],
            "Candidates": merged["n_candidates"],
            "Observed mean ΔR²": merged["mean_delta_r2"],
            "Fraction ΔR² > 0": merged["fraction_delta_positive"],
            "β0": merged["beta_intercept"],
            "SE": merged["se"],
            "95% CI": merged.apply(
                lambda row: f"{row['ci_lo']:.4f} to {row['ci_hi']:.4f}", axis=1
            ),
            "Two-sided P": merged["p_two_sided"],
            "Holm P": merged["holm_p"],
            "Directional P (secondary)": merged["p_positive_tail"],
            "Country ICC": merged["icc_country"],
            "n subjects": merged["n_subjects"],
        }
    ).reset_index(drop=True)


def build(
    results_path: Path, landscape_path: Path, source_data: Path, workbook_path: Path
) -> tuple[Path, Path]:
    results = pd.read_csv(results_path)
    landscape = pd.read_csv(landscape_path)

    primary = build_primary_frame(results, landscape)
    sensitivity = build_sensitivity_frame(results, landscape)

    source_data.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(source_data, index=False)
    sensitivity.to_csv(source_data.with_name("hierarchical_oof_region_sensitivity.csv"), index=False)

    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_table(
        workbook,
        PublicationTable(
            25, f"{PRIMARY_REGION} hierarchical OOF improvement over the covariate baseline",
            primary, NOTE,
        ),
    )
    _write_table(
        workbook,
        PublicationTable(
            26, "Hierarchical OOF improvement across nested candidate regions",
            sensitivity, SENSITIVITY_NOTE,
        ),
    )
    workbook["ST25"].title = "Top20 hierarchical improvement"
    workbook["ST26"].title = "TopK sensitivity"

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)
    return source_data, workbook_path


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "hierarchical_oof_model_results.csv")
    parser.add_argument(
        "--landscape", type=Path, default=root / "hierarchical_oof_landscape_summary.csv"
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/source_data/hierarchical_oof/hierarchical_oof_top20.csv",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT
        / "outputs/main/paper/complete/tables/Supplementary_Table_Hierarchical_OOF_Improvement.xlsx",
    )
    args = parser.parse_args()
    source_data, workbook = build(args.results, args.landscape, args.source_data, args.workbook)
    print(f"Saved: {source_data}")
    print(f"Saved: {workbook}")


if __name__ == "__main__":
    main()
