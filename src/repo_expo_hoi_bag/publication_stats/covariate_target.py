"""Paired country-level inference for publication Supplementary Table 17."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from repo_expo_hoi_bag.publication_stats.paths import (
    REPOSITORY_ROOT,
    required_dedup_root,
)


SENSITIVITY_ROOT = REPOSITORY_ROOT / "outputs/sensitivity/dedup"
OUTPUT = SENSITIVITY_ROOT / "covariate_target_matched_inference.csv"


def _paired_p(current: pd.Series, reference: pd.Series) -> tuple[int, float]:
    paired = pd.concat(
        [current.rename("current"), reference.rename("reference")], axis=1
    ).dropna()
    result = stats.wilcoxon(
        paired["current"] - paired["reference"], alternative="two-sided"
    )
    return int(len(paired)), float(result.pvalue)


def _covariate_rows(work_root: Path) -> list[dict[str, object]]:
    summary = pd.read_csv(
        SENSITIVITY_ROOT / "education_scanner_baseline/global_all_rungs.csv"
    )
    selected = summary[
        summary["rung_id"].eq("xgb_tree_d3")
        & summary["candidate_role"].eq("best_synergy")
    ]
    rows: list[dict[str, object]] = []
    for bag, bag_rows in selected.groupby("bag", observed=True):
        reference_row = bag_rows[
            bag_rows["covariate_set"].eq("baseline_covariates")
        ].iloc[0]
        reference_path = (
            work_root
            / "education_scanner_baseline"
            / bag
            / "baseline_covariates"
            / f"{bag}_xgb_tree_d3_country.csv"
        )
        reference_data = pd.read_csv(reference_path)
        reference = reference_data[
            reference_data["candidate_id"].eq(reference_row["candidate_id"])
        ].set_index("fold_country")["r2"]
        for row in bag_rows.to_dict("records"):
            specification = str(row["covariate_set"])
            if specification == "baseline_covariates":
                n_countries, p_value = int(reference.notna().sum()), None
            else:
                path = (
                    work_root
                    / "education_scanner_baseline"
                    / str(bag)
                    / specification
                    / f"{bag}_xgb_tree_d3_country.csv"
                )
                data = pd.read_csv(path)
                current = data[
                    data["candidate_id"].eq(row["candidate_id"])
                ].set_index("fold_country")["r2"]
                n_countries, p_value = _paired_p(current, reference)
            rows.append(
                {
                    "analysis": "Additional covariates",
                    "bag": bag,
                    "specification_or_model": specification,
                    "n_countries": n_countries,
                    "paired_country_wilcoxon_p": p_value,
                }
            )
    return rows


def _residualized_rows(work_root: Path) -> list[dict[str, object]]:
    summary = pd.read_csv(
        SENSITIVITY_ROOT / "residualized_bag/manuscript_target_comparison.csv"
    )
    models = {
        "Covariate baseline": None,
        "Best single exposure": "best_single_candidate_id",
        "Best synergy-arm model": "best_synergy_candidate_id",
        "Best redundancy-arm model": "best_redundancy_candidate_id",
    }
    rows: list[dict[str, object]] = []
    for row in summary.to_dict("records"):
        bag = str(row["bag"])
        path = (
            work_root
            / "residualized_bag"
            / bag
            / "residualized_eval_xgb_tree_d3"
            / f"{bag}_xgb_tree_d3_country.csv"
        )
        data = pd.read_csv(path)
        reference = data[data["candidate_id"].eq("__baseline__")].set_index(
            "fold_country"
        )["r2"]
        for model, candidate_column in models.items():
            if candidate_column is None:
                n_countries, p_value = int(reference.notna().sum()), None
            else:
                current = data[
                    data["candidate_id"].eq(row[candidate_column])
                ].set_index("fold_country")["r2"]
                n_countries, p_value = _paired_p(current, reference)
            rows.append(
                {
                    "analysis": "Residualised BAG target",
                    "bag": bag,
                    "specification_or_model": model,
                    "n_countries": n_countries,
                    "paired_country_wilcoxon_p": p_value,
                }
            )
    return rows


def main() -> None:
    bundle = required_dedup_root()
    work_root = bundle / "work/sensitivity_eval"
    result = pd.DataFrame(
        [*_covariate_rows(work_root), *_residualized_rows(work_root)]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
