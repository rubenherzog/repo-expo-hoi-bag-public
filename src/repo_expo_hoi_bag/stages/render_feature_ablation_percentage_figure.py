#!/usr/bin/env python3
"""Render relative LOFO R² losses from saved feature-ablation results only."""
from __future__ import annotations

import pandas as pd

from repo_expo_hoi_bag.analysis.feature_ablation import add_relative_r2_loss_percentage, cloud_summary
from scripts.run_feature_ablation_sensitivity import _feature_ablation_config, _plot_results
from scripts.sensitivity_common import (
    active_rungs,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)


STEM = "feature_ablation_percentage"


def _percentage_cloud_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Reuse the published cloud summariser on the displayed relative-loss scale."""
    cloud_input = results.drop(columns=["r2_loss"]).rename(columns={"r2_loss_percentage": "r2_loss"})
    return cloud_summary(cloud_input)


def main() -> None:
    cfg = load_sensitivity_config()
    summary_root = repo_sensitivity_root(cfg) / "feature_ablation"
    results_path = summary_root / "feature_ablation_results.csv"
    contrasts_path = summary_root / "feature_ablation_arm_contrasts.csv"
    missing = [str(path) for path in (results_path, contrasts_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Percentage rendering requires saved feature-ablation tables: " + ", ".join(missing))
    results = add_relative_r2_loss_percentage(pd.read_csv(results_path))
    contrasts = pd.read_csv(contrasts_path)
    bags = [bag for bag in selected_bags(cfg, include_combined=False) if bag in set(results["bag"].astype(str))]
    rungs = [rung for rung in active_rungs(cfg) if rung in set(results["rung_id"].astype(str))]
    _plot_results(
        results,
        _percentage_cloud_summary(results),
        contrasts,
        bags=bags,
        rungs=rungs,
        config=_feature_ablation_config(cfg),
        outdir=summary_root / "figures",
        stem=STEM,
        value_column="r2_loss_percentage",
        value_label="Relative ΔR² loss (% of complete-model R²)",
        value_description="Relative leave-one-exposure-out R² losses (% of complete-model R²)",
        source_value_description=(
            "100 times (parent_full_r2 minus ablated_full_r2) divided by parent_full_r2; signed percentage"
        ),
        source_paths=["feature_ablation_results.csv"],
    )


if __name__ == "__main__":
    main()
