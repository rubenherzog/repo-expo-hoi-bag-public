#!/usr/bin/env python3
"""Render the feature-ablation figure from existing saved analysis tables."""
from __future__ import annotations

import pandas as pd

from scripts.run_feature_ablation_sensitivity import _feature_ablation_config, _plot_results
from scripts.sensitivity_common import (
    active_rungs,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)


def main() -> None:
    cfg = load_sensitivity_config()
    config = _feature_ablation_config(cfg)
    summary_root = repo_sensitivity_root(cfg) / "feature_ablation"
    paths = {
        "results": summary_root / "feature_ablation_results.csv",
        "clouds": summary_root / "feature_ablation_cloud_summary.csv",
        "contrasts": summary_root / "feature_ablation_arm_contrasts.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Feature-ablation rendering requires saved analysis tables: " + ", ".join(missing))
    results = pd.read_csv(paths["results"])
    clouds = pd.read_csv(paths["clouds"])
    contrasts = pd.read_csv(paths["contrasts"])
    bags = [bag for bag in selected_bags(cfg, include_combined=False) if bag in set(results["bag"].astype(str))]
    rungs = [rung for rung in active_rungs(cfg) if rung in set(results["rung_id"].astype(str))]
    if not bags or not rungs:
        raise ValueError("Saved feature-ablation results contain no configured BAGs or model levels")
    _plot_results(
        results,
        clouds,
        contrasts,
        bags=bags,
        rungs=rungs,
        config=config,
        outdir=summary_root / "figures",
    )


if __name__ == "__main__":
    main()
