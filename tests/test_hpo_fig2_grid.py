from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages.plot_hpo_fig2_grid import _metrics_paths, _registry_for_bag


def test_hpo_fig2_uses_shared_endpoint_metrics_and_set_specific_candidates(tmp_path: Path) -> None:
    candidates, baseline, single = _metrics_paths(tmp_path, "k63", "structural", "xgb_tree_d2")

    assert candidates == (
        tmp_path
        / "results/analysis_runs/paper_reanalysis_k63/xgb/structural/xgb_tree_d2/k63/metrics_country.csv"
    )
    assert baseline == (
        tmp_path
        / "results/analysis_runs/paper_reanalysis_k10/xgb/structural/xgb_tree_d2/baseline/metrics_country.csv"
    )
    assert single == (
        tmp_path
        / "results/analysis_runs/paper_reanalysis_k10/xgb/structural/xgb_tree_d2/single/metrics_country.csv"
    )


def test_hpo_fig2_ols_metrics_are_shared_between_hpo_sets(tmp_path: Path) -> None:
    candidates, baseline, single = _metrics_paths(tmp_path, "k10", "functional", "ols")

    expected = tmp_path / "results/analysis_runs/paper_reanalysis_k10/ols/functional/ols"
    assert candidates == expected / "ols/metrics_country.csv"
    assert baseline == expected / "baseline/metrics_country.csv"
    assert single == expected / "single/metrics_country.csv"


def test_hpo_fig2_selects_the_bag_specific_candidate_registry() -> None:
    registry = pd.DataFrame(
        {
            "candidate_id": ["same-id", "same-id"],
            "objective": ["o_min", "o_max"],
            "order": [3, 3],
            "thoi_o": [-1.0, 1.0],
            "predictors_identity": ["a|b|c", "d|e|f"],
            "experiment_id": ["pooled_oinfo_ladder_structural", "pooled_oinfo_ladder_functional"],
        }
    )

    structural = _registry_for_bag(registry, "structural")

    assert structural.to_dict("records") == [
        {"candidate_id": "same-id", "objective": "o_min", "order": 3, "thoi_o": -1.0, "predictors_identity": "a|b|c"}
    ]
