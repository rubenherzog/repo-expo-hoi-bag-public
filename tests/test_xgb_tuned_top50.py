from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.xgb_tuned_top50 import (
    checkpoint_best_global_source,
    full_exposome_candidate,
    normalize_evaluator_global_rung,
    paired_arm_delta,
    resolve_comparison_candidate_mode,
    resolve_comparison_ensemble_size,
    resolve_comparison_top_n,
    select_historical_top_candidates,
    select_top_trial_ensemble_configs,
)


def test_checkpoint_best_global_source_reads_current_best_without_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "work" / "xgb_nested_loco_tuning" / "k10_current" / "functional" / "global"
    checkpoint.mkdir(parents=True)
    (checkpoint / "xgb_tree_d3.json").write_text(json.dumps({
        "signature_payload": {
            "bag": "functional", "rung_id": "xgb_tree_d3", "outer_country": "__global__",
            "countries": ["A", "B"], "feature_scope": "domain_balanced_k10",
            "tuning_cfg": {"optimization_direction": "minimize"},
        },
        "trials": [
            {"trial_index": 0, "objective_value": 2.0, "params": {"gamma": 1.0}},
            {"trial_index": 1, "objective_value": 1.0, "params": {"gamma": 0.0}},
        ],
    }))
    source = checkpoint_best_global_source(
        tmp_path, "k10_current", "functional", "xgb_tree_d3", tunable_keys={"gamma"}
    )
    assert source["n_trials_observed"] == 2
    assert source["best_trial_index"] == 1
    assert source["params"] == {"gamma": 0.0}
from repo_expo_hoi_bag.core.oinfo_bag_ladder.config import BAG_TARGETS


def test_top50_comparison_is_configured_for_the_deduplicated_paper_universe() -> None:
    config_text = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "repo_expo_hoi_bag"
        / "stages"
        / "resources"
        / "sensitivity.yaml"
    ).read_text()
    assert "per_experiment/pooled_oinfo_ladder_{bag}/metrics_global_long.parquet" in config_text
    assert "config/paper_reference.yaml" in config_text
    reference_text = (Path(__file__).resolve().parents[1] / "config" / "paper_reference.yaml").read_text()
    assert "read_only: true" in reference_text
    assert "expo_hoi_bag_repro_data_dedup" in reference_text
    assert "sensitivity_relative_path: sensitivity" in reference_text
    assert "runs/oinfo_only/variant_a/families/pooled_oinfo_ladder" not in config_text


def test_historical_top_selection_is_stable_and_excludes_baseline() -> None:
    metrics = pd.DataFrame({
        "candidate_id": ["b", "__baseline__", "a", "c"],
        "bag_target": ["functional"] * 4,
        "rung_id": ["xgb_tree_d3"] * 4,
        "full_r2": [0.4, 0.99, 0.4, 0.2],
        "predictors_identity": ["x|y", "", "z", "q"],
    })
    selected = select_historical_top_candidates(metrics, bag="functional", rung_id="xgb_tree_d3", top_n=2)
    assert selected["candidate_id"].tolist() == ["a", "b"]
    assert selected["nplet_vars"].tolist() == [["z"], ["x", "y"]]
    assert selected["candidate_family"].unique().tolist() == ["historical_functional_xgb_tree_d3_top50"]


def test_paired_delta_requires_exact_coverage_and_computes_tuned_minus_fixed() -> None:
    fixed = pd.DataFrame({"candidate_id": ["a"], "r2": [0.2]})
    tuned = pd.DataFrame({"candidate_id": ["a"], "r2": [0.3]})
    result = paired_arm_delta(fixed, tuned, keys=["candidate_id"], metrics=["r2"])
    assert result.loc[0, "delta_r2_tuned_minus_fixed"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="same pairing keys"):
        paired_arm_delta(fixed, pd.DataFrame({"candidate_id": ["b"], "r2": [0.3]}), keys=["candidate_id"], metrics=["r2"])


def test_top_n_smoke_override_is_bounded_by_the_versioned_full_comparison() -> None:
    assert resolve_comparison_top_n(50, {"XGB_TOP50_COMPARISON_TOP_N": "5"}) == 5
    with pytest.raises(ValueError, match="between 1 and configured top_n"):
        resolve_comparison_top_n(50, {"XGB_TOP50_COMPARISON_TOP_N": "51"})


def test_full_exposome_mode_has_one_complete_canonical_candidate() -> None:
    candidate = full_exposome_candidate(["x", "y", "z"])
    assert candidate.loc[0, "candidate_id"] == "__full_exposome__"
    assert candidate.loc[0, "nplet_vars"] == ["x", "y", "z"]
    assert candidate.loc[0, "order"] == 3
    assert resolve_comparison_candidate_mode({}) == "historical_top50"
    assert resolve_comparison_candidate_mode({"XGB_TUNED_COMPARISON_CANDIDATE_MODE": "full_exposome"}) == "full_exposome"
    with pytest.raises(ValueError, match="historical_top50 or full_exposome"):
        resolve_comparison_candidate_mode({"XGB_TUNED_COMPARISON_CANDIDATE_MODE": "all"})


def test_top_trial_ensemble_uses_only_best_diagnostic_trials_per_outer_country() -> None:
    diagnostics = pd.DataFrame({
        "bag": ["functional"] * 6,
        "rung_id": ["xgb_tree_d3"] * 6,
        "outer_country": ["A", "A", "A", "B", "B", "B"],
        "trial_index": [2, 0, 1, 0, 1, 2],
        "objective_value": [0.3, 0.5, 0.5, 0.1, 0.4, 0.2],
        "params_json": [
            '{"gamma": 2}', '{"gamma": 0}', '{"gamma": 1}',
            '{"gamma": 10}', '{"gamma": 11}', '{"gamma": 12}',
        ],
    })
    selected = select_top_trial_ensemble_configs(
        diagnostics, bag="functional", rung_id="xgb_tree_d3", countries=["A", "B"], ensemble_size=2
    )
    assert selected == {"A": [{"gamma": 0}, {"gamma": 1}], "B": [{"gamma": 11}, {"gamma": 12}]}
    assert resolve_comparison_ensemble_size({"XGB_TUNED_COMPARISON_ENSEMBLE_SIZE": "3"}) == 3
    with pytest.raises(ValueError, match="between 1 and 8"):
        resolve_comparison_ensemble_size({"XGB_TUNED_COMPARISON_ENSEMBLE_SIZE": "9"})


def test_functional_target_uses_the_shared_bag_target_contract() -> None:
    assert BAG_TARGETS["functional"] == "bag_func_resolved"


def test_global_evaluator_rung_merge_is_normalized_and_checked() -> None:
    normalized = normalize_evaluator_global_rung(
        pd.DataFrame({"rung_id_x": ["xgb_tree_d3"], "rung_id_y": ["xgb_tree_d3"]})
    )
    assert normalized["rung_id"].tolist() == ["xgb_tree_d3"]
    with pytest.raises(ValueError, match="inconsistent"):
        normalize_evaluator_global_rung(
            pd.DataFrame({"rung_id_x": ["xgb_tree_d2"], "rung_id_y": ["xgb_tree_d3"]})
        )
