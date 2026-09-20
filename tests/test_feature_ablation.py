from __future__ import annotations

import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.contracts import ResultContractError
from repo_expo_hoi_bag.analysis.feature_ablation import (
    ablation_candidate_table,
    add_relative_r2_loss_percentage,
    arm_contrasts,
    attach_ablation_metrics,
    build_ablation_requests,
    cloud_summary,
    parent_ablation_summary,
    select_ranked_top_models,
)


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bag in ("structural", "functional"):
        for rung in ("ols", "xgb_tree_d1"):
            for arm in ("o_min", "o_max"):
                for index, r2 in enumerate((0.40, 0.30, 0.20), start=1):
                    rows.append(
                        {
                            "bag_target": bag,
                            "rung_id": rung,
                            "objective": arm,
                            "candidate_id": f"{bag}-{rung}-{arm}-{index}",
                            "predictors_identity": f"a{index}|b{index}|c{index}",
                            "full_r2": r2,
                            "order": 3,
                        }
                    )
    return pd.DataFrame(rows)


def test_top_models_are_ranked_independently_with_deterministic_ties() -> None:
    metrics = _metrics()
    tied = metrics.iloc[0].copy()
    tied["candidate_id"] = "aaa-tie"
    tied["predictors_identity"] = "x|y|z"
    tied["full_r2"] = 0.40
    selected = select_ranked_top_models(
        pd.concat([metrics, pd.DataFrame([tied])], ignore_index=True),
        bags=("structural", "functional"),
        rungs=("ols", "xgb_tree_d1"),
        top_k=2,
        order_max=30,
    )
    assert len(selected) == 16
    first = selected[
        (selected["bag_target"] == "structural")
        & (selected["rung_id"] == "ols")
        & (selected["objective"] == "o_min")
    ]
    assert first["candidate_id"].tolist() == ["aaa-tie", "structural-ols-o_min-1"]
    assert first["rank_within_arm"].tolist() == [1, 2]


def test_ablation_requests_preserve_signed_losses_and_scope_metrics_to_bag_and_rung() -> None:
    parents = select_ranked_top_models(
        _metrics(), bags=("structural",), rungs=("ols",), top_k=1, order_max=30
    )
    requests = build_ablation_requests(parents)
    assert len(requests) == 6
    assert set(requests["removed_feature"]) == {"a1", "b1", "c1"}
    identities = requests[["bag", "rung_id", "ablation_predictors_identity"]].drop_duplicates().copy()
    identities["ablated_full_r2"] = 0.45
    identities["provenance"] = "new_refit"
    results = attach_ablation_metrics(requests, identities)
    assert (results["r2_loss"] < 0).all()


def test_relative_loss_percentage_uses_each_parent_complete_r2() -> None:
    relative = add_relative_r2_loss_percentage(
        pd.DataFrame({"parent_full_r2": [0.20, 0.40], "r2_loss": [0.02, -0.04]})
    )
    assert relative["r2_loss_percentage"].tolist() == pytest.approx([10.0, -10.0])
    with pytest.raises(ResultContractError, match="strictly positive"):
        add_relative_r2_loss_percentage(pd.DataFrame({"parent_full_r2": [0.0], "r2_loss": [0.01]}))


def test_candidate_table_deduplicates_physical_fits_by_identity() -> None:
    parents = pd.DataFrame(
        [
            {
                "bag_target": "structural",
                "rung_id": "ols",
                "objective": "o_min",
                "candidate_id": "one",
                "predictors_identity": "a|b|c",
                "full_r2": 0.3,
                "rank_within_arm": 1,
            },
            {
                "bag_target": "structural",
                "rung_id": "ols",
                "objective": "o_max",
                "candidate_id": "two",
                "predictors_identity": "a|b|c",
                "full_r2": 0.2,
                "rank_within_arm": 1,
            },
        ]
    )
    candidates = ablation_candidate_table(build_ablation_requests(parents))
    assert len(candidates) == 3
    assert candidates["candidate_id"].is_unique


def test_missing_ablation_metrics_fail_loudly() -> None:
    parents = select_ranked_top_models(
        _metrics(), bags=("structural",), rungs=("ols",), top_k=1, order_max=30
    )
    requests = build_ablation_requests(parents)
    with pytest.raises(ResultContractError, match="incomplete"):
        attach_ablation_metrics(
            requests,
            pd.DataFrame(
                {
                    "bag": ["structural"],
                    "rung_id": ["ols"],
                    "ablation_predictors_identity": ["a1|b1"],
                    "ablated_full_r2": [0.1],
                    "provenance": ["new_refit"],
                }
            ),
        )


def test_cloud_and_contrast_summaries_are_deterministic_and_equal_parent_weighted() -> None:
    results = pd.DataFrame(
        {
            "bag": ["structural"] * 8,
            "rung_id": ["ols"] * 8,
            "arm": ["o_min"] * 4 + ["o_max"] * 4,
            "rank_within_arm": [1, 1, 2, 2, 1, 1, 2, 2],
            "parent_candidate_id": ["s1", "s1", "s2", "s2", "r1", "r1", "r2", "r2"],
            "parent_predictors_identity": ["a|b"] * 8,
            "parent_full_r2": [0.3] * 8,
            "parent_order": [2] * 8,
            "r2_loss": [0.2, 0.4, 0.1, 0.3, 0.0, 0.2, 0.0, 0.2],
        }
    )
    parents = parent_ablation_summary(results)
    assert parents.loc[parents["arm"] == "o_min", "mean_r2_loss"].tolist() == pytest.approx([0.3, 0.2])
    clouds = cloud_summary(results)
    assert len(clouds) == 4
    first = arm_contrasts(parents, bootstrap_draws=100, permutation_draws=100, seed=17)
    second = arm_contrasts(parents, bootstrap_draws=100, permutation_draws=100, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "mean_difference_synergy_minus_redundancy"] == pytest.approx(0.15)
