import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.xgb_frozen_top50 import country_balanced_scores, endpoint_summary


def _scores(candidates):
    rows = []
    for candidate, value in candidates:
        for country in ("A", "B"):
            rows.append({"candidate_id": candidate, "rung_id": "xgb_tree_d3", "fold_country": country, "r2": value})
    return country_balanced_scores(pd.DataFrame(rows), expected_countries=2).assign(bag_target="structural")


def test_endpoint_summary_uses_country_balanced_scores_and_stable_best() -> None:
    summary = endpoint_summary(
        _scores([("__baseline__", 0.2)]),
        _scores([("__single__a", 0.3), ("__single__b", 0.25)]),
        _scores([(f"f{i}", 0.31 if i == 0 else 0.1) for i in range(50)]),
        _scores([(f"k{i}", 0.35 if i == 0 else 0.2) for i in range(50)]),
    )
    row = summary.iloc[0]
    assert row["best_single_candidate_id"] == "__single__a"
    assert row["delta_best_single_minus_baseline"] == pytest.approx(0.1)
    assert row["delta_top50_full63_best_minus_best_single"] == pytest.approx(0.01)
    assert row["delta_top50_k10_best_minus_best_single"] == pytest.approx(0.05)
