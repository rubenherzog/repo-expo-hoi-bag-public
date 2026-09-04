from __future__ import annotations

import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.contracts import ResultContractError, select_top_within_rung, write_canonical_csv


def test_select_top_within_rung_never_reuses_a_reference_rung() -> None:
    metrics = pd.DataFrame(
        {
            "rung_id": ["ols", "ols", "xgb_tree_d1", "xgb_tree_d1"],
            "objective": ["o_min"] * 4,
            "candidate_id": ["ols-best", "ols-other", "d1-other", "d1-best"],
            "full_r2": [0.4, 0.2, 0.3, 0.5],
        }
    )
    selected = select_top_within_rung(metrics, objective="o_min", top_k=1)
    assert selected.set_index("rung_id")["candidate_id"].to_dict() == {
        "ols": "ols-best",
        "xgb_tree_d1": "d1-best",
    }


def test_select_top_within_rung_requires_metric_contract() -> None:
    with pytest.raises(ResultContractError, match="rung_id"):
        select_top_within_rung(pd.DataFrame({"objective": ["o_min"]}), objective="o_min", top_k=1)


def test_canonical_writer_is_deterministic(tmp_path) -> None:
    frame = pd.DataFrame({"b": [2.0], "a": [1]})
    first = write_canonical_csv(frame, tmp_path / "first.csv")
    second = write_canonical_csv(frame, tmp_path / "second.csv")
    assert first == second
