from __future__ import annotations

import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.canonical import CanonicalTableStore
from repo_expo_hoi_bag.analysis.selection import (
    select_baseline_per_rung,
    select_best_per_rung,
    select_top_per_rung,
)


def test_xgb_depth_tracks_the_selected_rung() -> None:
    from repo_expo_hoi_bag.analysis.selection import xgb_depth_for_rung

    assert xgb_depth_for_rung("xgb_tree_d1") == 1
    assert xgb_depth_for_rung("xgb_tree_d2") == 2
    assert xgb_depth_for_rung("xgb_tree_d3") == 3
from repo_expo_hoi_bag.config.models import ConfigurationError


def test_selection_helpers_are_intra_rung() -> None:
    frame = pd.DataFrame(
        {
            "rung_id": ["ols", "ols", "xgb", "xgb"],
            "objective": ["o_min", "o_min", "o_min", "o_min"],
            "candidate_id": ["a", "b", "a", "c"],
            "full_r2": [0.2, 0.4, 0.1, 0.3],
        }
    )
    top = select_top_per_rung(frame, objective="o_min", top_k=1)
    assert top["candidate_id"].tolist() == ["b", "c"]

    best = select_best_per_rung(frame, objective="o_min")
    assert best["rung_id"].tolist() == ["ols", "xgb"]


def test_baseline_selection_is_one_row_per_rung() -> None:
    frame = pd.DataFrame(
        {
            "rung_id": ["ols", "ols", "xgb"],
            "candidate_family": ["single", "single", "single"],
            "full_r2": [0.1, 0.3, 0.2],
        }
    )
    baseline = select_baseline_per_rung(frame, baseline_label="single")
    assert baseline["full_r2"].tolist() == [0.3, 0.2]


def test_canonical_store_hashes_and_rejects_escape(tmp_path) -> None:
    root = tmp_path / "tables"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    store = CanonicalTableStore(root)
    assert store.load_csv("nested/a.csv")["x"].tolist() == [1]
    assert store.hashes(["nested/a.csv"])[0].name == "nested/a.csv"
    with pytest.raises(ConfigurationError, match="under"):
        store.resolve("../escape.csv")
