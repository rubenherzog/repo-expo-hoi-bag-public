from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from repo_expo_hoi_bag.stages.run_diagnosis_balance_sensitivity import (
    _load_reference_oof,
    _metric_rows,
)


def _reference_frame(bag: str = "structural") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(6),
            "N_MEGA": [f"id-{i}" for i in range(6)],
            "country": ["A", "A", "B", "B", "C", "C"],
            "diagnosis": ["CN", "AD", "FTD", "CN", "AD", "FTD"],
            "y_true": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "y_pred_full": [1.1, 2.1, 3.1, 1.6, 2.6, 3.6],
            "y_pred_base": [0.9, 1.9, 2.9, 1.4, 2.4, 3.4],
            "candidate_id": "candidate-1",
            "objective": "o_min",
            "rung_id": "xgb_tree_d3",
            # Main-k10 selected OOF files use ``bag`` rather than the legacy
            # diagnosis-balance name ``bag_target``.
            "bag": bag,
        }
    )


def test_reference_oof_normalizes_main_k10_bag_column(tmp_path: Path) -> None:
    source = tmp_path / "oof.parquet"
    _reference_frame().to_parquet(source, index=False)

    loaded = _load_reference_oof(
        {"unweighted_oof_source": str(source), "bag": "structural"},
        ("CN", "AD", "FTD"),
    )

    assert loaded["bag_target"].unique().tolist() == ["structural"]
    rows = _metric_rows(loaded, "unweighted", ("CN", "AD", "FTD"))
    assert rows
    assert {row["bag"] for row in rows} == {"structural"}


def test_reference_oof_rejects_a_different_bag(tmp_path: Path) -> None:
    source = tmp_path / "oof.parquet"
    _reference_frame("functional").to_parquet(source, index=False)

    with pytest.raises(ValueError, match="does not match the selected model"):
        _load_reference_oof(
            {"unweighted_oof_source": str(source), "bag": "structural"},
            ("CN", "AD", "FTD"),
        )


def test_cluster_jobs_persist_per_bag_before_combining() -> None:
    source = Path(
        "src/repo_expo_hoi_bag/stages/run_diagnosis_balance_sensitivity.py"
    ).read_text(encoding="utf-8")
    assert "bag_root = local_root / bag" in source
    assert "for bag in paper_cfg.primary_bags" in source
    assert "pd.concat(parts, ignore_index=True)" in source
