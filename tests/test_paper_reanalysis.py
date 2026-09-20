from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from repo_expo_hoi_bag.analysis.paper_reanalysis import (
    checkpoint_identity, country_balanced_ols_r2, pending_chunks, validate_candidate_pool,
    validate_ols_reuse_manifest, write_chunk_complete,
)
from repo_expo_hoi_bag.stages.run_paper_reanalysis import _completed_chunks


def _pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"experiment_id": "pooled_oinfo_ladder_structural", "candidate_id": f"{objective}-{order}-{rank}",
         "objective": objective, "order": order, "rank": rank}
        for objective in ("o_max", "o_min") for order in range(3, 31) for rank in range(1, 21)
    ])


def test_official_pool_contract_rejects_wrong_cardinality() -> None:
    assert len(validate_candidate_pool(_pool(), "structural")) == 1120
    with pytest.raises(ValueError, match="1,120"):
        validate_candidate_pool(_pool().iloc[:-1], "structural")


def test_partial_checkpoint_resume_only_returns_unfinished_chunks(tmp_path: Path) -> None:
    identity = checkpoint_identity(code_hash="code", pool_hash="pool", policy_hash="policy", hpo_hash="hpo",
                                   bag="structural", rung="xgb_tree_d1")
    ids = ["a", "b", "c", "d", "e"]
    write_chunk_complete(tmp_path, 0, identity, ids[:2], {"summary": "chunk-0000.csv"})
    assert pending_chunks(tmp_path, identity, ids, 2) == [(1, ["c", "d"]), (2, ["e"])]
    json.loads((tmp_path / "chunk-0000.json").read_text())


def test_ols_reuse_requires_provenance_and_balances_countries_equally() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_ols_reuse_manifest({}, cohort_sha256="a", policy_sha256="b", pool_sha256="c", contract_sha256="d")
    manifest = {"cohort_sha256": "a", "country_policy_sha256": "b", "candidate_pool_sha256": "c", "ols_contract_sha256": "d"}
    validate_ols_reuse_manifest(manifest, cohort_sha256="a", policy_sha256="b", pool_sha256="c", contract_sha256="d")
    scores = country_balanced_ols_r2(pd.DataFrame([
        {"candidate_id": "x", "country_full_r2": 0.2, "n_test_scored": 10},
        {"candidate_id": "x", "country_full_r2": 0.8, "n_test_scored": 1000},
    ]))
    assert scores.loc[0, "country_balanced_r2"] == pytest.approx(0.5)


def test_launcher_defines_six_xgb_units_plus_one_ols_unit() -> None:
    launcher = (Path(__file__).resolve().parents[1] / "scripts" / "submit_paper_reanalysis.sh").read_text()
    assert 'PAPER_REANALYSIS_MODE=ols' in launcher
    assert 'for bag in structural functional' in launcher
    assert 'for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3' in launcher
    assert 'PAPER_REANALYSIS_BASELINE_HPO' in launcher
    assert 'PAPER_REANALYSIS_SINGLE_HPO' in launcher
    assert 'PAPER_REANALYSIS_K10_HPO' in launcher
    assert 'PAPER_REANALYSIS_K63_HPO' in launcher


def test_historical_driver_launcher_has_seven_jobs_and_serializes_three_set_vectors() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "submit_historical_driver_hpo_sets.sh").read_text()
    assert 'historical_hpo_ols' in launcher
    assert 'for bag in structural functional' in launcher
    assert 'for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3' in launcher
    assert 'PAPER_REANALYSIS_SETS=k10,k63,single' in launcher
    assert 'run_historical_driver_reanalysis.sh' in launcher


def test_resume_accepts_completed_chunks_from_an_older_chunk_size(tmp_path: Path) -> None:
    identity = "same-scientific-unit"
    global_path, country_path = tmp_path / "old-global.csv", tmp_path / "old-country.csv"
    pd.DataFrame({"candidate_id": ["a", "b"]}).to_csv(global_path, index=False)
    pd.DataFrame({"candidate_id": ["a", "b"]}).to_csv(country_path, index=False)
    (tmp_path / "chunk-0000.json").write_text(json.dumps({
        "status": "complete", "identity": identity, "candidate_ids": ["a", "b"],
        "files": {"global": str(global_path), "country": str(country_path)},
    }))
    completed = _completed_chunks(tmp_path, identity, {"a", "b", "c"}, accept_legacy=False)
    assert set(completed) == {"a", "b"}
