from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages.sensitivity_common import route_main_k10_hpo


ROOT = Path(__file__).resolve().parents[1]


def _artifact(path: Path, feature_scope: str) -> str:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "provenance": {"feature_scope": feature_scope},
                "selected": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_k10_candidates_route_to_their_own_hpo_scopes(
    monkeypatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.json"
    single = tmp_path / "single.json"
    k10 = tmp_path / "k10.json"
    expected_hashes = {
        "baseline": _artifact(baseline, "baseline"),
        "single_exposure": _artifact(single, "single_exposure"),
        "k10": _artifact(k10, "domain_balanced_k10"),
    }
    monkeypatch.setenv("XGB_TUNING_BASELINE_ARTIFACT", str(baseline))
    monkeypatch.setenv("XGB_TUNING_SINGLE_ARTIFACT", str(single))

    candidates = pd.DataFrame(
        [
            {"candidate_id": "__baseline__", "candidate_family": "baseline"},
            {"candidate_id": "single_pm25", "candidate_family": "single_exposure"},
            {"candidate_id": "k10_candidate", "candidate_family": "oinfo_ladder"},
        ]
    )
    routed, artifacts = route_main_k10_hpo(candidates, k10)

    assert routed.set_index("candidate_id")["hpo_scope"].to_dict() == {
        "__baseline__": "baseline",
        "single_pm25": "single_exposure",
        "k10_candidate": "k10",
    }
    assert routed.set_index("hpo_scope")["hpo_artifact_sha256"].to_dict() == expected_hashes
    assert artifacts == {
        "baseline": baseline.resolve(),
        "single_exposure": single.resolve(),
        "k10": k10.resolve(),
    }


def test_cluster_runner_exports_all_three_frozen_hpo_artifacts() -> None:
    text = (ROOT / "scripts/jobs/run_main_k10_sensitivity_bag.sh").read_text()
    assert "XGB_TUNING_BASELINE_ARTIFACT" in text
    assert "XGB_TUNING_SINGLE_ARTIFACT" in text
    assert 'XGB_TUNING_ARTIFACT="$K10_HPO"' in text


def test_correction_launcher_reuses_the_assigned_namespace_and_only_affected_stages() -> None:
    path = ROOT / "scripts/submit_main_k10_hpo_scope_corrections.sh"
    text = path.read_text(encoding="utf-8")
    assert 'RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"' in text
    assert "prepare_main_k10_cluster_adapter.sh" not in text
    assert text.count("main_k10_hporoute_") == 10
    for stage in (
        "normative-transfer-xgb",
        "country-region",
        "education-scanner-baseline",
        "residualized-bag",
        "diagnosis-balance",
    ):
        assert stage in text
    for unaffected in ("country-block-null", "domain-imbalance", "whole-exposome-pca"):
        assert unaffected not in text
