from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages import sensitivity_common
from repo_expo_hoi_bag.stages.prepare_main_k10_sensitivity_adapter import (
    _materialize_effective_exposome,
)
from repo_expo_hoi_bag.stages.run_domain_imbalance_sensitivity import (
    _reuse_domain_pc1_oinfo,
)


def test_main_k10_stage_has_a_nonwriting_smoke_mode() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/build_main_k10_derivatives.py").read_text(encoding="utf-8")
    assert '"--smoke-test"' in source
    assert "if args.smoke_test:" in source
    assert "destination.mkdir" in source


def test_effective_exposome_materialization_uses_unique_public_signatures(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"
    domains = root / "data/metadata/exposome_feature_domains.csv"
    destination = tmp_path / "effective.csv"

    manifest = _materialize_effective_exposome(source, domains, destination)
    result = pd.read_csv(destination, low_memory=False)
    feature_names = pd.read_csv(domains)["feature_name"].astype(str).tolist()

    assert manifest["rows"] == 259
    assert manifest["features"] == 63
    assert result[feature_names].drop_duplicates().shape[0] == 259


def test_oinfo_cache_is_invalidated_when_reference_matrix_changes(
    monkeypatch, tmp_path: Path
) -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "order": [3],
            "nplet_vars": [["a", "b", "c"]],
        }
    )
    calls: list[float] = []

    def fake_score(reference, candidate_df, exposome_cols, *, batch_size):
        value = float(reference[exposome_cols].to_numpy().sum())
        calls.append(value)
        result = candidate_df.copy()
        result["score"] = value
        result["thoi_o"] = value
        return result

    monkeypatch.setattr(
        sensitivity_common,
        "score_oinfo_for_candidates_on_matrix",
        fake_score,
    )
    cache = tmp_path / "scores.csv"
    first = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    second = pd.DataFrame({"a": [4.0], "b": [5.0], "c": [6.0]})

    sensitivity_common.score_oinfo_with_cache(
        first, candidates, ["a", "b", "c"], cache_path=cache, batch_size=8
    )
    result = sensitivity_common.score_oinfo_with_cache(
        second, candidates, ["a", "b", "c"], cache_path=cache, batch_size=8
    )

    assert calls == [6.0, 15.0]
    assert result.loc[0, "score"] == 15.0
    assert pd.read_csv(cache)["reference_sha256"].nunique() == 1


def test_domain_pc1_oinfo_is_reused_by_exact_predictor_identity(
    tmp_path: Path,
) -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["candidate-b", "candidate-a"],
            "order": [2, 2],
            "predictors_identity": ["pc1_b|pc1_c", "pc1_a|pc1_b"],
        }
    )
    reference_path = tmp_path / "domain_pc1_oinfo.csv"
    pd.DataFrame(
        {
            "predictors_identity": ["pc1_a|pc1_b", "pc1_b|pc1_c"],
            "score": [-0.25, 0.75],
            "thoi_o": [-0.2, 0.7],
        }
    ).to_csv(reference_path, index=False)

    result = _reuse_domain_pc1_oinfo(candidates, reference_path)

    assert result["score"].tolist() == [0.75, -0.25]
    assert result["thoi_o"].tolist() == [0.7, -0.2]
    assert result["rank_o_min"].tolist() == [2.0, 1.0]
    assert result["rank_o_max"].tolist() == [1.0, 2.0]
    assert result["oinfo_reference_sha256"].nunique() == 1
