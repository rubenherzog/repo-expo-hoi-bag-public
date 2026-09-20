from pathlib import Path


def test_main_local_analyses_only_admits_k10_and_triplet_oinfo_inputs() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/run_main_local_analyses.py").read_text(encoding="utf-8")
    assert "per_candidate_diversity_scatter.csv" in source
    assert "--triplet-oinfo-index" in source
    assert "outputs/dedup" not in source
    assert "repro_data_dedup" not in source
