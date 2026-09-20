from pathlib import Path


def test_main_table_adapters_limit_historical_inputs_to_greedy_and_omega() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/run_main_table_adapters.py").read_text(encoding="utf-8")
    assert "--candidate-registry" in source and "--triplet-oinfo-dir" in source
    assert "outputs/dedup" not in source
    assert "repro_data_dedup" not in source
