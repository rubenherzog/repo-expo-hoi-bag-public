from pathlib import Path


def test_selection_sensitivities_only_read_new_k10_and_greedy_omega() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/run_main_selection_sensitivities.py").read_text(encoding="utf-8")
    assert "metrics_country.csv" in source and "thoi_o.lt(0)" in source
    assert "outputs/dedup" not in source
