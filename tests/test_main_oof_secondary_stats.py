from pathlib import Path


def test_secondary_oof_stats_use_only_k10_oof() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/run_main_oof_secondary_stats.py").read_text(encoding="utf-8")
    assert "level_best_" in source and "mixedlm" in source
    assert "outputs/dedup" not in source
