from pathlib import Path


def test_comparison_stage_can_separate_source_and_main_delivery() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/compute_hpo_model_comparisons.py").read_text(encoding="utf-8")
    assert '"--source-run-id"' in source
    assert '"--output-run-id"' in source
    assert '"--exclude-ols"' in source
