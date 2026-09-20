from pathlib import Path


def test_difference_audit_requires_changed_xgboost() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/audit_main_vs_historical.py").read_text(encoding="utf-8")
    assert "new_k10_xgboost_must_differ" in source
    assert "allowed_ols_reuse" in source
