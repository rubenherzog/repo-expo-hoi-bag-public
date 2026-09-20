from pathlib import Path


def test_main_k10_derivative_stage_is_nonfitting_and_main_scoped() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/build_main_k10_derivatives.py").read_text(encoding="utf-8")
    assert 'default="main"' in source
    assert '"country_variant": "historical_a"' in source
    assert "historical_ols_provenance" in source
    assert "_fit_candidate" not in source
    assert "xgboost" not in source.lower()
    assert "imshow" not in source
    assert "fig4_residual_bias_main_k10" not in source
