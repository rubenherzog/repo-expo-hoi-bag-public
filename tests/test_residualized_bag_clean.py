"""Contract tests for the clean residualized-BAG sensitivity.

These cover the properties the audit showed the delivered sensitivity got
wrong: the residualizer's covariate set must match what the downstream design
controls for, the downstream design must not reintroduce those covariates, the
comparator must be the training residual mean rather than a literal zero, and
the resume cache must not let the two target/design regimes mix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/repo_expo_hoi_bag/core"))
import repo_expo_hoi_bag.stages as _stages  # noqa: E402

sys.modules.setdefault("scripts", _stages)

from repo_expo_hoi_bag.stages.sensitivity_common import (  # noqa: E402
    _residualize_train_fit,
    _residualize_train_fit_with_year,
    build_fold_residualized_y_with_year,
)
from repo_expo_hoi_bag.stages.run_residualized_bag_clean_sensitivity import (  # noqa: E402
    NULL_ARM_ID,
    _rank_biserial,
    landscape_ordering,
    null_candidate_df,
    paired_error_tests,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(20260731)
    n = 400
    age = rng.normal(65, 8, n)
    year = rng.integers(2010, 2020, n).astype(float)
    sex = rng.choice(["M", "F"], n)
    diag = rng.choice(["CN", "AD", "FTD"], n)
    # BAG genuinely depends on year, so a residualizer omitting year leaves
    # year-related variance behind.
    y = (
        0.4 * (age - 65)
        + 1.5 * year
        + 2.0 * (sex == "M")
        + 3.0 * (diag == "AD")
        + rng.normal(0, 2, n)
    )
    train_idx = np.arange(0, 300)
    return y, age, year, sex, diag, train_idx


def test_year_residualizer_removes_year_variance(cohort):
    """The clean residualizer must strip year; the delivered one must not."""
    y, age, year, sex, diag, train_idx = cohort
    with_year, _ = _residualize_train_fit_with_year(
        y, age, sex, diag, year.reshape(-1, 1), train_idx
    )
    without_year = _residualize_train_fit(y, age, sex, diag, train_idx)

    corr_with = abs(np.corrcoef(with_year[train_idx], year[train_idx])[0, 1])
    corr_without = abs(np.corrcoef(without_year[train_idx], year[train_idx])[0, 1])
    assert corr_with < 1e-8, "year must be orthogonal to the clean residual on training rows"
    assert corr_without > 0.5, "the delivered residualizer is expected to leave year behind"


def test_residualizer_is_fit_on_training_rows_only(cohort):
    """A held-out country must never influence its own residualization."""
    y, age, year, sex, diag, train_idx = cohort
    base, _ = _residualize_train_fit_with_year(
        y, age, sex, diag, year.reshape(-1, 1), train_idx
    )
    perturbed_y = y.copy()
    perturbed_y[300:] += 50.0  # change held-out rows only
    moved, _ = _residualize_train_fit_with_year(
        perturbed_y, age, sex, diag, year.reshape(-1, 1), train_idx
    )
    np.testing.assert_allclose(base[train_idx], moved[train_idx], atol=1e-9)


def test_heldout_residual_mean_is_nonzero_by_construction(cohort):
    """The mechanism behind the delivered figure's negative R2, pinned.

    Because the residualizer never sees the held-out fold, that fold's residual
    mean is offset from zero. This is why a per-country R2 penalises every
    model by a constant and why the paired-error test is the primary statistic.
    """
    y, age, year, sex, diag, _ = cohort
    countries = ["A", "B"]
    train_by_country = {"A": np.arange(200, 400), "B": np.arange(0, 200)}
    _, diagnostics = build_fold_residualized_y_with_year(
        y, age, sex, diag,
        {c: year.reshape(-1, 1) for c in countries},
        countries, train_by_country,
    )
    assert set(diagnostics["fold_country"]) == set(countries)
    assert (diagnostics["train_residual_mean"].abs() < 1e-8).all()
    assert (diagnostics["heldout_residual_sd"] > 0).all()
    assert "residualizer_train_r2" in diagnostics.columns


def test_null_candidate_is_distinct_from_covariate_baseline():
    frame = null_candidate_df()
    assert frame["candidate_id"].iloc[0] == NULL_ARM_ID
    assert frame["candidate_id"].iloc[0] != "__baseline__"
    assert frame["predictors_identity_n"].iloc[0] == 0
    assert list(frame["nplet_vars"].iloc[0]) == []


def test_rank_biserial_sign_and_bounds():
    assert _rank_biserial(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)
    assert _rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert abs(_rank_biserial(np.array([-1.0, 1.0]))) < 1e-12
    assert np.isnan(_rank_biserial(np.array([0.0, 0.0])))


def test_paired_error_test_detects_a_real_improvement():
    """An arm that beats the null in every country must be flagged."""
    countries = [f"C{i}" for i in range(12)]
    rows = []
    for i, country in enumerate(countries):
        rows.append({
            "candidate_id": NULL_ARM_ID, "fold_country": country,
            "n_test": 100, "r2": -0.1, "rmse": 10.0, "mae": 8.0,
        })
        rows.append({
            "candidate_id": "cand_a", "fold_country": country,
            "n_test": 100, "r2": 0.02, "rmse": 9.0 - 0.01 * i, "mae": 7.0,
        })
    country = pd.DataFrame(rows)
    tests = paired_error_tests(
        country, {"null": NULL_ARM_ID, "o_min": "cand_a"}, "structural", "xgb_tree_d3"
    )
    row = tests.iloc[0]
    assert row["mean_delta_mse"] < 0
    assert row["countries_favouring_arm"] == len(countries)
    assert row["wilcoxon_p"] < 0.01
    assert row["rank_biserial"] == pytest.approx(-1.0)
    assert row["delta_mse_ci_high"] < 0, "bootstrap CI should exclude zero"
    # Audit columns are carried but are not the inferential statistic.
    assert row["audit_mean_country_r2_null"] == pytest.approx(-0.1)


def test_paired_error_test_reports_no_effect_when_arms_are_identical():
    rows = []
    for country in [f"C{i}" for i in range(10)]:
        for cid in (NULL_ARM_ID, "cand_a"):
            rows.append({
                "candidate_id": cid, "fold_country": country,
                "n_test": 50, "r2": 0.0, "rmse": 5.0, "mae": 4.0,
            })
    tests = paired_error_tests(
        pd.DataFrame(rows), {"null": NULL_ARM_ID, "o_min": "cand_a"}, "functional", "ols"
    )
    row = tests.iloc[0]
    assert row["mean_delta_mse"] == pytest.approx(0.0)
    assert np.isnan(row["wilcoxon_p"])


def test_landscape_ordering_is_reported_not_asserted():
    """Ordering must follow the data, including a reversal between rungs."""
    summary = pd.DataFrame([
        {"bag": "functional", "rung_id": "xgb_tree_d1", "arm": "o_min", "pooled_oof_r2_median": 0.0006},
        {"bag": "functional", "rung_id": "xgb_tree_d1", "arm": "o_max", "pooled_oof_r2_median": 0.0034},
        {"bag": "functional", "rung_id": "xgb_tree_d1", "arm": "single_exposure", "pooled_oof_r2_median": 0.0032},
        {"bag": "functional", "rung_id": "xgb_tree_d3", "arm": "o_min", "pooled_oof_r2_median": 0.0138},
        {"bag": "functional", "rung_id": "xgb_tree_d3", "arm": "o_max", "pooled_oof_r2_median": 0.0017},
        {"bag": "functional", "rung_id": "xgb_tree_d3", "arm": "single_exposure", "pooled_oof_r2_median": -0.0083},
    ])
    ordering = landscape_ordering(summary).set_index("rung_id")["observed_ordering"]
    assert ordering["xgb_tree_d1"] == "o_max > single_exposure > o_min"
    assert ordering["xgb_tree_d3"] == "o_min > o_max > single_exposure"


def test_target_design_identity_separates_resume_caches():
    """A covariate-free run must not resume a covariate-carrying cache."""
    import inspect

    from repo_expo_hoi_bag.stages import sensitivity_common

    source = inspect.getsource(sensitivity_common.evaluate_candidates_by_rung)
    assert "target_design_identity" in source
    assert "stamps_target_design" in source
    # Delivered stages must keep their existing identity so their caches stay valid.
    assert 'identity_columns = ["hpo_scope", "hpo_artifact_sha256"]' in source
