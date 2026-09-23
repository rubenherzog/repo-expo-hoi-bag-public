"""Focused contracts for paired hierarchical OOF contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.stages import compute_hierarchical_oof_comparisons as C


def _frame(loss_offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for country in ("a", "b"):
        for year in ("2018", "2019"):
            for subject in range(8):
                y = float(subject - 3) + (0.25 if country == "b" else 0.0)
                baseline = (y - 1.0) ** 2
                rows.append({"row_id": f"{country}-{year}-{subject}", "country": country,
                             "country_year": f"{country}_{year}", "y_true": y,
                             "mean_candidate_loss": baseline + loss_offset,
                             "baseline_loss": baseline})
    return pd.DataFrame(rows)


def test_observed_pair_is_country_balanced_r2_difference() -> None:
    left, right = _frame(-0.5), _frame(0.0)
    observed = C.observed_paired_r2(left, right, left_loss="mean_candidate_loss", right_loss="mean_candidate_loss")
    assert (observed["delta_r2_country"] > 0).all()
    assert np.isclose(observed["delta_r2_country"].mean(), observed["r2_left"].mean() - observed["r2_right"].mean())


def test_same_resample_makes_identical_losses_exactly_zero() -> None:
    frame = _frame()
    estimates, observed = C.bootstrap_paired_r2(frame, frame, draws=128, seed=12)
    assert np.array_equal(estimates, np.zeros(128))
    assert np.array_equal(observed["delta_r2_country"].to_numpy(), np.zeros(2))


def test_paired_bootstrap_is_worker_invariant() -> None:
    left, right = _frame(-0.5), _frame(0.0)
    sequential, _ = C.bootstrap_paired_r2(left, right, draws=128, seed=19, n_jobs=1)
    parallel, _ = C.bootstrap_paired_r2(left, right, draws=128, seed=19, n_jobs=4)
    assert np.array_equal(sequential, parallel)


def test_alignment_is_required() -> None:
    left, right = _frame(), _frame()
    right.loc[0, "country_year"] = "wrong"
    try:
        C.bootstrap_paired_r2(left, right, draws=4)
    except ValueError as error:
        assert "country_year" in str(error)
    else:
        raise AssertionError("misaligned OOF rows must be rejected")


def test_two_sided_p_is_bounded_and_holm_is_family_local() -> None:
    estimates = np.array([-1.0, -0.5, 0.2, 0.7])
    summary = C.summarise_paired(estimates, C.observed_paired_r2(_frame(), _frame(), left_loss="baseline_loss", right_loss="baseline_loss"))
    assert 0 < summary["p_two_sided"] <= 1
    rows = pd.DataFrame({"multiplicity_family": ["a", "a", "b"], "p_two_sided": [.01, .02, .01]})
    out = C.apply_holm(rows)
    assert out.loc[2, "holm_p"] == .01


def test_outer_parallel_contrasts_match_serial() -> None:
    left, right = _frame(-0.5), _frame(0.0)
    specs = [
        C.ContrastSpec("family", "question", "bag", "arm", "rung", "left", "right", left, right),
        C.ContrastSpec("family", "question", "bag", "arm", "rung", "left", "right", right, left),
    ]
    serial, serial_countries = C._execute_contrasts(specs, draws=64, seed=21, n_jobs=1)
    parallel, parallel_countries = C._execute_contrasts(specs, draws=64, seed=21, n_jobs=2)
    assert serial.equals(parallel)
    assert serial_countries.equals(parallel_countries)


def test_outer_contrast_preserves_its_region_label() -> None:
    frame = _frame()
    spec = C.ContrastSpec("family", "question", "bag", "arm", "rung", "Top5 set", "baseline", frame, frame, region="Top5", right_loss="baseline_loss")
    results, countries = C._execute_contrasts([spec], draws=8, seed=4, n_jobs=1)
    assert results.loc[0, "region"] == "Top5"
    assert set(countries["region"]) == {"Top5"}
