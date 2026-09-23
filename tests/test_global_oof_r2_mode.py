"""Contract tests for the global pooled OOF R2 estimand.

These guard the two failure modes that a dual-estimand delivery invites:
mixing the two estimands inside one artifact, and silently reporting one
estimand under the other one's name.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "repo_expo_hoi_bag" / "core"))
sys.path.insert(0, str(ROOT / "src" / "repo_expo_hoi_bag" / "stages"))


@pytest.fixture()
def sc(monkeypatch):
    import sensitivity_common

    monkeypatch.delenv("R2_MODE", raising=False)
    return sensitivity_common


# --------------------------------------------------------------------------- #
# Formula
# --------------------------------------------------------------------------- #
def test_global_oof_r2_matches_plain_r2_on_pooled_predictions(sc) -> None:
    rng = np.random.default_rng(20260304)
    y = rng.normal(70.0, 8.0, size=900)
    pred = 0.55 * y + rng.normal(0.0, 4.0, size=900)
    assert sc.global_oof_r2(y, pred) == pytest.approx(r2_score(y, pred), abs=1e-12)


def test_global_oof_r2_drops_only_non_finite_pairs(sc) -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    pred = y * 0.4 + rng.normal(scale=0.5, size=400)
    y_missing, pred_missing = y.copy(), pred.copy()
    y_missing[5] = np.nan
    pred_missing[11] = np.inf
    keep = np.isfinite(y_missing) & np.isfinite(pred_missing)
    assert sc.global_oof_r2(y_missing, pred_missing) == pytest.approx(
        r2_score(y_missing[keep], pred_missing[keep]), abs=1e-12
    )


def test_global_oof_r2_does_not_average_over_countries(sc) -> None:
    """Two countries with different target means must not collapse to a mean of R2."""
    rng = np.random.default_rng(11)
    y_a = rng.normal(75.0, 6.0, size=300)
    y_b = rng.normal(55.0, 5.0, size=200)
    pred_a = 0.5 * y_a + 37.0 + rng.normal(0.0, 4.0, size=300)
    pred_b = 0.5 * y_b + 27.0 + rng.normal(0.0, 4.0, size=200)
    pooled = sc.global_oof_r2(np.r_[y_a, y_b], np.r_[pred_a, pred_b])
    mean_of_country_r2 = float(np.mean([r2_score(y_a, pred_a), r2_score(y_b, pred_b)]))
    assert pooled != pytest.approx(mean_of_country_r2, abs=1e-6)
    # Between-country mean differences belong in the pooled total sum of squares.
    within_only = 1.0 - (
        np.sum((y_a - pred_a) ** 2) + np.sum((y_b - pred_b) ** 2)
    ) / (
        np.sum((y_a - y_a.mean()) ** 2) + np.sum((y_b - y_b.mean()) ** 2)
    )
    assert pooled != pytest.approx(float(within_only), abs=1e-6)


# --------------------------------------------------------------------------- #
# Mode selector
# --------------------------------------------------------------------------- #
def test_default_mode_is_country_balanced(sc) -> None:
    assert sc.r2_mode() == "country_balanced"
    assert sc.global_oof_mode() is False
    assert sc.r2_mode_suffix() == ""


def test_global_mode_is_opt_in_and_suffixed(sc, monkeypatch) -> None:
    monkeypatch.setenv("R2_MODE", "global_oof")
    assert sc.r2_mode() == "global_oof"
    assert sc.global_oof_mode() is True
    assert sc.r2_mode_suffix() == "_global_oof"


def test_unknown_mode_is_rejected_rather_than_defaulted(sc, monkeypatch) -> None:
    monkeypatch.setenv("R2_MODE", "country-balanced")  # wrong spelling
    with pytest.raises(ValueError, match="not a supported estimand"):
        sc.r2_mode()


def test_r2_column_refuses_to_fall_back_to_the_other_estimand(sc, monkeypatch) -> None:
    balanced_only = pd.DataFrame({"candidate_id": ["a"], "country_balanced_r2": [0.2]})
    assert sc.r2_column(balanced_only) == "country_balanced_r2"
    monkeypatch.setenv("R2_MODE", "global_oof")
    with pytest.raises(ValueError, match="global_oof"):
        sc.r2_column(balanced_only)


# --------------------------------------------------------------------------- #
# Delivery contract
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_global_delivery_is_a_separate_root_from_the_country_balanced_one() -> None:
    balanced = _load("main_paper_delivery.yaml")
    global_oof = _load("main_paper_delivery_global_oof.yaml")
    assert balanced["delivery_root"] == "outputs/main/paper/complete"
    assert global_oof["delivery_root"] == "outputs/main/paper/complete_global_oof"
    assert global_oof["delivery_root"] != balanced["delivery_root"]
    assert global_oof["r2_mode"] == "global_oof"
    assert global_oof["country_balanced_counterpart"] == balanced["delivery_root"]


def test_both_deliveries_carry_the_identical_scientific_inventory() -> None:
    """Only the estimand may differ: same figures, same 18 sheets, same exclusions."""
    balanced = _load("main_paper_delivery.yaml")
    global_oof = _load("main_paper_delivery_global_oof.yaml")
    for key in ("main_figures", "supplementary_figures", "excluded"):
        assert global_oof[key] == balanced[key], key
    assert global_oof["supplementary_tables"]["sheets"] == balanced["supplementary_tables"]["sheets"]
    assert len(global_oof["supplementary_tables"]["sheets"]) == 18


def test_country_block_null_labels_follow_the_active_mode(monkeypatch) -> None:
    """ST04 must never stamp one estimand's name on the other's draws."""
    import sensitivity_common

    monkeypatch.delenv("R2_MODE", raising=False)
    assert sensitivity_common.r2_mode() == "country_balanced"
    monkeypatch.setenv("R2_MODE", "global_oof")
    assert sensitivity_common.r2_mode() == "global_oof"
    source = (ROOT / "src/repo_expo_hoi_bag/stages/run_country_block_null.py").read_text()
    assert '"r2_estimand": "country_balanced"' not in source
    assert '"r2_estimand": r2_mode()' in source
    fast = (ROOT / "src/repo_expo_hoi_bag/stages/cbn_fast.py").read_text()
    assert '"country_balanced"' not in fast.replace('R2_MODE_COUNTRY_BALANCED', '')


# --------------------------------------------------------------------------- #
# Cross-mode write isolation
# --------------------------------------------------------------------------- #
def test_global_finalizer_never_writes_into_the_country_balanced_namespace() -> None:
    """The global run must not overwrite paper_reanalysis_k10 shared statistics."""
    script = (ROOT / "scripts/finalize_main_k10_global_oof.sh").read_text(encoding="utf-8")
    assert "R2_MODE=global_oof" in script
    # compute_hpo_model_comparisons defaults to the source run's namespace, so a
    # global run must pass an explicit --output-run-id.
    comparisons = script.split("compute_hpo_model_comparisons", 1)[1].split("\n\n", 1)[0]
    assert "--output-run-id" in comparisons
    assert "outputs/main/paper/complete\"" not in script
    assert "$RUN_ID" in script


def test_global_delivery_root_is_never_the_country_balanced_root() -> None:
    collector = (ROOT / "src/repo_expo_hoi_bag/stages/collect_global_oof_delivery.py").read_text()
    assert "Refusing to write the global delivery into the country-balanced root" in collector


def test_global_sensitivity_stages_refuse_to_overwrite_existing_figures() -> None:
    for name in (
        "render_global_oof_sensitivity_figures.py",
        "render_global_oof_set_size.py",
    ):
        text = (ROOT / "src/repo_expo_hoi_bag/stages" / name).read_text(encoding="utf-8")
        assert "FileExistsError" in text, name
        assert "Refusing to overwrite" in text, name


def test_country_block_null_launcher_is_global_and_non_overwriting() -> None:
    script = (ROOT / "scripts/submit_main_k10_global_oof_country_block_null.sh").read_text()
    assert "R2_MODE=global_oof" in script
    assert "Refusing to overwrite existing global ST04 result" in script
    assert "CBN_N_PERM" in script
