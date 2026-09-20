"""Pure helpers for the frozen-HPO endpoint and top-50 comparison."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def baseline_candidate() -> pd.DataFrame:
    """Return the covariate-only candidate in evaluator-compatible form."""
    return pd.DataFrame([{
        "candidate_id": "__baseline__",
        "feature_id": "__baseline__",
        "objective": "baseline",
        "order": 0,
        "rank": 1,
        "score": np.nan,
        "nplet_vars": [],
        "predictors_identity": "",
        "predictors_identity_n": 0,
        "candidate_family": "baseline",
        "source_label": "frozen_hpo_endpoint_comparison",
    }])


def single_exposure_candidates(features: Sequence[str]) -> pd.DataFrame:
    """Return one stable candidate per canonical exposure."""
    names = [str(feature) for feature in features]
    if not names or len(set(names)) != len(names):
        raise ValueError("Canonical single-exposure candidates require unique, non-empty features")
    return pd.DataFrame([{
        "candidate_id": f"__single__{feature}",
        "feature_id": feature,
        "objective": "single_exposure",
        "order": 1,
        "rank": index + 1,
        "score": np.nan,
        "nplet_vars": [feature],
        "predictors_identity": feature,
        "predictors_identity_n": 1,
        "candidate_family": "canonical_single_exposure",
        "source_label": "frozen_hpo_endpoint_comparison",
    } for index, feature in enumerate(names)])


def country_balanced_scores(country_metrics: pd.DataFrame, *, expected_countries: int) -> pd.DataFrame:
    """Compute equal-country R² for every fully evaluated candidate."""
    required = {"candidate_id", "rung_id", "fold_country", "r2"}
    missing = sorted(required.difference(country_metrics.columns))
    if missing:
        raise ValueError(f"Country metrics are missing required columns: {missing}")
    scoped = country_metrics.copy()
    scoped["r2"] = pd.to_numeric(scoped["r2"], errors="coerce")
    coverage = scoped.groupby(["candidate_id", "rung_id"], as_index=False).agg(
        country_balanced_r2=("r2", "mean"),
        countries_observed=("fold_country", "nunique"),
        r2_nonmissing=("r2", lambda values: int(pd.Series(values).notna().sum())),
    )
    valid = coverage[
        coverage["countries_observed"].eq(expected_countries)
        & coverage["r2_nonmissing"].eq(expected_countries)
    ].copy()
    if len(valid) != len(coverage):
        raise ValueError("At least one candidate lacks complete finite country-level R² coverage")
    return valid


def endpoint_summary(
    baseline: pd.DataFrame,
    single: pd.DataFrame,
    top50_full63: pd.DataFrame,
    top50_k10: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one deterministic country-balanced comparison row per BAG × rung."""
    inputs = {
        "baseline": baseline,
        "single": single,
        "top50_full63": top50_full63,
        "top50_k10": top50_k10,
    }
    for label, frame in inputs.items():
        required = {"bag_target", "rung_id", "candidate_id", "country_balanced_r2"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} scores are missing required columns: {missing}")
    keys = sorted(set(map(tuple, baseline[["bag_target", "rung_id"]].astype(str).to_numpy())))
    rows: list[dict] = []
    for bag, rung in keys:
        def subset(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[(frame["bag_target"].astype(str) == bag) & (frame["rung_id"].astype(str) == rung)].copy()
        base = subset(baseline)
        singles = subset(single)
        full = subset(top50_full63)
        k10 = subset(top50_k10)
        if len(base) != 1 or len(singles) == 0 or len(full) != 50 or len(k10) != 50:
            raise ValueError(f"Incomplete endpoint/top-50 score panel for {bag}/{rung}")
        choose = lambda frame: frame.sort_values(["country_balanced_r2", "candidate_id"], ascending=[False, True]).iloc[0]
        base_row, single_row, full_best, k10_best = choose(base), choose(singles), choose(full), choose(k10)
        baseline_r2 = float(base_row["country_balanced_r2"])
        single_r2 = float(single_row["country_balanced_r2"])
        row = {
            "bag_target": bag,
            "rung_id": rung,
            "baseline_country_balanced_r2": baseline_r2,
            "best_single_candidate_id": str(single_row["candidate_id"]),
            "best_single_country_balanced_r2": single_r2,
            "delta_best_single_minus_baseline": single_r2 - baseline_r2,
        }
        for label, panel, best in (("full63", full, full_best), ("k10", k10, k10_best)):
            best_r2 = float(best["country_balanced_r2"])
            median_r2 = float(panel["country_balanced_r2"].median())
            row.update({
                f"top50_{label}_best_candidate_id": str(best["candidate_id"]),
                f"top50_{label}_best_country_balanced_r2": best_r2,
                f"top50_{label}_median_country_balanced_r2": median_r2,
                f"delta_top50_{label}_best_minus_baseline": best_r2 - baseline_r2,
                f"delta_top50_{label}_best_minus_best_single": best_r2 - single_r2,
                f"delta_top50_{label}_median_minus_best_single": median_r2 - single_r2,
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["bag_target", "rung_id"]).reset_index(drop=True)
