#!/usr/bin/env python3
"""Landscape-level model comparisons for the main k10 delivery.

The winner-based analysis in ``compute_hpo_model_comparisons`` answers a
question the paper does not ask: how good is *the* selected candidate.  The
exact argmax is unstable, the top region is not, and winner R² carries
post-selection optimism.  This stage replaces that inference with a
comparison of complete candidate *landscapes*.

For every BAG x objective x rung x held-out country ``c`` the landscape score
is

    M[c, k] = median R2 across the 20 candidates at set size k
    L[c]    = mean over k of M[c, k]

Candidates are summarised robustly within a set size, every set size
contributes equally, and no candidate is ever chosen by its BAG performance,
so the statistic is BAG-blind by construction.  The statistical unit is the
held-out country, paired across the two conditions being compared.

This stage performs no model fitting whatsoever: it reads the cached
per-candidate ``metrics_country.csv`` files and aggregates them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]

BAGS = ("structural", "functional")
OBJECTIVES = ("o_min", "o_max")
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
# The six pairwise rung contrasts, always stated as higher minus lower so a
# positive delta always means "the more flexible model scores higher".
RUNG_PAIRS = (
    ("xgb_tree_d1", "ols"),
    ("xgb_tree_d2", "ols"),
    ("xgb_tree_d3", "ols"),
    ("xgb_tree_d2", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d2"),
)
CANDIDATES_PER_SET_SIZE = 20
ORDER_MIN, ORDER_MAX = 3, 30
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260922

AGGREGATION_DEFINITION = (
    "Landscape R2: within each held-out country, the median R2 across the 20 "
    "BAG-blind candidates at each set size k, averaged with equal weight over "
    "the available set sizes. Single-exposure landscape: median R2 across all "
    "single-exposure candidates within a country. Baseline: the unique "
    "covariate-only model. Inference is paired across held-out countries."
)


# ---------------------------------------------------------------------------
# candidate metrics
# ---------------------------------------------------------------------------
def candidate_metrics_path(run_root: Path, bag: str, rung: str, scope: str) -> Path:
    """Canonical per-candidate metrics file, mirroring the production layout.

    OLS candidates live under ``ols/<bag>/ols`` and XGBoost rungs under
    ``xgb/<bag>/<rung>``; the multivariate scope is named ``ols`` for the OLS
    rung and ``k10`` for the tuned rungs, exactly as the winner stage resolves
    them.
    """
    family = run_root / "ols" / bag / "ols" if rung == "ols" else run_root / "xgb" / bag / rung
    if scope == "multivariate":
        scope = "ols" if rung == "ols" else "k10"
    return family / scope / "metrics_country.csv"


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """The production eligibility rule: a scored fold with a finite R2."""
    frame = frame.copy()
    frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
    frame["n_test"] = pd.to_numeric(frame["n_test"], errors="coerce")
    return frame[frame["n_test"].gt(0) & np.isfinite(frame["r2"])].copy()


def _with_candidate_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the objective and set size encoded in every candidate id."""
    parsed = frame["candidate_id"].astype(str).str.extract(
        r"_(?P<objective>o_min|o_max)_ord(?P<order>\d+)_"
    )
    if parsed["objective"].isna().any():
        unparsed = sorted(frame.loc[parsed["objective"].isna(), "candidate_id"].unique())[:5]
        raise ValueError(f"Candidate ids without an objective/order: {unparsed}")
    frame = frame.copy()
    frame["objective"] = parsed["objective"]
    frame["order"] = parsed["order"].astype(int)
    return frame


def landscape_scores(
    frame: pd.DataFrame,
    *,
    expected_per_cell: int | None = CANDIDATES_PER_SET_SIZE,
    order_min: int = ORDER_MIN,
    order_max: int = ORDER_MAX,
) -> pd.DataFrame:
    """Country-level landscape scores from per-candidate country metrics.

    Returns one row per country x objective with the landscape R2, the number
    of set sizes contributing and the number of candidates used.  An
    incomplete set-size cell is an error rather than a silently smaller
    median, so a partially evaluated pool can never masquerade as a complete
    landscape.
    """
    frame = _with_candidate_metadata(_eligible(frame))
    frame = frame[frame["order"].between(order_min, order_max)]
    if frame.empty:
        raise ValueError("No eligible candidate rows in the requested order range")

    counts = frame.groupby(["fold_country", "objective", "order"], observed=True)["candidate_id"].nunique()
    if expected_per_cell is not None:
        incomplete = counts[counts != expected_per_cell]
        if len(incomplete):
            raise ValueError(
                f"Incomplete candidate cells (expected {expected_per_cell}): "
                f"{incomplete.head(10).to_dict()}"
            )

    per_set_size = (
        frame.groupby(["fold_country", "objective", "order"], observed=True)["r2"].median().rename("median_r2")
    )
    grouped = per_set_size.groupby(["fold_country", "objective"], observed=True)
    scores = pd.concat(
        [grouped.mean().rename("landscape_r2"), grouped.size().rename("n_orders")], axis=1
    ).reset_index()
    used = (
        frame.groupby(["fold_country", "objective"], observed=True)["candidate_id"]
        .nunique()
        .rename("n_candidates")
        .reset_index()
    )
    return scores.merge(used, on=["fold_country", "objective"], validate="one_to_one")


def single_exposure_landscape(frame: pd.DataFrame) -> pd.DataFrame:
    """Median R2 across every single-exposure candidate, per country.

    This is deliberately the whole single-exposure *landscape*, not the best
    single exposure: a selected maximum must not be compared against an
    unselected multivariate landscape.
    """
    grouped = _eligible(frame).groupby("fold_country", observed=True)["r2"]
    return pd.concat(
        [grouped.median().rename("single_landscape_r2"), grouped.size().rename("n_single_candidates")],
        axis=1,
    ).reset_index()


def covariate_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """The unique covariate-only R2 per country.

    The baseline must be a single model; more than one candidate identity here
    would mean the baseline is not uniquely defined for this BAG x rung.
    """
    frame = _eligible(frame)
    identities = frame["candidate_id"].astype(str).nunique()
    if identities != 1:
        raise ValueError(f"Covariate baseline is not unique: {identities} candidate identities")
    duplicated = frame["fold_country"].duplicated().any()
    if duplicated:
        raise ValueError("Covariate baseline has more than one row per country")
    return frame[["fold_country", "r2"]].rename(columns={"r2": "baseline_r2"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# paired country inference
# ---------------------------------------------------------------------------
def _rank_biserial(differences: np.ndarray) -> float:
    """Rank-biserial correlation for the paired signed-rank test."""
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank().to_numpy(float)
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total > 0 else 0.0


def paired_country_test(
    a: pd.Series,
    b: pd.Series,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Two-sided paired Wilcoxon across countries plus a paired bootstrap CI.

    ``a`` and ``b`` are indexed by country; only countries present in both are
    used, which keeps every comparison exactly paired.
    """
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if paired.empty:
        raise ValueError("No paired countries available")
    differences = (paired["a"] - paired["b"]).to_numpy(float)
    n = len(differences)

    nonzero = differences[differences != 0]
    if len(nonzero):
        statistic, p_raw = stats.wilcoxon(nonzero, alternative="two-sided")
        statistic, p_raw = float(statistic), float(p_raw)
    else:
        statistic, p_raw = np.nan, 1.0

    # Paired country bootstrap: resample countries, not participants, because
    # the country is the statistical unit throughout.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.repeat(1 / n, n), size=draws)
    delta = counts @ differences / n

    return {
        "n_countries": int(n),
        "estimate_a": float(paired["a"].mean()),
        "estimate_b": float(paired["b"].mean()),
        "delta_mean_r2": float(differences.mean()),
        "median_paired_delta": float(np.median(differences)),
        "ci_lo": float(np.percentile(delta, 2.5)),
        "ci_hi": float(np.percentile(delta, 97.5)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_raw": p_raw,
        "rank_biserial": _rank_biserial(differences),
    }


def apply_holm(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust within each declared multiplicity family."""
    result = frame.copy()
    result["holm_p"] = np.nan
    for _, index in result.groupby("multiplicity_family", sort=False).groups.items():
        values = result.loc[index, "wilcoxon_p_raw"].astype(float)
        result.loc[index, "holm_p"] = multipletests(values, method="holm")[1]
    return result


# ---------------------------------------------------------------------------
# analysis assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LandscapeInputs:
    """Every cached table one BAG needs, plus the paths behind them."""

    bag: str
    scores: pd.DataFrame          # country x objective x rung landscape R2
    single: pd.DataFrame          # country x rung single-exposure landscape
    baseline: pd.DataFrame        # country x rung covariate baseline
    paths: dict[str, str]


def _series(frame: pd.DataFrame, column: str, **filters: str) -> pd.Series:
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    return selected.set_index("fold_country")[column]


def load_bag(
    run_root: Path,
    bag: str,
    *,
    order_min: int = ORDER_MIN,
    order_max: int = ORDER_MAX,
    expected_per_cell: int | None = CANDIDATES_PER_SET_SIZE,
) -> LandscapeInputs:
    """Read and aggregate every cached metrics file for one BAG."""
    scores: list[pd.DataFrame] = []
    singles: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    paths: dict[str, str] = {}

    for rung in RUNGS:
        multivariate_path = candidate_metrics_path(run_root, bag, rung, "multivariate")
        single_path = candidate_metrics_path(run_root, bag, rung, "single")
        baseline_path = candidate_metrics_path(run_root, bag, rung, "baseline")
        paths[f"{bag}/{rung}/multivariate"] = str(multivariate_path)
        paths[f"{bag}/{rung}/single"] = str(single_path)
        paths[f"{bag}/{rung}/baseline"] = str(baseline_path)

        frame = landscape_scores(
            pd.read_csv(multivariate_path),
            expected_per_cell=expected_per_cell,
            order_min=order_min,
            order_max=order_max,
        )
        scores.append(frame.assign(bag=bag, rung=rung))
        singles.append(single_exposure_landscape(pd.read_csv(single_path)).assign(bag=bag, rung=rung))
        baselines.append(covariate_baseline(pd.read_csv(baseline_path)).assign(bag=bag, rung=rung))

    return LandscapeInputs(
        bag=bag,
        scores=pd.concat(scores, ignore_index=True),
        single=pd.concat(singles, ignore_index=True),
        baseline=pd.concat(baselines, ignore_index=True),
        paths=paths,
    )


def comparison_rows(inputs: LandscapeInputs, *, sensitivity: str = "main_k10") -> pd.DataFrame:
    """Every landscape contrast for one BAG, with its multiplicity family."""
    bag = inputs.bag
    rows: list[dict[str, object]] = []
    seed = BOOTSTRAP_SEED

    def record(**fields: object) -> None:
        rows.append({"sensitivity": sensitivity, "bag": bag, **fields})

    # A. synergy vs redundancy landscape, within each rung.
    for index, rung in enumerate(RUNGS):
        a = _series(inputs.scores, "landscape_r2", rung=rung, objective="o_min")
        b = _series(inputs.scores, "landscape_r2", rung=rung, objective="o_max")
        record(
            comparison_family="synergy_vs_redundancy",
            objective="o_min_minus_o_max",
            model_a="synergy_landscape",
            model_b="redundancy_landscape",
            rung_a=rung,
            rung_b=rung,
            multiplicity_family=f"{sensitivity}|{bag}|synergy_vs_redundancy",
            **paired_country_test(a, b, seed=seed + index),
        )

    # B. model complexity, within each objective.
    for objective_index, objective in enumerate(OBJECTIVES):
        for pair_index, (high, low) in enumerate(RUNG_PAIRS):
            a = _series(inputs.scores, "landscape_r2", rung=high, objective=objective)
            b = _series(inputs.scores, "landscape_r2", rung=low, objective=objective)
            record(
                comparison_family="model_complexity",
                objective=objective,
                model_a=f"{objective}_landscape",
                model_b=f"{objective}_landscape",
                rung_a=high,
                rung_b=low,
                multiplicity_family=f"{sensitivity}|{bag}|{objective}|model_complexity",
                **paired_country_test(a, b, seed=seed + 100 + 10 * objective_index + pair_index),
            )

    # C. multivariate landscape vs the single-exposure landscape. A sensitivity
    # that never evaluated a single-exposure scope simply omits this family.
    for objective_index, objective in enumerate(OBJECTIVES if not inputs.single.empty else ()):
        for rung_index, rung in enumerate(RUNGS):
            a = _series(inputs.scores, "landscape_r2", rung=rung, objective=objective)
            b = _series(inputs.single, "single_landscape_r2", rung=rung)
            record(
                comparison_family="vs_single_landscape",
                objective=objective,
                model_a=f"{objective}_landscape",
                model_b="single_exposure_landscape",
                rung_a=rung,
                rung_b=rung,
                multiplicity_family=f"{sensitivity}|{bag}|vs_single_landscape",
                **paired_country_test(a, b, seed=seed + 300 + 10 * objective_index + rung_index),
            )

    # D. multivariate landscape vs the covariate baseline.
    for objective_index, objective in enumerate(OBJECTIVES):
        for rung_index, rung in enumerate(RUNGS):
            a = _series(inputs.scores, "landscape_r2", rung=rung, objective=objective)
            b = _series(inputs.baseline, "baseline_r2", rung=rung)
            record(
                comparison_family="vs_covariate_baseline",
                objective=objective,
                model_a=f"{objective}_landscape",
                model_b="covariate_baseline",
                rung_a=rung,
                rung_b=rung,
                multiplicity_family=f"{sensitivity}|{bag}|vs_covariate_baseline",
                **paired_country_test(a, b, seed=seed + 500 + 10 * objective_index + rung_index),
            )

    frame = pd.DataFrame(rows)
    frame["aggregation_definition"] = AGGREGATION_DEFINITION
    return frame


def country_score_table(bundles: list[LandscapeInputs]) -> pd.DataFrame:
    """The intermediate country-level scores, with single and baseline joined."""
    frames = []
    for inputs in bundles:
        frame = inputs.scores.merge(inputs.single, on=["bag", "rung", "fold_country"], validate="many_to_one")
        frame = frame.merge(inputs.baseline, on=["bag", "rung", "fold_country"], validate="many_to_one")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"fold_country": "country"})
    return combined[
        [
            "bag",
            "country",
            "objective",
            "rung",
            "n_orders",
            "n_candidates",
            "landscape_r2",
            "single_landscape_r2",
            "n_single_candidates",
            "baseline_r2",
        ]
    ].sort_values(["bag", "objective", "rung", "country"], ignore_index=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


EXPECTED_FAMILY_SIZES = {
    "synergy_vs_redundancy": len(RUNGS),
    "model_complexity": len(RUNG_PAIRS),
    "vs_single_landscape": len(OBJECTIVES) * len(RUNGS),
    "vs_covariate_baseline": len(OBJECTIVES) * len(RUNGS),
}


def validate_families(frame: pd.DataFrame) -> None:
    """Every Holm family must hold exactly the number of tests it declares."""
    sizes = frame.groupby(["multiplicity_family", "comparison_family"], sort=False).size()
    for (family, kind), size in sizes.items():
        expected = EXPECTED_FAMILY_SIZES[kind]
        if size != expected:
            raise ValueError(f"Holm family {family!r} has {size} tests, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/landscape_comparisons/main_k10",
    )
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    run_root = args.repro_data_root.resolve() / "results/analysis_runs" / args.source_run_id
    bundles = [load_bag(run_root, bag) for bag in BAGS]

    scores = country_score_table(bundles)
    tests = apply_holm(pd.concat([comparison_rows(inputs) for inputs in bundles], ignore_index=True))
    validate_families(tests)

    columns = [
        "comparison_family", "sensitivity", "bag", "objective", "model_a", "model_b",
        "rung_a", "rung_b", "n_countries", "estimate_a", "estimate_b", "delta_mean_r2",
        "median_paired_delta", "ci_lo", "ci_hi", "wilcoxon_statistic", "wilcoxon_p_raw",
        "holm_p", "rank_biserial", "multiplicity_family", "aggregation_definition",
    ]
    tests = tests[columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "landscape_country_scores.csv", index=False)
    tests.to_csv(args.output_dir / "landscape_comparison_tests.csv", index=False)

    paths = {key: value for inputs in bundles for key, value in inputs.paths.items()}
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_landscape_model_comparisons",
        "source_run": str(run_root),
        "model_fitting_performed": False,
        "statement": (
            "No model fitting was performed. Every value is an aggregation of "
            "cached per-candidate held-out-country metrics."
        ),
        "inputs": {key: {"path": value, "sha256": _sha256(Path(value))} for key, value in sorted(paths.items())},
        "candidate_counts": {
            f"{row.bag}/{row.rung}/{row.objective}": int(row.n_candidates)
            for row in scores.drop_duplicates(["bag", "rung", "objective"]).itertuples()
        },
        "single_exposure_candidates": {
            f"{row.bag}/{row.rung}": int(row.n_single_candidates)
            for row in scores.drop_duplicates(["bag", "rung"]).itertuples()
        },
        "country_counts": scores.groupby("bag")["country"].nunique().to_dict(),
        "order_range": [ORDER_MIN, ORDER_MAX],
        "candidates_per_set_size": CANDIDATES_PER_SET_SIZE,
        "statistical_definitions": {
            "landscape_r2": AGGREGATION_DEFINITION,
            "primary_test": "two-sided Wilcoxon signed-rank across held-out countries",
            "effect_size": "rank-biserial correlation",
            "interval": "95% percentile CI from a paired country bootstrap",
            "eligibility": "n_test > 0 and finite R2",
        },
        "bootstrap": {"draws": args.draws, "seed": BOOTSTRAP_SEED},
        "multiplicity_families": {
            family: int(size)
            for family, size in tests.groupby("multiplicity_family", sort=False).size().items()
        },
    }
    (args.output_dir / "landscape_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )
    print(f"Saved: {args.output_dir / 'landscape_comparison_tests.csv'} ({len(tests)} tests)")


if __name__ == "__main__":
    main()
