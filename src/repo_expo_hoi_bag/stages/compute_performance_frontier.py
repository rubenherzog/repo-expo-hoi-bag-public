#!/usr/bin/env python3
"""Country-cross-selected performance-frontier analysis for the main k10 delivery.

This stage sits between two extremes that the paper has already rejected:

*   the single argmax candidate, which is exposed to winner's curse and whose
    identity the selection-stability diagnostic showed to be unstable;
*   the whole BAG-blind candidate pool, whose median answers a different
    question, because O-information discovery is independent of BAG and most
    candidates are not expected to carry BAG-relevant information.

The object of interest is the *high-performing region* of the candidate space.
For every BAG x arm x rung and every held-out country ``c`` the top-K
candidates are chosen by their unweighted mean R2 over the countries other than
``c`` -- the production selection rule, with the production tie-break -- and the
frontier score is the median of their already-computed R2 *in* ``c``:

    F[c, arm, rung, K] = median_{i in TopK[-c]} R2(i, c)

IMPORTANT INTERPRETATION LIMIT
------------------------------
This is a *country-cross-selected frontier*, not nested cross-validation.
Exclusion happens at the selection-metric level only. Each candidate's fits came
from the full LOCO design, so the fit whose R2 is read back for country ``c`` is
out-of-sample with respect to ``c``, but the candidate pool itself was not
re-derived without ``c``. Nothing here licenses the term "nested CV".

No model is fitted. Every number aggregates cached per-candidate per-country
LOCO metrics.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.stages.compute_landscape_model_comparisons import (
    apply_holm,
    paired_country_test,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
)
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import (
    _domain_map,
    _domains,
    _jaccard,
    _predictor_sets,
    _ranked,
    _registry,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[3]

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
# Unlike the winner diagnostic, the frontier analysis spans all four model
# levels, so OLS is included and its distinct on-disk layout is handled below.
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
RUNG_PAIRS = (
    ("xgb_tree_d1", "ols"),
    ("xgb_tree_d2", "ols"),
    ("xgb_tree_d3", "ols"),
    ("xgb_tree_d2", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d2"),
)
# The matched-candidate sensitivity only concerns the interaction-capable
# contrasts; OLS pairs are not at issue there.
MATCHED_PAIRS = (
    ("xgb_tree_d2", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d2"),
)
PRIMARY_K = 20
K_VALUES = (5, 10, 20, 50)
K_REPORTED = (10, 20, 50)
MAX_ORDER = 30

SELECTION_RULE = (
    "Candidates are ranked by the unweighted mean R2 across eligible held-out "
    "countries (n_test > 0) other than the evaluation country, restricted to "
    "set sizes <= 30, with ties broken on ascending candidate_id using a stable "
    "sort. This is the production selection rule with the evaluation country "
    "removed from the selection metric only."
)
FRONTIER_DEFINITION = (
    "F[c, arm, rung, K] is the median already-computed R2 in held-out country c "
    "across the K candidates with the highest mean R2 over all countries except "
    "c. Inference is paired across held-out countries."
)
NESTING_CAVEAT = (
    "Country-cross-selected, not nested cross-validation. The evaluation "
    "country is excluded from the selection metric only; each candidate's fits "
    "come from the full LOCO design and the candidate pool was not re-derived "
    "without the evaluation country."
)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def metrics_path(runtime: Path, source_run_id: str, bag: str, rung: str, scope: str) -> Path:
    """Canonical per-candidate metrics file for one BAG x rung x scope.

    Mirrors the production layout: OLS candidates live under ``ols/<bag>/ols``
    with a multivariate scope named ``ols``; tuned rungs live under
    ``xgb/<bag>/<rung>`` with scope ``k10``.
    """
    run_root = runtime / "results" / "analysis_runs" / source_run_id
    family = run_root / "ols" / bag / "ols" if rung == "ols" else run_root / "xgb" / bag / rung
    if scope == "multivariate":
        scope = "ols" if rung == "ols" else "k10"
    return family / scope / "metrics_country.csv"


def _eligible_metrics(path: Path) -> pd.DataFrame:
    """Scored folds with a finite R2, the production eligibility rule."""
    frame = pd.read_csv(path)
    for column in ("candidate_id", "fold_country", "n_test", "r2"):
        if column not in frame.columns:
            raise ValueError(f"{path} lacks required column {column!r}")
    frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
    frame = frame[pd.to_numeric(frame["n_test"], errors="coerce").gt(0)]
    if frame["r2"].isna().any():
        raise ValueError(f"Non-numeric R2 in a scored fold: {path}")
    return frame


def _wide(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Candidate x country R2 matrix; every cell must be populated."""
    wide = frame.pivot_table(index="candidate_id", columns="fold_country", values="r2", aggfunc="mean")
    if wide.isna().any().any():
        raise ValueError(f"Incomplete candidate x country grid for {label}")
    return wide


@dataclass(frozen=True)
class FrontierInputs:
    """Every cached matrix one BAG needs, keyed by rung."""

    bag: str
    multivariate: dict[str, pd.DataFrame]   # rung -> candidate x country R2
    single: dict[str, pd.DataFrame]         # rung -> single-exposure matrix
    baseline: dict[str, pd.Series]          # rung -> country -> baseline R2
    arms: dict[str, list[str]]              # arm -> candidate ids
    predictors: dict[str, frozenset[str]]   # candidate id -> exposures
    paths: dict[str, str]


def load_bag(runtime: Path, source_run_id: str, bag: str, registry: pd.DataFrame) -> FrontierInputs:
    identity = registry[registry["bag"].eq(bag)].drop(columns=["bag"])
    identity = identity[pd.to_numeric(identity["order"], errors="raise").le(MAX_ORDER)]
    arms = {
        arm: sorted(identity.loc[identity["objective"].eq(arm), "candidate_id"].astype(str))
        for arm in ARMS
    }
    predictors = _predictor_sets(identity)

    multivariate: dict[str, pd.DataFrame] = {}
    single: dict[str, pd.DataFrame] = {}
    baseline: dict[str, pd.Series] = {}
    paths: dict[str, str] = {}

    for rung in RUNGS:
        for scope, sink in (("multivariate", multivariate), ("single", single), ("baseline", None)):
            path = metrics_path(runtime, source_run_id, bag, rung, scope)
            if not path.is_file():
                raise FileNotFoundError(f"Required main k10 input is missing: {path}")
            paths[f"{bag}/{rung}/{scope}"] = str(path)
            frame = _eligible_metrics(path)
            if scope == "baseline":
                identities = frame["candidate_id"].astype(str).nunique()
                if identities != 1:
                    raise ValueError(f"Covariate baseline is not unique for {bag}/{rung}")
                if frame["fold_country"].duplicated().any():
                    raise ValueError(f"Baseline has repeated countries for {bag}/{rung}")
                baseline[rung] = frame.set_index("fold_country")["r2"].astype(float)
            else:
                sink[rung] = _wide(frame, f"{bag}/{rung}/{scope}")

        expected = set(arms["o_min"]) | set(arms["o_max"])
        found = set(multivariate[rung].index.astype(str))
        if found != expected:
            raise ValueError(f"Candidate membership mismatch for {bag}/{rung}")

    return FrontierInputs(bag, multivariate, single, baseline, arms, predictors, paths)


# ---------------------------------------------------------------------------
# the frontier itself
# ---------------------------------------------------------------------------
def top_k(wide: pd.DataFrame, k: int, *, exclude: str | None = None) -> list[str]:
    """The K best candidate ids under the production ranking rule.

    ``exclude`` removes a country from the *selection metric* only. The returned
    ids are ordered best-first.
    """
    scores = wide.drop(columns=[exclude]) if exclude is not None else wide
    if scores.shape[1] == 0:
        raise ValueError("No countries remain in the selection metric")
    ranked = _ranked(scores.mean(axis=1).rename("mean_r2").reset_index(), "mean_r2")
    if len(ranked) < k:
        raise ValueError(f"Only {len(ranked)} candidates available for K={k}")
    selected = list(ranked.head(k)["candidate_id"].astype(str))
    if len(set(selected)) != k:
        raise ValueError(f"Top-{k} selection produced duplicate candidate ids")
    return selected


def frontier_score(wide: pd.DataFrame, candidates: list[str], country: str) -> tuple[float, float]:
    """Median and mean R2 of a candidate set evaluated in one country."""
    values = wide.loc[candidates, country].to_numpy(float)
    return float(np.median(values)), float(values.mean())


def _union(candidates: list[str], predictors: dict[str, frozenset[str]]) -> frozenset[str]:
    return frozenset().union(*(predictors[candidate] for candidate in candidates))


def build_frontier(
    inputs: FrontierInputs,
    domain_map: dict[str, str],
    *,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Country scores, stability summaries and selection frequencies for one BAG."""
    score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []

    for arm in ARMS:
        members = inputs.arms[arm]
        for rung in RUNGS:
            wide = inputs.multivariate[rung].loc[members]
            countries = sorted(str(value) for value in wide.columns)
            global_ranked = _ranked(wide.mean(axis=1).rename("mean_r2").reset_index(), "mean_r2")
            global_rank = {
                str(row.candidate_id): index + 1
                for index, row in enumerate(global_ranked.itertuples(index=False))
            }

            summary: dict[str, object] = {
                "bag": inputs.bag,
                "arm": arm,
                "rung": rung,
                "n_candidates": int(wide.shape[0]),
                "n_countries": len(countries),
            }

            for k in k_values:
                global_top = top_k(wide, k)
                global_set = frozenset(global_top)
                global_exposures = _union(global_top, inputs.predictors)
                global_domains = frozenset().union(
                    *(_domains(inputs.predictors[c], domain_map) for c in global_top)
                )
                # Spread of the high-performing region: rank-1 minus rank-K.
                summary[f"top{k}_global_rank1_minus_rankK_r2"] = float(
                    global_ranked.iloc[0]["mean_r2"] - global_ranked.iloc[k - 1]["mean_r2"]
                )

                candidate_jaccards: list[float] = []
                exposure_jaccards: list[float] = []
                domain_jaccards: list[float] = []
                retained_counts: list[int] = []
                membership: dict[str, int] = {candidate: 0 for candidate in members}

                for country in countries:
                    selected = top_k(wide, k, exclude=country)
                    selected_set = frozenset(selected)
                    median_r2, mean_r2 = frontier_score(wide, selected, country)
                    exposures = _union(selected, inputs.predictors)
                    domains = frozenset().union(
                        *(_domains(inputs.predictors[c], domain_map) for c in selected)
                    )

                    candidate_jaccard = _jaccard(selected_set, global_set)
                    exposure_jaccard = _jaccard(exposures, global_exposures)
                    domain_jaccard = _jaccard(domains, global_domains)

                    candidate_jaccards.append(candidate_jaccard)
                    exposure_jaccards.append(exposure_jaccard)
                    domain_jaccards.append(domain_jaccard)
                    retained_counts.append(len(selected_set & global_set))
                    for candidate in selected:
                        membership[candidate] += 1

                    score_rows.append(
                        {
                            "bag": inputs.bag,
                            "arm": arm,
                            "rung": rung,
                            "excluded_country": country,
                            "K": k,
                            "frontier_median_r2": median_r2,
                            "frontier_mean_r2": mean_r2,
                            "selected_candidate_ids": "|".join(selected),
                            "global_frontier_candidate_jaccard": candidate_jaccard,
                            "global_frontier_exposure_jaccard": exposure_jaccard,
                            "global_frontier_domain_jaccard": domain_jaccard,
                            "n_global_topk_retained": len(selected_set & global_set),
                        }
                    )

                for candidate, count in membership.items():
                    if count == 0 and candidate not in global_set:
                        continue
                    frequency_rows.append(
                        {
                            "bag": inputs.bag,
                            "arm": arm,
                            "rung": rung,
                            "K": k,
                            "candidate_id": candidate,
                            "n_country_frontiers": count,
                            "fraction_of_countries": count / len(countries),
                            "global_rank": global_rank[candidate],
                            "in_global_topk": candidate in global_set,
                            "set_size": len(inputs.predictors[candidate]),
                            "predictors_identity": "|".join(sorted(inputs.predictors[candidate])),
                            "domains": "|".join(
                                sorted(_domains(inputs.predictors[candidate], domain_map))
                            ),
                        }
                    )

                summary.update(_stability_stats(candidate_jaccards, f"top{k}_candidate_jaccard"))
                summary.update(_stability_stats(exposure_jaccards, f"top{k}_exposure_jaccard"))
                summary.update(_stability_stats(domain_jaccards, f"top{k}_domain_jaccard"))
                summary[f"top{k}_n_global_retained_mean"] = float(np.mean(retained_counts))
                summary[f"top{k}_fraction_global_retained_mean"] = float(np.mean(retained_counts) / k)

            summary_rows.append(summary)

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(frequency_rows),
    )


def _stability_stats(values: list[float], prefix: str) -> dict[str, float]:
    """Mean plus the distribution summary. The mean is reported explicitly
    because the paper-facing stability tables are read off it directly."""
    array = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if array.size == 0:
        keys = ("mean", "median", "sd", "q1", "q3", "min", "max")
        return {f"{prefix}_{key}": float("nan") for key in keys}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_q1": float(np.percentile(array, 25)),
        f"{prefix}_q3": float(np.percentile(array, 75)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def cross_selected_single(inputs: FrontierInputs, rung: str) -> pd.Series:
    """S*[c, rung]: the best single exposure chosen without country c, scored in c.

    The selection excludes the evaluation country for exactly the same reason
    the multivariate frontier does, so the comparison is like-for-like.
    """
    wide = inputs.single[rung]
    return pd.Series(
        {
            str(country): float(wide.loc[top_k(wide, 1, exclude=str(country))[0], country])
            for country in wide.columns
        },
        dtype=float,
    )


def _frontier_series(scores: pd.DataFrame, bag: str, arm: str, rung: str, k: int) -> pd.Series:
    selected = scores[
        scores["bag"].eq(bag) & scores["arm"].eq(arm) & scores["rung"].eq(rung) & scores["K"].eq(k)
    ]
    return selected.set_index("excluded_country")["frontier_median_r2"]


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def comparison_rows(
    inputs: FrontierInputs,
    scores: pd.DataFrame,
    k: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Every frontier contrast for one BAG at one frontier size."""
    bag = inputs.bag
    rows: list[dict[str, object]] = []
    seed = BOOTSTRAP_SEED + 1000 * k

    def frontier(arm: str, rung: str) -> pd.Series:
        return _frontier_series(scores, bag, arm, rung, k)

    # A. synergy frontier vs redundancy frontier, within each rung.
    for index, rung in enumerate(RUNGS):
        rows.append(
            {
                "comparison_family": "synergy_vs_redundancy",
                "bag": bag,
                "arm": "o_min_minus_o_max",
                "model_a": "synergy_frontier",
                "model_b": "redundancy_frontier",
                "rung_a": rung,
                "rung_b": rung,
                "multiplicity_family": f"{bag}|synergy_vs_redundancy|K{k}",
                **paired_country_test(
                    frontier("o_min", rung), frontier("o_max", rung), draws=draws, seed=seed + index
                ),
            }
        )

    # B. model complexity within each arm, including the required d3 vs d1.
    for arm_index, arm in enumerate(ARMS):
        for pair_index, (high, low) in enumerate(RUNG_PAIRS):
            rows.append(
                {
                    "comparison_family": "model_complexity",
                    "bag": bag,
                    "arm": arm,
                    "model_a": f"{arm}_frontier",
                    "model_b": f"{arm}_frontier",
                    "rung_a": high,
                    "rung_b": low,
                    "multiplicity_family": f"{bag}|{arm}|model_complexity|K{k}",
                    **paired_country_test(
                        frontier(arm, high),
                        frontier(arm, low),
                        draws=draws,
                        seed=seed + 100 + 10 * arm_index + pair_index,
                    ),
                }
            )

    # C. frontier vs the canonical covariate baseline.
    for arm_index, arm in enumerate(ARMS):
        for rung_index, rung in enumerate(RUNGS):
            rows.append(
                {
                    "comparison_family": "vs_covariate_baseline",
                    "bag": bag,
                    "arm": arm,
                    "model_a": f"{arm}_frontier",
                    "model_b": "covariate_baseline",
                    "rung_a": rung,
                    "rung_b": rung,
                    "multiplicity_family": f"{bag}|vs_covariate_baseline|K{k}",
                    **paired_country_test(
                        frontier(arm, rung),
                        inputs.baseline[rung],
                        draws=draws,
                        seed=seed + 300 + 10 * arm_index + rung_index,
                    ),
                }
            )

    # D. frontier vs the country-cross-selected best single exposure.
    for arm_index, arm in enumerate(ARMS):
        for rung_index, rung in enumerate(RUNGS):
            rows.append(
                {
                    "comparison_family": "vs_cross_selected_single",
                    "bag": bag,
                    "arm": arm,
                    "model_a": f"{arm}_frontier",
                    "model_b": "cross_selected_best_single",
                    "rung_a": rung,
                    "rung_b": rung,
                    "multiplicity_family": f"{bag}|vs_cross_selected_single|K{k}",
                    **paired_country_test(
                        frontier(arm, rung),
                        cross_selected_single(inputs, rung),
                        draws=draws,
                        seed=seed + 500 + 10 * arm_index + rung_index,
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame["K"] = k
    frame["analysis_type"] = "country_cross_selected_frontier"
    frame["frontier_definition"] = FRONTIER_DEFINITION
    return frame


def matched_rung_rows(
    inputs: FrontierInputs, k: int = PRIMARY_K, *, draws: int = BOOTSTRAP_DRAWS
) -> pd.DataFrame:
    """Matched-candidate sensitivity: one shared candidate set, both rungs.

    For each excluded country the two rungs' Top-K selections are unioned, and
    that identical set of candidate identities is scored under both rungs in the
    held-out country. This separates a genuine model-level effect from a change
    in which candidates enter the frontier.
    """
    rows: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        members = inputs.arms[arm]
        for pair_index, (high, low) in enumerate(MATCHED_PAIRS):
            wide_high = inputs.multivariate[high].loc[members]
            wide_low = inputs.multivariate[low].loc[members]
            countries = sorted(str(value) for value in wide_high.columns)

            high_scores: dict[str, float] = {}
            low_scores: dict[str, float] = {}
            union_sizes: list[int] = []
            for country in countries:
                union = sorted(
                    set(top_k(wide_high, k, exclude=country))
                    | set(top_k(wide_low, k, exclude=country))
                )
                union_sizes.append(len(union))
                high_scores[country] = float(np.median(wide_high.loc[union, country].to_numpy(float)))
                low_scores[country] = float(np.median(wide_low.loc[union, country].to_numpy(float)))

            rows.append(
                {
                    "comparison_family": "matched_candidate_model_complexity",
                    "bag": inputs.bag,
                    "arm": arm,
                    "model_a": f"{arm}_matched_union",
                    "model_b": f"{arm}_matched_union",
                    "rung_a": high,
                    "rung_b": low,
                    "K": k,
                    "mean_union_size": float(np.mean(union_sizes)),
                    "min_union_size": int(np.min(union_sizes)),
                    "max_union_size": int(np.max(union_sizes)),
                    "multiplicity_family": f"{inputs.bag}|{arm}|matched_candidate|K{k}",
                    **paired_country_test(
                        pd.Series(high_scores),
                        pd.Series(low_scores),
                        draws=draws,
                        seed=BOOTSTRAP_SEED + 700 + 10 * arm_index + pair_index,
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["analysis_type"] = "matched_candidate_union_frontier"
    return frame


EXPECTED_FAMILY_SIZES = {
    "synergy_vs_redundancy": len(RUNGS),
    "model_complexity": len(RUNG_PAIRS),
    "vs_covariate_baseline": len(ARMS) * len(RUNGS),
    "vs_cross_selected_single": len(ARMS) * len(RUNGS),
    "matched_candidate_model_complexity": len(MATCHED_PAIRS),
}


def validate_families(frame: pd.DataFrame) -> None:
    sizes = frame.groupby(["multiplicity_family", "comparison_family"], sort=False).size()
    for (family, kind), size in sizes.items():
        expected = EXPECTED_FAMILY_SIZES[kind]
        if size != expected:
            raise ValueError(f"Holm family {family!r} has {size} tests, expected {expected}")


def k_sensitivity_table(tests: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast, with delta and Holm P side by side across K."""
    keys = ["comparison_family", "bag", "arm", "rung_a", "rung_b"]
    reported = tests[tests["K"].isin(K_REPORTED)]
    wide = reported.pivot_table(
        index=keys, columns="K", values=["delta_mean_r2", "holm_p", "rank_biserial"], aggfunc="first"
    )
    wide.columns = [f"{metric}_K{k}" for metric, k in wide.columns]
    wide = wide.reset_index()

    deltas = [f"delta_mean_r2_K{k}" for k in K_REPORTED]
    holms = [f"holm_p_K{k}" for k in K_REPORTED]
    signs = np.sign(wide[deltas].to_numpy(float))
    wide["sign_changes_across_k"] = [len(set(row[~np.isnan(row)])) > 1 for row in signs]
    significant = wide[holms].to_numpy(float) < 0.05
    wide["significance_changes_across_k"] = [len(set(row)) > 1 for row in significant]
    wide["max_abs_delta_change"] = np.nanmax(wide[deltas].to_numpy(float), axis=1) - np.nanmin(
        wide[deltas].to_numpy(float), axis=1
    )
    return wide.sort_values(keys, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/performance_frontier/main_k10",
    )
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    registry_path = _registry_path(runtime, args.source_run_id)
    registry = _registry(registry_path)
    domain_path = ROOT / "data/metadata/exposome_feature_domains.csv"
    domain_map = _domain_map(domain_path)

    bundles = [load_bag(runtime, args.source_run_id, bag, registry) for bag in BAGS]

    scores, summaries, frequencies = [], [], []
    for inputs in bundles:
        score, summary, frequency = build_frontier(inputs, domain_map)
        scores.append(score)
        summaries.append(summary)
        frequencies.append(frequency)
    scores = pd.concat(scores, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    frequency = pd.concat(frequencies, ignore_index=True)

    all_tests = []
    for k in K_VALUES:
        for inputs in bundles:
            all_tests.append(comparison_rows(inputs, scores, k, draws=args.draws))
    tests = apply_holm(pd.concat(all_tests, ignore_index=True))
    validate_families(tests)

    matched = apply_holm(
        pd.concat([matched_rung_rows(inputs, draws=args.draws) for inputs in bundles], ignore_index=True)
    )
    validate_families(matched)

    columns = [
        "comparison_family", "bag", "arm", "model_a", "model_b", "rung_a", "rung_b",
        "K", "n_countries", "estimate_a", "estimate_b", "delta_mean_r2",
        "median_paired_delta", "ci_lo", "ci_hi", "wilcoxon_statistic", "wilcoxon_p_raw",
        "holm_p", "rank_biserial", "multiplicity_family", "analysis_type",
        "frontier_definition",
    ]
    primary = tests[tests["K"].eq(PRIMARY_K)][columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "frontier_country_scores.csv", index=False)
    summary.to_csv(args.output_dir / "frontier_stability_summary.csv", index=False)
    frequency.to_csv(args.output_dir / "frontier_selection_frequencies.csv", index=False)
    primary.to_csv(args.output_dir / "frontier_comparison_tests.csv", index=False)
    k_sensitivity_table(tests).to_csv(args.output_dir / "frontier_K_sensitivity.csv", index=False)
    matched[[c for c in columns if c in matched.columns] + ["mean_union_size", "min_union_size", "max_union_size"]].to_csv(
        args.output_dir / "frontier_matched_rung_comparisons.csv", index=False
    )
    tests[columns].to_csv(args.output_dir / "frontier_comparison_tests_all_k.csv", index=False)

    paths = {key: value for inputs in bundles for key, value in inputs.paths.items()}
    paths["candidate_registry"] = str(registry_path)
    paths["exposome_feature_domains"] = str(domain_path)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_performance_frontier",
        "source_run": str(runtime / "results/analysis_runs" / args.source_run_id),
        "model_fitting_performed": False,
        "statement": (
            "No model fitting was performed. Every value aggregates cached "
            "per-candidate held-out-country LOCO metrics."
        ),
        "nesting_caveat": NESTING_CAVEAT,
        "selection_rule": SELECTION_RULE,
        "tie_breaking": "ascending candidate_id, stable mergesort",
        "frontier_definition": FRONTIER_DEFINITION,
        "k_values": list(K_VALUES),
        "primary_k": PRIMARY_K,
        "max_order": MAX_ORDER,
        "inputs": {
            key: {"path": value, "sha256": _sha256(Path(value))} for key, value in sorted(paths.items())
        },
        "candidate_counts": {
            f"{row.bag}/{row.arm}/{row.rung}": int(row.n_candidates)
            for row in summary.itertuples()
        },
        "country_counts": summary.groupby("bag")["n_countries"].max().to_dict(),
        "bootstrap": {"draws": args.draws, "seed": BOOTSTRAP_SEED},
        "multiplicity_families": {
            family: int(size)
            for family, size in pd.concat([tests, matched]).groupby("multiplicity_family", sort=False).size().items()
        },
    }
    (args.output_dir / "frontier_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir / 'frontier_comparison_tests.csv'} ({len(primary)} K=20 tests)")


def _registry_path(runtime: Path, source_run_id: str) -> Path:
    """The candidate registry recorded in the run's own manifest."""
    manifest = (
        runtime
        / "results/analysis_runs"
        / source_run_id
        / "xgb/structural/xgb_tree_d2/k10/manifest.json"
    )
    recorded = json.loads(manifest.read_text())["candidate_pool"]
    return Path(recorded)


if __name__ == "__main__":
    main()
