#!/usr/bin/env python3
"""Is the cross-selected Top-K frontier, as a region, above the covariate baseline?

This is narrower than testing a single winner and broader than testing the
median of the whole BAG-blind pool.

ALGEBRAIC NOTE (verified, not assumed)
--------------------------------------
Within a BAG x rung x country cell the covariate baseline is a single constant
``b`` shared by every candidate. Subtracting a constant is monotone, so it
commutes with any order statistic:

    median_j(R2_j - b) = median_j(R2_j) - b

The "median Top-K delta versus baseline" is therefore *identically* the
frontier-median analysis already completed, shifted by the baseline. It is
reported here only for continuity and is explicitly labelled as not a new
result.

The genuinely new estimand is the **mean** incremental performance across the
frontier. Unlike the median it is sensitive to the left tail, so it answers
whether the high-performing region as a whole -- including its weaker members --
carries incremental association.

The statistical unit is the held-out country. The K candidates within a
frontier overlap heavily and are strongly dependent, so they are never treated
as independent observations: they are averaged within a country first, and only
the resulting country-level values enter any test.

No model fitting is performed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.compute_landscape_model_comparisons import _rank_biserial
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import _registry, _sha256

ROOT = frontier.ROOT

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
# Primary scope. OLS and d1 are deliberately excluded from the correction
# family: d1 is a nonlinear-additive control and OLS is not used for a
# baseline-association conclusion.
RUNGS = ("xgb_tree_d2", "xgb_tree_d3")
K_VALUES = (10, 20, 50)
PRIMARY_K = 20
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260922

MEDIAN_EQUIVALENCE_NOTE = (
    "median_frontier_delta equals the previously reported frontier-median R2 minus "
    "the covariate baseline, because the baseline is constant within a country and "
    "subtracting a constant commutes with the median. It is reported for continuity "
    "and is not a new result."
)
DEPENDENCE_NOTE = (
    "The K candidates within a frontier overlap heavily and are not independent. "
    "They are averaged within each held-out country, and only the resulting "
    "country-level values are used for inference. No candidate-level P values are "
    "computed and the frontier members are never treated as independent observations."
)


def _wilcoxon_against_zero(values: np.ndarray) -> tuple[float, float]:
    nonzero = values[values != 0]
    if not len(nonzero):
        return float("nan"), 1.0
    statistic, p_value = stats.wilcoxon(nonzero, alternative="two-sided")
    return float(statistic), float(p_value)


def _paired_bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    """Percentile CI for the mean of country-level values, resampling countries."""
    rng = np.random.default_rng(seed)
    n = len(values)
    counts = rng.multinomial(n, np.repeat(1 / n, n), size=draws)
    means = counts @ values / n
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_country_summary(
    inputs: frontier.FrontierInputs,
    registry: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country-level frontier statistics and the candidate-level audit."""
    orders = dict(zip(registry["candidate_id"].astype(str), registry["order"]))
    summary_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for arm in ARMS:
        members = inputs.arms[arm]
        for rung in RUNGS:
            wide = inputs.multivariate[rung].loc[members]
            baseline = inputs.baseline[rung]
            countries = sorted(str(value) for value in wide.columns)
            global_ranked = frontier._ranked(
                wide.mean(axis=1).rename("mean_r2").reset_index(), "mean_r2"
            )
            global_rank = {
                str(row.candidate_id): index + 1
                for index, row in enumerate(global_ranked.itertuples(index=False))
            }

            for k in k_values:
                global_top = frozenset(frontier.top_k(wide, k))
                selections: dict[str, list[str]] = {
                    country: frontier.top_k(wide, k, exclude=country) for country in countries
                }
                frequency: dict[str, int] = {}
                for selected in selections.values():
                    for candidate in selected:
                        frequency[candidate] = frequency.get(candidate, 0) + 1

                for country in countries:
                    selected = selections[country]
                    if len(set(selected)) != k:
                        raise ValueError(f"Frontier for {country} does not hold {k} unique ids")
                    base_value = float(baseline[country])
                    candidate_r2 = wide.loc[selected, country].to_numpy(float)
                    deltas = candidate_r2 - base_value

                    summary_rows.append(
                        {
                            "bag": inputs.bag,
                            "arm": arm,
                            "rung": rung,
                            "country": country,
                            "K": k,
                            "mean_frontier_delta": float(deltas.mean()),
                            "median_frontier_delta": float(np.median(deltas)),
                            "baseline_country_r2": base_value,
                            "positive_fraction": float((deltas > 0).mean()),
                            "n_positive": int((deltas > 0).sum()),
                            "n_negative": int((deltas < 0).sum()),
                            "all_positive": int(bool((deltas > 0).all())),
                            "min_delta": float(deltas.min()),
                            "q1_delta": float(np.percentile(deltas, 25)),
                            "q3_delta": float(np.percentile(deltas, 75)),
                            "max_delta": float(deltas.max()),
                        }
                    )

                    for candidate, value, delta in zip(selected, candidate_r2, deltas):
                        audit_rows.append(
                            {
                                "bag": inputs.bag,
                                "arm": arm,
                                "rung": rung,
                                "country": country,
                                "K": k,
                                "candidate_id": candidate,
                                "global_rank": global_rank[candidate],
                                "set_size": int(orders[candidate]),
                                "candidate_country_r2": float(value),
                                "baseline_country_r2": base_value,
                                "delta_r2": float(delta),
                                "delta_positive": bool(delta > 0),
                                "in_global_topk": candidate in global_top,
                                "selection_frequency": frequency[candidate],
                                "selection_fraction": frequency[candidate] / len(countries),
                            }
                        )

    return pd.DataFrame(summary_rows), pd.DataFrame(audit_rows)


def run_tests(summary: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Primary mean-gain test and the secondary positivity test, per cell and K."""
    rows: list[dict[str, object]] = []
    for index, ((bag, arm, rung, k), cell) in enumerate(
        summary.groupby(["bag", "arm", "rung", "K"], sort=True)
    ):
        values = cell["mean_frontier_delta"].to_numpy(float)
        statistic, p_raw = _wilcoxon_against_zero(values)
        ci_lo, ci_hi = _paired_bootstrap_mean(values, draws=draws, seed=BOOTSTRAP_SEED + index)
        rows.append(
            {
                "test": "primary_mean_frontier_gain",
                "bag": bag,
                "arm": arm,
                "rung": rung,
                "K": k,
                "n_countries": int(len(values)),
                "mean_country_frontier_delta": float(values.mean()),
                "median_country_frontier_delta": float(np.median(values)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": p_raw,
                "rank_biserial": _rank_biserial(values),
                "n_countries_positive": int((values > 0).sum()),
                "fraction_countries_positive": float((values > 0).mean()),
                "multiplicity_family": f"{bag}|primary_mean_frontier_gain|K{k}",
            }
        )

        # Secondary, explicitly descriptive-leaning: does the typical frontier
        # member beat baseline more often than chance? The unit stays the country.
        fractions = cell["positive_fraction"].to_numpy(float)
        centred = fractions - 0.5
        statistic_p, p_positivity = _wilcoxon_against_zero(centred)
        rows.append(
            {
                "test": "secondary_positivity_vs_half",
                "bag": bag,
                "arm": arm,
                "rung": rung,
                "K": k,
                "n_countries": int(len(fractions)),
                "mean_positive_fraction": float(fractions.mean()),
                "median_positive_fraction": float(np.median(fractions)),
                "q1_positive_fraction": float(np.percentile(fractions, 25)),
                "q3_positive_fraction": float(np.percentile(fractions, 75)),
                "min_positive_fraction": float(fractions.min()),
                "max_positive_fraction": float(fractions.max()),
                "n_countries_majority_positive": int((fractions > 0.5).sum()),
                "n_countries_75pct_positive": int((fractions >= 0.75).sum()),
                "n_countries_all_positive": int((fractions == 1.0).sum()),
                "wilcoxon_statistic": statistic_p,
                "wilcoxon_p_raw": p_positivity,
                "rank_biserial": _rank_biserial(centred),
                "multiplicity_family": f"{bag}|secondary_positivity|K{k}",
            }
        )

    frame = pd.DataFrame(rows)
    frame["holm_p"] = np.nan
    for _, index in frame.groupby("multiplicity_family", sort=False).groups.items():
        values = frame.loc[index, "wilcoxon_p_raw"].astype(float)
        frame.loc[index, "holm_p"] = multipletests(values, method="holm")[1]
    frame["candidate_dependence_note"] = DEPENDENCE_NOTE
    return frame


def validate_families(tests: pd.DataFrame) -> None:
    """Every Holm family must contain exactly the four primary cells."""
    primary = tests[tests["test"].eq("primary_mean_frontier_gain")]
    sizes = primary.groupby("multiplicity_family").size()
    for family, size in sizes.items():
        if size != 4:
            raise ValueError(f"Primary Holm family {family!r} has {size} tests, expected 4")


def k_sensitivity(tests: pd.DataFrame) -> pd.DataFrame:
    """Mean gain, CI, Holm P and positivity side by side across K."""
    primary = tests[tests["test"].eq("primary_mean_frontier_gain")]
    positivity = tests[tests["test"].eq("secondary_positivity_vs_half")]
    keys = ["bag", "arm", "rung"]
    frame = primary.pivot_table(
        index=keys, columns="K", values=["mean_country_frontier_delta", "ci_lo", "ci_hi", "holm_p"],
        aggfunc="first",
    )
    frame.columns = [f"{metric}_K{k}" for metric, k in frame.columns]
    extra = positivity.pivot_table(
        index=keys,
        columns="K",
        values=["mean_positive_fraction", "median_positive_fraction", "n_countries_majority_positive"],
        aggfunc="first",
    )
    extra.columns = [f"{metric}_K{k}" for metric, k in extra.columns]
    combined = frame.join(extra).reset_index()

    deltas = [f"mean_country_frontier_delta_K{k}" for k in K_VALUES]
    holms = [f"holm_p_K{k}" for k in K_VALUES]
    signs = np.sign(combined[deltas].to_numpy(float))
    combined["sign_changes_across_k"] = [len(set(row[~np.isnan(row)])) > 1 for row in signs]
    significant = combined[holms].to_numpy(float) < 0.05
    combined["significance_changes_across_k"] = [len(set(row)) > 1 for row in significant]
    return combined.sort_values(keys, ignore_index=True)


def extended_estimand_comparison(
    tests: pd.DataFrame, summary: pd.DataFrame, cross_selected: Path, winner_path: Path
) -> pd.DataFrame:
    """A/B/C/D: winner, cross-selected Top-1, frontier mean, frontier median."""
    rows: list[dict[str, object]] = []

    # A. the delivered global winner. The production stage leaves the
    # vs_baseline family unadjusted, so the raw P is carried.
    winners = pd.read_csv(winner_path)
    winners = winners[winners["comparison_type"].eq("vs_baseline") & winners["model_a"].isin(RUNGS)]
    for row in winners.itertuples(index=False):
        rows.append(
            {
                "estimand": "A_global_winner",
                "bag": row.bag,
                "arm": row.objective,
                "rung": row.model_a,
                "delta_r2": row.delta_r2,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_raw": row.p_raw,
                "p_adjusted": np.nan,
                "fraction_countries_positive": np.nan,
                "note": "production leaves vs_baseline unadjusted; raw P shown",
            }
        )

    # B. the country-cross-selected Top-1 candidate.
    if cross_selected.is_file():
        selected = pd.read_csv(cross_selected)
        selected = selected[selected["rung"].isin(RUNGS)]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "estimand": "B_cross_selected_top1",
                    "bag": row.bag,
                    "arm": row.arm,
                    "rung": row.rung,
                    "delta_r2": row.delta_r2,
                    "ci_lo": row.bootstrap_ci_lo,
                    "ci_hi": row.bootstrap_ci_hi,
                    "p_raw": row.bootstrap_p_raw,
                    "p_adjusted": row.bootstrap_p_holm,
                    "fraction_countries_positive": row.fraction_countries_delta_positive,
                    "note": "Holm within its own primary family of four",
                }
            )

    # C/D. the Top-K frontier mean and median.
    primary = tests[tests["test"].eq("primary_mean_frontier_gain") & tests["K"].eq(PRIMARY_K)]
    for row in primary.itertuples(index=False):
        rows.append(
            {
                "estimand": "C_cross_selected_top20_mean",
                "bag": row.bag,
                "arm": row.arm,
                "rung": row.rung,
                "delta_r2": row.mean_country_frontier_delta,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_raw": row.wilcoxon_p_raw,
                "p_adjusted": row.holm_p,
                "fraction_countries_positive": row.fraction_countries_positive,
                "note": "primary estimand of this analysis",
            }
        )

    median_cell = summary[summary["K"].eq(PRIMARY_K)]
    for (bag, arm, rung), cell in median_cell.groupby(["bag", "arm", "rung"], sort=True):
        values = cell["median_frontier_delta"].to_numpy(float)
        statistic, p_raw = _wilcoxon_against_zero(values)
        rows.append(
            {
                "estimand": "D_cross_selected_top20_median",
                "bag": bag,
                "arm": arm,
                "rung": rung,
                "delta_r2": float(values.mean()),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "p_raw": p_raw,
                "p_adjusted": np.nan,
                "fraction_countries_positive": float((values > 0).mean()),
                "note": MEDIAN_EQUIVALENCE_NOTE,
            }
        )

    frame = pd.DataFrame(rows)
    frame["estimands_answer_different_questions"] = True
    return frame.sort_values(["bag", "arm", "rung", "estimand"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/frontier_vs_baseline/main_k10",
    )
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    registry_path = frontier._registry_path(runtime, args.source_run_id)
    registry = _registry(registry_path)

    summaries, audits, paths = [], [], {}
    for bag in BAGS:
        inputs = frontier.load_bag(runtime, args.source_run_id, bag, registry)
        paths.update(inputs.paths)
        summary, audit = build_country_summary(inputs, registry)
        summaries.append(summary)
        audits.append(audit)
    summary = pd.concat(summaries, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)

    tests = run_tests(summary, draws=args.draws)
    validate_families(tests)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "frontier_vs_baseline_country_summary.csv", index=False)
    tests.to_csv(args.output_dir / "frontier_vs_baseline_tests.csv", index=False)
    k_sensitivity(tests).to_csv(args.output_dir / "frontier_vs_baseline_K_sensitivity.csv", index=False)
    audit.to_csv(args.output_dir / "frontier_vs_baseline_candidate_audit.csv", index=False)
    extended_estimand_comparison(
        tests,
        summary,
        ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10/cross_selected_baseline_summary.csv",
        runtime / "results/analysis_runs" / args.source_run_id
        / "main_statistics/model_comparison/complexity_and_arm_comparisons.csv",
    ).to_csv(args.output_dir / "baseline_estimand_comparison_extended.csv", index=False)

    paths["candidate_registry"] = str(registry_path)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_frontier_vs_baseline",
        "model_fitting_performed": False,
        "statement": (
            "No model fitting and no candidate discovery were performed. Every value "
            "aggregates cached per-candidate held-out-country R2 and the matched "
            "canonical covariate baseline."
        ),
        "candidate_dependence": DEPENDENCE_NOTE,
        "median_equivalence": MEDIAN_EQUIVALENCE_NOTE,
        "selection_rule": frontier.SELECTION_RULE,
        "nesting_caveat": frontier.NESTING_CAVEAT,
        "k_values": list(K_VALUES),
        "primary_k": PRIMARY_K,
        "primary_estimand": "mean over Top-K frontier of (candidate country R2 - baseline country R2)",
        "baseline_source": "canonical covariate-only scope, unique per BAG x rung x country",
        "statistical_tests": {
            "primary": "two-sided Wilcoxon signed-rank of country-level mean frontier delta against zero",
            "interval": "95% percentile CI of the mean from a 10,000-draw country bootstrap",
            "secondary": "two-sided Wilcoxon signed-rank of (positive_fraction - 0.5) across countries",
            "effect_size": "rank-biserial correlation",
        },
        "holm_families": tests.groupby("multiplicity_family").size().to_dict(),
        "bootstrap": {"draws": args.draws, "seed": BOOTSTRAP_SEED},
        "country_counts": summary.groupby("bag")["country"].nunique().to_dict(),
        "inputs": {
            key: {"path": value, "sha256": _sha256(Path(value))} for key, value in sorted(paths.items())
        },
    }
    (args.output_dir / "frontier_vs_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir / 'frontier_vs_baseline_tests.csv'} ({len(tests)} tests)")


if __name__ == "__main__":
    main()
