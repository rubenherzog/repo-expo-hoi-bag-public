#!/usr/bin/env python3
"""Country-cross-selected exposome candidate versus the covariate baseline.

This answers a narrower question than either of the preceding analyses:

    Does a BAG-relevant exposome model, selected without using the evaluation
    country, add out-of-sample predictive information beyond the covariates?

That is an *incremental-association* question. The whole-landscape median was
too broad, because most BAG-blind candidates are not expected to carry BAG
information. The Top-20 frontier median is the right object for comparing
architectures, but it is deliberately conservative against a baseline because
it averages a high-performing region rather than evaluating a selected model.

For every BAG x arm x rung and every held-out country ``c`` the candidate is
chosen by its mean R2 over the countries other than ``c`` -- the production
rule, with the production tie-break -- and that candidate's subject-level OOF
predictions for the participants of ``c`` are taken. Concatenating over
countries yields one cross-selected prediction vector per cell.

ESTIMAND
--------
The canonical paper-facing estimand is the unweighted mean of per-country R2
(``unweighted_mean_country_r2``), which every delivered supplementary table
uses. That is the primary effect here. The pooled-subject OOF R2 is reported
alongside it as a secondary descriptive statistic, and the two are never mixed.

INTERPRETATION LIMIT
--------------------
Country-cross-selected, not nested cross-validation, and not one fixed exposome
model: the selected candidate is allowed to differ by evaluation country. The
candidate pool itself was not re-derived without the evaluation country.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import _registry, _sha256

ROOT = frontier.ROOT

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
PRIMARY_RUNGS = ("xgb_tree_d2", "xgb_tree_d3")
CONTROL_RUNGS = ("xgb_tree_d1",)
ALL_RUNGS = CONTROL_RUNGS + PRIMARY_RUNGS
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260922

SELECTION_RULE = (
    "For each held-out country c the candidate maximising the unweighted mean "
    "R2 across the eligible countries other than c (n_test > 0, set size <= 30), "
    "ties broken on ascending candidate_id with a stable sort. Country c never "
    "contributes to its own candidate ranking."
)
R2_DEFINITION = (
    "Primary: unweighted mean of per-country R2 (unweighted_mean_country_r2), the "
    "canonical paper-facing estimand. Secondary: pooled-subject OOF R2 computed "
    "on the concatenated cross-selected predictions."
)
NESTING_CAVEAT = (
    "Country-cross-selected, not nested cross-validation, and not one fixed "
    "exposome model: the selected candidate may differ by evaluation country, "
    "and the candidate pool was not re-derived without that country."
)


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    """Pooled R2, the same form the production comparison stage uses."""
    return float(1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum())


# ---------------------------------------------------------------------------
# assembling the cross-selected prediction vectors
# ---------------------------------------------------------------------------
def selection_plan(runtime: Path, source_run_id: str, registry: pd.DataFrame) -> pd.DataFrame:
    """The cross-selected candidate for every BAG x arm x rung x country."""
    predictors = dict(
        zip(registry["candidate_id"].astype(str), registry["predictors_identity"].astype(str))
    )
    orders = dict(zip(registry["candidate_id"].astype(str), registry["order"]))
    rows: list[dict[str, object]] = []
    for bag in BAGS:
        inputs = frontier.load_bag(runtime, source_run_id, bag, registry)
        for arm in ARMS:
            for rung in ALL_RUNGS:
                wide = inputs.multivariate[rung].loc[inputs.arms[arm]]
                global_choice = frontier.top_k(wide, 1)[0]
                for country in wide.columns:
                    country = str(country)
                    selected = frontier.top_k(wide, 1, exclude=country)[0]
                    rows.append(
                        {
                            "bag": bag,
                            "arm": arm,
                            "rung": rung,
                            "country": country,
                            "candidate_id": selected,
                            "predictors_identity": predictors[selected],
                            "order": int(orders[selected]),
                            "same_as_global_winner": selected == global_choice,
                            "global_winner_candidate_id": global_choice,
                        }
                    )
    return pd.DataFrame(rows)


def _cached_predictions(oof_root: Path) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Delivered subject-level OOF predictions, keyed by bag/rung/candidate."""
    store: dict[tuple[str, str, str], pd.DataFrame] = {}
    for path in sorted(oof_root.glob("*/*/*.parquet")):
        bag = path.parent.name
        rung = path.stem.removeprefix("oof_")
        frame = pd.read_parquet(
            path, columns=["row_id", "country", "y_true", "y_pred_full", "y_pred_base", "candidate_id"]
        )
        for candidate, group in frame.groupby("candidate_id"):
            store.setdefault((bag, rung, str(candidate)), group)
    return store


def _fitted_predictions(cache_dir: Path) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Predictions produced by run_cross_selected_oof for the uncached folds."""
    store: dict[tuple[str, str, str], pd.DataFrame] = {}
    for path in sorted(cache_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        key = (str(frame["bag"].iloc[0]), str(frame["rung_id"].iloc[0]), str(frame["candidate_id"].iloc[0]))
        store[key] = frame
    return store


def assemble_predictions(
    plan: pd.DataFrame, cached: dict, fitted: dict
) -> tuple[pd.DataFrame, list[str]]:
    """One prediction row per participant per BAG x arm x rung.

    Each participant's prediction comes from the candidate selected for that
    participant's own country, fitted on a training set excluding it.
    """
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for row in plan.itertuples(index=False):
        key = (row.bag, row.rung, row.candidate_id)
        source = cached.get(key)
        origin = "delivered_oof"
        if source is None:
            source = fitted.get(key)
            origin = "refitted_fold"
        if source is None:
            missing.append(f"{row.bag}/{row.rung}/{row.candidate_id}")
            continue
        block = source[source["country"].astype(str).eq(row.country)]
        if block.empty:
            missing.append(f"{row.bag}/{row.rung}/{row.candidate_id}@{row.country}")
            continue
        parts.append(
            block[["row_id", "country", "y_true", "y_pred_full", "y_pred_base"]].assign(
                bag=row.bag,
                arm=row.arm,
                rung=row.rung,
                candidate_id=row.candidate_id,
                prediction_source=origin,
            )
        )
    if missing:
        return pd.DataFrame(), sorted(set(missing))
    return pd.concat(parts, ignore_index=True), []


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def country_r2(frame: pd.DataFrame, prediction: str) -> pd.Series:
    return pd.Series(
        {
            str(country): _r2(
                group["y_true"].to_numpy(float), group[prediction].to_numpy(float)
            )
            for country, group in frame.groupby("country", sort=True)
        },
        dtype=float,
    )


def country_cluster_bootstrap(
    frame: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, seed: int
) -> dict[str, float]:
    """Paired country-cluster bootstrap of the cross-selected minus baseline gap.

    Countries are resampled with replacement and every participant of a sampled
    country is retained, so a country drawn twice contributes its participants
    twice. The cross-selected candidate identity for a country is fixed before
    resampling, so the bootstrap reflects sampling variation in countries rather
    than re-running the selection.

    Both estimands are recomputed on each draw: the country-balanced mean of
    per-country R2 (primary) and the pooled R2 (secondary).
    """
    groups = [group for _, group in frame.groupby("country", sort=True)]
    truth = [group["y_true"].to_numpy(float) for group in groups]
    full = [group["y_pred_full"].to_numpy(float) for group in groups]
    base = [group["y_pred_base"].to_numpy(float) for group in groups]
    per_country_full = np.array([_r2(t, f) for t, f in zip(truth, full)])
    per_country_base = np.array([_r2(t, b) for t, b in zip(truth, base)])

    observed_balanced = float(per_country_full.mean() - per_country_base.mean())
    observed_pooled = float(
        _r2(np.concatenate(truth), np.concatenate(full))
        - _r2(np.concatenate(truth), np.concatenate(base))
    )

    rng = np.random.default_rng(seed)
    n = len(groups)
    balanced = np.empty(draws)
    pooled = np.empty(draws)
    for draw in range(draws):
        picks = rng.integers(0, n, size=n)
        balanced[draw] = per_country_full[picks].mean() - per_country_base[picks].mean()
        y = np.concatenate([truth[i] for i in picks])
        pooled[draw] = _r2(y, np.concatenate([full[i] for i in picks])) - _r2(
            y, np.concatenate([base[i] for i in picks])
        )

    def two_sided(values: np.ndarray) -> float:
        return min(
            1.0,
            max(2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))), 1 / draws),
        )

    return {
        "country_balanced_delta_mean": observed_balanced,
        "bootstrap_ci_lo": float(np.percentile(balanced, 2.5)),
        "bootstrap_ci_hi": float(np.percentile(balanced, 97.5)),
        "bootstrap_p_raw": two_sided(balanced),
        "pooled_delta_r2": observed_pooled,
        "pooled_ci_lo": float(np.percentile(pooled, 2.5)),
        "pooled_ci_hi": float(np.percentile(pooled, 97.5)),
        "pooled_p_raw": two_sided(pooled),
    }


def summarise(predictions: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell summary and the per-country diagnostic table."""
    summary_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []

    for index, ((bag, arm, rung), cell) in enumerate(
        predictions.groupby(["bag", "arm", "rung"], sort=True)
    ):
        full = country_r2(cell, "y_pred_full")
        base = country_r2(cell, "y_pred_base")
        delta = (full - base).dropna()
        cell_plan = plan[plan["bag"].eq(bag) & plan["arm"].eq(arm) & plan["rung"].eq(rung)]
        chosen = dict(zip(cell_plan["country"].astype(str), cell_plan["candidate_id"].astype(str)))
        orders = dict(zip(cell_plan["country"].astype(str), cell_plan["order"]))

        counts = cell.groupby("country").size()
        for country in sorted(delta.index):
            country_rows.append(
                {
                    "bag": bag,
                    "arm": arm,
                    "rung": rung,
                    "country": country,
                    "selected_candidate_id": chosen[country],
                    "selected_candidate_order": orders[country],
                    "selected_model_country_r2": float(full[country]),
                    "baseline_country_r2": float(base[country]),
                    "delta_r2_country": float(delta[country]),
                    "n_test": int(counts[country]),
                }
            )

        bootstrap = country_cluster_bootstrap(cell, seed=BOOTSTRAP_SEED + index)
        selected_balanced = float(full.mean())
        baseline_balanced = float(base.mean())
        # Cohen's f2 on the country-balanced estimand, the convention the
        # delivered level_winner_cohen_f2 table uses.
        f2 = (
            (selected_balanced - baseline_balanced) / (1 - selected_balanced)
            if selected_balanced < 1
            else float("nan")
        )
        summary_rows.append(
            {
                "bag": bag,
                "arm": arm,
                "rung": rung,
                "n_countries": int(len(delta)),
                "n_participants": int(len(cell)),
                "selected_model_country_balanced_r2": selected_balanced,
                "baseline_country_balanced_r2": baseline_balanced,
                "delta_r2": selected_balanced - baseline_balanced,
                "selected_model_pooled_oof_r2": _r2(
                    cell["y_true"].to_numpy(float), cell["y_pred_full"].to_numpy(float)
                ),
                "baseline_pooled_oof_r2": _r2(
                    cell["y_true"].to_numpy(float), cell["y_pred_base"].to_numpy(float)
                ),
                "cohen_f2": f2,
                "country_delta_median": float(delta.median()),
                "country_delta_q1": float(np.percentile(delta, 25)),
                "country_delta_q3": float(np.percentile(delta, 75)),
                "country_delta_min": float(delta.min()),
                "country_delta_max": float(delta.max()),
                "fraction_countries_delta_positive": float((delta > 0).mean()),
                "n_countries_delta_positive": int((delta > 0).sum()),
                "same_as_global_winner_fraction": float(cell_plan["same_as_global_winner"].mean()),
                "n_unique_cross_selected_candidates": int(cell_plan["candidate_id"].nunique()),
                **bootstrap,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(country_rows)


def apply_primary_holm(summary: pd.DataFrame) -> pd.DataFrame:
    """Holm within each BAG across the four primary d2/d3 comparisons.

    d1 is a nonlinear-additive control and is corrected in its own descriptive
    family so it never inflates the primary family.
    """
    result = summary.copy()
    result["multiplicity_family"] = np.where(
        result["rung"].isin(PRIMARY_RUNGS),
        "primary: " + result["bag"] + " (4 arm x level tests)",
        "descriptive control: " + result["bag"] + " (d1)",
    )
    result["bootstrap_p_holm"] = np.nan
    for family, index in result.groupby("multiplicity_family", sort=False).groups.items():
        values = result.loc[index, "bootstrap_p_raw"].astype(float)
        result.loc[index, "bootstrap_p_holm"] = multipletests(values, method="holm")[1]
    return result


def validate_primary_families(summary: pd.DataFrame) -> None:
    primary = summary[summary["multiplicity_family"].str.startswith("primary")]
    sizes = primary.groupby("multiplicity_family").size()
    for family, size in sizes.items():
        if size != 4:
            raise ValueError(f"Primary Holm family {family!r} has {size} tests, expected 4")


def estimand_comparison(
    summary: pd.DataFrame, runtime: Path, source_run_id: str, frontier_tests: Path
) -> pd.DataFrame:
    """Side-by-side of the three non-interchangeable baseline estimands."""
    rows: list[dict[str, object]] = []

    for row in summary.itertuples(index=False):
        rows.append(
            {
                "estimand": "country_cross_selected",
                "bag": row.bag,
                "arm": row.arm,
                "rung": row.rung,
                "selected_model_r2": row.selected_model_country_balanced_r2,
                "baseline_r2": row.baseline_country_balanced_r2,
                "delta_r2": row.delta_r2,
                "ci_lo": row.bootstrap_ci_lo,
                "ci_hi": row.bootstrap_ci_hi,
                "p_raw": row.bootstrap_p_raw,
                "p_adjusted": row.bootstrap_p_holm,
                "p_note": "Holm within the primary family of this analysis",
            }
        )

    # A. the delivered winner-based estimate.
    winner_path = (
        runtime / "results/analysis_runs" / source_run_id
        / "main_statistics/model_comparison/complexity_and_arm_comparisons.csv"
    )
    winners = pd.read_csv(winner_path)
    winners = winners[winners["comparison_type"].eq("vs_baseline") & winners["model_a"].isin(ALL_RUNGS)]
    # The production stage leaves every Holm column empty for the vs_baseline
    # family; ST10 applies its own correction over the four deployed d3 rows.
    # Rather than invent an adjustment, carry the raw P and say so.
    for row in winners.itertuples(index=False):
        rows.append(
            {
                "estimand": "global_winner",
                "bag": row.bag,
                "arm": row.objective,
                "rung": row.model_a,
                "selected_model_r2": row.r2_a,
                "baseline_r2": row.r2_b,
                "delta_r2": row.delta_r2,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_raw": row.p_raw,
                "p_adjusted": np.nan,
                "p_note": "production leaves vs_baseline unadjusted; raw P shown",
            }
        )

    # C. the Top-20 cross-selected frontier median.
    if frontier_tests.is_file():
        frontier_frame = pd.read_csv(frontier_tests)
        frontier_frame = frontier_frame[
            frontier_frame["comparison_family"].eq("vs_covariate_baseline")
            & frontier_frame["rung_a"].isin(ALL_RUNGS)
        ]
        for row in frontier_frame.itertuples(index=False):
            rows.append(
                {
                    "estimand": "top20_frontier_median",
                    "bag": row.bag,
                    "arm": row.arm,
                    "rung": row.rung_a,
                    "selected_model_r2": row.estimate_a,
                    "baseline_r2": row.estimate_b,
                    "delta_r2": row.delta_mean_r2,
                    "ci_lo": row.ci_lo,
                    "ci_hi": row.ci_hi,
                    "p_raw": row.wilcoxon_p_raw,
                    "p_adjusted": row.holm_p,
                    "p_note": "Holm within the frontier analysis family",
                }
            )

    frame = pd.DataFrame(rows)
    frame["estimands_are_not_interchangeable"] = True
    return frame.sort_values(["bag", "arm", "rung", "estimand"], ignore_index=True)


def attach_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Reuse the existing stability outputs rather than recomputing them."""
    result = summary.copy()
    frontier_summary = ROOT / "outputs/sensitivity/performance_frontier/main_k10/frontier_stability_summary.csv"
    if frontier_summary.is_file():
        stability = pd.read_csv(frontier_summary)[
            ["bag", "arm", "rung", "top20_candidate_jaccard_mean", "top20_candidate_jaccard_median",
             "top20_exposure_jaccard_mean"]
        ]
        result = result.merge(stability, on=["bag", "arm", "rung"], how="left")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10",
    )
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    registry_path = frontier._registry_path(runtime, args.source_run_id)
    registry = _registry(registry_path)
    plan = selection_plan(runtime, args.source_run_id, registry)

    oof_root = (
        runtime / "results/analysis_runs" / args.source_run_id
        / "main_statistics/model_comparison/oof"
    )
    cache_dir = args.output_dir / "oof_cache"
    predictions, missing = assemble_predictions(
        plan, _cached_predictions(oof_root), _fitted_predictions(cache_dir)
    )
    if missing:
        raise SystemExit(
            "Missing OOF predictions for "
            f"{len(missing)} cross-selected folds; run run_cross_selected_oof first:\n  "
            + "\n  ".join(missing[:20])
        )

    summary, by_country = summarise(predictions, plan)
    summary = attach_stability(apply_primary_holm(summary))
    validate_primary_families(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "cross_selected_baseline_summary.csv", index=False)
    by_country.to_csv(args.output_dir / "cross_selected_baseline_by_country.csv", index=False)
    predictions.to_parquet(args.output_dir / "cross_selected_baseline_predictions.parquet", index=False)
    estimand_comparison(
        summary,
        runtime,
        args.source_run_id,
        ROOT / "outputs/sensitivity/performance_frontier/main_k10/frontier_comparison_tests.csv",
    ).to_csv(args.output_dir / "baseline_estimand_comparison.csv", index=False)

    inputs = {"candidate_registry": str(registry_path)}
    for path in sorted(oof_root.glob("*/*/*.parquet")):
        inputs[f"delivered_oof/{path.parent.parent.name}/{path.parent.name}/{path.name}"] = str(path)
    for path in sorted(cache_dir.glob("*.parquet")):
        inputs[f"refitted_oof/{path.name}"] = str(path)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_cross_selected_baseline",
        "candidate_discovery_performed": False,
        "model_fitting_performed": bool(list(cache_dir.glob("*.parquet"))),
        "fitting_statement": (
            "No candidate discovery was performed and no delivered artifact was "
            "modified. Subject-level OOF predictions exist only for globally "
            "selected winners, so the folds whose cross-selected candidate differs "
            "were refitted with the production fold builder, frozen HPO parameters, "
            "country policy and per-fold seeds. Refitting that path reproduces the "
            "delivered predictions bit-for-bit on a cached fold."
        ),
        "nesting_caveat": NESTING_CAVEAT,
        "selection_rule": SELECTION_RULE,
        "r2_definition": R2_DEFINITION,
        "primary_estimand": "unweighted_mean_country_r2",
        "secondary_estimand": "pooled_subject_oof_r2",
        "bootstrap": {
            "draws": args.draws,
            "seed": BOOTSTRAP_SEED,
            "unit": "country cluster, participants retained whole, paired",
        },
        "multiplicity_families": summary.groupby("multiplicity_family").size().to_dict(),
        "prediction_sources": predictions["prediction_source"].value_counts().to_dict(),
        "n_countries": summary.groupby("bag")["n_countries"].max().to_dict(),
        "n_participants": summary.groupby("bag")["n_participants"].max().to_dict(),
        "inputs": {
            key: {"path": value, "sha256": _sha256(Path(value))} for key, value in sorted(inputs.items())
        },
    }
    (args.output_dir / "cross_selected_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir / 'cross_selected_baseline_summary.csv'} ({len(summary)} cells)")


if __name__ == "__main__":
    main()
