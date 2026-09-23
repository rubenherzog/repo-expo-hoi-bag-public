#!/usr/bin/env python3
"""Which sensitivity analyses can carry the landscape tests without refitting.

A landscape test needs one thing the winner-based tests never needed: the
*complete* candidate pool scored per held-out fold.  Many sensitivity analyses
only ever evaluated selected winners, or report a pooled R2 with no fold
column, so the test is simply not available for them from cached outputs.
This stage classifies every sensitivity, and runs the landscape comparison
wherever the cached data already make it possible.

It never fits a model.  A sensitivity that would require refitting hundreds of
candidates is recorded as unavailable rather than manufactured.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.stages import compute_landscape_model_comparisons as landscape

ROOT = landscape.ROOT


@dataclass(frozen=True)
class Scope:
    """One sensitivity's classification row."""

    sensitivity: str
    source_path: str
    full_candidate_landscape_available: bool
    per_country_or_per_fold_metrics_available: bool
    synergy_vs_redundancy_applicable: bool
    rung_comparison_applicable: bool
    single_landscape_comparison_applicable: bool
    baseline_comparison_applicable: bool
    requires_refit: bool
    reason: str
    native_order_range: str
    native_fold_unit: str
    runnable: bool = False
    runner: str = ""
    options: dict = field(default_factory=dict)


def _exists(path: Path) -> bool:
    return path.is_file() or path.is_dir()


def build_scope(repro_data_root: Path, main_run_id: str, release_run_id: str) -> list[Scope]:
    """Classify every sensitivity represented in the configured namespaces."""
    runtime = repro_data_root.resolve()
    main_run = runtime / "results/analysis_runs" / main_run_id
    delivery = ROOT / "outputs/main" / release_run_id / "sensitivity"
    dedup = ROOT / "outputs/sensitivity/dedup"
    hc_only = runtime / "sensitivity/hc_only"

    rows: list[Scope] = [
        Scope(
            sensitivity="main_k10_reference",
            source_path=str(main_run),
            full_candidate_landscape_available=True,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=True,
            rung_comparison_applicable=True,
            single_landscape_comparison_applicable=True,
            baseline_comparison_applicable=True,
            requires_refit=False,
            reason=(
                "Reference analysis. Complete BAG-blind pool of 20 candidates per "
                "objective and set size, scored for every held-out country at all "
                "four model levels, with single-exposure and covariate-baseline scopes."
            ),
            native_order_range="3-30",
            native_fold_unit="held-out country",
            runnable=True,
            runner="main",
        ),
        Scope(
            sensitivity="order_cap",
            source_path=str(main_run),
            full_candidate_landscape_available=True,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=True,
            rung_comparison_applicable=True,
            single_landscape_comparison_applicable=True,
            baseline_comparison_applicable=True,
            requires_refit=False,
            reason=(
                "The set-size cap sensitivity re-caps the same cached candidate pool, "
                "so the landscape is rebuilt from the set sizes at or below each cap "
                "without any refitting. Reported here at the manuscript's cap of 21."
            ),
            native_order_range="3-21 (cap applied to the 3-30 pool)",
            native_fold_unit="held-out country",
            runnable=True,
            runner="order_cap",
            options={"order_max": 21},
        ),
        Scope(
            sensitivity="hc_only",
            source_path=str(hc_only),
            full_candidate_landscape_available=_exists(hc_only / "structural/hc_only_country_all.csv"),
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=True,
            rung_comparison_applicable=True,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=True,
            requires_refit=False,
            reason=(
                "Healthy-control-only re-evaluation retained the complete greedy pool "
                "(20 candidates per objective and set size) per held-out country at all "
                "four model levels, with an embedded covariate baseline. It carries no "
                "single-exposure scope, so that contrast is not available."
            ),
            native_order_range="3-30 (pool extends to 50; capped to the paper range)",
            native_fold_unit="held-out country",
            runnable=_exists(hc_only / "structural/hc_only_country_all.csv"),
            runner="hc_only",
        ),
        Scope(
            sensitivity="domain_imbalance",
            source_path=str(delivery / "domain_imbalance"),
            full_candidate_landscape_available=True,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=True,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=False,
            reason=(
                "Domain-level candidates are enumerated over the ~10 exposome domains "
                "under a single 'o_domain' objective, so there are no synergy and "
                "redundancy arms to contrast. The pool is complete per held-out country "
                "for d1-d3, so rung comparisons use its native order range. Its baseline "
                "is stored per model level only, not per country, so the paired baseline "
                "contrast is unavailable."
            ),
            native_order_range="3-10 (domain space admits no higher order)",
            native_fold_unit="held-out country",
            runnable=True,
            runner="domain_imbalance",
            options={"order_min": 3, "order_max": 10},
        ),
        Scope(
            sensitivity="whole_exposome_pca",
            source_path=str(delivery / "whole_exposome_pca"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "A nested incremental principal-component ladder with exactly one "
                "candidate per order and no O-information arms. There is no candidate "
                "landscape to summarise: a median over one candidate is that candidate."
            ),
            native_order_range="1-10 principal components",
            native_fold_unit="held-out country",
        ),
        Scope(
            sensitivity="country_region_generalization",
            source_path=str(delivery / "country_region"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "Region-level generalization stores per-fold medians over the top-20 "
                "selected candidates only, not per-candidate region metrics, and the "
                "top-20 pool is chosen on performance. Reconstructing a region-level "
                "landscape would require re-evaluating the full pool under leave-one-"
                "region-out."
            ),
            native_order_range="not retained per candidate",
            native_fold_unit="held-out region (9) / held-out country (22)",
        ),
        Scope(
            sensitivity="normative_transfer",
            source_path=str(delivery / "normative_transfer"),
            full_candidate_landscape_available=True,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "The complete pool of 20 candidates per objective and set size is "
                "retained for every transfer condition and model level, but R2 is pooled "
                "across folds with no country or fold column. The statistical unit of the "
                "landscape test does not exist in the cached output."
            ),
            native_order_range="3-30",
            native_fold_unit="none retained (pooled across folds)",
        ),
        Scope(
            sensitivity="residualized_bag",
            source_path=str(delivery / "residualized_bag"),
            full_candidate_landscape_available=True,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "The residualised-target re-evaluation covers the full candidate pool at "
                "every model level but stores a single global R2 per candidate. No "
                "per-country metrics were retained, so no paired-country test is possible "
                "without re-evaluating the pool."
            ),
            native_order_range="3-30",
            native_fold_unit="none retained (global R2 per candidate)",
        ),
        Scope(
            sensitivity="education_scanner_baseline",
            source_path=str(delivery / "education_scanner_baseline"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "Only the selected best synergy and redundancy candidates were "
                "re-evaluated under the extended covariate set, and only as a global "
                "out-of-fold R2. Both the candidate landscape and the fold unit are "
                "absent."
            ),
            native_order_range="selected winners only",
            native_fold_unit="none retained (global out-of-fold R2)",
        ),
        Scope(
            sensitivity="diagnosis_balance",
            source_path=str(delivery / "diagnosis_balance"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "Diagnosis-balanced re-weighting was applied to the deployed winner "
                "models at one model level. No candidate pool was re-evaluated, so there "
                "is no landscape to summarise."
            ),
            native_order_range="selected winners only",
            native_fold_unit="held-out country",
        ),
        Scope(
            sensitivity="country_block_null",
            source_path=str(delivery / "country_block_null"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "A permutation null for the deployed winner models. By design it "
                "evaluates one model per arm against permuted exposures, not a candidate "
                "landscape."
            ),
            native_order_range="selected winners only",
            native_fold_unit="permuted country blocks",
        ),
        Scope(
            sensitivity="feature_ablation",
            source_path=str(dedup / "feature_ablation"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "Leave-one-exposure-out ablations are defined relative to the top-20 "
                "performance-selected parents and stored as global R2 losses. The parent "
                "set is performance-selected, which is exactly what the landscape "
                "statistic avoids."
            ),
            native_order_range="derived from top-20 parents",
            native_fold_unit="none retained (global R2 loss)",
        ),
        Scope(
            sensitivity="residual_confounds",
            source_path=str(dedup / "residual_confounds"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=False,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "Mixed-effects models of the deployed winners' residuals. The analysis "
                "target is a residual variance decomposition, not a candidate R2 "
                "landscape."
            ),
            native_order_range="selected winners only",
            native_fold_unit="participant within country",
        ),
        Scope(
            sensitivity="negative_o_arm_comparison",
            source_path=str(ROOT / "outputs/sensitivity/dedup_neg_o"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=True,
            reason=(
                "The alternative negative-O candidate pool is summarised as per-country "
                "R2 for the selected best synergy and redundancy models only. Its "
                "per-candidate country metrics are not part of the delivered namespace, "
                "so a landscape would require re-evaluation."
            ),
            native_order_range="selected winners only",
            native_fold_unit="held-out country",
        ),
        Scope(
            sensitivity="selection_stability",
            source_path=str(ROOT / "outputs/sensitivity/selection_stability/main_k10"),
            full_candidate_landscape_available=False,
            per_country_or_per_fold_metrics_available=True,
            synergy_vs_redundancy_applicable=False,
            rung_comparison_applicable=False,
            single_landscape_comparison_applicable=False,
            baseline_comparison_applicable=False,
            requires_refit=False,
            reason=(
                "A diagnostic of winner instability rather than a re-evaluation. It "
                "motivates the landscape framework but carries no separate candidate "
                "pool of its own; its underlying metrics are the main-k10 pool already "
                "analysed above."
            ),
            native_order_range="3-30 (main pool)",
            native_fold_unit="held-out country",
        ),
    ]
    return rows


# ---------------------------------------------------------------------------
# runners for the sensitivities whose cached data suffice
# ---------------------------------------------------------------------------
def _run_main(repro_data_root: Path, run_id: str, sensitivity: str, **options) -> pd.DataFrame:
    run_root = repro_data_root.resolve() / "results/analysis_runs" / run_id
    frames = []
    for bag in landscape.BAGS:
        inputs = landscape.load_bag(run_root, bag, **options)
        frames.append(landscape.comparison_rows(inputs, sensitivity=sensitivity))
    return pd.concat(frames, ignore_index=True)


def _hc_only_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the HC-only table into its candidate pool and its baseline."""
    frame = pd.read_csv(path, low_memory=False)
    baseline = frame[frame["candidate_family"].astype(str).eq("baseline")]
    pool = frame[~frame["candidate_family"].astype(str).eq("baseline")]
    return pool, baseline


def _run_hc_only(hc_root: Path, sensitivity: str) -> pd.DataFrame:
    rows = []
    for bag in landscape.BAGS:
        pool, baseline_frame = _hc_only_frames(hc_root / bag / "hc_only_country_all.csv")
        scores, baselines = [], []
        for rung in landscape.RUNGS:
            scores.append(
                landscape.landscape_scores(pool[pool["rung_id"].eq(rung)]).assign(bag=bag, rung=rung)
            )
            baselines.append(
                landscape.covariate_baseline(baseline_frame[baseline_frame["rung_id"].eq(rung)]).assign(
                    bag=bag, rung=rung
                )
            )
        inputs = landscape.LandscapeInputs(
            bag=bag,
            scores=pd.concat(scores, ignore_index=True),
            single=pd.DataFrame(columns=["fold_country", "single_landscape_r2", "bag", "rung"]),
            baseline=pd.concat(baselines, ignore_index=True),
            paths={f"{bag}/hc_only": str(hc_root / bag / "hc_only_country_all.csv")},
        )
        # The empty single frame makes comparison_rows omit that family.
        rows.append(landscape.comparison_rows(inputs, sensitivity=sensitivity))
    return pd.concat(rows, ignore_index=True)


def _run_domain_imbalance(delivery: Path, sensitivity: str, order_min: int, order_max: int) -> pd.DataFrame:
    """Rung comparisons over the complete domain-level candidate pool.

    The domain pool has a single objective and a combinatorially varying number
    of candidates per order, so the per-cell count is not fixed at 20 and the
    arm, single and baseline contrasts do not apply.
    """
    rows: list[dict[str, object]] = []
    for bag in landscape.BAGS:
        path = delivery / "domain_imbalance" / bag / "domain_imbalance_country_all.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["candidate_family"].astype(str).eq("best_single_per_domain")]
        frame = landscape._eligible(frame)
        frame["order"] = pd.to_numeric(frame["order"], errors="coerce").astype("Int64")
        frame = frame[frame["order"].between(order_min, order_max)]

        scores: dict[str, pd.Series] = {}
        for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
            subset = frame[frame["rung_id"].eq(rung)]
            per_set_size = subset.groupby(["fold_country", "order"], observed=True)["r2"].median()
            scores[rung] = per_set_size.groupby("fold_country", observed=True).mean()

        pairs = [
            ("xgb_tree_d2", "xgb_tree_d1"),
            ("xgb_tree_d3", "xgb_tree_d1"),
            ("xgb_tree_d3", "xgb_tree_d2"),
        ]
        for index, (high, low) in enumerate(pairs):
            rows.append(
                {
                    "sensitivity": sensitivity,
                    "bag": bag,
                    "comparison_family": "model_complexity",
                    "objective": "o_domain",
                    "model_a": "domain_landscape",
                    "model_b": "domain_landscape",
                    "rung_a": high,
                    "rung_b": low,
                    "multiplicity_family": f"{sensitivity}|{bag}|o_domain|model_complexity",
                    **landscape.paired_country_test(
                        scores[high], scores[low], seed=landscape.BOOTSTRAP_SEED + 700 + index
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["aggregation_definition"] = (
        "Landscape R2 over the complete domain-level candidate pool: median R2 across "
        "all candidates at each order, averaged with equal weight over orders 3-10. "
        "The domain pool has a single objective, so no arm contrast is defined."
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--release-run-id", default="main_k10_release_20260916")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/landscape_comparisons",
    )
    args = parser.parse_args()

    rows = build_scope(args.repro_data_root, args.source_run_id, args.release_run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scope_columns = [
        "sensitivity", "source_path", "full_candidate_landscape_available",
        "per_country_or_per_fold_metrics_available", "synergy_vs_redundancy_applicable",
        "rung_comparison_applicable", "single_landscape_comparison_applicable",
        "baseline_comparison_applicable", "requires_refit", "reason",
        "native_order_range", "native_fold_unit",
    ]
    scope = pd.DataFrame([{key: getattr(row, key) for key in scope_columns} for row in rows])
    scope.to_csv(args.output_dir / "sensitivity_landscape_test_scope.csv", index=False)

    delivery = ROOT / "outputs/main" / args.release_run_id / "sensitivity"
    hc_root = args.repro_data_root.resolve() / "sensitivity/hc_only"
    results: list[pd.DataFrame] = []
    for row in rows:
        if not row.runnable or row.runner == "main":
            continue
        if row.runner == "order_cap":
            results.append(
                _run_main(
                    args.repro_data_root,
                    args.source_run_id,
                    row.sensitivity,
                    order_max=row.options["order_max"],
                )
            )
        elif row.runner == "hc_only":
            results.append(_run_hc_only(hc_root, row.sensitivity))
        elif row.runner == "domain_imbalance":
            results.append(_run_domain_imbalance(delivery, row.sensitivity, **row.options))

    if results:
        combined = landscape.apply_holm(pd.concat(results, ignore_index=True))
        columns = [
            "comparison_family", "sensitivity", "bag", "objective", "model_a", "model_b",
            "rung_a", "rung_b", "n_countries", "estimate_a", "estimate_b", "delta_mean_r2",
            "median_paired_delta", "ci_lo", "ci_hi", "wilcoxon_statistic", "wilcoxon_p_raw",
            "holm_p", "rank_biserial", "multiplicity_family", "aggregation_definition",
        ]
        combined[columns].to_csv(args.output_dir / "sensitivity_landscape_comparisons.csv", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_sensitivity_landscape_scope",
        "model_fitting_performed": False,
        "statement": (
            "No model fitting was performed. Sensitivities whose cached outputs lack a "
            "complete candidate pool or a fold unit are recorded as unavailable rather "
            "than refitted."
        ),
        "n_sensitivities": len(rows),
        "runnable": sorted(row.sensitivity for row in rows if row.runnable),
        "not_runnable_without_refit": sorted(row.sensitivity for row in rows if row.requires_refit),
    }
    (args.output_dir / "sensitivity_landscape_scope_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir / 'sensitivity_landscape_test_scope.csv'} ({len(scope)} sensitivities)")


if __name__ == "__main__":
    main()
