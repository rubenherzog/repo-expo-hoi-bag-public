#!/usr/bin/env python3
"""Which sensitivities can carry the cross-selected frontier without refitting.

The frontier test is less demanding than the whole-landscape test: it needs a
complete candidate x fold R2 grid that can be ranked with the evaluation fold
withheld, but it does not need exactly 20 candidates at every set size. This
stage re-asks the scope questions in frontier terms, updates the scope table,
and runs the K=20 frontier contrasts wherever the cached data already suffice.

No model is fitted. A sensitivity that would require refitting is recorded as
unavailable rather than manufactured.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.compute_landscape_model_comparisons import apply_holm
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import _domain_map, _jaccard

ROOT = frontier.ROOT
PRIMARY_K = frontier.PRIMARY_K


@dataclass(frozen=True)
class Scope:
    sensitivity: str
    source_path: str
    can_rank_excluding_evaluation_fold: bool
    per_candidate_fold_r2_available: bool
    top20_frontier_definable: bool
    multiple_rungs_available: bool
    baseline_available: bool
    single_exposure_available: bool
    requires_refit: bool
    reason: str
    native_fold_unit: str
    native_k: str
    runnable: bool = False
    runner: str = ""
    options: dict = field(default_factory=dict)


def build_scope(repro_data_root: Path, source_run_id: str, release_run_id: str) -> list[Scope]:
    runtime = repro_data_root.resolve()
    main_run = runtime / "results/analysis_runs" / source_run_id
    delivery = ROOT / "outputs/main" / release_run_id / "sensitivity"
    dedup = ROOT / "outputs/sensitivity/dedup"
    hc_only = runtime / "sensitivity/hc_only"

    return [
        Scope(
            sensitivity="main_k10_reference",
            source_path=str(main_run),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=True,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=True,
            requires_refit=False,
            reason=(
                "Reference analysis. Complete 560-candidate x country grid per arm at "
                "all four model levels, with single-exposure and covariate-baseline "
                "scopes, so every frontier contrast is available."
            ),
            native_fold_unit="held-out country",
            native_k="20",
            runnable=True,
            runner="main",
        ),
        Scope(
            sensitivity="order_cap",
            source_path=str(main_run),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=True,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=True,
            requires_refit=False,
            reason=(
                "Re-caps the same cached grid to set sizes <= 21, then ranks and "
                "evaluates exactly as the main analysis. No refitting."
            ),
            native_fold_unit="held-out country",
            native_k="20",
            runnable=True,
            runner="order_cap",
            options={"max_order": 21},
        ),
        Scope(
            sensitivity="hc_only",
            source_path=str(hc_only),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=True,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=False,
            requires_refit=False,
            reason=(
                "Healthy-control-only re-evaluation retains a complete 560-candidate x "
                "country grid per arm at all four model levels with an embedded "
                "covariate baseline. It has no single-exposure scope, so that one "
                "contrast is unavailable."
            ),
            native_fold_unit="held-out country",
            native_k="20",
            runnable=True,
            runner="hc_only",
        ),
        Scope(
            sensitivity="domain_imbalance",
            source_path=str(delivery / "domain_imbalance"),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=True,
            multiple_rungs_available=True,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=False,
            reason=(
                "Complete domain-level candidate x country grid for d1-d3, so a Top-20 "
                "frontier and the interaction-capable rung contrasts are definable. The "
                "pool has a single 'o_domain' objective, so there is no arm contrast, "
                "and its baseline is stored per model level rather than per country."
            ),
            native_fold_unit="held-out country",
            native_k="20",
            runnable=True,
            runner="domain_imbalance",
        ),
        Scope(
            sensitivity="whole_exposome_pca",
            source_path=str(delivery / "whole_exposome_pca"),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Only 10 nested principal-component candidates exist per model level, "
                "fewer than K=20, and they form an incremental ladder rather than a "
                "candidate pool. K is not silently reduced; the frontier is undefined."
            ),
            native_fold_unit="held-out country",
            native_k="not definable (10 candidates < K=20)",
        ),
        Scope(
            sensitivity="country_region_generalization",
            source_path=str(delivery / "country_region"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Stores per-fold medians over an already performance-selected top-20, "
                "not per-candidate region metrics. Candidates cannot be re-ranked with "
                "a region withheld, so no cross-selected frontier can be built."
            ),
            native_fold_unit="held-out region (9) / held-out country (22)",
            native_k="not definable",
        ),
        Scope(
            sensitivity="normative_transfer",
            source_path=str(delivery / "normative_transfer"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=True,
            requires_refit=True,
            reason=(
                "The complete candidate pool is retained per transfer condition and "
                "model level, but R2 is pooled across folds with no country column. "
                "Neither the ranking exclusion nor the paired statistical unit exists."
            ),
            native_fold_unit="none retained (pooled across folds)",
            native_k="not definable",
        ),
        Scope(
            sensitivity="residualized_bag",
            source_path=str(delivery / "residualized_bag"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Covers the full candidate pool at every model level but stores one "
                "global R2 per candidate. Without per-country values a candidate cannot "
                "be ranked with a country withheld."
            ),
            native_fold_unit="none retained (global R2 per candidate)",
            native_k="not definable",
        ),
        Scope(
            sensitivity="education_scanner_baseline",
            source_path=str(delivery / "education_scanner_baseline"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Only the selected winners were re-evaluated under the extended "
                "covariate set, as a global out-of-fold R2. No pool and no fold unit."
            ),
            native_fold_unit="none retained (global out-of-fold R2)",
            native_k="not definable",
        ),
        Scope(
            sensitivity="diagnosis_balance",
            source_path=str(delivery / "diagnosis_balance"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=False,
            baseline_available=True,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Re-weighting applied to the deployed winner models at one model level. "
                "No candidate pool was re-evaluated, so there is no frontier."
            ),
            native_fold_unit="held-out country",
            native_k="not definable",
        ),
        Scope(
            sensitivity="country_block_null",
            source_path=str(delivery / "country_block_null"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=False,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "A permutation null for the deployed winners. By design it evaluates "
                "one model per arm against permuted exposures, not a candidate pool."
            ),
            native_fold_unit="permuted country blocks",
            native_k="not definable",
        ),
        Scope(
            sensitivity="feature_ablation",
            source_path=str(dedup / "feature_ablation"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Ablations are defined relative to performance-selected top-20 parents "
                "and stored as global R2 losses, with no per-country values."
            ),
            native_fold_unit="none retained (global R2 loss)",
            native_k="not definable",
        ),
        Scope(
            sensitivity="residual_confounds",
            source_path=str(dedup / "residual_confounds"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=False,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "Mixed-effects decomposition of the deployed winners' residuals. The "
                "target is residual variance, not a candidate R2 frontier."
            ),
            native_fold_unit="participant within country",
            native_k="not definable",
        ),
        Scope(
            sensitivity="negative_o_arm_comparison",
            source_path=str(ROOT / "outputs/sensitivity/dedup_neg_o"),
            can_rank_excluding_evaluation_fold=False,
            per_candidate_fold_r2_available=False,
            top20_frontier_definable=False,
            multiple_rungs_available=True,
            baseline_available=False,
            single_exposure_available=False,
            requires_refit=True,
            reason=(
                "The alternative negative-O pool is delivered only as per-country R2 "
                "for the selected best synergy and redundancy models; its per-candidate "
                "country metrics are not in the delivered namespace."
            ),
            native_fold_unit="held-out country",
            native_k="not definable",
        ),
        Scope(
            sensitivity="selection_stability",
            source_path=str(ROOT / "outputs/sensitivity/selection_stability/main_k10"),
            can_rank_excluding_evaluation_fold=True,
            per_candidate_fold_r2_available=True,
            top20_frontier_definable=True,
            multiple_rungs_available=True,
            baseline_available=True,
            single_exposure_available=True,
            requires_refit=False,
            reason=(
                "Not a separate pool: it is the exact-winner diagnostic computed on the "
                "main k10 grid already analysed above. It motivates the frontier "
                "framework rather than providing an independent test of it."
            ),
            native_fold_unit="held-out country",
            native_k="20 (main pool)",
        ),
    ]


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def _frontier_from_grids(
    label: str,
    grids: dict[tuple[str, str], pd.DataFrame],
    baselines: dict[tuple[str, str], pd.Series],
    rungs: tuple[str, ...],
    arms: tuple[str, ...],
    bag: str,
    k: int = PRIMARY_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frontier scores and contrasts from explicit candidate x country grids."""
    scores: dict[tuple[str, str], pd.Series] = {}
    stability: list[dict[str, object]] = []
    for arm in arms:
        for rung in rungs:
            wide = grids[(arm, rung)]
            countries = sorted(str(value) for value in wide.columns)
            global_top = frozenset(frontier.top_k(wide, k))
            per_country: dict[str, float] = {}
            jaccards: list[float] = []
            for country in countries:
                selected = frontier.top_k(wide, k, exclude=country)
                per_country[country] = frontier.frontier_score(wide, selected, country)[0]
                jaccards.append(_jaccard(frozenset(selected), global_top))
            scores[(arm, rung)] = pd.Series(per_country, dtype=float)
            stability.append(
                {
                    "sensitivity": label,
                    "bag": bag,
                    "arm": arm,
                    "rung": rung,
                    "K": k,
                    "n_candidates": int(wide.shape[0]),
                    "n_countries": len(countries),
                    **frontier._stability_stats(jaccards, f"top{k}_candidate_jaccard"),
                }
            )

    rows: list[dict[str, object]] = []
    seed = frontier.BOOTSTRAP_SEED + 2000

    if len(arms) == 2:
        for index, rung in enumerate(rungs):
            rows.append(
                {
                    "sensitivity": label,
                    "comparison_family": "synergy_vs_redundancy",
                    "bag": bag,
                    "arm": "o_min_minus_o_max",
                    "rung_a": rung,
                    "rung_b": rung,
                    "multiplicity_family": f"{label}|{bag}|synergy_vs_redundancy",
                    **frontier.paired_country_test(
                        scores[("o_min", rung)], scores[("o_max", rung)], seed=seed + index
                    ),
                }
            )

    pairs = tuple((a, b) for a, b in frontier.RUNG_PAIRS if a in rungs and b in rungs)
    for arm_index, arm in enumerate(arms):
        for pair_index, (high, low) in enumerate(pairs):
            rows.append(
                {
                    "sensitivity": label,
                    "comparison_family": "model_complexity",
                    "bag": bag,
                    "arm": arm,
                    "rung_a": high,
                    "rung_b": low,
                    "multiplicity_family": f"{label}|{bag}|{arm}|model_complexity",
                    **frontier.paired_country_test(
                        scores[(arm, high)],
                        scores[(arm, low)],
                        seed=seed + 100 + 10 * arm_index + pair_index,
                    ),
                }
            )

    for arm_index, arm in enumerate(arms):
        for rung_index, rung in enumerate(rungs):
            baseline = baselines.get((arm, rung))
            if baseline is None:
                continue
            rows.append(
                {
                    "sensitivity": label,
                    "comparison_family": "vs_covariate_baseline",
                    "bag": bag,
                    "arm": arm,
                    "rung_a": rung,
                    "rung_b": rung,
                    "multiplicity_family": f"{label}|{bag}|vs_covariate_baseline",
                    **frontier.paired_country_test(
                        scores[(arm, rung)],
                        baseline,
                        seed=seed + 300 + 10 * arm_index + rung_index,
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame["K"] = k
    return frame, pd.DataFrame(stability)


def _run_main(runtime: Path, source_run_id: str, label: str, max_order: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = frontier._registry(frontier._registry_path(runtime, source_run_id))
    results, stabilities = [], []
    for bag in frontier.BAGS:
        inputs = frontier.load_bag(runtime, source_run_id, bag, registry)
        identity = registry[registry["bag"].eq(bag)]
        keep = set(
            identity.loc[pd.to_numeric(identity["order"], errors="raise").le(max_order), "candidate_id"]
            .astype(str)
        )
        grids = {
            (arm, rung): inputs.multivariate[rung].loc[
                [c for c in inputs.arms[arm] if c in keep]
            ]
            for arm in frontier.ARMS
            for rung in frontier.RUNGS
        }
        baselines = {
            (arm, rung): inputs.baseline[rung] for arm in frontier.ARMS for rung in frontier.RUNGS
        }
        result, stability = _frontier_from_grids(
            label, grids, baselines, frontier.RUNGS, frontier.ARMS, bag
        )
        results.append(result)
        stabilities.append(stability)
    return pd.concat(results, ignore_index=True), pd.concat(stabilities, ignore_index=True)


def _run_hc_only(hc_root: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, stabilities = [], []
    for bag in frontier.BAGS:
        frame = pd.read_csv(hc_root / bag / "hc_only_country_all.csv", low_memory=False)
        frame = frame[pd.to_numeric(frame["n_test"], errors="coerce").gt(0)]
        frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
        pool = frame[~frame["candidate_family"].astype(str).eq("baseline")].copy()
        parsed = pool["candidate_id"].astype(str).str.extract(r"_(o_min|o_max)_ord(\d+)_")
        pool["arm"] = parsed[0]
        pool["order"] = pd.to_numeric(parsed[1], errors="coerce")
        pool = pool[pool["arm"].notna() & pool["order"].le(frontier.MAX_ORDER)]

        grids = {}
        for arm in frontier.ARMS:
            for rung in frontier.RUNGS:
                subset = pool[pool["arm"].eq(arm) & pool["rung_id"].eq(rung)]
                grids[(arm, rung)] = frontier._wide(subset, f"hc_only/{bag}/{arm}/{rung}")

        baseline_frame = frame[frame["candidate_family"].astype(str).eq("baseline")]
        baselines = {}
        for rung in frontier.RUNGS:
            rung_baseline = baseline_frame[baseline_frame["rung_id"].eq(rung)]
            series = rung_baseline.set_index("fold_country")["r2"].astype(float)
            for arm in frontier.ARMS:
                baselines[(arm, rung)] = series

        result, stability = _frontier_from_grids(
            label, grids, baselines, frontier.RUNGS, frontier.ARMS, bag
        )
        results.append(result)
        stabilities.append(stability)
    return pd.concat(results, ignore_index=True), pd.concat(stabilities, ignore_index=True)


def _run_domain_imbalance(delivery: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rungs = ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
    results, stabilities = [], []
    for bag in frontier.BAGS:
        frame = pd.read_csv(
            delivery / "domain_imbalance" / bag / "domain_imbalance_country_all.csv", low_memory=False
        )
        frame = frame[frame["candidate_family"].astype(str).eq("best_single_per_domain")]
        frame = frame[pd.to_numeric(frame["n_test"], errors="coerce").gt(0)]
        frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
        grids = {
            ("o_domain", rung): frontier._wide(
                frame[frame["rung_id"].eq(rung)], f"domain/{bag}/{rung}"
            )
            for rung in rungs
        }
        result, stability = _frontier_from_grids(label, grids, {}, rungs, ("o_domain",), bag)
        results.append(result)
        stabilities.append(stability)
    return pd.concat(results, ignore_index=True), pd.concat(stabilities, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--release-run-id", default="main_k10_release_20260916")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/sensitivity/performance_frontier"
    )
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    rows = build_scope(runtime, args.source_run_id, args.release_run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "sensitivity", "source_path", "can_rank_excluding_evaluation_fold",
        "per_candidate_fold_r2_available", "top20_frontier_definable",
        "multiple_rungs_available", "baseline_available", "single_exposure_available",
        "requires_refit", "reason", "native_fold_unit", "native_k",
    ]
    scope = pd.DataFrame([{key: getattr(row, key) for key in columns} for row in rows])
    scope.to_csv(args.output_dir / "sensitivity_frontier_test_scope.csv", index=False)

    delivery = ROOT / "outputs/main" / args.release_run_id / "sensitivity"
    hc_root = runtime / "sensitivity/hc_only"
    results, stabilities = [], []
    for row in rows:
        if not row.runnable or row.runner == "main":
            continue
        if row.runner == "order_cap":
            result, stability = _run_main(
                runtime, args.source_run_id, row.sensitivity, row.options["max_order"]
            )
        elif row.runner == "hc_only":
            result, stability = _run_hc_only(hc_root, row.sensitivity)
        elif row.runner == "domain_imbalance":
            result, stability = _run_domain_imbalance(delivery, row.sensitivity)
        else:
            continue
        results.append(result)
        stabilities.append(stability)

    if results:
        combined = apply_holm(pd.concat(results, ignore_index=True))
        combined.to_csv(args.output_dir / "sensitivity_frontier_comparisons.csv", index=False)
        pd.concat(stabilities, ignore_index=True).to_csv(
            args.output_dir / "sensitivity_frontier_stability.csv", index=False
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "compute_sensitivity_frontier_scope",
        "model_fitting_performed": False,
        "primary_k": PRIMARY_K,
        "statement": (
            "No model fitting was performed. K is never silently reduced: a pool with "
            "fewer than K candidates is recorded as not definable."
        ),
        "nesting_caveat": frontier.NESTING_CAVEAT,
        "n_sensitivities": len(rows),
        "runnable": sorted(row.sensitivity for row in rows if row.runnable),
        "not_runnable_without_refit": sorted(row.sensitivity for row in rows if row.requires_refit),
    }
    (args.output_dir / "sensitivity_frontier_scope_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir / 'sensitivity_frontier_test_scope.csv'} ({len(scope)} sensitivities)")


if __name__ == "__main__":
    main()
