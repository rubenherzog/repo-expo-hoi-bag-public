#!/usr/bin/env python3
"""Redraw the four estimand-dependent sensitivity figures under global OOF R2.

The delivered country-balanced package converts these four figures at delivery
time (``finalize_main_k10_delivery.refresh_country_balanced_sensitivity_figures``)
by replacing each candidate's score with the unweighted mean of its held-out
country R2.  The global delivery skips that conversion and uses the pooled
global OOF R2 that the sensitivity stages already computed and stored.

Nothing else changes: the same parent plotting functions, the same candidate
pools, rungs, baselines and BAG order are used, and no model is refitted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUNGS = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
ALL_RUNGS = ["ols", *RUNGS]
BAGS = ("structural", "functional")  # structural precedes functional throughout
R2_ESTIMAND = "global pooled out-of-fold LOCO R²"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument(
        "--source-run-id",
        default="main_k10_release_20260916",
        help="Completed sensitivity run supplying the stored per-candidate evaluations.",
    )
    parser.add_argument("--run-id", required=True, help="New immutable global-OOF run identifier.")
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required sensitivity evaluation is missing: {path}")
    return path


def _global_scores(frame: pd.DataFrame, path: Path) -> pd.Series:
    """The stored pooled global OOF R2, keyed by candidate."""
    if "global_oof_r2" not in frame.columns:
        raise ValueError(f"{path} lacks global_oof_r2")
    return (
        frame.assign(global_oof_r2=pd.to_numeric(frame["global_oof_r2"], errors="coerce"))
        .set_index(frame["candidate_id"].astype(str))["global_oof_r2"]
    )


def _global_reference_lines(main_run: Path, adapter_root: Path, bag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Covariate-only baseline and best main-analysis model per rung, under global OOF.

    The delivered country-balanced package stores these two reference lines as
    country-balanced values.  Reusing them in a global figure would mix the two
    estimands inside one panel, so they are rebuilt here from the same evaluated
    candidates using the pooled global OOF R2."""
    candidates = pd.read_parquet(
        _require(adapter_root / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet")
    )
    candidates["global_oof_r2"] = pd.to_numeric(candidates["global_oof_r2"], errors="coerce")
    best_rows, baseline_rows = [], []
    for rung in RUNGS:
        scoped = candidates[candidates["rung_id"].astype(str).eq(rung)]
        if scoped.empty:
            raise ValueError(f"No {bag}/{rung} candidates for the global reference lines")
        winner = scoped.loc[scoped["global_oof_r2"].idxmax()]
        best_rows.append(
            {
                # Two consumer contracts share this table: the PCA figure reads
                # best_model_r2 and the domain-imbalance figure reads the
                # original_complete_best_* family.  Both are emitted here.
                "rung_id": rung,
                "best_model_r2": float(winner["global_oof_r2"]),
                "original_complete_best_r2": float(winner["global_oof_r2"]),
                "original_complete_best_candidate_id": str(winner["candidate_id"]),
                "original_complete_best_objective": str(winner["objective"]),
                "original_complete_best_order": int(winner["order"]),
                "original_complete_best_predictors": str(winner["predictors_identity"]),
            }
        )
        baseline_global = pd.read_csv(_require(main_run / "xgb" / bag / rung / "baseline/metrics_global.csv"))
        baseline_rows.append(
            {"rung_id": rung, "baseline_r2": float(pd.to_numeric(baseline_global["global_oof_r2"], errors="coerce").iloc[0])}
        )
    return pd.DataFrame(baseline_rows), pd.DataFrame(best_rows)


def _domain_imbalance(run_root: Path, main_run: Path, adapter_root: Path, figures: Path) -> None:
    from repo_expo_hoi_bag.stages.run_domain_imbalance_sensitivity import _plot_domain_imbalance

    inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for bag in BAGS:
        bag_root = run_root / "domain_imbalance" / bag
        country = pd.read_csv(_require(bag_root / "domain_imbalance_country_all.csv"), low_memory=False)
        keys = [
            "candidate_id", "candidate_family", "source_label", "order", "score",
            "predictors_identity", "fit_candidate_id", "rung_id", "bag",
        ]
        metadata = country[keys].drop_duplicates(["candidate_id", "rung_id"])
        stored = pd.read_csv(_require(bag_root / "domain_imbalance_global_all.csv"))
        if "global_oof_r2" not in stored.columns:
            raise ValueError(f"{bag_root} global evaluation lacks global_oof_r2")
        scores = stored[["candidate_id", "rung_id", "global_oof_r2"]].copy()
        scores["global_oof_r2"] = pd.to_numeric(scores["global_oof_r2"], errors="coerce")
        summary = metadata.merge(scores, on=["candidate_id", "rung_id"], validate="one_to_one")
        if summary["global_oof_r2"].isna().any():
            raise ValueError(f"Missing global OOF R² for one or more {bag} domain candidates")

        best_single = pd.read_csv(_require(bag_root / "candidates/best_single_by_rung_source.csv"))
        for rung in RUNGS:
            rung_root = main_run / "xgb" / bag / rung
            single_global = pd.read_csv(_require(rung_root / "single/metrics_global.csv"))
            single_scores = _global_scores(single_global, rung_root / "single/metrics_global.csv")
            mask = best_single["source_rung"].astype(str).eq(rung)
            best_single.loc[mask, "global_oof_r2"] = (
                best_single.loc[mask, "candidate_id"].astype(str).map(single_scores)
            )
        baselines, original_best = _global_reference_lines(main_run, adapter_root, bag)
        inputs[bag] = {
            "summary": summary,
            "candidate_scores": pd.read_csv(_require(bag_root / "domain_imbalance_candidate_scores.csv")),
            "best_single": best_single,
            "baselines": baselines,
            "original_best": original_best,
        }
    _plot_domain_imbalance(inputs, figures, RUNGS, 20)


def _whole_exposome_pca(run_root: Path, main_run: Path, adapter_root: Path, figures: Path) -> None:
    from repo_expo_hoi_bag.stages.run_whole_exposome_pca_sensitivity import _plot_pca

    inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for bag in BAGS:
        bag_root = run_root / "whole_exposome_pca" / bag
        # Deliberately NOT country_balanced_summary: the stored global frame
        # already carries the pooled global OOF R2 in global_oof_r2.
        summary = pd.read_csv(_require(bag_root / "whole_exposome_pca_global_all.csv"))
        if "global_oof_r2" not in summary.columns:
            raise ValueError(f"{bag_root} PCA evaluation lacks global_oof_r2")
        baselines, original_best = _global_reference_lines(main_run, adapter_root, bag)
        inputs[bag] = {
            "variance": pd.read_csv(_require(bag_root / "whole_exposome_pca_variance.csv")),
            "summary": summary,
            "baselines": baselines,
            "original_best": original_best,
        }
    previous = os.environ.get("WHOLE_PCA_PERFORMANCE_ESTIMATOR")
    os.environ["WHOLE_PCA_PERFORMANCE_ESTIMATOR"] = "global-oof"
    try:
        _plot_pca(inputs, figures, RUNGS)
    finally:
        if previous is None:
            os.environ.pop("WHOLE_PCA_PERFORMANCE_ESTIMATOR", None)
        else:
            os.environ["WHOLE_PCA_PERFORMANCE_ESTIMATOR"] = previous


def _education_scanner(work: Path, figures: Path) -> list[str]:
    from repo_expo_hoi_bag.stages.run_education_scanner_baseline_sensitivity import (
        BASELINE_LABEL,
        _plot_comparison,
    )

    specifications = ("baseline_covariates", "plus_education", "plus_scanner", "plus_education_scanner")
    rungs = [
        rung
        for rung in ALL_RUNGS
        if all(
            (work / "education_scanner_baseline" / bag / specification / f"{bag}_{rung}_global.csv").is_file()
            for bag in BAGS
            for specification in specifications
        )
    ]
    inputs: dict[str, dict[str, object]] = {}
    for bag in BAGS:
        summaries: dict[str, pd.DataFrame] = {}
        for specification in specifications:
            pieces = []
            for rung in rungs:
                folder = work / "education_scanner_baseline" / bag / specification
                frame = pd.read_csv(_require(folder / f"{bag}_{rung}_global.csv"))
                if "global_oof_r2" not in frame.columns:
                    raise ValueError(f"{folder} lacks global_oof_r2")
                pieces.append(frame)
            summaries[specification] = pd.concat(pieces, ignore_index=True)
        inputs[bag] = {
            "baseline_summary": summaries[BASELINE_LABEL],
            "dist_by_variant": {k: v for k, v in summaries.items() if k != BASELINE_LABEL},
        }
    _plot_comparison(inputs, rungs, figures, r2_estimand=R2_ESTIMAND)
    return rungs


def _residualized_bag(work: Path, run_root: Path, figures: Path) -> pd.DataFrame:
    from repo_expo_hoi_bag.stages.run_residualized_bag_sensitivity import (
        _build_summary_table,
        _plot_residualized,
    )

    rungs = [
        rung
        for rung in ALL_RUNGS
        if all(
            (work / "residualized_bag" / bag / f"residualized_eval_{rung}" / f"{bag}_{rung}_global.csv").is_file()
            for bag in BAGS
        )
    ]
    rows = []
    for bag in BAGS:
        bag_rows = []
        for rung in rungs:
            folder = work / "residualized_bag" / bag / f"residualized_eval_{rung}"
            frame = pd.read_csv(_require(folder / f"{bag}_{rung}_global.csv"))
            if "global_oof_r2" not in frame.columns:
                raise ValueError(f"{folder} lacks global_oof_r2")
            frame["r2"] = pd.to_numeric(frame["global_oof_r2"], errors="coerce")
            frame["candidate_role"] = frame["objective"].astype(str)
            bag_rows.append(frame)
        rows.append(_build_summary_table(pd.concat(bag_rows, ignore_index=True), bag, rungs))
    summary = pd.concat(rows, ignore_index=True)
    destination = run_root / "residualized_bag"
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "residualized_bag_best_by_rung.csv", index=False)
    _plot_residualized(summary, list(BAGS), figures, "residualized_bag", r2_estimand=R2_ESTIMAND)
    return summary


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.resolve()
    source_root = ROOT / "outputs" / "main" / args.source_run_id / "sensitivity"
    run_root = ROOT / "outputs" / "main" / args.run_id / "sensitivity"
    figures = run_root / "figures"
    for stem in ("domain_imbalance_sensitivity", "whole_exposome_pca_sensitivity",
                 "education_scanner_baseline", "residualized_bag"):
        existing = figures / f"{stem}.pdf"
        if existing.exists():
            raise FileExistsError(f"Refusing to overwrite existing global figure: {existing}")
    figures.mkdir(parents=True, exist_ok=True)
    main_run = runtime / "results/analysis_runs/paper_reanalysis_k10"
    work = runtime / "work/analysis_runs" / args.source_run_id / "sensitivity_eval"

    adapter_root = runtime / "results/analysis_runs" / args.source_run_id / "input_adapter"
    _domain_imbalance(source_root, main_run, adapter_root, figures)
    _whole_exposome_pca(source_root, main_run, adapter_root, figures)
    education_rungs = _education_scanner(work, figures)
    residual_summary = _residualized_bag(work, run_root, figures)

    manifest = {
        "r2_mode": "global_oof",
        "r2_estimand": R2_ESTIMAND,
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "repro_data_root": str(runtime),
        "figures": sorted(p.name for p in figures.rglob("*.pdf")),
        "education_rungs": education_rungs,
        "residualized_rows": int(len(residual_summary)),
        "refitted_models": 0,
    }
    (run_root / "global_oof_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Rendered global-OOF sensitivity figures: {figures}")


if __name__ == "__main__":
    main()
