#!/usr/bin/env python3
"""Ridge LOCO comparator pipeline for exposome-BAG.

This runner is intentionally separate from the canonical OLS/XGB pipeline. It
reuses the existing candidate pool and LOCO machinery, then evaluates Ridge on
main-effect and pairwise feature expansions. Triples are available only when
explicitly requested through RIDGE_RUNGS.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.ridge_common import (
    evaluate_ridge_candidates,
    ridge_analysis_cfg,
    ridge_bags,
    ridge_candidate_pool,
    ridge_eval_root,
    ridge_families,
    ridge_output_root,
    ridge_rungs,
    build_ridge_model_df,
)
from scripts.sensitivity_common import log_msg


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y"}


def _write_run_manifest(
    root: Path,
    *,
    eval_root: Path,
    bags: list[str],
    rungs: list[str],
    n_jobs: int,
    smoke: bool,
) -> None:
    rows = [
        {"key": "local_summary_root", "value": str(root)},
        {"key": "external_eval_root", "value": str(eval_root)},
        {"key": "bags", "value": ",".join(bags)},
        {"key": "rungs", "value": ",".join(rungs)},
        {"key": "n_jobs", "value": str(n_jobs)},
        {"key": "smoke", "value": str(bool(smoke))},
        {"key": "RIDGE_ORDER_MAX", "value": os.environ.get("RIDGE_ORDER_MAX", "30")},
        {"key": "RIDGE_ORDER_MAX_TRIPLE", "value": os.environ.get("RIDGE_ORDER_MAX_TRIPLE", "")},
        {"key": "RIDGE_TOP_K_PER_ORDER", "value": os.environ.get("RIDGE_TOP_K_PER_ORDER", "20")},
        {"key": "RIDGE_ALPHAS", "value": os.environ.get("RIDGE_ALPHAS", "logspace(-3,4,8)")},
        {"key": "RIDGE_EVAL_CHUNK_SIZE", "value": os.environ.get("RIDGE_EVAL_CHUNK_SIZE", "120")},
    ]
    pd.DataFrame(rows).to_csv(root / "ridge_run_manifest.csv", index=False)


def main() -> None:
    smoke = _env_bool("RIDGE_SMOKE") or _env_bool("SMOKE_TEST")
    include_combined = _env_bool("RIDGE_INCLUDE_COMBINED", default=False)
    bags = ridge_bags(include_combined=include_combined)
    rungs = ridge_rungs()
    if smoke:
        os.environ.setdefault("RIDGE_TOP_K_PER_ORDER", "1")
    n_jobs = int(os.environ.get("RIDGE_N_JOBS", "40"))
    local_root = ridge_output_root()
    eval_root = ridge_eval_root(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(local_root, eval_root=eval_root, bags=bags, rungs=rungs, n_jobs=n_jobs, smoke=smoke)
    _write_run_manifest(eval_root, eval_root=eval_root, bags=bags, rungs=rungs, n_jobs=n_jobs, smoke=smoke)

    model_df, feature_names = build_ridge_model_df()
    all_summary = []
    eval_country = []
    eval_alpha = []
    eval_complexity = []

    for bag in bags:
        pool = ridge_candidate_pool(bag, smoke=smoke)
        pool.to_csv(local_root / f"{bag}_ridge_candidate_pool.csv", index=False)
        pool.to_csv(eval_root / f"{bag}_ridge_candidate_pool.csv", index=False)
        log_msg(f"Ridge candidate pool bag={bag}: rows={len(pool)}")
        for family in ridge_families():
            analysis_cfg = ridge_analysis_cfg(include_diagnosis=family.include_diagnosis)
            outdir = eval_root / bag / family.family_id / f"train_{family.train_label}" / f"test_{family.test_label}"
            outdir.mkdir(parents=True, exist_ok=True)
            summary, country, alpha, complexity = evaluate_ridge_candidates(
                model_df=model_df,
                candidate_df=pool,
                exposome_cols=feature_names,
                bag=bag,
                family=family,
                rungs=rungs,
                analysis_cfg=analysis_cfg,
                outdir=outdir,
                n_jobs=n_jobs,
            )
            for df in (summary, country, alpha, complexity):
                if not df.empty:
                    df["bag"] = bag
            all_summary.append(summary)
            eval_country.append(country)
            eval_alpha.append(alpha)
            eval_complexity.append(complexity)

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        summary_all.to_csv(local_root / "ridge_global_all.csv", index=False)
        summary_all.to_csv(eval_root / "ridge_global_all.csv", index=False)
    if eval_country:
        pd.concat(eval_country, ignore_index=True).to_csv(eval_root / "ridge_country_all.csv", index=False)
    if eval_alpha:
        pd.concat(eval_alpha, ignore_index=True).to_csv(eval_root / "ridge_alpha_all.csv", index=False)
    if eval_complexity:
        pd.concat(eval_complexity, ignore_index=True).to_csv(eval_root / "ridge_complexity_all.csv", index=False)

    log_msg(f"Ridge pipeline complete: local summaries={local_root} detailed eval={eval_root}")


if __name__ == "__main__":
    main()
