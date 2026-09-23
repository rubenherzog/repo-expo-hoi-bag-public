#!/usr/bin/env python3
"""Fit missing education-adjusted single-exposure models and render the no-scanner figure.

This is deliberately a narrow companion to the established education/scanner
sensitivity. It reuses its complete-case cohort and +Education covariate
specification so the new best-single marks are directly comparable with the
already evaluated greedy pool. Scanner identity is used only to reproduce the
existing cohort; it is never included as a predictor in these fits or shown in
the rendered figure.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.run_education_scanner_baseline_sensitivity import (
    _attach_education_and_scanner,
    _plot_education_no_scanner,
)
from scripts.sensitivity_common import (
    analysis_cfg_from_config,
    build_original_model_df,
    evaluate_candidates_by_rung,
    load_raw_and_domains,
    load_sensitivity_config,
)


def _single_pool(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate_id": f"single__{feature}",
            "feature_id": f"single__{feature}",
            "objective": "single_exposure",
            "order": 1,
            "rank": index + 1,
            "score": pd.NA,
            "nplet_vars": [feature],
            "predictors_identity": feature,
            "predictors_identity_n": 1,
            "candidate_family": "single_exposure",
            "source_label": "single_exposure",
        }
        for index, feature in enumerate(feature_names)
    ])


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is required")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")
    return path


def main() -> None:
    source_eval = _required_path("EDUCATION_NO_SCANNER_SOURCE_EVAL_ROOT")
    figure_root = _required_path("EDUCATION_NO_SCANNER_FIGURE_ROOT")
    result_root = _required_path("EDUCATION_NO_SCANNER_RESULT_ROOT")
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "20"))
    rungs = [r.strip() for r in os.environ.get(
        "SENSITIVITY_RUNGS", "ols,xgb_tree_d1,xgb_tree_d2,xgb_tree_d3"
    ).split(",") if r.strip()]
    expected_rungs = {"ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"}
    if not set(rungs).issubset(expected_rungs):
        raise ValueError(f"Unsupported rung(s): {rungs}")

    cfg = load_sensitivity_config()
    analysis_cfg = {**analysis_cfg_from_config(cfg), "include_education": True}
    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = _attach_education_and_scanner(
        build_original_model_df(raw, feature_names, analysis_cfg), raw
    )
    complete_case = full_model[full_model["Edu"].notna() & full_model["scanner_id"].notna()].copy()
    if complete_case.empty:
        raise ValueError("Education/scanner complete-case cohort is empty")
    pool = _single_pool(feature_names)
    pool.to_csv(result_root / "education_adjusted_single_candidate_pool.csv", index=False)

    bag_inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for bag in ("structural", "functional"):
        education_dir = source_eval / bag / "plus_education"
        # The legacy aggregate was written once per independently completed
        # rung and can therefore contain only the last writer. Assemble the
        # authoritative per-rung files instead.
        education_parts = []
        for rung in rungs:
            education_path = education_dir / f"{bag}_{rung}_global.csv"
            if not education_path.is_file():
                raise FileNotFoundError(f"Missing existing +Education summary: {education_path}")
            education_parts.append(pd.read_csv(education_path))
        education = pd.concat(education_parts, ignore_index=True)
        missing_rungs = set(rungs) - set(education["rung_id"].astype(str))
        if missing_rungs:
            raise ValueError(f"Existing +Education source is missing {bag} rungs: {sorted(missing_rungs)}")
        singles, _country = evaluate_candidates_by_rung(
            model_df=complete_case,
            candidate_df=pool,
            exposome_cols=feature_names,
            bag=bag,
            rungs=rungs,
            analysis_cfg=analysis_cfg,
            outdir=result_root / bag,
            n_jobs=n_jobs,
        )
        singles.to_csv(result_root / bag / f"{bag}_education_adjusted_singles.csv", index=False)
        bag_inputs[bag] = {"education": education, "singles": singles}

    _plot_education_no_scanner(
        bag_inputs, rungs, figure_root,
        source_paths=[str(source_eval), str(result_root)],
    )


if __name__ == "__main__":
    main()
