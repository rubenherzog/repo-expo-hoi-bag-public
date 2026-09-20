#!/usr/bin/env python3
"""Evaluate frozen endpoint HPOs and top-50 candidates under one country policy."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.analysis.xgb_frozen_top50 import (
    baseline_candidate, country_balanced_scores, endpoint_summary, single_exposure_candidates,
)
from repo_expo_hoi_bag.analysis.xgb_tuned_top50 import (
    load_comparison_bag_policy, select_historical_top_candidates,
)
from repo_expo_hoi_bag.config.models import load_historical_paper_reference
from scripts.sensitivity_common import analysis_cfg_from_config, build_original_model_df, evaluate_candidates_by_rung, load_raw_and_domains, load_sensitivity_config
from xgb_nested_loco_tuning import resolve_tuned_fold_configs


STAGE_NAME = "xgb_frozen_cap500_top50"
ARTIFACT_ENV = {
    "baseline": "XGB_FROZEN_BASELINE_ARTIFACT",
    "single": "XGB_FROZEN_SINGLE_ARTIFACT",
    "full63": "XGB_FROZEN_FULL63_ARTIFACT",
    "k10": "XGB_FROZEN_K10_ARTIFACT",
}
EXPECTED_SCOPE = {
    "baseline": "baseline", "single": "single_exposure",
    "full63": "full_exposome", "k10": "domain_balanced_k10",
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _artifact(checkout: Path, label: str, policy: dict) -> tuple[Path, dict]:
    raw = os.environ.get(ARTIFACT_ENV[label], "").strip()
    if not raw:
        raise ValueError(f"{ARTIFACT_ENV[label]} is required")
    path = Path(raw)
    path = path if path.is_absolute() else checkout / path
    path = path.resolve()
    root = (checkout / "outputs" / "xgb_nested_loco_tuning").resolve()
    if path.name != "selected_xgb_configs.json" or not path.is_file() or path.parent.parent != root:
        raise ValueError(f"{ARTIFACT_ENV[label]} must be a completed local selected-config artifact")
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    observed = manifest["two_phase_country_config"]
    if (
        manifest.get("feature_scope") != EXPECTED_SCOPE[label]
        or observed.get("variant") != policy["variant"]
        or list(observed.get("bag_exclude_countries", [])) != policy["exclude_countries"]
        or list(observed.get("bag_exclude_diagnosis", [])) != policy["exclude_diagnosis"]
    ):
        raise ValueError(f"{label} artifact does not match required scope/country policy: {path}")
    return path, {"path": str(path), "sha256": _sha256(path), "manifest_sha256": _sha256(manifest_path)}


def _top50_candidates(checkout: Path, cfg: dict, bag: str, rung_id: str) -> pd.DataFrame:
    reference = load_historical_paper_reference(checkout / "config" / "paper_reference.yaml")
    template = cfg["xgb_tuned_top50_comparison"]["historical_metrics_relative_path_template"]
    path = (reference.canonical_root / str(template).format(bag=bag)).resolve()
    path.relative_to(reference.root)
    return select_historical_top_candidates(pd.read_parquet(path), bag=bag, rung_id=rung_id, top_n=50)


def main() -> None:
    runtime = Path(os.environ["REPRO_DATA_ROOT"]).resolve()
    checkout = Path(os.environ["REPO_CHECKOUT_ROOT"]).resolve()
    variant = os.environ.get("XGB_FROZEN_TOP50_VARIANT", "").strip()
    run_id = os.environ.get("XGB_FROZEN_TOP50_RUN_ID", "").strip()
    n_jobs = int(os.environ.get("XGB_FROZEN_TOP50_N_JOBS", "20"))
    if variant not in {"a", "historical_a"} or not run_id.replace("_", "").replace("-", "").isalnum() or n_jobs < 1:
        raise ValueError("Set a valid XGB_FROZEN_TOP50_VARIANT, RUN_ID, and positive N_JOBS")
    policy = load_comparison_bag_policy(checkout / "config" / "country_exclusions.yaml", variant)
    artifacts = {label: _artifact(checkout, label, policy) for label in ARTIFACT_ENV}
    cfg = load_sensitivity_config()
    raw, _domains, features, _domain_map = load_raw_and_domains()
    analysis_cfg = analysis_cfg_from_config(cfg)
    analysis_cfg["exclude_countries"] = policy["exclude_countries"]
    analysis_cfg["exclude_diagnosis"] = policy["exclude_diagnosis"]
    model_df = build_original_model_df(raw, features, analysis_cfg)
    outdir = runtime / "results" / "sensitivity" / STAGE_NAME / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "stage": STAGE_NAME, "variant": variant, "country_policy": policy,
        "artifacts": {label: meta for label, (_path, meta) in artifacts.items()},
        "n_jobs": n_jobs, "country_metric": "mean per-country LOCO R2", "top_n": 50,
    }
    (outdir / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    panels: dict[str, list[pd.DataFrame]] = {key: [] for key in ARTIFACT_ENV}
    for bag in ("structural", "functional"):
        for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
            candidates = {
                "baseline": baseline_candidate(),
                "single": single_exposure_candidates(features),
                "full63": _top50_candidates(checkout, cfg, bag, rung),
                "k10": _top50_candidates(checkout, cfg, bag, rung),
            }
            for label, candidate_df in candidates.items():
                artifact_path, _meta = artifacts[label]
                arm_dir = outdir / bag / rung / label
                _global, country = evaluate_candidates_by_rung(
                    model_df=model_df, candidate_df=candidate_df, exposome_cols=features,
                    bag=bag, rungs=[rung], analysis_cfg=analysis_cfg, outdir=arm_dir,
                    n_jobs=n_jobs, tuning_artifact_path=artifact_path, tuning_strict=True,
                )
                expected_countries = int(country["fold_country"].nunique())
                scores = country_balanced_scores(country, expected_countries=expected_countries)
                scores.insert(0, "bag_target", bag)
                panels[label].append(scores)
                print(f"[frozen-top50] variant={variant} bag={bag} rung={rung} arm={label} candidates={len(scores)}", flush=True)
    combined = {label: pd.concat(parts, ignore_index=True) for label, parts in panels.items()}
    for label, table in combined.items():
        table.to_csv(outdir / f"{label}_country_balanced_scores.csv", index=False)
    summary = endpoint_summary(combined["baseline"], combined["single"], combined["full63"], combined["k10"])
    summary.to_csv(outdir / "frozen_cap500_top50_summary.csv", index=False)
    print(f"Completed {STAGE_NAME}: {outdir}", flush=True)


if __name__ == "__main__":
    main()
