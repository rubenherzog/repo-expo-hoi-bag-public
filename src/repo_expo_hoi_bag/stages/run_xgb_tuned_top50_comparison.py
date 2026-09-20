#!/usr/bin/env python3
"""Paired LOCO comparison of fixed and country-tuned functional XGB d3 models."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.config.models import load_historical_paper_reference
from repo_expo_hoi_bag.analysis.xgb_tuned_top50 import (
    full_exposome_candidate,
    checkpoint_best_global_source,
    load_comparison_bag_policy,
    normalize_evaluator_global_rung,
    paired_arm_delta,
    resolve_comparison_candidate_mode,
    resolve_comparison_ensemble_size,
    resolve_comparison_top_n,
    select_historical_top_candidates,
    select_top_trial_ensemble_configs,
)
from oinfo_bag_ladder.config import BAG_TARGETS
from oinfo_bag_ladder.io_utils import stable_hash
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
from scripts.sensitivity_common import (
    analysis_cfg_from_config,
    build_original_model_df,
    evaluate_candidates_by_rung,
    load_raw_and_domains,
    load_sensitivity_config,
)
from xgb_nested_loco_tuning import TUNABLE_KEYS, resolve_tuned_fold_configs


STAGE_NAME = "xgb_tuned_top50_comparison"


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be beneath {root}: {resolved}") from exc
    return resolved


def _required_tuning_artifact(checkout_root: Path) -> Path:
    raw = os.environ.get("XGB_TUNING_ARTIFACT", "").strip()
    if not raw:
        raise ValueError(
            "XGB_TUNING_ARTIFACT is required and must name a completed local "
            "outputs/xgb_nested_loco_tuning/<signature>/selected_xgb_configs.json file"
        )
    path = Path(raw)
    path = path if path.is_absolute() else checkout_root / path
    expected_root = checkout_root / "outputs" / "xgb_nested_loco_tuning"
    path = _under(path, expected_root)
    if path.name != "selected_xgb_configs.json" or not path.is_file():
        raise ValueError(f"XGB_TUNING_ARTIFACT is not a completed selected-config artifact: {path}")
    return path


def _validate_artifact_policy(tuning_artifact: Path, policy: dict) -> dict:
    """Reject a comparison whose refit population differs from the tuning run."""
    manifest_path = tuning_artifact.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tuning manifest is required beside the selected artifact: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tuned_policy = dict(manifest.get("two_phase_country_config", {}))
    expected = {
        "variant": policy["variant"],
        "exclude_countries": policy["exclude_countries"],
        "exclude_diagnosis": policy["exclude_diagnosis"],
    }
    observed = {
        "variant": tuned_policy.get("variant"),
        "exclude_countries": list(tuned_policy.get("bag_exclude_countries", [])),
        "exclude_diagnosis": list(tuned_policy.get("bag_exclude_diagnosis", [])),
    }
    if observed != expected:
        raise ValueError(
            "Tuning artifact country policy does not match the requested comparison policy: "
            f"artifact={observed}, requested={expected}"
        )
    return {"path": str(manifest_path), "sha256": _sha256_file(manifest_path), **observed}


def _source_metrics_path(checkout_root: Path, cfg: dict, bag: str) -> tuple[Path, dict]:
    """Resolve paper candidates from the configured read-only reference runtime."""
    reference_path = checkout_root / "config" / "paper_reference.yaml"
    reference = load_historical_paper_reference(reference_path)
    template = str(cfg[STAGE_NAME]["historical_metrics_relative_path_template"])
    try:
        relative = Path(template.format(bag=bag))
    except KeyError as exc:
        raise ValueError("historical_metrics_relative_path_template may use only the {bag} placeholder") from exc
    if relative.is_absolute():
        raise ValueError("historical_metrics_relative_path must be relative to the historical paper reference")
    path = _under(reference.canonical_root / relative, reference.root)
    if not path.is_file():
        raise FileNotFoundError(f"Historical paper metrics are missing: {path}")
    return path, {
        "config_path": str(reference_path),
        "config_sha256": _sha256_file(reference_path),
        "root": str(reference.root),
        "canonical_root": str(reference.canonical_root),
        "read_only": True,
    }


def _unique_run_dir(runtime_root: Path, fingerprint: str) -> Path:
    test_dir = os.environ.get("XGB_TOP50_COMPARISON_TEST_DIR", "").strip()
    base = runtime_root / "results" / "sensitivity" / STAGE_NAME
    if test_dir:
        if not test_dir.replace("-", "").replace("_", "").isalnum():
            raise ValueError("XGB_TOP50_COMPARISON_TEST_DIR may contain only letters, digits, '-' and '_'")
        # Explicitly opt-in test mode: callers may overwrite only this one
        # bounded external directory, never an arbitrary path or production run.
        destination = base / test_dir
        destination.mkdir(parents=True, exist_ok=True)
        return destination
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + fingerprint[:16]
    destination = base / label
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def main() -> None:
    runtime_raw = os.environ.get("REPRO_DATA_ROOT", "").strip()
    checkout_raw = os.environ.get("REPO_CHECKOUT_ROOT", "").strip()
    if not runtime_raw or not checkout_raw:
        raise EnvironmentError("REPRO_DATA_ROOT and REPO_CHECKOUT_ROOT are required")
    runtime_root = Path(runtime_raw).resolve()
    checkout_root = Path(checkout_raw).resolve()
    cfg = load_sensitivity_config()
    stage_cfg = dict(cfg[STAGE_NAME])
    bag = os.environ.get("XGB_TUNED_COMPARISON_BAG", str(stage_cfg["bag"])).strip()
    rung_id = os.environ.get("XGB_TUNED_COMPARISON_RUNG", str(stage_cfg["rung_id"])).strip()
    stage_cfg["bag"] = bag
    stage_cfg["rung_id"] = rung_id
    top_n = resolve_comparison_top_n(int(stage_cfg["top_n"]), os.environ)
    stage_cfg["top_n"] = top_n
    candidate_mode = resolve_comparison_candidate_mode(os.environ)
    ensemble_size = resolve_comparison_ensemble_size(os.environ)
    if bag not in {"structural", "functional"}:
        raise ValueError(f"{STAGE_NAME} supports structural or functional BAGs, got {bag!r}")
    if rung_id not in {"xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"}:
        raise ValueError(f"{STAGE_NAME} supports XGBoost d1, d2, or d3, got {rung_id!r}")
    checkpoint_signature = os.environ.get("XGB_TUNING_CHECKPOINT_RUN_SIGNATURE", "").strip()
    if checkpoint_signature and os.environ.get("XGB_TUNING_ARTIFACT", "").strip():
        raise ValueError("Set either XGB_TUNING_ARTIFACT or XGB_TUNING_CHECKPOINT_RUN_SIGNATURE, not both")
    tuning_artifact = None if checkpoint_signature else _required_tuning_artifact(checkout_root)
    comparison_variant = os.environ.get("XGB_TUNED_COMPARISON_VARIANT", "a").strip() or "a"
    comparison_policy = load_comparison_bag_policy(
        checkout_root / "config" / "country_exclusions.yaml", comparison_variant
    )
    artifact_policy = _validate_artifact_policy(tuning_artifact, comparison_policy) if tuning_artifact else None
    checkpoint_source = (
        checkpoint_best_global_source(
            runtime_root, checkpoint_signature, bag, rung_id, tunable_keys=TUNABLE_KEYS
        ) if checkpoint_signature else None
    )
    diagnostics_path = tuning_artifact.parent / "trial_diagnostics.parquet" if tuning_artifact else None
    if checkpoint_source and ensemble_size > 1:
        raise ValueError("An active checkpoint supports only a one-config comparison")
    if ensemble_size > 1 and (diagnostics_path is None or not diagnostics_path.is_file()):
        raise FileNotFoundError(
            f"The {ensemble_size}-member tuned ensemble requires trial diagnostics: {diagnostics_path}"
        )
    source_metrics, paper_reference = (
        _source_metrics_path(checkout_root, cfg, bag)
        if candidate_mode == "historical_top50"
        else (None, None)
    )
    comparison_purpose = os.environ.get("XGB_TOP50_COMPARISON_PURPOSE", "internal_check").strip() or "internal_check"
    provenance = {
        "stage": STAGE_NAME,
        "comparison_purpose": comparison_purpose,
        "stage_cfg": stage_cfg,
        "candidate_mode": candidate_mode,
        "comparison_country_policy": comparison_policy,
        "tuning_artifact_country_policy": artifact_policy,
        "tuned_ensemble_size": ensemble_size,
        "tuning_artifact": str(tuning_artifact) if tuning_artifact else None,
        "tuning_artifact_sha256": _sha256_file(tuning_artifact) if tuning_artifact else None,
        "tuning_checkpoint_source": checkpoint_source,
        "trial_diagnostics": str(diagnostics_path) if ensemble_size > 1 and diagnostics_path else None,
        "trial_diagnostics_sha256": _sha256_file(diagnostics_path) if ensemble_size > 1 and diagnostics_path else None,
        "historical_metrics": str(source_metrics) if source_metrics else None,
        "historical_metrics_sha256": _sha256_file(source_metrics) if source_metrics else None,
        "historical_paper_reference": paper_reference,
        "country_policy_sha256": _sha256_file(checkout_root / "config" / "country_exclusions.yaml"),
        "stage_sha256": _sha256_file(Path(__file__)),
    }
    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    if candidate_mode == "historical_top50":
        historical = pd.read_parquet(source_metrics)
        candidates = select_historical_top_candidates(historical, bag=bag, rung_id=rung_id, top_n=top_n)
        candidate_output_name = "historical_top50_candidates.parquet"
    else:
        candidates = full_exposome_candidate(feature_names)
        candidate_output_name = "full_exposome_candidate.parquet"
    analysis_cfg = analysis_cfg_from_config(cfg)
    analysis_cfg["exclude_countries"] = list(comparison_policy["exclude_countries"])
    analysis_cfg["exclude_diagnosis"] = list(comparison_policy["exclude_diagnosis"])
    model_df = build_original_model_df(raw, feature_names, analysis_cfg)
    rung = next((spec for spec in get_rung_specs() if spec["rung_id"] == rung_id), None)
    if rung is None:
        raise ValueError(f"Configured rung is unavailable: {rung_id}")
    xgb_cfg = build_xgb_cfg_for_rung(rung)

    # Validate coverage before any refit so the comparison can never mix tuned
    # countries with silently fixed countries.
    from loco_fusion_matrix_engine import prepare_bag_context
    from scripts.sensitivity_common import cv_cfg

    context = prepare_bag_context(
        model_df, BAG_TARGETS[bag], bag, analysis_cfg, cv_cfg(), feature_names
    )
    direct_fold_xgb_cfgs = None
    if checkpoint_source:
        if checkpoint_source["countries"] != list(context["countries"]):
            raise ValueError(
                "Active checkpoint countries do not match the requested comparison policy: "
                f"checkpoint={checkpoint_source['countries']}, comparison={list(context['countries'])}"
            )
        direct_fold_xgb_cfgs = {
            str(country): {**xgb_cfg, **checkpoint_source["params"]}
            for country in context["countries"]
        }
    else:
        resolve_tuned_fold_configs(tuning_artifact, bag, rung_id, xgb_cfg, context["countries"], strict=True)
    output_dir = _unique_run_dir(runtime_root, stable_hash(provenance))
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    candidates.to_parquet(output_dir / candidate_output_name, index=False)
    tuned_ensembles = None
    if ensemble_size > 1:
        tuned_ensembles = select_top_trial_ensemble_configs(
            pd.read_parquet(diagnostics_path),
            bag=bag,
            rung_id=rung_id,
            countries=context["countries"],
            ensemble_size=ensemble_size,
        )
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    fixed_global, fixed_country = evaluate_candidates_by_rung(
        model_df=model_df, candidate_df=candidates, exposome_cols=feature_names,
        bag=bag, rungs=[rung_id], analysis_cfg=analysis_cfg,
        outdir=output_dir / "fixed", n_jobs=n_jobs, tuning_artifact_path="", tuning_strict=False,
    )
    tuned_global, tuned_country = evaluate_candidates_by_rung(
        model_df=model_df, candidate_df=candidates, exposome_cols=feature_names,
        bag=bag, rungs=[rung_id], analysis_cfg=analysis_cfg,
        outdir=output_dir / ("tuned" if ensemble_size == 1 else f"tuned_top{ensemble_size}_ensemble"), n_jobs=n_jobs,
        tuning_artifact_path=tuning_artifact if ensemble_size == 1 else "", tuning_strict=True,
        fold_xgb_cfgs=direct_fold_xgb_cfgs,
        fold_xgb_ensembles=tuned_ensembles,
    )
    fixed_global = normalize_evaluator_global_rung(fixed_global)
    tuned_global = normalize_evaluator_global_rung(tuned_global)
    global_delta = paired_arm_delta(
        fixed_global, tuned_global, keys=["candidate_id", "rung_id"],
        metrics=["global_oof_r2", "global_rmse", "global_mae", "global_corr2"],
    )
    country_delta = paired_arm_delta(
        fixed_country, tuned_country, keys=["candidate_id", "rung_id", "fold_country"],
        metrics=["r2", "rmse", "mae", "corr2"],
    )
    global_delta.to_parquet(output_dir / "paired_global_delta.parquet", index=False)
    country_delta.to_parquet(output_dir / "paired_country_delta.parquet", index=False)
    summary = global_delta["delta_global_oof_r2_tuned_minus_fixed"].describe().to_frame("delta_global_oof_r2").T
    summary.to_csv(output_dir / "paired_summary.csv", index=False)
    print(f"Completed {STAGE_NAME}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
