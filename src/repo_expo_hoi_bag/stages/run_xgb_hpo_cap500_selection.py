#!/usr/bin/env python3
"""Freeze paper-facing global XGBoost HPO choices from source checkpoints."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import yaml

from xgb_nested_loco_tuning import ARTIFACT_SCHEMA_VERSION, select_trial_cap_or_practical_convergence


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is required")
    return value


def _safe_id(value: str, name: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"{name} may contain only letters, digits, '-' and '_'")
    return value


def _load_rule(checkout: Path) -> tuple[dict, Path]:
    path = checkout / "config" / "xgb_hpo_selection.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Invalid HPO selection config: {path}")
    rule = dict(data.get("cap_or_practical_convergence", {}))
    practical = dict(rule.get("practical_convergence", {}))
    return {
        "max_trials": int(rule["max_trials"]),
        "minimum_completed_trials": int(practical["minimum_completed_trials"]),
        "lookback_trials": int(practical["lookback_trials"]),
        "minimum_incumbent_improvement_mse": float(practical["minimum_incumbent_improvement_mse"]),
    }, path


def _load_policy(checkout: Path, variant: str) -> tuple[dict, Path]:
    path = checkout / "config" / "country_exclusions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = data.get("bag", {}).get("variants", {}) if isinstance(data, dict) else {}
    if data.get("schema_version") != 1 or variant not in variants:
        raise ValueError(f"Unknown BAG country-policy variant: {variant}")
    value = dict(variants[variant])
    return {
        "variant": variant,
        "bag_exclude_countries": list(value.get("exclude_countries", [])),
        "bag_exclude_diagnosis": list(value.get("exclude_diagnosis", [])),
        "config_path": str(path),
    }, path


def main() -> None:
    checkout = Path(_required_env("REPO_CHECKOUT_ROOT")).resolve()
    runtime = Path(_required_env("REPRO_DATA_ROOT")).resolve()
    source_signature = _safe_id(_required_env("XGB_HPO_SELECTION_SOURCE_RUN_SIGNATURE"), "XGB_HPO_SELECTION_SOURCE_RUN_SIGNATURE")
    output_id = _safe_id(_required_env("XGB_HPO_SELECTION_RUN_ID"), "XGB_HPO_SELECTION_RUN_ID")
    feature_scope = _required_env("XGB_HPO_SELECTION_FEATURE_SCOPE")
    variant = _required_env("XGB_HPO_SELECTION_VARIANT")
    rule, rule_path = _load_rule(checkout)
    policy, country_path = _load_policy(checkout, variant)
    source_root = runtime / "work" / "xgb_nested_loco_tuning" / source_signature
    output_root = checkout / "outputs" / "xgb_nested_loco_tuning" / output_id
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source HPO checkpoint root is missing: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite final HPO selection: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    selected, audit, checkpoint_hashes = [], [], {}
    for bag in ("structural", "functional"):
        for rung_id in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
            checkpoint = source_root / bag / "global" / f"{rung_id}.json"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Required global HPO checkpoint is missing: {checkpoint}")
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            payload = dict(data.get("signature_payload", {}))
            if (payload.get("bag"), payload.get("rung_id"), payload.get("outer_country")) != (bag, rung_id, "__global__"):
                raise ValueError(f"Invalid global HPO checkpoint identity: {checkpoint}")
            observed_scope = str(payload.get("feature_scope") or "full_exposome")
            if observed_scope != feature_scope:
                raise ValueError(f"Checkpoint feature scope does not match {feature_scope}: {checkpoint}")
            cfg = dict(payload.get("tuning_cfg", {}))
            if cfg.get("selection_objective") != "mean_country_mse" or cfg.get("optimization_direction") != "minimize":
                raise ValueError(f"Checkpoint does not minimize mean country MSE: {checkpoint}")
            best, decision = select_trial_cap_or_practical_convergence(data.get("trials", []), **rule)
            row = {
                "bag_target": bag, "outer_country": "__global__", "rung_id": rung_id,
                "params": dict(best["params"]), "selection_objective": str(best["selection_objective"]),
                "selection_objective_value": float(best["objective_value"]),
                "inner_global_oof_r2": float(best["inner_global_oof_r2"]),
                "inner_country_balanced_oof_r2": float(best["inner_country_balanced_oof_r2"]),
                "inner_country_r2_p25": float(best["inner_country_r2_p25"]),
                "inner_country_r2_median": float(best["inner_country_r2_median"]),
                "mean_country_mse": float(best["mean_country_mse"]),
                "median_country_mse": float(best["median_country_mse"]),
                "country_r2_json": str(best["country_r2_json"]), "country_mse_json": str(best["country_mse_json"]),
                # Older full-exposome checkpoints predate these optional
                # diagnostics. They are not part of the downstream resolver
                # contract and must not block a scientifically compatible freeze.
                "country_feature_r2_json": str(best.get("country_feature_r2_json", "")),
                "country_feature_mse_json": str(best.get("country_feature_mse_json", "")),
                "feature_scope": feature_scope, "is_manual_incumbent": bool(best["is_manual_incumbent"]),
                "selected_trial_index": int(best["trial_index"]), "best_iteration_median": best.get("best_iteration_median"),
                "n_trials": int(decision["observed_trials"]), "seed": cfg.get("seed"),
            }
            selected.append(row)
            key = f"{bag}/{rung_id}"
            checkpoint_hashes[key] = _sha256(checkpoint)
            audit.append({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hashes[key], **decision, **row})
            print(f"[selection] bag={bag} rung={rung_id} reason={decision['stop_reason']} trial={row['selected_trial_index'] + 1}/{decision['selection_trial_limit']} mean_country_mse={row['mean_country_mse']:.6f}", flush=True)
    provenance = {
        "artifact_kind": "paper_final_hpo_selection", "source_run_signature": source_signature,
        "source_checkpoint_root": str(source_root), "source_checkpoint_sha256": checkpoint_hashes,
        "feature_scope": feature_scope, "selection_rule": rule,
        "selection_rule_config": str(rule_path), "selection_rule_config_sha256": _sha256(rule_path),
        "two_phase_country_config": policy, "country_policy_sha256": _sha256(country_path),
        "output_root": str(output_root),
    }
    selected_path = output_root / "selected_xgb_configs.json"
    selected_path.write_text(json.dumps({"schema_version": ARTIFACT_SCHEMA_VERSION, "provenance": provenance, "selected": selected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "selection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps({"selected_sha256": _sha256(selected_path), **provenance}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "selection.log").write_text("\n".join(f"bag={row['bag_target']} rung={row['rung_id']} trial={row['selected_trial_index'] + 1} reason={row['stop_reason']} mean_country_mse={row['mean_country_mse']:.6f}" for row in audit) + "\n", encoding="utf-8")
    print(f"Completed final HPO selection: {output_root}", flush=True)


if __name__ == "__main__":
    main()
