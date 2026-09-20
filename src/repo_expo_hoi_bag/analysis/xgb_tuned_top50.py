"""Pure contracts for the paired fixed-versus-tuned XGBoost comparison."""
from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import yaml


def checkpoint_best_global_source(
    runtime_root: Path,
    run_signature: str,
    bag: str,
    rung_id: str,
    *,
    tunable_keys: set[str] | frozenset[str],
) -> dict:
    """Read the best observed global HPO trial without creating a snapshot artifact."""
    if not run_signature.replace("-", "").replace("_", "").isalnum():
        raise ValueError("XGB_TUNING_CHECKPOINT_RUN_SIGNATURE has invalid characters")
    root = Path(runtime_root).resolve() / "work" / "xgb_nested_loco_tuning"
    path = (root / run_signature / bag / "global" / f"{rung_id}.json").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Checkpoint path must be beneath {root}: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Active HPO checkpoint is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature_payload = dict(payload.get("signature_payload", {}))
    if signature_payload.get("bag") != bag or signature_payload.get("rung_id") != rung_id:
        raise ValueError(f"Checkpoint BAG/rung does not match requested comparison: {path}")
    if signature_payload.get("outer_country") != "__global__":
        raise ValueError(f"Checkpoint is not a global HPO unit: {path}")
    trials = [row for row in payload.get("trials", []) if isinstance(row, dict)]
    valid = [row for row in trials if pd.notna(row.get("objective_value"))]
    if not valid:
        raise ValueError(f"Checkpoint has no valid trials: {path}")
    direction = str(signature_payload.get("tuning_cfg", {}).get("optimization_direction", "minimize"))
    if direction != "minimize":
        raise ValueError(f"Checkpoint must minimize mean country MSE, got {direction!r}: {path}")
    best = min(valid, key=lambda row: (float(row["objective_value"]), int(row["trial_index"])))
    params = dict(best.get("params", {}))
    unexpected = sorted(set(params).difference(tunable_keys))
    if unexpected:
        raise ValueError(f"Checkpoint has non-tunable parameters: {unexpected}")
    if not params:
        raise ValueError(f"Checkpoint best trial has no parameters: {path}")
    return {
        "kind": "active_checkpoint_best_so_far",
        "run_signature": run_signature,
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "countries": list(signature_payload.get("countries", [])),
        "feature_scope": signature_payload.get("feature_scope"),
        "selection_objective": best.get("selection_objective"),
        "n_trials_observed": len(trials),
        "best_trial_index": int(best["trial_index"]),
        "best_objective_value": float(best["objective_value"]),
        "params": params,
    }


def resolve_comparison_top_n(default: int, environ: Mapping[str, str]) -> int:
    """Read an explicit bounded smoke override without changing the full default."""
    raw = environ.get("XGB_TOP50_COMPARISON_TOP_N", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"XGB_TOP50_COMPARISON_TOP_N must be a positive integer, got {raw!r}") from exc
    if value < 1 or value > default:
        raise ValueError(
            f"XGB_TOP50_COMPARISON_TOP_N must be between 1 and configured top_n ({default}), got {value}"
        )
    return value


def resolve_comparison_candidate_mode(environ: Mapping[str, str]) -> str:
    """Resolve the explicitly bounded candidate set used in a paired check."""
    mode = environ.get("XGB_TUNED_COMPARISON_CANDIDATE_MODE", "historical_top50").strip()
    if mode not in {"historical_top50", "full_exposome"}:
        raise ValueError(
            "XGB_TUNED_COMPARISON_CANDIDATE_MODE must be historical_top50 or full_exposome"
        )
    return mode


def resolve_comparison_ensemble_size(environ: Mapping[str, str]) -> int:
    """Resolve the number of already-scored trials to average per outer country."""
    raw = environ.get("XGB_TUNED_COMPARISON_ENSEMBLE_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "XGB_TUNED_COMPARISON_ENSEMBLE_SIZE must be a positive integer"
        ) from exc
    if value < 1 or value > 8:
        raise ValueError("XGB_TUNED_COMPARISON_ENSEMBLE_SIZE must be between 1 and 8")
    return value


def load_comparison_bag_policy(country_config_path: Path, variant: str) -> dict:
    """Load one explicit BAG policy from the versioned country contract."""
    with Path(country_config_path).open("r", encoding="utf-8") as handle:
        country_cfg = yaml.safe_load(handle)
    variants = country_cfg.get("bag", {}).get("variants", {}) if isinstance(country_cfg, dict) else {}
    if country_cfg.get("schema_version") != 1 or variant not in variants:
        raise ValueError(f"Unknown XGB comparison country-policy variant {variant!r}")
    selected = dict(variants[variant])
    return {
        "variant": variant,
        "exclude_countries": list(selected.get("exclude_countries", [])),
        "exclude_diagnosis": list(selected.get("exclude_diagnosis", [])),
        "config_path": str(country_config_path),
    }


def select_top_trial_ensemble_configs(
    diagnostics: pd.DataFrame,
    *,
    bag: str,
    rung_id: str,
    countries: Sequence[str],
    ensemble_size: int,
) -> dict[str, list[dict]]:
    """Return the top already-evaluated trial parameters for every outer country.

    This is deliberately a deterministic read of trial diagnostics: it performs
    neither further optimization nor any reference to the manually fixed arm.
    """
    bag_column = "bag_target" if "bag_target" in diagnostics.columns else "bag"
    required = {bag_column, "rung_id", "outer_country", "trial_index", "objective_value", "params_json"}
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise ValueError(f"Trial diagnostics are missing required columns: {missing}")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    scoped = diagnostics[
        diagnostics[bag_column].astype(str).eq(bag)
        & diagnostics["rung_id"].astype(str).eq(rung_id)
    ].copy()
    scoped["objective_value"] = pd.to_numeric(scoped["objective_value"], errors="coerce")
    scoped["trial_index"] = pd.to_numeric(scoped["trial_index"], errors="coerce")
    output: dict[str, list[dict]] = {}
    for country in countries:
        rows = scoped[scoped["outer_country"].astype(str).eq(str(country))].dropna(
            subset=["objective_value", "trial_index", "params_json"]
        )
        rows = rows.sort_values(["objective_value", "trial_index"], ascending=[False, True]).head(ensemble_size)
        if len(rows) != ensemble_size:
            raise ValueError(
                f"Trial diagnostics provide only {len(rows)} valid trials for {bag}/{rung_id}/{country}; "
                f"{ensemble_size} are required"
            )
        try:
            configs = [json.loads(value) for value in rows["params_json"]]
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Trial diagnostics have invalid params_json for {country}") from exc
        if not all(isinstance(config, dict) and config for config in configs):
            raise ValueError(f"Trial diagnostics have empty parameter configurations for {country}")
        output[str(country)] = configs
    return output


def full_exposome_candidate(exposome_cols: Sequence[str]) -> pd.DataFrame:
    """Return the one canonical candidate containing every exposome feature."""
    features = [str(feature) for feature in exposome_cols]
    if not features or len(set(features)) != len(features):
        raise ValueError("The full-exposome comparison requires unique, non-empty feature names")
    return pd.DataFrame([{
        "candidate_id": "__full_exposome__",
        "feature_id": "__full_exposome__",
        "objective": "canonical_full_exposome",
        "order": len(features),
        "rank": 1,
        "score": float("nan"),
        "nplet_vars": features,
        "predictors_identity": "|".join(features),
        "predictors_identity_n": len(features),
        "candidate_family": "canonical_full_exposome",
        "source_label": "canonical_full_exposome",
    }])


def select_historical_top_candidates(
    metrics: pd.DataFrame,
    *,
    bag: str,
    rung_id: str,
    top_n: int,
) -> pd.DataFrame:
    """Select a stable top-N candidate set from an immutable historical table."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    required = {"candidate_id", "rung_id", "full_r2", "predictors_identity"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Historical metrics are missing required columns: {missing}")
    subset = metrics[metrics["rung_id"].astype(str).eq(rung_id)].copy()
    if "bag_target" in subset.columns:
        subset = subset[subset["bag_target"].astype(str).eq(bag)].copy()
    subset = subset[subset["candidate_id"].astype(str).ne("__baseline__")].copy()
    subset["full_r2"] = pd.to_numeric(subset["full_r2"], errors="coerce")
    subset = subset.dropna(subset=["full_r2", "predictors_identity"])
    subset = subset.sort_values(["full_r2", "candidate_id"], ascending=[False, True])
    subset = subset.drop_duplicates("candidate_id", keep="first").head(top_n).reset_index(drop=True)
    if len(subset) != top_n:
        raise ValueError(
            f"Historical metrics provide only {len(subset)} valid {bag}/{rung_id} candidates; "
            f"{top_n} are required"
        )
    output = subset.copy()
    output["feature_id"] = output["candidate_id"].astype(str)
    output["nplet_vars"] = output["predictors_identity"].astype(str).str.split("|")
    output["predictors_identity_n"] = output["nplet_vars"].map(len)
    output["candidate_family"] = f"historical_{bag}_{rung_id}_top50"
    output["source_label"] = "historical_global_oof_r2"
    return output


def paired_arm_delta(
    fixed: pd.DataFrame,
    tuned: pd.DataFrame,
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Join fixed and tuned outcomes and calculate tuned-minus-fixed metrics."""
    for name, frame in (("fixed", fixed), ("tuned", tuned)):
        missing = sorted(set(keys).union(metrics).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} comparison table is missing columns: {missing}")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{name} comparison table has duplicate pairing keys: {list(keys)}")
    fixed_keys = set(map(tuple, fixed.loc[:, keys].astype(str).to_numpy()))
    tuned_keys = set(map(tuple, tuned.loc[:, keys].astype(str).to_numpy()))
    if fixed_keys != tuned_keys:
        raise ValueError(
            "Fixed and tuned comparison tables do not cover the same pairing keys: "
            f"fixed_only={len(fixed_keys - tuned_keys)} tuned_only={len(tuned_keys - fixed_keys)}"
        )
    fixed_values = fixed.loc[:, [*keys, *metrics]].rename(
        columns={metric: f"{metric}_fixed" for metric in metrics}
    )
    tuned_values = tuned.loc[:, [*keys, *metrics]].rename(
        columns={metric: f"{metric}_tuned" for metric in metrics}
    )
    output = fixed_values.merge(tuned_values, on=list(keys), how="inner", validate="one_to_one")
    for metric in metrics:
        output[f"delta_{metric}_tuned_minus_fixed"] = (
            pd.to_numeric(output[f"{metric}_tuned"], errors="coerce")
            - pd.to_numeric(output[f"{metric}_fixed"], errors="coerce")
        )
    return output


def normalize_evaluator_global_rung(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the evaluator's metadata/fit merge to one rung-id column."""
    if "rung_id" in frame.columns:
        return frame.copy()
    if "rung_id_y" not in frame.columns:
        raise ValueError("Evaluator global table has no rung_id or rung_id_y column")
    output = frame.copy()
    if "rung_id_x" in output.columns:
        disagree = (
            output["rung_id_x"].notna()
            & output["rung_id_y"].notna()
            & output["rung_id_x"].astype(str).ne(output["rung_id_y"].astype(str))
        )
        if disagree.any():
            raise ValueError("Evaluator global table has inconsistent metadata and fitted rung identifiers")
    return output.rename(columns={"rung_id_y": "rung_id"})
