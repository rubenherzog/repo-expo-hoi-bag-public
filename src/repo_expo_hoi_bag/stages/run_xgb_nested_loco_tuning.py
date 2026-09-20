#!/usr/bin/env python3
"""Run nested-LOCO XGBoost tuning with local lightweight delivery artifacts."""
from __future__ import annotations

import os
from pathlib import Path
from hashlib import sha256

from joblib import Parallel, delayed
import pandas as pd
import yaml

from loco_fusion_matrix_engine import (
    align_exposome_to_greedy_features,
    build_model_df,
    load_greedy_feature_names,
    pin_blas_threads,
)
from oinfo_bag_ladder.config import (
    ANALYSIS_CFG, BAG_TARGETS, CV_CFG, DATA_PATHS, EARLY_STOP_CFG, XGB_TUNING_CFG,
)
from oinfo_bag_ladder.io_utils import stable_hash
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
from xgb_nested_loco_tuning import (
    FITTING_MODES,
    FEATURE_SCOPES,
    OPTIMIZERS,
    SELECTION_OBJECTIVES,
    append_progress_line,
    build_global_loco_fold_designs,
    build_nested_fold_designs,
    create_tuning_run_paths,
    deterministic_seed,
    tune_unit,
    write_tuning_artifact,
)
from loco_fusion_matrix_engine import prepare_bag_context


def _runtime_root() -> Path:
    root = os.environ.get("REPRO_DATA_ROOT", "").strip()
    if not root:
        raise EnvironmentError("REPRO_DATA_ROOT is required for XGBoost nested tuning")
    return Path(root)


def _checkout_root() -> Path:
    root = os.environ.get("REPO_CHECKOUT_ROOT", "").strip()
    if not root:
        raise EnvironmentError("REPO_CHECKOUT_ROOT is required to route local XGB tuning artifacts")
    checkout = Path(root).resolve()
    if not (checkout / "outputs").is_dir():
        raise FileNotFoundError(f"Checkout outputs directory is missing: {checkout / 'outputs'}")
    return checkout


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_domain_balanced_k10_panel(checkout_root: Path, feature_names: list[str]) -> tuple[list[dict], dict]:
    """Load and validate the fixed, domain-balanced intermediate HPO panel."""
    path = checkout_root / "config" / "xgb_hpo_feature_panels.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Invalid XGB HPO feature-panel configuration: {path}")
    panel = dict(dict(payload.get("panels", {})).get("domain_balanced_k10", {}))
    subsets = panel.get("subsets")
    if int(panel.get("subset_size", -1)) != 10 or int(panel.get("n_subsets", -1)) != 10:
        raise ValueError("domain_balanced_k10 must declare ten subsets of size ten")
    if not isinstance(subsets, list) or len(subsets) != 10:
        raise ValueError("domain_balanced_k10 must contain exactly ten subsets")
    known = set(feature_names)
    feature_domains = pd.read_csv(checkout_root / "data" / "metadata" / "exposome_feature_domains.csv")
    domain_by_feature = dict(zip(feature_domains["feature_name"].astype(str), feature_domains["domain"].astype(str)))
    expected_domains = set(domain_by_feature.values())
    normalized: list[dict] = []
    seen_ids: set[str] = set()
    seen_subsets: set[tuple[str, ...]] = set()
    for item in subsets:
        if not isinstance(item, dict):
            raise ValueError("domain_balanced_k10 subsets must be mappings")
        identifier = str(item.get("id", "")).strip()
        features = [str(value) for value in item.get("features", [])]
        if not identifier or identifier in seen_ids:
            raise ValueError("domain_balanced_k10 subset ids must be non-empty and unique")
        if len(features) != 10 or len(set(features)) != 10:
            raise ValueError(f"domain_balanced_k10 subset {identifier} must contain ten distinct exposures")
        unknown = sorted(set(features).difference(known))
        if unknown:
            raise ValueError(f"domain_balanced_k10 subset {identifier} has unknown exposures: {unknown}")
        domains = [domain_by_feature[feature] for feature in features]
        if set(domains) != expected_domains or len(set(domains)) != 10:
            raise ValueError(f"domain_balanced_k10 subset {identifier} must contain one exposure from every domain")
        subset_key = tuple(sorted(features))
        if subset_key in seen_subsets:
            raise ValueError(f"domain_balanced_k10 repeats the full subset {identifier}")
        seen_ids.add(identifier)
        seen_subsets.add(subset_key)
        normalized.append({"id": identifier, "features": features})
    panel["subsets"] = normalized
    return normalized, {"path": str(path), "sha256": _file_sha256(path), "definition": panel}


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "").strip()
    return [value.strip() for value in raw.split(",") if value.strip()] if raw else list(default)


def _apply_runtime_overrides(base_cfg: dict, environ: dict[str, str] | None = None) -> dict:
    """Apply explicit operational overrides without changing scientific defaults."""
    cfg = dict(base_cfg)
    env = os.environ if environ is None else environ
    for key, cfg_key in (
        ("XGB_TUNING_N_JOBS", "n_jobs"),
        ("XGB_TUNING_N_TRIALS", "n_trials"),
        ("XGB_TUNING_N_INITIAL_TRIALS", "n_initial_trials"),
    ):
        raw = env.get(key, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be a positive integer, got {raw!r}") from exc
        if value < 1:
            raise ValueError(f"{key} must be a positive integer, got {raw!r}")
        cfg[cfg_key] = value
    objective = env.get("XGB_TUNING_SELECTION_OBJECTIVE", "").strip()
    if objective:
        if objective not in SELECTION_OBJECTIVES:
            raise ValueError(
                "XGB_TUNING_SELECTION_OBJECTIVE must be one of "
                f"{sorted(SELECTION_OBJECTIVES)}, got {objective!r}"
            )
        cfg["selection_objective"] = objective
    fitting_mode = env.get("XGB_TUNING_FITTING_MODE", "").strip()
    if fitting_mode:
        if fitting_mode not in FITTING_MODES:
            raise ValueError(
                f"XGB_TUNING_FITTING_MODE must be one of {sorted(FITTING_MODES)}, got {fitting_mode!r}"
            )
        cfg["fitting_mode"] = fitting_mode
    optimizer = env.get("XGB_TUNING_OPTIMIZER", "").strip()
    if optimizer:
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"XGB_TUNING_OPTIMIZER must be one of {sorted(OPTIMIZERS)}, got {optimizer!r}")
        cfg["optimizer"] = optimizer
    feature_scope = env.get("XGB_TUNING_FEATURE_SCOPE", "").strip()
    if feature_scope:
        if feature_scope not in FEATURE_SCOPES:
            raise ValueError(
                f"XGB_TUNING_FEATURE_SCOPE must be one of {sorted(FEATURE_SCOPES)}, got {feature_scope!r}"
            )
        cfg["feature_scope"] = feature_scope
    single_parallel_axis = env.get("XGB_TUNING_SINGLE_PARALLEL_AXIS", "").strip()
    if single_parallel_axis:
        if single_parallel_axis not in {"country", "country_exposure"}:
            raise ValueError("XGB_TUNING_SINGLE_PARALLEL_AXIS must be country or country_exposure")
        cfg["single_parallel_axis"] = single_parallel_axis
    parent_run_signature = env.get("XGB_TUNING_PARENT_RUN_SIGNATURE", "").strip()
    if parent_run_signature:
        if not parent_run_signature.replace("-", "").replace("_", "").isalnum():
            raise ValueError("XGB_TUNING_PARENT_RUN_SIGNATURE may contain only letters, digits, '-' and '_'")
        cfg["parent_run_signature"] = parent_run_signature
    if cfg.get("fitting_mode") == "global_loco_country_mse":
        # This alternative procedure has a fixed scientific objective: minimize
        # the equally weighted mean of held-out-country MSE values.
        cfg["selection_objective"] = "mean_country_mse"
        cfg["optimization_direction"] = "minimize"
        cfg["include_manual_incumbent"] = True
    elif parent_run_signature:
        raise ValueError("XGB_TUNING_PARENT_RUN_SIGNATURE is supported only for global_loco_country_mse")
    manual_incumbent = env.get("XGB_TUNING_INCLUDE_MANUAL_INCUMBENT", "").strip().lower()
    if manual_incumbent:
        if manual_incumbent not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError("XGB_TUNING_INCLUDE_MANUAL_INCUMBENT must be a boolean value")
        cfg["include_manual_incumbent"] = manual_incumbent in {"1", "true", "yes"}
    if int(cfg["n_trials"]) < int(cfg["n_initial_trials"]):
        raise ValueError(
            "XGB_TUNING_N_TRIALS must be at least n_initial_trials "
            f"({cfg['n_initial_trials']})"
        )
    return cfg


def _resolve_global_parallelism(cfg: dict, n_units: int, environ: dict[str, str] | None = None) -> None:
    """Allocate the fixed CPU budget across concurrent global tuning units.

    The unit plan is deterministic and spends the entire requested budget when
    there are at least as many country folds as allocated workers.  Individual
    BAG × rung units receive either ``floor`` or ``ceil`` of the budget, so a
    40-core six-unit run gets 7/7/7/7/6/6 cores rather than silently idling 4.
    """
    env = os.environ if environ is None else environ
    active_units = max(1, min(int(cfg["n_jobs"]), int(n_units)))
    total_cores = int(cfg["n_jobs"])
    base_cores, remainder = divmod(total_cores, active_units)
    per_unit_cores = max(1, base_cores)
    raw_threads = env.get("XGB_TUNING_XGB_NTHREAD", "").strip()
    raw_folds = env.get("XGB_TUNING_FOLD_N_JOBS", "").strip()
    if raw_threads or raw_folds:
        xgb_nthread = int(raw_threads) if raw_threads else 1
        fold_n_jobs = int(raw_folds) if raw_folds else max(1, per_unit_cores // xgb_nthread)
        if xgb_nthread < 1 or fold_n_jobs < 1 or fold_n_jobs * xgb_nthread > per_unit_cores:
            raise ValueError("Global XGB fold parallelism exceeds the allocated cores per BAG × rung unit")
        unit_parallelism = [
            {"core_budget": per_unit_cores, "fold_n_jobs": fold_n_jobs, "xgb_nthread": xgb_nthread}
            for _ in range(active_units)
        ]
    else:
        unit_parallelism = []
        for index in range(active_units):
            core_budget = base_cores + (1 if index < remainder else 0)
            xgb_nthread = 1 if cfg.get("feature_scope") in {"single_exposure", "domain_balanced_k10"} else (2 if core_budget >= 8 else 1)
            fold_n_jobs = max(1, core_budget // xgb_nthread)
            unit_parallelism.append(
                {"core_budget": core_budget, "fold_n_jobs": fold_n_jobs, "xgb_nthread": xgb_nthread}
            )
    if any(item["fold_n_jobs"] * item["xgb_nthread"] > item["core_budget"] for item in unit_parallelism):
        raise ValueError("Global XGB fold parallelism exceeds the allocated cores per BAG × rung unit")
    cfg["global_unit_parallelism"] = unit_parallelism
    # These fields preserve a concise top-level summary for provenance and
    # legacy consumers; each unit receives its exact values below.
    cfg["fold_n_jobs"] = unit_parallelism[0]["fold_n_jobs"]
    cfg["xgb_nthread"] = unit_parallelism[0]["xgb_nthread"]
    cfg["parallel_core_budget"] = total_cores
    cfg["parallel_active_units"] = active_units


def _canonical_two_phase_population_config() -> tuple[dict, dict]:
    """Load the source-of-truth greedy and BAG population rules.

    ``country_exclusions.yaml`` intentionally has two independent country exclusions:
    greedy discovery uses ``greedy.countries_to_remove`` while BAG prediction
    uses the main-analysis variant.  Tuning is a BAG prediction stage, so only
    the latter filters model rows; both are fingerprinted in its artifact.
    """
    config_path = os.environ.get("XGB_TUNING_COUNTRY_CONFIG", "").strip()
    if config_path:
        country_config_path = Path(config_path)
    else:
        runtime_candidate = Path.cwd() / "config" / "country_exclusions.yaml"
        source_candidate = Path(__file__).resolve().parents[3] / "config" / "country_exclusions.yaml"
        country_config_path = runtime_candidate if runtime_candidate.is_file() else source_candidate
    with country_config_path.open("r", encoding="utf-8") as handle:
        country_cfg = yaml.safe_load(handle)
    if not isinstance(country_cfg, dict) or country_cfg.get("schema_version") != 1:
        raise ValueError(f"Invalid country-exclusion configuration: {country_config_path}")
    greedy_cfg = dict(country_cfg.get("greedy", {}))
    variants = dict(country_cfg.get("bag", {}).get("variants", {}))
    variant_key = os.environ.get("XGB_TUNING_VARIANT", "a").strip() or "a"
    if variant_key not in variants:
        raise ValueError(
            f"XGB_TUNING_VARIANT={variant_key!r} is not defined in {country_config_path}"
        )
    variant_cfg = dict(variants[variant_key])
    analysis_cfg = dict(ANALYSIS_CFG)
    analysis_cfg["exclude_countries"] = _env_list(
        "V3_EXCLUDE_COUNTRIES", list(variant_cfg.get("exclude_countries", []))
    )
    analysis_cfg["exclude_diagnosis"] = _env_list(
        "V3_EXCLUDE_DIAGNOSIS", list(variant_cfg.get("exclude_diagnosis", []))
    )
    two_phase_cfg = {
        "variant": variant_key,
        "config_path": str(country_config_path),
        "greedy_countries_to_remove": list(greedy_cfg.get("countries_to_remove", [])),
        "bag_exclude_countries": list(analysis_cfg["exclude_countries"]),
        "bag_exclude_diagnosis": list(analysis_cfg["exclude_diagnosis"]),
    }
    return analysis_cfg, two_phase_cfg


def main() -> None:
    cfg = _apply_runtime_overrides(XGB_TUNING_CFG)
    analysis_cfg, two_phase_cfg = _canonical_two_phase_population_config()
    requested_bags = [value.strip() for value in os.environ.get("XGB_TUNING_BAGS", "").split(",") if value.strip()]
    if requested_bags:
        unknown_bags = sorted(set(requested_bags).difference(BAG_TARGETS))
        if unknown_bags:
            raise ValueError(f"Unknown XGB_TUNING_BAGS values: {unknown_bags}")
        cfg["bags"] = requested_bags
    requested_rungs = [value.strip() for value in os.environ.get("XGB_TUNING_RUNGS", "").split(",") if value.strip()]
    smoke = os.environ.get("XGB_TUNING_SMOKE", "").strip().lower() in {"1", "true", "yes"}
    if smoke:
        cfg.update({"n_trials": 4, "n_initial_trials": 3, "n_jobs": 1})
    if cfg["fitting_mode"] == "global_loco_country_mse":
        _resolve_global_parallelism(
            cfg, len(cfg["bags"]) * (1 if smoke else (len(requested_rungs) or 3))
        )
    # Tuning does not depend on O-information candidates or greedy outputs. Its
    # fixed, canonical feature panel is selected explicitly by feature_scope.
    raw = pd.read_csv(DATA_PATHS["input_csv"], low_memory=False)
    raw["N_MEGA"] = raw["N_MEGA"].astype(str).str.strip()
    raw = raw.drop_duplicates(subset="N_MEGA", keep="first").reset_index(drop=True)
    feature_names = load_greedy_feature_names(Path("data/exposome_feature_names.csv"))
    exposome = align_exposome_to_greedy_features(raw, feature_names).apply(pd.to_numeric, errors="coerce")
    if int(exposome.isna().sum().sum()) != 0:
        raise ValueError("The complete-exposome tuning matrix contains missing values")
    model_df = build_model_df(raw, exposome, analysis_cfg)
    feature_names = exposome.columns.astype(str).tolist()
    runtime_root = _runtime_root()
    checkout_root = _checkout_root()
    feature_panel_provenance = None
    if cfg.get("feature_scope") == "domain_balanced_k10":
        panel_sets, feature_panel_provenance = _load_domain_balanced_k10_panel(checkout_root, feature_names)
        cfg["feature_panel"] = feature_panel_provenance["definition"]
    parent_run_signature = str(cfg.get("parent_run_signature", ""))
    parent_work_root = None
    continuation_parent = None
    if parent_run_signature:
        parent_work_root = runtime_root / "work" / "xgb_nested_loco_tuning" / parent_run_signature
        parent_manifest = checkout_root / "outputs" / "xgb_nested_loco_tuning" / parent_run_signature / "manifest.json"
        if not parent_work_root.is_dir() or not parent_manifest.is_file():
            raise FileNotFoundError(
                "Continuation requires the complete parent checkpoint and manifest: "
                f"{parent_run_signature}"
            )
        continuation_parent = {
            "run_signature": parent_run_signature,
            "checkpoint_root": str(parent_work_root),
            "manifest_sha256": _file_sha256(parent_manifest),
        }
    provenance = {
        "method": str(cfg.get("optimizer", "custom_gp")),
        "tuning_cfg": cfg,
        "early_stop_cfg": EARLY_STOP_CFG,
        "exposome_features": feature_names,
        "feature_scope": str(cfg.get("feature_scope", "full_exposome")),
        "feature_panel": feature_panel_provenance,
        "two_phase_country_config": two_phase_cfg,
        "smoke": smoke,
        "input_sha256": _file_sha256(Path(DATA_PATHS["input_csv"])),
        "feature_names_sha256": _file_sha256(Path("data/exposome_feature_names.csv")),
        "country_policy_sha256": _file_sha256(Path(two_phase_cfg["config_path"])),
        "stage_sha256": _file_sha256(Path(__file__)),
        "continuation_parent": continuation_parent,
    }
    analysis_signature = stable_hash(provenance)
    run_signature, output_root, work_root = create_tuning_run_paths(
        checkout_root,
        runtime_root,
        analysis_signature,
        run_id=os.environ.get("XGB_TUNING_RUN_ID", "").strip() or None,
    )
    log_path = output_root / "tuning.log"
    provenance.update({
        "analysis_signature": analysis_signature,
        "run_signature": run_signature,
        "checkpoint_root": str(work_root),
        "output_root": str(output_root),
    })
    pin_blas_threads(int(cfg.get("xgb_nthread", 1)))
    contexts = {
        bag: dict(
            prepare_bag_context(model_df, BAG_TARGETS[bag], bag, analysis_cfg, dict(CV_CFG), feature_names),
            exposome_features=feature_names,
            hpo_feature_sets=panel_sets if feature_panel_provenance is not None else None,
        )
        for bag in cfg["bags"]
    }
    xgb_rungs = [spec for spec in get_rung_specs() if spec["model_family"] == "xgb"]
    if requested_rungs:
        available_rungs = {spec["rung_id"] for spec in xgb_rungs}
        unknown_rungs = sorted(set(requested_rungs).difference(available_rungs))
        if unknown_rungs:
            raise ValueError(f"Unknown XGB_TUNING_RUNGS values: {unknown_rungs}")
        xgb_rungs = [spec for spec in xgb_rungs if spec["rung_id"] in requested_rungs]
    if smoke:
        bag = cfg["bags"][0]
        contexts = {bag: contexts[bag]}
        contexts[bag] = dict(contexts[bag], countries=contexts[bag]["countries"][:2])
        xgb_rungs = xgb_rungs[:1]

    def run_outer(bag: str, outer_country: str):
        selected, diagnostics = [], []
        append_progress_line(
            log_path,
            f"[tuning] start bag={bag} outer={outer_country} rungs={[spec['rung_id'] for spec in xgb_rungs]}",
        )
        nested_designs = build_nested_fold_designs(
            contexts[bag], outer_country, analysis_cfg, dict(EARLY_STOP_CFG),
            deterministic_seed(cfg["seed"], bag, outer_country),
        )
        for rung in xgb_rungs:
            selected_row, trial_rows = tune_unit(
                contexts[bag], outer_country, rung["rung_id"], build_xgb_cfg_for_rung(rung),
                analysis_cfg, dict(EARLY_STOP_CFG), cfg,
                work_root / bag / str(outer_country) / f"{rung['rung_id']}.json",
                nested_designs,
                log_path,
            )
            selected.append(selected_row)
            diagnostics.extend(trial_rows)
            append_progress_line(
                log_path,
                f"[tuning] selected bag={bag} outer={outer_country} rung={rung['rung_id']} "
                f"selection_objective={selected_row['selection_objective']} "
                f"selection_value={selected_row['selection_objective_value']:.6f} "
                f"inner_global_r2={selected_row['inner_global_oof_r2']:.6f} "
                f"inner_country_balanced_r2={selected_row['inner_country_balanced_oof_r2']:.6f} "
                f"trial={selected_row['selected_trial_index'] + 1}/{selected_row['n_trials']} "
                f"best_iteration_median={selected_row['best_iteration_median']} "
                f"params={selected_row['params']}",
            )
        return selected, diagnostics

    def run_global(bag: str, rung: dict, unit_parallelism: dict):
        unit_cfg = dict(cfg)
        unit_cfg.update(unit_parallelism)
        scope = "__global__"
        append_progress_line(
            log_path,
            f"[tuning] start bag={bag} scope=global_loco rung={rung['rung_id']} "
            f"countries={contexts[bag]['countries']} core_budget={unit_cfg['core_budget']} "
            f"fold_workers={unit_cfg['fold_n_jobs']} xgb_nthread={unit_cfg['xgb_nthread']} "
            f"feature_scope={unit_cfg.get('feature_scope', 'full_exposome')}",
        )
        designs = build_global_loco_fold_designs(
            contexts[bag], analysis_cfg, dict(EARLY_STOP_CFG), deterministic_seed(cfg["seed"], bag, scope),
        )
        selected_row, trial_rows = tune_unit(
            contexts[bag], scope, rung["rung_id"], build_xgb_cfg_for_rung(rung),
            analysis_cfg, dict(EARLY_STOP_CFG), unit_cfg,
            work_root / bag / "global" / f"{rung['rung_id']}.json", designs, log_path,
            parent_checkpoint_path=(
                parent_work_root / bag / "global" / f"{rung['rung_id']}.json"
                if parent_work_root is not None else None
            ),
        )
        append_progress_line(
            log_path,
            f"[tuning] selected bag={bag} scope=global_loco rung={rung['rung_id']} "
            f"selection_objective=mean_country_mse selection_value={selected_row['selection_objective_value']:.6f} "
            f"mean_country_mse={selected_row['mean_country_mse']:.6f} params={selected_row['params']}",
        )
        return [selected_row], trial_rows

    global_mode = cfg["fitting_mode"] == "global_loco_country_mse"
    units = (
        [(bag, rung) for bag in contexts for rung in xgb_rungs]
        if global_mode else [(bag, country) for bag, context in contexts.items() for country in context["countries"]]
    )
    start_message = (
        f"Nested-LOCO tuning start: run_signature={run_signature} bags={cfg['bags']} "
        f"rungs={[r['rung_id'] for r in xgb_rungs]} fitting_mode={cfg['fitting_mode']} units={len(units)} "
        f"feature_scope={cfg.get('feature_scope', 'full_exposome')} "
        f"trials={cfg['n_trials']} workers={min(int(cfg['n_jobs']), len(units))} "
        f"parallel_plan={cfg.get('global_unit_parallelism', []) if global_mode else []} "
        f"continuation_parent={parent_run_signature or '<none>'}"
    )
    print(start_message, flush=True)
    append_progress_line(log_path, start_message)
    if global_mode:
        results = Parallel(n_jobs=min(int(cfg["n_jobs"]), len(units)), backend="loky", verbose=10)(
            delayed(run_global)(bag, rung, cfg["global_unit_parallelism"][index])
            for index, (bag, rung) in enumerate(units)
        )
    else:
        results = Parallel(n_jobs=min(int(cfg["n_jobs"]), len(units)), backend="loky", verbose=10)(
            delayed(run_outer)(bag, country) for bag, country in units
        )
    selected = [row for rows, _diagnostics in results for row in rows]
    diagnostics = [row for _selected, rows in results for row in rows]
    artifact = write_tuning_artifact(output_root, selected, diagnostics, provenance)
    complete_message = f"Nested-LOCO XGB tuning complete: {artifact}"
    print(complete_message, flush=True)
    append_progress_line(log_path, complete_message)


if __name__ == "__main__":
    main()
