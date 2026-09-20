#!/usr/bin/env python3
"""Equal-diagnosis-weight sensitivity for the configured paper models.

Within every LOCO fold, HC (CN), AD, and FTLD (FTD) receive equal total weight
in the inner training sample. The early-stopping validation sample is weighted
the same way over the diagnoses present there. Held-out-country evaluation is
never weighted.

The sensitivity deliberately keeps the deployed model definitions fixed by the
shared ``paper_analysis`` configuration. Both the full exposome model and its
covariate-only baseline are refit with the same weights.

Lightweight summaries -> outputs/sensitivity/dedup/diagnosis_balance/
Heavy OOF predictions  -> $REPRO_DATA_ROOT/work/sensitivity_eval/diagnosis_balance/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline_utils import default_xgb_cfg, load_stage_config
from scripts.sensitivity_common import (
    CONFIG_PATH,
    analysis_cfg_from_config,
    build_original_model_df,
    bundle_sensitivity_root,
    copy_tree_contents,
    cv_cfg,
    early_stop_cfg,
    load_raw_and_domains,
    load_sensitivity_config,
    paper_analysis_config,
    repo_sensitivity_root,
    route_main_k10_hpo,
    sensitivity_eval_work_root,
    selected_bags,
)
from loco_fusion_matrix_engine import (
    prepare_bag_context,
    regression_metrics_extended,
    target_map,
)
from xgb_loco_engine import (
    _append_predictors,
    _build_base_fold_mats,
    _fit_xgb_fold,
    require_xgboost,
)
from oinfo_bag_ladder.rungs import get_rung_specs
from xgb_nested_loco_tuning import resolve_tuned_fold_configs


def _canonical_metrics_path(root: Path, bag: str) -> Path:
    adapter = os.environ.get("MAIN_K10_CANONICAL_ROOT", "").strip()
    if adapter:
        return Path(adapter) / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    return (
        root
        / "runs"
        / "oinfo_only"
        / "variant_a"
        / "families"
        / "pooled_oinfo_ladder"
        / "canonical"
        / "per_experiment"
        / f"pooled_oinfo_ladder_{bag}"
        / "metrics_global_long.parquet"
    )


def _reference_oof_path(
    root: Path,
    bag: str,
    objective: str,
    file_suffix: str,
    reference_oof_dir_template: str,
    order_max: int,
) -> Path:
    main_oof = os.environ.get("MAIN_K10_SELECTED_OOF_ROOT", "").strip()
    if main_oof:
        return Path(main_oof) / f"level_best_{file_suffix}" / bag / "oof_xgb_tree_d3.parquet"
    reference_dir = reference_oof_dir_template.format(order_max=order_max)
    return (
        root
        / reference_dir
        / bag
        / f"oof_predictions_best_{file_suffix}.parquet"
    )


def _best_fixed_spec(
    root: Path,
    bag: str,
    objective: str,
    *,
    rung_id: str,
    order_max: int,
    file_suffix: str,
    reference_oof_dir_template: str,
) -> dict[str, object]:
    metrics_path = _canonical_metrics_path(root, bag)
    metrics = pd.read_parquet(metrics_path)
    pool = metrics[
        (metrics["objective"].astype(str) == objective)
        & (metrics["rung_id"].astype(str) == rung_id)
        & (pd.to_numeric(metrics["order"], errors="coerce") <= order_max)
    ].copy()
    if pool.empty:
        raise ValueError(
            f"No {bag} {objective} candidate at {rung_id} with order <= {order_max}"
        )
    best = pool.nlargest(1, "full_r2").iloc[0]
    predictors = [x for x in str(best["predictors_identity"]).split("|") if x]

    reference_path = _reference_oof_path(
        root,
        bag,
        objective,
        file_suffix,
        reference_oof_dir_template,
        order_max,
    )
    reference = pd.read_parquet(
        reference_path,
        columns=["candidate_id", "rung_id", "objective"],
    )
    reference_spec = reference.iloc[0]
    expected = (str(best["candidate_id"]), rung_id, objective)
    observed = (
        str(reference_spec["candidate_id"]),
        str(reference_spec["rung_id"]),
        str(reference_spec["objective"]),
    )
    if observed != expected:
        raise ValueError(
            "Fixed candidate does not match the configured paper OOF reference: "
            f"expected={expected}, observed={observed}, source={reference_path}"
        )
    return {
        "bag": bag,
        "objective": objective,
        "candidate_id": str(best["candidate_id"]),
        "rung_id": rung_id,
        "order_max": order_max,
        "order": int(best["order"]),
        "predictors": predictors,
        "canonical_full_r2": float(best["full_r2"]),
        "metrics_source": str(metrics_path),
        "unweighted_oof_source": str(reference_path),
    }


def _equal_diagnosis_weights(
    diagnosis: np.ndarray,
    indices: np.ndarray,
    *,
    diagnoses: tuple[str, ...],
    require_all: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = diagnosis[indices].astype(str)
    counts = {dx: int(np.sum(labels == dx)) for dx in diagnoses}
    present = [dx for dx, n in counts.items() if n > 0]
    if require_all and present != list(diagnoses):
        raise ValueError(f"Training split lacks a primary diagnosis: {counts}")
    if not present:
        raise ValueError("Cannot compute diagnosis weights for an empty split")

    n_rows = len(labels)
    n_groups = len(present)
    weight_by_dx = {dx: n_rows / (n_groups * counts[dx]) for dx in present}
    weights = np.asarray([weight_by_dx[str(dx)] for dx in labels], dtype=float)
    return weights, counts


def _fit_balanced_oof(
    *,
    model_df: pd.DataFrame,
    feature_names: list[str],
    bag: str,
    spec: dict[str, object],
    analysis_cfg: dict,
    diagnoses: tuple[str, ...],
    rung_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_col = target_map(bag)[bag]
    context = prepare_bag_context(
        model_df,
        y_col,
        bag,
        analysis_cfg,
        cv_cfg(),
        feature_names,
    )
    diagnosis = np.asarray(context["diag"]).astype(str)
    unexpected = sorted(set(diagnosis).difference(diagnoses))
    if unexpected:
        raise ValueError(f"Diagnosis-balanced cohort contains unexpected groups: {unexpected}")

    stage_cfg = load_stage_config("oof_predictions", smoke=False)
    rung_specs = {
        str(rung_spec["rung_id"]): rung_spec for rung_spec in get_rung_specs()
    }
    if rung_id not in rung_specs:
        raise ValueError(f"Configured paper rung is undefined: {rung_id}")
    xgb_cfg = {**default_xgb_cfg(), **stage_cfg.get("xgb_cfg", {})}
    xgb_cfg.update(rung_specs[rung_id].get("xgb_overrides", {}))
    tuning_path = os.environ.get("XGB_TUNING_ARTIFACT", "").strip()
    main_k10_mode = bool(os.environ.get("MAIN_K10_MODE", "").strip())
    if main_k10_mode and not tuning_path:
        raise ValueError("MAIN_K10_MODE requires XGB_TUNING_ARTIFACT")
    routing = pd.DataFrame(
        [
            {"candidate_id": str(spec["candidate_id"]), "objective": str(spec["objective"])},
            {"candidate_id": "__baseline__", "objective": "baseline"},
        ]
    )
    if main_k10_mode:
        routing, hpo_artifacts = route_main_k10_hpo(routing, tuning_path)
    else:
        routing["hpo_scope"] = "configured"
        routing["hpo_artifact_sha256"] = ""
        hpo_artifacts = {"configured": Path(tuning_path)} if tuning_path else {}
    fold_cfgs_by_scope = {
        scope: resolve_tuned_fold_configs(
            path,
            bag,
            rung_id,
            xgb_cfg,
            context["countries"],
            strict=main_k10_mode,
        )
        for scope, path in hpo_artifacts.items()
    }
    full_scope = str(routing.iloc[0]["hpo_scope"])
    baseline_scope = str(routing.iloc[1]["hpo_scope"])
    full_hpo_hash = str(routing.iloc[0]["hpo_artifact_sha256"])
    baseline_hpo_hash = str(routing.iloc[1]["hpo_artifact_sha256"])
    base_seed = int(xgb_cfg["random_state"])
    fold_designs = {
        country: _build_base_fold_mats(
            context,
            country,
            analysis_cfg,
            early_stop_cfg(),
            seed=base_seed + fold_i,
        )
        for fold_i, country in enumerate(context["countries"])
    }

    predictor_indices = [feature_names.index(str(name)) for name in spec["predictors"]]
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)
    y_pred_full = np.full(len(y), np.nan, dtype=float)
    y_pred_base = np.full(len(y), np.nan, dtype=float)
    weight_rows: list[dict[str, object]] = []
    xgb = require_xgboost()

    for fold_i, country in enumerate(context["countries"]):
        fold = fold_designs[country]
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        tr_inner = np.asarray(fold["tr_inner_idx"], dtype=int)
        val_idx = np.asarray(fold["val_idx"], dtype=int)
        test_idx = np.asarray(fold["test_idx"], dtype=int)

        train_weight, train_counts = _equal_diagnosis_weights(
            diagnosis, tr_inner, diagnoses=diagnoses, require_all=True
        )
        val_weight, val_counts = _equal_diagnosis_weights(
            diagnosis, val_idx, diagnoses=diagnoses, require_all=False
        )

        used = list(predictor_indices)
        if used:
            variance = np.nanvar(x_exp[np.ix_(train_idx, used)], axis=0)
            used = [idx for idx, keep in zip(used, variance > 0) if bool(keep)]

        x_train_full = _append_predictors(
            fold["Xb_train_inner"], x_exp, tr_inner, used
        )
        x_val_full = _append_predictors(fold["Xb_val"], x_exp, val_idx, used)
        x_test_full = _append_predictors(fold["Xb_test"], x_exp, test_idx, used)
        full_params = {
            **(fold_cfgs_by_scope.get(full_scope) or {}).get(country, xgb_cfg),
            "random_state": base_seed + fold_i,
        }
        baseline_params = {
            **(fold_cfgs_by_scope.get(baseline_scope) or {}).get(country, xgb_cfg),
            "random_state": base_seed + fold_i,
        }

        full_model = _fit_xgb_fold(
            xgb,
            full_params,
            x_train_full,
            y[tr_inner],
            x_val_full,
            y[val_idx],
            sample_weight=train_weight,
            sample_weight_val=val_weight,
        )
        baseline_model = _fit_xgb_fold(
            xgb,
            baseline_params,
            fold["Xb_train_inner"],
            y[tr_inner],
            fold["Xb_val"],
            y[val_idx],
            sample_weight=train_weight,
            sample_weight_val=val_weight,
        )
        y_pred_full[test_idx] = full_model.predict(x_test_full)
        y_pred_base[test_idx] = baseline_model.predict(fold["Xb_test"])

        for split, counts, weights in (
            ("inner_train", train_counts, train_weight),
            ("validation", val_counts, val_weight),
        ):
            labels = diagnosis[tr_inner] if split == "inner_train" else diagnosis[val_idx]
            for dx in diagnoses:
                mask = labels == dx
                weight_rows.append(
                    {
                        "bag": bag,
                        "objective": spec["objective"],
                        "candidate_id": spec["candidate_id"],
                        "fold_country": country,
                        "split": split,
                        "diagnosis": dx,
                        "n": counts[dx],
                        "weight_sum": float(weights[mask].sum()) if mask.any() else 0.0,
                    }
                )

    oof = pd.DataFrame(
        {
            "row_id": context["row_id"],
            "N_MEGA": context["N_MEGA"],
            "country": context["country"],
            "diagnosis": diagnosis,
            "age": context["age"],
            "sex": context["sex"],
            "y_true": y,
            "y_pred_full": y_pred_full,
            "y_pred_base": y_pred_base,
            "bag_target": bag,
            "candidate_id": spec["candidate_id"],
            "best_rung": rung_id,
            "objective": spec["objective"],
            "training_scheme": "equal_diagnosis_weight",
            "full_hpo_scope": full_scope,
            "full_hpo_artifact_sha256": full_hpo_hash,
            "baseline_hpo_scope": baseline_scope,
            "baseline_hpo_artifact_sha256": baseline_hpo_hash,
        }
    )
    oof["bias_full"] = oof["y_pred_full"] - oof["y_true"]
    oof["bias_base"] = oof["y_pred_base"] - oof["y_true"]
    return oof, pd.DataFrame(weight_rows)


def _metric_rows(
    oof: pd.DataFrame,
    scheme: str,
    diagnoses: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = [("all", oof)] + [
        (dx, oof[oof["diagnosis"] == dx]) for dx in diagnoses
    ]
    for group, subset in groups:
        for model, prediction in (("full", "y_pred_full"), ("baseline", "y_pred_base")):
            clean = subset[["y_true", prediction]].dropna()
            metrics = regression_metrics_extended(
                clean["y_true"].to_numpy(),
                clean[prediction].to_numpy(),
                min_n=5,
            )
            error = clean[prediction].to_numpy() - clean["y_true"].to_numpy()
            rows.append(
                {
                    "bag": str(oof["bag_target"].iloc[0]),
                    "objective": str(oof["objective"].iloc[0]),
                    "candidate_id": str(oof["candidate_id"].iloc[0]),
                    "training_scheme": scheme,
                    "model": model,
                    "diagnosis": group,
                    "n": len(clean),
                    "bias_mean": float(np.mean(error)),
                    "bias_definition": "predicted_minus_observed",
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                }
            )
    return rows


def _paired_cluster_bootstrap(
    merged: pd.DataFrame,
    *,
    bag: str,
    objective: str,
    model: str,
    diagnosis: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    pred_balanced = f"y_pred_{model}_balanced"
    pred_unweighted = f"y_pred_{model}_unweighted"
    subset = merged if diagnosis == "all" else merged[merged["diagnosis"] == diagnosis]
    subset = subset.dropna(
        subset=["y_true", pred_balanced, pred_unweighted, "country"]
    ).copy()
    subset["error_balanced"] = subset[pred_balanced] - subset["y_true"]
    subset["error_unweighted"] = subset[pred_unweighted] - subset["y_true"]

    countries = sorted(subset["country"].astype(str).unique())
    country_stats = (
        subset.groupby(subset["country"].astype(str), observed=True)
        .agg(
            n=("y_true", "size"),
            bias_balanced=("error_balanced", "sum"),
            bias_unweighted=("error_unweighted", "sum"),
            mae_balanced=("error_balanced", lambda values: np.abs(values).sum()),
            mae_unweighted=("error_unweighted", lambda values: np.abs(values).sum()),
        )
        .reindex(countries)
    )
    values = country_stats.to_numpy(dtype=float)
    n_country = len(countries)
    rng = np.random.default_rng(
        bootstrap_seed
        + sum(ord(char) for char in f"{bag}|{objective}|{model}|{diagnosis}")
    )
    draw_index = rng.integers(0, n_country, size=(bootstrap_draws, n_country))
    sampled = values[draw_index].sum(axis=1)
    delta_bias = sampled[:, 1] / sampled[:, 0] - sampled[:, 2] / sampled[:, 0]
    delta_mae = sampled[:, 3] / sampled[:, 0] - sampled[:, 4] / sampled[:, 0]

    def two_sided_p(values: np.ndarray) -> float:
        return min(
            1.0,
            max(
                2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))),
                1 / bootstrap_draws,
            ),
        )

    error_balanced = subset["error_balanced"].to_numpy()
    error_unweighted = subset["error_unweighted"].to_numpy()
    return {
        "bag": bag,
        "objective": objective,
        "model": model,
        "diagnosis": diagnosis,
        "n_subjects": len(subset),
        "n_countries": n_country,
        "delta_bias_balanced_minus_unweighted": float(
            np.mean(error_balanced) - np.mean(error_unweighted)
        ),
        "delta_bias_ci_low": float(np.quantile(delta_bias, 0.025)),
        "delta_bias_ci_high": float(np.quantile(delta_bias, 0.975)),
        "delta_bias_bootstrap_p": two_sided_p(delta_bias),
        "delta_mae_balanced_minus_unweighted": float(
            np.mean(np.abs(error_balanced)) - np.mean(np.abs(error_unweighted))
        ),
        "delta_mae_ci_low": float(np.quantile(delta_mae, 0.025)),
        "delta_mae_ci_high": float(np.quantile(delta_mae, 0.975)),
        "delta_mae_bootstrap_p": two_sided_p(delta_mae),
        "bootstrap_unit": "country",
        "bootstrap_draws": bootstrap_draws,
    }


def _load_reference_oof(
    spec: dict[str, object], diagnoses: tuple[str, ...]
) -> pd.DataFrame:
    path = Path(str(spec["unweighted_oof_source"]))
    reference = pd.read_parquet(path)
    expected_bag = str(spec["bag"])
    if "bag_target" not in reference.columns:
        if "bag" not in reference.columns:
            raise KeyError(
                "Reference OOF must contain either 'bag_target' or 'bag': "
                f"{path}"
            )
        reference["bag_target"] = reference["bag"].astype(str)
    observed_bags = set(reference["bag_target"].dropna().astype(str))
    if observed_bags != {expected_bag}:
        raise ValueError(
            "Reference OOF BAG does not match the selected model: "
            f"expected={expected_bag!r}, observed={sorted(observed_bags)!r}, "
            f"source={path}"
        )
    reference = reference[reference["diagnosis"].astype(str).isin(diagnoses)].copy()
    reference["training_scheme"] = "unweighted"
    return reference


def main() -> None:
    root_text = os.environ.get("REPRO_DATA_ROOT", "").strip()
    if not root_text:
        raise EnvironmentError("REPRO_DATA_ROOT must point to the dedup analysis bundle")
    bundle_root = Path(root_text)
    cfg = load_sensitivity_config()
    paper_cfg = paper_analysis_config(cfg)
    balance_cfg = paper_cfg.diagnosis_balance
    bootstrap_draws = int(balance_cfg["bootstrap_draws"])
    bootstrap_seed = int(balance_cfg["bootstrap_seed"])
    if bootstrap_draws < 1:
        raise ValueError("diagnosis_balance.bootstrap_draws must be positive")
    analysis_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / "diagnosis_balance"
    heavy_root = sensitivity_eval_work_root(cfg, "diagnosis_balance")
    local_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, analysis_cfg)

    manifest_rows: list[dict[str, object]] = []
    weight_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for bag in selected_bags(cfg, include_combined=False):
        for objective in paper_cfg.objectives:
            spec = _best_fixed_spec(
                bundle_root,
                bag,
                objective,
                rung_id=paper_cfg.deployed_rung,
                order_max=paper_cfg.order_max,
                file_suffix=paper_cfg.objective_metadata[objective]["file_suffix"],
                reference_oof_dir_template=paper_cfg.reference_oof_dir_template,
            )
            print(
                f"Diagnosis-balanced LOCO | {bag} | {objective} | "
                f"{spec['candidate_id']} | order={spec['order']} | {paper_cfg.deployed_rung}",
                flush=True,
            )
            balanced, weights = _fit_balanced_oof(
                model_df=model_df,
                feature_names=feature_names,
                bag=bag,
                spec=spec,
                analysis_cfg=analysis_cfg,
                diagnoses=paper_cfg.primary_diagnoses,
                rung_id=paper_cfg.deployed_rung,
            )
            file_suffix = paper_cfg.objective_metadata[objective]["file_suffix"]
            oof_path = (
                heavy_root
                / bag
                / f"oof_predictions_{file_suffix}_equal_diagnosis.parquet"
            )
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            balanced.to_parquet(oof_path, index=False)

            unweighted = _load_reference_oof(spec, paper_cfg.primary_diagnoses)
            metric_rows.extend(
                _metric_rows(
                    balanced,
                    "equal_diagnosis_weight",
                    paper_cfg.primary_diagnoses,
                )
            )
            metric_rows.extend(
                _metric_rows(unweighted, "unweighted", paper_cfg.primary_diagnoses)
            )

            join_keys = ["row_id", "N_MEGA", "country", "diagnosis", "y_true"]
            merged = balanced[join_keys + ["y_pred_full", "y_pred_base"]].merge(
                unweighted[join_keys + ["y_pred_full", "y_pred_base"]],
                on=join_keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_balanced", "_unweighted"),
            )
            if len(merged) != len(balanced) or len(merged) != len(unweighted):
                raise ValueError(
                    f"Balanced/reference OOF mismatch for {bag} {objective}: "
                    f"balanced={len(balanced)}, unweighted={len(unweighted)}, merged={len(merged)}"
                )
            for model in ("full", "base"):
                for diagnosis in ("all", *paper_cfg.primary_diagnoses):
                    bootstrap_rows.append(
                        _paired_cluster_bootstrap(
                            merged,
                            bag=bag,
                            objective=objective,
                            model=model,
                            diagnosis=diagnosis,
                            bootstrap_draws=bootstrap_draws,
                            bootstrap_seed=bootstrap_seed,
                        )
                    )

            manifest_rows.append(
                {
                    key: value if key != "predictors" else "|".join(value)
                    for key, value in spec.items()
                }
                | {
                    "balanced_oof_output": str(oof_path),
                    "weighting": "inverse diagnosis frequency within each inner-train and validation split",
                    "held_out_evaluation_weighted": False,
                    "primary_diagnoses": "|".join(paper_cfg.primary_diagnoses),
                    "configuration_source": str(CONFIG_PATH.resolve()),
                    "full_hpo_scope": str(balanced["full_hpo_scope"].iloc[0]),
                    "full_hpo_artifact_sha256": str(
                        balanced["full_hpo_artifact_sha256"].iloc[0]
                    ),
                    "baseline_hpo_scope": str(balanced["baseline_hpo_scope"].iloc[0]),
                    "baseline_hpo_artifact_sha256": str(
                        balanced["baseline_hpo_artifact_sha256"].iloc[0]
                    ),
                }
            )
            weight_parts.append(weights)

    manifest = pd.DataFrame(manifest_rows)
    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["bag", "objective", "training_scheme", "model", "diagnosis"]
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    weights = pd.concat(weight_parts, ignore_index=True)

    tables = {
        "diagnosis_balance_model_manifest.csv": manifest,
        "diagnosis_balance_metrics.csv": metrics,
        "diagnosis_balance_country_bootstrap.csv": bootstrap,
        "diagnosis_balance_fold_weights.csv": weights,
    }
    # Cluster jobs run one BAG at a time. Persist each BAG first, then rebuild
    # the shared paper table in primary-BAG order. Whichever job finishes last
    # sees both completed BAG directories, avoiding last-writer data loss.
    completed_bags = [
        bag
        for bag in paper_cfg.primary_bags
        if not manifest.loc[manifest["bag"].astype(str).eq(bag)].empty
    ]
    for bag in completed_bags:
        bag_root = local_root / bag
        bag_root.mkdir(parents=True, exist_ok=True)
        for filename, table in tables.items():
            table.loc[table["bag"].astype(str).eq(bag)].to_csv(
                bag_root / filename, index=False
            )

    for filename in tables:
        parts = []
        for bag in paper_cfg.primary_bags:
            path = local_root / bag / filename
            if path.is_file():
                parts.append(pd.read_csv(path))
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(
                local_root / filename, index=False
            )
    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "diagnosis_balance")

    print(f"Wrote lightweight summaries to {local_root}", flush=True)
    print(f"Wrote heavy balanced OOF predictions to {heavy_root}", flush=True)


if __name__ == "__main__":
    main()
