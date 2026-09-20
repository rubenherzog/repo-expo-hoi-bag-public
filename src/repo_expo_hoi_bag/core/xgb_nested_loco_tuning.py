"""Leakage-free nested-LOCO XGBoost hyperparameter selection.

This deliberately is not a generic HPO framework.  It tunes the canonical
baseline plus every canonical exposome feature for one fixed-depth XGBoost rung.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from loco_fusion_matrix_engine import regression_metrics_extended
from oinfo_bag_ladder.io_utils import stable_hash, write_parquet
from xgb_loco_engine import _append_predictors, _build_fold_mats_from_indices, _fit_xgb_fold, require_xgboost


TUNABLE_KEYS = (
    "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
    "reg_alpha", "reg_lambda", "gamma",
)
ARTIFACT_SCHEMA_VERSION = 1
SELECTION_OBJECTIVES = {
    "pooled_global_r2", "country_r2_mean", "country_r2_p25", "country_balanced_r2",
    "mean_country_mse",
}
FITTING_MODES = {"nested_per_outer_country", "global_loco_country_mse"}
OPTIMIZERS = {"custom_gp", "optuna_tpe", "optuna_gp"}
FEATURE_SCOPES = {"baseline", "single_exposure", "full_exposome", "domain_balanced_k10"}


def deterministic_seed(*parts: object) -> int:
    """Stable seed independent of worker ordering and Python hash randomization."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "little") % (2**31 - 1)


def inner_fold_fit_seed(unit_seed: int, inner_country: str, fold_index: int) -> int:
    """Return a fit seed shared by every trial in one inner LOCO fold.

    The seed deliberately excludes the trial index. Otherwise Bayesian
    optimization would rank hyperparameters and independent XGBoost random
    draws together, then deploy the selected hyperparameters with a different
    draw. ``unit_seed`` already identifies BAG × outer-country × rung.
    """
    return deterministic_seed(unit_seed, "inner_fit", inner_country, fold_index)


def selection_objective_value(
    selection_objective: str,
    pooled_r2: float,
    country_r2: Sequence[float],
    country_balanced_r2: float | None = None,
    country_mse: Sequence[float] | None = None,
) -> float:
    """Aggregate inner-LOCO performance using the configured transfer target."""
    if selection_objective not in SELECTION_OBJECTIVES:
        raise ValueError(
            f"Unsupported XGB tuning selection_objective {selection_objective!r}; "
            f"expected one of {sorted(SELECTION_OBJECTIVES)}"
        )
    if selection_objective == "pooled_global_r2":
        return float(pooled_r2)
    if selection_objective == "country_balanced_r2":
        return float("nan") if country_balanced_r2 is None else float(country_balanced_r2)
    if selection_objective == "mean_country_mse":
        scores = np.asarray(country_mse, dtype=float)
        return float("nan") if scores.size == 0 or not np.isfinite(scores).all() else float(np.mean(scores))
    scores = np.asarray(country_r2, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        return float("nan")
    if selection_objective == "country_r2_mean":
        return float(np.mean(scores))
    return float(np.percentile(scores, 25))


def select_trial_cap_or_practical_convergence(
    history: Sequence[Mapping[str, Any]], *, max_trials: int,
    minimum_completed_trials: int, lookback_trials: int,
    minimum_incumbent_improvement_mse: float,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Freeze a global-HPO incumbent under the paper's bounded selection rule."""
    if max_trials < 1 or minimum_completed_trials < 1 or lookback_trials < 1:
        raise ValueError("HPO cap-or-convergence trial counts must be positive")
    if minimum_completed_trials > max_trials or lookback_trials >= minimum_completed_trials:
        raise ValueError("HPO cap-or-convergence windows are inconsistent")
    if minimum_incumbent_improvement_mse < 0:
        raise ValueError("minimum_incumbent_improvement_mse must be non-negative")
    valid = [row for row in history if np.isfinite(float(row.get("objective_value", np.nan)))]
    if len(valid) != len(history) or not valid:
        raise ValueError("HPO history must contain contiguous finite objective values")
    indices = [int(row.get("trial_index", -1)) for row in valid]
    if indices != list(range(len(valid))):
        raise ValueError("HPO history trial indices must be contiguous from zero")
    observed = len(valid)
    if observed >= max_trials:
        eligible = valid[:max_trials]
        reason = "max_trials_cap"
        improvement = None
    else:
        if observed < minimum_completed_trials:
            raise ValueError(
                f"HPO checkpoint has {observed} trials; at least {minimum_completed_trials} are required"
            )
        prior = valid[:observed - lookback_trials]
        incumbent_prior = min(float(row["objective_value"]) for row in prior)
        incumbent_now = min(float(row["objective_value"]) for row in valid)
        improvement = incumbent_prior - incumbent_now
        if improvement >= minimum_incumbent_improvement_mse:
            raise ValueError(
                "HPO checkpoint has not practically converged: "
                f"incumbent improvement={improvement:.6f} >= {minimum_incumbent_improvement_mse:.6f}"
            )
        eligible = valid
        reason = "practical_convergence"
    best = min(eligible, key=lambda row: (float(row["objective_value"]), int(row["trial_index"])))
    return best, {
        "observed_trials": observed,
        "selection_trial_limit": len(eligible),
        "stop_reason": reason,
        "incumbent_improvement_over_lookback_mse": improvement,
    }


def country_balanced_r2(
    y_true: Sequence[float], y_pred: Sequence[float], countries: Sequence[object]
) -> float:
    """Compute OOF R² with every observed country contributing equal total weight."""
    observed = np.asarray(countries, dtype=object)
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if not (len(observed) == len(truth) == len(predicted)):
        raise ValueError("Country-balanced R² inputs must have the same length")
    valid = np.isfinite(truth) & np.isfinite(predicted)
    if not valid.any():
        return float("nan")
    observed = observed[valid]
    truth = truth[valid]
    predicted = predicted[valid]
    _, inverse, counts = np.unique(observed.astype(str), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weighted_mean = float(np.sum(weights * truth) / np.sum(weights))
    total = float(np.sum(weights * np.square(truth - weighted_mean)))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum(weights * np.square(truth - predicted)))
    return float(1.0 - residual / total)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_continuation_history(
    parent_checkpoint_path: Path, child_signature_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load a compatible completed/interrupted parent history without refitting it.

    A continuation deliberately has a new ``n_trials`` and therefore a new
    checkpoint identity.  The scientific inputs and optimizer definition must
    otherwise be identical; accepting merely a nearby JSON file would mix
    incompatible LOCO populations or search spaces.
    """
    path = Path(parent_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Continuation parent checkpoint is missing: {path}")
    parent = json.loads(path.read_text(encoding="utf-8"))
    parent_payload = parent.get("signature_payload")
    if not isinstance(parent_payload, dict) or not isinstance(parent.get("trials"), list):
        raise ValueError(f"Invalid continuation parent checkpoint: {path}")
    for key in ("schema", "bag", "outer_country", "rung_id", "base_xgb_cfg", "early_stop_cfg", "countries", "x_shape"):
        if parent_payload.get(key) != child_signature_payload.get(key):
            raise ValueError(f"Continuation parent is incompatible at {key}: {path}")
    if parent_payload.get("feature_scope", "full_exposome") != child_signature_payload.get("feature_scope", "full_exposome"):
        raise ValueError(f"Continuation parent is incompatible at feature_scope: {path}")
    parent_cfg = dict(parent_payload.get("tuning_cfg", {}))
    child_cfg = dict(child_signature_payload.get("tuning_cfg", {}))
    parent_cfg.setdefault("feature_scope", "full_exposome")
    child_cfg.setdefault("feature_scope", "full_exposome")
    # Trial budget and explicit provenance naturally differ between a parent
    # and its immutable continuation; every modeling/optimizer field must not.
    for key in ("n_trials", "parent_run_signature"):
        parent_cfg.pop(key, None)
        child_cfg.pop(key, None)
    if parent_cfg != child_cfg:
        raise ValueError(f"Continuation parent has an incompatible tuning configuration: {path}")
    history = [dict(row) for row in parent["trials"]]
    indices = [int(row.get("trial_index", -1)) for row in history]
    if indices != list(range(len(history))):
        raise ValueError(f"Continuation parent has non-contiguous trial indices: {path}")
    if not history:
        raise ValueError(f"Continuation parent has no trials: {path}")
    if len(history) >= int(child_signature_payload["tuning_cfg"]["n_trials"]):
        raise ValueError("Continuation target n_trials must exceed its parent history")
    return history


def create_tuning_run_paths(
    checkout_root: Path,
    runtime_root: Path,
    analysis_signature: str,
    *,
    run_id: str | None = None,
) -> tuple[str, Path, Path]:
    """Create isolated local-delivery and external-checkpoint directories.

    An automatic invocation receives a time-qualified signature with a short
    scientific fingerprint. A caller may instead provide a concise,
    human-readable immutable label. In both cases an existing destination is
    rejected rather than overwritten, and the manifest records the complete
    scientific fingerprint.
    """
    checkout = Path(checkout_root).resolve()
    local_base = checkout / "outputs" / "xgb_nested_loco_tuning"
    requested_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not requested_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("XGB tuning run id may contain only letters, digits, '-' and '_'")
    # A caller-provided run id is an intentional, human-readable delivery
    # label (for example ``historical5_20260910T230000Z``).  Its immutable
    # manifest carries the full scientific fingerprint; appending an opaque
    # hash to that label makes routine result discovery needlessly difficult.
    # Automatically generated ids retain the short fingerprint.
    run_signature = requested_id if run_id else f"{requested_id}-{analysis_signature[:16]}"
    output_dir = (local_base / run_signature).resolve()
    if output_dir.parent != local_base.resolve():
        raise ValueError("Refusing an XGB tuning output path outside local outputs/")
    checkpoint_dir = Path(runtime_root).resolve() / "work" / "xgb_nested_loco_tuning" / run_signature
    # mkdir(exist_ok=False) is deliberate: delivered artifacts are immutable
    # across runs, and a collision must be resolved explicitly rather than
    # silently replacing a prior result.
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        # The local directory documents the failed allocation; it is not
        # removed because repository rules prohibit inferred deletions.
        raise
    return run_signature, output_dir, checkpoint_dir


def append_progress_line(path: Path, message: str) -> None:
    """Append one worker-safe, line-oriented progress record to the local log."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


def _decode_point(point: np.ndarray, search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value, key in zip(point, TUNABLE_KEYS):
        spec = search_space[key]
        low, high = float(spec["low"]), float(spec["high"])
        if spec["kind"] == "log_or_zero" and float(value) <= 0.0:
            resolved = 0.0
        elif spec["kind"] in {"log", "log_or_zero"}:
            resolved = float(np.exp(np.log(low) + float(value) * (np.log(high) - np.log(low))))
        else:
            resolved = low + float(value) * (high - low)
        params[key] = int(round(resolved)) if spec["kind"] == "int" else float(resolved)
    return params


def _point_from_params(params: Mapping[str, Any], search_space: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    vals: list[float] = []
    for key in TUNABLE_KEYS:
        spec = search_space[key]
        low, high = float(spec["low"]), float(spec["high"])
        value = float(params[key])
        if spec["kind"] == "log_or_zero" and value == 0.0:
            vals.append(0.0)
        elif spec["kind"] in {"log", "log_or_zero"}:
            vals.append((np.log(value) - np.log(low)) / (np.log(high) - np.log(low)))
        else:
            vals.append((value - low) / (high - low))
    return np.asarray(vals, dtype=float)


def _next_point(
    history: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], unit_seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    n_initial = int(cfg["n_initial_trials"])
    search_space = cfg["search_space"]
    trial_index = len(history)
    if trial_index < n_initial:
        point = qmc.LatinHypercube(d=len(TUNABLE_KEYS), seed=unit_seed).random(n_initial)[trial_index]
        # The continuous design cannot land exactly on zero. Reserve trial one
        # in every BAG/country/rung unit for the unregularized L1 case.
        if trial_index == 0 and search_space["reg_alpha"]["kind"] == "log_or_zero":
            point[TUNABLE_KEYS.index("reg_alpha")] = 0.0
        return (
            point,
            {"optimizer_warning_count": 0, "optimizer_warnings": "", "optimizer_kernel": ""},
        )

    x = np.vstack([_point_from_params(row["params"], search_space) for row in history])
    y = np.asarray([float(row["objective_value"]) for row in history], dtype=float)
    # The GP acquisition is written as a maximizer.  The global alternative
    # stores its scientific objective as positive mean country MSE, so invert
    # only the surrogate target while retaining the auditable MSE value.
    if str(cfg.get("optimization_direction", "maximize")) == "minimize":
        y = -y
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return (
            qmc.Sobol(d=len(TUNABLE_KEYS), scramble=True, seed=unit_seed + trial_index).random_base2(11)[trial_index],
            {"optimizer_warning_count": 0, "optimizer_warnings": "", "optimizer_kernel": ""},
        )
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-6),
        normalize_y=True,
        random_state=unit_seed,
        n_restarts_optimizer=0,
    )
    # These convergence warnings describe only the auxiliary GP surrogate, not
    # an XGBoost fit or a scientific score. Preserve them in trial diagnostics,
    # while leaving every other warning visible to the caller.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(x[valid], y[valid])
    gp_warnings = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(warning.message, warning.category, stacklevel=2)
    candidates = qmc.Sobol(d=len(TUNABLE_KEYS), scramble=True, seed=unit_seed + trial_index).random_base2(
        int(np.ceil(np.log2(int(cfg["sobol_candidates"]))))
    )
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-12)
    improvement = mean - np.max(y[valid])
    z = improvement / std
    expected_improvement = improvement * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return candidates[int(np.argmax(expected_improvement))], {
        "optimizer_warning_count": len(gp_warnings),
        "optimizer_warnings": " | ".join(str(warning.message) for warning in gp_warnings),
        "optimizer_kernel": str(gp.kernel_),
    }


def _optuna_suggest(trial: Any, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", space["min_child_weight"]["low"], space["min_child_weight"]["high"]),
        "subsample": trial.suggest_float("subsample", space["subsample"]["low"], space["subsample"]["high"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", space["colsample_bytree"]["low"], space["colsample_bytree"]["high"]),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, space["reg_alpha"]["high"]),
        "reg_lambda": trial.suggest_float("reg_lambda", space["reg_lambda"]["low"], space["reg_lambda"]["high"], log=True),
        "gamma": trial.suggest_float("gamma", space["gamma"]["low"], space["gamma"]["high"]),
    }


def _restore_optuna_study(
    history: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], unit_seed: int
) -> Any:
    """Restore an Optuna study once, preserving its sampler cache thereafter."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna optimizers require the versioned optuna dependency") from exc
    space = cfg["search_space"]
    direction = "minimize" if str(cfg.get("optimization_direction", "maximize")) == "minimize" else "maximize"
    optimizer = str(cfg.get("optimizer", "optuna_tpe"))
    n_startup = max(0, int(cfg["n_initial_trials"]) - 1)
    if optimizer == "optuna_gp":
        sampler = optuna.samplers.GPSampler(
            seed=int(unit_seed), n_startup_trials=n_startup, deterministic_objective=True,
        )
    else:
        sampler = optuna.samplers.TPESampler(
            seed=int(unit_seed), n_startup_trials=n_startup, multivariate=True,
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=sampler)
    for row in history:
        value = float(row["objective_value"])
        if not np.isfinite(value):
            continue
        # Replaying ask/tell, rather than only adding finished trials, advances
        # TPE's seeded random state exactly as in the original invocation.
        if int(row["trial_index"]) == 0 and bool(row.get("is_manual_incumbent", False)):
            study.enqueue_trial(dict(row["params"]))
        replay = study.ask()
        _optuna_suggest(replay, space)
        study.tell(replay, value)
    return study


def build_nested_fold_designs(
    context: Mapping[str, Any], outer_country: str, analysis_cfg: Mapping[str, Any],
    early_stop_cfg: Mapping[str, Any], seed: int,
) -> dict[str, dict[str, Any]]:
    """Return inner LOCO designs with hard outer/inner country separation."""
    country = np.asarray(context["country"]).astype(str)
    outer_mask = country == str(outer_country)
    designs: dict[str, dict[str, Any]] = {}
    for inner_index, inner_country in enumerate(context["countries"]):
        if str(inner_country) == str(outer_country):
            continue
        score_idx = np.flatnonzero(country == str(inner_country))
        train_idx = np.flatnonzero(~outer_mask & (country != str(inner_country)))
        design = _build_fold_mats_from_indices(
            context, str(inner_country), train_idx, score_idx, analysis_cfg, early_stop_cfg,
            deterministic_seed(seed, outer_country, inner_country, inner_index),
        )
        designs[str(inner_country)] = design
    return designs


def build_global_loco_fold_designs(
    context: Mapping[str, Any], analysis_cfg: Mapping[str, Any],
    early_stop_cfg: Mapping[str, Any], seed: int,
) -> dict[str, dict[str, Any]]:
    """Build one conventional LOCO design for every available country.

    Each held-out country is absent from both fitting and early stopping because
    validation is drawn only from the supplied non-held-out training indices.
    """
    return {
        str(country): _build_fold_mats_from_indices(
            context, str(country), context["train_idx_by_country"][country],
            context["test_idx_by_country"][country], analysis_cfg, early_stop_cfg,
            deterministic_seed(seed, "global_loco", country, fold_index),
        )
        for fold_index, country in enumerate(context["countries"])
    }


def _feature_sets(context: Mapping[str, Any], feature_scope: str) -> list[tuple[str, list[int]]]:
    """Return the fixed predictor panels used by one HPO objective.

    ``single_exposure`` deliberately contains every canonical exposure.  It is
    not a sampled calibration panel and therefore cannot privilege a manually
    chosen feature.  Every trial receives the same panel.
    """
    if feature_scope not in FEATURE_SCOPES:
        raise ValueError(f"Unsupported XGB tuning feature_scope {feature_scope!r}")
    n_features = int(np.asarray(context["X_exp"]).shape[1])
    if feature_scope == "baseline":
        return [("__baseline__", [])]
    if feature_scope == "full_exposome":
        return [("__full_exposome__", list(range(n_features)))]
    names = list(context.get("exposome_features", []))
    if len(names) != n_features:
        names = [f"feature_{index}" for index in range(n_features)]
    if feature_scope == "domain_balanced_k10":
        configured_sets = context.get("hpo_feature_sets")
        if not isinstance(configured_sets, Sequence) or not configured_sets:
            raise ValueError("domain_balanced_k10 requires validated hpo_feature_sets in its context")
        name_to_index = {str(name): index for index, name in enumerate(names)}
        resolved_sets: list[tuple[str, list[int]]] = []
        for item in configured_sets:
            if not isinstance(item, Mapping):
                raise ValueError("domain_balanced_k10 feature-set configuration must contain mappings")
            identifier = str(item.get("id", "")).strip()
            members = [str(value) for value in item.get("features", [])]
            if not identifier or not members:
                raise ValueError("domain_balanced_k10 feature sets require id and features")
            unknown = sorted(set(members).difference(name_to_index))
            if unknown:
                raise ValueError(f"domain_balanced_k10 contains unknown exposures: {unknown}")
            resolved_sets.append((identifier, [name_to_index[feature] for feature in members]))
        return resolved_sets
    return [(str(name), [index]) for index, name in enumerate(names)]


def _evaluate_trial_fold(
    context: Mapping[str, Any], inner_country: str, design: Mapping[str, Any], fold_index: int,
    base_xgb_cfg: Mapping[str, Any], params: Mapping[str, Any], seed: int, xgb_nthread: int,
    feature_scope: str,
    feature_sets_override: Sequence[tuple[str, list[int]]] | None = None,
) -> dict[str, Any]:
    """Fit and score the fixed HPO panel for one held-out LOCO country."""
    xgb = require_xgboost()
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)
    feature_sets = list(feature_sets_override) if feature_sets_override is not None else _feature_sets(context, feature_scope)
    try:
        config = {
            **base_xgb_cfg, **params, "nthread": int(xgb_nthread),
        }
        score_idx = design["test_idx"]
        predictions: list[pd.DataFrame] = []
        feature_r2: dict[str, float] = {}
        feature_mse: dict[str, float] = {}
        best_iterations: list[int] = []
        fit_warnings: list[str] = []
        failures: list[str] = []
        for feature_name, requested_indices in feature_sets:
            # A single-exposure panel contains independent models. Preserve a
            # trial-invariant draw for each one without making all exposures
            # share the same XGBoost randomness.
            config["random_state"] = (
                deterministic_seed(seed, "panel_fit", inner_country, fold_index, feature_name)
                if feature_scope in {"single_exposure", "domain_balanced_k10"}
                else inner_fold_fit_seed(seed, inner_country, fold_index)
            )
            train_variance = np.nanvar(x_exp[np.ix_(design["train_idx"], requested_indices)], axis=0)
            predictor_indices = [
                index for index, keep in zip(requested_indices, train_variance > 0) if bool(keep)
            ]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reg = _fit_xgb_fold(
                        xgb, config,
                        _append_predictors(design["Xb_train_inner"], x_exp, design["tr_inner_idx"], predictor_indices),
                        y[design["tr_inner_idx"]],
                        _append_predictors(design["Xb_val"], x_exp, design["val_idx"], predictor_indices),
                        y[design["val_idx"]],
                    )
                prediction = reg.predict(_append_predictors(design["Xb_test"], x_exp, score_idx, predictor_indices))
                metrics = regression_metrics_extended(
                    y[score_idx], prediction, min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5))
                )
                valid = np.isfinite(y[score_idx]) & np.isfinite(prediction)
                mse = float(np.mean(np.square(y[score_idx][valid] - prediction[valid]))) if valid.any() else np.nan
                feature_r2[feature_name] = float(metrics["r2"])
                feature_mse[feature_name] = mse
                predictions.append(pd.DataFrame({
                    "y_true": y[score_idx], "y_pred": prediction, "country": str(inner_country),
                    "feature_scope_member": feature_name,
                }))
                best_iteration = getattr(reg, "best_iteration", None)
                if best_iteration is not None:
                    best_iterations.append(int(best_iteration))
                fit_warnings.extend(
                    f"{inner_country}/{feature_name}:{warning.category.__name__}:{warning.message}"
                    for warning in caught
                )
            except Exception as exc:
                failures.append(f"{inner_country}/{feature_name}:{exc!r}")
        if failures or len(predictions) != len(feature_sets):
            return {
                "country": str(inner_country), "prediction": None, "r2": np.nan, "mse": np.nan,
                "best_iteration": None, "fit_warnings": fit_warnings,
                "failure": " | ".join(failures) or f"{inner_country}:incomplete feature scope",
                "feature_r2": feature_r2, "feature_mse": feature_mse,
            }
        country_r2 = float(np.mean(list(feature_r2.values())))
        country_mse = float(np.mean(list(feature_mse.values())))
        return {
            "country": str(inner_country), "prediction": pd.concat(predictions, ignore_index=True),
            "r2": country_r2, "mse": country_mse,
            "best_iteration": float(np.median(best_iterations)) if best_iterations else None,
            "fit_warnings": fit_warnings, "failure": "", "feature_r2": feature_r2, "feature_mse": feature_mse,
        }
    except Exception as exc:  # trial diagnostics must survive an individual XGBoost failure
        return {"country": str(inner_country), "prediction": None, "r2": np.nan, "mse": np.nan,
                "best_iteration": None, "fit_warnings": [], "failure": f"{inner_country}:{exc!r}",
                "feature_r2": {}, "feature_mse": {}}


def _evaluate_trial(
    context: Mapping[str, Any], designs: Mapping[str, Mapping[str, Any]], base_xgb_cfg: Mapping[str, Any],
    params: Mapping[str, Any], seed: int, selection_objective: str, *, fold_n_jobs: int = 1,
    xgb_nthread: int = 1, feature_scope: str = "full_exposome",
    single_parallel_axis: str = "country_exposure",
) -> dict[str, Any]:
    items = list(enumerate(designs.items()))
    if feature_scope in {"single_exposure", "domain_balanced_k10"} and single_parallel_axis == "country_exposure":
        # A one-exposure fit is too small to benefit materially from internal
        # XGBoost threads. Schedule the independent country×exposure pairs in
        # one shared pool instead of making each country process 63 fits in
        # series. This preserves fit seeds and the exact country×exposure
        # objective while removing country-duration load imbalance.
        pair_items = [
            (fold_index, country, design, member)
            for fold_index, (country, design) in items
            for member in _feature_sets(context, feature_scope)
        ]
        pair_evaluate = [
            delayed(_evaluate_trial_fold)(
                context, country, design, fold_index, base_xgb_cfg, params, seed, xgb_nthread,
                feature_scope, [member],
            )
            for fold_index, country, design, member in pair_items
        ]
        pair_outcomes = (
            Parallel(n_jobs=min(int(fold_n_jobs), len(pair_evaluate)), backend="threading")(pair_evaluate)
            if int(fold_n_jobs) > 1 else [
                _evaluate_trial_fold(
                    context, country, design, fold_index, base_xgb_cfg, params, seed, xgb_nthread,
                    feature_scope, [member],
                )
                for fold_index, country, design, member in pair_items
            ]
        )
        outcomes = []
        for _fold_index, (country, _design) in items:
            country_pairs = [item for item in pair_outcomes if item["country"] == str(country)]
            failed = [item for item in country_pairs if item["prediction"] is None]
            if failed:
                outcomes.append({
                    "country": str(country), "prediction": None, "r2": np.nan, "mse": np.nan,
                    "best_iteration": None,
                    "fit_warnings": [warning for item in country_pairs for warning in item["fit_warnings"]],
                    "failure": " | ".join(str(item["failure"]) for item in failed),
                    "feature_r2": {key: value for item in country_pairs for key, value in item["feature_r2"].items()},
                    "feature_mse": {key: value for item in country_pairs for key, value in item["feature_mse"].items()},
                })
                continue
            feature_r2 = {key: value for item in country_pairs for key, value in item["feature_r2"].items()}
            feature_mse = {key: value for item in country_pairs for key, value in item["feature_mse"].items()}
            predictions = [item["prediction"] for item in country_pairs]
            iterations = [item["best_iteration"] for item in country_pairs if item["best_iteration"] is not None]
            outcomes.append({
                "country": str(country), "prediction": pd.concat(predictions, ignore_index=True),
                "r2": float(np.mean(list(feature_r2.values()))), "mse": float(np.mean(list(feature_mse.values()))),
                "best_iteration": float(np.median(iterations)) if iterations else None,
                "fit_warnings": [warning for item in country_pairs for warning in item["fit_warnings"]], "failure": "",
                "feature_r2": feature_r2, "feature_mse": feature_mse,
            })
    else:
        evaluate = [
            delayed(_evaluate_trial_fold)(
                context, country, design, fold_index, base_xgb_cfg, params, seed, xgb_nthread, feature_scope
            )
            for fold_index, (country, design) in items
        ]
        outcomes = (
            Parallel(n_jobs=min(int(fold_n_jobs), len(evaluate)), backend="threading")(evaluate)
            if int(fold_n_jobs) > 1 else [
                _evaluate_trial_fold(
                    context, country, design, fold_index, base_xgb_cfg, params, seed, xgb_nthread, feature_scope
                )
                for fold_index, (country, design) in items
            ]
        )
    predictions = [item["prediction"] for item in outcomes if item["prediction"] is not None]
    best_iterations = [int(item["best_iteration"]) for item in outcomes if item["best_iteration"] is not None]
    failures = [str(item["failure"]) for item in outcomes if item["failure"]]
    fit_warnings = [warning for item in outcomes for warning in item["fit_warnings"]]
    country_r2 = [float(item["r2"]) for item in outcomes if item["prediction"] is not None]
    country_mse = [float(item["mse"]) for item in outcomes if item["prediction"] is not None]
    country_r2_by_name = {str(item["country"]): float(item["r2"]) for item in outcomes if item["prediction"] is not None}
    country_mse_by_name = {str(item["country"]): float(item["mse"]) for item in outcomes if item["prediction"] is not None}
    country_feature_r2 = {str(item["country"]): item["feature_r2"] for item in outcomes}
    country_feature_mse = {str(item["country"]): item["feature_mse"] for item in outcomes}
    merged = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
        columns=["y_true", "y_pred", "country"]
    )
    metrics = regression_metrics_extended(
        merged["y_true"].to_numpy(), merged["y_pred"].to_numpy(),
        min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
    )
    pooled_r2 = float(metrics["r2"])
    balanced_r2 = country_balanced_r2(
        merged["y_true"].to_numpy(), merged["y_pred"].to_numpy(), merged["country"].to_numpy()
    )
    complete_country_coverage = len(predictions) == len(designs)
    return {
        "objective_value": selection_objective_value(
            selection_objective, pooled_r2, country_r2, balanced_r2,
            country_mse if complete_country_coverage else None,
        ),
        "selection_objective": selection_objective,
        "inner_global_oof_r2": pooled_r2,
        "inner_country_balanced_oof_r2": balanced_r2,
        "inner_country_r2_p25": float(np.percentile(country_r2, 25)) if country_r2 else np.nan,
        "inner_country_r2_median": float(np.median(country_r2)) if country_r2 else np.nan,
        "mean_country_mse": float(np.mean(country_mse)) if country_mse and np.isfinite(country_mse).all() else np.nan,
        "median_country_mse": float(np.median(country_mse)) if country_mse and np.isfinite(country_mse).all() else np.nan,
        "country_r2_json": json.dumps(country_r2_by_name, sort_keys=True),
        "country_mse_json": json.dumps(country_mse_by_name, sort_keys=True),
        "country_feature_r2_json": json.dumps(country_feature_r2, sort_keys=True),
        "country_feature_mse_json": json.dumps(country_feature_mse, sort_keys=True),
        "n_scored": int(metrics["n_scored"]),
        "inner_folds_n": len(designs),
        "inner_folds_succeeded": len(predictions),
        "complete_country_coverage": complete_country_coverage,
        "best_iteration_median": float(np.median(best_iterations)) if best_iterations else np.nan,
        "failures": " | ".join(failures),
        "fit_warning_count": len(fit_warnings),
        "fit_warnings": " | ".join(fit_warnings),
    }


def tune_unit(
    context: Mapping[str, Any], outer_country: str, rung_id: str, base_xgb_cfg: Mapping[str, Any],
    analysis_cfg: Mapping[str, Any], early_stop_cfg: Mapping[str, Any], tuning_cfg: Mapping[str, Any],
    checkpoint_path: Path,
    nested_designs: Mapping[str, Mapping[str, Any]] | None = None,
    progress_log_path: Path | None = None,
    parent_checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune one BAG × scope × rung, resuming exact compatible checkpoints."""
    signature_payload = {
        "schema": ARTIFACT_SCHEMA_VERSION, "bag": context["bag_name"], "outer_country": outer_country,
        "rung_id": rung_id, "base_xgb_cfg": dict(base_xgb_cfg), "early_stop_cfg": dict(early_stop_cfg),
        "tuning_cfg": dict(tuning_cfg), "countries": list(context["countries"]), "x_shape": list(np.asarray(context["X_exp"]).shape),
        "feature_scope": str(tuning_cfg.get("feature_scope", "full_exposome")),
    }
    signature = stable_hash(signature_payload)
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checkpoint = {}
    history = checkpoint.get("trials", []) if checkpoint.get("signature") == signature else []
    if not history and parent_checkpoint_path is not None:
        history = load_continuation_history(parent_checkpoint_path, signature_payload)
    unit_seed = deterministic_seed(tuning_cfg["seed"], context["bag_name"], outer_country, rung_id)
    designs = nested_designs or build_nested_fold_designs(
        context, outer_country, analysis_cfg, early_stop_cfg,
        deterministic_seed(tuning_cfg["seed"], context["bag_name"], outer_country),
    )
    optuna_study = None
    optuna_trial = None
    while len(history) < int(tuning_cfg["n_trials"]):
        started = time.monotonic()
        if not history and bool(tuning_cfg.get("include_manual_incumbent", False)):
            params = {key: base_xgb_cfg[key] for key in TUNABLE_KEYS}
            optimizer_diag = {
                "optimizer_warning_count": 0,
                "optimizer_warnings": "",
                "optimizer_kernel": "manual_incumbent",
            }
            is_manual_incumbent = True
        else:
            if str(tuning_cfg.get("optimizer", "custom_gp")).startswith("optuna_"):
                if optuna_study is None:
                    optuna_study = _restore_optuna_study(history, tuning_cfg, unit_seed)
                    # ``reg_alpha=0`` is a scientifically valid no-L1 case,
                    # not an arbitrary manual incumbent. Optuna's continuous
                    # sampler almost never lands exactly on zero, so reserve
                    # one otherwise ordinary proposal for that boundary.
                    if not history and bool(tuning_cfg.get("include_zero_reg_alpha_trial", True)):
                        optuna_study.enqueue_trial({"reg_alpha": 0.0})
                optuna_trial = optuna_study.ask()
                params = _optuna_suggest(optuna_trial, tuning_cfg["search_space"])
                optimizer_diag = {
                    "optimizer_warning_count": 0, "optimizer_warnings": "",
                    "optimizer_kernel": str(tuning_cfg["optimizer"]),
                }
            else:
                point, optimizer_diag = _next_point(history, tuning_cfg, unit_seed)
                params = _decode_point(point, tuning_cfg["search_space"])
            is_manual_incumbent = False
        # All trials share the same deterministic seed in each inner fold. The
        # optimizer must select hyperparameters, not a lucky random draw.
        result = _evaluate_trial(
            context, designs, base_xgb_cfg, params, unit_seed,
            str(tuning_cfg["selection_objective"]),
            fold_n_jobs=int(tuning_cfg.get("fold_n_jobs", 1)),
            xgb_nthread=int(tuning_cfg.get("xgb_nthread", 1)),
            feature_scope=str(tuning_cfg.get("feature_scope", "full_exposome")),
            single_parallel_axis=str(tuning_cfg.get("single_parallel_axis", "country_exposure")),
        )
        if optuna_trial is not None:
            value = float(result["objective_value"])
            if np.isfinite(value):
                optuna_study.tell(optuna_trial, value)
            else:
                optuna_study.tell(optuna_trial, state=__import__("optuna").trial.TrialState.FAIL)
            optuna_trial = None
        row = {
            "trial_index": len(history), "params": params, **result, **optimizer_diag,
            "elapsed_sec": time.monotonic() - started, "is_manual_incumbent": is_manual_incumbent,
        }
        history.append(row)
        _atomic_json(checkpoint_path, {"signature": signature, "signature_payload": signature_payload, "trials": history})
        valid_values = [float(item["objective_value"]) for item in history if np.isfinite(float(item["objective_value"]))]
        best_so_far = (
            min(valid_values) if str(tuning_cfg.get("optimization_direction", "maximize")) == "minimize"
            else max(valid_values)
        ) if valid_values else np.nan
        message = (
            f"[tuning] bag={context['bag_name']} outer={outer_country} rung={rung_id} "
            f"trial={len(history)}/{tuning_cfg['n_trials']} "
            f"selection_objective={row['selection_objective']} "
            f"selection_value={row['objective_value']:.6f} best_value={best_so_far:.6f} "
            f"inner_global_r2={row['inner_global_oof_r2']:.6f} "
            f"inner_country_balanced_r2={row['inner_country_balanced_oof_r2']:.6f} "
            f"inner_country_r2_p25={row['inner_country_r2_p25']:.6f} "
            f"mean_country_mse={row['mean_country_mse']:.6f} "
            f"elapsed_sec={row['elapsed_sec']:.1f} "
            f"gp_warnings={row['optimizer_warning_count']} fit_warnings={row['fit_warning_count']} "
            f"params={json.dumps(params, sort_keys=True)}"
        )
        print(message, flush=True)
        if progress_log_path is not None:
            append_progress_line(progress_log_path, message)
    valid = [row for row in history if np.isfinite(float(row["objective_value"]))]
    if not valid:
        raise RuntimeError(f"No successful nested-LOCO trial for {context['bag_name']}/{outer_country}/{rung_id}")
    if str(tuning_cfg.get("optimization_direction", "maximize")) == "minimize":
        best = min(valid, key=lambda row: (float(row["objective_value"]), int(row["trial_index"])))
    else:
        best = max(valid, key=lambda row: (float(row["objective_value"]), -int(row["trial_index"])))
    selected = {
        "bag_target": context["bag_name"], "outer_country": outer_country, "rung_id": rung_id,
        "params": dict(best["params"]),
        "selection_objective": str(best["selection_objective"]),
        "selection_objective_value": float(best["objective_value"]),
        "inner_global_oof_r2": float(best["inner_global_oof_r2"]),
        "inner_country_balanced_oof_r2": float(best["inner_country_balanced_oof_r2"]),
        "inner_country_r2_p25": float(best["inner_country_r2_p25"]),
        "inner_country_r2_median": float(best["inner_country_r2_median"]),
        "mean_country_mse": float(best["mean_country_mse"]),
        "median_country_mse": float(best["median_country_mse"]),
            "country_r2_json": str(best["country_r2_json"]),
        "country_mse_json": str(best["country_mse_json"]),
        "country_feature_r2_json": str(best["country_feature_r2_json"]),
        "country_feature_mse_json": str(best["country_feature_mse_json"]),
        "feature_scope": str(tuning_cfg.get("feature_scope", "full_exposome")),
        "is_manual_incumbent": bool(best["is_manual_incumbent"]),
        "selected_trial_index": int(best["trial_index"]), "best_iteration_median": best["best_iteration_median"],
        "n_trials": len(history), "seed": unit_seed,
    }
    diagnostics = [
        {
            "bag_target": context["bag_name"], "outer_country": outer_country, "rung_id": rung_id,
            **{key: value for key, value in row.items() if key != "params"},
            "params_json": json.dumps(row["params"], sort_keys=True),
            **{f"param_{key}": value for key, value in row["params"].items()},
        }
        for row in history
    ]
    return selected, diagnostics


def write_tuning_artifact(output_dir: Path, selected: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> Path:
    """Write the compact downstream resolver contract and diagnostic table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": ARTIFACT_SCHEMA_VERSION, "provenance": dict(provenance), "selected": list(selected)}
    selected_path = output_dir / "selected_xgb_configs.json"
    _atomic_json(selected_path, payload)
    diagnostics_path = output_dir / "trial_diagnostics.parquet"
    diagnostics_tmp = diagnostics_path.with_suffix(".parquet.tmp")
    write_parquet(pd.DataFrame(diagnostics), diagnostics_tmp)
    diagnostics_tmp.replace(diagnostics_path)
    _atomic_json(output_dir / "manifest.json", {"selected_sha256": sha256(selected_path.read_bytes()).hexdigest(), **dict(provenance)})
    return selected_path


def resolve_tuned_fold_configs(
    artifact_path: Path | str | None, bag_target: str, rung_id: str, base_xgb_cfg: Mapping[str, Any],
    countries: Sequence[str], *, strict: bool = False,
) -> dict[str, dict[str, Any]] | None:
    """Load selected overrides, validating fixed rung depth and complete coverage."""
    if not artifact_path:
        return None
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported XGB tuning artifact schema")
    selected = {(str(row["outer_country"])): row for row in data.get("selected", [])
                if str(row.get("bag_target")) == str(bag_target) and str(row.get("rung_id")) == str(rung_id)}
    global_row = selected.get("__global__")
    missing = [] if global_row is not None else sorted(set(map(str, countries)).difference(selected))
    if strict and missing:
        raise ValueError(f"Tuning artifact lacks {bag_target}/{rung_id} countries: {missing}")
    if not selected:
        return None
    resolved: dict[str, dict[str, Any]] = {}
    for country in countries:
        row = global_row or selected.get(str(country))
        if row is None:
            continue
        params = dict(row.get("params", {}))
        unexpected = set(params).difference(TUNABLE_KEYS)
        if unexpected:
            raise ValueError(f"Tuning artifact contains non-tunable keys: {sorted(unexpected)}")
        resolved[str(country)] = {**base_xgb_cfg, **params}
    return resolved
