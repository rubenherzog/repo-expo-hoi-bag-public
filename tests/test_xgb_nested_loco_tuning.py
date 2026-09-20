from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


CORE = Path(__file__).resolve().parents[1] / "src" / "repo_expo_hoi_bag" / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import xgb_nested_loco_tuning as tuning_module
from xgb_loco_engine import _build_base_fold_mats, baseline_cache_signature
from xgb_nested_loco_tuning import (
    TUNABLE_KEYS,
    _feature_sets,
    _evaluate_trial,
    _decode_point,
    _next_point,
    append_progress_line,
    build_global_loco_fold_designs,
    build_nested_fold_designs,
    country_balanced_r2,
    create_tuning_run_paths,
    inner_fold_fit_seed,
    load_continuation_history,
    resolve_tuned_fold_configs,
    selection_objective_value,
    select_trial_cap_or_practical_convergence,
    tune_unit,
)


def _context() -> dict:
    countries = np.repeat(np.array(["A", "B", "C", "D"]), 3)
    n_rows = len(countries)
    return {
        "bag_name": "structural",
        "country": countries,
        "countries": ["A", "B", "C", "D"],
        "age": np.arange(n_rows, dtype=float),
        "year": np.full(n_rows, 2020.0),
        "sex": np.repeat(["F", "M"], n_rows // 2),
        "diag": np.repeat("CN", n_rows),
        "y": np.arange(n_rows, dtype=float),
        "X_exp": np.zeros((n_rows, 2), dtype=np.float32),
        "analysis_cfg": {"min_n_obs_for_metrics": 2},
        "train_idx_by_country": {c: np.flatnonzero(countries != c) for c in "ABCD"},
        "test_idx_by_country": {c: np.flatnonzero(countries == c) for c in "ABCD"},
    }


def test_nested_designs_exclude_outer_and_scored_country_from_fit_and_validation() -> None:
    context = _context()
    designs = build_nested_fold_designs(context, "A", {"include_year": True}, {"val_country_frac": 0.5}, 7)
    assert set(designs) == {"B", "C", "D"}
    for inner_country, design in designs.items():
        used = np.concatenate([design["tr_inner_idx"], design["val_idx"]])
        used_countries = set(context["country"][used])
        assert "A" not in used_countries
        assert inner_country not in used_countries
        assert set(context["country"][design["test_idx"]]) == {inner_country}


def test_global_loco_designs_exclude_each_scored_country_from_fit_and_validation() -> None:
    context = _context()
    designs = build_global_loco_fold_designs(context, {"include_year": True}, {"val_country_frac": 0.5}, 7)
    assert set(designs) == set(context["countries"])
    for held_out, design in designs.items():
        fitted = np.concatenate([design["tr_inner_idx"], design["val_idx"]])
        assert held_out not in set(context["country"][fitted])
        assert set(context["country"][design["test_idx"]]) == {held_out}


def test_fold_config_resolver_validates_strict_coverage_and_preserves_fixed_depth(tmp_path: Path) -> None:
    artifact = tmp_path / "selected.json"
    artifact.write_text(json.dumps({
        "schema_version": 1,
        "selected": [{
            "bag_target": "structural", "outer_country": "A", "rung_id": "xgb_tree_d1",
            "params": {"learning_rate": 0.05, "min_child_weight": 7},
        }],
    }))
    base = {"max_depth": 1, "learning_rate": 0.03}
    resolved = resolve_tuned_fold_configs(artifact, "structural", "xgb_tree_d1", base, ["A", "B"])
    assert resolved == {"A": {"max_depth": 1, "learning_rate": 0.05, "min_child_weight": 7}}
    try:
        resolve_tuned_fold_configs(artifact, "structural", "xgb_tree_d1", base, ["A", "B"], strict=True)
    except ValueError as exc:
        assert "B" in str(exc)
    else:
        raise AssertionError("strict resolver accepted incomplete artifact")


def test_global_tuning_config_is_reused_for_every_loco_country(tmp_path: Path) -> None:
    artifact = tmp_path / "selected.json"
    artifact.write_text(json.dumps({
        "schema_version": 1,
        "selected": [{
            "bag_target": "functional", "outer_country": "__global__", "rung_id": "xgb_tree_d3",
            "params": {"learning_rate": 0.05, "min_child_weight": 7},
        }],
    }))
    base = {"max_depth": 3, "learning_rate": 0.03}
    resolved = resolve_tuned_fold_configs(
        artifact, "functional", "xgb_tree_d3", base, ["A", "B", "C"], strict=True
    )
    assert resolved == {
        country: {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 7}
        for country in ["A", "B", "C"]
    }


def test_cap_or_practical_convergence_selects_cap_or_early_stable_incumbent() -> None:
    history = [
        {"trial_index": index, "objective_value": 10.0 - (1.0 if index == 25 else 0.0)}
        for index in range(500)
    ]
    best, decision = select_trial_cap_or_practical_convergence(
        history, max_trials=500, minimum_completed_trials=350, lookback_trials=300,
        minimum_incumbent_improvement_mse=0.1,
    )
    assert best["trial_index"] == 25
    assert decision["stop_reason"] == "max_trials_cap"
    best, decision = select_trial_cap_or_practical_convergence(
        history[:382], max_trials=500, minimum_completed_trials=350, lookback_trials=300,
        minimum_incumbent_improvement_mse=0.1,
    )
    assert best["trial_index"] == 25
    assert decision["stop_reason"] == "practical_convergence"


def test_continuation_history_accepts_only_matching_parent_except_trial_budget(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.json"
    parent_payload = {
        "schema": 1,
        "bag": "functional",
        "outer_country": "__global__",
        "rung_id": "xgb_tree_d3",
        "base_xgb_cfg": {"max_depth": 3},
        "early_stop_cfg": {"rounds": 10},
        "tuning_cfg": {"n_trials": 200, "optimizer": "optuna_gp", "seed": 1},
        "countries": ["A", "B"],
        "x_shape": [8, 2],
    }
    parent_path.write_text(json.dumps({
        "signature": "parent", "signature_payload": parent_payload,
        "trials": [{"trial_index": 0, "objective_value": 1.0, "params": {"gamma": 0}}],
    }))
    child_payload = {
        **parent_payload,
        "tuning_cfg": {
            "n_trials": 1000, "optimizer": "optuna_gp", "seed": 1,
            "parent_run_signature": "parent_run",
        },
    }
    restored = load_continuation_history(parent_path, child_payload)
    assert restored[0]["trial_index"] == 0

    incompatible = {**child_payload, "rung_id": "xgb_tree_d2"}
    try:
        load_continuation_history(parent_path, incompatible)
    except ValueError as exc:
        assert "rung_id" in str(exc)
    else:
        raise AssertionError("incompatible continuation parent was accepted")


def test_baseline_signature_changes_for_fold_specific_configuration() -> None:
    context = _context()
    designs = {
        country: _build_base_fold_mats(context, country, {"include_year": True}, {"val_country_frac": 0.5}, 7)
        for country in context["countries"]
    }
    # Signature only requires deterministic fold designs; the selected mapping is
    # independently represented so cache reuse cannot cross configurations.
    fixed = baseline_cache_signature(context, designs, {"max_depth": 1})
    tuned = baseline_cache_signature(context, designs, {"max_depth": 1}, {"B": {"learning_rate": 0.05}})
    assert fixed != tuned


def test_tuning_paths_keep_delivery_local_and_checkpoints_external(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "outputs").mkdir(parents=True)
    runtime = tmp_path / "runtime"
    signature, output_dir, checkpoint_dir = create_tuning_run_paths(
        checkout, runtime, "a" * 64, run_id="controlled-run"
    )
    assert signature == "controlled-run"
    assert output_dir == checkout / "outputs" / "xgb_nested_loco_tuning" / signature
    assert checkpoint_dir == runtime / "work" / "xgb_nested_loco_tuning" / signature
    append_progress_line(output_dir / "tuning.log", "[tuning] bag=functional outer=Spain rung=xgb_tree_d3 trial=1/32")
    assert "bag=functional" in (output_dir / "tuning.log").read_text()
    try:
        create_tuning_run_paths(checkout, runtime, "a" * 64, run_id="controlled-run")
    except FileExistsError:
        pass
    else:
        raise AssertionError("A prior tuning delivery directory was accepted for overwrite")


def test_reg_alpha_zero_is_a_mandatory_initial_trial_candidate() -> None:
    search_space = {
        key: {"kind": "float", "low": 0.1, "high": 1.0}
        for key in TUNABLE_KEYS
    }
    search_space["reg_alpha"] = {"kind": "log_or_zero", "low": 0.01, "high": 5.0}
    point, _ = _next_point([], {"n_initial_trials": 3, "search_space": search_space}, 17)
    assert point[TUNABLE_KEYS.index("reg_alpha")] == 0.0
    assert _decode_point(point, search_space)["reg_alpha"] == 0.0


def test_optuna_first_nonmanual_trial_enqueues_zero_l1(monkeypatch, tmp_path: Path) -> None:
    class Study:
        def __init__(self): self.enqueued = []
        def enqueue_trial(self, params): self.enqueued.append(params)
        def ask(self): return object()
        def tell(self, *_args, **_kwargs): pass
    study = Study()
    monkeypatch.setattr(tuning_module, "_restore_optuna_study", lambda *_args: study)
    monkeypatch.setattr(tuning_module, "_optuna_suggest", lambda *_args: {key: 1.0 for key in TUNABLE_KEYS})
    monkeypatch.setattr(tuning_module, "_evaluate_trial", lambda *_args, **_kwargs: {
        "objective_value": 1.0, "selection_objective": "mean_country_mse", "inner_global_oof_r2": 0.0,
        "inner_country_balanced_oof_r2": 0.0, "inner_country_r2_p25": 0.0, "inner_country_r2_median": 0.0,
        "mean_country_mse": 1.0, "median_country_mse": 1.0, "country_r2_json": "{}", "country_mse_json": "{}",
        "country_feature_r2_json": "{}", "country_feature_mse_json": "{}", "n_scored": 1, "inner_folds_n": 1,
        "inner_folds_succeeded": 1, "complete_country_coverage": True, "best_iteration_median": 1.0,
        "failures": "", "fit_warning_count": 0, "fit_warnings": "",
    })
    cfg = {"seed": 1, "n_trials": 1, "n_initial_trials": 1, "sobol_candidates": 2, "optimizer": "optuna_gp",
           "selection_objective": "mean_country_mse", "optimization_direction": "minimize", "search_space": {}}
    tune_unit(_context(), "__global__", "xgb_tree_d3", {key: 1.0 for key in TUNABLE_KEYS}, {}, {}, cfg, tmp_path / "x.json", nested_designs={"A": {}})
    assert study.enqueued == [{"reg_alpha": 0.0}]


def test_inner_fold_fit_seed_is_independent_of_trial_index() -> None:
    unit_seed = 12345
    # The trial index is intentionally absent from this API: every candidate
    # must experience the same XGBoost stochastic draw in a given inner fold.
    assert inner_fold_fit_seed(unit_seed, "Spain", 2) == inner_fold_fit_seed(unit_seed, "Spain", 2)
    assert inner_fold_fit_seed(unit_seed, "Spain", 2) != inner_fold_fit_seed(unit_seed, "Japan", 2)


def test_country_r2_p25_selection_objective_is_country_balanced() -> None:
    scores = [0.10, 0.20, 0.30, 0.90]
    assert np.isclose(selection_objective_value("country_r2_p25", 0.80, scores), 0.175)
    assert np.isclose(selection_objective_value("country_r2_mean", 0.80, scores), 0.375)
    assert np.isclose(selection_objective_value("pooled_global_r2", 0.80, scores), 0.80)


def test_country_balanced_r2_gives_each_country_equal_total_weight() -> None:
    y_true = [0.0, 0.0, 0.0, 10.0]
    y_pred = [0.0, 0.0, 0.0, 0.0]
    countries = ["A", "B", "B", "B"]
    # A receives weight 1 and each B subject weight 1/3. The weighted mean is
    # 5/3, weighted residual is 100/3 and weighted total is 250/9: R² = -0.2.
    assert np.isclose(country_balanced_r2(y_true, y_pred, countries), -0.2)
    assert np.isclose(
        selection_objective_value("country_balanced_r2", 0.8, [0.1, 0.2], -0.2), -0.2
    )


def test_mean_country_mse_selection_gives_every_country_one_vote() -> None:
    assert np.isclose(
        selection_objective_value("mean_country_mse", 0.8, [0.1, 0.2], country_mse=[1.0, 9.0]), 5.0
    )


def test_feature_scopes_are_canonical_and_single_exposure_uses_every_feature() -> None:
    context = _context()
    context["exposome_features"] = ["expo_a", "expo_b"]
    assert _feature_sets(context, "baseline") == [("__baseline__", [])]
    assert _feature_sets(context, "full_exposome") == [("__full_exposome__", [0, 1])]
    assert _feature_sets(context, "single_exposure") == [("expo_a", [0]), ("expo_b", [1])]


def test_domain_balanced_k10_scope_uses_its_fixed_configured_panel() -> None:
    context = _context()
    context["exposome_features"] = ["expo_a", "expo_b"]
    context["hpo_feature_sets"] = [
        {"id": "panel_01", "features": ["expo_a", "expo_b"]},
        {"id": "panel_02", "features": ["expo_b", "expo_a"]},
    ]
    assert _feature_sets(context, "domain_balanced_k10") == [
        ("panel_01", [0, 1]), ("panel_02", [1, 0]),
    ]


def test_single_exposure_trial_schedules_country_feature_pairs_and_aggregates_by_country(monkeypatch) -> None:
    context = _context()
    context["exposome_features"] = ["expo_a", "expo_b"]
    designs = {country: {"test_idx": np.flatnonzero(context["country"] == country)} for country in ["A", "B"]}
    calls = []

    def fake_fold(context, country, design, fold_index, *_args, **_kwargs):
        member = _args[-1][0][0]
        calls.append((country, member))
        idx = design["test_idx"]
        prediction = np.asarray(context["y"])[idx]
        return {
            "country": country,
            "prediction": __import__("pandas").DataFrame({"y_true": prediction, "y_pred": prediction, "country": country}),
            "r2": 1.0, "mse": 1.0 if member == "expo_a" else 3.0, "best_iteration": 4,
            "fit_warnings": [], "failure": "", "feature_r2": {member: 1.0}, "feature_mse": {member: 1.0 if member == "expo_a" else 3.0},
        }

    monkeypatch.setattr(tuning_module, "_evaluate_trial_fold", fake_fold)
    result = _evaluate_trial(context, designs, {}, {}, 7, "mean_country_mse", fold_n_jobs=1, feature_scope="single_exposure")
    assert sorted(calls) == [("A", "expo_a"), ("A", "expo_b"), ("B", "expo_a"), ("B", "expo_b")]
    assert np.isclose(result["mean_country_mse"], 2.0)


def test_global_mode_starts_with_the_current_manual_parameter_vector(tmp_path: Path, monkeypatch) -> None:
    manual = {
        "learning_rate": 0.03, "min_child_weight": 15, "subsample": 0.7,
        "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 0.5,
    }

    def fake_evaluate(*_args, **_kwargs):
        return {
            "objective_value": 2.5, "selection_objective": "mean_country_mse",
            "inner_global_oof_r2": 0.1, "inner_country_balanced_oof_r2": 0.2,
            "inner_country_r2_p25": 0.0, "inner_country_r2_median": 0.1,
            "mean_country_mse": 2.5, "median_country_mse": 2.5,
            "country_r2_json": '{"A": 0.1}', "country_mse_json": '{"A": 2.5}',
            "country_feature_r2_json": '{"A": {"__full_exposome__": 0.1}}',
            "country_feature_mse_json": '{"A": {"__full_exposome__": 2.5}}',
            "n_scored": 3, "inner_folds_n": 1, "inner_folds_succeeded": 1,
            "complete_country_coverage": True, "best_iteration_median": 10.0,
            "failures": "", "fit_warning_count": 0, "fit_warnings": "",
        }

    monkeypatch.setattr(tuning_module, "_evaluate_trial", fake_evaluate)
    cfg = {
        "seed": 7, "n_trials": 1, "n_initial_trials": 1, "sobol_candidates": 2,
        "selection_objective": "mean_country_mse", "optimization_direction": "minimize",
        "include_manual_incumbent": True,
        "search_space": {
            "learning_rate": {"kind": "log", "low": 0.01, "high": 0.10},
            "min_child_weight": {"kind": "int", "low": 5, "high": 30},
            "subsample": {"kind": "float", "low": 0.50, "high": 1.00},
            "colsample_bytree": {"kind": "float", "low": 0.50, "high": 1.00},
            "reg_alpha": {"kind": "log_or_zero", "low": 0.01, "high": 5.0},
            "reg_lambda": {"kind": "log", "low": 0.10, "high": 10.0},
            "gamma": {"kind": "float", "low": 0.0, "high": 5.0},
        },
    }
    selected, diagnostics = tune_unit(
        _context(), "__global__", "xgb_tree_d3", {**manual, "max_depth": 3},
        {"include_year": True}, {"val_country_frac": 0.5}, cfg,
        tmp_path / "global.json", nested_designs={"A": {}},
    )
    assert selected["params"] == manual
    assert selected["is_manual_incumbent"] is True
    assert diagnostics[0]["is_manual_incumbent"] is True
