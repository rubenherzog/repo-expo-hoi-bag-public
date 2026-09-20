from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "src" / "repo_expo_hoi_bag" / "stages"
CORE = ROOT / "src" / "repo_expo_hoi_bag" / "core"
for path in (STAGES, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_xgb_nested_loco_tuning import (
    _apply_runtime_overrides,
    _load_domain_balanced_k10_panel,
    _resolve_global_parallelism,
)
from repo_expo_hoi_bag.analysis.xgb_tuned_top50 import load_comparison_bag_policy
from oinfo_bag_ladder.config import XGB_TUNING_CFG


def test_cluster_launcher_is_syntactically_valid_and_requests_one_cpu_job() -> None:
    submit = ROOT / "scripts" / "submit_xgb_nested_loco_tuning.sh"
    runner = ROOT / "scripts" / "jobs" / "run_xgb_nested_loco_tuning.sh"
    historical = ROOT / "scripts" / "submit_xgb_nested_loco_tuning_historical5.sh"
    balanced = ROOT / "scripts" / "submit_xgb_nested_loco_tuning_country_balanced.sh"
    global_gp = ROOT / "scripts" / "submit_xgb_global_loco_optuna_gp.sh"
    extension = ROOT / "scripts" / "submit_xgb_global_loco_optuna_gp_extend1000.sh"
    extremes = ROOT / "scripts" / "submit_xgb_hpo_extremes_optuna_gp_1000.sh"
    domain_k10 = ROOT / "scripts" / "submit_xgb_hpo_domaink10_optuna_gp_1000.sh"
    for script in (submit, runner, historical, balanced, global_gp, extension, extremes, domain_k10):
        subprocess.run(["bash", "-n", str(script)], check=True)

    assert "run -t \"${TIME_LIMIT}\" -j \"${JOB_NAME}\" -c \"${N_JOBS}\" -m 40" in submit.read_text()
    runner_text = runner.read_text()
    assert 'export XGB_TUNING_BAGS="structural,functional"' in runner_text
    assert 'export XGB_TUNING_RUNGS="xgb_tree_d1,xgb_tree_d2,xgb_tree_d3"' in runner_text
    assert 'export XGB_TUNING_N_TRIALS="${N_TRIALS}"' in runner_text
    assert 'export XGB_TUNING_VARIANT="${COUNTRY_VARIANT}"' in runner_text
    assert 'export XGB_TUNING_RUN_ID="${RUN_ID}"' in runner_text
    assert 'export XGB_TUNING_FEATURE_SCOPE="${FEATURE_SCOPE}"' in runner_text
    assert 'export XGB_TUNING_INCLUDE_MANUAL_INCUMBENT="${INCLUDE_MANUAL_INCUMBENT}"' in runner_text
    assert 'if [ "${PARENT_RUN_SIGNATURE}" = "-" ]; then' in runner_text
    assert 'historical_a)' in submit.read_text()
    assert 'RUN_ID="historical5${OBJECTIVE_LABEL}_${SUBMITTED_AT}"' in submit.read_text()
    assert 'submit_xgb_nested_loco_tuning.sh" historical_a' in historical.read_text()
    assert 'XGB_TUNING_SELECTION_OBJECTIVE="country_balanced_r2"' in balanced.read_text()
    global_gp_text = global_gp.read_text()
    assert 'export N_JOBS="${N_JOBS:-40}"' in global_gp_text
    assert 'export N_TRIALS="${N_TRIALS:-200}"' in global_gp_text
    assert 'export XGB_TUNING_FITTING_MODE="global_loco_country_mse"' in global_gp_text
    assert 'export XGB_TUNING_SELECTION_OBJECTIVE="mean_country_mse"' in global_gp_text
    assert 'export XGB_TUNING_OPTIMIZER="optuna_gp"' in global_gp_text
    extension_text = extension.read_text()
    assert 'export N_TRIALS="${N_TRIALS:-1000}"' in extension_text
    assert 'historical5_globalmse_optuna_gp_20260911T111017Z' in extension_text
    assert 'current3_globalmse_optuna_gp_20260911T111017Z' in extension_text
    extremes_text = extremes.read_text()
    assert 'export N_TRIALS="${N_TRIALS:-1000}"' in extremes_text
    assert 'submit_scope baseline historical_a' in extremes_text
    assert 'submit_scope single_exposure a' in extremes_text
    assert 'XGB_TUNING_INCLUDE_MANUAL_INCUMBENT=0' in extremes_text
    assert 'XGB_TUNING_SINGLE_PARALLEL_AXIS="country_exposure"' in extremes_text
    domain_k10_text = domain_k10.read_text()
    assert 'export N_JOBS="${N_JOBS:-40}"' in domain_k10_text
    assert 'export N_TRIALS="${N_TRIALS:-1000}"' in domain_k10_text
    assert 'export XGB_TUNING_FEATURE_SCOPE="domain_balanced_k10"' in domain_k10_text
    assert 'export XGB_TUNING_INCLUDE_MANUAL_INCUMBENT=0' in domain_k10_text
    assert 'submit_xgb_nested_loco_tuning.sh" historical_a' in domain_k10_text
    assert 'submit_xgb_nested_loco_tuning.sh" a' in domain_k10_text
    submit_text = submit.read_text()
    assert 'PARENT_RUN_ARGUMENT="${PARENT_RUN_SIGNATURE:--}"' in submit_text
    assert '"${PARENT_RUN_ARGUMENT}" "${FEATURE_SCOPE}" "${INCLUDE_MANUAL_INCUMBENT}"' in submit_text
    assert 'baseline|single_exposure|full_exposome|domain_balanced_k10)' in submit_text
    assert 'OBJECTIVE_LABEL="_domaink10_gp${N_TRIALS}"' in submit_text


def test_runtime_overrides_allow_the_cluster_budget_without_changing_default() -> None:
    base = {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10}
    resolved = _apply_runtime_overrides(
        base, {"XGB_TUNING_N_JOBS": "42", "XGB_TUNING_N_TRIALS": "200"}
    )
    assert resolved == {"n_jobs": 42, "n_trials": 200, "n_initial_trials": 10}
    assert base == {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10}


def test_runtime_override_allows_a_one_trial_fixed_reference() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10},
        {"XGB_TUNING_N_TRIALS": "1", "XGB_TUNING_N_INITIAL_TRIALS": "1"},
    )
    assert resolved["n_trials"] == resolved["n_initial_trials"] == 1


def test_runtime_override_allows_explicit_country_balanced_selection() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10, "selection_objective": "country_r2_p25"},
        {"XGB_TUNING_SELECTION_OBJECTIVE": "country_balanced_r2"},
    )
    assert resolved["selection_objective"] == "country_balanced_r2"


def test_global_country_mse_mode_has_its_fixed_objective_and_manual_incumbent() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10, "selection_objective": "country_r2_p25",
         "fitting_mode": "nested_per_outer_country"},
        {"XGB_TUNING_FITTING_MODE": "global_loco_country_mse"},
    )
    assert resolved["selection_objective"] == "mean_country_mse"
    assert resolved["optimization_direction"] == "minimize"
    assert resolved["include_manual_incumbent"] is True


def test_runtime_override_supports_a_canonical_single_exposure_panel_without_manual_incumbent() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10, "fitting_mode": "global_loco_country_mse"},
        {
            "XGB_TUNING_FEATURE_SCOPE": "single_exposure",
            "XGB_TUNING_INCLUDE_MANUAL_INCUMBENT": "0",
        },
    )
    assert resolved["feature_scope"] == "single_exposure"
    assert resolved["include_manual_incumbent"] is False


def test_runtime_override_selects_an_explicit_single_parallel_axis() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10},
        {"XGB_TUNING_SINGLE_PARALLEL_AXIS": "country_exposure"},
    )
    assert resolved["single_parallel_axis"] == "country_exposure"


def test_global_continuation_records_a_validated_parent_signature() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10, "fitting_mode": "global_loco_country_mse"},
        {"XGB_TUNING_PARENT_RUN_SIGNATURE": "current3_globalmse_optuna_gp_20260911T111017Z"},
    )
    assert resolved["parent_run_signature"] == "current3_globalmse_optuna_gp_20260911T111017Z"


def test_global_parallelism_spends_the_40_core_budget_without_oversubscription() -> None:
    cfg = {"n_jobs": 40}
    _resolve_global_parallelism(cfg, n_units=6, environ={})
    plan = cfg["global_unit_parallelism"]
    assert [item["core_budget"] for item in plan] == [7, 7, 7, 7, 6, 6]
    assert sum(item["fold_n_jobs"] * item["xgb_nthread"] for item in plan) == 40


def test_single_exposure_parallelism_uses_one_xgb_thread_per_independent_fit() -> None:
    cfg = {"n_jobs": 20, "feature_scope": "single_exposure"}
    _resolve_global_parallelism(cfg, n_units=2, environ={})
    assert cfg["global_unit_parallelism"] == [
        {"core_budget": 10, "fold_n_jobs": 10, "xgb_nthread": 1},
        {"core_budget": 10, "fold_n_jobs": 10, "xgb_nthread": 1},
    ]


def test_domain_balanced_k10_panel_is_complete_and_uses_one_thread_per_fit() -> None:
    features = (ROOT / "data" / "metadata" / "exposome_feature_names.csv").read_text().splitlines()
    panel, provenance = _load_domain_balanced_k10_panel(ROOT, features)
    assert len(panel) == 10
    assert all(len(item["features"]) == 10 for item in panel)
    assert provenance["definition"]["sampling_seed"] == 20260911
    cfg = {"n_jobs": 20, "feature_scope": "domain_balanced_k10"}
    _resolve_global_parallelism(cfg, n_units=2, environ={})
    assert all(item["xgb_nthread"] == 1 for item in cfg["global_unit_parallelism"])


def test_runtime_override_allows_optuna_tpe() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10},
        {"XGB_TUNING_OPTIMIZER": "optuna_tpe"},
    )
    assert resolved["optimizer"] == "optuna_tpe"


def test_runtime_override_allows_optuna_gp() -> None:
    resolved = _apply_runtime_overrides(
        {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10},
        {"XGB_TUNING_OPTIMIZER": "optuna_gp"},
    )
    assert resolved["optimizer"] == "optuna_gp"


def test_production_tuning_default_uses_the_validated_country_p25_objective() -> None:
    assert XGB_TUNING_CFG["selection_objective"] == "country_r2_p25"


def test_hpo_search_space_expands_the_pilot_boundaries_without_negative_gamma() -> None:
    space = XGB_TUNING_CFG["search_space"]
    assert space["learning_rate"]["low"] == 0.003
    assert space["min_child_weight"]["low"] == 1
    assert space["subsample"]["low"] == 0.30
    assert space["reg_alpha"]["high"] == 10.0
    assert space["gamma"]["low"] == 0.0


def test_comparison_policy_is_versioned() -> None:
    policy = load_comparison_bag_policy(ROOT / "config" / "country_exclusions.yaml", "historical_a")
    assert policy["exclude_countries"] == ["France", "Italy", "Egypt", "Greece", "Poland"]


@pytest.mark.parametrize("value", ["0", "nine"])
def test_runtime_trial_override_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="XGB_TUNING_N_TRIALS"):
        _apply_runtime_overrides(
            {"n_jobs": 4, "n_trials": 64, "n_initial_trials": 10},
            {"XGB_TUNING_N_TRIALS": value},
        )
