from __future__ import annotations

import os
from pathlib import Path


PROJECT_CODE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("V3_OINFO_OUTPUT_ROOT", "outputs/variant_a"))
FAMILIES_ROOT = OUTPUT_ROOT / "families"
LEGACY_CANONICAL_ROOT = OUTPUT_ROOT / "canonical"
LEGACY_FIGURES_ROOT = OUTPUT_ROOT / "figures"
REFERENCE_SNAPSHOT_ROOT = Path(os.environ.get(
    "V3_OINFO_REFERENCE_SNAPSHOT_ROOT",
    "outputs_oinfo_bag_ladder_snapshots/pre_fullrun_example_20260319_182109",
))
REFERENCE_SNAPSHOT_CANONICAL_ROOT = REFERENCE_SNAPSHOT_ROOT / "canonical"

GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", "outputs/greedy"))

ANALYSIS_FAMILIES = [
    "pooled_oinfo_ladder",
    "single_dx_oinfo_ladder",
    "single_dx_portability",
    "cn_normative_transfer",
]
DEFAULT_ANALYSIS_FAMILY = "pooled_oinfo_ladder"

DATA_PATHS = {
    "input_csv": "data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
    "greedy_topk_csv": str(GREEDY_ROOT / "greedy_topk_by_objective_order.csv"),
    "greedy_feature_names_csv": str(GREEDY_ROOT / "exposome_feature_names.csv"),
}

BAG_TARGETS = {
    "structural": "bag_struct_resolved",
    "functional": "bag_func_resolved",
    "combined": "bag_comb_resolved",
}
BAG_ORDER = ["structural", "functional", "combined"]
OBJECTIVE_ORDER = ["o_max", "o_min"]
DEFAULT_BAG_TARGET_MODE = "all"
DIAGNOSIS_ORDER = ["CN", "AD", "MCI", "FTD"]

ANALYSIS_CFG = {
    "objectives": ["o_max", "o_min"],
    "initial_order": 3,
    "max_order": 30,
    "top_k_per_order": 10,  # v3: top-10 per order → 48 orders × 10 × 2 objectives = 960 candidates total (480 per objective). v1 used 5 (480 total).
    "max_candidates_per_objective": 0,
    "max_models_total": 0,
    "objective_filter": [],
    "order_min": None,
    "order_max": None,
    "exclude_diagnosis": ["Other", "AFM"],
    "exclude_countries": ["North Macedonia", "New Zeland", "Belgium"],
    "include_sex": True,
    "include_diagnosis": True,
    "include_year": True,
    "min_n_obs": 80,
    "min_n_obs_for_metrics": 5,
}

CV_CFG = {
    "split_col": "country_clean",
    "min_country_size_test": 0,
    "unseen_diag_policy": "drop_test_rows",
}

PERF_CFG_OLS = {
    "n_jobs": 40,
    "backend": "loky",
    "blas_threads": 1,
    "chunk_size_models": 100,
    "enable_identity_prune": True,
    "enable_resume": False,
}

MATRIX_CFG = {"rcond": None}

PERF_CFG_XGB = {
    "n_jobs": 40,
    "backend": "loky",
    "blas_threads": 1,
    "chunk_size_models": 100,
    "enable_identity_prune": True,
}

BASE_XGB_CFG = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 15,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "gamma": 0.5,
    "n_estimators": 4000,
    "early_stopping_rounds": 100,
    "random_state": 20260304,
    "nthread": 1,
    "missing": None,
}

EARLY_STOP_CFG = {
    "val_country_frac": 0.20,
    "val_country_min": 1,
}

# Disabled by default: the canonical fixed-hyperparameter analysis remains the
# default scientific contract until a validated tuning artifact is supplied.
XGB_TUNING_CFG = {
    "enabled": False,
    "artifact_path": "",
    "strict_artifact": False,
    "bags": ["structural", "functional"],
    "n_trials": 64,
    "n_initial_trials": 10,
    "n_jobs": 4,
    "seed": 20260304,
    "sobol_candidates": 2048,
    # Preserve the existing nested-per-country procedure unless an explicitly
    # requested alternative fitting mode is supplied at the runtime boundary.
    "fitting_mode": "nested_per_outer_country",
    "optimizer": "custom_gp",
    # Canonical predictor panel for the HPO objective. Runtime may select the
    # baseline-only or all-single-exposure panels for controlled pilot studies.
    "feature_scope": "full_exposome",
    "include_zero_reg_alpha_trial": True,
    "single_parallel_axis": "country_exposure",
    # Hyperparameters are selected for country transfer, not pooled subjects:
    # each inner LOCO country contributes one R² and the lower quartile is the
    # robust trial objective selected by the local development comparison.
    "selection_objective": "country_r2_p25",
    "search_space": {
        # The domain-balanced k=10 pilot reached these former boundaries.
        # Broaden only the implicated directions while retaining conservative,
        # valid XGBoost domains for all HPO scopes.
        "learning_rate": {"kind": "log", "low": 0.003, "high": 0.10},
        "min_child_weight": {"kind": "int", "low": 1, "high": 30},
        "subsample": {"kind": "float", "low": 0.30, "high": 1.00},
        "colsample_bytree": {"kind": "float", "low": 0.50, "high": 1.00},
        # Zero is a valid no-L1-penalty case. The tuner evaluates it once for
        # every outer country before exploring positive values logarithmically.
        "reg_alpha": {"kind": "log_or_zero", "low": 0.01, "high": 10.0},
        "reg_lambda": {"kind": "log", "low": 0.10, "high": 10.0},
        # XGBoost gamma is a non-negative minimum-loss reduction.
        "gamma": {"kind": "float", "low": 0.0, "high": 5.0},
    },
}

TOP_TAIL_FRAC = 0.10
TOP_TAIL_MIN_N = 10
FRONTIER_N_BINS = 40
FRONTIER_MIN_BIN_SIZE = 25
FIGURE_DPI = 220


def family_root(analysis_family: str) -> Path:
    return FAMILIES_ROOT / str(analysis_family)


def experiments_root(analysis_family: str) -> Path:
    return family_root(analysis_family) / "experiments"


def canonical_root(analysis_family: str) -> Path:
    return family_root(analysis_family) / "canonical"


def per_experiment_canonical_root(analysis_family: str) -> Path:
    return canonical_root(analysis_family) / "per_experiment"


def figures_root(analysis_family: str) -> Path:
    return family_root(analysis_family) / "figures"


def experiment_dir(analysis_family: str, experiment_id: str) -> Path:
    return experiments_root(analysis_family) / experiment_id


def rung_dir(analysis_family: str, experiment_id: str, rung_id: str) -> Path:
    return experiment_dir(analysis_family, experiment_id) / rung_id


def per_experiment_canonical_dir(analysis_family: str, experiment_id: str) -> Path:
    return per_experiment_canonical_root(analysis_family) / experiment_id


# Backward-compatible pooled defaults used by the original V1 figure code.
EXPERIMENTS_ROOT = experiments_root(DEFAULT_ANALYSIS_FAMILY)
CANONICAL_ROOT = canonical_root(DEFAULT_ANALYSIS_FAMILY)
PER_EXPERIMENT_CANONICAL_ROOT = per_experiment_canonical_root(DEFAULT_ANALYSIS_FAMILY)
FIGURES_ROOT = figures_root(DEFAULT_ANALYSIS_FAMILY)


# ── V3 env var overrides ──────────────────────────────────────────────────────
# Set these env vars to redirect a run without editing this file.
# All are optional; unset (or empty string) → values above remain unchanged.
#
#   V3_OUTPUT_ROOT          override OUTPUT_ROOT (and all derived roots)
#   V3_EXCLUDE_COUNTRIES    comma-separated country names to exclude
#   V3_EXCLUDE_DIAGNOSIS    comma-separated diagnoses to exclude
#
# Used by run_full_analysis_v3.py to drive Variant A / B without forking config.
import os as _os

_v3_out = _os.environ.get("V3_OUTPUT_ROOT", "").strip()
if _v3_out:
    OUTPUT_ROOT    = Path(_v3_out)
    FAMILIES_ROOT  = OUTPUT_ROOT / "families"
    LEGACY_CANONICAL_ROOT = OUTPUT_ROOT / "canonical"
    LEGACY_FIGURES_ROOT   = OUTPUT_ROOT / "figures"
    EXPERIMENTS_ROOT          = experiments_root(DEFAULT_ANALYSIS_FAMILY)
    CANONICAL_ROOT            = canonical_root(DEFAULT_ANALYSIS_FAMILY)
    PER_EXPERIMENT_CANONICAL_ROOT = per_experiment_canonical_root(DEFAULT_ANALYSIS_FAMILY)
    FIGURES_ROOT              = figures_root(DEFAULT_ANALYSIS_FAMILY)

_v3_excl_c = _os.environ.get("V3_EXCLUDE_COUNTRIES", "").strip()
if _v3_excl_c:
    ANALYSIS_CFG["exclude_countries"] = [c.strip() for c in _v3_excl_c.split(",") if c.strip()]

_v3_excl_d = _os.environ.get("V3_EXCLUDE_DIAGNOSIS", "").strip()
if _v3_excl_d:
    ANALYSIS_CFG["exclude_diagnosis"] = [d.strip() for d in _v3_excl_d.split(",") if d.strip()]

_v3_topk = _os.environ.get("V3_TOP_K_PER_ORDER", "").strip()
if _v3_topk:
    ANALYSIS_CFG["top_k_per_order"] = int(_v3_topk)

# Candidate order range for evaluation. The runner's RUN_SETTINGS reads these as
# the --order-min/--order-max defaults; without an override the runner default
# order_max is only 9 (a debug value), so the canonical pipeline evaluates
# order_min=3..order_max=30. Set these to reproduce that (or to cap differently).
_v3_order_min = _os.environ.get("V3_ORDER_MIN", "").strip()
if _v3_order_min:
    ANALYSIS_CFG["order_min"] = int(_v3_order_min)

_v3_order_max = _os.environ.get("V3_ORDER_MAX", "").strip()
if _v3_order_max:
    ANALYSIS_CFG["order_max"] = int(_v3_order_max)

_v3_xgb_n_estimators = _os.environ.get("V3_XGB_N_ESTIMATORS", "").strip()
if _v3_xgb_n_estimators:
    BASE_XGB_CFG["n_estimators"] = int(_v3_xgb_n_estimators)

_v3_xgb_early_stopping = _os.environ.get("V3_XGB_EARLY_STOPPING_ROUNDS", "").strip()
if _v3_xgb_early_stopping:
    BASE_XGB_CFG["early_stopping_rounds"] = int(_v3_xgb_early_stopping)

del _os, _v3_out, _v3_excl_c, _v3_excl_d, _v3_topk, _v3_order_min, _v3_order_max, _v3_xgb_n_estimators, _v3_xgb_early_stopping
