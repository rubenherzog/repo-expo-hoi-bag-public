#!/bin/bash
# Continue the two completed global Optuna-GP studies to 1,000 total trials.
# The first 200 observations are restored from the immutable parent checkpoints;
# only trials 201--1,000 are fitted.  Running this script submits the jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export N_JOBS="${N_JOBS:-40}"
export N_TRIALS="${N_TRIALS:-1000}"
export XGB_TUNING_FITTING_MODE="global_loco_country_mse"
export XGB_TUNING_SELECTION_OBJECTIVE="mean_country_mse"
export XGB_TUNING_OPTIMIZER="optuna_gp"

export XGB_TUNING_PARENT_RUN_SIGNATURE="historical5_globalmse_optuna_gp_20260911T111017Z"
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" historical_a

export XGB_TUNING_PARENT_RUN_SIGNATURE="current3_globalmse_optuna_gp_20260911T111017Z"
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" a
