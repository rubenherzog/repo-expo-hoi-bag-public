#!/bin/bash
# Prepare the two immutable 1,000-trial global-LOCO HPO submissions for the
# fixed, outcome-independent, domain-balanced k=10 panel: current3 and the
# five-country historical reproduction. The user submits these jobs explicitly
# by running this script; defining it does not submit anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_JOBS="${N_JOBS:-40}"
export N_TRIALS="${N_TRIALS:-1000}"
export XGB_TUNING_TIME_LIMIT="${DOMAIN_K10_TIME_LIMIT:-3-00:00}"
export XGB_TUNING_FITTING_MODE="global_loco_country_mse"
export XGB_TUNING_SELECTION_OBJECTIVE="mean_country_mse"
export XGB_TUNING_OPTIMIZER="optuna_gp"
export XGB_TUNING_FEATURE_SCOPE="domain_balanced_k10"
export XGB_TUNING_SINGLE_PARALLEL_AXIS="country_exposure"
# The manually chosen vector is a benchmark, not an official HPO proposal.
export XGB_TUNING_INCLUDE_MANUAL_INCUMBENT=0

bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" historical_a
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" a
