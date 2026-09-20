#!/bin/bash
# Submit the four 1,000-trial global-LOCO HPO studies missing after full-exposome HPO:
# baseline and all canonical single exposures, each under current3 and historical5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export N_JOBS="${N_JOBS:-40}"
export N_TRIALS="${N_TRIALS:-1000}"
export XGB_TUNING_FITTING_MODE="global_loco_country_mse"
export XGB_TUNING_SELECTION_OBJECTIVE="mean_country_mse"
export XGB_TUNING_OPTIMIZER="optuna_gp"
export XGB_TUNING_SINGLE_PARALLEL_AXIS="country_exposure"
# The fixed vector remains an external benchmark only, never an official HPO trial.
export XGB_TUNING_INCLUDE_MANUAL_INCUMBENT=0

submit_scope() {
    local scope="$1" variant="$2"
    export XGB_TUNING_FEATURE_SCOPE="$scope"
    # Single exposure is substantially more expensive (country × 63 models per
    # trial); use a conservative Slurm limit while allowing user override.
    if [ "$scope" = "single_exposure" ]; then
        export XGB_TUNING_TIME_LIMIT="${SINGLE_TIME_LIMIT:-3-00:00}"
    else
        export XGB_TUNING_TIME_LIMIT="${BASELINE_TIME_LIMIT:-12:00}"
    fi
    bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" "$variant"
}

submit_scope baseline historical_a
submit_scope baseline a
submit_scope single_exposure historical_a
submit_scope single_exposure a
