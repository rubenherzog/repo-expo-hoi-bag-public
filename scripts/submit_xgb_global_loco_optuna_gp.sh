#!/bin/bash
# Prepare two immutable production submissions for the global LOCO HPO:
# 200 Optuna-GP trials, one global configuration per BAG × depth, and the
# two versioned country policies.  It is an explicit launcher: merely adding
# this file does not submit anything; execution is left to the user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_JOBS="${N_JOBS:-40}"
export N_TRIALS="${N_TRIALS:-200}"
export XGB_TUNING_FITTING_MODE="global_loco_country_mse"
export XGB_TUNING_SELECTION_OBJECTIVE="mean_country_mse"
export XGB_TUNING_OPTIMIZER="optuna_gp"

# Each invocation creates a fresh, informative immutable run id.  Submit the
# historical and current policies separately so their scheduler jobs and
# manifests remain unambiguous.
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" historical_a
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" a
