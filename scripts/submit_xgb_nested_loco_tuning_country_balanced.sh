#!/bin/bash
# Submit the two production country-balanced-R² nested-LOCO tuning jobs:
# current three-country policy and historical five-country policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XGB_TUNING_SELECTION_OBJECTIVE="country_balanced_r2"

bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" historical_a
bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" a
