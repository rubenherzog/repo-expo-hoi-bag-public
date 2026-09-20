#!/bin/bash
# Submit the paired fixed-versus-country-tuned functional XGB d3 top-50 check.
# Usage: bash scripts/submit_xgb_tuned_top50_comparison.sh <selected_xgb_configs.json>
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <outputs/xgb_nested_loco_tuning/<signature>/selected_xgb_configs.json>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}"
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/env.sh"
fi
REPRO_DATA_ROOT="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
TUNING_ARTIFACT="$1"
N_JOBS="${N_JOBS:-42}"
LOG_DIR="${SCHEDULER_LOG_DIR:-${REPRO_DATA_ROOT}/work/xgb_tuned_top50_comparison/scheduler_logs}"
mkdir -p "${LOG_DIR}"
run -t 12:00 -j xgb_tuned_top50 -c "${N_JOBS}" -m 40 \
    -o "${LOG_DIR}/xgb_tuned_top50_%j.out" \
    -e "${LOG_DIR}/xgb_tuned_top50_%j.err" \
    bash "${PROJECT_ROOT}/scripts/jobs/run_xgb_tuned_top50_comparison.sh" \
    "${REPRO_DATA_ROOT}" "${N_JOBS}" "${TUNING_ARTIFACT}"
