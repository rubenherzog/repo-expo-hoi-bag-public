#!/bin/bash
# Usage: bash scripts/jobs/run_xgb_tuned_top50_comparison.sh <repro_data_root> <n_jobs> <artifact>
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <repro_data_root> <n_jobs> <selected_xgb_configs.json>" >&2
    exit 1
fi
REPRO_DATA_ROOT="$1"
N_JOBS="$2"
TUNING_ARTIFACT="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/../..}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export XGB_TUNING_ARTIFACT="${TUNING_ARTIFACT}"
export SENSITIVITY_N_JOBS="${N_JOBS}"
export MPLBACKEND="Agg"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset XGB_TUNING_SMOKE XGB_TUNING_RUN_ID V3_EXCLUDE_COUNTRIES V3_EXCLUDE_DIAGNOSIS
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m repo_expo_hoi_bag.cli --repro-data-root "${REPRO_DATA_ROOT}" \
    run sensitivity xgb-tuned-top50-comparison
