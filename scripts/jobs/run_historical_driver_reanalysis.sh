#!/bin/bash
set -euo pipefail
: "${REPRO_DATA_ROOT:?}"; : "${PROJECT_ROOT:?}"; : "${PAPER_REANALYSIS_ID:?}"; : "${PAPER_REANALYSIS_BAGS:?}"; : "${PAPER_REANALYSIS_RUNG:?}"; : "${PAPER_REANALYSIS_SETS:?}"; : "${PAPER_REANALYSIS_REFERENCE_ROOT:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/repo_expo_hoi_bag/core:${PROJECT_ROOT}/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
args=(--run-id "$PAPER_REANALYSIS_ID" --bags ${PAPER_REANALYSIS_BAGS//,/ } --rung "$PAPER_REANALYSIS_RUNG" --set ${PAPER_REANALYSIS_SETS//,/ } --n-jobs "${PAPER_REANALYSIS_N_JOBS:-40}" --chunk-size "${PAPER_REANALYSIS_CHUNK_SIZE:-40}")
if [ "${PAPER_REANALYSIS_VALIDATE_REFERENCE:-0}" = 1 ]; then args+=(--validate-reference); fi
python -m repo_expo_hoi_bag.stages.run_historical_driver_reanalysis "${args[@]}"
