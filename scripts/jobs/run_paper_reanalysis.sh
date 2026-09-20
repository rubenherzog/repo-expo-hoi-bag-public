#!/bin/bash
set -euo pipefail

: "${REPRO_DATA_ROOT:?REPRO_DATA_ROOT is required}"
: "${PAPER_REANALYSIS_ID:?PAPER_REANALYSIS_ID is required}"
: "${PAPER_REANALYSIS_MODE:?PAPER_REANALYSIS_MODE is required}"
: "${PAPER_REANALYSIS_BAGS:?PAPER_REANALYSIS_BAGS is required}"
: "${PAPER_REANALYSIS_REFERENCE_ROOT:?PAPER_REANALYSIS_REFERENCE_ROOT is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/repo_expo_hoi_bag/core:${PROJECT_ROOT}/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"

args=(--mode "${PAPER_REANALYSIS_MODE}" --run-id "${PAPER_REANALYSIS_ID}" --bags ${PAPER_REANALYSIS_BAGS//,/ } --n-jobs "${PAPER_REANALYSIS_N_JOBS:-40}" --chunk-size "${PAPER_REANALYSIS_CHUNK_SIZE:-40}")
if [ "${PAPER_REANALYSIS_MODE}" = "xgb" ]; then
  : "${PAPER_REANALYSIS_RUNG:?PAPER_REANALYSIS_RUNG is required for xgb}"
  : "${PAPER_REANALYSIS_BASELINE_HPO:?baseline artifact is required}"
  : "${PAPER_REANALYSIS_SINGLE_HPO:?single artifact is required}"
  : "${PAPER_REANALYSIS_K10_HPO:?k10 artifact is required}"
  : "${PAPER_REANALYSIS_K63_HPO:?k63 artifact is required}"
  args+=(--rung "${PAPER_REANALYSIS_RUNG}")
fi
if [ "${PAPER_REANALYSIS_ACCEPT_LEGACY_CHECKPOINTS:-0}" = "1" ]; then
  args+=(--accept-legacy-checkpoints)
fi
python -m repo_expo_hoi_bag.stages.run_paper_reanalysis "${args[@]}"
