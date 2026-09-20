#!/bin/bash
# Prepare, but do not submit, the seven paper-reanalysis scheduler jobs which
# reuse the completed single-exposure HPO vector for every candidate feature set.
set -euo pipefail

REPRO_DATA_ROOT="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}"
if [[ -n "${PAPER_REANALYSIS_REFERENCE_ROOT:-}" ]]; then
  REFERENCE_ROOT="$PAPER_REANALYSIS_REFERENCE_ROOT"
else
  PAPER_REFERENCE_ROOT="${PAPER_REFERENCE_ROOT:?Set PAPER_REFERENCE_ROOT or PAPER_REANALYSIS_REFERENCE_ROOT}"
  REFERENCE_ROOT="$PAPER_REFERENCE_ROOT/results/variant_a/families/pooled_oinfo_ladder/canonical"
fi
SCHEDULER_LOG_DIR="${PROJECT_ROOT}/logs/paper_reanalysis_single_set_params"
mkdir -p "${SCHEDULER_LOG_DIR}"

BASELINE_HPO="${PROJECT_ROOT}/outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json"
SINGLE_HPO="${PROJECT_ROOT}/outputs/xgb_nested_loco_tuning/historical5_single_cap500/selected_xgb_configs.json"
RUN_ID="paper_reanalysis_single_set_params_k10"
COMMON=(PROJECT_ROOT="$PROJECT_ROOT" REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" PAPER_REANALYSIS_N_JOBS=40 PAPER_REANALYSIS_CHUNK_SIZE=40 PAPER_REANALYSIS_SET_HPO_SOURCE=single)

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/repo_expo_hoi_bag/core:${PROJECT_ROOT}/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" \
PAPER_REANALYSIS_BASELINE_HPO="$BASELINE_HPO" PAPER_REANALYSIS_SINGLE_HPO="$SINGLE_HPO" \
PAPER_REANALYSIS_K10_HPO="$SINGLE_HPO" PAPER_REANALYSIS_K63_HPO="$SINGLE_HPO" PAPER_REANALYSIS_SET_HPO_SOURCE=single \
python -m repo_expo_hoi_bag.stages.run_paper_reanalysis --mode xgb --bags structural functional --rung xgb_tree_d1 --run-id "$RUN_ID" --preflight

env "${COMMON[@]}" PAPER_REANALYSIS_ID="$RUN_ID" PAPER_REANALYSIS_MODE=ols PAPER_REANALYSIS_BAGS=structural,functional \
  run -t 1-00:00 -j paper_reanalysis_singleparams_ols -c 40 -m 160 \
  -o "${SCHEDULER_LOG_DIR}/paper_reanalysis_ols_%j.out" -e "${SCHEDULER_LOG_DIR}/paper_reanalysis_ols_%j.err" \
  bash "${SCRIPT_DIR}/jobs/run_paper_reanalysis.sh"

for bag in structural functional; do
  for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3; do
    env "${COMMON[@]}" PAPER_REANALYSIS_ID="$RUN_ID" PAPER_REANALYSIS_MODE=xgb PAPER_REANALYSIS_BAGS="$bag" PAPER_REANALYSIS_RUNG="$rung" \
      PAPER_REANALYSIS_BASELINE_HPO="$BASELINE_HPO" PAPER_REANALYSIS_SINGLE_HPO="$SINGLE_HPO" PAPER_REANALYSIS_K10_HPO="$SINGLE_HPO" PAPER_REANALYSIS_K63_HPO="$SINGLE_HPO" \
      run -t 1-00:00 -j "paper_reanalysis_singleparams_xgb_${bag}_${rung}" -c 40 -m 160 \
      -o "${SCHEDULER_LOG_DIR}/paper_reanalysis_xgb_${bag}_${rung}_%j.out" -e "${SCHEDULER_LOG_DIR}/paper_reanalysis_xgb_${bag}_${rung}_%j.err" \
      bash "${SCRIPT_DIR}/jobs/run_paper_reanalysis.sh"
  done
done
