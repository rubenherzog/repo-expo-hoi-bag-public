#!/bin/bash
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
# User-directed exception: scheduler stdout/stderr is retained with this
# checkout for direct inspection. Numerical checkpoints and results remain in
# the canonical external runtime.
SCHEDULER_LOG_DIR="${PROJECT_ROOT}/logs/paper_reanalysis"
mkdir -p "${SCHEDULER_LOG_DIR}"
# One-time migration: accepts verified completed 32-model chunks written by
# the canceled jobs, then schedules the remaining candidates in chunks of 40.
COMMON=(PROJECT_ROOT="$PROJECT_ROOT" REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" PAPER_REANALYSIS_N_JOBS=40 PAPER_REANALYSIS_CHUNK_SIZE=40 PAPER_REANALYSIS_ACCEPT_LEGACY_CHECKPOINTS=1)
BASELINE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json"
SINGLE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_single_cap500/selected_xgb_configs.json"
K10_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json"
K63_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_full63_cap500_final/selected_xgb_configs.json"

# Fail before scheduler submission if the worker cannot import, validate all
# frozen artifacts, load the official pool/cohort, or construct every LOCO
# fold. This is deliberately model-free and writes no runtime state.
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/src/repo_expo_hoi_bag/core:${PROJECT_ROOT}/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" \
PAPER_REANALYSIS_BASELINE_HPO="$BASELINE_HPO" PAPER_REANALYSIS_SINGLE_HPO="$SINGLE_HPO" \
PAPER_REANALYSIS_K10_HPO="$K10_HPO" PAPER_REANALYSIS_K63_HPO="$K63_HPO" \
python -m repo_expo_hoi_bag.stages.run_paper_reanalysis --mode xgb --bags structural functional \
  --rung xgb_tree_d1 --run-id paper_reanalysis_k10 --preflight
REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" \
python -m repo_expo_hoi_bag.stages.run_paper_reanalysis --mode ols --bags structural functional \
  --run-id paper_reanalysis_k10 --preflight

# One 40-core job refits OLS for both BAGs. OLS has no HPO and no XGB rung.
env "${COMMON[@]}" PAPER_REANALYSIS_ID=paper_reanalysis_k10 PAPER_REANALYSIS_MODE=ols PAPER_REANALYSIS_BAGS=structural,functional \
  run -t 1-00:00 -j paper_reanalysis_ols -c 40 -m 160 \
  -o "${SCHEDULER_LOG_DIR}/paper_reanalysis_ols_%j.out" -e "${SCHEDULER_LOG_DIR}/paper_reanalysis_ols_%j.err" \
  bash "${SCRIPT_DIR}/jobs/run_paper_reanalysis.sh"

for bag in structural functional; do
  for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3; do
    env "${COMMON[@]}" PAPER_REANALYSIS_ID=paper_reanalysis_k10 PAPER_REANALYSIS_MODE=xgb PAPER_REANALYSIS_BAGS="$bag" PAPER_REANALYSIS_RUNG="$rung" \
      PAPER_REANALYSIS_BASELINE_HPO="$BASELINE_HPO" PAPER_REANALYSIS_SINGLE_HPO="$SINGLE_HPO" PAPER_REANALYSIS_K10_HPO="$K10_HPO" PAPER_REANALYSIS_K63_HPO="$K63_HPO" \
    run -t 1-00:00 -j "paper_reanalysis_xgb_${bag}_${rung}" -c 40 -m 160 \
      -o "${SCHEDULER_LOG_DIR}/paper_reanalysis_xgb_${bag}_${rung}_%j.out" -e "${SCHEDULER_LOG_DIR}/paper_reanalysis_xgb_${bag}_${rung}_%j.err" \
      bash "${SCRIPT_DIR}/jobs/run_paper_reanalysis.sh"
  done
done
