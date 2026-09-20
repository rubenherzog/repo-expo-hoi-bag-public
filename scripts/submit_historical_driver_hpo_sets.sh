#!/bin/bash
# Prepare exactly seven jobs: one OLS job and six BAG × rung XGBoost jobs.
# Each XGBoost job sequentially evaluates the same candidate pool with k10,
# k63, and single-exposure frozen parameter vectors.
set -euo pipefail
REPRO_DATA_ROOT="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}"
if [[ -n "${PAPER_REANALYSIS_REFERENCE_ROOT:-}" ]]; then
  REFERENCE_ROOT="$PAPER_REANALYSIS_REFERENCE_ROOT"
else
  PAPER_REFERENCE_ROOT="${PAPER_REFERENCE_ROOT:?Set PAPER_REFERENCE_ROOT or PAPER_REANALYSIS_REFERENCE_ROOT}"
  REFERENCE_ROOT="$PAPER_REFERENCE_ROOT/results/variant_a/families/pooled_oinfo_ladder/canonical"
fi
K10_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json"
K63_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_full63_cap500_final/selected_xgb_configs.json"
SINGLE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_single_cap500/selected_xgb_configs.json"
# OLS has no hyperparameters; retain its established evaluator and place this
# independent refit in the same new delivery namespace.
env PROJECT_ROOT="$PROJECT_ROOT" REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" PAPER_REANALYSIS_ID=paper_reanalysis_historical_driver PAPER_REANALYSIS_MODE=ols PAPER_REANALYSIS_BAGS=structural,functional PAPER_REANALYSIS_N_JOBS=40 PAPER_REANALYSIS_CHUNK_SIZE=40 \
  run -t 1-00:00 -j historical_hpo_ols -c 40 -m 160 -o "${PROJECT_ROOT}/logs/historical_hpo_ols_%j.out" -e "${PROJECT_ROOT}/logs/historical_hpo_ols_%j.err" bash "${SCRIPT_DIR}/jobs/run_paper_reanalysis.sh"
for bag in structural functional; do for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3; do
  env PROJECT_ROOT="$PROJECT_ROOT" REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" PAPER_REANALYSIS_ID=paper_reanalysis_historical_driver PAPER_REANALYSIS_BAGS="$bag" PAPER_REANALYSIS_RUNG="$rung" PAPER_REANALYSIS_SETS=k10,k63,single PAPER_REANALYSIS_N_JOBS=40 PAPER_REANALYSIS_CHUNK_SIZE=40 PAPER_REANALYSIS_K10_HPO="$K10_HPO" PAPER_REANALYSIS_K63_HPO="$K63_HPO" PAPER_REANALYSIS_SINGLE_HPO="$SINGLE_HPO" \
  run -t 3-00:00 -j "historical_hpo_${bag}_${rung}" -c 40 -m 160 -o "${PROJECT_ROOT}/logs/historical_hpo_${bag}_${rung}_%j.out" -e "${PROJECT_ROOT}/logs/historical_hpo_${bag}_${rung}_%j.err" bash "${SCRIPT_DIR}/jobs/run_historical_driver_reanalysis.sh"
done; done
