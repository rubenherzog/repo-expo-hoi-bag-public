#!/usr/bin/env bash
# Run the remaining main-k10 sensitivity stages locally, one at a time, using
# the established one-BAG runner.  It deliberately does not contact Slurm.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <run-id> <n-jobs>" >&2
  exit 2
fi

RUN_ID="$1"
N_JOBS="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

run_stage() {
  local stage="$1"
  local bag="$2"
  MAIN_K10_CLUSTER_RUN_ID="$RUN_ID" REPRO_DATA_ROOT="$RUNTIME" \
    bash "$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh" "$stage" "$bag" "$N_JOBS"
}

# Structural precedes functional throughout the primary-BAG analysis.
for stage in residualized-bag domain-imbalance whole-exposome-pca country-region diagnosis-balance normative-transfer-xgb country-block-null; do
  for bag in structural functional; do
    # A caller that has already started the structural pass may explicitly
    # suppress only that duplicate unit while it waits for its PID above.
    if [[ "${SKIP_RESIDUALIZED_STRUCTURAL:-0}" == "1" && "$stage" == "residualized-bag" && "$bag" == "structural" ]]; then
      continue
    fi
    run_stage "$stage" "$bag"
  done
done

PYTHONPATH="$PROJECT_ROOT/src" python -m repo_expo_hoi_bag.stages.collect_main_paper_delivery --run-id "$RUN_ID"
