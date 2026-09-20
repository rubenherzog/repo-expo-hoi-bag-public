#!/bin/bash
# Prepare six validation jobs; submit manually after reviewing this command.
set -euo pipefail
REPRO_DATA_ROOT="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}"
if [[ -n "${PAPER_REANALYSIS_REFERENCE_ROOT:-}" ]]; then
  REFERENCE_ROOT="$PAPER_REANALYSIS_REFERENCE_ROOT"
else
  PAPER_REFERENCE_ROOT="${PAPER_REFERENCE_ROOT:?Set PAPER_REFERENCE_ROOT or PAPER_REANALYSIS_REFERENCE_ROOT}"
  REFERENCE_ROOT="$PAPER_REFERENCE_ROOT/results/variant_a/families/pooled_oinfo_ladder/canonical"
fi
RUN_ID="${HISTORICAL_VALIDATION_RUN_ID:-historical_driver_validation_$(date -u +%Y%m%dT%H%M%SZ)}"
for bag in structural functional; do for rung in xgb_tree_d1 xgb_tree_d2 xgb_tree_d3; do
  env PROJECT_ROOT="$PROJECT_ROOT" REPRO_DATA_ROOT="$REPRO_DATA_ROOT" PAPER_REANALYSIS_REFERENCE_ROOT="$REFERENCE_ROOT" PAPER_REANALYSIS_ID="$RUN_ID" PAPER_REANALYSIS_BAGS="$bag" PAPER_REANALYSIS_RUNG="$rung" PAPER_REANALYSIS_SETS=historical PAPER_REANALYSIS_VALIDATE_REFERENCE=1 PAPER_REANALYSIS_N_JOBS=40 PAPER_REANALYSIS_CHUNK_SIZE=40 \
  run -t 1-00:00 -j "historical_validate_${bag}_${rung}" -c 40 -m 160 -o "${PROJECT_ROOT}/logs/historical_validation_${bag}_${rung}_%j.out" -e "${PROJECT_ROOT}/logs/historical_validation_${bag}_${rung}_%j.err" bash "${SCRIPT_DIR}/jobs/run_historical_driver_reanalysis.sh"
done; done
