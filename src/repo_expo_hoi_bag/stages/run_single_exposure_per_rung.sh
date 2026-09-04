#!/usr/bin/env bash
# Run single-exposure LOCO evaluation for each missing rung: OLS, d1, d3.
# d2 already exists at $OUTPUT_ROOT/single_exposure_eval/{bag}/.
#
# Each rung writes to:
#   $OUTPUT_ROOT/single_exposure_eval_<rung>/{bag}/single_exposure_{global,country}.csv

set -euo pipefail

CONDA_PYTHON="${CONDA_PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to a writable external directory before running this stage.}"
OUTPUT_ROOT="${V3_OUTPUT_ROOT:-$REPRO_DATA_ROOT/results/variant_a}"
GREEDY_ROOT="${V3_GREEDY_ROOT:-$REPRO_DATA_ROOT/work/greedy}"
N_JOBS="${SINGLE_EXPOSURE_N_JOBS:-48}"

cd "$REPO"

# ── per-rung runner ───────────────────────────────────────────────────────────
run_rung() {
    local RUNG="$1"
    local MAX_DEPTH="$2"   # 0 = OLS/gblinear, 1/3 = gbtree

    local OUTDIR="$OUTPUT_ROOT/single_exposure_eval_${RUNG}"
    echo ""
    echo "========================================"
    echo "RUNG=$RUNG  MAX_DEPTH=$MAX_DEPTH  -> $OUTDIR"
    echo "========================================"

    V3_RAW_PATH="$REPO/data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv" \
    V3_GREEDY_ROOT="$GREEDY_ROOT" \
    V3_OUTDIR_BASE="$OUTDIR" \
    V3_EXCLUDE_COUNTRIES="France,Italy,Egypt,Greece,Poland" \
    V3_EXCLUDE_DIAGNOSIS="Other,AFM,MCI" \
    BAG_TARGET_MODE="all" \
    SMOKE_TEST="0" \
    SINGLE_EXPOSURE_MAX_DEPTH="$MAX_DEPTH" \
    SINGLE_EXPOSURE_N_JOBS="$N_JOBS" \
    "$CONDA_PYTHON" -m repo_expo_hoi_bag.stages._single_exposure_rung_runner

    echo "DONE: $RUNG"
}

run_rung "ols"         0
run_rung "xgb_tree_d1" 1
run_rung "xgb_tree_d3" 3

echo ""
echo "All rungs complete. Outputs:"
for RUNG in ols xgb_tree_d1 xgb_tree_d3; do
    echo "  $OUTPUT_ROOT/single_exposure_eval_${RUNG}/"
done
