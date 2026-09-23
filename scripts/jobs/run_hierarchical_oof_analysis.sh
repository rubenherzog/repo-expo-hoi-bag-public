#!/usr/bin/env bash
# Hierarchical models, publication table and figure for the OOF improvement
# analysis. This script submits nothing; run it inside an allocation.
#
# It consumes the per-subject loss accumulators written by
# run_hierarchical_oof_cell.sh and performs no model fitting of its own.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

PREFLIGHT="${1:-}"
[[ -z "$PREFLIGHT" || "$PREFLIGHT" == "--preflight" ]] || { echo "Unknown option: $PREFLIGHT" >&2; exit 2; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
SOURCE_RUN_ID="${HIER_OOF_SOURCE_RUN_ID:-paper_reanalysis_k10}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_main_k10}"
CACHE_DIR="${HIER_OOF_OUTPUT_DIR:-$RUNTIME/work/analysis_runs/$RUN_ID/hierarchical_oof/loss_cache}"

export REPRO_DATA_ROOT="$RUNTIME"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

expected=0
missing=0
for bag in structural functional; do
  for arm in o_min o_max; do
    for rung in xgb_tree_d2 xgb_tree_d3; do
      expected=$((expected + 1))
      [[ -f "$CACHE_DIR/all__${bag}__${arm}__${rung}.parquet" ]] || {
        echo "Missing loss accumulator: all__${bag}__${arm}__${rung}.parquet" >&2
        missing=$((missing + 1))
      }
    done
  done
done
[[ "$missing" -eq 0 ]] || { echo "$missing of $expected cells are not fitted yet" >&2; exit 1; }

if [[ "$PREFLIGHT" == "--preflight" ]]; then
  python -c "
from repo_expo_hoi_bag.stages import compute_hierarchical_oof_improvement, build_hierarchical_oof_table, plot_hierarchical_oof_improvement
print('  imports resolved')
"
  printf 'Preflight passed: %s cells present under %s\n' "$expected" "$CACHE_DIR"
  exit 0
fi

echo "hierarchical oof analysis run=$RUN_ID cache=$CACHE_DIR"
python -m repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement \
  --repro-data-root "$RUNTIME" \
  --source-run-id "$SOURCE_RUN_ID" \
  --run-id "$RUN_ID" \
  --cache-dir "$CACHE_DIR"

python -m repo_expo_hoi_bag.stages.build_hierarchical_oof_table
python -m repo_expo_hoi_bag.stages.plot_hierarchical_oof_improvement
echo "hierarchical oof analysis complete"
