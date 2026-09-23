#!/usr/bin/env bash
# Paired hierarchical bootstrap of country-balanced OOF R2 contrasts, plus
# the Top-K frontier and S13-style complexity figure.
# This script submits nothing; run it inside an allocation.
#
# It consumes the per-subject loss accumulators from the fine-grid fitting run
# and refits no model.
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [n_jobs] [--preflight]" >&2
  exit 2
fi

if [[ "${1:-}" == "--preflight" ]]; then
  N_JOBS=40
  PREFLIGHT="--preflight"
else
  N_JOBS="${1:-40}"
  PREFLIGHT="${2:-}"
fi
[[ -z "$PREFLIGHT" || "$PREFLIGHT" == "--preflight" ]] || { echo "Unknown option: $PREFLIGHT" >&2; exit 2; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_kgrid}"
CACHE_DIR="${HIER_OOF_OUTPUT_DIR:-$RUNTIME/work/analysis_runs/$RUN_ID/hierarchical_oof/loss_cache}"
DRAWS="${HIER_OOF_BOOTSTRAP_DRAWS:-10000}"

export REPRO_DATA_ROOT="$RUNTIME"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

missing=0
for bag in structural functional; do
  for arm in o_min o_max; do
    for rung in ols xgb_tree_d1 xgb_tree_d2 xgb_tree_d3; do
      [[ -f "$CACHE_DIR/all__${bag}__${arm}__${rung}.parquet" ]] || {
        echo "Missing loss accumulator: all__${bag}__${arm}__${rung}.parquet" >&2
        missing=$((missing + 1))
      }
    done
  done
done
[[ "$missing" -eq 0 ]] || { echo "$missing cells are not fitted yet" >&2; exit 1; }

if [[ "$PREFLIGHT" == "--preflight" ]]; then
  python - "$CACHE_DIR" <<'PY'
import sys
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages import compute_hierarchical_oof_comparisons  # noqa: F401
from repo_expo_hoi_bag.stages import plot_paired_hierarchical_oof_complexity  # noqa: F401
from repo_expo_hoi_bag.stages import plot_paired_hierarchical_oof_frontier  # noqa: F401
from repo_expo_hoi_bag.stages import build_paired_hierarchical_oof_tables  # noqa: F401

cache = Path(sys.argv[1])
frame = pd.read_parquet(cache / "all__structural__o_min__xgb_tree_d1.parquet")
needed = {"row_id", "country", "y_true", "baseline_loss", "mean_candidate_loss", "region"}
missing = needed.difference(frame.columns)
if missing:
    raise SystemExit(f"Stored losses lack required columns: {sorted(missing)}")
regions = sorted(frame["region"].unique())
print(f"  imports resolved; {len(regions)} regions present: {regions}")
print("  every quantity is re-aggregated from stored losses; zero refitting required")
PY
  printf 'Preflight passed: cache=%s\n' "$CACHE_DIR"
  exit 0
fi

echo "paired hierarchical OOF bootstrap run=$RUN_ID draws=$DRAWS workers=$N_JOBS cache=$CACHE_DIR"
python -m repo_expo_hoi_bag.stages.compute_hierarchical_oof_comparisons \
  --repro-data-root "$RUNTIME" \
  --run-id "$RUN_ID" \
  --cache-dir "$CACHE_DIR" \
  --draws "$DRAWS" \
  --n-jobs "$N_JOBS"

python -m repo_expo_hoi_bag.stages.plot_paired_hierarchical_oof_complexity
python -m repo_expo_hoi_bag.stages.plot_paired_hierarchical_oof_frontier
python -m repo_expo_hoi_bag.stages.build_paired_hierarchical_oof_tables
echo "paired hierarchical OOF bootstrap complete"
