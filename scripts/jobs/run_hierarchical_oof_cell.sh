#!/usr/bin/env bash
# Subject-level OOF loss accumulators for the hierarchical improvement analysis.
# This script submits nothing; run it inside an allocation for one cell.
#
# A cell is one BAG x arm x rung. Each cell fits the 560 eligible candidates of
# that arm across every LOCO fold and writes a single parquet of per-subject
# candidate losses, from which the Top10/Top20/Top50/ALL regions are sliced.
# Cells already present are skipped, so a resubmission completes only the gaps.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <structural|functional> <o_min|o_max> <ols|xgb_tree_d1|xgb_tree_d2|xgb_tree_d3> <n_jobs> [--preflight]" >&2
  exit 2
fi

BAG="$1"
ARM="$2"
RUNG="$3"
N_JOBS="$4"
PREFLIGHT="${5:-}"
[[ -z "$PREFLIGHT" || "$PREFLIGHT" == "--preflight" ]] || { echo "Unknown option: $PREFLIGHT" >&2; exit 2; }
case "$BAG" in structural|functional) ;; *) echo "Unsupported BAG: $BAG" >&2; exit 2;; esac
case "$ARM" in o_min|o_max) ;; *) echo "Unsupported arm: $ARM" >&2; exit 2;; esac
case "$RUNG" in ols|xgb_tree_d1|xgb_tree_d2|xgb_tree_d3) ;; *) echo "Unsupported rung: $RUNG" >&2; exit 2;; esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
SOURCE_RUN_ID="${HIER_OOF_SOURCE_RUN_ID:-paper_reanalysis_k10}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_main_k10}"
# The per-subject loss accumulators are heavy generated results, so they live
# under the external runtime, not in the checkout.
OUTPUT_DIR="${HIER_OOF_OUTPUT_DIR:-$RUNTIME/work/analysis_runs/$RUN_ID/hierarchical_oof/loss_cache}"
K10_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json"
BASELINE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json"

for hpo in "$BASELINE_HPO" "$K10_HPO"; do
  [[ -f "$hpo" ]] || { echo "Missing frozen HPO selection: $hpo" >&2; exit 1; }
done
python - "$BASELINE_HPO" "$K10_HPO" <<'PY'
import json
import sys

expected = ("baseline", "domain_balanced_k10")
for raw_path, expected_scope in zip(sys.argv[1:], expected):
    with open(raw_path, encoding="utf-8") as handle:
        actual_scope = str(json.load(handle).get("provenance", {}).get("feature_scope", ""))
    if actual_scope != expected_scope:
        raise ValueError(
            f"HPO artifact {raw_path} declares feature_scope={actual_scope!r}; "
            f"expected {expected_scope!r}"
        )
PY

export REPRO_DATA_ROOT="$RUNTIME"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
# XGBoost stays single-threaded: parallelism comes from the candidate workers,
# and nested threading would oversubscribe the allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export XGB_TUNING_BASELINE_ARTIFACT="$BASELINE_HPO"
export XGB_TUNING_ARTIFACT="$K10_HPO" XGB_TUNING_STRICT=1

if [[ "$PREFLIGHT" == "--preflight" ]]; then
  # Import the stage and resolve the candidate plan for this cell without
  # fitting or writing anything, mirroring the other job runners' preflight.
  python - "$RUNTIME" "$SOURCE_RUN_ID" "$BAG" "$ARM" "$RUNG" <<'PY'
from pathlib import Path
import sys

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.run_hierarchical_oof_predictions import region_members

runtime, source_run_id, bag, arm, rung = Path(sys.argv[1]), *sys.argv[2:]
registry = frontier._registry(frontier._registry_path(runtime, source_run_id))
inputs = frontier.load_bag(runtime, source_run_id, bag, registry)
regions, everything = region_members(inputs, arm, rung)
folds = inputs.multivariate[rung].shape[1]
for smaller, larger in (("Top10", "Top20"), ("Top20", "Top50")):
    assert set(regions[smaller]) <= set(regions[larger]), f"{smaller} not nested in {larger}"
assert set(regions["Top50"]) <= set(everything), "Top50 not contained in ALL"
print(f"  candidates={len(everything)} folds={folds} fits={len(everything) * folds}")
PY
  printf 'Preflight passed: bag=%s arm=%s rung=%s\n' "$BAG" "$ARM" "$RUNG"
  exit 0
fi

echo "hierarchical oof cell bag=$BAG arm=$ARM rung=$RUNG workers=$N_JOBS source=$SOURCE_RUN_ID"
python -m repo_expo_hoi_bag.stages.run_hierarchical_oof_predictions \
  --repro-data-root "$RUNTIME" \
  --source-run-id "$SOURCE_RUN_ID" \
  --scope all \
  --bag "$BAG" --arm "$ARM" --rung "$RUNG" \
  --output-dir "$OUTPUT_DIR" \
  --n-jobs "$N_JOBS"
