#!/usr/bin/env bash
# Submit the pending hierarchical-OOF loss-accumulator cells, one job per cell.
# The default is a read-only preflight; only --submit contacts the scheduler.
#
# Each cell is one BAG x arm x rung and fits 560 candidates across every LOCO
# fold. Cells whose parquet already exists are skipped, so this completes only
# the gaps left by an earlier run.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_main_k10}"
JOB="$PROJECT_ROOT/scripts/jobs/run_hierarchical_oof_cell.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"
# Heavy loss accumulators stay on the cluster runtime; only logs land in the
# checkout, under logs/<run id>/.
CACHE_DIR="${HIER_OOF_OUTPUT_DIR:-$RUNTIME/work/analysis_runs/$RUN_ID/hierarchical_oof/loss_cache}"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json" ]] || {
  echo "Missing frozen k10 HPO artifact" >&2
  exit 1
}

# time|name|cores|memory|bag|arm|rung
# Structural cells carry 22 folds over 10,847 subjects and functional 16 folds
# over 9,441, so structural is given the longer wall clock. OLS is a closed-form
# least-squares fit with no boosting rounds, so it needs far less time.
jobs=(
  "04:00|hier_oof_struct_omin_ols|40|160|structural|o_min|ols"
  "04:00|hier_oof_struct_omax_ols|40|160|structural|o_max|ols"
  "04:00|hier_oof_func_omin_ols|40|160|functional|o_min|ols"
  "04:00|hier_oof_func_omax_ols|40|160|functional|o_max|ols"
  "12:00|hier_oof_struct_omin_d1|40|160|structural|o_min|xgb_tree_d1"
  "12:00|hier_oof_struct_omax_d1|40|160|structural|o_max|xgb_tree_d1"
  "08:00|hier_oof_func_omin_d1|40|160|functional|o_min|xgb_tree_d1"
  "08:00|hier_oof_func_omax_d1|40|160|functional|o_max|xgb_tree_d1"
  "12:00|hier_oof_struct_omin_d2|40|160|structural|o_min|xgb_tree_d2"
  "12:00|hier_oof_struct_omax_d2|40|160|structural|o_max|xgb_tree_d2"
  "12:00|hier_oof_struct_omin_d3|40|160|structural|o_min|xgb_tree_d3"
  "12:00|hier_oof_struct_omax_d3|40|160|structural|o_max|xgb_tree_d3"
  "08:00|hier_oof_func_omin_d2|40|160|functional|o_min|xgb_tree_d2"
  "08:00|hier_oof_func_omax_d2|40|160|functional|o_max|xgb_tree_d2"
  "08:00|hier_oof_func_omin_d3|40|160|functional|o_min|xgb_tree_d3"
  "08:00|hier_oof_func_omax_d3|40|160|functional|o_max|xgb_tree_d3"
)

export REPRO_DATA_ROOT="$RUNTIME" HIER_OOF_OUTPUT_DIR="$CACHE_DIR"

pending=()
for item in "${jobs[@]}"; do
  IFS='|' read -r _ _ _ _ bag arm rung <<< "$item"
  if [[ -f "$CACHE_DIR/all__${bag}__${arm}__${rung}.parquet" ]]; then
    printf '  skip (already complete): %s/%s/%s\n' "$bag" "$arm" "$rung"
  else
    pending+=("$item")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  printf 'All %s cells are already complete under %s; nothing to submit.\n' "${#jobs[@]}" "$CACHE_DIR"
  exit 0
fi

if [[ "$MODE" == "--dry-run" ]]; then
  for item in "${pending[@]}"; do
    IFS='|' read -r _ name cores _ bag arm rung <<< "$item"
    bash "$JOB" "$bag" "$arm" "$rung" "$cores" --preflight
    printf '  %s: %s/%s/%s\n' "$name" "$bag" "$arm" "$rung"
  done
  printf 'dry-run: validated %s pending cells (of %s); no job sent.\n' "${#pending[@]}" "${#jobs[@]}"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT" "$CACHE_DIR"

for item in "${pending[@]}"; do
  IFS='|' read -r time_limit name cores memory bag arm rung <<< "$item"
  run -t "$time_limit" -j "$name" -c "$cores" -m "$memory" \
    -o "$LOG_ROOT/${name}_%j.out" -e "$LOG_ROOT/${name}_%j.err" \
    bash "$JOB" "$bag" "$arm" "$rung" "$cores"
done

printf 'Submitted %s pending hierarchical-OOF cells under %s\n' "${#pending[@]}" "$RUN_ID"
