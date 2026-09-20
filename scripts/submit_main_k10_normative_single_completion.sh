#!/usr/bin/env bash
# Complete the two missing single-exposure normative references in the existing
# main-k10 run. The default is a read-only preflight; only --submit contacts the
# cluster scheduler.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"
JOB="$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"
ADAPTER_ROOT="$RUNTIME/results/analysis_runs/$RUN_ID/input_adapter"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -f "$ADAPTER_ROOT/main_k10_adapter_manifest.json" ]] || {
  echo "Missing assigned main-k10 adapter: $ADAPTER_ROOT/main_k10_adapter_manifest.json" >&2
  exit 1
}

jobs=(
  "12:00|main_k10_single_norm_structural|40|160|structural"
  "12:00|main_k10_single_norm_functional|40|160|functional"
)

export REPRO_DATA_ROOT="$RUNTIME" MAIN_K10_CLUSTER_RUN_ID="$RUN_ID"

if [[ "$MODE" == "--dry-run" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r _ name cores _ bag <<< "$item"
    bash "$JOB" normative-transfer-single-xgb "$bag" "$cores" --preflight
    printf '  %s: normative-transfer-single-xgb/%s\n' "$name" "$bag"
  done
  printf 'dry-run: validated %s completion jobs in existing run %s; no job sent.\n' "${#jobs[@]}" "$RUN_ID"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

for item in "${jobs[@]}"; do
  IFS='|' read -r time_limit name cores memory bag <<< "$item"
  run -t "$time_limit" -j "$name" -c "$cores" -m "$memory" \
    -o "$LOG_ROOT/${name}_%j.out" -e "$LOG_ROOT/${name}_%j.err" \
    bash "$JOB" normative-transfer-single-xgb "$bag" "$cores"
done

printf 'Submitted %s normative single-exposure completion jobs into existing run %s\n' "${#jobs[@]}" "$RUN_ID"
