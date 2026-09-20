#!/usr/bin/env bash
# Submit only the corrected diagnosis-balance jobs into the assigned main-k10 run.
# Safe default: --dry-run validates the existing adapter/imports and never contacts Slurm.
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
FINAL_SUMMARY_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/diagnosis_balance"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -f "$ADAPTER_ROOT/main_k10_adapter_manifest.json" ]] || {
  echo "Missing assigned main-k10 adapter: $ADAPTER_ROOT/main_k10_adapter_manifest.json" >&2
  exit 1
}
[[ -f "$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json" ]] || {
  echo "Missing frozen k10 HPO artifact" >&2
  exit 1
}

export REPRO_DATA_ROOT="$RUNTIME" MAIN_K10_CLUSTER_RUN_ID="$RUN_ID"

if [[ "$MODE" == "--dry-run" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
  bash "$JOB" diagnosis-balance structural 40 --preflight
  bash "$JOB" diagnosis-balance functional 40 --preflight
  printf 'dry-run passed for diagnosis-balance Structural and Functional; no job sent.\n'
  printf 'submit with: MAIN_K10_CLUSTER_RUN_ID=%q REPRO_DATA_ROOT=%q bash %q --submit\n' \
    "$RUN_ID" "$RUNTIME" "$0"
  exit 0
fi

if [[ -d "$FINAL_SUMMARY_ROOT" ]] && find "$FINAL_SUMMARY_ROOT" -type f -print -quit | grep -q .; then
  echo "Refusing to replace completed diagnosis-balance summaries: $FINAL_SUMMARY_ROOT" >&2
  exit 1
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}

mkdir -p "$LOG_ROOT"

run -t 12:00 -j main_k10_diagbalance_structural -c 40 -m 160 \
  -o "$LOG_ROOT/main_k10_diagbalance_structural_%j.out" \
  -e "$LOG_ROOT/main_k10_diagbalance_structural_%j.err" \
  bash "$JOB" diagnosis-balance structural 40

run -t 12:00 -j main_k10_diagbalance_functional -c 40 -m 160 \
  -o "$LOG_ROOT/main_k10_diagbalance_functional_%j.out" \
  -e "$LOG_ROOT/main_k10_diagbalance_functional_%j.err" \
  bash "$JOB" diagnosis-balance functional 40

printf 'Submitted 2 diagnosis-balance jobs into assigned run %s\n' "$RUN_ID"
