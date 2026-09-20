#!/usr/bin/env bash
# Re-run only the corrected country-balanced country-block null in the assigned
# main-k10 run. The default preflight is read-only and never contacts Slurm.
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

export REPRO_DATA_ROOT="$RUNTIME" MAIN_K10_CLUSTER_RUN_ID="$RUN_ID"

if [[ "$MODE" == "--dry-run" ]]; then
  bash "$JOB" country-block-null structural 40 --preflight
  bash "$JOB" country-block-null functional 40 --preflight
  printf 'dry-run passed for country-balanced country-block null, Structural and Functional; no job sent.\n'
  printf 'submit with: MAIN_K10_CLUSTER_RUN_ID=%q REPRO_DATA_ROOT=%q bash %q --submit\n' \
    "$RUN_ID" "$RUNTIME" "$0"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

run -t 72:00 -j main_k10_cbn_cb_structural -c 40 -m 160 \
  -o "$LOG_ROOT/main_k10_cbn_cb_structural_%j.out" \
  -e "$LOG_ROOT/main_k10_cbn_cb_structural_%j.err" \
  env CBN_N_PERM=10000 CBN_FAST=0 CBN_RUNGS=xgb_tree_d3 \
  bash "$JOB" country-block-null structural 40

run -t 72:00 -j main_k10_cbn_cb_functional -c 40 -m 160 \
  -o "$LOG_ROOT/main_k10_cbn_cb_functional_%j.out" \
  -e "$LOG_ROOT/main_k10_cbn_cb_functional_%j.err" \
  env CBN_N_PERM=10000 CBN_FAST=0 CBN_RUNGS=xgb_tree_d3 \
  bash "$JOB" country-block-null functional 40

printf 'Submitted 2 corrected country-block-null jobs into assigned run %s\n' "$RUN_ID"
