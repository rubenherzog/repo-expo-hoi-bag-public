#!/usr/bin/env bash
# Country-block permutation null (ST04) under the global pooled OOF R2 estimand.
#
# Scientific contract is identical to the delivered country-balanced ST04: same
# cohort, folds, exclusions, candidate pool, order cap, winners-per-rung design,
# 10,000 country-block permutations and seed.  The single difference is the test
# statistic, selected through R2_MODE=global_oof.
#
# It writes only into its own run namespace and never touches the delivered
# country-balanced ST04.  The default preflight is read-only and never contacts
# Slurm.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
# The adapter is read from the delivered main-k10 run; the global estimand is a
# selection policy over the same evaluated candidates, so no new adapter build
# and no refit of the candidate pool is required.
SOURCE_RUN_ID="${MAIN_K10_SOURCE_CLUSTER_RUN_ID:-main_k10_release_20260916}"
RUN_ID="${MAIN_K10_GLOBAL_RUN_ID:-main_k10_global_oof_20260920}"
JOB="$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"
ADAPTER_ROOT="$RUNTIME/results/analysis_runs/$SOURCE_RUN_ID/input_adapter"
N_PERM="${CBN_N_PERM:-10000}"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -f "$ADAPTER_ROOT/main_k10_adapter_manifest.json" ]] || {
  echo "Missing main-k10 adapter: $ADAPTER_ROOT/main_k10_adapter_manifest.json" >&2
  exit 1
}

# Refuse to overwrite an existing global ST04 delivery.
for bag in structural functional; do
  existing="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/country_block_null/${bag}_winner_pvalues.csv"
  [[ -e "$existing" ]] && {
    echo "Refusing to overwrite existing global ST04 result: $existing" >&2
    echo "Set MAIN_K10_GLOBAL_RUN_ID to a new identifier." >&2
    exit 1
  }
done

export REPRO_DATA_ROOT="$RUNTIME"
export MAIN_K10_CLUSTER_RUN_ID="$SOURCE_RUN_ID"
export R2_MODE=global_oof

if [[ "$MODE" == "--dry-run" ]]; then
  R2_MODE=global_oof bash "$JOB" country-block-null structural 40 --preflight
  R2_MODE=global_oof bash "$JOB" country-block-null functional 40 --preflight
  printf 'dry-run passed for global-OOF country-block null, Structural and Functional; no job sent.\n'
  printf 'submit with: MAIN_K10_GLOBAL_RUN_ID=%q REPRO_DATA_ROOT=%q bash %q --submit\n' \
    "$RUN_ID" "$RUNTIME" "$0"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

for bag in structural functional; do
  run -t 72:00 -j "main_k10_cbn_global_${bag}" -c 40 -m 160 \
    -o "$LOG_ROOT/main_k10_cbn_global_${bag}_%j.out" \
    -e "$LOG_ROOT/main_k10_cbn_global_${bag}_%j.err" \
    env R2_MODE=global_oof CBN_N_PERM="$N_PERM" CBN_FAST=0 CBN_INCLUDE_SINGLE=0 \
    bash "$JOB" country-block-null "$bag" 40
done

printf 'Submitted 2 global-OOF country-block-null jobs (run=%s, n_perm=%s)\n' "$RUN_ID" "$N_PERM"
