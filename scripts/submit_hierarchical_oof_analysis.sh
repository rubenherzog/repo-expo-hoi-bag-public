#!/usr/bin/env bash
# Submit the hierarchical-model / table / figure stage for the OOF improvement
# analysis. The default is a read-only preflight; only --submit contacts the
# scheduler. This stage fits no XGBoost models: it consumes the per-subject loss
# accumulators produced by submit_hierarchical_oof_cells.sh.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_main_k10}"
JOB="$PROJECT_ROOT/scripts/jobs/run_hierarchical_oof_analysis.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }

export REPRO_DATA_ROOT="$RUNTIME" HIER_OOF_RUN_ID="$RUN_ID"

if [[ "$MODE" == "--dry-run" ]]; then
  bash "$JOB" --preflight
  printf 'dry-run: analysis inputs and imports validated for %s; no job sent.\n' "$RUN_ID"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

# Mixed models over ~10,800 subjects per cell: modest and single-threaded, so
# this asks for far less than the fitting jobs.
run -t "02:00" -j "hier_oof_analysis" -c 4 -m 32 \
  -o "$LOG_ROOT/hier_oof_analysis_%j.out" -e "$LOG_ROOT/hier_oof_analysis_%j.err" \
  bash "$JOB"

printf 'Submitted the hierarchical-OOF analysis stage for %s\n' "$RUN_ID"
