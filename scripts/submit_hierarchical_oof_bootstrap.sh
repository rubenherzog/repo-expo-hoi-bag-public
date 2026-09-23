#!/usr/bin/env bash
# Submit the hierarchical bootstrap of the Top-K incremental R2. The default is
# a read-only preflight; only --submit contacts the scheduler. This stage refits
# nothing: it re-aggregates the stored per-subject losses.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${HIER_OOF_RUN_ID:-hierarchical_oof_kgrid}"
JOB="$PROJECT_ROOT/scripts/jobs/run_hierarchical_oof_bootstrap.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }

export REPRO_DATA_ROOT="$RUNTIME" HIER_OOF_RUN_ID="$RUN_ID"

if [[ "$MODE" == "--dry-run" ]]; then
  bash "$JOB" --preflight
  printf 'dry-run: bootstrap inputs validated for %s; no job sent.\n' "$RUN_ID"
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

# Paired 10,000-draw contrasts include the Top20 primary families and the
# descriptive K grid. Each comparison reuses one hierarchical draw for both
# models; the one-hour allocation leaves slack for the expanded comparison set.
run -t "01:00" -j "hier_oof_bootstrap" -c 40 -m 160 \
  -o "$LOG_ROOT/hier_oof_bootstrap_%j.out" -e "$LOG_ROOT/hier_oof_bootstrap_%j.err" \
  bash "$JOB" 40

printf 'Submitted the hierarchical-OOF bootstrap for %s\n' "$RUN_ID"
