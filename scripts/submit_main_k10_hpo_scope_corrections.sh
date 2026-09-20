#!/usr/bin/env bash
# Re-run only main-k10 sensitivities whose heterogeneous endpoint refits require
# scope-specific HPO. Reuses the assigned run and output roots in place.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--submit|--ols-dry-run|--ols-submit) ;;
  *) echo "Usage: $0 [--dry-run|--submit|--ols-dry-run|--ols-submit]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"
JOB="$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"
ADAPTER_ROOT="$RUNTIME/results/analysis_runs/$RUN_ID/input_adapter"
SCANNER_XLSX_PATH="${SCANNER_XLSX_PATH:-$PROJECT_ROOT/data/metadata/Megabase_ODQ_scanners.xlsx}"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -f "$ADAPTER_ROOT/main_k10_adapter_manifest.json" ]] || {
  echo "Missing assigned main-k10 adapter: $ADAPTER_ROOT/main_k10_adapter_manifest.json" >&2
  exit 1
}
for scope in baseline single k10; do
  [[ -f "$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_${scope}_cap500/selected_xgb_configs.json" ]] || {
    echo "Missing frozen $scope HPO artifact" >&2
    exit 1
  }
done
[[ -f "$SCANNER_XLSX_PATH" ]] || {
  echo "Scanner metadata file not found: $SCANNER_XLSX_PATH" >&2
  exit 1
}

jobs=(
  "24:00|main_k10_hporoute_norm_structural|40|160|normative-transfer-xgb|structural"
  "24:00|main_k10_hporoute_norm_functional|40|160|normative-transfer-xgb|functional"
  "24:00|main_k10_hporoute_loro_structural|40|160|country-region|structural"
  "24:00|main_k10_hporoute_loro_functional|40|160|country-region|functional"
  "12:00|main_k10_hporoute_edu_structural|40|160|education-scanner-baseline|structural"
  "12:00|main_k10_hporoute_edu_functional|40|160|education-scanner-baseline|functional"
  "24:00|main_k10_hporoute_resid_structural|40|160|residualized-bag|structural"
  "24:00|main_k10_hporoute_resid_functional|40|160|residualized-bag|functional"
  "12:00|main_k10_hporoute_diag_structural|40|160|diagnosis-balance|structural"
  "12:00|main_k10_hporoute_diag_functional|40|160|diagnosis-balance|functional"
)

if [[ "$MODE" == "--ols-dry-run" || "$MODE" == "--ols-submit" ]]; then
  jobs=(
    "12:00|main_k10_ols_edu_structural|40|160|education-scanner-baseline|structural"
    "12:00|main_k10_ols_edu_functional|40|160|education-scanner-baseline|functional"
    "24:00|main_k10_ols_resid_structural|40|160|residualized-bag|structural"
    "24:00|main_k10_ols_resid_functional|40|160|residualized-bag|functional"
  )
  export MAIN_K10_SENSITIVITY_RUNGS=ols
fi

export REPRO_DATA_ROOT="$RUNTIME" MAIN_K10_CLUSTER_RUN_ID="$RUN_ID" SCANNER_XLSX_PATH

if [[ "$MODE" == "--dry-run" || "$MODE" == "--ols-dry-run" ]]; then
  for item in "${jobs[@]}"; do
    IFS='|' read -r _ name cores _ stage bag <<< "$item"
    bash "$JOB" "$stage" "$bag" "$cores" --preflight
    printf '  %s: %s/%s\n' "$name" "$stage" "$bag"
  done
  printf 'dry-run: validated %s corrected jobs in existing run %s; no job sent.\n' "${#jobs[@]}" "$RUN_ID"
  if [[ "$MODE" == "--ols-dry-run" ]]; then
    printf 'OLS completion uses the existing education/scanner and residualized-BAG runtime paths.\n'
  fi
  exit 0
fi

command -v run >/dev/null || {
  echo "Cluster scheduler command 'run' is unavailable" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"

for item in "${jobs[@]}"; do
  IFS='|' read -r time_limit name cores memory stage bag <<< "$item"
  extra_env=()
  [[ "$stage" == "education-scanner-baseline" ]] && extra_env+=(SCANNER_XLSX_PATH="$SCANNER_XLSX_PATH")
  env "${extra_env[@]}" run -t "$time_limit" -j "$name" -c "$cores" -m "$memory" \
    -o "$LOG_ROOT/${name}_%j.out" -e "$LOG_ROOT/${name}_%j.err" \
    bash "$JOB" "$stage" "$bag" "$cores"
done

printf 'Submitted %s sensitivity corrections into existing run %s\n' "${#jobs[@]}" "$RUN_ID"
