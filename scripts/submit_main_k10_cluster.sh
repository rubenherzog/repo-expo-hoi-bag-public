#!/usr/bin/env bash
# Submit the pending main-k10 refit bundle.  --dry-run is local-only.
set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in --dry-run|--submit) ;; *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2;; esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
PAPER_REFERENCE_ROOT="${PAPER_REFERENCE_ROOT:?Set PAPER_REFERENCE_ROOT to the read-only historical runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_cluster_k10_20260915_v1}"
JOB="$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"

[[ -x "$JOB" ]] || { echo "Missing job runner: $JOB" >&2; exit 1; }
[[ -x "$PROJECT_ROOT/scripts/prepare_main_k10_cluster_adapter.sh" ]] || { echo "Missing adapter launcher" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json" ]] || { echo "Missing frozen k10 HPO artifact" >&2; exit 1; }

jobs=(
  "24:00|main_k10_cbn_structural|40|160|country-block-null|structural"
  "24:00|main_k10_cbn_functional|40|160|country-block-null|functional"
  "24:00|main_k10_norm_structural|40|160|normative-transfer-xgb|structural"
  "24:00|main_k10_norm_functional|40|160|normative-transfer-xgb|functional"
  "24:00|main_k10_domain_structural|40|160|domain-imbalance|structural"
  "24:00|main_k10_domain_functional|40|160|domain-imbalance|functional"
  "24:00|main_k10_pca_structural|40|160|whole-exposome-pca|structural"
  "24:00|main_k10_pca_functional|40|160|whole-exposome-pca|functional"
  "24:00|main_k10_loro_structural|40|160|country-region|structural"
  "24:00|main_k10_loro_functional|40|160|country-region|functional"
  "12:00|main_k10_edu_structural|40|160|education-scanner-baseline|structural"
  "12:00|main_k10_edu_functional|40|160|education-scanner-baseline|functional"
  "24:00|main_k10_residtarget_structural|40|160|residualized-bag|structural"
  "24:00|main_k10_residtarget_functional|40|160|residualized-bag|functional"
  "12:00|main_k10_diagbalance_structural|40|160|diagnosis-balance|structural"
  "12:00|main_k10_diagbalance_functional|40|160|diagnosis-balance|functional"
)

export REPRO_DATA_ROOT="$RUNTIME" MAIN_K10_CLUSTER_RUN_ID="$RUN_ID"
if [[ "$MODE" == "--dry-run" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
  python -m repo_expo_hoi_bag.stages.prepare_main_k10_sensitivity_adapter \
    --repro-data-root "$RUNTIME" \
    --candidate-registry "${MAIN_K10_CANDIDATE_REGISTRY:-$PAPER_REFERENCE_ROOT/results/variant_a/families/pooled_oinfo_ladder/canonical/candidate_registry.parquet}" \
    --adapter-run-id "$RUN_ID" \
    --effective-exposome-source "$PROJECT_ROOT/data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv" \
    --feature-domains "$PROJECT_ROOT/data/metadata/exposome_feature_domains.csv" \
    --domain-pc1-oinfo-reference "${MAIN_K10_DOMAIN_PC1_OINFO_REFERENCE:-$PAPER_REFERENCE_ROOT/sensitivity/domain_imbalance_dedup/domain_imbalance/global_hoi_cache/within_domain_pc1_thoi_scores.csv}" \
    --smoke-test
  python -m repo_expo_hoi_bag.stages.prepare_main_k10_residual_inputs \
    --repro-data-root "$RUNTIME" --adapter-run-id "$RUN_ID" --smoke-test
  for item in "${jobs[@]}"; do
    IFS='|' read -r _ name cores _ stage bag <<< "$item"
    bash "$JOB" "$stage" "$bag" "$cores" --preflight
    printf '  %s: %s/%s\n' "$name" "$stage" "$bag"
  done
  printf 'dry-run: adapter inputs, HPO artifact, compatibility imports and %s scheduler commands validated; no job sent.\n' "${#jobs[@]}"
  exit 0
fi

SCANNER_XLSX_PATH="${SCANNER_XLSX_PATH:?Set SCANNER_XLSX_PATH to the restricted scanner metadata workbook}"
export SCANNER_XLSX_PATH
[[ -f "$SCANNER_XLSX_PATH" ]] || { echo "Scanner metadata file not found: $SCANNER_XLSX_PATH" >&2; exit 1; }
for protected in \
  "$RUNTIME/results/analysis_runs/$RUN_ID" \
  "$RUNTIME/work/analysis_runs/$RUN_ID" \
  "$PROJECT_ROOT/outputs/main/$RUN_ID"; do
  [[ ! -e "$protected" ]] || { echo "Refusing to reuse existing run namespace: $protected" >&2; exit 1; }
done
command -v run >/dev/null || { echo "Cluster scheduler command 'run' is unavailable" >&2; exit 1; }
mkdir -p "$LOG_ROOT"
bash "$PROJECT_ROOT/scripts/prepare_main_k10_cluster_adapter.sh"
for item in "${jobs[@]}"; do
  IFS='|' read -r time_limit name cores memory stage bag <<< "$item"
  extra_env=()
  [[ "$stage" == "country-block-null" ]] && extra_env+=(CBN_N_PERM=10000)
  [[ "$stage" == "education-scanner-baseline" ]] && extra_env+=(SCANNER_XLSX_PATH="$SCANNER_XLSX_PATH")
  env "${extra_env[@]}" run -t "$time_limit" -j "$name" -c "$cores" -m "$memory" \
    -o "$LOG_ROOT/${name}_%j.out" -e "$LOG_ROOT/${name}_%j.err" \
    bash "$JOB" "$stage" "$bag" "$cores"
done
printf 'Submitted %s pending main-k10 refit jobs under %s\n' "${#jobs[@]}" "$RUN_ID"
