#!/usr/bin/env bash
# Parent sensitivity runner, routed only to the main k10 adapter.  This script
# submits nothing; run it inside an allocation for one BAG and one stage.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <stage> <structural|functional> <n_jobs> [--preflight]" >&2
  exit 2
fi

STAGE="$1"
BAG="$2"
N_JOBS="$3"
PREFLIGHT="${4:-}"
[[ -z "$PREFLIGHT" || "$PREFLIGHT" == "--preflight" ]] || { echo "Unknown option: $PREFLIGHT" >&2; exit 2; }
case "$BAG" in structural|functional) ;; *) echo "Unsupported BAG: $BAG" >&2; exit 2;; esac
case "$STAGE" in
  country-block-null|normative-transfer-xgb|normative-transfer-single-xgb|domain-imbalance|whole-exposome-pca|country-region|education-scanner-baseline|residualized-bag|residualized-bag-clean|residual-confounds|diagnosis-balance)
    ;;
  *) echo "Unsupported main-k10 stage: $STAGE" >&2; exit 2;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:?Set MAIN_K10_CLUSTER_RUN_ID to a new immutable cluster run ID}"
ADAPTER_ROOT="$RUNTIME/results/analysis_runs/$RUN_ID/input_adapter"
EFFECTIVE_EXPOSOME="$ADAPTER_ROOT/effective_exposome_country_year.csv"
DOMAIN_PC1_OINFO="$ADAPTER_ROOT/domain_pc1_oinfo_scores.csv"
BASELINE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json"
SINGLE_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_single_cap500/selected_xgb_configs.json"
K10_HPO="$PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json"

for hpo in "$BASELINE_HPO" "$SINGLE_HPO" "$K10_HPO"; do
  [[ -f "$hpo" ]] || { echo "Missing frozen HPO selection: $hpo" >&2; exit 1; }
done
python - "$BASELINE_HPO" "$SINGLE_HPO" "$K10_HPO" <<'PY'
import json
import sys

expected = ("baseline", "single_exposure", "domain_balanced_k10")
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
export MAIN_K10_MODE=1 MAIN_K10_CLUSTER_RUN_ID="$RUN_ID" MAIN_K10_CANONICAL_ROOT="$ADAPTER_ROOT"
[[ -f "$EFFECTIVE_EXPOSOME" ]] || { echo "Missing main-k10 effective exposome: $EFFECTIVE_EXPOSOME" >&2; exit 1; }
[[ -f "$DOMAIN_PC1_OINFO" ]] || { echo "Missing precomputed domain-PC1 O-information: $DOMAIN_PC1_OINFO" >&2; exit 1; }
export DEDUP_EXPOSOME_CSV="$EFFECTIVE_EXPOSOME"
export DOMAIN_PC1_OINFO_REFERENCE="$DOMAIN_PC1_OINFO"
export MAIN_K10_SOURCE_RUN_ID=paper_reanalysis_k10
export XGB_TUNING_BASELINE_ARTIFACT="$BASELINE_HPO"
export XGB_TUNING_SINGLE_ARTIFACT="$SINGLE_HPO"
export XGB_TUNING_ARTIFACT="$K10_HPO" XGB_TUNING_STRICT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export SENSITIVITY_N_JOBS="$N_JOBS" SENSITIVITY_BAGS="$BAG" SENSITIVITY_RUNGS="xgb_tree_d1,xgb_tree_d2,xgb_tree_d3"
if [[ -n "${MAIN_K10_SENSITIVITY_RUNGS:-}" ]]; then
  export SENSITIVITY_RUNGS="$MAIN_K10_SENSITIVITY_RUNGS"
fi
export CBN_BAGS="$BAG" CBN_RUNGS="${CBN_RUNGS:-xgb_tree_d1,xgb_tree_d2,xgb_tree_d3}" CBN_INCLUDE_SINGLE=0
export XGB_NORM_BAGS="$BAG" XGB_NORM_RUNGS="xgb_tree_d1,xgb_tree_d2,xgb_tree_d3" XGB_NORM_N_JOBS="$N_JOBS"
export XGB_NORM_OUTPUT_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/$BAG"
export XGB_NORM_EVAL_ROOT="$RUNTIME/work/analysis_runs/$RUN_ID/normative_transfer/$BAG"
export XGB_NORM_BUNDLE_SUBDIR="results/analysis_runs/$RUN_ID/sensitivity_bundle/normative_transfer/$BAG"
export SINGLE_NORM_BAGS="$BAG" SINGLE_NORM_MODELS=xgb SINGLE_NORM_N_JOBS="$N_JOBS"
export SINGLE_NORM_OUTPUT_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/$BAG"
export SINGLE_NORM_EVAL_ROOT="$RUNTIME/work/analysis_runs/$RUN_ID/normative_transfer/$BAG/single_exposure"
export SINGLE_NORM_BUNDLE_SUBDIR="results/analysis_runs/$RUN_ID/sensitivity_bundle/normative_transfer/$BAG/single_exposure"
export SINGLE_NORM_EVAL_CHUNK_SIZE=63
export PAPER_FIG_ORDER_MAX=30
export V3_RESIDUALS_DIR="$RUNTIME/results/analysis_runs/$RUN_ID/main_statistics/residuals"
export MAIN_K10_SELECTED_OOF_ROOT="$RUNTIME/results/analysis_runs/paper_reanalysis_k10/main_statistics/model_comparison/oof"

if [[ "$PREFLIGHT" == "--preflight" ]]; then
  # Build exactly the compatibility runtime used by the CLI, then import the
  # selected parent module.  This validates the copied-module layout and all
  # imports without fitting, writing an analysis result, or contacting Slurm.
  module=""
  case "$STAGE" in
    country-block-null) module="scripts.run_country_block_null" ;;
    normative-transfer-xgb) module="scripts.run_xgb_normative_loco_pipeline" ;;
    normative-transfer-single-xgb) module="scripts.run_single_exposure_normative_pipeline" ;;
    domain-imbalance) module="scripts.run_domain_imbalance_sensitivity" ;;
    whole-exposome-pca) module="scripts.run_whole_exposome_pca_sensitivity" ;;
    country-region) module="scripts.run_country_region_sensitivity" ;;
    education-scanner-baseline) module="scripts.run_education_scanner_baseline_sensitivity" ;;
    residualized-bag) module="scripts.run_residualized_bag_sensitivity" ;;
    residualized-bag-clean) module="scripts.run_residualized_bag_clean_sensitivity" ;;
    residual-confounds) module="scripts.compute_residual_confounds" ;;
    diagnosis-balance) module="scripts.run_diagnosis_balance_sensitivity" ;;
  esac
  legacy_root="$(python - "$PROJECT_ROOT" "$RUNTIME" <<'PY'
from pathlib import Path
import sys
from repo_expo_hoi_bag.config.models import RuntimePaths
from repo_expo_hoi_bag.legacy_runtime import prepare_legacy_runtime

print(prepare_legacy_runtime(Path(sys.argv[1]), RuntimePaths.from_environment(sys.argv[2])))
PY
)"
  PYTHONPATH="$legacy_root:$PROJECT_ROOT/src:${PYTHONPATH:-}" \
    python -c "import importlib; importlib.import_module('$module')"
  printf 'Preflight passed: stage=%s bag=%s module=%s\n' "$STAGE" "$BAG" "$module"
  exit 0
fi

[[ -f "$ADAPTER_ROOT/main_k10_adapter_manifest.json" ]] || { echo "Missing prepared adapter: $ADAPTER_ROOT" >&2; exit 1; }

if [[ "$STAGE" == "education-scanner-baseline" ]]; then
  : "${SCANNER_XLSX_PATH:?SCANNER_XLSX_PATH is required for education-scanner-baseline}"
fi
if [[ "$STAGE" == "country-block-null" ]]; then
  : "${CBN_N_PERM:=10000}"
  # The parent correctness path delegates XGBoost parameters through the shared
  # evaluator, which validates the frozen k10 artifact for every held-out country.
  export CBN_N_PERM CBN_N_JOBS="$N_JOBS"
fi

echo "main k10 cluster stage=$STAGE bag=$BAG workers=$N_JOBS run=$RUN_ID"
python -m repo_expo_hoi_bag.cli --repro-data-root "$RUNTIME" \
  --analysis-run-id "${RUN_ID}_${STAGE//-/_}_${BAG}" run sensitivity "$STAGE"
