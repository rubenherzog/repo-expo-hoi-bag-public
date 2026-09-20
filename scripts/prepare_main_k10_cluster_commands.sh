#!/usr/bin/env bash
# Print-only cluster handoff.  It never submits or executes a scheduler job.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_cluster_k10_20260915_v1}"
JOB="$PROJECT_ROOT/scripts/jobs/run_main_k10_sensitivity_bag.sh"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_ID"

cat <<EOF
# Main k10 cluster handoff — print-only; review then run manually.
# New output namespace: $RUN_ID
# Frozen HPO: $PROJECT_ROOT/outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json
# OLS: never refitted; excluded from every job below.

export REPRO_DATA_ROOT=$RUNTIME
export MAIN_K10_CLUSTER_RUN_ID=$RUN_ID
export SCANNER_XLSX_PATH=$PROJECT_ROOT/data/metadata/Megabase_ODQ_scanners.xlsx
export PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:\${PYTHONPATH:-}
mkdir -p $LOG_ROOT
bash $PROJECT_ROOT/scripts/prepare_main_k10_cluster_adapter.sh

# ST04 — country-block permutation null.  Two independent BAG jobs.
CBN_N_PERM=10000 run -t 24:00 -j main_k10_cbn_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_cbn_structural_%j.out -e $LOG_ROOT/main_k10_cbn_structural_%j.err bash $JOB country-block-null structural 40
CBN_N_PERM=10000 run -t 24:00 -j main_k10_cbn_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_cbn_functional_%j.out -e $LOG_ROOT/main_k10_cbn_functional_%j.err bash $JOB country-block-null functional 40

# ST11 / normative context — XGB k10 only, one BAG per job.
run -t 24:00 -j main_k10_norm_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_norm_structural_%j.out -e $LOG_ROOT/main_k10_norm_structural_%j.err bash $JOB normative-transfer-xgb structural 40
run -t 24:00 -j main_k10_norm_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_norm_functional_%j.out -e $LOG_ROOT/main_k10_norm_functional_%j.err bash $JOB normative-transfer-xgb functional 40

# ST13 — alternative representations.  Domain representative and PCA are independent.
run -t 24:00 -j main_k10_domain_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_domain_structural_%j.out -e $LOG_ROOT/main_k10_domain_structural_%j.err bash $JOB domain-imbalance structural 40
run -t 24:00 -j main_k10_domain_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_domain_functional_%j.out -e $LOG_ROOT/main_k10_domain_functional_%j.err bash $JOB domain-imbalance functional 40
run -t 24:00 -j main_k10_pca_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_pca_structural_%j.out -e $LOG_ROOT/main_k10_pca_structural_%j.err bash $JOB whole-exposome-pca structural 40
run -t 24:00 -j main_k10_pca_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_pca_functional_%j.out -e $LOG_ROOT/main_k10_pca_functional_%j.err bash $JOB whole-exposome-pca functional 40

# ST16 — LORO region refit; country-LOCO rows come from the main k10 adapter.
run -t 24:00 -j main_k10_loro_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_loro_structural_%j.out -e $LOG_ROOT/main_k10_loro_structural_%j.err bash $JOB country-region structural 40
run -t 24:00 -j main_k10_loro_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_loro_functional_%j.out -e $LOG_ROOT/main_k10_loro_functional_%j.err bash $JOB country-region functional 40

# ST17 — covariate and residualized-target refits.
run -t 12:00 -j main_k10_edu_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_edu_structural_%j.out -e $LOG_ROOT/main_k10_edu_structural_%j.err bash $JOB education-scanner-baseline structural 40
run -t 12:00 -j main_k10_edu_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_edu_functional_%j.out -e $LOG_ROOT/main_k10_edu_functional_%j.err bash $JOB education-scanner-baseline functional 40
run -t 24:00 -j main_k10_residtarget_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_residtarget_structural_%j.out -e $LOG_ROOT/main_k10_residtarget_structural_%j.err bash $JOB residualized-bag structural 40
run -t 24:00 -j main_k10_residtarget_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_residtarget_functional_%j.out -e $LOG_ROOT/main_k10_residtarget_functional_%j.err bash $JOB residualized-bag functional 40

# ST15 — equal-diagnosis weighting, selected main d3 model only.
run -t 12:00 -j main_k10_diagbalance_structural -c 40 -m 160 -o $LOG_ROOT/main_k10_diagbalance_structural_%j.out -e $LOG_ROOT/main_k10_diagbalance_structural_%j.err bash $JOB diagnosis-balance structural 40
run -t 12:00 -j main_k10_diagbalance_functional -c 40 -m 160 -o $LOG_ROOT/main_k10_diagbalance_functional_%j.out -e $LOG_ROOT/main_k10_diagbalance_functional_%j.err bash $JOB diagnosis-balance functional 40

# ST12 residual-confounds is already completed and delivered locally; it is
# deliberately omitted to avoid a duplicate bootstrap run.

# ST14 and ST18 are selection/re-aggregation analyses, already prepared locally;
# they require no cluster refit.  Fig. 2 and Fig. 3 are intentionally absent.
EOF
