#!/usr/bin/env bash
# Build the adapter once on the cluster login node.  It performs no model fit.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:?Set a new immutable MAIN_K10_CLUSTER_RUN_ID}"
PAPER_REFERENCE_ROOT="${PAPER_REFERENCE_ROOT:?Set PAPER_REFERENCE_ROOT to the read-only historical runtime}"
REGISTRY="${MAIN_K10_CANDIDATE_REGISTRY:-$PAPER_REFERENCE_ROOT/results/variant_a/families/pooled_oinfo_ladder/canonical/candidate_registry.parquet}"
EFFECTIVE_SOURCE="$PROJECT_ROOT/data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"
FEATURE_DOMAINS="$PROJECT_ROOT/data/metadata/exposome_feature_domains.csv"
DOMAIN_PC1_OINFO_REFERENCE="${MAIN_K10_DOMAIN_PC1_OINFO_REFERENCE:-$PAPER_REFERENCE_ROOT/sensitivity/domain_imbalance_dedup/domain_imbalance/global_hoi_cache/within_domain_pc1_thoi_scores.csv}"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
python -m repo_expo_hoi_bag.stages.prepare_main_k10_sensitivity_adapter \
  --repro-data-root "$RUNTIME" --source-run-id paper_reanalysis_k10 \
  --candidate-registry "$REGISTRY" --adapter-run-id "$RUN_ID" \
  --effective-exposome-source "$EFFECTIVE_SOURCE" --feature-domains "$FEATURE_DOMAINS" \
  --domain-pc1-oinfo-reference "$DOMAIN_PC1_OINFO_REFERENCE"
python -m repo_expo_hoi_bag.stages.prepare_main_k10_residual_inputs \
  --repro-data-root "$RUNTIME" --source-run-id paper_reanalysis_k10 \
  --adapter-run-id "$RUN_ID"
