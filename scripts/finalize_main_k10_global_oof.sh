#!/usr/bin/env bash
# Finalize the global pooled OOF R² delivery: normative figures, complexity
# comparison, supplementary tables and the delivery manifest.
#
# It is the estimand counterpart of finalize_main_k10_table_alignment.sh and
# differs from it only by R2_MODE=global_oof and by writing into the separate
# global delivery root.  It never writes into the country-balanced delivery.
#
# Requires the global ST04 country-block null to be complete (see
# scripts/submit_main_k10_global_oof_country_block_null.sh).  Set
# ALLOW_MISSING_ST04=1 to build everything else first and record ST04 as
# pending; the delivery verifier will then still report it as incomplete.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_GLOBAL_RUN_ID:-main_k10_global_oof_20260920}"
SOURCE_RUN_ID="${MAIN_K10_SOURCE_CLUSTER_RUN_ID:-main_k10_release_20260916}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CBN_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/country_block_null"

for bag in structural functional; do
  result="$CBN_ROOT/${bag}_winner_pvalues.csv"
  if [[ ! -f "$result" ]]; then
    if [[ "${ALLOW_MISSING_ST04:-0}" == "1" ]]; then
      echo "WARNING: global ST04 result is not yet available: $result" >&2
      continue
    fi
    echo "Missing completed global ST04 result: $result" >&2
    echo "Submit it with scripts/submit_main_k10_global_oof_country_block_null.sh, or set ALLOW_MISSING_ST04=1." >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$result" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
estimands = set(frame.get("r2_estimand", pd.Series(dtype=str)).astype(str))
if estimands != {"global_oof"}:
    raise ValueError(f"Not a global-OOF ST04 result ({estimands}): {sys.argv[1]}")
if set(frame.get("n_perm", pd.Series(dtype=int)).astype(int)) != {10000}:
    raise ValueError(f"ST04 result does not contain 10,000 permutations: {sys.argv[1]}")
PY
done

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
export REPRO_DATA_ROOT="$RUNTIME"
# The estimand selector.  Every stage below resolves it through
# sensitivity_common.r2_mode(); nothing else about the analysis changes.
export R2_MODE=global_oof

# Heavy normative evaluations are estimand-invariant and are reused from the
# completed country-balanced run; only the selection and reporting change.
export NORMATIVE_TRANSFER_SOURCE_DIR="$RUNTIME/work/analysis_runs/$SOURCE_RUN_ID/normative_transfer"
export NORMATIVE_TRANSFER_OUTPUT_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/stats"
export NORM_XGB_ROOT="$NORMATIVE_TRANSFER_SOURCE_DIR"
export NORM_OLS_ROOT="$RUNTIME/ols_normative_loco"
export NORM_SINGLE_NORMATIVE_ROOT="$NORMATIVE_TRANSFER_SOURCE_DIR"
export NORM_POOLED_CANONICAL_ROOT="$RUNTIME/results/analysis_runs/$RUN_ID/input_adapter"
export NORM_POOLED_OLS_CANONICAL_ROOT="$RUNTIME/results/variant_a/paper_dedup_max30/canonical"
export NORM_POOLED_MAIN_RUN_ROOT="$RUNTIME/results/analysis_runs/paper_reanalysis_k10"
export SUBCOMB_BASELINE_PARQUET="$RUNTIME/work/subcomb_oinfo/subcomb_baseline.parquet"
export REPRO_FIGURES_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/figures"
export NORM_TRANSFER_STATS_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/plot_stats"
export NORM_DIVERSITY_STATS_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_diversity"
export NORM_FIG_STEM_TEMPLATE='normative_transfer_{bag}{suffix}'
export NORM_DIVERSITY_FIG_STEM=normative_diversity
export NORM_POOLED_RUNG=xgb_tree_d3
export NORM_DIVERSITY_RUNG=xgb_tree_d3
mkdir -p "$REPRO_FIGURES_ROOT"

# --output-run-id keeps the global comparisons in their own namespace; without
# it this stage would overwrite the country-balanced file in paper_reanalysis_k10.
"$PYTHON_BIN" -m repo_expo_hoi_bag.stages.compute_hpo_model_comparisons \
  --repro-data-root "$RUNTIME" --hpo-set k10 --source-run-id paper_reanalysis_k10 \
  --output-run-id "$RUN_ID" \
  --draws 10000 --n-jobs "${MAIN_K10_STATS_WORKERS:-20}"
"$PYTHON_BIN" -m repo_expo_hoi_bag.stages.plot_hpo_complexity_comparison_heatmaps \
  --repro-data-root "$RUNTIME" --hpo-set k10 \
  --output-dir "$REPRO_FIGURES_ROOT"
"$PYTHON_BIN" -m repo_expo_hoi_bag.stages.compute_normative_transfer_stats
"$PYTHON_BIN" -m repo_expo_hoi_bag.stages.plot_normative_transfer_grid
"$PYTHON_BIN" -m repo_expo_hoi_bag.stages.plot_normative_diversity_r2

printf 'Finalized the global-OOF normative figures and statistics for run %s\n' "$RUN_ID"
