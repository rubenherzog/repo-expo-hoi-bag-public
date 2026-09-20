#!/usr/bin/env bash
# Finalize corrected table/figure alignment after both ST04 cluster jobs finish.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"
RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"
CBN_ROOT="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/country_block_null"

for bag in structural functional; do
  result="$CBN_ROOT/${bag}_winner_pvalues.csv"
  [[ -f "$result" ]] || { echo "Missing completed ST04 result: $result" >&2; exit 1; }
  python - "$result" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
if set(frame.get("r2_estimand", pd.Series(dtype=str)).astype(str)) != {"country_balanced"}:
    raise ValueError(f"Not a corrected country-balanced ST04 result: {sys.argv[1]}")
if set(frame.get("n_perm", pd.Series(dtype=int)).astype(int)) != {10000}:
    raise ValueError(f"ST04 result does not contain 10,000 permutations: {sys.argv[1]}")
PY
done

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/repo_expo_hoi_bag/core:$PROJECT_ROOT/src/repo_expo_hoi_bag/stages:${PYTHONPATH:-}"
export REPRO_DATA_ROOT="$RUNTIME"
export NORMATIVE_TRANSFER_SOURCE_DIR="$RUNTIME/work/analysis_runs/$RUN_ID/normative_transfer"
export NORMATIVE_TRANSFER_OUTPUT_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/stats"
export NORM_XGB_ROOT="$NORMATIVE_TRANSFER_SOURCE_DIR"
export NORM_OLS_ROOT="$RUNTIME/ols_normative_loco"
export NORM_SINGLE_NORMATIVE_ROOT="$NORMATIVE_TRANSFER_SOURCE_DIR"
export NORM_POOLED_CANONICAL_ROOT="$RUNTIME/results/analysis_runs/$RUN_ID/input_adapter"
export NORM_POOLED_OLS_CANONICAL_ROOT="$RUNTIME/results/variant_a/paper_dedup_max30/canonical"
export NORM_POOLED_MAIN_RUN_ROOT="$RUNTIME/results/analysis_runs/paper_reanalysis_k10"
export SUBCOMB_BASELINE_PARQUET="$RUNTIME/work/subcomb_oinfo/subcomb_baseline.parquet"
export REPRO_FIGURES_ROOT="$PROJECT_ROOT/outputs/main/paper/complete/figures/supplementary"
export NORM_TRANSFER_STATS_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_transfer/plot_stats"
export NORM_DIVERSITY_STATS_DIR="$PROJECT_ROOT/outputs/main/$RUN_ID/sensitivity/normative_diversity"
export NORM_FIG_STEM_TEMPLATE='normative_transfer_{bag}{suffix}'
export NORM_DIVERSITY_FIG_STEM=normative_diversity
export NORM_POOLED_RUNG=xgb_tree_d3
export NORM_DIVERSITY_RUNG=xgb_tree_d3

python -m repo_expo_hoi_bag.stages.compute_hpo_model_comparisons \
  --repro-data-root "$RUNTIME" --hpo-set k10 --source-run-id paper_reanalysis_k10 \
  --draws 10000 --n-jobs "${MAIN_K10_STATS_WORKERS:-20}"
python -m repo_expo_hoi_bag.stages.plot_hpo_complexity_comparison_heatmaps \
  --repro-data-root "$RUNTIME" --hpo-set k10 \
  --output-dir "$REPRO_FIGURES_ROOT"
python -m repo_expo_hoi_bag.stages.compute_normative_transfer_stats
python -m repo_expo_hoi_bag.stages.plot_normative_transfer_grid
python -m repo_expo_hoi_bag.stages.plot_normative_diversity_r2
python -m repo_expo_hoi_bag.stages.finalize_main_k10_delivery \
  --repro-data-root "$RUNTIME"
python -m repo_expo_hoi_bag.stages.verify_main_paper_delivery --refresh-manifest

printf 'Finalized the country-balanced figures, supplementary tables, parent statistical tests, and manifest in the existing main-k10 delivery.\n'
