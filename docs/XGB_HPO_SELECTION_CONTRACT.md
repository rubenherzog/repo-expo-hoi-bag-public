# XGBoost HPO final-selection contract

Exploratory XGBoost HPO checkpoints may contain more trials than the paper uses.
Only derived, immutable `selected_xgb_configs.json` artifacts produced by the
`xgb-hpo-cap500-selection` stage are paper-facing. Downstream re-analysis must
not select a new best trial directly from an exploratory checkpoint.

## Final rule

`config/xgb_hpo_selection.yaml` is the sole versioned definition.

1. When a global BAG × rung checkpoint has at least 500 completed trials,
   select the lowest `mean_country_MSE` among trials 0--499. Later trials are
   exploratory and ineligible for paper selection.
2. Below 500, selection is permitted only after practical convergence: at
   least 350 trials, and incumbent improvement below 0.10 mean-country MSE
   over the preceding 300 trials. Select the lowest-MSE observed trial.
3. A checkpoint below 500 that fails this condition is ineligible.

The freeze artifact records source hashes, observed trials, selection limit,
reason, incumbent improvement, country policy, and rule hash. It never changes
the source checkpoints.

## Released paper-selection artifacts

The following immutable local artifacts implement this rule for the current
paper re-analysis. Each contains `selected_xgb_configs.json`, `manifest.json`,
`selection_audit.json`, and `selection.log`. Consumers must use the selected
configuration file from this table, never a raw checkpoint or an unlisted
scratch/provisional directory.

| Country policy | Feature scope | Artifact directory |
| --- | --- | --- |
| `historical_a` (France, Italy, Egypt, Greece, Poland excluded) | baseline | `outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/` |
| `historical_a` | full exposome (63) | `outputs/xgb_nested_loco_tuning/historical5_full63_cap500_final/` |
| `historical_a` | best single exposure | `outputs/xgb_nested_loco_tuning/historical5_single_cap500/` |
| `historical_a` | domain-balanced k10 | `outputs/xgb_nested_loco_tuning/historical5_k10_cap500/` |
| `a` (Egypt, Greece, Poland excluded) | baseline | `outputs/xgb_nested_loco_tuning/current3_baseline_cap500/` |
| `a` | full exposome (63) | `outputs/xgb_nested_loco_tuning/current3_full63_cap500/` |
| `a` | best single exposure | `outputs/xgb_nested_loco_tuning/current3_single_cap500/` |
| `a` | domain-balanced k10 | `outputs/xgb_nested_loco_tuning/current3_k10_cap500/` |

`current3_single_cap500` has one permitted early selection: structural d1 had
389 completed trials, its best trial was 183 (one-based), and its incumbent
improvement over the preceding 300 trials was 0.002532 mean-country MSE. This
is below the configured 0.10 threshold. Every other BAG × rung in the table
was selected from its first 500 trials.

## Scope

The objective is equally weighted held-out-country MSE. A trial is one complete
Optuna proposal, not an XGBoost boosting iteration. Practical convergence is an
operational stopping rule, not a claim of mathematical convergence.
