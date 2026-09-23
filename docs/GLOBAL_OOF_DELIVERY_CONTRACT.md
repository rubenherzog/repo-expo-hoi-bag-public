# Global pooled OOF R² delivery contract

The main k10 paper is delivered under two R² estimands computed over **the same**
OOF predictions, folds, cohort, exclusions, diagnostic definitions and candidate
sets. Only the aggregation of the out-of-fold residuals differs:

| Estimand | Definition | Delivery root |
| --- | --- | --- |
| `country_balanced` | unweighted mean of the per-held-out-country R² | `outputs/main/paper/complete` |
| `global_oof` | ordinary R² applied once to the concatenated valid OOF predictions | `outputs/main/paper/complete_global_oof` |

The country-balanced delivery is the established one and is never modified by a
global run.

## Selecting the estimand

`R2_MODE` is the single switch (`sensitivity_common.r2_mode()`); it defaults to
`country_balanced`, and an unknown value is rejected rather than defaulted.
Stages that predate it expose the equivalent `--r2-estimator
{country-balanced,global-oof}` flag. Global-mode artifacts live in sibling
paths (`oof_global_oof/`, `fig3_diversity_d3_global/`,
`level_winners_global_oof.csv`, a separate `--output-run-id` namespace) so a
global run can never overwrite a country-balanced one.

`global_oof_r2` never averages R² across countries, folds or subgroups. Where a
secondary analysis deliberately clusters by country — the country-cluster
bootstrap and the paired country Wilcoxon tests — the resampling/pairing unit
stays the country under both estimands; only the statistic recomputed on each
resample changes.

## Hyperparameters are estimand-independent

All three frozen HPO scopes (`baseline`, `single_exposure`,
`domain_balanced_k10`) were selected on `mean_country_mse`, an MSE objective.
The R² definition therefore plays no part in hyperparameter selection, and both
deliveries reuse the identical frozen artifacts, kept separate per scope.

## Column name-shadowing

Several stages carry the selected estimand in a column literally named
`country_balanced_r2`, and `run_whole_exposome_pca_sensitivity` assigns
country-balanced values into a column named `global_oof_r2`. **Never audit these
artifacts by column name alone**; check the emitted `r2_mode` / `r2_estimand`
field or the producing stage.

## Required artifacts

`config/main_paper_delivery_global_oof.yaml` carries the same 16-figure and
18-sheet inventory as the country-balanced contract and adds `r2_mode:
global_oof`. `verify_main_paper_delivery.py --config <that file>` must pass
before handoff, and `PROVENANCE.json` records the run ids, config/input hashes,
HPO sources and final status.

## Cluster dependency

ST04 (country-block permutation null) is the only analysis whose statistic must
be recomputed on the cluster, because its null distribution is built from the
test statistic itself. Submit it with
`scripts/submit_main_k10_global_oof_country_block_null.sh`; every other global
artifact is produced locally without refitting a model, except the 56
level-winner OOF fits and the ST15 diagnosis-balance refit, whose selected
candidates change with the estimand.
