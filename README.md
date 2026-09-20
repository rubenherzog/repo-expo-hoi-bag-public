# Exposome HOI-BAG

Reproducible empirical analysis of exposome higher-order interactions and BAG
outcomes. The public release uses the deduplicated cohort, order cap 30,
structural and functional BAGs, OLS and XGBoost rungs d1-d3, and top-20
selection.

## Inputs and runtime

The required input files are versioned under data/. Generated results are
written outside this repository through REPRO_DATA_ROOT. Set that variable to
a writable directory before running any pipeline command.

    export REPRO_DATA_ROOT=/path/to/repro-data
    export PYTHONPATH=$PWD/src
    python -m repo_expo_hoi_bag.cli validate-data

### Selecting the external runtime

`REPRO_DATA_ROOT` is an external runtime selected by the operator. It is not a
directory inside this checkout, a YAML setting, or a path inferred from a
sibling checkout. Production commands must provide it explicitly either as an
environment variable or with the equivalent CLI argument:

    export REPRO_DATA_ROOT=/path/to/external/repro-data
    python -m repo_expo_hoi_bag.cli --repro-data-root "$REPRO_DATA_ROOT" validate-data

Do not substitute a sibling checkout, a local scratch directory, or a prior
ad-hoc runtime. The CLI creates its standard external `data/`, `results/`,
`work/`, and `figures/` subtrees under this root. New tuning signatures still
prevent a new run from overwriting an older run with different code or policy.

Nested-LOCO XGBoost tuning has one deliberate delivery exception: resumable
checkpoints remain at
`REPRO_DATA_ROOT/work/xgb_nested_loco_tuning/<run_signature>/`, while its
lightweight delivery files (`tuning.log`, `selected_xgb_configs.json`,
`manifest.json`, and `trial_diagnostics.parquet`) are written to
`outputs/xgb_nested_loco_tuning/<run_signature>/` in this checkout. A new run
always creates a new signature directory and rejects a collision.

The fixed intermediate-size HPO panel is versioned in
`config/xgb_hpo_feature_panels.yaml`. Its `domain_balanced_k10` scope contains
ten outcome-independent, size-10 subsets with one exposure from each canonical
domain; the exact panel and its hash are recorded in every tuning manifest.

The input cohort and feature metadata are listed in config/paper.yaml.

The complete prior-paper runtime is an immutable, external read-only reference
catalogued in `config/paper_reference.yaml`. It supplies the paper candidate
universe, OOF data, and all historical sensitivity artifacts on demand; its
paths and hashes are recorded by any comparison that reads them. The small
copy at `REPRO_DATA_ROOT/results/variant_a/paper_dedup_max30/canonical/` is a
verified backup only, not the default source. Neither is the older
`runs/oinfo_only/variant_a` universe a valid paper reference.

New main analyses require a deliberate, stable identifier and never write into
that reference bundle. For example:

    python -m repo_expo_hoi_bag.cli --repro-data-root "$REPRO_DATA_ROOT" --analysis-run-id hpo_optuna_gp_current3_v1 run greedy

This routes outputs to `results/analysis_runs/hpo_optuna_gp_current3_v1/` and
greedy work to `work/analysis_runs/hpo_optuna_gp_current3_v1/` under the same
canonical runtime.

Repository operating rules for contributors and agents are in [AGENTS.md](AGENTS.md), with the detailed [architecture contract](docs/REPOSITORY_CONTRACT.md) and [country-exclusion contract](docs/COUNTRY_EXCLUSION_CONTRACT.md).

## Production examples

Run the complete workflow:

    bash examples/run_full_production.sh

Run the deduplication sensitivity families, including negative-O and cap-21
views where available:

    bash examples/run_dedup_sensitivities.sh

Run the leave-one-exposure-out ablation sensitivity against the deduplicated
bundle (top-20 models are selected independently per BAG, model level and arm):

    SENSITIVITY_DEDUP=1 python -m repo_expo_hoi_bag.cli run sensitivity feature-ablation

Submit the full nested-LOCO XGBoost tuning job to the cluster (both primary
BAGs, rungs d1--d3, and 200 trials per BAG/country/rung):

    bash scripts/submit_xgb_nested_loco_tuning.sh

For the global country-balanced-MSE HPO used for the current analysis and its
five-country historical reproduction, submit the two jobs explicitly with:

    bash scripts/submit_xgb_global_loco_optuna_gp.sh

That launcher fixes Optuna-GP, 200 trials, and 40 CPU cores. It creates one
immutable run each for `historical_a` (France, Italy, Egypt, Greece, Poland)
and `a` (Egypt, Greece, Poland). Its six concurrent BAG×depth units split the
40-core allocation deterministically as 7/7/7/7/6/6 cores; each unit evaluates
its held-out countries in parallel, while trials remain sequential because GP
proposals depend on prior scores.

To continue those exact completed studies to 1,000 total trials without
refitting the first 200 observations, use the explicit continuation launcher:

    bash scripts/submit_xgb_global_loco_optuna_gp_extend1000.sh

It creates new immutable deliveries, records the parent signature and manifest
hash, and reads only the parent checkpoints under the canonical runtime.

The remaining endpoint HPO studies (baseline and all canonical single
exposures), each at 1,000 Optuna-GP trials for both current3 and historical5,
are prepared — but not submitted until the user runs — with:

    bash scripts/submit_xgb_hpo_extremes_optuna_gp_1000.sh

It creates four jobs, all with 40 cores and no manual-incumbent trial. For
single exposures it uses one XGBoost thread per fit and schedules independent
country×exposure fits across the allocated workers. The default time limit is
12 hours for baseline and 3 days for singles; `BASELINE_TIME_LIMIT` and
`SINGLE_TIME_LIMIT` override these explicitly.

The intermediate domain-balanced k=10 HPO, also at 1,000 Optuna-GP trials for
current3 and historical5, is prepared with:

    bash scripts/submit_xgb_hpo_domaink10_optuna_gp_1000.sh

It requests 40 cores and a three-day limit by default, uses the versioned panel
in `config/xgb_hpo_feature_panels.yaml`, and keeps the manual fixed vector out
of its optimization trials. `DOMAIN_K10_TIME_LIMIT` can override the limit.

Both launchers pass the canonical
`REPRO_DATA_ROOT` explicitly. It does not parallelize Bayesian-optimization
trials within a country: later trials depend on earlier scores. With the
current main-policy cohort there are 42 BAG/country units (24 structural and
18 functional), so the legacy nested procedure can run up to its requested
number concurrently. Scheduler stdout/stderr
are external under `REPRO_DATA_ROOT/work/xgb_nested_loco_tuning/`, and the
stage's readable scientific log and delivery artifacts remain under its new
local `outputs/xgb_nested_loco_tuning/<run_signature>/` directory.

The command line stages are validate-data, run greedy, run evaluate, run
analyses, render, and verify. Figure targets can be rendered selectively with
--target.

## Delivered results

Main figures and their source data are under outputs/figures/dedup/. Summary
tables and deduplicated sensitivity outputs are under outputs/dedup/ and
outputs/sensitivity/. The lightweight supplementary workbooks are under
results/.

Figure 2 is the cap-30 grid and includes both candidate set size and the
fraction of synergistic triplets. The manifest also covers the education/scanner,
residual-confounds, and residualized-BAG sensitivity figures.

Large generated intermediates, out-of-fold predictions, per-fold files,
temporary logs, and local runtime state are intentionally externalized through
REPRO_DATA_ROOT.

## Development checks

    python -m compileall -q src
    python -m pytest -q
    bash -n examples/run_full_production.sh examples/run_dedup_sensitivities.sh

The figure contract is stored in outputs/figures/manifest.yaml. Run verify
after rendering to check inputs, final figure references, and delivered source
data.
