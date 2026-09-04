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

The input cohort and feature metadata are listed in config/paper.yaml.

## Production examples

Run the complete workflow:

    bash examples/run_full_production.sh

Run the deduplication sensitivity families, including negative-O and cap-21
views where available:

    bash examples/run_dedup_sensitivities.sh

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
