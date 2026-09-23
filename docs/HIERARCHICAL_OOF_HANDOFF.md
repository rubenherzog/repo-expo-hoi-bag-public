# Hierarchical OOF improvement analyses — technical handoff

Reproducible paths and identifiers for the hierarchical out-of-fold analyses.
Written for the next agent picking this up; the science is summarised only where
it is needed to use a path correctly.

## Repository identifiers

| Key | Value |
|---|---|
| `REPO_ROOT` | `/home/rherzog/Documents/Brainlat/repo-expo-hoi-bag-public` |
| Branch | `main-k10-paper-delivery` |
| Commit | `9319ad8245f0396d35322184da6d6b38eaf36636` ("Consolidate portable main-k10 paper workflow") |
| `REPRO_DATA_ROOT` | `/data/workspaces/neuromodelling/rherzog/Brainlat/expo_hoi_bag_repro_data` |
| `SOURCE_RUN_ID` | `paper_reanalysis_k10` |
| Python | `/home/rherzog/miniconda3/envs/general/bin/python` |
| `PYTHONPATH` | `src:src/repo_expo_hoi_bag/core:src/repo_expo_hoi_bag/stages` |

Every stage listed below is **untracked** at that commit. Nothing here has been
committed yet.

### Operating rules for this work

- Everything heavy runs on the cluster through `run`; the user submits, never the
  agent. `run` does not exist on the dev node, so `--submit` is issued from the
  cluster login. `--dry-run` is local and only validates.
- Cluster allocations follow the existing jobs: `40|160` cores/memory.
- joblib backend is always `loky`.
- Logs live in `logs/<RUN_ID>/` inside the checkout; heavy generated output lives
  under `$REPRO_DATA_ROOT/work/analysis_runs/<RUN_ID>/...` and never in `outputs/`
  (the checkout filesystem sits at ~97% full).

---

## 1. Hierarchical bootstrap ΔR² analysis

Country-balanced incremental R² of the Top-K candidate region over the covariate
baseline, on the R² scale, by nested resampling. No model fitting.

### Scripts

```
src/repo_expo_hoi_bag/stages/compute_hierarchical_oof_bootstrap.py
src/repo_expo_hoi_bag/stages/plot_hierarchical_oof_bootstrap.py
src/repo_expo_hoi_bag/stages/build_hierarchical_oof_bootstrap_table.py
scripts/jobs/run_hierarchical_oof_bootstrap.sh
scripts/submit_hierarchical_oof_bootstrap.sh
tests/test_hierarchical_oof_bootstrap.py                     # 30 tests
```

### Outputs

```
outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap/
├── hierarchical_oof_bootstrap_results.csv        # 72 rows = 2 BAG x 2 arm x 2 rung x 9 regions
├── hierarchical_oof_bootstrap_by_country.csv     # per-country ΔR², SSE_baseline, SSE_candidate, SST
└── hierarchical_oof_bootstrap_manifest.json
```

`hierarchical_oof_bootstrap_results.csv` columns:
`bag, arm, rung, region, K, observed_delta_r2, bootstrap_median, ci_lo, ci_hi,
fraction_bootstrap_positive, p_one_sided, n_countries, n_subjects, holm_p,
multiplicity_family`.

The Top-20 primary family is `region == "Top20"`; only those rows carry a
non-null `holm_p`.

**Bootstrap replicates are not persisted**, only their summaries. They are exactly
reproducible from `bootstrap_cell(..., seed=20260922)`, which is invariant to
`n_jobs`.

### Table and figure

```
outputs/main/paper/complete/tables/Supplementary_Table_Hierarchical_OOF_Bootstrap.xlsx
    sheets: "Top20 primary" (ST27), "K-grid sensitivity" (ST28)
outputs/main/paper/complete/tables/source_data/hierarchical_oof/hierarchical_oof_bootstrap_top20.csv
outputs/main/paper/complete/tables/source_data/hierarchical_oof/hierarchical_oof_bootstrap_k_grid.csv

outputs/main/paper/complete/figures/supplementary/fig_hierarchical_oof_bootstrap_k_curve.{png,pdf,svg,tiff}
outputs/main/paper/complete/figures/supplementary/source_data/fig_hierarchical_oof_bootstrap_k_curve_source_data.xlsx
```

The delivered CSVs and figure currently cover **d2/d3 only**. Re-run the bootstrap
once the d1 cells land (see §4) so OLS and d1 rows appear.

---

## 2. Subject-level OOF data

Authoritative per-subject loss accumulators, RUN_ID `hierarchical_oof_kgrid`:

```
$REPRO_DATA_ROOT/work/analysis_runs/hierarchical_oof_kgrid/hierarchical_oof/loss_cache/
├── all__{structural,functional}__{o_min,o_max}__{ols,xgb_tree_d1,xgb_tree_d2,xgb_tree_d3}.parquet
├── all__..._cumulative.npz
└── fit_manifest_all.json
```

Parquet schema — one row per **subject x region**, 9 regions per cell:

```
row_id, N_MEGA, country, y_true, baseline_loss, mean_candidate_loss,
median_candidate_loss, n_candidate_predictions, region, bag, arm, rung,
mean_improvement, median_improvement
```

Structural: 10,847 subjects / 22 countries. Functional: 9,441 / 16.

> **`country_year` is not stored in the parquet.** Join it by `row_id` with
> `compute_hierarchical_oof_improvement.country_year_map(bag)` — 174 structural and
> 150 functional country-years, no nulls. The nested bootstrap needs this column.

Candidate-level OOF prediction cache from the earlier cross-selected analysis
(partial: 34 parquets covering 48 candidate-folds, not a full grid):

```
outputs/sensitivity/cross_selected_baseline/main_k10/oof_cache/
```

Consolidated subject scores from the earlier mixed-model run (d2/d3 only):

```
outputs/sensitivity/hierarchical_oof_improvement/main_k10/hierarchical_oof_subject_scores.parquet
```

---

## 3. Fine K grid

```python
NESTED_K = (5, 10, 15, 20, 30, 40, 50, 100)   # plus "ALL" = 560 candidates per arm
PRIMARY_K = 20
```

Defined in `src/repo_expo_hoi_bag/stages/run_hierarchical_oof_predictions.py`.

### Any K without refitting

Each cell stores `all__<bag>__<arm>__<rung>__cumulative.npz` (~40–46 MB) with keys
`cumulative` (560 x n_subjects, **float64**), `row_id`, `country`, `y_true`,
`baseline_loss`, `candidate_order`.

```
mean_loss_TopK[i] = cumulative[K - 1, i] / K
```

Candidates are stored in global LOCO rank order, so any Top-K is a prefix and the
identity is exact for every K from 1 to 560. float64 is deliberate: float32 would
inject ~3e-7 into every derived mean.

### Command used

```bash
export REPRO_DATA_ROOT=/data/workspaces/neuromodelling/rherzog/Brainlat/expo_hoi_bag_repro_data
export HIER_OOF_RUN_ID=hierarchical_oof_kgrid
bash scripts/submit_hierarchical_oof_cells.sh --dry-run
bash scripts/submit_hierarchical_oof_cells.sh --submit
```

---

## 4. OLS and d1 cells

Script `scripts/jobs/run_hierarchical_oof_cell.sh`, submitted by
`scripts/submit_hierarchical_oof_cells.sh`, RUN_ID `hierarchical_oof_kgrid`.

| Job IDs | Cells | Status at handoff |
|---|---|---|
| 36684107–36684110 | OLS x 4 | complete (parquet + npz written) |
| 36684111–36684114 | d1 x 4 | running on slurm11–14 |

Expected outputs:
`…/loss_cache/all__{bag}__{arm}__xgb_tree_d1.parquet` and `…__cumulative.npz`.

Check completion:

```bash
squeue -u rherzog
ls $REPRO_DATA_ROOT/work/analysis_runs/hierarchical_oof_kgrid/hierarchical_oof/loss_cache/*xgb_tree_d1*.parquet | wc -l   # expect 4
tail -2 logs/hierarchical_oof_kgrid/hier_oof_*_d1_*.out                                                                   # expect "Saved:"
```

A cell writes only when all 560 candidates finish, so there is no partial output.
Resubmitting skips completed cells.

### OLS is a different estimator

OLS is not "XGBoost without a config". The production design is
intercept + age + sex dummies + year spline + diagnosis dummies + exposome +
diagnosis x exposome interactions, built by
`sensitivity_common._build_canonical_ols_design` and gated on
`sensitivity_common._ols_dx_interactions()`. `fit_cell` imports both rather than
reimplementing them. Recomputed OLS R² matches the delivered
`metrics_country.csv` to ~2.5e-10.

---

## 5. d2/d3 cells (already complete)

Same directory and schema as §2:

```
$REPRO_DATA_ROOT/work/analysis_runs/hierarchical_oof_kgrid/hierarchical_oof/loss_cache/
    all__{structural,functional}__{o_min,o_max}__{xgb_tree_d2,xgb_tree_d3}.parquet
```

`compute_hierarchical_oof_improvement.load_subjects(cache_dir, country_year)`
already concatenates every cell and attaches country-year, so combining OLS/d1
with d2/d3 needs no new code.

A superseded d2/d3-only copy exists under RUN_ID `hierarchical_oof_main_k10`
(8 parquets, **no npz**). Keep it for provenance; do not analyse from it.

---

## 6. Fig. S13 model-complexity figure

```
Generator:  src/repo_expo_hoi_bag/stages/plot_hpo_complexity_comparison_heatmaps.py
              SPECS (L27) — the five panel columns per BAG row
Parent:     src/repo_expo_hoi_bag/stages/plot_complexity_comparison_heatmaps.py
              _draw_heatmap (L277), _cell_value (L126), COMMON_COLOR_LIMIT = 0.06 (L53)
Style:      src/repo_expo_hoi_bag/figures/style.py
              SYN_COLOR, RED_COLOR, LEVEL_LABELS, LEVEL_AXIS_LABEL,
              BAG_ROW_ORDER, BAG_LABELS, ROW_LETTERS, style_axis
Source data helper: src/repo_expo_hoi_bag/figures/source_data.py  (Panel, write_source_data)
```

Input CSV (external runtime):

```
$REPRO_DATA_ROOT/results/analysis_runs/paper_reanalysis_k10/main_statistics/model_comparison/complexity_and_arm_comparisons.csv
```

Delivered outputs:

```
outputs/main/paper/complete/figures/supplementary/fig_s13_model_complexity_comparisons_hpo_k10.{png,pdf,svg,tiff}
outputs/main/paper/complete/figures/supplementary/source_data/fig_s13_model_complexity_comparisons_hpo_k10_source_data.xlsx
    plus panel CSVs a1–a5 (structural) and b1–b5 (functional)
```

Regenerate:

```bash
python -m repo_expo_hoi_bag.stages.plot_hpo_complexity_comparison_heatmaps \
  --repro-data-root "$REPRO_DATA_ROOT" --hpo-set k10 --output-dir <dir>
```

Its default `--output-dir` is `outputs/figures/HPO/k10/supplementary`, **not** the
delivered supplementary path above. Pass `--output-dir` explicitly to refresh the
delivered copy.

---

## 7. Bootstrap implementation

`src/repo_expo_hoi_bag/stages/compute_hierarchical_oof_bootstrap.py`

| Symbol | Line | Role |
|---|---|---|
| `bootstrap_cell` | 117 | countries fixed → resample country-years → resample subjects |
| `_country_year_blocks` | 84 | block index per country-year, with its owning country |
| `observed_delta_r2` | 96 | per-country `(SSE_base − SSE_cand)/SST` |
| `summarise` | 176 | CI, fraction > 0, one-sided P |
| `apply_holm` | 192 | `multipletests(method="holm")` |
| `DRAWS = 10_000` | 63 | replicates |
| `SEED = 20260922` | 64 | seed |

- **Estimand**: `mean_c[(SSE_baseline_c − mean_SSE_TopK_c) / SST_c]`, equal weight
  per country — the canonical country-balanced quantity.
- **Seeding**: `np.random.SeedSequence(seed).spawn(draws)` gives each replicate its
  own seed, so results are identical for any `n_jobs`. A test enforces this.
- **Holm**: within each BAG across exactly the four
  `Top20 x {o_min, o_max} x {d2, d3}` tests. `PRIMARY_RUNGS = ("xgb_tree_d2",
  "xgb_tree_d3")` keeps OLS and d1 out of the family; the K grid and ALL are
  reported uncorrected.
- **One-sided P**: `(Σ[est ≤ 0] + 1) / (B + 1)`, so it is never exactly zero.
- Measured runtime: 1.8 ms per replicate on a real cell; the 72-cell grid takes
  about 9 minutes.

---

## Known discrepancy, already investigated

Recomputed per-country R² matches the delivered `metrics_country.csv` to ~1e-3 for
the XGB levels and ~1e-10 for OLS. The residual XGB gap is **not** introduced by
this stage: the production helper `sensitivity_common._fit_candidate`, invoked with
the same frozen configuration, departs from the same delivered CSV by ~9e-2 — two
orders of magnitude more. The delivered metrics were produced through the main-k10
HPO routing, which this stage does not replicate because it needs subject-level
predictions rather than aggregated metrics.

What matters here is the prediction path, and that path was verified bit-exact
(max |diff| = 0.0) against the delivered subject-level OOF store before any cell
was fitted. This is recorded in the module docstring of
`run_hierarchical_oof_predictions.py`.

---

## Unrelated pre-existing failure

`tests/test_hardcoding_guardrails.py::test_active_sources_do_not_contain_personal_absolute_paths`
fails on `src/repo_expo_hoi_bag/stages/update_education_no_scanner_tables.py`, an
**untracked** file that is not part of this work. Leave it alone. The rest of the
suite passes (249 tests).

---

## Codex starting point

Inspect in this order:

1. `src/repo_expo_hoi_bag/stages/compute_hierarchical_oof_bootstrap.py`
2. `src/repo_expo_hoi_bag/stages/run_hierarchical_oof_predictions.py`
3. `$REPRO_DATA_ROOT/work/analysis_runs/hierarchical_oof_kgrid/hierarchical_oof/loss_cache/`
4. `outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap/hierarchical_oof_bootstrap_results.csv`
5. `src/repo_expo_hoi_bag/stages/compute_hierarchical_oof_improvement.py`
6. `scripts/submit_hierarchical_oof_cells.sh`
7. `src/repo_expo_hoi_bag/stages/plot_hpo_complexity_comparison_heatmaps.py`
8. `src/repo_expo_hoi_bag/figures/style.py`
9. `AGENTS.md`

### Immediate next step

Once the four d1 cells finish (§4), re-run the bootstrap so OLS and d1 enter the
K-grid outputs:

```bash
export REPRO_DATA_ROOT=/data/workspaces/neuromodelling/rherzog/Brainlat/expo_hoi_bag_repro_data
export HIER_OOF_RUN_ID=hierarchical_oof_kgrid
bash scripts/submit_hierarchical_oof_bootstrap.sh --dry-run
bash scripts/submit_hierarchical_oof_bootstrap.sh --submit
```

That job also rebuilds the table and the figure. The Top-20 Holm family stays at
four d2/d3 tests per BAG; OLS and d1 are reported as descriptive levels.
