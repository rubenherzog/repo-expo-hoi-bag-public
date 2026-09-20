# Main k10 paper delivery contract

The complete lightweight paper delivery lives under
`outputs/main/paper/complete/`.  It is the sole handoff location for the main
k10 paper and contains main figures, supplementary figures, source data, and
the complete 18-sheet supplementary-table workbook.

`REPRO_DATA_ROOT` contains only heavy runtime products: participant-level OOF,
fold-level metrics, bootstrap/permutation draws, candidate matrices,
checkpoints, logs and compatibility-runtime copies.  A runtime product is not
a paper delivery artifact.

## Required layout

```
outputs/main/paper/complete/
  figures/main/                 # Figure 2, Figure 3, Figure 4 + source_data/
  figures/supplementary/        # each included supplementary figure + source_data/
  tables/Supplementary_Tables_main_k10.xlsx
  tables/source_data/           # lightweight CSV/XLSX inputs for ST01--ST18
  MANIFEST.csv
```

The exhaustive required figure/table inventory is versioned in
`config/main_paper_delivery.yaml`.  `verify_main_paper_delivery.py` must pass
before the paper package is handed off.

Cluster sensitivity jobs first write their lightweight figures, source data and
summary tables locally under `outputs/main/<run-id>/sensitivity/`; they never
write paper delivery artifacts into `REPRO_DATA_ROOT`.  After every required
job is complete, `collect_main_paper_delivery.py --run-id <run-id>` copies
only allowed lightweight file types into this delivery root and refuses any
overwrite.  It never copies OOF, fold metrics, checkpoints, logs, or matrices.

## Scientific exclusions

The delivery excludes the generative Figure 5, any standalone residual figure,
SHAP, and combined BAG outputs.  The parent set-size sensitivity is included as
Supplementary Fig. S10 using the main-k10 country-balanced estimand; ST18 remains
required as its compact tabular summary.

## Non-overwrite rule

Each delivery file is copied or written only into a new, empty target.  A
collision is a failure requiring an explicit new delivery identifier; existing
`outputs/` content is never deleted, moved, or overwritten.

`MANIFEST.csv` is the sole generated status record rather than a scientific
delivery artifact. It may be intentionally refreshed with
`--refresh-manifest`; all figures, source data and tables remain strictly
non-overwriteable.
