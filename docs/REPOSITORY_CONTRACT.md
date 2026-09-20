# Repository architecture and operating contract

This document is the local operational contract for `repo-expo-hoi-bag-public`. It ports the durable repository rules needed for safe agent work while keeping this checkout independent of any earlier repository.

## What belongs where

| Location | Purpose | Write policy |
| --- | --- | --- |
| `src/repo_expo_hoi_bag/` | Active package code, numerical engines, stages, and compatibility bridge | Change only within the requested implementation scope. |
| `config/` | Versioned, public configuration contracts | Add or change configuration here when the policy is repository-wide. |
| `data/` | Versioned public cohort and metadata | Read-only unless the user explicitly requests a data change. |
| `outputs/` | Delivered lightweight references, figures, and summaries | Preserve structure; never delete or invent parallel output trees. |
| `tests/` | Contract and regression tests | Extend when behavior or a contract changes. |
| `REPRO_DATA_ROOT` | Operator-selected external runtime | Heavy results, work, checkpoints, logs, OOF, fold metrics and resampling draws belong here; pass it explicitly to production commands. |

The public CLI builds a `RunContext` and `RuntimePaths`. Generated runtime content is externalized as `data`, `results`, `work`, and `figures` beneath the selected `REPRO_DATA_ROOT`. `legacy_runtime.py` is a compatibility bridge: it copies active stage/core code into the external runtime and links only the public `data/` directory. It is not permission to create an alternative checkout or an external convenience path.

The main k10 paper has a binding delivery exception: its final lightweight
figures, supplementary figures, source data and all supplementary tables must
be copied into `outputs/main/paper/complete/` in this checkout.  The exact
inventory and verification procedure are in
[MAIN_PAPER_DELIVERY_CONTRACT.md](MAIN_PAPER_DELIVERY_CONTRACT.md).

### Selecting `REPRO_DATA_ROOT`

`REPRO_DATA_ROOT` is an explicit external runtime selected by the operator. It
must be supplied to every production command as an environment variable or as
the equivalent `--repro-data-root` argument; the repository contains no
machine-specific default. Agents must not substitute a prior analysis runtime
or a sibling checkout. Existing external results may be inspected read-only
when requested, but they do not authorize overwriting an existing signature.

The nested-LOCO XGBoost tuning stage is an explicit output-contract exception:
its checkpoints are external at
`REPRO_DATA_ROOT/work/xgb_nested_loco_tuning/<run_signature>/`, whereas its
lightweight log, selected configuration, manifest, and trial diagnostics are
delivered locally at
`outputs/xgb_nested_loco_tuning/<run_signature>/`. The stage must allocate a
new signature directory and refuse to overwrite an existing one.

Final paper-facing XGBoost HPO choices are a separate, immutable derivation
governed by [XGB_HPO_SELECTION_CONTRACT.md](XGB_HPO_SELECTION_CONTRACT.md).
Do not substitute the unconstrained best trial from a 1000-trial exploratory
checkpoint.

### Immutable paper candidate reference

The complete previous-paper runtime is an external, immutable reference
catalogued in `config/paper_reference.yaml`; its root is supplied explicitly
through `PAPER_REFERENCE_ROOT`.
It covers the canonical candidate metrics, subject-level OOF files, and all
historical heavy sensitivity artifacts. In particular, the top-50 stage reads
`per_experiment/pooled_oinfo_ladder_{bag}/metrics_global_long.parquet` from
that read-only reference. It is deliberately distinct from
`runs/oinfo_only/variant_a/...`, which is an older non-deduplicated candidate
universe and must not be used to support paper-facing HPO comparisons. The
22-MB copy at `REPRO_DATA_ROOT/results/variant_a/paper_dedup_max30/canonical/`
is retained only as a verified backup, never the default source. Every stage
that reads the historical reference must record the selected artifact's path
and hash in its manifest; no stage may write beneath the historical root.

### Main re-analysis namespaces

The historical paper reference is read-only. A new main analysis must pass an
explicit `--analysis-run-id <id>` to the CLI; it writes only beneath
`REPRO_DATA_ROOT/results/analysis_runs/<id>/variant_a/`, with its greedy work
under `REPRO_DATA_ROOT/work/analysis_runs/<id>/greedy/`. This keeps the
historical paper, sensitivity references, and each HPO-based re-analysis
separate. The CLI refuses a main `greedy`, `evaluate`, or `analyses` stage
without an explicit identifier.

## Configuration sources

- `config/paper.yaml` defines the public input files, BAG scope, rung order, order range, top-k value, and random seed.
- `config/country_exclusions.yaml` is the explicit two-phase population policy. It is independent of `paper.yaml` because greedy discovery and BAG evaluation intentionally filter different populations.
- `config/xgb_hpo_feature_panels.yaml` defines any fixed, outcome-independent predictor panels used by XGBoost HPO. A stage must validate the panel against the canonical exposure/domain metadata and include its definition and hash in its manifest/checkpoint identity.
- `src/repo_expo_hoi_bag/stages/resources/` contains configuration owned by legacy stage implementations. Do not put a new repository-wide policy there when `config/` is the appropriate owner.
- Environment variables are compatibility/runtime overrides, not a place to hide new scientific defaults. A new default must be versioned in `config/` and hashed in the produced artifact.

## Output and run discipline

1. Inspect the stage, its configuration, and its runtime routing before running it.
2. Use the user-provided runtime root. Never substitute a directory near the checkout.
3. Do not write generated files in the checkout unless the invoked stage's established contract explicitly names a lightweight `outputs/` destination. The main-k10 delivery contract explicitly requires `outputs/main/paper/complete/`.
4. Never create a top-level `figures/`, nested `outputs/outputs/`, an ad-hoc log directory, a duplicate repository, or a repository-root symlink to runtime results.
5. Runs that can resume must validate their checkpoint identity against code, inputs, configuration, and relevant feature/country policies. A mismatch starts a new compatible unit; it must not mix results.
6. For a long-running numerical stage, make progress observable and warnings actionable. Record diagnostics rather than globally hiding warnings.

## Migration protocol

When the user asks to port behavior from another repository:

1. Read the exact source files and the current destination architecture.
2. Identify whether each item is code, a global configuration policy, a stage-local configuration, an input, a test, or historical context.
3. Port only the durable code/configuration/tests required by the request. Re-home global policies in `config/`.
4. Replace all source-checkout assumptions with local files and validate that imports, configuration lookup, and runtime execution do not reference the source checkout.
5. Add or update focused tests, then run the repository test suite appropriate to the change.

Do not copy historical handoffs wholesale: they may contain stale execution state, machine-specific locations, or already-resolved incidents. Port the rule they establish, not their accidental context.

## Change safety

- Preserve the existing worktree. Do not revert, clean, or overwrite unrelated modifications.
- Ask before a destructive operation, a raw-data modification, an external write not covered by the configured runtime, or any move/rename whose ownership is unclear.
- A request to inspect, diagnose, review, or plan authorizes read-only work only. A request to implement authorizes only the scoped implementation and its proportionate verification.
- After changing code, run `python -m compileall -q src`, targeted tests, and `git diff --check`; run the full suite when shared runtime/configuration code changes.
