#!/usr/bin/env python3
"""Subject-level OOF losses for every candidate region, streamed to accumulators.

The hierarchical analysis needs, for each participant, the mean and median
squared prediction error across the candidates of a region. Persisting one
prediction column per candidate would be ~0.7 GB of parquet for the ALL region
and is unnecessary: the required per-subject statistics are additive.

What is accumulated per BAG x arm x rung x subject
--------------------------------------------------
* ``sum_loss[region]``  -- running sum of candidate squared errors, giving the
  mean loss exactly (verified to 2.2e-16 against the explicit formulation).
* ``median_candidate_loss`` -- the exact per-subject median candidate loss. The
  descriptive median improvement is then recovered from the baseline-constant
  identity::

      median_j(L_base_i - L_cand_ij) = L_base_i - median_j(L_cand_ij)

  which holds because ``L_base_i`` is constant across j within a subject and
  ``x -> L_base_i - x`` is order-reversing. The loss matrix is held in memory
  only for the duration of one cell (at most 560 x ~10,800 float64, ~0.05 GB),
  so the median is exact for every region including ALL.

Because Top10 c Top20 c Top50, a single pass over the Top-50 union serves all
three nested regions. The ALL region is a separate pass over every eligible
candidate.

Critically this never averages candidate *predictions*: it accumulates
candidate *losses*. Baseline loss minus mean candidate loss is the individual
prediction improvement; baseline loss minus the loss of an averaged prediction
would evaluate an ensemble, a different scientific question (section 3).

Reuses the production fold builder, frozen HPO parameters, country policy and
per-fold seeds via ``run_cross_selected_oof.fit_folds``, so a fit here is
numerically the computation the delivery performed. Verified bit-exact against
a delivered fold before launch (max |diff| = 0.0).

Agreement with the delivered per-country metrics
------------------------------------------------
Recomputed per-country R2 matches ``metrics_country.csv`` to ~1e-3 for the XGB
levels and to ~1e-10 for OLS. The residual XGB gap is not introduced here: the
production helper ``sensitivity_common._fit_candidate``, invoked with the same
frozen configuration, departs from the same delivered CSV by ~9e-2, an order of
magnitude more. The delivered metrics were produced through the main-k10 HPO
routing, which this stage does not replicate because it needs subject-level
predictions rather than aggregated metrics. What matters for this analysis is
the prediction path, and that path was verified bit-exact against the delivered
subject-level OOF store.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from repo_expo_hoi_bag.analysis.paper_reanalysis import load_main_policy
from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.run_cross_selected_oof import _hpo_artifacts, fit_folds

ROOT = frontier.ROOT

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
# OLS and d1 are reported alongside d2/d3; the primary Holm family stays the
# four d2/d3 tests per BAG, so the added levels are descriptive.
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
# Nested candidate regions. Top10 c Top20 c Top50 c ALL, verified explicitly.
NESTED_K = (5, 10, 15, 20, 30, 40, 50, 100)
PRIMARY_K = 20
CACHE_K = 50

SELECTION_NOTE = (
    "Candidates are ranked by the unweighted mean R2 across all eligible held-out "
    "countries (n_test > 0), restricted to set sizes <= 30, ties broken on ascending "
    "candidate_id with a stable sort. This is the original *global* LOCO ranking used "
    "by the paper and the previous frontier analysis. No country-by-country "
    "cross-selection is applied here: the regions are conditional on the observed "
    "candidate landscape."
)


def region_members(
    inputs: frontier.FrontierInputs, arm: str, rung: str
) -> tuple[dict[str, list[str]], list[str]]:
    """Global Top-K membership for each nested region, plus the full ALL pool.

    Uses ``compute_performance_frontier.top_k`` with no exclusion, which is the
    production ranking rule evaluated on every country.
    """
    wide = inputs.multivariate[rung]
    pool = wide.loc[wide.index.astype(str).isin(inputs.arms[arm])]
    regions = {f"Top{k}": frontier.top_k(pool, k) for k in NESTED_K}
    everything = frontier._ranked(
        pool.mean(axis=1).rename("mean_r2").reset_index(), "mean_r2"
    )["candidate_id"].astype(str).tolist()

    for smaller, larger in zip(NESTED_K, NESTED_K[1:]):
        if not set(regions[f"Top{smaller}"]).issubset(regions[f"Top{larger}"]):
            raise ValueError(f"Top{smaller} is not nested inside Top{larger}")
    if not set(regions[f"Top{NESTED_K[-1]}"]).issubset(everything):
        raise ValueError("Top-K membership is not contained in the ALL pool")
    return regions, everything


def plan(
    runtime: Path, source_run_id: str, scope: str
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, list[str]]]]:
    """Candidates to fit, and the region membership each cell will slice.

    ``scope="all"`` fits every eligible candidate, which is a superset of the
    nested Top-K regions, so one pass serves all four regions and no fit is ever
    duplicated. ``scope="topk"`` restricts the pass to the Top-50 union.
    """
    registry = frontier._registry(frontier._registry_path(runtime, source_run_id))
    rows: list[dict[str, object]] = []
    membership: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    for bag in BAGS:
        inputs = frontier.load_bag(runtime, source_run_id, bag, registry)
        for rung in RUNGS:
            for arm in ARMS:
                regions, everything = region_members(inputs, arm, rung)
                if scope == "all":
                    regions = {**regions, "ALL": everything}
                wanted = everything if scope == "all" else regions[f"Top{CACHE_K}"]
                membership[(bag, arm, rung)] = regions
                for rank, candidate in enumerate(wanted, start=1):
                    rows.append(
                        {
                            "bag": bag,
                            "arm": arm,
                            "rung": rung,
                            "candidate_id": candidate,
                            "global_rank": rank,
                            "predictors_identity": "|".join(
                                sorted(inputs.predictors[candidate])
                            ),
                        }
                    )
    return pd.DataFrame(rows), membership


def fit_cell(
    bag: str,
    rung: str,
    group: pd.DataFrame,
    hpo_rows: dict[str, dict],
    countries: list[str],
    n_jobs: int = 1,
) -> list[tuple[str, pd.DataFrame]]:
    """Fit every candidate of one cell, building the LOCO context only once.

    This is ``run_cross_selected_oof.fit_folds`` with the per-candidate model
    load and fold construction hoisted out of the loop: the context, folds,
    frozen HPO parameters and per-fold seeds are identical, so each fit is the
    same computation, but the shared setup is paid once per cell instead of once
    per candidate. The baseline is refitted per fold exactly as before and is
    asserted constant across candidates by the caller.
    """
    from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
    from xgb_loco_engine import _append_predictors, _fit_xgb_fold, require_xgboost

    from repo_expo_hoi_bag.stages.run_paper_reanalysis import _build_context, _load_model
    from repo_expo_hoi_bag.stages.run_cross_selected_oof import FOLD_SEED_BASE

    is_ols = rung == "ols"
    if is_ols:
        # OLS is a different estimator, not XGBoost without a config: the
        # production design carries intercept + age + sex dummies + year spline
        # + diagnosis dummies + exposome + diagnosis x exposome interactions.
        # Both helpers are imported from the stage that owns that design rather
        # than reimplemented here.
        from repo_expo_hoi_bag.stages.sensitivity_common import (
            _build_canonical_ols_design,
            _ols_dx_interactions,
        )

    wanted = {str(value) for value in countries}
    model_df, features = _load_model()
    context, folds = _build_context(model_df, bag, features)
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)

    if is_ols:
        cfg = base_cfg = None
        xgb = None
    else:
        specs = {spec["rung_id"]: spec for spec in get_rung_specs()}
        cfg = {
            **build_xgb_cfg_for_rung(specs[rung]),
            **hpo_rows["k10"][(bag, rung, "__global__")]["params"],
        }
        base_cfg = {
            **build_xgb_cfg_for_rung(specs[rung]),
            **hpo_rows["baseline"][(bag, rung, "__global__")]["params"],
        }
        xgb = require_xgboost()

    scored = [
        (index, country)
        for index, country in enumerate(context["countries"])
        if str(country) in wanted
    ]

    # The covariate baseline does not depend on the candidate, so it is fitted
    # once per fold rather than once per candidate x fold.
    base = np.full(len(y), np.nan)
    for fold_index, country in scored:
        fold = folds[country]
        train, test = fold["train_idx"], fold["test_idx"]
        if is_ols:
            # Covariate-only OLS baseline: the canonical design with an empty
            # exposome block, fitted on the full training fold.
            if _ols_dx_interactions():
                design_train, design_test = _build_canonical_ols_design(
                    context, country, train, test, x_exp, []
                )
            else:
                design_train = np.column_stack(
                    [np.ones(len(train)), fold["Xb_train_full"].astype(float)]
                )
                design_test = np.column_stack(
                    [np.ones(len(test)), fold["Xb_test"].astype(float)]
                )
            beta, *_ = np.linalg.lstsq(design_train, y[train], rcond=None)
            base[test] = design_test @ beta
            continue
        inner, validation = fold["tr_inner_idx"], fold["val_idx"]
        base_model = _fit_xgb_fold(
            xgb,
            {**base_cfg, "random_state": FOLD_SEED_BASE + fold_index},
            fold["Xb_train_inner"],
            y[inner],
            fold["Xb_val"],
            y[validation],
        )
        base[test] = base_model.predict(fold["Xb_test"])

    keep = np.array([str(value) in wanted for value in context["country"]])
    identity = pd.DataFrame(
        {
            "row_id": context["row_id"],
            "N_MEGA": context["N_MEGA"],
            "country": context["country"],
            "y_true": y,
            "y_pred_base": base,
        }
    )

    def one_candidate(row) -> tuple[str, pd.DataFrame]:
        predictors = [value for value in str(row.predictors_identity).split("|") if value]
        indices = [features.index(value) for value in predictors]
        full = np.full(len(y), np.nan)
        for fold_index, country in scored:
            fold = folds[country]
            train, test = fold["train_idx"], fold["test_idx"]
            used = [
                index
                for index, flag in zip(
                    indices, np.nanvar(x_exp[np.ix_(train, indices)], axis=0) > 0
                )
                if flag
            ]
            try:
                if is_ols:
                    if _ols_dx_interactions():
                        design_train, design_test = _build_canonical_ols_design(
                            context, country, train, test, x_exp, used
                        )
                    else:
                        design_train = np.column_stack(
                            [
                                np.ones(len(train)),
                                _append_predictors(
                                    fold["Xb_train_full"], x_exp, train, used
                                ).astype(float),
                            ]
                        )
                        design_test = np.column_stack(
                            [
                                np.ones(len(test)),
                                _append_predictors(
                                    fold["Xb_test"], x_exp, test, used
                                ).astype(float),
                            ]
                        )
                    beta, *_ = np.linalg.lstsq(design_train, y[train], rcond=None)
                    full[test] = design_test @ beta
                    continue
                inner, validation = fold["tr_inner_idx"], fold["val_idx"]
                model = _fit_xgb_fold(
                    xgb,
                    {**cfg, "random_state": FOLD_SEED_BASE + fold_index},
                    _append_predictors(fold["Xb_train_inner"], x_exp, inner, used),
                    y[inner],
                    _append_predictors(fold["Xb_val"], x_exp, validation, used),
                    y[validation],
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(f"OOF fit failed for {bag}/{rung}/{country}") from exc
            full[test] = model.predict(_append_predictors(fold["Xb_test"], x_exp, test, used))

        frame = identity.assign(y_pred_full=full)[keep].sort_values("row_id").reset_index(drop=True)
        return str(row.candidate_id), frame

    rows = list(group.itertuples(index=False))
    if n_jobs <= 1:
        return [one_candidate(row) for row in rows]
    # Candidates are independent, so they are fanned out one per worker. Each
    # booster stays single-threaded via OMP_NUM_THREADS=1, so the cores are
    # spent on candidates rather than on nested boosting threads. Order is
    # preserved, keeping the result identical to the sequential path.
    # ``max_nbytes`` memory-maps the shared fold matrices instead of pickling a
    # copy of them into every task, so the fan-out cost stays bounded.
    return Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
        delayed(one_candidate)(row) for row in rows
    )


def accumulate(
    group: pd.DataFrame,
    hpo_rows: dict[str, dict],
    countries: list[str],
    regions: dict[str, list[str]],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit every candidate of one BAG x arm x rung and reduce to per-subject stats.

    Candidates are fitted once; because the regions are nested each region's
    statistics are taken as a row slice of the same loss matrix. Returns one row
    per subject x region. Candidate predictions are discarded as soon as their
    loss is recorded, and are never averaged.
    """
    bag = str(group["bag"].iloc[0])
    rung = str(group["rung"].iloc[0])
    arm = str(group["arm"].iloc[0])

    frames = fit_cell(bag, rung, group, hpo_rows, countries, n_jobs=n_jobs)

    losses: list[np.ndarray] = []
    order: list[str] = []
    identity: pd.DataFrame | None = None
    baseline_loss: np.ndarray | None = None

    for candidate, frame in frames:
        if identity is None:
            identity = frame[["row_id", "N_MEGA", "country", "y_true"]].copy()
            baseline_loss = (frame["y_true"] - frame["y_pred_base"]).to_numpy(float) ** 2
        elif not identity["row_id"].equals(frame["row_id"]):
            raise ValueError(f"Subject alignment differs across candidates for {bag}/{rung}")
        else:
            delivered = (frame["y_true"] - frame["y_pred_base"]).to_numpy(float) ** 2
            if not np.allclose(delivered, baseline_loss, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Baseline prediction is not constant across candidates for {bag}/{rung}"
                )

        losses.append((frame["y_true"] - frame["y_pred_full"]).to_numpy(float) ** 2)
        order.append(candidate)

    if identity is None or baseline_loss is None:
        raise ValueError("No candidates supplied")

    matrix = np.vstack(losses)
    position = {candidate: index for index, candidate in enumerate(order)}

    # The cumulative sum over the global ranking is kept so that *any* Top-K can
    # be recovered later without refitting: the summed loss over the first K
    # candidates is cumulative[K - 1], and a mean is that divided by K. Storing
    # the running sums rather than the full 560 x subjects matrix keeps the file
    # small while making the region grid exact for every K.
    cumulative = np.cumsum(matrix, axis=0)

    parts: list[pd.DataFrame] = []
    for region, members in regions.items():
        rows = [position[candidate] for candidate in members]
        block = matrix[rows, :]
        part = identity.copy()
        part["baseline_loss"] = baseline_loss
        part["mean_candidate_loss"] = block.mean(axis=0)
        part["median_candidate_loss"] = np.median(block, axis=0)
        part["n_candidate_predictions"] = block.shape[0]
        part["region"] = region
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out["bag"] = bag
    out["arm"] = arm
    out["rung"] = rung
    # improvement = baseline squared error - candidate squared error (section 2).
    out["mean_improvement"] = out["baseline_loss"] - out["mean_candidate_loss"]
    out["median_improvement"] = out["baseline_loss"] - out["median_candidate_loss"]
    return out, cumulative, identity, baseline_loss, order


def _key(bag: str, arm: str, rung: str, scope: str) -> str:
    return f"{scope}__{bag}__{arm}__{rung}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--scope", choices=("topk", "all"), default="topk")
    # One cluster job owns one cell, so it can be restricted to a single
    # BAG x arm x rung. Omitting these runs every cell in one process.
    parser.add_argument("--bag", choices=BAGS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--rung", choices=RUNGS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/loss_cache",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    policy, _ = load_main_policy(ROOT / "config/country_exclusions.yaml")
    hpo_rows = _hpo_artifacts(policy)

    tasks, membership = plan(runtime, args.source_run_id, args.scope)
    for column, wanted in (("bag", args.bag), ("arm", args.arm), ("rung", args.rung)):
        if wanted is not None:
            tasks = tasks[tasks[column].eq(wanted)]
    if tasks.empty:
        raise SystemExit("No candidates match the requested bag/arm/rung selection")
    registry = frontier._registry(frontier._registry_path(runtime, args.source_run_id))
    country_map = {
        bag: sorted(
            frontier.load_bag(runtime, args.source_run_id, bag, registry)
            .multivariate[RUNGS[0]]
            .columns.astype(str)
        )
        for bag in BAGS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        (bag, arm, rung, group)
        for (bag, arm, rung), group in tasks.groupby(["bag", "arm", "rung"], sort=True)
    ]
    pending = [
        item
        for item in groups
        if not (args.output_dir / f"{_key(item[0], item[1], item[2], args.scope)}.parquet").is_file()
    ]
    print(
        f"{len(tasks)} candidate fits across {len(groups)} cells, {len(pending)} cells pending",
        flush=True,
    )

    def run(item: tuple[str, str, str, pd.DataFrame]) -> str:
        bag, arm, rung, group = item
        frame, cumulative, identity, baseline_loss, order = accumulate(
            group, hpo_rows, country_map[bag], membership[(bag, arm, rung)], n_jobs=args.n_jobs
        )
        key = _key(bag, arm, rung, args.scope)
        target = args.output_dir / f"{key}.parquet"
        frame.to_parquet(target, index=False)

        # Cumulative loss sums over the global ranking, so any Top-K mean can be
        # recovered later as cumulative[K - 1] / K without refitting.
        np.savez_compressed(
            args.output_dir / f"{key}__cumulative.npz",
            # float64: the sums are recovered by differencing, so float32 would
            # inject a ~3e-7 rounding error into every derived Top-K mean.
            cumulative=cumulative,
            row_id=identity["row_id"].to_numpy(),
            country=identity["country"].astype(str).to_numpy(),
            y_true=identity["y_true"].to_numpy(),
            baseline_loss=baseline_loss,
            candidate_order=np.array(order, dtype=object),
        )
        print(f"  {target.name}: {len(group)} candidates, {len(frame)} rows", flush=True)
        return str(target)

    # Cells run one at a time and the workers are spent on the candidates inside
    # each cell. A cluster job owns a single cell, so parallelising over cells
    # would leave every allocated core but one idle.
    for item in pending:
        run(item)

    manifest = {
        "stage": "run_hierarchical_oof_predictions",
        "scope": args.scope,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidate_fits": int(len(tasks)),
        "n_cells": len(groups),
        "selection_rule": SELECTION_NOTE,
        "fidelity_gate": (
            "A cached delivered fold was refitted through this exact path before launch; "
            "max absolute prediction difference was 0.0."
        ),
        "statement": (
            "Accumulates per-subject candidate squared-error losses. Candidate "
            "predictions are never averaged and never persisted. Writes only "
            "beneath its own output root."
        ),
    }
    (args.output_dir / f"fit_manifest_{args.scope}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
