#!/usr/bin/env python3
"""Fit the OOF folds the country-cross-selected analysis needs and cache misses.

The delivered subject-level OOF store holds predictions only for the globally
selected winner of each BAG x arm x rung. A country-cross-selected analysis
picks the candidate using the countries *other than* the evaluation country, so
for the folds where that choice differs from the global winner the required
predictions were never computed.

This stage fits exactly those missing (candidate, rung, held-out country) folds
and nothing else. Folds whose cross-selected candidate equals a candidate that
already has cached predictions are reused rather than refitted.

It reuses the production fold builder and fit helpers from
``run_hpo_level_winner_oof`` / ``run_paper_reanalysis`` so a refitted fold is
numerically the same computation the delivery performed, with the same frozen
HPO parameters, country policy and per-fold random seed. It writes only beneath
its own output root and never into the delivered ``model_comparison/oof`` tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from repo_expo_hoi_bag.analysis.paper_reanalysis import load_frozen_hpo, load_main_policy
from repo_expo_hoi_bag.stages.run_paper_reanalysis import _build_context, _load_model

ROOT = Path(__file__).resolve().parents[3]

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
# d2/d3 are the primary target; d1 is a nonlinear-additive control kept
# descriptively. OLS is deliberately excluded from the baseline conclusion.
PRIMARY_RUNGS = ("xgb_tree_d2", "xgb_tree_d3")
CONTROL_RUNGS = ("xgb_tree_d1",)
# The per-fold seed convention of the delivered OOF stage, reproduced exactly.
FOLD_SEED_BASE = 20260304


def _hpo_artifacts(policy) -> dict[str, dict]:
    """The frozen HPO parameter sets used by the delivered k10 OOF fits."""
    return {
        "k10": load_frozen_hpo(
            ROOT / "outputs/xgb_nested_loco_tuning/historical5_k10_cap500/selected_xgb_configs.json",
            feature_scope="domain_balanced_k10",
            country_policy=policy,
        )["rows"],
        "baseline": load_frozen_hpo(
            ROOT / "outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json",
            feature_scope="baseline",
            country_policy=policy,
        )["rows"],
    }


def fit_folds(task: dict[str, object], hpo_rows: dict[str, dict]) -> pd.DataFrame:
    """Fit one candidate at one rung, for a named subset of held-out countries.

    Returns subject-level predictions for the participants of those countries
    only. Each prediction comes from a model trained without that participant's
    country, exactly as in the delivered LOCO design.
    """
    from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
    from xgb_loco_engine import _append_predictors, _fit_xgb_fold, require_xgboost

    bag, rung = str(task["bag"]), str(task["rung"])
    wanted = set(str(value) for value in task["countries"])

    model_df, features = _load_model()
    context, folds = _build_context(model_df, bag, features)
    predictors = [value for value in str(task["predictors_identity"]).split("|") if value]
    indices = [features.index(value) for value in predictors]
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)

    specs = {spec["rung_id"]: spec for spec in get_rung_specs()}
    cfg = {**build_xgb_cfg_for_rung(specs[rung]), **hpo_rows["k10"][(bag, rung, "__global__")]["params"]}
    base_cfg = {**build_xgb_cfg_for_rung(specs[rung]), **hpo_rows["baseline"][(bag, rung, "__global__")]["params"]}
    xgb = require_xgboost()

    full = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    fitted: list[str] = []
    # The fold index drives the per-fold seed, so it must be the position in the
    # full country list, not the position within the requested subset.
    for fold_index, country in enumerate(context["countries"]):
        if str(country) not in wanted:
            continue
        fold = folds[country]
        train, test = fold["train_idx"], fold["test_idx"]
        used = [
            index
            for index, keep in zip(indices, np.nanvar(x_exp[np.ix_(train, indices)], axis=0) > 0)
            if keep
        ]
        try:
            params = {**cfg, "random_state": FOLD_SEED_BASE + fold_index}
            base_params = {**base_cfg, "random_state": FOLD_SEED_BASE + fold_index}
            inner, validation = fold["tr_inner_idx"], fold["val_idx"]
            model = _fit_xgb_fold(
                xgb,
                params,
                _append_predictors(fold["Xb_train_inner"], x_exp, inner, used),
                y[inner],
                _append_predictors(fold["Xb_val"], x_exp, validation, used),
                y[validation],
            )
            base_model = _fit_xgb_fold(
                xgb, base_params, fold["Xb_train_inner"], y[inner], fold["Xb_val"], y[validation]
            )
            full[test] = model.predict(_append_predictors(fold["Xb_test"], x_exp, test, used))
            base[test] = base_model.predict(fold["Xb_test"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"OOF fit failed for {bag}/{rung}/{country}") from exc
        fitted.append(str(country))

    frame = pd.DataFrame(
        {
            "row_id": context["row_id"],
            "N_MEGA": context["N_MEGA"],
            "country": context["country"],
            "diagnosis": context["diag"],
            "age": context["age"],
            "sex": context["sex"],
            "y_true": y,
            "y_pred_full": full,
            "y_pred_base": base,
        }
    )
    frame = frame[frame["country"].astype(str).isin(set(fitted))].copy()
    frame["candidate_id"] = str(task["candidate_id"])
    frame["rung_id"] = rung
    frame["bag"] = bag
    return frame


def _task_key(task: dict[str, object]) -> str:
    return f"{task['bag']}__{task['rung']}__{task['candidate_id']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="Task table from the planning stage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10/oof_cache",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    policy, _ = load_main_policy(ROOT / "config/country_exclusions.yaml")
    hpo_rows = _hpo_artifacts(policy)

    plan = pd.read_csv(args.tasks)
    tasks: list[dict[str, object]] = []
    for (bag, rung, candidate, predictors), group in plan.groupby(
        ["bag", "rung", "candidate_id", "predictors_identity"], sort=True
    ):
        tasks.append(
            {
                "bag": bag,
                "rung": rung,
                "candidate_id": candidate,
                "predictors_identity": predictors,
                "countries": sorted(group["country"].astype(str).unique()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        task for task in tasks if not (args.output_dir / f"{_task_key(task)}.parquet").is_file()
    ] if args.resume else tasks
    print(f"{len(tasks)} fit tasks, {len(pending)} pending", flush=True)

    def run(task: dict[str, object]) -> str:
        frame = fit_folds(task, hpo_rows)
        target = args.output_dir / f"{_task_key(task)}.parquet"
        frame.to_parquet(target, index=False)
        print(f"  fitted {_task_key(task)} ({len(task['countries'])} folds)", flush=True)
        return str(target)

    if pending:
        Parallel(n_jobs=min(args.n_jobs, len(pending)), backend="loky")(
            delayed(run)(task) for task in pending
        )

    manifest = {
        "stage": "run_cross_selected_oof",
        "n_tasks": len(tasks),
        "output_dir": str(args.output_dir),
        "fold_seed_base": FOLD_SEED_BASE,
        "statement": (
            "Fits only the (candidate, rung, held-out country) folds whose "
            "cross-selected candidate has no cached OOF predictions. Reuses the "
            "production fold builder, frozen HPO parameters, country policy and "
            "per-fold seeds. Writes only beneath its own output root."
        ),
    }
    (args.output_dir / "fit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
