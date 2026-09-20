#!/usr/bin/env python3
"""Regenerate OOF predictions for the country-balanced Figure 2 contrasts.

Direct local port of the raw-exposome path in the paper's
``regen_oof_models.py``. It evaluates the 16 level-selected winners, the 16
depth-2-fixed trajectories, the 16 depth-3-fixed trajectories, and the eight
best-single models with the frozen HPO parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from repo_expo_hoi_bag.analysis.paper_reanalysis import load_frozen_hpo, load_main_policy
from repo_expo_hoi_bag.config.models import load_historical_paper_reference
from repo_expo_hoi_bag.stages.run_paper_reanalysis import ANALYSIS_CFG, EARLY_STOP_CFG, _build_context, _load_model

ROOT = Path(__file__).resolve().parents[3]
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
BAGS = ("structural", "functional")
OBJECTIVES = ("o_min", "o_max")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--hpo-set", choices=("k1", "k10", "k63"), default="k10")
    parser.add_argument("--write-manifest-only", action="store_true", help="Record the selected Figure 2 tasks after OOF files exist")
    parser.add_argument("--resume", action="store_true", help="Fit only Figure 2 OOF files that are still missing")
    return parser.parse_args()


def _winner_table(root: Path, hpo_set: str) -> pd.DataFrame:
    # k1 is the completed candidate-set evaluation with the frozen
    # single-exposure configuration.  Its input directory retains the
    # descriptive historical name; its OOF results get a short, distinct k1
    # output directory below so neither completed evaluation is overwritten.
    analysis_id = "paper_reanalysis_single_set_params_k10" if hpo_set == "k1" else f"paper_reanalysis_{hpo_set}"
    candidate_scope = "k10" if hpo_set == "k1" else hpo_set
    run_root = root / "results/analysis_runs" / analysis_id
    path = run_root / "main_statistics/fig3_diversity/candidate_country_balanced_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        reference = load_historical_paper_reference(
            ROOT / "config/paper_reference.yaml", repro_data_root=root
        ).root
        registry = pd.read_parquet(reference / "results/variant_a/families/pooled_oinfo_ladder/canonical/candidate_registry.parquet")
        parts = []
        for bag in BAGS:
            source_root = root / "results/analysis_runs/paper_reanalysis_k10" if hpo_set == "k63" else run_root
            for rung in RUNGS:
                folder = source_root / "ols" / bag / "ols" / "ols" if rung == "ols" else run_root / "xgb" / bag / rung / candidate_scope
                scores = pd.read_csv(folder / "metrics_country.csv"); scores = scores[scores.n_test > 0].groupby("candidate_id", as_index=False).r2.mean().rename(columns={"r2":"country_balanced_r2"})
                part = scores.merge(registry[registry.experiment_id.eq(f"pooled_oinfo_ladder_{bag}")][["candidate_id","objective","order","predictors_identity"]], on="candidate_id", validate="one_to_one")
                part["bag"], part["rung"] = bag, rung; parts.append(part)
        df = pd.concat(parts, ignore_index=True)
    df = df[pd.to_numeric(df["order"], errors="raise") <= 30].copy()
    rows = []
    for (bag, objective, rung), group in df.groupby(["bag", "objective", "rung"], observed=True):
        rows.append(group.sort_values(["country_balanced_r2", "candidate_id"], ascending=[False, True], kind="mergesort").iloc[0])
    winners = pd.DataFrame(rows).reset_index(drop=True)
    winners["family"] = np.where(winners["objective"].eq("o_min"), "level_best_syn", "level_best_red")
    winners["scope"] = "single" if hpo_set == "k1" else hpo_set
    winners["selection_source_rung"] = winners["rung"]
    fixed_rows = []
    for (bag, objective), group in winners.groupby(["bag", "objective"], observed=True):
        for source_rung in ("xgb_tree_d2", "xgb_tree_d3"):
            deployed = group.loc[group["rung"].eq(source_rung)].copy()
            if len(deployed) != 1:
                raise ValueError(f"Expected one {source_rung} winner for {bag}/{objective}, got {len(deployed)}")
            for rung in RUNGS:
                fixed = deployed.iloc[0].copy()
                fixed["rung"] = rung
                fixed["family"] = f"fixed_{source_rung.removeprefix('xgb_tree_')}_{'syn' if objective == 'o_min' else 'red'}"
                fixed["selection_source_rung"] = source_rung
                fixed_rows.append(fixed)
    single_rows = []
    for bag in BAGS:
        for rung in RUNGS:
            source_root = root / "results/analysis_runs/paper_reanalysis_k10"
            path = source_root / ("ols" if rung == "ols" else "xgb") / bag / ("ols" if rung == "ols" else rung) / "single" / "metrics_country.csv"
            scores = pd.read_csv(path); scores = scores[scores.n_test > 0].groupby("candidate_id", as_index=False).r2.mean()
            winner = scores.sort_values(["r2", "candidate_id"], ascending=[False, True], kind="mergesort").iloc[0]
            feature = str(winner.candidate_id).removeprefix("__single__")
            single_rows.append({"bag": bag, "objective": "single", "rung": rung, "candidate_id": str(winner.candidate_id), "predictors_identity": feature, "country_balanced_r2": float(winner.r2), "family": "level_best_single", "scope": "single"})
    winners = pd.concat([winners, pd.DataFrame(fixed_rows), pd.DataFrame(single_rows)], ignore_index=True)
    if len(winners) != 56:
        raise ValueError(f"Expected 56 Figure 2 OOF tasks, got {len(winners)}")
    return winners


def _fit_one(row: dict[str, object], repro_root: Path, hpo_rows: dict[str, dict[tuple[str, str, str], dict]]) -> dict[str, object]:
    # These are the exact fold builders and XGB/OLS fit helpers used by the
    # completed evaluation, imported only after the local compatibility bridge.
    from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
    from xgb_loco_engine import _append_predictors, _fit_xgb_fold, require_xgboost

    bag, objective, rung = str(row["bag"]), str(row["objective"]), str(row["rung"])
    model_df, features = _load_model()
    context, folds = _build_context(model_df, bag, features)
    predictors = [value for value in str(row["predictors_identity"]).split("|") if value]
    indices = [features.index(value) for value in predictors]
    y, x_exp = np.asarray(context["y"], dtype=float), np.asarray(context["X_exp"], dtype=np.float32)
    full, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
    specs = {spec["rung_id"]: spec for spec in get_rung_specs()}
    cfg = None if rung == "ols" else {**build_xgb_cfg_for_rung(specs[rung]), **hpo_rows[str(row["scope"])][(bag, rung, "__global__")]["params"]}
    base_cfg = None if rung == "ols" else {**build_xgb_cfg_for_rung(specs[rung]), **hpo_rows["baseline"][(bag, rung, "__global__")]["params"]}
    xgb = None if rung == "ols" else require_xgboost()
    for fold_index, country in enumerate(context["countries"]):
        fold = folds[country]
        train, test = fold["train_idx"], fold["test_idx"]
        used = [index for index, keep in zip(indices, np.nanvar(x_exp[np.ix_(train, indices)], axis=0) > 0) if keep]
        try:
            if rung == "ols":
                # The HPO OLS delivery used sensitivity_common's default linear
                # covariate matrix (the diagnosis×exposure option is opt-in and
                # was not enabled). Rebuild that exact branch for the OOF gate.
                x_train = _append_predictors(fold["Xb_train_full"], x_exp, train, used).astype(float)
                x_test = _append_predictors(fold["Xb_test"], x_exp, test, used).astype(float)
                x_train = np.column_stack([np.ones(len(train)), x_train])
                x_test = np.column_stack([np.ones(len(test)), x_test])
                beta, *_ = np.linalg.lstsq(x_train, y[train], rcond=None)
                xb_train = np.column_stack([np.ones(len(train)), fold["Xb_train_full"]])
                xb_test = np.column_stack([np.ones(len(test)), fold["Xb_test"]])
                beta_b, *_ = np.linalg.lstsq(xb_train, y[train], rcond=None)
                full[test], base[test] = x_test @ beta, xb_test @ beta_b
            else:
                params = {**cfg, "random_state": 20260304 + fold_index}; base_params = {**base_cfg, "random_state": 20260304 + fold_index}
                tri, val = fold["tr_inner_idx"], fold["val_idx"]
                model = _fit_xgb_fold(xgb, params, _append_predictors(fold["Xb_train_inner"], x_exp, tri, used), y[tri], _append_predictors(fold["Xb_val"], x_exp, val, used), y[val])
                bmodel = _fit_xgb_fold(xgb, base_params, fold["Xb_train_inner"], y[tri], fold["Xb_val"], y[val])
                full[test] = model.predict(_append_predictors(fold["Xb_test"], x_exp, test, used))
                base[test] = bmodel.predict(fold["Xb_test"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"OOF fit failed for {bag}/{objective}/{rung}/{country}") from exc
    out = pd.DataFrame({"row_id": context["row_id"], "N_MEGA": context["N_MEGA"], "country": context["country"], "diagnosis": context["diag"], "age": context["age"], "sex": context["sex"], "y_true": y, "y_pred_full": full, "y_pred_base": base})
    out["candidate_id"], out["objective"], out["rung_id"], out["bag"] = str(row["candidate_id"]), objective, rung, bag
    out["residual"] = out["y_pred_full"] - out["y_true"]
    out["abs_residual"] = out["residual"].abs()
    target = repro_root / "results/analysis_runs" / f"paper_reanalysis_{str(row['analysis_hpo_set'])}" / "main_statistics/model_comparison/oof" / str(row["family"]) / bag / f"oof_{rung}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return {"bag": bag, "objective": objective, "rung": rung, "candidate_id": str(row["candidate_id"]), "predictors_identity": str(row["predictors_identity"]), "family": str(row["family"]), "scope": str(row["scope"]), "selection_source_rung": str(row["selection_source_rung"]), "expected_country_balanced_r2": float(row["country_balanced_r2"]), "oof_path": str(target)}


def main() -> None:
    args = _args(); repro_root = args.repro_data_root.resolve()
    policy, _ = load_main_policy(ROOT / "config/country_exclusions.yaml")
    candidate_artifact = {
        "k1": ("historical5_single_cap500", "single_exposure"),
        "k10": ("historical5_k10_cap500", "domain_balanced_k10"),
        "k63": ("historical5_full63_cap500_final", "full_exposome"),
    }[args.hpo_set]
    artifacts = {"single" if args.hpo_set == "k1" else args.hpo_set: load_frozen_hpo(ROOT / f"outputs/xgb_nested_loco_tuning/{candidate_artifact[0]}/selected_xgb_configs.json", feature_scope=candidate_artifact[1], country_policy=policy), "baseline": load_frozen_hpo(ROOT / "outputs/xgb_nested_loco_tuning/historical5_baseline_cap500/selected_xgb_configs.json", feature_scope="baseline", country_policy=policy), "single": load_frozen_hpo(ROOT / "outputs/xgb_nested_loco_tuning/historical5_single_cap500/selected_xgb_configs.json", feature_scope="single_exposure", country_policy=policy)}
    winners = _winner_table(repro_root, args.hpo_set)
    winners["analysis_hpo_set"] = args.hpo_set
    manifest = repro_root / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}" / "main_statistics/model_comparison/level_winners.csv"
    if args.write_manifest_only:
        rows = []
        for row in winners.to_dict("records"):
            target = repro_root / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}" / "main_statistics/model_comparison/oof" / str(row["family"]) / str(row["bag"]) / f"oof_{row['rung']}.parquet"
            if not target.exists():
                raise FileNotFoundError(f"Cannot record missing OOF file: {target}")
            rows.append({"bag": row["bag"], "objective": row["objective"], "rung": row["rung"], "candidate_id": row["candidate_id"], "predictors_identity": row["predictors_identity"], "family": row["family"], "scope": row["scope"], "selection_source_rung": row["selection_source_rung"], "analysis_hpo_set": row["analysis_hpo_set"], "expected_country_balanced_r2": row["country_balanced_r2"], "oof_path": str(target)})
        manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(manifest, index=False)
        print(f"Saved: {manifest} ({len(rows)} Figure 2 OOF tasks)")
        return
    records = winners.to_dict("records")
    if args.resume:
        oof_root = repro_root / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}" / "main_statistics/model_comparison/oof"
        records = [row for row in records if not (oof_root / str(row["family"]) / str(row["bag"]) / f"oof_{row['rung']}.parquet").exists()]
        print(f"Resuming {len(records)} missing Figure 2 OOF tasks", flush=True)
    rows = Parallel(n_jobs=min(args.n_jobs, len(records)), backend="loky")(delayed(_fit_one)(row, repro_root, {key: value["rows"] for key, value in artifacts.items()}) for row in records) if records else []
    manifest.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Saved: {manifest} ({len(rows)} Figure 2 OOF tasks)")


if __name__ == "__main__": main()
