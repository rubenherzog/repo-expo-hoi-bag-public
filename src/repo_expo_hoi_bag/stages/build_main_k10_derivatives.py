#!/usr/bin/env python3
"""Build the inexpensive main-paper k10 derivatives from immutable OOF files.

This stage deliberately does not fit a model.  It materializes a new ``main``
delivery namespace from the completed five-country k10 OOF predictions and an
explicit historical-OLS provenance record.  Expensive sensitivity refits are
outside this stage and are scheduled separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

BAGS = ("structural", "functional")
OBJECTIVES = (("o_min", "syn"), ("o_max", "red"))
def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--main-run-id", default="main")
    parser.add_argument("--historical-ols-provenance", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true", help="Validate sources and policy without writing outputs.")
    parser.add_argument(
        "--r2-estimator",
        choices=("country-balanced", "global-oof"),
        default="country-balanced",
        help="Estimand whose level winners supply the residuals; global-oof reads the sibling oof_global_oof/ root.",
    )
    return parser.parse_args()


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum())


def _oof(source: Path, bag: str, suffix: str, oof_dirname: str = "oof") -> pd.DataFrame:
    path = source / "main_statistics" / "model_comparison" / oof_dirname / f"level_best_{suffix}" / bag / "oof_xgb_tree_d3.parquet"
    frame = pd.read_parquet(path)
    required = {"row_id", "country", "diagnosis", "age", "sex", "y_true", "y_pred_full", "candidate_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OOF source {path} is missing {sorted(missing)}")
    return frame.loc[np.isfinite(frame.y_true) & np.isfinite(frame.y_pred_full)].copy()


def _country_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, float]:
    grouped = frame.assign(
        y2=frame.y_true**2,
        sse=np.square(frame.y_true - frame.y_pred_full),
        sse_base=np.square(frame.y_true - frame.y_pred_base),
    ).groupby("country", observed=True).agg(n=("y_true", "size"), sy=("y_true", "sum"), sy2=("y2", "sum"), sse=("sse", "sum"), sse_base=("sse_base", "sum"))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(grouped), np.repeat(1 / len(grouped), len(grouped)), size=draws)
    n = counts @ grouped.n.to_numpy(float)
    denominator = counts @ grouped.sy2.to_numpy(float) - np.square(counts @ grouped.sy.to_numpy(float)) / n
    delta = (counts @ grouped.sse_base.to_numpy(float) - counts @ grouped.sse.to_numpy(float)) / denominator
    return {"delta_r2_vs_base": _r2(frame.y_true.to_numpy(), frame.y_pred_full.to_numpy()) - _r2(frame.y_true.to_numpy(), frame.y_pred_base.to_numpy()), "ci_lo": float(np.percentile(delta, 2.5)), "ci_hi": float(np.percentile(delta, 97.5)), "p_raw": max(1 / draws, min(1.0, 2 * min(float((delta <= 0).mean()), float((delta >= 0).mean()))))}


def _summary(frame: pd.DataFrame, bag: str, objective: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = frame.assign(residual=frame.y_pred_full - frame.y_true)
    subject = []
    for diagnosis, group in x.assign(diagnosis=x.diagnosis.where(x.diagnosis.isin(["CN", "AD", "FTD"]), "other")).groupby("diagnosis", observed=True):
        subject.append({"bag": bag, "objective": objective, "diagnosis": diagnosis, "n_subjects": len(group), "bias_mean": group.residual.mean(), "mae": group.residual.abs().mean(), "rmse": float(np.sqrt(np.square(group.residual).mean()))})
    country = x.groupby("country", observed=True).agg(bias_mean=("residual", "mean"), mae=("residual", lambda s: s.abs().mean()), n_subjects=("residual", "size")).reset_index()
    country["bag"], country["objective"] = bag, objective
    return pd.DataFrame(subject), country


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.resolve()
    source = runtime / "results" / "analysis_runs" / args.source_run_id
    destination = runtime / "results" / "analysis_runs" / args.main_run_id / "derived"
    r2_mode = "global_oof" if args.r2_estimator == "global-oof" else "country_balanced"
    oof_dirname = "oof" if args.r2_estimator == "country-balanced" else "oof_global_oof"
    ols = json.loads(args.historical_ols_provenance.read_text(encoding="utf-8"))
    if ols.get("analysis_label") != "main" or ols.get("country_variant") != "historical_a":
        raise ValueError("Historical OLS provenance must declare main / historical_a compatibility")
    prepared: list[tuple[int, str, str, str, pd.DataFrame]] = []
    for index, (bag, objective, suffix) in enumerate((bag, objective, suffix) for bag in BAGS for objective, suffix in OBJECTIVES):
        frame = _oof(source, bag, suffix, oof_dirname)
        if frame.country.nunique() < 2:
            raise ValueError(f"OOF source has fewer than two countries: {bag}/{objective}")
        prepared.append((index, bag, objective, suffix, frame))
    if args.smoke_test:
        print(f"Smoke test passed: main k10 sources={len(prepared)}, OLS={args.historical_ols_provenance}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing main delivery: {destination}")
    destination.mkdir(parents=True)
    all_subject, all_country = [], []
    for index, bag, objective, suffix, frame in prepared:
        subject, country = _summary(frame, bag, objective)
        all_subject.append(subject); all_country.append(country)
    def test_row(item: tuple[int, str, str, str, pd.DataFrame]) -> dict[str, object]:
        index, bag, objective, _, frame = item
        return {"bag": bag, "objective": objective, "n_countries": frame.country.nunique(), "r2": _r2(frame.y_true.to_numpy(), frame.y_pred_full.to_numpy()), **_country_bootstrap(frame, args.draws, 20260915 + index)}
    tests = Parallel(n_jobs=min(args.n_jobs, len(prepared)), backend="loky")(delayed(test_row)(item) for item in prepared)
    tests_frame = pd.DataFrame(tests)
    tests_frame["holm_p"] = multipletests(tests_frame.p_raw, method="holm")[1]
    subject_frame, country_frame = pd.concat(all_subject, ignore_index=True), pd.concat(all_country, ignore_index=True)
    subject_frame.to_csv(destination / "main_residual_subject_summary.csv", index=False)
    country_frame.to_csv(destination / "main_residual_country_summary.csv", index=False)
    tests_frame.to_csv(destination / "main_residual_tests.csv", index=False)
    (destination / "manifest.json").write_text(json.dumps({"analysis_label": "main", "country_variant": "historical_a", "source_k10_run": str(source), "historical_ols_provenance": str(args.historical_ols_provenance), "performance_estimator": "global_oof_r2" if args.r2_estimator == "global-oof" else "country_balanced_r2", "r2_mode": r2_mode, "oof_source_dir": oof_dirname, "draws": args.draws}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved main local derivatives: {destination}")


if __name__ == "__main__":
    main()
