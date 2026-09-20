#!/usr/bin/env python3
"""Gate HPO Figure 2 inference on exact per-country OOF metric reproduction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--hpo-set", choices=("k10", "k63"), default="k10")
    return parser.parse_args()


def _r2(frame: pd.DataFrame, prediction: str) -> pd.Series:
    def score(group: pd.DataFrame) -> float:
        y = group.y_true.to_numpy(float)
        return float(1.0 - np.square(y - group[prediction].to_numpy(float)).sum() / np.square(y - y.mean()).sum())

    return frame.groupby("country", sort=True).apply(score)


def _metrics_path(run_root: Path, bag: str, rung: str, scope: str) -> Path:
    if scope in {"baseline", "single"} or rung == "ols":
        run_root = run_root.parent / "paper_reanalysis_k10"
    if rung == "ols":
        return run_root / "ols" / bag / "ols" / ("ols" if scope in {"k10", "k63"} else scope) / "metrics_country.csv"
    return run_root / "xgb" / bag / rung / scope / "metrics_country.csv"


def _expected(run_root: Path, row: pd.Series, scope: str, candidate_id: str) -> pd.Series:
    metrics = pd.read_csv(_metrics_path(run_root, str(row.bag), str(row.rung), scope))
    selected = metrics.loc[(metrics.candidate_id.astype(str) == candidate_id) & (metrics.n_test > 0), ["fold_country", "r2"]].rename(columns={"fold_country": "country"})
    if selected.empty:
        raise ValueError(f"No canonical metric for {row.bag}/{row.rung}/{scope}/{candidate_id}")
    return selected.set_index("country").r2.sort_index()


def main() -> None:
    args = _args()
    root = args.repro_data_root.resolve()
    run_root = root / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}"
    output_root = run_root / "main_statistics/model_comparison"
    manifest = pd.read_csv(output_root / "level_winners.csv")
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        oof = pd.read_parquet(row.oof_path, columns=["country", "y_true", "y_pred_full", "y_pred_base"])
        scope = str(row.scope)
        expected_full = _expected(run_root, row, scope, str(row.candidate_id))
        observed_full = _r2(oof, "y_pred_full")
        for country in expected_full.index:
            rows.append({"family": row.family, "bag": row.bag, "objective": row.objective, "rung": row.rung, "candidate_id": row.candidate_id, "prediction": "full", "country": country, "expected_r2": expected_full.loc[country], "observed_r2": observed_full.loc[country], "absolute_difference": abs(expected_full.loc[country] - observed_full.loc[country])})
        baseline = pd.read_csv(_metrics_path(run_root, str(row.bag), str(row.rung), "baseline"))
        baseline_id = str(baseline.loc[baseline.n_test > 0, "candidate_id"].iloc[0])
        expected_base = _expected(run_root, row, "baseline", baseline_id)
        observed_base = _r2(oof, "y_pred_base")
        for country in expected_base.index:
            rows.append({"family": row.family, "bag": row.bag, "objective": row.objective, "rung": row.rung, "candidate_id": baseline_id, "prediction": "baseline", "country": country, "expected_r2": expected_base.loc[country], "observed_r2": observed_base.loc[country], "absolute_difference": abs(expected_base.loc[country] - observed_base.loc[country])})
    report = pd.DataFrame(rows)
    report["passed"] = report.absolute_difference <= args.tolerance
    target = output_root / "oof_qa_country_r2.csv"
    report.to_csv(target, index=False)
    failed = report.loc[~report.passed]
    print(f"Saved: {target} ({len(report)} country-level checks; {len(failed)} failed)")
    if not failed.empty:
        raise SystemExit("OOF QA failed; statistical comparisons are not valid")


if __name__ == "__main__":
    main()
