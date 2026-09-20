#!/usr/bin/env python3
"""Audit that main k10 XGBoost outputs differ from the historical reference."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


BAGS = ("structural", "functional"); RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-data-root", type=Path, required=True); p.add_argument("--historical-root", type=Path, required=True); p.add_argument("--source-run-id", default="paper_reanalysis_k10"); p.add_argument("--main-run-id", default="main")
    p.add_argument("--output-name", default="main_effective_vs_historical_difference.csv")
    return p.parse_args()


def _new(root: Path, bag: str, rung: str) -> pd.Series:
    path = root / ("ols" if rung == "ols" else "xgb") / bag / ("ols" if rung == "ols" else rung) / ("ols" if rung == "ols" else "k10") / "metrics_country.csv"
    x = pd.read_csv(path); return x[x.n_test > 0].groupby("candidate_id", observed=True).r2.mean()


def _old(root: Path, bag: str, rung: str) -> pd.Series:
    path = root / "results/variant_a/families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
    x = pd.read_parquet(path, filters=[("rung_id", "=", rung)])
    return x.groupby("candidate_id", observed=True).country_full_r2.mean()


def main() -> None:
    a = _args(); runtime = a.repro_data_root.resolve(); new_root = runtime / "results/analysis_runs" / a.source_run_id
    rows = []
    for bag in BAGS:
        for rung in RUNGS:
            old = _old(a.historical_root.resolve(), bag, rung)
            # Main policy imports historical OLS directly; only XGBoost is k10-new.
            new = old.copy() if rung == "ols" else _new(new_root, bag, rung)
            joined = pd.concat([new.rename("new"), old.rename("historical")], axis=1, join="inner").dropna()
            delta = joined.new - joined.historical
            rows.append({"bag": bag, "rung": rung, "n_shared_candidates": len(joined), "n_changed_gt_1e12": int((delta.abs() > 1e-12).sum()), "fraction_changed": float((delta.abs() > 1e-12).mean()), "mean_abs_delta_country_balanced_r2": float(delta.abs().mean()), "max_abs_delta_country_balanced_r2": float(delta.abs().max()), "classification": "allowed_ols_reuse" if rung == "ols" else "new_k10_xgboost_must_differ"})
    report = pd.DataFrame(rows)
    xgb = report[report.rung.ne("ols")]
    if not (xgb.n_changed_gt_1e12 > 0).all(): raise ValueError("At least one k10 XGBoost rung is identical to the historical reference")
    output = runtime / "results/analysis_runs" / a.main_run_id / "audit"
    output.mkdir(parents=True, exist_ok=True); path = output / a.output_name
    if path.exists(): raise FileExistsError(f"Refusing to overwrite {path}")
    report.to_csv(path, index=False)
    print(report.to_string(index=False)); print(f"Saved: {path}")


if __name__ == "__main__": main()
