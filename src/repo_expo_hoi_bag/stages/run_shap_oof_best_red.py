#!/usr/bin/env python3
"""
Compute out-of-fold SHAP values for the best redundant model per BAG.

This is a thin wrapper around ``scripts.run_shap_oof`` that swaps the model
selection rule from best synergistic (o_min) to best redundant (o_max) using
the already computed per-candidate diversity scatter table.

Outputs are written under:
  outputs/variant_a/shap_oof_red/{bag}/
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts import run_shap_oof as base

ROOT = Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
STATS = ROOT / "stats"
OUTDIR_BASE = Path(os.environ.get("V3_OUTDIR_BASE", str(ROOT / "shap_oof_red")))


def _get_best_model_spec_red(bag_name: str) -> dict:
    """Return best redundant (o_max) model spec for ``bag_name``.

    Uses the diversity scatter table already present in ``ROOT/stats``.
    """
    path = STATS / "per_candidate_diversity_scatter.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate table: {path}")

    df = pd.read_csv(path)
    sub = df[(df["bag"] == bag_name) & (df["objective"] == "o_max")].copy()
    if sub.empty:
        raise ValueError(f"No redundant candidates found for BAG '{bag_name}' in {path}")

    best = sub.sort_values(
        ["global_oof_r2", "rank", "order"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    predictors = [p for p in str(best["predictors_identity"]).split("|") if p.strip()]
    print(
        f"  [red] Best redundant model for {bag_name}: "
        f"{best['model_id']}  order={len(predictors)}  R²={best['global_oof_r2']:.4f}"
    )
    return {
        "candidate_id": str(best["model_id"]),
        "predictors": predictors,
        "expected_r2": float(best["global_oof_r2"]),
    }


def parse_args():
    return base.parse_args()


def main() -> None:
    args = parse_args()
    base.pin_blas_threads(1)

    if args.n_jobs != 1:
        base.XGB_CFG["nthread"] = args.n_jobs

    base._get_best_model_spec = _get_best_model_spec_red  # type: ignore[attr-defined]

    print(f"\n{'='*70}")
    print("OOF SHAP values — best redundant model per BAG")
    print(f"  SMOKE_TEST={base.SMOKE_TEST}  bags={args.bags}  nthread={base.XGB_CFG['nthread']}")
    print(f"  OUTDIR_BASE={OUTDIR_BASE}")
    print(f"{'='*70}\n")

    print("Loading artifacts...")
    raw_df, expo_df, _, _ = base.load_artifacts_greedy_only(base.PATHS, base.ANALYSIS_CFG)
    model_df = base.build_model_df(raw_df, expo_df, base.ANALYSIS_CFG)
    feature_names = expo_df.columns.tolist()
    print(f"  model_df: {model_df.shape}  |  n_features={len(feature_names)}")

    for bag_name in args.bags:
        y_col = base.target_map(bag_name)[bag_name]
        base.run_bag(bag_name, y_col, model_df, feature_names, OUTDIR_BASE, base.SMOKE_TEST)

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
