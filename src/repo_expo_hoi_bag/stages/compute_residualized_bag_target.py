#!/usr/bin/env python3
"""Residualized BAG target (informational only): BAG ~ age + sex + diagnosis
(OLS), fit once on the pooled cohort.

Fits the demographic-only model on each BAG and saves the residual as a
diagnostic CSV (e.g. overall residual SD per bag). Country is deliberately
NOT in this formula: country is the LOCO grouping variable for the main
analysis and the baseline candidate, not a confound to strip out of BAG --
residualizing it out would change what every candidate (including the
baseline) is scored against and conflate "control for country" with the LOCO
fold structure the main analysis already uses. Year is also excluded -- this
residualization isolates whatever the exposome-BAG association is "on top of"
age + sex + diagnosis alone, not on top of the full k_base=6 main-analysis
baseline.

This pooled, fit-once-on-everyone version is kept only as a quick eyeball
diagnostic (this script never feeds the LOCO evaluator). The actual Pass 2
modeling target in run_residualized_bag_sensitivity.py uses
evaluate_candidates_by_rung(fold_residualize=True), which refits this same
formula independently per LOCO fold using only that fold's training countries
(sensitivity_common.build_fold_residualized_y / _residualize_train_fit) --
mirroring the existing fold_pca leakage-free pattern -- so a held-out test
country's target is never informed by a model that saw that same country's
data. Do not re-wire this module's pooled-fit output back into Pass 2.

Lightweight stats -> outputs/sensitivity/residualized_bag/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from loco_fusion_matrix_engine import encode_dummies, target_map
from scripts.sensitivity_common import (
    analysis_cfg_from_config,
    build_original_model_df,
    bundle_sensitivity_root,
    copy_tree_contents,
    filtered_rows_for_bag,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)


def _residualize(work: pd.DataFrame, y_col: str) -> pd.Series:
    y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=float)
    age = pd.to_numeric(work["Age"], errors="coerce").to_numpy(dtype=float)
    sex = work["Sex"].astype(str).to_numpy()
    diag = work["Diagnosis"].astype(str).to_numpy()

    sex_d, _, _ = encode_dummies(sex, sex)
    diag_d, _, _ = encode_dummies(diag, diag)

    X = np.hstack([np.ones((len(work), 1)), age.reshape(-1, 1), sex_d, diag_d])
    keep = np.isfinite(y) & np.isfinite(age)
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    residual = np.full(len(work), np.nan)
    residual[keep] = y[keep] - X[keep] @ beta
    return pd.Series(residual, index=work.index, name="residualized_bag_value")


def main() -> None:
    cfg = load_sensitivity_config()
    local_root = repo_sensitivity_root(cfg) / "residualized_bag"
    local_root.mkdir(parents=True, exist_ok=True)

    base_cfg = analysis_cfg_from_config(cfg)
    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = build_original_model_df(raw, feature_names, base_cfg)

    for bag in selected_bags(cfg, include_combined=True):
        y_col = target_map(bag)[bag]
        work = filtered_rows_for_bag(full_model, bag, base_cfg).copy()
        residual = _residualize(work, y_col)

        out = pd.DataFrame({
            "row_id": work["row_id"].to_numpy(),
            "N_MEGA": work["N_MEGA"].to_numpy(),
            "country_clean": work["country_clean"].to_numpy(),
            "original_bag_value": pd.to_numeric(work[y_col], errors="coerce").to_numpy(),
            "residualized_bag_value": residual.to_numpy(),
        })
        out_path = local_root / f"{bag}_residualized_target.csv"
        out.to_csv(out_path, index=False)
        n_resolved = int(out["residualized_bag_value"].notna().sum())
        print(
            f"=== residualized-bag-target | bag={bag} y_col={y_col} "
            f"n_rows={len(out)} n_resolved={n_resolved} "
            f"residual_sd={out['residualized_bag_value'].std():.4f} ==="
        )

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "residualized_bag")


if __name__ == "__main__":
    main()
