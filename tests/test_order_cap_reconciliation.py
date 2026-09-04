from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repo_expo_hoi_bag.config.models import RuntimePaths
from repo_expo_hoi_bag.legacy_runtime import prepare_legacy_runtime


def _load_legacy_stage(tmp_path: Path):
    repository_root = Path(__file__).resolve().parents[1]
    runtime = RuntimePaths.from_environment(str(tmp_path / "runtime"))
    compatibility_root = prepare_legacy_runtime(repository_root, runtime)
    domain_csv = repository_root / "data" / "metadata" / "exposome_feature_domains.csv"
    legacy_domain_csv = compatibility_root / "data" / "exposome_feature_domains.csv"
    if domain_csv.exists() and not legacy_domain_csv.exists():
        legacy_domain_csv.symlink_to(domain_csv)
    sys.path.insert(0, str(compatibility_root))
    try:
        return importlib.import_module("scripts.compute_order_cap_reconciliation")
    finally:
        sys.path.pop(0)


def test_reconciliation_long_exposes_best_and_top20_order(tmp_path: Path) -> None:
    module = _load_legacy_stage(tmp_path)
    module.ORDER_CAPS = [5, 6]
    assert module.OUT_DIR.as_posix().endswith("outputs/sensitivity/order_cap")

    frames = {
        "functional": pd.DataFrame(
            {
                "bag": ["functional"] * 6,
                "condition": ["Pooled"] * 6,
                "rung_id": ["ols", "ols", "ols", "xgb_tree_d1", "xgb_tree_d1", "xgb_tree_d1"],
                "objective": ["o_min", "o_max", "o_min", "o_min", "o_max", "o_min"],
                "order": [5, 5, 6, 5, 5, 6],
                "candidate_id": ["a", "b", "c", "d", "e", "f"],
                "predictors_identity": ["p1", "p2", "p3", "p4", "p5", "p6"],
                "r2": [0.10, 0.20, 0.90, 0.30, 0.40, 0.80],
                "baseline_r2": [0.05] * 6,
            }
        )
    }

    recon = module.reconciliation_long(frames, {("functional", "Pooled"): 1.23})

    assert {"best_order", "top20_median_order", "top20_median_r2"}.issubset(recon.columns)
    ols = recon[(recon["analysis"] == "Pooled") & (recon["rung_id"] == "ols") & (recon["order_cap"] == 6)]
    assert ols.iloc[0]["best_order"] == 6
    assert ols.iloc[0]["top20_median_order"] == 5.0
