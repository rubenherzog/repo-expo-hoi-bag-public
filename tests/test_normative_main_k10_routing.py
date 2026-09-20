from __future__ import annotations

import importlib

import pandas as pd
import pytest


def test_normative_loaders_accept_explicit_main_k10_roots(monkeypatch, tmp_path) -> None:
    pooled = tmp_path / "pooled"
    xgb = tmp_path / "xgb"
    ols = tmp_path / "ols"
    stats = tmp_path / "stats"
    monkeypatch.setenv("NORM_POOLED_CANONICAL_ROOT", str(pooled))
    monkeypatch.setenv("NORM_POOLED_OLS_CANONICAL_ROOT", str(tmp_path / "pooled_ols"))
    monkeypatch.setenv("NORM_XGB_ROOT", str(xgb))
    monkeypatch.setenv("NORM_OLS_ROOT", str(ols))
    monkeypatch.setenv("NORM_TRANSFER_STATS_DIR", str(stats))

    from repo_expo_hoi_bag.stages import plot_normative_transfer_grid

    module = importlib.reload(plot_normative_transfer_grid)
    assert module.POOLED_CANONICAL_ROOT == pooled
    assert module.LOCAL_STATS_DIR == stats


def test_normative_diversity_accepts_delivery_stem(monkeypatch) -> None:
    monkeypatch.setenv("NORM_DIVERSITY_FIG_STEM", "normative_diversity")

    from repo_expo_hoi_bag.stages import plot_normative_diversity_r2

    module = importlib.reload(plot_normative_diversity_r2)
    assert module.FIG_STEM == "normative_diversity"


def test_country_balanced_r2_is_unweighted_over_countries(tmp_path) -> None:
    from repo_expo_hoi_bag.stages.plot_normative_transfer_grid import _country_balanced_r2

    path = tmp_path / "metrics_country.csv"
    pd.DataFrame(
        {
            "candidate_id": ["a", "a", "b", "b"],
            "fold_country": ["x", "y", "x", "y"],
            "n_test": [1, 100, 1, 100],
            "r2": [0.1, 0.5, 0.2, 0.4],
        }
    ).to_csv(path, index=False)
    result = _country_balanced_r2(path)
    assert result.to_dict() == pytest.approx({"a": 0.3, "b": 0.3})


def test_country_balanced_r2_excludes_empty_and_nonfinite_country_cells(tmp_path) -> None:
    from repo_expo_hoi_bag.stages.plot_normative_transfer_grid import _country_balanced_r2

    path = tmp_path / "metrics_country.csv"
    pd.DataFrame(
        {
            "candidate_id": ["a", "a", "a", "b", "b"],
            "fold_country": ["valid", "empty", "invalid", "x", "y"],
            "n_test": [10, 0, 5, 2, 200],
            "r2": [0.2, -99.0, float("inf"), -0.1, 0.5],
        }
    ).to_csv(path, index=False)
    result = _country_balanced_r2(path)
    assert result.to_dict() == pytest.approx({"a": 0.2, "b": 0.2})


def test_normative_diversity_can_fix_one_rung_across_contexts(monkeypatch) -> None:
    monkeypatch.setenv("NORM_DIVERSITY_RUNG", "xgb_tree_d3")
    from repo_expo_hoi_bag.stages import plot_normative_diversity_r2

    module = importlib.reload(plot_normative_diversity_r2)
    rows = []
    for objective in ("o_min", "o_max"):
        for rung, best in (("ols", 0.9), ("xgb_tree_d3", 0.4)):
            for index in range(3):
                rows.append(
                    {
                        "bag": "functional",
                        "condition": "CN->CN",
                        "objective": objective,
                        "rung_id": rung,
                        "candidate_id": f"{objective}-{rung}-{index}",
                        "r2": best - 0.01 * index,
                    }
                )
    selected, audit = module.select_best_rung_by_objective(pd.DataFrame(rows))
    assert set(selected["rung_id"]) == {"xgb_tree_d3"}
    assert set(audit["best_rung"]) == {"xgb_tree_d3"}


def test_normative_model_loader_uses_country_balanced_r2(monkeypatch, tmp_path) -> None:
    root = tmp_path / "xgb"
    bag_root = root / "structural"
    bag_root.mkdir(parents=True)
    keys = {
        "rung_id": "xgb_tree_d3",
        "family_id": "cn_norm",
        "train_dx": "CN",
        "test_dx": "CN",
    }
    pd.DataFrame(
        [
            {"candidate_id": "__baseline__", "objective": "baseline", "order": 0, "global_oof_r2": 0.99, **keys},
            {"candidate_id": "model", "objective": "o_min", "order": 3, "global_oof_r2": 0.98, "predictors_identity": "a|b|c", "score": -0.1, **keys},
        ]
    ).to_csv(bag_root / "xgb_norm_global_all.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": candidate, "fold_country": country, "n_test": n, "r2": r2, **keys}
            for candidate, values in {
                "__baseline__": [("small", 1, 0.1), ("large", 100, 0.5)],
                "model": [("small", 1, 0.2), ("large", 100, 0.6)],
            }.items()
            for country, n, r2 in values
        ]
    ).to_csv(bag_root / "xgb_norm_country_all.csv", index=False)
    monkeypatch.setenv("NORM_XGB_ROOT", str(root))

    from repo_expo_hoi_bag.stages import plot_normative_transfer_grid

    module = importlib.reload(plot_normative_transfer_grid)
    result = module._load_normative_model("structural", "xgb")
    assert result.loc[result["candidate_id"].eq("model"), "r2"].iloc[0] == pytest.approx(0.4)
    assert result.loc[result["candidate_id"].eq("model"), "baseline_r2"].iloc[0] == pytest.approx(0.3)
