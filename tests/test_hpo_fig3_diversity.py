from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages.plot_hpo_fig3_diversity import (
    _adjusted_regression_table,
    _build_tables,
    _country_balanced,
    _metrics_path,
)


def test_country_balanced_r2_is_an_unweighted_mean(tmp_path: Path) -> None:
    path = tmp_path / "metrics_country.csv"
    pd.DataFrame(
        {
            "candidate_id": ["a", "a", "a", "b"],
            "n_test": [1, 100, 0, 3],
            "r2": [0.2, 0.8, 99.0, 0.4],
        }
    ).to_csv(path, index=False)

    result = _country_balanced(path).set_index("candidate_id")["country_balanced_r2"]

    assert result.to_dict() == {"a": 0.5, "b": 0.4}


def test_fig3_uses_the_k10_candidate_metric_paths(tmp_path: Path) -> None:
    assert _metrics_path(tmp_path, "structural", "ols") == (
        tmp_path / "results/analysis_runs/paper_reanalysis_k10/ols/structural/ols/ols/metrics_country.csv"
    )
    assert _metrics_path(tmp_path, "functional", "xgb_tree_d3") == (
        tmp_path / "results/analysis_runs/paper_reanalysis_k10/xgb/functional/xgb_tree_d3/k10/metrics_country.csv"
    )


def test_fig3_selects_the_best_rung_then_its_candidates() -> None:
    rows = []
    for bag in ("structural", "functional"):
        for objective in ("o_min", "o_max"):
            for rung, score in (("ols", 0.1), ("xgb_tree_d3", 0.3)):
                for candidate in range(20):
                    rows.append(
                        {
                            "bag": bag,
                            "objective": objective,
                            "rung": rung,
                            "candidate_id": f"{bag}-{objective}-{rung}-{candidate}",
                            "order": 3,
                            "predictors_identity": "air|income|temp",
                            "country_balanced_r2": score + candidate / 10_000,
                        }
                    )
    scatter, recipe, selected = _build_tables(pd.DataFrame(rows), {"air": "Air Pollution", "income": "Socioeconomic", "temp": "Temperature"})

    assert set(selected["selected_rung"]) == {"xgb_tree_d3"}
    assert len(scatter) == 80
    assert len(recipe) == 240
    assert set(scatter["rung"]) == {"xgb_tree_d3"}


def test_adjusted_diversity_regression_reports_entropy_and_set_size_terms() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "bag": "structural",
                "objective": "o_min",
                "country_balanced_r2": 0.1 + index * 0.01,
                "shannon_h": index / 10,
                "order": 3 + (index % 4),
            }
        )

    result = _adjusted_regression_table(pd.DataFrame(rows))

    assert result.loc[0, "formula"] == "country_balanced_r2 ~ shannon_h + order"
    assert int(result.loc[0, "n_candidates"]) == 12
