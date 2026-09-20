from __future__ import annotations

import pandas as pd

from repo_expo_hoi_bag.stages.run_whole_exposome_pca_sensitivity import (
    country_balanced_summary,
)


def test_country_balanced_summary_preserves_global_and_uses_country_mean() -> None:
    summary = pd.DataFrame(
        {
            "candidate_id": ["pc1"],
            "rung_id": ["xgb_tree_d3"],
            "global_oof_r2": [0.9],
        }
    )
    country = pd.DataFrame(
        {
            "candidate_id": ["pc1", "pc1"],
            "rung_id": ["xgb_tree_d3", "xgb_tree_d3"],
            "fold_country": ["small", "large"],
            "n_test": [1, 100],
            "r2": [0.1, 0.5],
        }
    )
    result = country_balanced_summary(summary, country)
    assert result.loc[0, "participant_global_oof_r2"] == 0.9
    assert result.loc[0, "country_balanced_r2"] == 0.3
    assert result.loc[0, "global_oof_r2"] == 0.3
