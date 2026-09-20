from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "repo_expo_hoi_bag" / "core"))
sys.path.insert(0, str(ROOT / "src" / "repo_expo_hoi_bag" / "stages"))

from repo_expo_hoi_bag.stages.run_country_block_null import country_balanced_r2


def test_country_balanced_r2_is_unweighted_across_scored_countries() -> None:
    country = pd.DataFrame(
        {
            "predictors_identity": ["a", "a", "a", "b", "b"],
            "n_test": [1000, 10, 0, 30, 40],
            "r2": [0.1, 0.9, 999.0, -0.2, 0.4],
        }
    )

    result = country_balanced_r2(country)

    assert result["a"] == pytest.approx(0.5)
    assert result["b"] == pytest.approx(0.1)


def test_country_block_fix_launcher_reuses_assigned_namespace() -> None:
    text = Path("scripts/submit_main_k10_country_block_null_fix.sh").read_text()
    assert 'RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"' in text
    assert "country-block-null structural 40" in text
    assert "country-block-null functional 40" in text
    assert "CBN_N_PERM=10000" in text
    assert "CBN_FAST=0" in text
    assert "CBN_RUNGS=xgb_tree_d3" in text
    dry_run = text.split('if [[ "$MODE" == "--dry-run" ]]; then', 1)[1].split("fi", 1)[0]
    assert " run -t " not in dry_run
