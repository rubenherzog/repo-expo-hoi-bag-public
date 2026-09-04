from __future__ import annotations

from itertools import combinations

import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.stages.plot_normative_transfer_grid import (
    _build_transfer_panels,
    add_synergistic_triplet_fraction,
)


def test_triplet_fraction_uses_every_triplet_in_the_candidate() -> None:
    exposures = ["a", "b", "c", "d"]
    triplets = list(combinations(exposures, 3))
    omega = {
        frozenset(triplet): value
        for triplet, value in zip(triplets, [-1.0, 0.5, -0.2, 0.1])
    }
    top = pd.DataFrame({"predictors_identity": ["|".join(exposures)]})

    result = add_synergistic_triplet_fraction(top, omega, baseline_fraction=0.25)

    assert result.loc[0, "n_triplets"] == 4
    assert result.loc[0, "n_synergistic_triplets"] == 2
    assert result.loc[0, "pct_synergistic_triplets"] == 50.0
    assert result.loc[0, "exposome_baseline_pct"] == 25.0


def test_normative_source_data_matches_the_three_fig2_quantities() -> None:
    top = pd.DataFrame(
        {
            "condition": ["Pooled"],
            "rung_id": ["ols"],
            "objective": ["o_min"],
            "candidate_id": ["candidate"],
            "order": [3],
            "r2": [0.3],
            "baseline_r2": [0.2],
            "n_triplets": [1],
            "n_synergistic_triplets": [1],
            "pct_synergistic_triplets": [100.0],
            "exposome_baseline_pct": [33.2],
        }
    )
    single = pd.DataFrame(
        {
            "condition": ["Pooled"],
            "rung_id": ["ols"],
            "single_r2": [0.25],
            "single_feature": ["a"],
        }
    )

    panels = _build_transfer_panels("structural", top, single)

    assert [panel.panel_id for panel in panels] == [
        "a_structural_loco_r2",
        "b_structural_set_size",
        "c_structural_triplets",
        "best_single_structural",
    ]
    assert all("f2" not in panel.panel_id and "enrichment" not in panel.panel_id for panel in panels)


def test_source_data_writer_removes_obsolete_panel_csv(tmp_path) -> None:
    source_dir = tmp_path / "source_data"
    source_dir.mkdir()
    stale = source_dir / "figure_source_data_old_panel.csv"
    stale.write_text("old\n", encoding="utf-8")
    unrelated = source_dir / "another_figure_source_data_old_panel.csv"
    unrelated.write_text("keep\n", encoding="utf-8")

    write_source_data(
        "figure",
        [Panel(panel_id="a", frame=pd.DataFrame({"value": [1]}), description="test")],
        tmp_path,
    )

    assert not stale.exists()
    assert unrelated.exists()
    assert (source_dir / "figure_source_data_a.csv").exists()
