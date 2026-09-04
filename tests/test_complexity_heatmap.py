from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages.plot_complexity_comparison_heatmaps import (
    COMMON_COLOR_LIMIT,
    SPECIFICATIONS,
    build_panel_frame,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = ROOT / "outputs/dedup/model_comparison/complexity_pairwise_comparisons.csv"
D2_COMPARISONS = ROOT / "outputs/dedup/model_comparison/complexity_d2_fixed_comparisons.csv"
ARM_COMPARISONS = ROOT / "outputs/dedup/model_comparison/complexity_arm_comparisons.csv"


def test_complexity_heatmap_triangles_use_the_intended_comparisons() -> None:
    comparisons = pd.read_csv(COMPARISONS)
    comparisons = pd.concat([comparisons, pd.read_csv(D2_COMPARISONS)], ignore_index=True)
    arm_comparisons = pd.read_csv(ARM_COMPARISONS)
    for bag in ("structural", "functional"):
        selected = build_panel_frame(
            comparisons, arm_comparisons, bag=bag, specification=SPECIFICATIONS[0]
        )
        d2_fixed = build_panel_frame(
            comparisons, arm_comparisons, bag=bag, specification=SPECIFICATIONS[1]
        )
        d3_fixed = build_panel_frame(
            comparisons, arm_comparisons, bag=bag, specification=SPECIFICATIONS[2]
        )
        single = build_panel_frame(
            comparisons, arm_comparisons, bag=bag, specification=SPECIFICATIONS[3]
        )
        baseline = build_panel_frame(
            comparisons, arm_comparisons, bag=bag, specification=SPECIFICATIONS[4]
        )
        assert len(selected) == len(d2_fixed) == len(d3_fixed) == 16
        assert len(single) == len(baseline) == 6
        for fixed in (selected, d2_fixed, d3_fixed):
            assert set(
                fixed[fixed["triangle"].eq("diagonal")]["comparison_direction"]
            ) == {"synergy arm minus redundancy arm"}
        assert set(single["triangle"]) == {"single"}
        assert set(baseline["triangle"]) == {"baseline"}


def test_complexity_heatmap_uses_one_explicit_symmetric_colour_limit() -> None:
    assert COMMON_COLOR_LIMIT == 0.06
    arm_comparisons = pd.read_csv(ARM_COMPARISONS)
    assert (
        arm_comparisons["delta_r2_synergy_minus_redundancy"].abs()
        > COMMON_COLOR_LIMIT
    ).any()
