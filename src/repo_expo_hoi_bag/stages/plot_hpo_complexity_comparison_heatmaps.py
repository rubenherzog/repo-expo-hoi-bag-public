#!/usr/bin/env python3
"""HPO port of Supplementary Fig. S13 model-complexity matrices."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.stages import plot_complexity_comparison_heatmaps as parent_figure

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ("OLS", "d1", "d2", "d3")
RUNG_TO_LEVEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
LEVEL_INDEX = {value: index for index, value in enumerate(LEVELS)}
LIMIT = 0.06
SPECS = (
    ("selected", "Winner selected at each level", "level_selected", "arm_level_selected"),
    ("d2_fixed", "d2 winner held fixed", "fixed_d2_candidate", "arm_fixed_d2_candidate"),
    ("d3_fixed", "d3 winner held fixed", "fixed_d3_candidate", "arm_fixed_d3_candidate"),
    ("single", "Best single exposure", "single_trajectory", None),
    ("baseline", "Covariate baseline", "baseline_trajectory", None),
)


def _stars(value: float) -> str:
    return "***" if value < .001 else "**" if value < .01 else "*" if value < .05 else ""


def _p(row: pd.Series, arm: bool = False) -> float:
    column = "holm_p_across_four_levels" if arm else "holm_p_within_bag_trajectory"
    return float(row[column])


def _rows(stats: pd.DataFrame, bag: str, key: str, comparison_type: str, arm_type: str | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if arm_type:
        for objective, triangle in (("o_min", "synergy"), ("o_max", "redundancy")):
            for row in stats[(stats.comparison_type == comparison_type) & (stats.bag == bag) & (stats.objective == objective)].itertuples(index=False):
                high, low = RUNG_TO_LEVEL[row.model_a], RUNG_TO_LEVEL[row.model_b]
                ri, ci = (LEVEL_INDEX[low], LEVEL_INDEX[high]) if triangle == "synergy" else (LEVEL_INDEX[high], LEVEL_INDEX[low])
                rows.append({"bag": bag, "panel": key, "triangle": triangle, "matrix_row": LEVELS[ri], "matrix_column": LEVELS[ci], "delta_r2": row.delta_r2, "ci_lo": row.ci_lo, "ci_hi": row.ci_hi, "p_raw": row.p_raw, "p_holm_adjusted": _p(pd.Series(row._asdict())), "significance": _stars(_p(pd.Series(row._asdict()))), "adjustment_family": "six pairwise model-level comparisons within BAG measure and trajectory", "n_subjects": row.n_subjects, "n_countries": row.n_countries})
        for row in stats[(stats.comparison_type == arm_type) & (stats.bag == bag)].itertuples(index=False):
            level = RUNG_TO_LEVEL[row.model_a]; pvalue = _p(pd.Series(row._asdict()), arm=True)
            rows.append({"bag": bag, "panel": key, "triangle": "diagonal", "matrix_row": level, "matrix_column": level, "delta_r2": row.delta_r2, "ci_lo": row.ci_lo, "ci_hi": row.ci_hi, "p_raw": row.p_raw, "p_holm_adjusted": pvalue, "significance": _stars(pvalue), "adjustment_family": "four within-level arm comparisons within BAG measure and selection rule", "n_subjects": row.n_subjects, "n_countries": row.n_countries})
    else:
        for row in stats[(stats.comparison_type == comparison_type) & (stats.bag == bag)].itertuples(index=False):
            high, low = RUNG_TO_LEVEL[row.model_a], RUNG_TO_LEVEL[row.model_b]; pvalue = _p(pd.Series(row._asdict()))
            rows.append({"bag": bag, "panel": key, "triangle": key, "matrix_row": low, "matrix_column": high, "delta_r2": row.delta_r2, "ci_lo": row.ci_lo, "ci_hi": row.ci_hi, "p_raw": row.p_raw, "p_holm_adjusted": pvalue, "significance": _stars(pvalue), "adjustment_family": "six pairwise model-level comparisons within BAG measure and trajectory", "n_subjects": row.n_subjects, "n_countries": row.n_countries})
    frame = pd.DataFrame(rows)
    expected = 16 if arm_type else 6
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} cells for {bag}/{key}, found {len(frame)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--hpo-set", choices=("k10", "k63"), default="k10")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "outputs/figures/HPO" / args.hpo_set / "supplementary"
    source = args.repro_data_root.resolve() / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}" / "main_statistics/model_comparison/complexity_and_arm_comparisons.csv"
    stats = pd.read_csv(source); fig, axes = plt.subplots(2, 5, figsize=(12.6, 5.65), squeeze=False, gridspec_kw={"hspace": .42, "wspace": .28}); panels = []
    for row_index, bag in enumerate(("structural", "functional")):
        axes[row_index, 0].text(-.08, 1.27 if row_index == 0 else 1.12, f"{'ab'[row_index]}. {'Structural' if bag == 'structural' else 'Functional'} BAG", transform=axes[row_index, 0].transAxes, fontsize=11)
        for column, (key, title, kind, arm) in enumerate(SPECS):
            ax = axes[row_index, column]; frame = _rows(stats, bag, key, kind, arm)
            parent_figure._draw_heatmap(ax, frame)
            if row_index == 0:
                ax.set_xlabel("")
            if column > 0:
                ax.set_ylabel("")
            if row_index == 0:
                ax.set_title(title + ("\nSynergy upper · redundancy lower" if arm else "\nUpper triangle"), fontsize=10, pad=7)
            panels.append(Panel(panel_id=f"{'ab'[row_index]}{column+1}_{bag}_{key}", frame=frame, description=f"Matched model-complexity comparisons for {bag}, {title.lower()}.", test="Two-sided country-cluster bootstrap, 10,000 draws; Holm correction within adjustment_family."))
    fig.subplots_adjust(left=.065, right=.90, top=.85, bottom=.08); cax = fig.add_axes([.92,.20,.014,.58]); fig.colorbar(ScalarMappable(norm=Normalize(-LIMIT,LIMIT,clip=True), cmap="seismic"), cax=cax).set_label("ΔR²")
    output_dir.mkdir(parents=True, exist_ok=True); stem=f"fig_s13_model_complexity_comparisons_hpo_{args.hpo_set}"
    for extension in ("pdf", "svg", "png", "tiff"): fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", dpi=600 if extension in {"png","tiff"} else None)
    plt.close(fig); write_source_data(stem, panels, output_dir, source_paths=[str(source)]); print(f"Saved: {output_dir / (stem + '.png')}")


if __name__ == "__main__": main()
