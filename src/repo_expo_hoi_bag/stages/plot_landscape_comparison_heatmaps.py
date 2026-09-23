#!/usr/bin/env python3
"""Supplementary figure for the landscape-level model comparisons.

Replaces the winner-selected / d2-fixed / d3-fixed panels of Fig. S13 with
three conceptual columns per BAG, all computed on complete candidate
landscapes rather than on a selected candidate:

  1. a 4x4 model-level matrix whose upper triangle is the synergy landscape,
     lower triangle the redundancy landscape and diagonal the arm contrast;
  2. multivariate landscape minus single-exposure landscape;
  3. multivariate landscape minus covariate baseline.

The visual grammar (structural row above functional, diverging map centred on
zero, numeric delta inside each cell, Holm stars, arm outlines) follows the
parent figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import RED_COLOR, SYN_COLOR
from repo_expo_hoi_bag.stages import plot_complexity_comparison_heatmaps as parent_figure
from repo_expo_hoi_bag.stages import plot_hpo_complexity_comparison_heatmaps as parent_hpo
from repo_expo_hoi_bag.stages.compute_landscape_model_comparisons import ROOT

LEVELS = ("OLS", "d1", "d2", "d3")
LEVEL_INDEX = {value: index for index, value in enumerate(LEVELS)}
RUNG_TO_LEVEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
ARM_ROWS = ("Synergy", "Redundancy")
OBJECTIVE_ROW = {"o_min": "Synergy", "o_max": "Redundancy"}

# Kept at the parent figure's symmetric limit. Every contrast that does not
# involve the OLS level falls inside it; the OLS landscape is a genuine
# outlier because unregularised OLS degrades badly at large set sizes, so its
# cells saturate. The exact value is always printed inside the cell and the
# caption records the limit, so no number is hidden by the clip.
LIMIT = 0.06


def _stars(value: float) -> str:
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def _matrix_cells(tests: pd.DataFrame, bag: str) -> pd.DataFrame:
    """Column 1: the 4x4 model-level matrix for one BAG."""
    rows: list[dict[str, object]] = []
    selected = tests[tests["bag"].eq(bag)]

    complexity = selected[selected["comparison_family"].eq("model_complexity")]
    for objective, triangle in (("o_min", "synergy"), ("o_max", "redundancy")):
        for row in complexity[complexity["objective"].eq(objective)].itertuples(index=False):
            high, low = RUNG_TO_LEVEL[row.rung_a], RUNG_TO_LEVEL[row.rung_b]
            # Synergy fills the upper triangle, redundancy the lower one.
            row_index, column_index = (
                (LEVEL_INDEX[low], LEVEL_INDEX[high])
                if triangle == "synergy"
                else (LEVEL_INDEX[high], LEVEL_INDEX[low])
            )
            rows.append(
                {
                    "bag": bag,
                    "panel": "landscape_matrix",
                    "comparison_family": "model_complexity",
                    "objective": objective,
                    "triangle": triangle,
                    "matrix_row": LEVELS[row_index],
                    "matrix_column": LEVELS[column_index],
                    "rung_a": row.rung_a,
                    "rung_b": row.rung_b,
                    "delta_r2": row.delta_mean_r2,
                    "ci_lo": row.ci_lo,
                    "ci_hi": row.ci_hi,
                    "p_raw": row.wilcoxon_p_raw,
                    "p_holm_adjusted": row.holm_p,
                    "significance": _stars(float(row.holm_p)),
                    "rank_biserial": row.rank_biserial,
                    "n_countries": row.n_countries,
                    "adjustment_family": row.multiplicity_family,
                }
            )

    arm = selected[selected["comparison_family"].eq("synergy_vs_redundancy")]
    for row in arm.itertuples(index=False):
        level = RUNG_TO_LEVEL[row.rung_a]
        rows.append(
            {
                "bag": bag,
                "panel": "landscape_matrix",
                "comparison_family": "synergy_vs_redundancy",
                "objective": "o_min_minus_o_max",
                "triangle": "diagonal",
                "matrix_row": level,
                "matrix_column": level,
                "rung_a": row.rung_a,
                "rung_b": row.rung_b,
                "delta_r2": row.delta_mean_r2,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_raw": row.wilcoxon_p_raw,
                "p_holm_adjusted": row.holm_p,
                "significance": _stars(float(row.holm_p)),
                "rank_biserial": row.rank_biserial,
                "n_countries": row.n_countries,
                "adjustment_family": row.multiplicity_family,
            }
        )

    frame = pd.DataFrame(rows)
    if len(frame) != 16:
        raise ValueError(f"Expected 16 matrix cells for {bag}, found {len(frame)}")
    return frame


def _arm_by_level_cells(tests: pd.DataFrame, bag: str, family: str, panel: str) -> pd.DataFrame:
    """Columns 2 and 3: a 2x4 arm-by-level heatmap."""
    rows = []
    selected = tests[tests["bag"].eq(bag) & tests["comparison_family"].eq(family)]
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "bag": bag,
                "panel": panel,
                "comparison_family": family,
                "objective": row.objective,
                "matrix_row": OBJECTIVE_ROW[str(row.objective)],
                "matrix_column": RUNG_TO_LEVEL[row.rung_a],
                "rung_a": row.rung_a,
                "rung_b": row.rung_b,
                "delta_r2": row.delta_mean_r2,
                "ci_lo": row.ci_lo,
                "ci_hi": row.ci_hi,
                "p_raw": row.wilcoxon_p_raw,
                "p_holm_adjusted": row.holm_p,
                "significance": _stars(float(row.holm_p)),
                "rank_biserial": row.rank_biserial,
                "n_countries": row.n_countries,
                "adjustment_family": row.multiplicity_family,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 8:
        raise ValueError(f"Expected 8 cells for {bag}/{panel}, found {len(frame)}")
    return frame


def _draw_arm_by_level(axis: plt.Axes, frame: pd.DataFrame) -> None:
    """A 2 (arm) x 4 (model level) diverging heatmap in the parent's style."""
    matrix = np.full((len(ARM_ROWS), len(LEVELS)), np.nan)
    for _, row in frame.iterrows():
        matrix[ARM_ROWS.index(str(row["matrix_row"]))][LEVEL_INDEX[str(row["matrix_column"])]] = float(
            row["delta_r2"]
        )
    colormap = matplotlib.colormaps["seismic"].copy()
    colormap.set_bad("#F2F2F2")
    # Cells are half as tall as they are wide. Square cells would leave the
    # 2x4 grid stranded in whitespace beside the 4x4 matrix; filling the axes
    # entirely would stretch two rows over four rows' height. This keeps the
    # panel legible at the same width as its neighbours.
    axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        norm=Normalize(vmin=-LIMIT, vmax=LIMIT, clip=True),
        interpolation="nearest",
        aspect=0.5,
    )
    axis.set_xticks(range(len(LEVELS)), LEVELS)
    axis.set_yticks(range(len(ARM_ROWS)), ARM_ROWS)
    axis.set_xlabel("Model level", labelpad=3)
    axis.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ARM_ROWS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for _, row in frame.iterrows():
        delta = float(row["delta_r2"])
        stars = str(row["significance"])
        label = parent_figure._cell_value(delta) + (f"\n{stars}" if stars else "")
        axis.text(
            LEVEL_INDEX[str(row["matrix_column"])],
            ARM_ROWS.index(str(row["matrix_row"])),
            label,
            ha="center",
            va="center",
            color="white" if abs(delta) >= LIMIT * 0.52 else "black",
            fontsize=8.2,
            linespacing=0.82,
        )


def build(tests_path: Path, output_dir: Path, *, skip_source_data: bool = False) -> Path:
    tests = pd.read_csv(tests_path)
    # wspace is generous because columns 2 and 3 carry "Synergy"/"Redundancy"
    # row labels, which are far wider than the matrix column's level ticks.
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 6.1),
        squeeze=False,
        gridspec_kw={"hspace": 0.46, "wspace": 0.52, "width_ratios": [1.15, 1, 1]},
    )
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    specifications = (
        ("landscape_matrix", "Candidate landscape\nMin O-info ↑ · Max O-info ↓", None),
        (
            "vs_single",
            "Multivariate vs\nsingle-exposure landscape",
            "vs_single_landscape",
        ),
        ("vs_baseline", "Multivariate landscape vs\ncovariate baseline", "vs_covariate_baseline"),
    )

    for row_index, bag in enumerate(("structural", "functional")):
        axes[row_index, 0].text(
            -0.14,
            1.30 if row_index == 0 else 1.22,
            f"{'ab'[row_index]}. {'Structural' if bag == 'structural' else 'Functional'} BAG",
            transform=axes[row_index, 0].transAxes,
            fontsize=11,
        )
        for column, (key, title, family) in enumerate(specifications):
            axis = axes[row_index, column]
            if family is None:
                frame = _matrix_cells(tests, bag)
                parent_figure._draw_heatmap(axis, frame)
                parent_hpo._outline_arm_triangles(axis, frame)
            else:
                frame = _arm_by_level_cells(tests, bag, family, key)
                _draw_arm_by_level(axis, frame)
                # imshow with a fixed aspect centres the 2x4 grid vertically in
                # a cell sized for the 4x4 matrix. Anchoring to the top aligns
                # its first row with the matrix's first row.
                axis.set_anchor("N")
            drawn.append(frame)

            if row_index == 0:
                axis.set_xlabel("")
                axis.set_title(title, fontsize=10, pad=7)
            if column > 0:
                axis.set_ylabel("")
            panels.append(
                Panel(
                    panel_id=f"{'ab'[row_index]}{column + 1}_{bag}_{key}",
                    frame=frame,
                    description=(
                        f"Landscape-level comparisons for {bag} BAG, {title.replace(chr(10), ' ')}."
                    ),
                    columns={
                        "delta_r2": "Mean paired difference in landscape R² across held-out countries",
                        "p_holm_adjusted": "Holm-adjusted P within the family named in adjustment_family",
                        "rank_biserial": "Rank-biserial correlation effect size",
                    },
                    test=(
                        "Two-sided Wilcoxon signed-rank test across held-out countries; "
                        "95% CI from a 10,000-draw paired country bootstrap; Holm "
                        "correction within adjustment_family."
                    ),
                    notes=(
                        "Landscape R²: median R² across the 20 BAG-blind candidates at each "
                        "set size, averaged over k=3–30. No candidate is selected on "
                        f"performance. Colour is clipped at ±{LIMIT}; every value is printed."
                    ),
                )
            )

    figure.subplots_adjust(left=0.085, right=0.88, top=0.84, bottom=0.09)
    colorbar_axis = figure.add_axes([0.905, 0.20, 0.016, 0.56])
    figure.colorbar(
        ScalarMappable(norm=Normalize(-LIMIT, LIMIT, clip=True), cmap="seismic"),
        cax=colorbar_axis,
    ).set_label("ΔR² (landscape)")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_s13_landscape_model_comparisons_hpo_k10"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        # write_source_data appends its own source_data/ subfolder.
        write_source_data(stem, panels, output_dir, source_paths=[str(tests_path)])
        # A single flat mirror of every drawn cell across all panels, used by
        # the figure/table consistency test. The per-panel sheets and CSVs
        # written above keep their standard names.
        pd.concat(drawn, ignore_index=True).to_csv(
            output_dir / "source_data" / f"{stem}_source_data_all_cells.csv", index=False
        )
    return output_dir / f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        type=Path,
        default=ROOT / "outputs/sensitivity/landscape_comparisons/main_k10/landscape_comparison_tests.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/main/paper/complete/figures/supplementary",
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.tests, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
