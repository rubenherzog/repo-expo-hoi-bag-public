#!/usr/bin/env python3
"""Supplementary figure for the K=20 country-cross-selected performance frontier.

Follows the Fig. S13 visual grammar: structural BAG above functional, diverging
ΔR² heatmaps centred on zero, the numeric ΔR² inside every cell, significance
stars from Holm-adjusted P, Arial with editable vector text.

Three conceptual columns per BAG:

  1. a 4x4 model-level matrix -- upper triangle the synergy arm, lower triangle
     the redundancy arm, diagonal the synergy-minus-redundancy arm contrast, so
     d2-d1, d3-d1 and d3-d2 all appear directly;
  2. frontier minus the country-cross-selected best single exposure;
  3. frontier minus the covariate baseline.

A narrow strip at the right of each row carries the mean Top-20 candidate
Jaccard, so frontier stability is visible without competing with the effect
estimates.
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
from matplotlib.gridspec import GridSpec

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.stages import plot_complexity_comparison_heatmaps as parent_figure
from repo_expo_hoi_bag.stages import plot_hpo_complexity_comparison_heatmaps as parent_hpo
from repo_expo_hoi_bag.stages.compute_performance_frontier import PRIMARY_K, ROOT

LEVELS = ("OLS", "d1", "d2", "d3")
LEVEL_INDEX = {value: index for index, value in enumerate(LEVELS)}
RUNG_TO_LEVEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
ARM_ROWS = ("Synergy", "Redundancy")
ARM_OF_ROW = {"Synergy": "o_min", "Redundancy": "o_max"}
ROW_OF_ARM = {"o_min": "Synergy", "o_max": "Redundancy"}

# Same symmetric limit as the parent figure. Every frontier contrast that does
# not involve the OLS level falls well inside it; OLS saturates because
# unregularised OLS degrades at large set sizes. Every value is printed.
LIMIT = 0.06
# Stability is a similarity in [0, 1]; a sequential map keeps it visually
# distinct from the diverging effect panels.
STABILITY_RANGE = (0.5, 1.0)


def _stars(value: float) -> str:
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def _matrix_cells(tests: pd.DataFrame, bag: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = tests[tests["bag"].eq(bag)]

    complexity = selected[selected["comparison_family"].eq("model_complexity")]
    for arm, triangle in (("o_min", "synergy"), ("o_max", "redundancy")):
        for row in complexity[complexity["arm"].eq(arm)].itertuples(index=False):
            high, low = RUNG_TO_LEVEL[row.rung_a], RUNG_TO_LEVEL[row.rung_b]
            row_index, column_index = (
                (LEVEL_INDEX[low], LEVEL_INDEX[high])
                if triangle == "synergy"
                else (LEVEL_INDEX[high], LEVEL_INDEX[low])
            )
            rows.append(_cell(bag, "frontier_matrix", row, triangle, LEVELS[row_index], LEVELS[column_index]))

    arm_rows = selected[selected["comparison_family"].eq("synergy_vs_redundancy")]
    for row in arm_rows.itertuples(index=False):
        level = RUNG_TO_LEVEL[row.rung_a]
        rows.append(_cell(bag, "frontier_matrix", row, "diagonal", level, level))

    frame = pd.DataFrame(rows)
    if len(frame) != 16:
        raise ValueError(f"Expected 16 matrix cells for {bag}, found {len(frame)}")
    return frame


def _cell(bag: str, panel: str, row, triangle: str, matrix_row: str, matrix_column: str) -> dict[str, object]:
    holm = float(row.holm_p)
    return {
        "bag": bag,
        "panel": panel,
        "comparison_family": row.comparison_family,
        "arm": row.arm,
        "triangle": triangle,
        "matrix_row": matrix_row,
        "matrix_column": matrix_column,
        "rung_a": row.rung_a,
        "rung_b": row.rung_b,
        "K": row.K,
        "delta_r2": row.delta_mean_r2,
        "ci_lo": row.ci_lo,
        "ci_hi": row.ci_hi,
        "p_raw": row.wilcoxon_p_raw,
        "p_holm_adjusted": holm,
        "significance": _stars(holm),
        "rank_biserial": row.rank_biserial,
        "n_countries": row.n_countries,
        "adjustment_family": row.multiplicity_family,
    }


def _arm_by_level_cells(tests: pd.DataFrame, bag: str, family: str, panel: str) -> pd.DataFrame:
    selected = tests[tests["bag"].eq(bag) & tests["comparison_family"].eq(family)]
    rows = [
        _cell(bag, panel, row, panel, ROW_OF_ARM[str(row.arm)], RUNG_TO_LEVEL[str(row.rung_a)])
        for row in selected.itertuples(index=False)
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != 8:
        raise ValueError(f"Expected 8 cells for {bag}/{panel}, found {len(frame)}")
    return frame


def _stability_cells(summary: pd.DataFrame, bag: str, k: int = PRIMARY_K) -> pd.DataFrame:
    selected = summary[summary["bag"].eq(bag)]
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "bag": bag,
                "panel": "frontier_stability",
                "arm": row.arm,
                "matrix_row": ROW_OF_ARM[str(row.arm)],
                "matrix_column": RUNG_TO_LEVEL[str(row.rung)],
                "K": k,
                "candidate_jaccard_mean": getattr(row, f"top{k}_candidate_jaccard_mean"),
                "candidate_jaccard_median": getattr(row, f"top{k}_candidate_jaccard_median"),
                "exposure_jaccard_mean": getattr(row, f"top{k}_exposure_jaccard_mean"),
            }
        )
    return pd.DataFrame(rows)


def _draw_grid(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    norm,
    cmap: str,
    annotate_stars: bool,
    white_above: float,
    fmt,
) -> None:
    """A rows x columns heatmap in the parent figure's cell style."""
    matrix = np.full((len(rows), len(columns)), np.nan)
    for _, row in frame.iterrows():
        matrix[rows.index(str(row["matrix_row"]))][columns.index(str(row["matrix_column"]))] = float(
            row[value_column]
        )
    colormap = matplotlib.colormaps[cmap].copy()
    colormap.set_bad("#F2F2F2")
    axis.imshow(
        np.ma.masked_invalid(matrix), cmap=colormap, norm=norm, interpolation="nearest", aspect=0.5
    )
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for _, row in frame.iterrows():
        value = float(row[value_column])
        label = fmt(value)
        if annotate_stars and str(row.get("significance", "")):
            label += f"\n{row['significance']}"
        axis.text(
            columns.index(str(row["matrix_column"])),
            rows.index(str(row["matrix_row"])),
            label,
            ha="center",
            va="center",
            color="white" if abs(value) >= white_above else "black",
            fontsize=8.2,
            linespacing=0.82,
        )


def build(
    tests_path: Path, summary_path: Path, output_dir: Path, *, skip_source_data: bool = False
) -> Path:
    tests = pd.read_csv(tests_path)
    summary = pd.read_csv(summary_path)

    figure = plt.figure(figsize=(12.4, 6.3))
    grid = GridSpec(
        2,
        4,
        figure=figure,
        width_ratios=[1.18, 1.0, 1.0, 0.72],
        hspace=0.5,
        wspace=0.62,
        left=0.075,
        right=0.86,
        top=0.83,
        bottom=0.09,
    )
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    effect_norm = Normalize(-LIMIT, LIMIT, clip=True)
    stability_norm = Normalize(*STABILITY_RANGE, clip=True)

    specifications = (
        ("frontier_matrix", "Cross-selected frontier\nMin O-info ↑ · Max O-info ↓", None),
        ("vs_single", "Frontier vs cross-selected\nbest single exposure", "vs_cross_selected_single"),
        ("vs_baseline", "Frontier vs\ncovariate baseline", "vs_covariate_baseline"),
    )

    for row_index, bag in enumerate(("structural", "functional")):
        first_axis = figure.add_subplot(grid[row_index, 0])
        first_axis.text(
            -0.16,
            1.30 if row_index == 0 else 1.22,
            f"{'ab'[row_index]}. {'Structural' if bag == 'structural' else 'Functional'} BAG",
            transform=first_axis.transAxes,
            fontsize=11,
        )
        for column, (key, title, family) in enumerate(specifications):
            axis = first_axis if column == 0 else figure.add_subplot(grid[row_index, column])
            if family is None:
                frame = _matrix_cells(tests, bag)
                parent_figure._draw_heatmap(axis, frame)
                parent_hpo._outline_arm_triangles(axis, frame)
            else:
                frame = _arm_by_level_cells(tests, bag, family, key)
                _draw_grid(
                    axis,
                    frame,
                    "delta_r2",
                    rows=ARM_ROWS,
                    columns=LEVELS,
                    norm=effect_norm,
                    cmap="seismic",
                    annotate_stars=True,
                    white_above=LIMIT * 0.52,
                    fmt=parent_figure._cell_value,
                )
                axis.set_anchor("N")
            drawn.append(frame)

            if row_index == 0:
                axis.set_title(title, fontsize=10, pad=7)
                axis.set_xlabel("")
            else:
                axis.set_xlabel("Model level", labelpad=3)
            if column > 0:
                axis.set_ylabel("")
            panels.append(
                Panel(
                    panel_id=f"{'ab'[row_index]}{column + 1}_{bag}_{key}",
                    frame=frame,
                    description=f"K={PRIMARY_K} frontier comparisons for {bag} BAG, {title.replace(chr(10), ' ')}.",
                    columns={
                        "delta_r2": "Mean paired difference in frontier R² across held-out countries",
                        "p_holm_adjusted": "Holm-adjusted P within adjustment_family",
                        "rank_biserial": "Rank-biserial correlation effect size",
                    },
                    test=(
                        "Two-sided Wilcoxon signed-rank across held-out countries; 95% CI from a "
                        "10,000-draw paired country bootstrap; Holm within adjustment_family."
                    ),
                    notes=(
                        f"Frontier: median held-out R² across the Top-{PRIMARY_K} candidates ranked "
                        "without the evaluation country. Country-cross-selected, not nested CV. "
                        f"Colour clipped at ±{LIMIT}; every value is printed."
                    ),
                )
            )

        stability_axis = figure.add_subplot(grid[row_index, 3])
        stability = _stability_cells(summary, bag)
        _draw_grid(
            stability_axis,
            stability,
            "candidate_jaccard_mean",
            rows=ARM_ROWS,
            columns=LEVELS,
            norm=stability_norm,
            cmap="Greens",
            annotate_stars=False,
            white_above=0.92,
            fmt=lambda value: f"{value:.2f}",
        )
        stability_axis.set_anchor("N")
        stability_axis.set_ylabel("")
        if row_index == 0:
            stability_axis.set_title(f"Top-{PRIMARY_K} frontier\nstability (Jaccard)", fontsize=10, pad=7)
            stability_axis.set_xlabel("")
        else:
            stability_axis.set_xlabel("Model level", labelpad=3)
        drawn.append(stability)
        panels.append(
            Panel(
                panel_id=f"{'ab'[row_index]}4_{bag}_stability",
                frame=stability,
                description=f"Mean Top-{PRIMARY_K} candidate-identity Jaccard for {bag} BAG.",
                columns={
                    "candidate_jaccard_mean": "Mean Jaccard of leave-one-country-out vs global Top-K, by candidate identity",
                    "candidate_jaccard_median": "Median of the same quantity",
                    "exposure_jaccard_mean": "Mean Jaccard of the exposure unions (a separate quantity)",
                },
                test="Descriptive; no test.",
                notes="Candidate-identity Jaccard, not exposure Jaccard.",
            )
        )

    effect_bar = figure.add_axes([0.885, 0.42, 0.013, 0.38])
    figure.colorbar(ScalarMappable(norm=effect_norm, cmap="seismic"), cax=effect_bar).set_label("ΔR²")
    stability_bar = figure.add_axes([0.885, 0.10, 0.013, 0.22])
    figure.colorbar(ScalarMappable(norm=stability_norm, cmap="Greens"), cax=stability_bar).set_label(
        "Mean Jaccard"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_s13_performance_frontier_comparisons_hpo_k10"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        write_source_data(stem, panels, output_dir, source_paths=[str(tests_path), str(summary_path)])
        effect_cells = [frame for frame in drawn if "delta_r2" in frame.columns]
        pd.concat(effect_cells, ignore_index=True).to_csv(
            output_dir / "source_data" / f"{stem}_source_data_all_cells.csv", index=False
        )
    return output_dir / f"{stem}.png"


def main() -> None:
    frontier = ROOT / "outputs/sensitivity/performance_frontier/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=frontier / "frontier_comparison_tests.csv")
    parser.add_argument("--stability", type=Path, default=frontier / "frontier_stability_summary.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary"
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.tests, args.stability, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
