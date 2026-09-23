#!/usr/bin/env python3
"""Supplementary Fig. S13: paired-bootstrap Top1 and Top20 heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import BAG_LABELS, RED_COLOR, SYN_COLOR
from repo_expo_hoi_bag.stages import plot_complexity_comparison_heatmaps as parent
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import ROOT
from repo_expo_hoi_bag.stages.plot_hpo_complexity_comparison_heatmaps import RUNG_TO_LEVEL

LEVELS = ("OLS", "d1", "d2", "d3")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
ARM_LABEL = {"o_min": "Min O", "o_max": "Max O"}


def _stars(value: float) -> str:
    return "***" if value < .001 else "**" if value < .01 else "*" if value < .05 else ""


def complexity_panel(
    results: pd.DataFrame, bag: str, *, complexity_question: str, arm_question: str, region: str
) -> pd.DataFrame:
    """One full matrix: Min O upper, Max O lower, Min O−Max O diagonal."""
    complexity = results[results.question.eq(complexity_question) & results.bag.eq(bag)]
    arm_contrast = results[results.question.eq(arm_question) & results.bag.eq(bag)]
    rows: list[dict[str, object]] = []
    for row in complexity.itertuples(index=False):
        high, low = RUNG_TO_LEVEL[row.left_model], RUNG_TO_LEVEL[row.right_model]
        if row.arm == "o_min":
            matrix_row, matrix_column, triangle = low, high, "o_min"
        else:
            matrix_row, matrix_column, triangle = high, low, "o_max"
        rows.append({**row._asdict(), "matrix_row": matrix_row, "matrix_column": matrix_column,
                     "triangle": triangle, "delta_r2": row.observed_delta_r2,
                     "p_raw": row.p_two_sided, "p_holm_adjusted": getattr(row, "holm_p", np.nan),
                     "significance": _stars(row.p_two_sided),
                     "comparison_direction": "higher model level minus lower model level",
                     "adjustment_family": "six model-level contrasts within BAG x selection arm"})
    for row in arm_contrast.itertuples(index=False):
        level = RUNG_TO_LEVEL[row.rung]
        rows.append({**row._asdict(), "matrix_row": level, "matrix_column": level,
                     "triangle": "diagonal_min_o_minus_max_o", "delta_r2": row.observed_delta_r2,
                     "p_raw": row.p_two_sided, "p_holm_adjusted": row.holm_p,
                     "significance": _stars(row.p_two_sided),
                     "comparison_direction": f"Min O {region} set minus Max O {region} set",
                     "adjustment_family": "four Min O versus Max O contrasts within BAG"})
    frame = pd.DataFrame(rows)
    if len(frame) != 16:
        raise ValueError(f"Expected 16 {region} complexity cells for {bag}, found {len(frame)}")
    return frame


def reference_panel(results: pd.DataFrame, bag: str, question: str) -> pd.DataFrame:
    frame = results[results.question.eq(question) & results.bag.eq(bag)].copy()
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        rows.append({**row._asdict(), "matrix_row": RUNG_TO_LEVEL[row.rung],
                     "matrix_column": ARM_LABEL[row.arm], "delta_r2": row.observed_delta_r2,
                     "p_raw": row.p_two_sided, "p_holm_adjusted": getattr(row, "holm_p", np.nan),
                     "significance": _stars(row.p_two_sided),
                     "comparison_direction": "set minus " + str(row.right_model),
                     "adjustment_family": str(row.multiplicity_family)})
    output = pd.DataFrame(rows)
    if len(output) != 8:
        raise ValueError(f"Expected eight {question} cells for {bag}, found {len(output)}")
    return output


def _outline_min_max_triangles(axis: plt.Axes) -> None:
    axis.add_patch(Polygon([(0.5, -.5), (3.5, -.5), (3.5, 2.5), (2.5, 2.5),
                            (2.5, 1.5), (1.5, 1.5), (1.5, .5), (.5, .5)],
                           closed=True, fill=False, edgecolor=SYN_COLOR, linewidth=3.4,
                           joinstyle="round", clip_on=False, zorder=10))
    axis.add_patch(Polygon([(-.5, .5), (-.5, 3.5), (2.5, 3.5), (2.5, 2.5),
                            (1.5, 2.5), (1.5, 1.5), (.5, 1.5), (.5, .5)],
                           closed=True, fill=False, edgecolor=RED_COLOR, linewidth=3.4,
                           joinstyle="round", clip_on=False, zorder=10))


def _draw_reference_heatmap(axis: plt.Axes, frame: pd.DataFrame) -> None:
    matrix = np.full((len(LEVELS), len(ARM_LABEL)), np.nan)
    arms = tuple(ARM_LABEL.values())
    for row in frame.itertuples(index=False):
        matrix[LEVEL_INDEX[row.matrix_row], arms.index(row.matrix_column)] = row.delta_r2
    colormap = matplotlib.colormaps["seismic"].copy()
    colormap.set_bad("#F2F2F2")
    axis.imshow(np.ma.masked_invalid(matrix), cmap=colormap,
                norm=Normalize(-parent.COMMON_COLOR_LIMIT, parent.COMMON_COLOR_LIMIT, clip=True),
                interpolation="nearest", aspect="auto")
    axis.set_xticks(range(len(arms)), arms)
    axis.set_yticks(range(len(LEVELS)), LEVELS)
    axis.set_xticks(np.arange(-.5, len(arms), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(LEVELS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in frame.itertuples(index=False):
        value = float(row.delta_r2)
        label = parent._cell_value(value) + (f"\n{row.significance}" if row.significance else "")
        color = "white" if abs(value) >= parent.COMMON_COLOR_LIMIT * .52 else "black"
        axis.text(arms.index(row.matrix_column), LEVEL_INDEX[row.matrix_row], label,
                  ha="center", va="center", color=color, fontsize=7.8, linespacing=.82)
    axis.add_patch(Rectangle((-.5, -.5), 1, 4, fill=False, edgecolor=SYN_COLOR,
                             linewidth=3.4, joinstyle="round", clip_on=False, zorder=10))
    axis.add_patch(Rectangle((.5, -.5), 1, 4, fill=False, edgecolor=RED_COLOR,
                             linewidth=3.4, joinstyle="round", clip_on=False, zorder=10))


def _set_cell_text_black(axis: plt.Axes) -> None:
    """Keep every numeric cell label readable in the common publication style."""
    for text in axis.texts:
        text.set_color("black")
        text.set_fontsize(text.get_fontsize() + 2)


def build(
    top20_path: Path, top1_sensitivity_path: Path, top1_complexity_path: Path,
    output_dir: Path, *, skip_source_data: bool = False,
) -> Path:
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    top20 = pd.read_csv(top20_path)
    top1_sensitivity = pd.read_csv(top1_sensitivity_path)
    top1_complexity = pd.read_csv(top1_complexity_path)
    figure, axes = plt.subplots(2, 6, figsize=(17.2, 6.3), squeeze=False,
                                gridspec_kw={"hspace": .30, "wspace": .10,
                                             "width_ratios": [1, 1, 1, 1, 1, 1]})
    panels: list[Panel] = []
    specs = (
        ("top1_complexity", "Min O upper · Max O lower", "complexity"),
        ("top1_single", "Best single exposure", "top1_vs_best_single_sensitivity"),
        ("top1_baseline", "Covariate baseline", "top1_vs_baseline_sensitivity"),
        ("top20_complexity", "Min O upper · Max O lower", "complexity"),
        ("top20_single", "Best single exposure", "top20_vs_best_single"),
        ("top20_baseline", "Covariate baseline", "top20_vs_baseline"),
    )
    for row_index, bag in enumerate(("structural", "functional")):
        group_titles = (
            ("a. Top 1 Structural BAG", "b. Top 20 Structural BAG")
            if bag == "structural"
            else ("c. Top 1 Functional BAG", "d. Top 20 Functional BAG")
        )
        for group_column, title in zip((0, 3), group_titles):
            axes[row_index, group_column].text(
                -.08, 1.23 if row_index == 0 else 1.08, title,
                transform=axes[row_index, group_column].transAxes,
                ha="left", va="bottom", fontsize=13,
            )
        for column, (key, title, question) in enumerate(specs):
            axis = axes[row_index, column]
            if key == "top1_complexity":
                frame = complexity_panel(top1_complexity, bag, complexity_question="top1_model_complexity",
                                         arm_question="top1_min_o_vs_max_o", region="Top1")
                parent._draw_heatmap(axis, frame)
                _outline_min_max_triangles(axis)
            elif key == "top20_complexity":
                frame = complexity_panel(top20, bag, complexity_question="model_complexity",
                                         arm_question="synergy_vs_redundancy", region="Top20")
                parent._draw_heatmap(axis, frame)
                _outline_min_max_triangles(axis)
            else:
                source = top1_sensitivity if key.startswith("top1") else top20
                frame = reference_panel(source, bag, question)
                _draw_reference_heatmap(axis, frame)
            _set_cell_text_black(axis)
            axis.set_xlabel("Model level" if key.endswith("complexity") else "Selection arm", labelpad=3)
            axis.set_ylabel("Model level" if column == 0 else "", labelpad=3)
            if column > 0:
                axis.set_yticks([])
            if row_index == 0:
                axis.set_xlabel("")
                axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            if row_index == 0:
                axis.set_title(title, fontsize=11.4, pad=7)
            panels.append(Panel(
                panel_id=f"{'ab'[row_index]}{column + 1}_{'str' if bag == 'structural' else 'fun'}_{key}",
                frame=frame,
                description=f"Paired country-balanced {key.replace('_', ' ')} comparisons for {BAG_LABELS[bag]}.",
                test="Paired stratified cluster bootstrap, 10,000 draws; asterisks denote nominal two-sided bootstrap P values. Holm-adjusted P values are retained in source data.",
            ))
    figure.subplots_adjust(left=.055, right=.90, top=.85, bottom=.10)
    colorbar_axis = figure.add_axes([.92, .20, .014, .58])
    colorbar = figure.colorbar(ScalarMappable(norm=Normalize(-parent.COMMON_COLOR_LIMIT,
                                                               parent.COMMON_COLOR_LIMIT,
                                                               clip=True), cmap="seismic"),
                              cax=colorbar_axis)
    colorbar.set_label("Paired ΔR²", fontsize=12)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_s13_model_complexity_comparisons_hpo_k10"
    for extension in ("png", "pdf", "svg", "tiff"):
        figure.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight",
                       dpi=600 if extension in {"png", "tiff"} else None)
    plt.close(figure)
    if not skip_source_data:
        write_source_data(stem, panels, output_dir,
                          source_paths=[str(top20_path), str(top1_sensitivity_path), str(top1_complexity_path)])
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/paired_bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top20", type=Path, default=root / "paired_hierarchical_oof_comparisons.csv")
    parser.add_argument("--top1-sensitivity", type=Path, default=root / "paired_hierarchical_oof_top1_sensitivity.csv")
    parser.add_argument("--top1-complexity", type=Path, default=root / "paired_hierarchical_oof_top1_complexity.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/main/paper/complete/figures/supplementary")
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.top20, args.top1_sensitivity, args.top1_complexity, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
