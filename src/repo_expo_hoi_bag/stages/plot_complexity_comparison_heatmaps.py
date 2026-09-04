#!/usr/bin/env python3
"""Supplementary Fig. S13: matched model-complexity comparison heatmaps.

Each BAG measure occupies one row. Off-diagonal cells show the higher-complexity
model level minus the lower-complexity model level. For multivariate candidates,
the upper triangle contains the synergy arm and the lower triangle the
redundancy arm; diagonal cells show synergy minus redundancy within a model
level. Best-single-exposure and covariate-baseline panels use the upper triangle
only. Holm-adjusted significance is marked without recomputing any plotted
between-level statistic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402
from repo_expo_hoi_bag.figures.style import (  # noqa: E402
    BAG_LABELS,
    BAG_ROW_ORDER,
    BAG_SHORT,
    LEVEL_AXIS_LABEL,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISONS = (
    ROOT / "outputs" / "dedup" / "model_comparison" / "complexity_pairwise_comparisons.csv"
)
DEFAULT_ARM_COMPARISONS = (
    ROOT / "outputs" / "dedup" / "model_comparison" / "complexity_arm_comparisons.csv"
)
DEFAULT_D2_COMPARISONS = (
    ROOT / "outputs" / "dedup" / "model_comparison" / "complexity_d2_fixed_comparisons.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figures" / "dedup" / "supplementary"
FIGURE_STEM = "fig_s13_model_complexity_comparisons"
LEVELS = ("OLS", "d1", "d2", "d3")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
COMMON_COLOR_LIMIT = 0.06


@dataclass(frozen=True)
class HeatmapSpecification:
    key: str
    title: str
    scope: str
    synergy_trajectory: str
    redundancy_trajectory: str | None
    diagonal_rule: str | None
    triangle_label: str


SPECIFICATIONS = (
    HeatmapSpecification(
        "selected",
        "Winner selected at each level",
        "level_selected",
        "Level-selected synergy arm",
        "Level-selected redundancy arm",
        "winner_at_each_level",
        "Synergy upper · redundancy lower",
    ),
    HeatmapSpecification(
        "d2_fixed",
        "d2 winner held fixed",
        "fixed_d2_candidate",
        "Fixed d2 synergy-arm candidate",
        "Fixed d2 redundancy-arm candidate",
        "fixed_d2_candidates",
        "Synergy upper · redundancy lower",
    ),
    HeatmapSpecification(
        "d3_fixed",
        "d3 winner held fixed",
        "fixed_candidate",
        "Fixed deployed synergy-arm candidate",
        "Fixed deployed redundancy-arm candidate",
        "fixed_d3_candidates",
        "Synergy upper · redundancy lower",
    ),
    HeatmapSpecification(
        "single",
        "Best single exposure",
        "level_selected",
        "Best single exposure",
        None,
        None,
        "Upper triangle",
    ),
    HeatmapSpecification(
        "baseline",
        "Covariate baseline",
        "level_selected",
        "Covariate baseline",
        None,
        None,
        "Upper triangle",
    ),
)


def _stars(adjusted_p: float) -> str:
    if adjusted_p < 0.001:
        return "***"
    if adjusted_p < 0.01:
        return "**"
    if adjusted_p < 0.05:
        return "*"
    return ""


def _cell_value(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _off_diagonal_rows(
    comparisons: pd.DataFrame,
    *,
    bag: str,
    specification: HeatmapSpecification,
    trajectory: str,
    triangle: str,
) -> list[dict[str, object]]:
    selected = comparisons[
        comparisons["analysis_scope"].eq(specification.scope)
        & comparisons["bag"].eq(bag)
        & comparisons["trajectory"].eq(trajectory)
    ]
    if len(selected) != 6:
        raise ValueError(
            f"Expected six comparisons for {bag}, {trajectory}; found {len(selected)}"
        )
    rows: list[dict[str, object]] = []
    for _, comparison in selected.iterrows():
        high = str(comparison["model_a"])
        low = str(comparison["model_b"])
        high_index = LEVEL_INDEX[high]
        low_index = LEVEL_INDEX[low]
        if high_index <= low_index:
            raise ValueError(f"Comparison is not ordered high minus low: {high} vs {low}")
        if triangle in {"synergy", "single", "baseline"}:
            matrix_row, matrix_column = low_index, high_index
        else:
            matrix_row, matrix_column = high_index, low_index
        adjusted_p = float(comparison["holm_p_within_bag_trajectory"])
        delta = float(comparison["delta_r2_a_minus_b"])
        rows.append(
            {
                "bag": bag,
                "panel": specification.key,
                "triangle": triangle,
                "matrix_row": LEVELS[matrix_row],
                "matrix_column": LEVELS[matrix_column],
                "higher_complexity_level": high,
                "lower_complexity_level": low,
                "comparison_direction": "higher complexity minus lower complexity",
                "delta_r2": delta,
                "delta_r2_ci_lower": float(comparison["delta_r2_ci_lower"]),
                "delta_r2_ci_upper": float(comparison["delta_r2_ci_upper"]),
                "p_raw": float(comparison["country_cluster_bootstrap_p_raw"]),
                "p_holm_adjusted": adjusted_p,
                "significance": _stars(adjusted_p),
                "adjustment_family": (
                    "six pairwise model-level comparisons within BAG measure and trajectory"
                ),
                "n_subjects": int(comparison["n_subjects"]),
                "n_countries": int(comparison["n_countries"]),
                "color_clipped": abs(delta) > COMMON_COLOR_LIMIT,
            }
        )
    return rows


def _diagonal_rows(
    arm_comparisons: pd.DataFrame,
    *,
    bag: str,
    specification: HeatmapSpecification,
) -> list[dict[str, object]]:
    if specification.diagonal_rule is None:
        return []
    selected = arm_comparisons[
        arm_comparisons["selection_rule"].eq(specification.diagonal_rule)
        & arm_comparisons["bag"].eq(bag)
    ]
    if len(selected) != 4:
        raise ValueError(
            f"Expected four arm comparisons for {bag}, {specification.key}; "
            f"found {len(selected)}"
        )
    rows: list[dict[str, object]] = []
    for _, comparison in selected.iterrows():
        level = str(comparison["model_level"])
        adjusted_p = float(comparison["holm_p_across_four_levels"])
        delta = float(comparison["delta_r2_synergy_minus_redundancy"])
        rows.append(
            {
                "bag": bag,
                "panel": specification.key,
                "triangle": "diagonal",
                "matrix_row": level,
                "matrix_column": level,
                "higher_complexity_level": level,
                "lower_complexity_level": level,
                "comparison_direction": "synergy arm minus redundancy arm",
                "delta_r2": delta,
                "delta_r2_ci_lower": float(comparison["delta_r2_ci_lower"]),
                "delta_r2_ci_upper": float(comparison["delta_r2_ci_upper"]),
                "p_raw": float(comparison["country_cluster_bootstrap_p_raw"]),
                "p_holm_adjusted": adjusted_p,
                "significance": _stars(adjusted_p),
                "adjustment_family": (
                    "four within-level arm comparisons within BAG measure and selection rule"
                ),
                "n_subjects": int(comparison["n_subjects"]),
                "n_countries": int(comparison["n_countries"]),
                "color_clipped": abs(delta) > COMMON_COLOR_LIMIT,
            }
        )
    return rows


def build_panel_frame(
    comparisons: pd.DataFrame,
    arm_comparisons: pd.DataFrame,
    *,
    bag: str,
    specification: HeatmapSpecification,
) -> pd.DataFrame:
    rows = _off_diagonal_rows(
        comparisons,
        bag=bag,
        specification=specification,
        trajectory=specification.synergy_trajectory,
        triangle=(
            specification.key if specification.key in {"single", "baseline"} else "synergy"
        ),
    )
    if specification.redundancy_trajectory is not None:
        rows.extend(
            _off_diagonal_rows(
                comparisons,
                bag=bag,
                specification=specification,
                trajectory=specification.redundancy_trajectory,
                triangle="redundancy",
            )
        )
    rows.extend(
        _diagonal_rows(
            arm_comparisons, bag=bag, specification=specification
        )
    )
    return pd.DataFrame(rows)


def _draw_heatmap(axis: plt.Axes, frame: pd.DataFrame) -> None:
    matrix = np.full((len(LEVELS), len(LEVELS)), np.nan)
    for _, row in frame.iterrows():
        matrix[LEVEL_INDEX[str(row["matrix_row"])]][
            LEVEL_INDEX[str(row["matrix_column"])]
        ] = float(row["delta_r2"])
    colormap = mpl.colormaps["seismic"].copy()
    colormap.set_bad("#F2F2F2")
    axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        norm=Normalize(vmin=-COMMON_COLOR_LIMIT, vmax=COMMON_COLOR_LIMIT, clip=True),
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xticks(range(len(LEVELS)), LEVELS)
    axis.set_yticks(range(len(LEVELS)), LEVELS)
    axis.set_xlabel(LEVEL_AXIS_LABEL, labelpad=3)
    axis.set_ylabel(LEVEL_AXIS_LABEL, labelpad=3)
    axis.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for _, row in frame.iterrows():
        row_index = LEVEL_INDEX[str(row["matrix_row"])]
        column_index = LEVEL_INDEX[str(row["matrix_column"])]
        delta = float(row["delta_r2"])
        stars = str(row["significance"])
        label = _cell_value(delta) + (f"\n{stars}" if stars else "")
        color = "white" if abs(delta) >= COMMON_COLOR_LIMIT * 0.52 else "black"
        axis.text(
            column_index,
            row_index,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=8.2,
            linespacing=0.82,
        )


def _save(figure: plt.Figure, output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in ("pdf", "svg", "png", "tiff"):
        output = output_dir / f"{FIGURE_STEM}.{extension}"
        kwargs: dict[str, object] = {}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(output, bbox_inches="tight", **kwargs)
        outputs.append(output)
    return tuple(outputs)


def render(
    comparisons_path: Path,
    d2_comparisons_path: Path,
    arm_comparisons_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    comparisons = pd.read_csv(comparisons_path)
    comparisons = pd.concat(
        [comparisons, pd.read_csv(d2_comparisons_path)], ignore_index=True
    )
    arm_comparisons = pd.read_csv(arm_comparisons_path)
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.7,
        }
    )
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(12.6, 5.65),
        squeeze=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.28},
    )
    panels: list[Panel] = []
    for row_index, bag in enumerate(BAG_ROW_ORDER):
        axes[row_index, 0].text(
            -0.08,
            1.27 if row_index == 0 else 1.12,
            f"{'ab'[row_index]}. {BAG_LABELS[bag]}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
        )
        for column_index, specification in enumerate(SPECIFICATIONS):
            axis = axes[row_index, column_index]
            frame = build_panel_frame(
                comparisons,
                arm_comparisons,
                bag=bag,
                specification=specification,
            )
            _draw_heatmap(axis, frame)
            if row_index == 0:
                axis.set_xlabel("")
            if column_index > 0:
                axis.set_ylabel("")
            if row_index == 0:
                axis.set_title(
                    f"{specification.title}\n{specification.triangle_label}",
                    fontsize=10,
                    pad=7,
                )
            panel_id = f"{'ab'[row_index]}{column_index + 1}_{BAG_SHORT[bag]}_{specification.key}"
            panels.append(
                Panel(
                    panel_id=panel_id,
                    frame=frame,
                    description=(
                        f"Matched model-complexity comparisons for {BAG_LABELS[bag]}, "
                        f"{specification.title.lower()}."
                    ),
                    columns={
                        "matrix_row": "model-level label printed on the heatmap row",
                        "matrix_column": "model-level label printed on the heatmap column",
                        "comparison_direction": "direction used for the plotted delta R²",
                        "delta_r2": "pooled out-of-fold R² difference shown by colour and cell label",
                        "delta_r2_ci_lower": "lower endpoint of the 95% country-cluster bootstrap interval",
                        "delta_r2_ci_upper": "upper endpoint of the 95% country-cluster bootstrap interval",
                        "p_raw": "raw two-sided country-cluster bootstrap p-value",
                        "p_holm_adjusted": "Holm-adjusted p-value used for asterisks",
                        "significance": "* p<0.05, ** p<0.01, *** p<0.001 after Holm adjustment",
                        "color_clipped": "true when colour is saturated beyond the common ±0.06 scale",
                    },
                    test=(
                        "Two-sided country-cluster bootstrap, 10,000 draws; Holm correction "
                        "within the family stated in adjustment_family."
                    ),
                    notes=(
                        "Off-diagonal cells are higher-complexity minus lower-complexity. "
                        "Diagonal cells are synergy minus redundancy. A common seismic colour "
                        "scale of −0.06 to 0.06 is used across all panels; exact values remain "
                        "printed when fixed-candidate OLS contrasts exceed the colour range."
                    ),
                )
            )

    figure.subplots_adjust(left=0.065, right=0.90, top=0.85, bottom=0.08)
    colorbar_axis = figure.add_axes([0.92, 0.20, 0.014, 0.58])
    colorbar = figure.colorbar(
        ScalarMappable(
            norm=Normalize(-COMMON_COLOR_LIMIT, COMMON_COLOR_LIMIT, clip=True),
            cmap="seismic",
        ),
        cax=colorbar_axis,
        extend="both",
    )
    colorbar.set_label("ΔR²", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    outputs = _save(figure, output_dir)
    plt.close(figure)
    source_outputs = write_source_data(
        FIGURE_STEM,
        panels,
        output_dir,
        source_paths=[
            str(comparisons_path),
            str(d2_comparisons_path),
            str(arm_comparisons_path),
        ],
    )
    return (*outputs, *source_outputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument(
        "--d2-comparisons", type=Path, default=DEFAULT_D2_COMPARISONS
    )
    parser.add_argument(
        "--arm-comparisons", type=Path, default=DEFAULT_ARM_COMPARISONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = render(
        args.comparisons,
        args.d2_comparisons,
        args.arm_comparisons,
        args.output_dir,
    )
    print(f"Wrote {len(outputs)} figure and Source Data files")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
