#!/usr/bin/env python3
"""Top-K paired-bootstrap OOF ΔR² frontier versus the stored best single."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import BAG_LABELS, LEVEL_LABELS, style_axis
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import ROOT
from repo_expo_hoi_bag.stages.plot_paired_hierarchical_oof_frontier import (
    ARM_COLOR,
    ARM_LABEL,
    K_ORDER,
    LEVEL_STYLE,
    _format_p,
    _k,
)


def build(
    primary_path: Path,
    sensitivity_path: Path,
    output_dir: Path,
    *,
    top1_path: Path,
    skip_source_data: bool = False,
) -> Path:
    matplotlib.rcParams.update({
        "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    })
    primary = pd.read_csv(primary_path)
    primary = primary[primary.question.eq("top20_vs_best_single")].copy()
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity = sensitivity[sensitivity.question.eq("topk_vs_best_single_sensitivity")].copy()
    top1 = pd.read_csv(top1_path)
    top1 = top1[top1.question.eq("top1_vs_best_single_sensitivity")].copy()
    grid = pd.concat([primary, sensitivity, top1], ignore_index=True)
    grid["K"] = grid.region.map(_k)
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 7.6),
                                gridspec_kw={"hspace": .34, "wspace": .18})
    panels: list[Panel] = []
    for row_index, bag in enumerate(("structural", "functional")):
        frame = grid[grid.bag.eq(bag)].copy()
        axes[row_index, 0].text(
            -.08, 1.13 if row_index == 0 else 1.12,
            f"{'ab'[row_index]}. {BAG_LABELS[bag]}", transform=axes[row_index, 0].transAxes,
            ha="left", va="bottom", fontsize=13,
        )
        for column, rung in enumerate(LEVEL_STYLE):
            axis = axes[row_index, column]
            axis.axhline(0, color="#444444", linewidth=.9)
            cell = frame[frame.rung.eq(rung)].copy()
            for arm in ("o_min", "o_max"):
                plot = cell[cell.arm.eq(arm) & cell.K.notna()].sort_values("K")
                line, marker = LEVEL_STYLE[rung]
                colour = ARM_COLOR[arm]
                axis.fill_between(plot.K, plot.ci_lo, plot.ci_hi, color=colour, alpha=.10)
                axis.plot(plot.K, plot.observed_delta_r2, line, marker=marker, color=colour,
                          linewidth=1.5, markersize=4.2, label=ARM_LABEL[arm])
                selected = plot[plot.region.eq("Top20")]
                axis.plot(selected.K, selected.observed_delta_r2, "o", color=colour, markersize=9,
                          markerfacecolor="white", markeredgewidth=2.0)
            axis.set_xscale("log")
            axis.set_xticks(K_ORDER)
            axis.set_xticklabels([str(value) for value in K_ORDER], fontsize=12)
            axis.minorticks_off()
            style_axis(axis)
            if row_index == 0:
                axis.set_title(LEVEL_LABELS[rung], fontsize=13, pad=8)
            if column == 0:
                axis.set_ylabel("Paired ΔR² vs best single")
            else:
                axis.set_ylabel("")
                axis.set_yticks([])
            axis.set_xlabel("K (log scale)" if row_index == 1 else "")
            top20 = cell[cell.question.eq("top20_vs_best_single")]
            p_handles = []
            for arm in ("o_min", "o_max"):
                value = top20[top20.arm.eq(arm)].iloc[0].p_two_sided
                p_handles.append(Line2D(
                    [], [], linestyle="None", marker="o", markersize=8.5,
                    markerfacecolor="white", markeredgecolor=ARM_COLOR[arm], markeredgewidth=2.0,
                    label=f"P={_format_p(value)}",
                ))
            p_legend = axis.legend(handles=p_handles, frameon=False, fontsize=10.2,
                                   loc="lower left", handletextpad=.4, borderaxespad=.35)
            if row_index == 0 and column == 3:
                axis.add_artist(p_legend)
                axis.legend(frameon=False, fontsize=12.5, loc="upper right")
            panels.append(Panel(
                panel_id=f"{'ab'[row_index]}_{bag}_{rung.removeprefix('xgb_tree_')}_topk_single",
                frame=cell[cell.K.notna()],
                description=(f"Top-K country-balanced OOF ΔR² versus the stored best single exposure "
                             f"for {BAG_LABELS[bag]}, {LEVEL_LABELS[rung]}."),
                test=("Paired stratified cluster bootstrap, 10,000 draws. Top20 is primary; "
                      "other K values are descriptive sensitivity points."),
            ))
        y_min = min(axis.get_ylim()[0] for axis in axes[row_index])
        y_max = max(axis.get_ylim()[1] for axis in axes[row_index])
        for axis in axes[row_index]:
            axis.set_ylim(y_min, y_max)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_hierarchical_oof_best_single_k_curve"
    for extension in ("png", "pdf", "svg", "tiff"):
        figure.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight",
                       dpi=600 if extension in {"png", "tiff"} else None)
    plt.close(figure)
    if not skip_source_data:
        write_source_data(stem, panels, output_dir,
                          source_paths=[str(primary_path), str(sensitivity_path), str(top1_path)])
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/paired_bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=root / "paired_hierarchical_oof_comparisons.csv")
    parser.add_argument("--sensitivity", type=Path,
                        default=root / "paired_hierarchical_oof_best_single_frontier_sensitivity.csv")
    parser.add_argument("--top1", type=Path, default=root / "paired_hierarchical_oof_top1_sensitivity.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/main/paper/complete/figures/supplementary")
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.primary, args.sensitivity, args.output_dir, top1_path=args.top1, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
