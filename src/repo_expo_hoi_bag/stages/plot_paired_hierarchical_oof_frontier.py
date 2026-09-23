#!/usr/bin/env python3
"""Top-K paired-bootstrap OOF ΔR² frontier, with Top20 marked primary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import BAG_LABELS, LEVEL_LABELS, RED_COLOR, SYN_COLOR, style_axis
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import ROOT

ARM_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
ARM_LABEL = {"o_min": "Min O", "o_max": "Max O"}
LEVEL_STYLE = {"ols": ("-", "o"), "xgb_tree_d1": ("-", "o"), "xgb_tree_d2": ("-", "o"), "xgb_tree_d3": ("-", "o")}
K_ORDER = (1, 5, 10, 20, 50, 100)


def _k(region: str) -> int | None:
    return None if region == "ALL" else int(region.removeprefix("Top"))


def _format_p(value: float) -> str:
    return f"{value:.4f}"


def build(
    results_path: Path, output_dir: Path, *, top1_path: Path,
    skip_source_data: bool = False,
) -> Path:
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    results = pd.read_csv(results_path)
    baseline = results[results.question.eq("top20_vs_baseline")].copy()
    sensitivity = results[results.question.eq("topk_frontier_sensitivity")].copy()
    top1 = pd.read_csv(top1_path)
    top1 = top1[top1.question.eq("top1_vs_baseline_sensitivity")].copy()
    grid = pd.concat([baseline, sensitivity, top1], ignore_index=True)
    grid["K"] = grid.region.map(_k)
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.6), gridspec_kw={"hspace": .34, "wspace": .18})
    panels: list[Panel] = []
    for index, bag in enumerate(("structural", "functional")):
        frame = grid[grid.bag.eq(bag)].copy()
        axes[index, 0].text(
            -0.08,
            1.13 if index == 0 else 1.12,
            f"{'ab'[index]}. {BAG_LABELS[bag]}",
            transform=axes[index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
        )
        for column, rung in enumerate(LEVEL_STYLE):
            ax = axes[index, column]
            ax.axhline(0, color="#444444", linewidth=.9)
            cell_frame = frame[frame.rung.eq(rung)].copy()
            for arm in ("o_min", "o_max"):
                cells = cell_frame[
                    cell_frame.arm.eq(arm)
                    & cell_frame.left_model.str.endswith(" set")
                    & cell_frame.right_model.eq("matched baseline")
                ]
                plot = cells[cells.K.notna()].sort_values("K")
                if plot.empty:
                    continue
                line, marker = LEVEL_STYLE[rung]
                colour = ARM_COLOR[arm]
                ax.fill_between(plot.K, plot.ci_lo, plot.ci_hi, color=colour, alpha=.10)
                ax.plot(plot.K, plot.observed_delta_r2, line, marker=marker, color=colour,
                        linewidth=1.5, markersize=4.2, label=ARM_LABEL[arm])
                primary = plot[plot.region.eq("Top20")]
                if not primary.empty:
                    ax.plot(primary.K, primary.observed_delta_r2, "o", color=colour, markersize=9,
                            markerfacecolor="white", markeredgewidth=2.0)
            ax.set_xscale("log"); ax.set_xticks(K_ORDER); ax.set_xticklabels([str(k) for k in K_ORDER], fontsize=12)
            ax.minorticks_off(); style_axis(ax)
            if index == 0:
                ax.set_title(LEVEL_LABELS[rung], fontsize=13, pad=8)
            if column == 0:
                ax.set_ylabel("Paired ΔR² vs baseline")
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
            if index == 1:
                ax.set_xlabel("K (log scale)")
            else:
                ax.set_xlabel("")
            primary_tests = cell_frame[cell_frame.question.eq("top20_vs_baseline")]
            p_handles = []
            for arm in ("o_min", "o_max"):
                row = primary_tests[primary_tests.arm.eq(arm)].iloc[0]
                p_handles.append(Line2D(
                    [], [], linestyle="None", marker="o", markersize=8.5,
                    markerfacecolor="white", markeredgecolor=ARM_COLOR[arm],
                    markeredgewidth=2.0, label=f"P={_format_p(row.p_two_sided)}",
                ))
            p_legend = ax.legend(
                handles=p_handles, frameon=False, fontsize=10.2, loc="lower left",
                handletextpad=.4, borderaxespad=.35,
            )
            if index == 0 and column == 3:
                ax.add_artist(p_legend)
                ax.legend(frameon=False, fontsize=12.5, loc="upper right")
            panels.append(Panel(panel_id=f"{'ab'[index]}_{bag}_{rung.removeprefix('xgb_tree_')}_topk", frame=cell_frame[cell_frame.K.notna()],
                                description=f"Top-K country-balanced OOF ΔR² for {BAG_LABELS[bag]}, {LEVEL_LABELS[rung]}, comparing Min O and Max O.",
                                test="Paired stratified cluster bootstrap, 10,000 draws. Top20 is primary; other K values are descriptive sensitivity points."))
        y_min = min(axis.get_ylim()[0] for axis in axes[index])
        y_max = max(axis.get_ylim()[1] for axis in axes[index])
        for axis in axes[index]:
            axis.set_ylim(y_min, y_max)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_hierarchical_oof_bootstrap_k_curve"
    for extension in ("png", "pdf", "svg", "tiff"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", dpi=600 if extension in {"png", "tiff"} else None)
    plt.close(fig)
    if not skip_source_data:
        write_source_data(stem, panels, output_dir,
                          source_paths=[str(results_path), str(top1_path)])
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/paired_bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "paired_hierarchical_oof_comparisons.csv")
    parser.add_argument("--top1", type=Path, default=root / "paired_hierarchical_oof_top1_sensitivity.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary")
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.results, args.output_dir, top1_path=args.top1, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
