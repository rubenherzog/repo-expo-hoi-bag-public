#!/usr/bin/env python3
"""K versus incremental R2, with 95% hierarchical-bootstrap intervals.

One panel per BAG. The x axis is the region size K on a log scale, ending in the
whole BAG-blind pool (ALL) drawn apart from the grid, because it is a control
rather than another point on the curve. Top20 is marked as the prespecified
primary region.

The curve shows how the incremental R2 decays as the high-performing region is
widened; it is not a device for choosing K.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_ROW_ORDER,
    LEVEL_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    SYN_COLOR,
    style_axis,
)
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import PRIMARY_REGION, ROOT

ARM_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
# Publication spelling of the two discovery arms, as used by the other
# sensitivity figures of this series.
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
# d2 dotted, d3 solid: the two model levels stay distinguishable in greyscale.
LEVEL_STYLE = {"xgb_tree_d2": (":", "s"), "xgb_tree_d3": ("-", "o")}


def _k_of(region: str) -> int | None:
    return None if region == "ALL" else int(region.removeprefix("Top"))


def build(results_path: Path, output_dir: Path, *, skip_source_data: bool = False) -> Path:
    results = pd.read_csv(results_path)
    grid = results[results["region"].ne("ALL")].copy()
    grid["K_region"] = grid["region"].map(_k_of)
    grid = grid.sort_values("K_region")
    control = results[results["region"].eq("ALL")]

    ks = sorted(grid["K_region"].unique())
    # ALL sits to the right of the grid, visually detached from the curve.
    all_x = ks[-1] * 2.2

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), gridspec_kw={"hspace": 0.32})
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    for row_index, bag in enumerate(BAG_ROW_ORDER):
        axis = axes[row_index]
        axis.axhline(0.0, color="#444444", linewidth=0.9, zorder=1)
        rows = []

        for arm in ("o_min", "o_max"):
            for rung in ("xgb_tree_d2", "xgb_tree_d3"):
                cells = grid[
                    grid["bag"].eq(bag) & grid["arm"].eq(arm) & grid["rung"].eq(rung)
                ]
                if cells.empty:
                    continue
                colour = ARM_COLOR[arm]
                style, marker = LEVEL_STYLE[rung]
                x = cells["K_region"].to_numpy(float)
                y = cells["observed_delta_r2"].to_numpy(float)

                axis.fill_between(
                    x, cells["ci_lo"], cells["ci_hi"], color=colour, alpha=0.12, linewidth=0
                )
                axis.plot(
                    x, y, style, color=colour, linewidth=1.6, marker=marker, markersize=4.2,
                    label=f"{ARM_LABEL[arm]} {LEVEL_LABELS[rung]}", zorder=3,
                )

                primary = cells[cells["region"].eq(PRIMARY_REGION)]
                if not primary.empty:
                    axis.plot(
                        primary["K_region"], primary["observed_delta_r2"], marker,
                        color=colour, markersize=9, markerfacecolor="white",
                        markeredgewidth=2.0, zorder=4,
                    )

                pool = control[
                    control["bag"].eq(bag) & control["arm"].eq(arm) & control["rung"].eq(rung)
                ]
                if not pool.empty:
                    pool = pool.iloc[0]
                    axis.errorbar(
                        all_x, pool["observed_delta_r2"],
                        yerr=[[pool["observed_delta_r2"] - pool["ci_lo"]],
                              [pool["ci_hi"] - pool["observed_delta_r2"]]],
                        fmt=marker, color=colour, markersize=5, capsize=3,
                        linewidth=1.2, alpha=0.75, zorder=3,
                    )

                rows.append(
                    cells[
                        ["bag", "arm", "rung", "region", "K_region", "observed_delta_r2",
                         "ci_lo", "ci_hi", "p_one_sided", "holm_p", "n_countries", "n_subjects"]
                    ]
                )

        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        drawn.append(frame)

        axis.set_xscale("log")
        axis.set_xticks([*ks, all_x])
        axis.set_xticklabels([*[str(int(k)) for k in ks], "ALL"], fontsize=8.6)
        axis.minorticks_off()
        axis.set_ylabel("Incremental R² over baseline")
        axis.set_title(
            f"{ROW_LETTERS[row_index]}. {BAG_LABELS[bag]}", loc="left", fontsize=11, pad=6
        )
        style_axis(axis)
        if row_index == 0:
            axis.legend(frameon=False, fontsize=8.4, ncol=4, loc="upper center",
                        bbox_to_anchor=(0.5, 1.26), columnspacing=1.2, handletextpad=0.5)

        panels.append(
            Panel(
                panel_id=f"{ROW_LETTERS[row_index]}_{bag}_boot_k",
                frame=frame,
                description=(
                    f"Incremental R² of the Top-K region over the covariate baseline, "
                    f"{BAG_LABELS[bag]}, with 95% hierarchical-bootstrap intervals."
                ),
                columns={
                    "observed_delta_r2": "Unweighted mean over countries of (SSE_base - SSE_TopK)/SST",
                    "ci_lo": "2.5th percentile of the nested bootstrap",
                    "holm_p": f"Holm-adjusted within the four primary {PRIMARY_REGION} tests; blank elsewhere",
                },
                test=(
                    "Nested bootstrap, 10,000 draws: countries fixed, country-years resampled "
                    "within country, subjects resampled within country-year."
                ),
                notes=(
                    "Open markers mark the prespecified primary region. ALL is the BAG-blind "
                    "control, drawn apart from the grid. The K grid is a sensitivity curve and "
                    "is not used to select K."
                ),
            )
        )

    axes[-1].set_xlabel("Candidate region size K (log scale)")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_hierarchical_oof_bootstrap_k_curve"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}", bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        write_source_data(stem, panels, output_dir, source_paths=[str(results_path)])
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "hierarchical_oof_bootstrap_results.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary"
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.results, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
