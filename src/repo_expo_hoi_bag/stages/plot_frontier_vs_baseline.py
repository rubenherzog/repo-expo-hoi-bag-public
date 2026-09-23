#!/usr/bin/env python3
"""Compact supplementary figure: Top-20 frontier region versus covariate baseline.

One panel per BAG, four positions per panel (synergy/redundancy x d2/d3). The
point is the country-level mean frontier gain with its 95% country-bootstrap
interval; faint points behind it are the individual held-out countries, so the
heterogeneity driving each estimate stays visible.

The 20 frontier candidates are never drawn as independent observations: each
country contributes exactly one value, already averaged over its frontier.
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
from repo_expo_hoi_bag.figures.style import RED_COLOR, SYN_COLOR
from repo_expo_hoi_bag.stages.compute_frontier_vs_baseline import PRIMARY_K, ROOT

ARM_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
LEVEL_LABEL = {"xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
COLUMNS = (
    ("o_min", "xgb_tree_d2"),
    ("o_max", "xgb_tree_d2"),
    ("o_min", "xgb_tree_d3"),
    ("o_max", "xgb_tree_d3"),
)


def _stars(value: float) -> str:
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def build(
    tests_path: Path, summary_path: Path, output_dir: Path, *, skip_source_data: bool = False
) -> Path:
    tests = pd.read_csv(tests_path)
    summary = pd.read_csv(summary_path)
    primary = tests[tests["test"].eq("primary_mean_frontier_gain") & tests["K"].eq(PRIMARY_K)]
    positivity = tests[tests["test"].eq("secondary_positivity_vs_half") & tests["K"].eq(PRIMARY_K)]
    countries = summary[summary["K"].eq(PRIMARY_K)]

    figure, axes = plt.subplots(2, 1, figsize=(6.8, 6.8), gridspec_kw={"hspace": 0.36})
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    for row_index, bag in enumerate(("structural", "functional")):
        axis = axes[row_index]
        axis.axhline(0.0, color="#444444", linewidth=0.9, zorder=1)
        rows = []

        for position, (arm, rung) in enumerate(COLUMNS):
            cell = primary[
                primary["bag"].eq(bag) & primary["arm"].eq(arm) & primary["rung"].eq(rung)
            ]
            positive = positivity[
                positivity["bag"].eq(bag) & positivity["arm"].eq(arm) & positivity["rung"].eq(rung)
            ]
            if cell.empty:
                continue
            cell, positive = cell.iloc[0], positive.iloc[0]
            colour = ARM_COLOR[arm]

            values = countries[
                countries["bag"].eq(bag) & countries["arm"].eq(arm) & countries["rung"].eq(rung)
            ]["mean_frontier_delta"]
            jitter = np.random.default_rng(position).normal(scale=0.045, size=len(values))
            axis.scatter(
                position + jitter, values, s=11, color=colour, alpha=0.22, linewidths=0, zorder=2
            )

            mean = float(cell["mean_country_frontier_delta"])
            axis.errorbar(
                position,
                mean,
                yerr=[[mean - cell["ci_lo"]], [cell["ci_hi"] - mean]],
                fmt="o",
                color=colour,
                markersize=7,
                capsize=4,
                linewidth=1.8,
                zorder=3,
            )
            stars = _stars(float(cell["holm_p"]))
            if stars:
                axis.text(
                    position, cell["ci_hi"] + 0.002, stars, ha="center", va="bottom",
                    fontsize=11, color=colour,
                )
            rows.append(
                {
                    "bag": bag,
                    "arm": arm,
                    "rung": rung,
                    "K": PRIMARY_K,
                    "mean_frontier_delta": mean,
                    "ci_lo": cell["ci_lo"],
                    "ci_hi": cell["ci_hi"],
                    "p_raw": cell["wilcoxon_p_raw"],
                    "p_holm_adjusted": cell["holm_p"],
                    "rank_biserial": cell["rank_biserial"],
                    "n_countries": cell["n_countries"],
                    "median_percent_above_baseline": positive["median_positive_fraction"] * 100,
                    "significance": stars,
                }
            )

        frame = pd.DataFrame(rows)
        drawn.append(frame)

        # The median % of frontier members above baseline, annotated beneath each
        # position so the breadth of the effect is readable next to its size.
        axis.set_xticks(range(len(COLUMNS)))
        axis.set_xticklabels(
            [
                f"{ARM_LABEL[a]}\n{LEVEL_LABEL[r]}\n"
                f"{frame.loc[(frame.arm == a) & (frame.rung == r), 'median_percent_above_baseline'].iloc[0]:.0f}% > base"
                for a, r in COLUMNS
            ],
            fontsize=8.6,
        )
        axis.set_ylabel(f"Mean Top-{PRIMARY_K} frontier ΔR²")
        axis.set_title(
            f"{'ab'[row_index]}. {'Structural' if bag == 'structural' else 'Functional'} BAG",
            loc="left",
            fontsize=11,
            pad=6,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.5, len(COLUMNS) - 0.5)

        panels.append(
            Panel(
                panel_id=f"{'ab'[row_index]}_{bag}_frontier_vs_base",
                frame=frame,
                description=(
                    f"Mean Top-{PRIMARY_K} frontier incremental R² over the covariate baseline, {bag} BAG."
                ),
                columns={
                    "mean_frontier_delta": "Country-level mean of candidate ΔR², averaged over countries",
                    "p_holm_adjusted": "Holm-adjusted within the four primary tests of this BAG",
                    "median_percent_above_baseline": "Median across countries of the % of frontier members above baseline",
                },
                test=(
                    "Two-sided Wilcoxon signed-rank of country-level mean frontier ΔR² against "
                    "zero; 95% CI from a 10,000-draw country bootstrap."
                ),
                notes=(
                    "Faint points are held-out countries. The 20 frontier candidates are averaged "
                    "within a country first and are never treated as independent observations."
                ),
            )
        )

    axes[-1].set_xlabel("Discovery arm and model level")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_frontier_vs_baseline_hpo_k10"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        write_source_data(stem, panels, output_dir, source_paths=[str(tests_path), str(summary_path)])
        pd.concat(drawn, ignore_index=True).to_csv(
            output_dir / "source_data" / f"{stem}_source_data_all_cells.csv", index=False
        )
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/frontier_vs_baseline/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=root / "frontier_vs_baseline_tests.csv")
    parser.add_argument("--summary", type=Path, default=root / "frontier_vs_baseline_country_summary.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary"
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.tests, args.summary, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
