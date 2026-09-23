#!/usr/bin/env python3
"""Compact supplementary figure: cross-selected exposome model vs baseline.

Two panels, structural above functional, each showing the four primary d2/d3
comparisons as the observed country-balanced ΔR² with its 95% paired
country-cluster bootstrap interval. Individual country ΔR² values sit lightly
in the background so the heterogeneity behind each estimate stays visible.
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
from repo_expo_hoi_bag.stages.compute_cross_selected_baseline import PRIMARY_RUNGS, ROOT

ARM_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
LEVEL_LABEL = {"xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
# x order: synergy d2, redundancy d2, synergy d3, redundancy d3.
COLUMNS = (("o_min", "xgb_tree_d2"), ("o_max", "xgb_tree_d2"), ("o_min", "xgb_tree_d3"), ("o_max", "xgb_tree_d3"))


def _stars(value: float) -> str:
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def build(
    summary_path: Path, by_country_path: Path, output_dir: Path, *, skip_source_data: bool = False
) -> Path:
    summary = pd.read_csv(summary_path)
    by_country = pd.read_csv(by_country_path)
    summary = summary[summary["rung"].isin(PRIMARY_RUNGS)]

    figure, axes = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.22})
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    for row_index, bag in enumerate(("structural", "functional")):
        axis = axes[row_index]
        axis.axhline(0.0, color="#444444", linewidth=0.9, zorder=1)

        rows = []
        for position, (arm, rung) in enumerate(COLUMNS):
            cell = summary[summary["bag"].eq(bag) & summary["arm"].eq(arm) & summary["rung"].eq(rung)]
            if cell.empty:
                continue
            cell = cell.iloc[0]
            colour = ARM_COLOR[arm]

            countries = by_country[
                by_country["bag"].eq(bag) & by_country["arm"].eq(arm) & by_country["rung"].eq(rung)
            ]
            jitter = np.random.default_rng(position).normal(scale=0.045, size=len(countries))
            axis.scatter(
                position + jitter,
                countries["delta_r2_country"],
                s=11,
                color=colour,
                alpha=0.22,
                linewidths=0,
                zorder=2,
            )

            axis.errorbar(
                position,
                cell["delta_r2"],
                yerr=[[cell["delta_r2"] - cell["bootstrap_ci_lo"]], [cell["bootstrap_ci_hi"] - cell["delta_r2"]]],
                fmt="o",
                color=colour,
                markersize=7,
                capsize=4,
                linewidth=1.8,
                zorder=3,
            )
            stars = _stars(float(cell["bootstrap_p_holm"]))
            if stars:
                axis.text(
                    position,
                    cell["bootstrap_ci_hi"] + 0.004,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color=colour,
                )
            rows.append(
                {
                    "bag": bag,
                    "arm": arm,
                    "rung": rung,
                    "delta_r2": cell["delta_r2"],
                    "ci_lo": cell["bootstrap_ci_lo"],
                    "ci_hi": cell["bootstrap_ci_hi"],
                    "p_raw": cell["bootstrap_p_raw"],
                    "p_holm_adjusted": cell["bootstrap_p_holm"],
                    "cohen_f2": cell["cohen_f2"],
                    "n_countries": cell["n_countries"],
                    "n_countries_delta_positive": cell["n_countries_delta_positive"],
                    "significance": stars,
                }
            )

        frame = pd.DataFrame(rows)
        drawn.append(frame)
        axis.set_xticks(range(len(COLUMNS)))
        axis.set_xticklabels([f"{ARM_LABEL[a]}\n{LEVEL_LABEL[r]}" for a, r in COLUMNS])
        axis.set_ylabel("ΔR² over baseline")
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
                # Kept short: xlsx caps sheet names at 31 characters.
                panel_id=f"{'ab'[row_index]}_{bag}_xsel_vs_base",
                frame=frame,
                description=(
                    f"Country-cross-selected exposome model minus covariate baseline, {bag} BAG."
                ),
                columns={
                    "delta_r2": "Unweighted mean of per-country R² difference",
                    "ci_lo": "2.5th percentile of the paired country-cluster bootstrap",
                    "ci_hi": "97.5th percentile of the paired country-cluster bootstrap",
                    "p_holm_adjusted": "Holm-adjusted within the four primary tests of this BAG",
                },
                test=(
                    "Two-sided paired country-cluster bootstrap, 10,000 draws resampling whole "
                    "countries; Holm across the four primary d2/d3 comparisons within a BAG."
                ),
                notes=(
                    "Faint points are individual held-out countries. The candidate is chosen "
                    "without the evaluation country; it may differ by country, so this is not "
                    "one fixed exposome model."
                ),
            )
        )

    axes[-1].set_xlabel("Discovery arm and model level")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_cross_selected_exposome_vs_baseline_hpo_k10"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        write_source_data(
            stem, panels, output_dir, source_paths=[str(summary_path), str(by_country_path)]
        )
        pd.concat(drawn, ignore_index=True).to_csv(
            output_dir / "source_data" / f"{stem}_source_data_all_cells.csv", index=False
        )
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=root / "cross_selected_baseline_summary.csv")
    parser.add_argument("--by-country", type=Path, default=root / "cross_selected_baseline_by_country.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary"
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.summary, args.by_country, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
