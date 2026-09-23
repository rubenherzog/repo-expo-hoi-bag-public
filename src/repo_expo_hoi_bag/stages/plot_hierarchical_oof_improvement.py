#!/usr/bin/env python3
"""Compact supplementary figure: hierarchical OOF improvement over baseline.

One panel per BAG. Four x positions (synergy/redundancy x d2/d3), each holding
the nested candidate regions Top10, Top20, Top50 and ALL as offset points with
95% confidence intervals. Top20 is the primary region and is drawn filled and
slightly larger; the others are open markers.

Participants are never plotted individually: each point is the intercept of the
random-intercept model for that cell, already collapsed over candidates and
countries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import RED_COLOR, SYN_COLOR
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import (
    PRIMARY_REGION,
    REGION_ORDER,
    ROOT,
)

ARM_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
ARM_LABEL = {"o_min": "Synergy", "o_max": "Redundancy"}
LEVEL_LABEL = {"xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
COLUMNS = (
    ("o_min", "xgb_tree_d2"),
    ("o_max", "xgb_tree_d2"),
    ("o_min", "xgb_tree_d3"),
    ("o_max", "xgb_tree_d3"),
)
# Offsets keep the four nested regions legible inside one x position.
OFFSET = {"Top10": -0.21, "Top20": -0.07, "Top50": 0.07, "ALL": 0.21}
MARKER = {"Top10": "^", "Top20": "o", "Top50": "s", "ALL": "D"}


def _stars(value: float) -> str:
    if not pd.notna(value):
        return ""
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def build(results_path: Path, output_dir: Path, *, skip_source_data: bool = False) -> Path:
    results = pd.read_csv(results_path)
    regions = [name for name in REGION_ORDER if name in set(results["region"])]

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), gridspec_kw={"hspace": 0.34})
    panels: list[Panel] = []
    drawn: list[pd.DataFrame] = []

    for row_index, bag in enumerate(("structural", "functional")):
        axis = axes[row_index]
        axis.axhline(0.0, color="#444444", linewidth=0.9, zorder=1)
        rows = []

        for position, (arm, rung) in enumerate(COLUMNS):
            colour = ARM_COLOR[arm]
            for region in regions:
                cell = results[
                    results["bag"].eq(bag)
                    & results["arm"].eq(arm)
                    & results["rung"].eq(rung)
                    & results["region"].eq(region)
                ]
                if cell.empty:
                    continue
                cell = cell.iloc[0]
                primary = region == PRIMARY_REGION
                x = position + OFFSET[region]
                beta = float(cell["beta_intercept"])
                axis.errorbar(
                    x,
                    beta,
                    yerr=[[beta - cell["ci_lo"]], [cell["ci_hi"] - beta]],
                    fmt=MARKER[region],
                    color=colour,
                    markersize=7.0 if primary else 4.6,
                    markerfacecolor=colour if primary else "white",
                    markeredgewidth=1.2,
                    capsize=3,
                    linewidth=1.7 if primary else 1.0,
                    zorder=4 if primary else 3,
                )
                if primary:
                    stars = _stars(cell["holm_p"])
                    if stars:
                        axis.text(
                            x, cell["ci_hi"], f" {stars}", ha="center", va="bottom",
                            fontsize=10, color=colour,
                        )
                rows.append(
                    {
                        "bag": bag,
                        "arm": arm,
                        "rung": rung,
                        "region": region,
                        "beta_intercept": beta,
                        "ci_lo": cell["ci_lo"],
                        "ci_hi": cell["ci_hi"],
                        "p_two_sided": cell["p_two_sided"],
                        "holm_p": cell["holm_p"],
                        "icc_country": cell["icc_country"],
                        "n_subjects": cell["n_subjects"],
                        "n_countries": cell["n_countries"],
                        "is_primary_region": primary,
                    }
                )

        frame = pd.DataFrame(rows)
        drawn.append(frame)
        axis.set_xticks(range(len(COLUMNS)))
        axis.set_xticklabels(
            [f"{ARM_LABEL[arm]}\n{LEVEL_LABEL[rung]}" for arm, rung in COLUMNS], fontsize=9
        )
        axis.set_ylabel("Mean individual OOF\nloss improvement")
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
                panel_id=f"{'ab'[row_index]}_{bag}_hier_oof",
                frame=frame,
                description=(
                    f"Hierarchical mean individual OOF loss improvement over the covariate "
                    f"baseline, {bag} BAG, by candidate region."
                ),
                columns={
                    "beta_intercept": "Intercept of the country random-intercept REML model",
                    "holm_p": (
                        f"Holm-adjusted within the four primary {PRIMARY_REGION} tests of this "
                        "BAG; blank for the ALL control"
                    ),
                    "icc_country": "Share of outcome variance between countries",
                },
                test=(
                    "REML random-intercept model of per-participant mean improvement with "
                    "country as grouping factor; two-sided Wald P on the intercept."
                ),
                notes=(
                    "beta0 is on the squared-error scale, not the R2 scale. Candidates are "
                    "averaged within each participant before fitting, so subject x candidate "
                    "rows are never independent observations. Top-K regions are conditional on "
                    "the observed candidate landscape."
                ),
            )
        )

    handles = [
        Line2D(
            [], [], color="#444444", linestyle="none",
            marker=MARKER[region], markersize=7.0 if region == PRIMARY_REGION else 4.6,
            markerfacecolor="#444444" if region == PRIMARY_REGION else "white",
            markeredgewidth=1.2,
            label=f"{region} (primary)" if region == PRIMARY_REGION else region,
        )
        for region in regions
    ]
    axes[0].legend(
        handles=handles, frameon=False, fontsize=8.4, ncol=len(handles),
        loc="upper center", bbox_to_anchor=(0.5, 1.30), handletextpad=0.4, columnspacing=1.2,
    )
    axes[-1].set_xlabel("Discovery arm and model level")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_hierarchical_oof_improvement_hpo_k10"
    for extension in ("pdf", "svg", "png", "tiff"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=600 if extension in {"png", "tiff"} else None,
        )
    plt.close(figure)

    if not skip_source_data:
        write_source_data(stem, panels, output_dir, source_paths=[str(results_path)])
        pd.concat(drawn, ignore_index=True).to_csv(
            output_dir / "source_data" / f"{stem}_source_data_all_cells.csv", index=False
        )
    return output_dir / f"{stem}.png"


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "hierarchical_oof_model_results.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/figures/supplementary"
    )
    parser.add_argument("--skip-source-data", action="store_true")
    args = parser.parse_args()
    print(f"Saved: {build(args.results, args.output_dir, skip_source_data=args.skip_source_data)}")


if __name__ == "__main__":
    main()
