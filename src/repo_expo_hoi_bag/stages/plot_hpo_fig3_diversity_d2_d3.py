#!/usr/bin/env python3
"""Render the diversity-only d2+d3 Figure 3 companion and partial correlations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]


def _residual(values: pd.Series, order: pd.Series) -> np.ndarray:
    design = np.column_stack([np.ones(len(order)), order.to_numpy(float)])
    return values.to_numpy(float) - design @ np.linalg.lstsq(design, values.to_numpy(float), rcond=None)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--hpo-set", choices=("k10", "k63"), required=True)
    args = parser.parse_args(); run = args.repro_data_root.resolve() / "results/analysis_runs" / f"paper_reanalysis_{args.hpo_set}" / "main_statistics"
    parts = []
    for rung in ("d2", "d3"):
        frame = pd.read_csv(run / f"fig3_diversity_{rung}/per_candidate_diversity_scatter.csv")
        frame["model_level"] = rung; parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    rows = []
    for level, frame in [("d2", data[data.model_level.eq("d2")]), ("d3", data[data.model_level.eq("d3")]), ("d2_d3", data)]:
        for (bag, objective), group in frame.groupby(["bag", "objective"], observed=True):
            r, p = stats.pearsonr(_residual(group.shannon_h, group.order), _residual(group.country_balanced_r2, group.order))
            rows.append({"hpo_set": args.hpo_set, "model_level": level, "bag": bag, "objective": objective, "n": len(group), "partial_pearson_r": r, "p_two_sided": p, "control": "set size (order), linear residualization"})
    partial = pd.DataFrame(rows); partial.to_csv(run / "fig3_diversity_d2_d3_partial_correlations.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9), sharex=True)
    colors = {"o_min": "#1B6B2E", "o_max": "#4B0082"}; markers = {"d2": "o", "d3": "^"}
    for ax, bag in zip(axes, ("structural", "functional")):
        for (objective, level), group in data[data.bag.eq(bag)].groupby(["objective", "model_level"], observed=True):
            ax.scatter(group.shannon_h, group.country_balanced_r2, c=colors[objective], marker=markers[level], alpha=.35, s=13, label=f"{'Min' if objective=='o_min' else 'Max'} O-info {level}")
        ax.set_ylabel(f"{bag.title()} BAG\nLOCO R²"); ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Shannon domain entropy H (bits)"); fig.tight_layout()
    out = ROOT / "outputs/figures/HPO" / args.hpo_set / "paper/complete"; out.mkdir(parents=True, exist_ok=True); stem=f"fig3_diversity_only_max_30_hpo_{args.hpo_set}_d2_d3"
    for extension in ("png", "pdf", "svg"): fig.savefig(out / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig); print(partial.to_string(index=False)); print(f"Saved: {out / (stem + '.png')}")


if __name__ == "__main__": main()
