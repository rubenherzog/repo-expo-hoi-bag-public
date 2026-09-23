"""Render the set-size-adjusted no-scanner education figure.

The education comparison panels are deliberately redrawn from the delivered
unsuffixed figure's Source Data.  This makes their candidate values, baseline,
and best-single lines an exact copy of the delivered figure.  Only the d3
entropy scatter is transformed: both variables are residualized on predictor
set size within each O-information arm.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_ROW_ORDER,
    BAG_SHORT,
    LEVEL_LABELS,
    OBJECTIVE_ARM_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    RUNG_ORDER,
    SINGLE_COLOR,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from scripts.run_education_scanner_baseline_sensitivity import (
    TOP_K,
    _draw_best_single_lines,
    _draw_split_boxes,
)


_ORIGINAL_STEM = "education_scanner_baseline_no_scanner"
_PARTIAL_STEM = f"{_ORIGINAL_STEM}_partial"


def _required_directory(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is required")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _source_frame(source_data: Path, panel_id: str) -> pd.DataFrame:
    path = source_data / f"{_ORIGINAL_STEM}_source_data_{panel_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing delivered Source Data: {path}")
    return pd.read_csv(path)


def _predictor_count(identity: object) -> int:
    """Recover set size from the exact plotted predictor identity."""
    return len([feature for feature in str(identity).split("|") if feature.strip()])


def _residualize_scatter(scatter: pd.DataFrame) -> pd.DataFrame:
    out = scatter.copy()
    required = {"objective", "shannon_h", "global_oof_r2", "predictors_identity"}
    if missing := required.difference(out.columns):
        raise ValueError(f"Entropy Source Data is missing columns: {sorted(missing)}")
    out["order"] = out["predictors_identity"].map(_predictor_count)
    out["entropy_residual"] = np.nan
    out["r2_residual"] = np.nan
    for _objective, points in out.groupby("objective", observed=True):
        valid = points[["shannon_h", "global_oof_r2", "order"]].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 4 or valid["order"].nunique() < 2:
            continue
        design = np.column_stack([np.ones(len(valid)), valid["order"].to_numpy(dtype=float)])
        entropy = valid["shannon_h"].to_numpy(dtype=float)
        r2 = valid["global_oof_r2"].to_numpy(dtype=float)
        out.loc[valid.index, "entropy_residual"] = entropy - design @ np.linalg.lstsq(
            design, entropy, rcond=None
        )[0]
        out.loc[valid.index, "r2_residual"] = r2 - design @ np.linalg.lstsq(
            design, r2, rcond=None
        )[0]
    return out


def _best_single_lines(left: pd.DataFrame) -> pd.DataFrame:
    required = {"rung_id", "best_single_r2"}
    if missing := required.difference(left.columns):
        raise ValueError(f"Education Source Data is missing columns: {sorted(missing)}")
    return (
        left.dropna(subset=["best_single_r2"])[["rung_id", "best_single_r2"]]
        .drop_duplicates("rung_id")
        .rename(columns={"best_single_r2": "global_oof_r2"})
    )


def main() -> None:
    figure_root = _required_directory("EDUCATION_NO_SCANNER_FIGURE_ROOT")
    source_data = _required_directory("EDUCATION_NO_SCANNER_SOURCE_DATA_ROOT")
    rungs = list(RUNG_ORDER)
    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 8.8), squeeze=False,
        gridspec_kw={"hspace": 0.34, "wspace": 0.32},
    )
    panels: list[Panel] = []

    for row_i, bag in enumerate(BAG_ROW_ORDER):
        prefix = "a" if bag == "structural" else "b"
        left_id, scatter_id = f"{prefix}1_{BAG_SHORT[bag]}_edu", f"{prefix}2_{BAG_SHORT[bag]}_entropy"
        # These are the exact rows drawn by the original figure, not a fresh
        # extraction from the sensitivity summaries.
        left = _source_frame(source_data, left_id)
        scatter = _residualize_scatter(_source_frame(source_data, scatter_id))
        ax_left, ax_right = axes[row_i]

        left_values = _draw_split_boxes(ax_left, left, rungs)
        left_values.extend(_draw_best_single_lines(ax_left, _best_single_lines(left), rungs))
        if left_values:
            lo, hi = min(left_values), max(left_values)
            pad = (hi - lo) * 0.10 if hi > lo else 0.02
            ax_left.set_ylim(lo - pad, hi + pad)
        ax_left.set_title(
            f"{ROW_LETTERS[row_i]}. {BAG_LABELS[bag]} — + Education", fontsize=10, loc="left"
        )
        ax_left.set_ylabel("R² LOCO")
        style_axis(ax_left)

        for objective, color in (("o_min", SYN_COLOR), ("o_max", RED_COLOR)):
            points = scatter[scatter["objective"] == objective].dropna(
                subset=["entropy_residual", "r2_residual"]
            )
            ax_right.scatter(
                points["entropy_residual"], points["r2_residual"], s=20,
                color=color, alpha=0.20, linewidths=0, zorder=2,
            )
            if len(points) > 2 and points["entropy_residual"].nunique() > 1:
                fit = scipy_stats.linregress(points["entropy_residual"], points["r2_residual"])
                xs = np.linspace(points["entropy_residual"].min(), points["entropy_residual"].max(), 100)
                ax_right.plot(xs, fit.intercept + fit.slope * xs, color=color, lw=2.2, zorder=3)
        ax_right.set_title("Partial R² LOCO vs entropy (d3)", fontsize=10)
        ax_right.set_xlabel("Entropy residual (adjusted for set size)")
        ax_right.set_ylabel("R² LOCO residual\n(adjusted for set size)")
        style_axis(ax_right)

        panels.extend([
            Panel(
                panel_id=left_id,
                frame=left,
                description=(
                    f"Exact copy of the delivered {BAG_LABELS[bag]} education-adjusted top-{TOP_K} "
                    "comparison panel, including its baseline and best-single values."
                ),
                columns={
                    "rung_id": "model level", "objective": "o_min = Min O-info; o_max = Max O-info",
                    "full_r2": "country-balanced LOCO R² for plotted top-20 point",
                    "base_r2": "education-adjusted covariate baseline R² (black line)",
                    "best_single_r2": "best education-adjusted single-exposure LOCO R² (orange line)",
                },
                notes="Rows are copied verbatim from the unsuffixed figure Source Data.",
            ),
            Panel(
                panel_id=scatter_id,
                frame=scatter,
                description=(
                    f"{BAG_LABELS[bag]} d3 candidates: partial LOCO R²–entropy scatter after "
                    "residualizing both variables by set size within O-information arm."
                ),
                columns={
                    "shannon_h": "ordinary Shannon entropy of predictor domains (bits)",
                    "global_oof_r2": "LOCO R²", "objective": "Min O-info or Max O-info",
                    "order": "set size recovered from predictors_identity",
                    "entropy_residual": "entropy residual after linear adjustment for set size -- plotted x axis",
                    "r2_residual": "LOCO R² residual after linear adjustment for set size -- plotted y axis",
                },
                notes="Input rows are copied from the unsuffixed figure Source Data; no single-exposure point is plotted.",
            ),
        ])

    legend = [
        mpatches.Patch(color=SYN_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_min']}"),
        mpatches.Patch(color=RED_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_max']}"),
        mlines.Line2D([0], [0], color="black", lw=2.5, label="Baseline + Education"),
        mlines.Line2D([0], [0], color=SINGLE_COLOR, lw=3.0, label="Best single + Education"),
    ]
    axes[0][0].legend(handles=legend, fontsize=7, loc="best")
    save_figure(fig, _PARTIAL_STEM, figure_root)
    plt.close(fig)
    write_source_data(_PARTIAL_STEM, panels, figure_root, source_paths=[str(source_data)])


if __name__ == "__main__":
    main()
