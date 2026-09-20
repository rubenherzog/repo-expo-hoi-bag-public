#!/usr/bin/env python3
"""Render the parent-set O-information versus LOFO-fragility scatter.

This is a read-only visualisation of the completed feature-ablation analysis;
it never evaluates or refits a model.  Each point represents one selected
parent set, with the largest observed leave-one-exposure-out R² loss on the
y-axis and the parent set's O-information on the x-axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402
from repo_expo_hoi_bag.figures.style import (  # noqa: E402
    BAG_LABELS,
    BAG_ROW_ORDER,
    LEVEL_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from scripts.sensitivity_common import (  # noqa: E402
    active_rungs,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)


STEM = "feature_ablation_oinfo_max_loss"
ARM_LABELS = {"o_min": "Synergy arm", "o_max": "Redundancy arm"}
ARM_COLORS = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
MIN_MARKER_AREA = 24.0
MAX_MARKER_AREA = 132.0


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_parent_scatter_frame(selection: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Join saved parent metrics into the exact one-row-per-point plotting table."""
    _require_columns(
        selection,
        {"bag_target", "rung_id", "objective", "candidate_id", "thoi_o", "full_r2", "order", "rank_within_arm"},
        label="Feature-ablation parent selection",
    )
    _require_columns(
        summaries,
        {"bag", "rung_id", "arm", "parent_candidate_id", "parent_order", "max_r2_loss", "mean_r2_loss"},
        label="Feature-ablation parent summary",
    )
    parent = selection.rename(
        columns={
            "bag_target": "bag",
            "objective": "arm",
            "candidate_id": "parent_candidate_id",
            "order": "selection_order",
            "full_r2": "parent_full_r2",
        }
    ).loc[
        :,
        ["bag", "rung_id", "arm", "parent_candidate_id", "rank_within_arm", "thoi_o", "parent_full_r2", "selection_order"],
    ]
    summary = summaries.loc[
        :,
        ["bag", "rung_id", "arm", "parent_candidate_id", "parent_order", "max_r2_loss", "mean_r2_loss"],
    ]
    frame = parent.merge(
        summary,
        on=["bag", "rung_id", "arm", "parent_candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(parent):
        raise ValueError("Parent scatter merge lost selected feature-ablation parents")
    if not np.array_equal(frame["selection_order"].to_numpy(), frame["parent_order"].to_numpy()):
        raise ValueError("Parent set order differs between selection and ablation summary")
    numeric = ["thoi_o", "parent_full_r2", "parent_order", "max_r2_loss", "mean_r2_loss"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame[numeric].isna().any().any():
        raise ValueError("Parent scatter requires finite O-information, order, and R² metrics")
    if (frame["parent_order"] < 2).any():
        raise ValueError("Parent scatter requires sets with at least two exposures")
    frame["arm_label"] = frame["arm"].map(ARM_LABELS)
    if frame["arm_label"].isna().any():
        unknown = sorted(frame.loc[frame["arm_label"].isna(), "arm"].astype(str).unique())
        raise ValueError(f"Unexpected feature-ablation arm labels: {unknown}")
    return frame.sort_values(["bag", "rung_id", "arm", "rank_within_arm", "parent_candidate_id"], kind="stable").reset_index(drop=True)


def _marker_areas(orders: pd.Series) -> np.ndarray:
    values = orders.to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    if low == high:
        return np.full(len(values), (MIN_MARKER_AREA + MAX_MARKER_AREA) / 2.0)
    return MIN_MARKER_AREA + (values - low) * (MAX_MARKER_AREA - MIN_MARKER_AREA) / (high - low)


def _bag_y_limits(frame: pd.DataFrame, bags: list[str]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for bag in bags:
        values = frame.loc[frame["bag"] == bag, "max_r2_loss"]
        lower = min(float(values.min()), 0.0)
        upper = max(float(values.max()), 0.01)
        limits[bag] = (lower - 0.06 * upper, 1.08 * upper)
    return limits


def render_parent_scatter(frame: pd.DataFrame, *, bags: list[str], rungs: list[str], outdir: Path) -> None:
    """Render the 2 × 4 paper-style scatter using only saved parent tables."""
    present_bags = [bag for bag in BAG_ROW_ORDER if bag in bags]
    if not present_bags or not rungs:
        raise ValueError("Parent scatter has no configured BAGs or model levels to render")
    frame = frame.copy()
    frame["marker_area"] = _marker_areas(frame["parent_order"])
    y_limits = _bag_y_limits(frame, present_bags)
    figure, axes = plt.subplots(
        len(present_bags),
        len(rungs),
        figsize=(4.5 * len(rungs), 4.6 * len(present_bags)),
        squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.34},
    )
    panels: list[Panel] = []
    for row_index, bag in enumerate(present_bags):
        for column_index, rung in enumerate(rungs):
            axis = axes[row_index, column_index]
            cell = frame.loc[(frame["bag"] == bag) & (frame["rung_id"] == rung)].copy()
            if cell.empty:
                raise ValueError(f"Parent scatter has no points for bag={bag}, rung={rung}")
            for arm in ("o_min", "o_max"):
                points = cell.loc[cell["arm"] == arm]
                axis.scatter(
                    points["thoi_o"],
                    points["max_r2_loss"],
                    s=points["marker_area"],
                    color=ARM_COLORS[arm],
                    alpha=0.68,
                    linewidths=0,
                    zorder=3,
                )
            axis.axhline(0, color="#888888", lw=0.8, zorder=0)
            axis.set_ylim(*y_limits[bag])
            axis.margins(x=0.07)
            axis.set_xlabel("O-information of parent set" if row_index == len(present_bags) - 1 else "")
            if row_index == 0 and column_index == 0:
                axis.set_ylabel("Maximum ΔR² after LOFO refit")
                axis.set_title(
                    f"{ROW_LETTERS[row_index]}. {BAG_LABELS[bag]} — {LEVEL_LABELS[rung]}",
                    loc="left",
                    fontsize=13,
                )
            elif row_index == 0:
                axis.set_title(LEVEL_LABELS[rung], fontsize=13)
            elif column_index == 0:
                axis.set_ylabel("Maximum ΔR² after LOFO refit")
                axis.set_title(f"{ROW_LETTERS[row_index]}. {BAG_LABELS[bag]}", loc="left", fontsize=13)
            style_axis(axis)
            cell["model_level"] = LEVEL_LABELS[rung]
            panels.append(
                Panel(
                    panel_id=f"{ROW_LETTERS[row_index]}_{bag[:6]}_{LEVEL_LABELS[rung]}",
                    frame=cell[
                        [
                            "model_level",
                            "arm",
                            "arm_label",
                            "rank_within_arm",
                            "parent_candidate_id",
                            "thoi_o",
                            "max_r2_loss",
                            "mean_r2_loss",
                            "parent_order",
                            "marker_area",
                            "parent_full_r2",
                        ]
                    ],
                    description=(
                        f"Maximum leave-one-exposure-out R² loss versus O-information for top-20 "
                        f"{ARM_LABELS['o_min'].lower()} and {ARM_LABELS['o_max'].lower()} parent sets in "
                        f"the {BAG_LABELS[bag]} at {LEVEL_LABELS[rung]}."
                    ),
                    columns={
                        "thoi_o": "O-information of the complete parent exposure set",
                        "max_r2_loss": "largest parent_full_r2 minus ablated_full_r2 across leave-one-exposure-out refits",
                        "parent_order": "number of exposures in the complete parent set; encoded by point area",
                        "marker_area": "matplotlib point area in points squared used to render parent_order",
                        "parent_full_r2": "complete parent model held-out LOCO R²",
                    },
                    notes="Each point is one parent set; point area encodes the original set size.",
                )
            )
    order_min, order_max = int(frame["parent_order"].min()), int(frame["parent_order"].max())
    color_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SYN_COLOR, label="Synergy arm", markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=RED_COLOR, label="Redundancy arm", markersize=6),
    ]
    size_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#777777", alpha=0.68, label=f"Set size: {order_min}", markersize=np.sqrt(MIN_MARKER_AREA)),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#777777", alpha=0.68, label=f"Set size: {order_max}", markersize=np.sqrt(MAX_MARKER_AREA)),
    ]
    axes[0, -1].legend(handles=[*color_handles, *size_handles], frameon=False, fontsize=8.5, loc="upper right")
    save_figure(figure, STEM, outdir)
    plt.close(figure)
    write_source_data(
        STEM,
        panels,
        outdir,
        source_paths=["feature_ablation_parent_selection.csv", "feature_ablation_parent_summary.csv"],
    )


def main() -> None:
    cfg = load_sensitivity_config()
    summary_root = repo_sensitivity_root(cfg) / "feature_ablation"
    selection_path = summary_root / "feature_ablation_parent_selection.csv"
    summary_path = summary_root / "feature_ablation_parent_summary.csv"
    missing = [str(path) for path in (selection_path, summary_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("O-information scatter requires saved feature-ablation tables: " + ", ".join(missing))
    frame = build_parent_scatter_frame(pd.read_csv(selection_path), pd.read_csv(summary_path))
    bags = [bag for bag in selected_bags(cfg, include_combined=False) if bag in set(frame["bag"])]
    rungs = [rung for rung in active_rungs(cfg) if rung in set(frame["rung_id"])]
    render_parent_scatter(frame, bags=bags, rungs=rungs, outdir=summary_root / "figures")


if __name__ == "__main__":
    main()
