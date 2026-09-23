#!/usr/bin/env python3
"""
Figure 4 - Residual bias by country and diagnosis.

Layout: 3 rows (BAG) x 3 columns
  Col 1: paired heatmaps of median signed bias for the best synergistic and redundant models
  Col 2: residual distribution by diagnosis for the best synergistic model
  Col 3: residual distribution by diagnosis for the best redundant model

Outputs:
  paper_figures/fig4_residual_bias.{pdf,svg,png}
  paper_figures/fig4_residual_bias_subject.csv
  paper_figures/fig4_residual_bias_country_dx.csv
  paper_figures/fig4_residual_bias_selection.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
from pathlib import Path as _Path
for _fp in [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/local/share/fonts/Arial.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]:
    if _Path(_fp).exists():
        fm.fontManager.addfont(_fp)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Arimo', 'Liberation Sans', 'DejaVu Sans']

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from scripts.plot_regression_residuals import (
    country_order_from,
    draw_bias_panel,
    pivot_ctry_dx,
)
from scripts.sensitivity_common import (
    load_sensitivity_config,
    paper_analysis_config,
)

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data

# Re-assert after imports in case any imported module mutates rcParams at
# module load time (see plot_fig3_diversity_v2.py for the failure mode).
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Arimo', 'Liberation Sans', 'DejaVu Sans']

REPO_ROOT = Path(__file__).resolve().parents[1]
# Residual stat tables + best_rung_selection are read from STATS_BASE; capped
# variants point this at a cap-specific staging dir via V3_FIG_STATS_DIR.
STATS_BASE = Path(os.environ["V3_FIG_STATS_DIR"]) if os.environ.get("V3_FIG_STATS_DIR") else (REPO_ROOT / "outputs" / "variant_a" / "stats")
RES_DIR = STATS_BASE / "residuals"
PAPER_FIGS = REPO_ROOT / "paper_figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
# Shared paper-figure variant contract.
SUFFIX = os.environ.get("PAPER_FIG_SUFFIX", "")
OUT_DIR = Path(os.environ["PAPER_FIG_OUTDIR"]) if os.environ.get("PAPER_FIG_OUTDIR") else PAPER_FIGS

_SENSITIVITY_CFG = load_sensitivity_config()
_PAPER_CFG = paper_analysis_config(_SENSITIVITY_CFG)
_RESIDUAL_CFG = _PAPER_CFG.residual_bias
_BAGS_ENV = [
    value.strip()
    for value in os.environ.get(
        "PAPER_FIG_BAGS", ",".join(_PAPER_CFG.primary_bags)
    ).split(",")
    if value.strip()
]
BAGS = [bag for bag in _PAPER_CFG.primary_bags if bag in _BAGS_ENV]
_OBJECTIVE_PRESENTATION = _PAPER_CFG.objective_metadata
OBJECTIVES = [
    (
        objective,
        _OBJECTIVE_PRESENTATION[objective]["file_suffix"],
        _OBJECTIVE_PRESENTATION[objective]["arm_label"],
    )
    for objective in _PAPER_CFG.objectives
]
OBJECTIVE_BY_SUFFIX = {
    values["file_suffix"]: objective
    for objective, values in _OBJECTIVE_PRESENTATION.items()
}
SYNERGY_OBJECTIVE = OBJECTIVE_BY_SUFFIX["syn"]
REDUNDANCY_OBJECTIVE = OBJECTIVE_BY_SUFFIX["red"]
BAG_LABELS = {
    "functional": "Functional BAG",
    "structural": "Structural BAG",
    "combined": "Combined BAG",
}

DX_ORDER = list(_PAPER_CFG.primary_diagnoses)
DX_LABELS = dict(_PAPER_CFG.diagnosis_labels)
SYN_BORDER = "#1B5E20"
RED_BORDER = "#4B0082"
OBJ_COLORS = {
    SYNERGY_OBJECTIVE: SYN_BORDER,
    REDUNDANCY_OBJECTIVE: RED_BORDER,
}
DX_LINESTYLES = {
    "CN": ("-", 1.0),
    "AD": ((0, (4.0, 2.5)), 0.95),
    "FTD": ((0, (8.0, 2.5)), 0.95),
}

FS = 19
FS_TK = 16


def _subject_path(bag: str, suffix: str) -> Path:
    return RES_DIR / f"residuals_subject_{bag}_{suffix}.csv"


def _country_path(bag: str, objective: str) -> Path:
    return RES_DIR / f"residuals_ctry_dx_{bag}_{objective}.csv"


def _selection_table() -> pd.DataFrame:
    p = STATS_BASE / "best_rung_selection.csv"
    return pd.read_csv(p)


def _load_subject_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = _selection_table()
    subject_rows = []
    country_rows = []
    selection_rows = []

    for bag in BAGS:
        for objective, suffix, label in OBJECTIVES:
            subj = pd.read_csv(_subject_path(bag, suffix))
            subj = subj.copy()
            subj["bag"] = bag
            subj["objective"] = objective
            subj["objective_label"] = label
            subj["diagnosis_label"] = subj["diagnosis"].map(DX_LABELS)
            row = selection[
                (selection["analysis"] == _RESIDUAL_CFG["best_rung_analysis"])
                & (selection["bag"] == bag)
                & (selection["objective"] == objective)
            ]
            best_rung = str(row["best_rung"].iloc[0]) if not row.empty else ""
            if "candidate_id" not in subj.columns:
                subj["candidate_id"] = ""
            subj["best_rung"] = best_rung
            cols = [
                "bag",
                "objective",
                "objective_label",
                "country",
                "diagnosis",
                "diagnosis_label",
                "age",
                "sex",
                "y_true",
                "y_pred_full",
                "residual",
                "abs_residual",
            ]
            cols.extend(["candidate_id", "best_rung"])
            subject_rows.append(
                subj[cols]
            )

            cdx = (
                subj.groupby(["country", "diagnosis"], observed=True)
                .agg(
                    median_residual=("residual", "median"),
                    n_subjects=("residual", "size"),
                )
                .reset_index()
            )
            cdx["abs_median_residual"] = cdx["median_residual"].abs()
            cdx["bag"] = bag
            cdx["objective"] = objective
            cdx["objective_label"] = label
            cdx["diagnosis_label"] = cdx["diagnosis"].map(DX_LABELS)
            country_rows.append(cdx)

            selection_rows.append(
                {
                    "bag": bag,
                    "objective": objective,
                    "objective_label": label,
                    "best_rung": best_rung,
                    "candidate_id": str(subj["candidate_id"].dropna().iloc[0]) if "candidate_id" in subj.columns and subj["candidate_id"].notna().any() else "",
                    "source_file": str(_subject_path(bag, suffix)),
                }
            )

    subject_df = pd.concat(subject_rows, ignore_index=True)
    country_df = pd.concat(country_rows, ignore_index=True)
    selection_df = pd.DataFrame(selection_rows)
    return subject_df, country_df, selection_df


def _draw_distribution_panel(
    ax: plt.Axes,
    subj: pd.DataFrame,
    *,
    show_legend: bool,
    x_lo: float,
    x_hi: float,
    y_hi: float,
    objective: str,
) -> None:
    xs = np.linspace(x_lo, x_hi, 240)
    for dx in DX_ORDER:
        sub = subj[(subj["diagnosis"] == dx) & (subj["objective"] == objective)]["residual"].dropna().values
        if len(sub) < 2:
            continue
        kde = gaussian_kde(sub, bw_method=0.30)
        linestyle, alpha = DX_LINESTYLES[dx]
        ax.plot(
            xs,
            kde(xs),
            color=OBJ_COLORS[objective],
            linewidth=3.9,
            linestyle=linestyle,
            alpha=alpha,
            label=DX_LABELS[dx],
        )
    ax.axvline(0, color="black", linewidth=0.9, linestyle="-", alpha=0.55)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_hi)
    ax.set_ylabel("Density", fontsize=FS)
    ax.tick_params(axis="x", labelsize=FS_TK + 2)
    ax.tick_params(axis="y", labelsize=FS_TK + 2)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        from matplotlib.lines import Line2D
        dx_handles = [
            Line2D([], [], color="black", linestyle=DX_LINESTYLES["CN"][0], linewidth=1.8, label="HC"),
            Line2D([], [], color="black", linestyle=DX_LINESTYLES["AD"][0], linewidth=1.8, label="AD"),
            Line2D([], [], color="black", linestyle=DX_LINESTYLES["FTD"][0], linewidth=1.8, label="FTLD"),
        ]
        ax.legend(handles=dx_handles, fontsize=FS_TK - 1, framealpha=0.85, loc="upper left")


def _draw_heatmap_panel(
    ax: plt.Axes,
    bag_df: pd.DataFrame,
    *,
    ctry_order: list[str],
    abs_max: float,
    objective: str,
    show_row_labels: bool,
    title: str | None = None,
):
    plot_df = bag_df.copy()
    plot_df["diagnosis"] = plot_df["diagnosis"].map(DX_LABELS)
    present_ctry = [c for c in ctry_order if c in plot_df["country"].unique()]
    present_dx = [DX_LABELS[d] for d in DX_ORDER if d in bag_df["diagnosis"].unique()]
    b_vals, n_vals = pivot_ctry_dx(plot_df, "median_residual", present_ctry, present_dx)
    im = draw_bias_panel(
        ax,
        b_vals,
        n_vals,
        row_labels=present_ctry,
        col_labels=present_dx,
        vmin=-abs_max,
        vmax=abs_max,
        cmap=plt.cm.RdBu_r,
        show_row_labels=show_row_labels,
        annotate_threshold=10,
        annotate_fontsize=13.5,
    )
    border_color = (
        SYN_BORDER if objective == SYNERGY_OBJECTIVE else RED_BORDER
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3.4)
    ax.set_xlabel("")
    if not show_row_labels:
        ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title, fontsize=FS, pad=7)
    return im


_BAG_SHORT = {"structural": "struct", "functional": "func", "combined": "comb"}
_ROW_LETTERS = ("a", "b", "c")


def _build_fig4_panels(
    subject_df: pd.DataFrame,
    country_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    ctry_order: list[str],
    abs_max: float,
) -> list[Panel]:
    """One heatmap panel and one density panel per BAG.

    The heatmap frame holds the median bias actually shown in each cell (plus the
    n behind it, which drives the in-cell annotation); the density frame holds
    the per-subject residuals the kernel density estimate is drawn from, since a
    KDE curve is not itself a data series a reader could check.
    """
    panels: list[Panel] = []
    obj_gloss = (
        "o_min = best synergistic (min O-information) model; "
        "o_max = best redundant (max O-information) model"
    )
    for letter, bag in zip(_ROW_LETTERS, BAGS):
        short = _BAG_SHORT.get(bag, bag)

        heat = country_df[country_df["bag"] == bag].copy()
        heat = heat[heat["country"].isin(ctry_order)]
        heat["country"] = pd.Categorical(heat["country"], categories=ctry_order, ordered=True)
        heat = heat.sort_values(["objective", "country", "diagnosis"])
        panels.append(Panel(
            panel_id=f"{letter}1_{short}_bias_heatmap",
            frame=heat[[
                "objective", "objective_label", "country", "diagnosis_label",
                "median_residual", "n_subjects",
            ]].reset_index(drop=True),
            description=(
                f"{BAG_LABELS[bag]} bias stratified by country and diagnosis, for the best "
                "synergistic and best redundant model."
            ),
            columns={
                "objective": obj_gloss,
                "objective_label": "arm label as printed above the heatmap",
                "country": "country of the held-out LOCO fold; rows are ordered by absolute median bias",
                "diagnosis_label": "diagnosis group as printed on the x axis",
                "median_residual": "median BAG bias (predicted - observed, years) in that cell -- the plotted colour",
                "n_subjects": "participants in that cell; cells with n >= 10 carry the printed annotation",
            },
            notes=(
                f"Colour scale is symmetric and clipped at +/-{abs_max:.1f} years, so cells beyond "
                "that saturate; the tabulated median is the unclipped value. Country order is shared "
                "across all rows and taken from the configured reference BAG and arm."
            ),
        ))

        dens = subject_df[subject_df["bag"] == bag]
        panels.append(Panel(
            panel_id=f"{letter}2_{short}_bias_density",
            frame=dens[[
                "objective", "objective_label", "diagnosis_label", "country", "residual",
            ]].reset_index(drop=True),
            description=(
                f"{BAG_LABELS[bag]} bias pooled by diagnosis: the per-participant residuals "
                "underlying each plotted density curve."
            ),
            columns={
                "objective": obj_gloss,
                "objective_label": "arm label, distinguishing the two curve colours",
                "diagnosis_label": "diagnosis group, distinguishing the three line styles",
                "country": "country of the held-out LOCO fold",
                "residual": "BAG bias (predicted - observed, years) for that participant",
            },
            notes=(
                "One row per participant. The figure draws a Gaussian kernel density estimate "
                "(bw_method=0.30) over these values, one curve per diagnosis and arm; the curve "
                "itself is derived, so the underlying residuals are given instead."
            ),
        ))

    panels.append(Panel(
        panel_id="model_selection",
        frame=selection_df,
        description="The model whose residuals each panel shows, per BAG and arm.",
        columns={
            "bag": "brain-age-gap modality",
            "objective": obj_gloss,
            "best_rung": "model level selected for that BAG and arm",
            "candidate_id": "identifier of the selected exposure set",
            "source_file": "residual table the panel was built from",
        },
        notes="Provenance for the panels above; nothing in this sheet is itself plotted.",
    ))
    return panels


def main() -> None:
    subject_df, country_df, selection_df = _load_subject_frames()

    if not SUFFIX:
        subject_df.to_csv(PAPER_FIGS / "fig4_residual_bias_subject.csv", index=False)
        country_df.to_csv(PAPER_FIGS / "fig4_residual_bias_country_dx.csv", index=False)
        selection_df.to_csv(PAPER_FIGS / "fig4_residual_bias_selection.csv", index=False)

    reference_country_order = country_df[
        (country_df["bag"] == _RESIDUAL_CFG["country_order_reference_bag"])
        & (
            country_df["objective"]
            == _RESIDUAL_CFG["country_order_reference_objective"]
        )
    ]
    if reference_country_order.empty:
        raise ValueError("Configured Fig. 4 country-order reference has no rows")
    ctry_order = country_order_from(
        reference_country_order, metric="abs_median_residual"
    )

    all_residuals = subject_df["residual"].dropna().values
    x_lo = float(np.nanpercentile(all_residuals, 0.5))
    x_hi = float(np.nanpercentile(all_residuals, 99.5))
    xs_global = np.linspace(x_lo, x_hi, 240)

    heatmap_vals = country_df["median_residual"].dropna().values
    # abs_max = max(float(np.nanpercentile(np.abs(heatmap_vals), 95)), 0.5)
    abs_max = 5.0


    y_hi = 0.0
    for bag in BAGS:
        for objective, _, _ in OBJECTIVES:
            subj = subject_df[(subject_df["bag"] == bag) & (subject_df["objective"] == objective)]
            for dx in DX_ORDER:
                sub = subj[subj["diagnosis"] == dx]["residual"].dropna().values
                if len(sub) < 2:
                    continue
                kde = gaussian_kde(sub, bw_method=0.30)
                y_hi = max(y_hi, float(np.nanmax(kde(xs_global))))
    y_hi *= 1.04

    row_heights = []
    for bag in BAGS:
        bag_df = country_df[country_df["bag"] == bag]
        present = [c for c in ctry_order if c in bag_df["country"].unique()]
        row_heights.append(len(present))
    max_ctry = max(row_heights)
    fig_h = max(12.7, 0.48 * max_ctry + 4.5) * 0.9

    fig = plt.figure(figsize=(23.5, fig_h))
    gs = gridspec.GridSpec(
        len(BAGS),
        2,
        figure=fig,
        left=0.11,
        right=0.99,
        top=0.92,
        bottom=0.07,
        wspace=0.21,
        hspace=0.11,
        width_ratios=[1.15, 1.12],
    )

    heat_axes = []
    distribution_axes = []
    for ri, bag in enumerate(BAGS):
        heat_spec = gs[ri, 0].subgridspec(1, 2, wspace=0.03, width_ratios=[1.0, 1.0])
        distribution_spec = gs[ri, 1].subgridspec(1, 2, wspace=0.12)
        ax_h1 = fig.add_subplot(heat_spec[0, 0])
        ax_h2 = fig.add_subplot(heat_spec[0, 1])
        ax_d_max = fig.add_subplot(distribution_spec[0, 0])
        ax_d_min = fig.add_subplot(distribution_spec[0, 1])
        heat_axes.extend([ax_h1, ax_h2])
        distribution_axes.extend([ax_d_max, ax_d_min])

        ax_h1.set_ylabel(BAG_LABELS[bag], fontsize=FS)
        if ri == 0:
            ax_h1.set_title("a. Stratified BAG bias", fontsize=FS + 5, pad=28, loc="left", x=-0.55)
            ax_d_max.set_title("b. BAG bias pooled by diagnosis", fontsize=FS + 5, pad=52, loc="left")
            ax_h1.text(0.5, 1.02, _OBJECTIVE_PRESENTATION[SYNERGY_OBJECTIVE]["arm_label"], transform=ax_h1.transAxes, ha="center", va="bottom", fontsize=FS)
            ax_h2.text(0.5, 1.02, _OBJECTIVE_PRESENTATION[REDUNDANCY_OBJECTIVE]["arm_label"], transform=ax_h2.transAxes, ha="center", va="bottom", fontsize=FS)
            ax_d_max.text(0.5, 1.02, "Max O-info", transform=ax_d_max.transAxes, ha="center", va="bottom", fontsize=FS)
            ax_d_min.text(0.5, 1.02, "Min O-info", transform=ax_d_min.transAxes, ha="center", va="bottom", fontsize=FS)

        bag_syn = country_df[(country_df["bag"] == bag) & (country_df["objective"] == SYNERGY_OBJECTIVE)].copy()
        bag_red = country_df[(country_df["bag"] == bag) & (country_df["objective"] == REDUNDANCY_OBJECTIVE)].copy()

        im = _draw_heatmap_panel(
            ax_h1,
            bag_syn,
            ctry_order=ctry_order,
            abs_max=abs_max,
            objective=SYNERGY_OBJECTIVE,
            show_row_labels=True,
            title=None,
        )
        _draw_heatmap_panel(
            ax_h2,
            bag_red,
            ctry_order=ctry_order,
            abs_max=abs_max,
            objective=REDUNDANCY_OBJECTIVE,
            show_row_labels=False,
            title=None,
        )

        if ri < len(BAGS) - 1:
            ax_h1.tick_params(axis="x", labelbottom=False)
            ax_h2.tick_params(axis="x", labelbottom=False)
        else:
            ax_h1.set_xticklabels([DX_LABELS[d] for d in DX_ORDER if d in bag_syn["diagnosis"].unique()])
            ax_h2.set_xticklabels([DX_LABELS[d] for d in DX_ORDER if d in bag_red["diagnosis"].unique()])
        ax_h1.spines["top"].set_visible(False)
        ax_h2.spines["top"].set_visible(False)
        ax_h1.spines["left"].set_visible(True)
        ax_h1.spines["bottom"].set_visible(True)
        ax_h1.spines["top"].set_visible(True)
        ax_h1.spines["right"].set_visible(True)
        ax_h2.spines["left"].set_visible(True)
        ax_h2.spines["bottom"].set_visible(True)
        ax_h2.spines["top"].set_visible(True)
        ax_h2.spines["right"].set_visible(True)
        for spine in ["left", "bottom", "top", "right"]:
            ax_h1.spines[spine].set_color(SYN_BORDER)
            ax_h2.spines[spine].set_color(RED_BORDER)
            ax_h1.spines[spine].set_linewidth(4.4)
            ax_h2.spines[spine].set_linewidth(4.4)
        ax_h1.spines["right"].set_color(SYN_BORDER)
        ax_h2.spines["left"].set_color(RED_BORDER)
        ax_h1.tick_params(axis="y", which="both", length=0, pad=2)
        ax_h2.tick_params(axis="y", which="both", length=0, pad=2)
        ax_h1.tick_params(axis="y", labelsize=FS_TK + 3)
        ax_h1.tick_params(axis="x", labelsize=FS_TK + 2)
        ax_h2.tick_params(axis="x", labelsize=FS_TK + 2)

        subj_syn = subject_df[(subject_df["bag"] == bag) & (subject_df["objective"] == SYNERGY_OBJECTIVE)].copy()
        subj_red = subject_df[(subject_df["bag"] == bag) & (subject_df["objective"] == REDUNDANCY_OBJECTIVE)].copy()
        _draw_distribution_panel(
            ax_d_max,
            subj_red,
            show_legend=(ri == 0),
            x_lo=x_lo,
            x_hi=x_hi,
            y_hi=y_hi,
            objective=REDUNDANCY_OBJECTIVE,
        )
        _draw_distribution_panel(
            ax_d_min,
            subj_syn,
            show_legend=False,
            x_lo=x_lo,
            x_hi=x_hi,
            y_hi=y_hi,
            objective=SYNERGY_OBJECTIVE,
        )
        if ri < len(BAGS) - 1:
            ax_d_max.set_xticklabels([])
            ax_d_min.set_xticklabels([])
        else:
            ax_d_max.set_xlabel("")
            ax_d_min.set_xlabel("")
        for ax_d in (ax_d_max, ax_d_min):
            ax_d.tick_params(axis="x", labelsize=FS_TK + 2)
            ax_d.tick_params(axis="y", labelsize=FS_TK + 2)
            ax_d.grid(alpha=0.18, linewidth=0.6)
        ax_d_min.set_ylabel("")
        ax_d_min.tick_params(axis="y", labelleft=False)

    heat_right = max(ax.get_position().x1 for ax in heat_axes)
    hist_left = min(ax.get_position().x0 for ax in distribution_axes)
    hist_right = max(ax.get_position().x1 for ax in distribution_axes)
    fig.text(
        (hist_left + hist_right) / 2,
        0.025,
        "BAG bias (predicted − observed, years)",
        ha="center",
        va="bottom",
        fontsize=FS,
    )
    gap = max(hist_left - heat_right, 0.01)
    cbar_width = min(0.009, gap * 0.225)
    cbar_x = heat_right + gap * 0.06
    cbar_bottom = min(ax.get_position().y0 for ax in heat_axes)
    cbar_top = max(ax.get_position().y1 for ax in heat_axes)
    cbar_ax = fig.add_axes([cbar_x, cbar_bottom, cbar_width, cbar_top - cbar_bottom])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max])
    cbar.set_ticklabels(
        [
            f"{-abs_max:.1f}",
            f"{-abs_max/2:.1f}",
            "0",
            f"{abs_max/2:.1f}",
            f"{abs_max:.1f}",
        ]
    )
    cbar.ax.tick_params(labelsize=FS_TK - 1)
    cbar.ax.text(
        0.5,
        0.5,
        "Median BAG bias (years)",
        transform=cbar.ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        color="black",
        fontsize=FS,
        zorder=5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"fig4_residual_bias{SUFFIX}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"fig4_residual_bias{SUFFIX}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"fig4_residual_bias{SUFFIX}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if os.environ.get("PAPER_FIG_WRITE_SOURCE_DATA", "1") != "0":
        for path in write_source_data(
            f"fig4_residual_bias{SUFFIX}",
            _build_fig4_panels(subject_df, country_df, selection_df, ctry_order, abs_max),
            OUT_DIR,
            source_paths=[str(RES_DIR), str(STATS_BASE / "best_rung_selection.csv")],
        ):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
