"""
plot_regression_residuals.py
=============================
Figures for the regression pipeline residual analysis — the threshold-free
continuous analogue of the classifier misclassification analysis.

Signed BAG bias = y_pred_full − y_true (years):
  bias > 0  → model overestimates BAG
  bias < 0  → model underestimates BAG
  |bias|    → magnitude of error regardless of direction

Figures produced (all → reports/paper_report_v1/figures/):
  resid_bias_heatmap_{bag}_omin.png    — country × diagnosis bias heatmap (o_min)
  resid_bias_heatmap_{bag}_omax.png    — same for o_max
  resid_rmse_heatmap_{bag}_omin.png    — country × diagnosis RMSE heatmap (o_min)
  resid_delta_{bag}.png                — Δbias (o_min − o_max) heatmap
  resid_by_rung_combined.png           — bias & RMSE vs. rung (combined, o_min)
  resid_age_continuous.png             — residual vs. continuous age, per-dx regression lines
  resid_scatter_{bag}.png              — y_true vs y_pred scatter, best-syn, colored by dx

Usage:
    python -m scripts.plot_regression_residuals
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, linregress
from scripts.sensitivity_common import (
    load_sensitivity_config,
    paper_analysis_config,
)

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR   = REPO_ROOT / "outputs" / "variant_a" / "stats" / "residuals"
FIG_DIR   = REPO_ROOT / "reports" / "paper_report_v1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_SENSITIVITY_CFG = load_sensitivity_config()
_PAPER_CFG = paper_analysis_config(_SENSITIVITY_CFG)
_RESIDUAL_CFG = _PAPER_CFG.residual_bias
BAGS = list(_PAPER_CFG.residual_bias_bags)
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}
OBJECTIVES = list(_PAPER_CFG.objectives)
OBJ_LABELS = {
    objective: values["arm_label"]
    for objective, values in _PAPER_CFG.objective_metadata.items()
}

RUNG_ORDER = [str(value) for value in _SENSITIVITY_CFG["defaults"]["active_rungs"]]
RUNG_LABELS = {
    "ols":          "OLS",
    "xgb_tree_d1":  "XGB d=1",
    "xgb_tree_d2":  "XGB d=2",
    "xgb_tree_d3":  "XGB d=3",
}
DX_ORDER = list(_PAPER_CFG.primary_diagnoses)
AGE_ORDER = [str(value) for value in _RESIDUAL_CFG["age_labels"]]
DX_COLORS = dict(zip(DX_ORDER, ("#4292c6", "#e6550d", "#74c476")))

N_SUBJ_MIN = 10   # gray-hatch below this


# ── drawing helpers ─────────────────────────────────────────────────────────

def draw_bias_panel(
    ax, data, n_subj,
    row_labels, col_labels,
    vmin, vmax, cmap,
    show_row_labels=True,
    annotate_threshold=30,
    annotate_fontsize=7.5,
):
    """Draw a single heatmap panel for signed bias (diverging) or RMSE (sequential)."""
    nrows, ncols = data.shape
    im = ax.imshow(data, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    for ri in range(nrows):
        for ci in range(ncols):
            n = n_subj[ri, ci] if n_subj is not None else np.nan
            val = data[ri, ci]
            if np.isnan(val) or (n < N_SUBJ_MIN):
                ax.add_patch(mpatches.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    fill=True, facecolor="#cfcfcf", edgecolor="#cfcfcf",
                    linewidth=0))
            elif n >= annotate_threshold:
                # For diverging maps pick text color by position
                if vmin < 0:
                    mid = 0.0
                    text_col = "white" if abs(val) > 0.65 * max(abs(vmin), vmax) else "black"
                else:
                    text_col = "white" if val > 0.65 * vmax else "black"
                _bold_path = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
                _bold_fp = (
                    fm.FontProperties(fname=_bold_path, size=annotate_fontsize)
                    if os.path.exists(_bold_path)
                    else fm.FontProperties(weight="bold", size=annotate_fontsize)
                )
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        color=text_col, fontproperties=_bold_fp)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(nrows))
    ax.tick_params(axis="y", which="both", length=0, pad=3)
    ax.set_yticklabels(row_labels if show_row_labels else [""] * nrows, fontsize=8.5)
    for y in np.arange(-0.5, nrows - 0.5):
        ax.axhline(y, color="white", linewidth=0.5)
    for x in np.arange(-0.5, ncols - 0.5):
        ax.axvline(x, color="white", linewidth=0.5)
    return im


def add_colorbar(fig, ax, im, ticks, tick_labels):
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.04, fraction=0.06, shrink=0.85)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)


def pivot_ctry_dx(df_cdx, metric, country_order, present_dx):
    rate_piv = df_cdx.pivot(index="country", columns="diagnosis", values=metric)
    n_piv    = df_cdx.pivot(index="country", columns="diagnosis", values="n_subjects")
    rate_piv = rate_piv.reindex(index=country_order, columns=present_dx)
    n_piv    = n_piv.reindex(index=country_order, columns=present_dx)
    return rate_piv.values, n_piv.values


def country_order_from(df_cdx, metric="abs_bias"):
    avg = df_cdx.groupby("country")[metric].mean().sort_values(ascending=False)
    return avg.index.tolist()


# ── Figure: country × diagnosis bias heatmap ────────────────────────────────

def plot_bias_heatmap(bag, objective, ctry_order):
    """2×2 (bias/RMSE × o_min/o_max is handled by separate calls)."""
    # Load country-level bias (top-20 from cluster)
    df_ctry = pd.read_csv(RES_DIR / f"residuals_country_{bag}_{objective}.csv")
    # Load subject-level for dx breakdown (best-syn, richer)
    df_subj = pd.read_csv(RES_DIR / f"residuals_ctry_dx_{bag}.csv")

    present_dx   = [d for d in DX_ORDER if d in df_subj["diagnosis"].unique()]
    present_ctry = [c for c in ctry_order if c in df_subj["country"].unique()]
    n_ctry = len(present_ctry)

    # Colour scales
    bias_vals = df_subj["bias_mean"].values
    abs_max   = np.nanpercentile(np.abs(bias_vals), 95)
    abs_max   = max(abs_max, 0.5)
    rmse_vmax = np.nanpercentile(df_subj["rmse"].values, 95)
    rmse_vmax = max(rmse_vmax, 1.0)

    row_h = max(7.0, 0.38 * n_ctry + 2.5)
    fig = plt.figure(figsize=(13.0, row_h))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.13, right=0.97, top=0.90, bottom=0.10,
                            wspace=0.08)

    # Panel 0: signed bias (country average, from cluster top-20)
    ax0 = fig.add_subplot(gs[0, 0])
    ctry_avg = df_ctry.set_index("country")["bias_mean"].reindex(present_ctry).values.reshape(-1, 1)
    n_avg    = df_ctry.set_index("country")["n_subjects"].reindex(present_ctry).values.reshape(-1, 1)
    im0 = draw_bias_panel(ax0, ctry_avg, n_avg,
                          row_labels=present_ctry, col_labels=["All Dx"],
                          vmin=-abs_max, vmax=abs_max,
                          cmap=plt.cm.RdBu_r, show_row_labels=True)
    ax0.set_title("Signed bias\n(country avg, top-20)", fontsize=9.5, pad=6)
    add_colorbar(fig, ax0, im0,
                 [-abs_max, 0, abs_max],
                 [f"{-abs_max:.1f}", "0", f"+{abs_max:.1f}"])

    # Panel 1: bias per country × diagnosis (from subject OOF)
    ax1 = fig.add_subplot(gs[0, 1])
    b_vals, n_vals = pivot_ctry_dx(df_subj, "bias_mean", present_ctry, present_dx)
    im1 = draw_bias_panel(ax1, b_vals, n_vals,
                          row_labels=present_ctry, col_labels=present_dx,
                          vmin=-abs_max, vmax=abs_max,
                          cmap=plt.cm.RdBu_r, show_row_labels=False)
    ax1.set_title("Signed bias by diagnosis\n(best-syn model)", fontsize=9.5, pad=6)
    add_colorbar(fig, ax1, im1,
                 [-abs_max, 0, abs_max],
                 [f"{-abs_max:.1f}", "0", f"+{abs_max:.1f}"])

    # Panel 2: RMSE per country × diagnosis
    ax2 = fig.add_subplot(gs[0, 2])
    r_vals, n_vals2 = pivot_ctry_dx(df_subj, "rmse", present_ctry, present_dx)
    im2 = draw_bias_panel(ax2, r_vals, n_vals2,
                          row_labels=present_ctry, col_labels=present_dx,
                          vmin=0, vmax=rmse_vmax,
                          cmap=plt.cm.YlOrRd, show_row_labels=False)
    ax2.set_title("RMSE by diagnosis\n(best-syn model, years)", fontsize=9.5, pad=6)
    add_colorbar(fig, ax2, im2,
                 [0, rmse_vmax / 2, rmse_vmax],
                 ["0", f"{rmse_vmax/2:.1f}", f"≥{rmse_vmax:.1f}"])

    obj_str = OBJ_LABELS[objective]
    fig.suptitle(
        f"{BAG_LABELS[bag]}  —  Regression residuals by country × diagnosis\n"
        f"({obj_str}, top-20 models / best-syn OOF)\n"
        "Blue = BAG underestimation; Red = BAG overestimation",
        fontsize=10.5, y=0.97)

    out = FIG_DIR / f"resid_bias_heatmap_{bag}_{objective}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: delta bias (o_min − o_max) ─────────────────────────────────────

def plot_delta_heatmap(bag, ctry_order):
    df_min = pd.read_csv(RES_DIR / f"residuals_country_{bag}_o_min.csv")
    df_max = pd.read_csv(RES_DIR / f"residuals_country_{bag}_o_max.csv")

    present_ctry = [c for c in ctry_order
                    if c in df_min["country"].values and c in df_max["country"].values]
    n_ctry = len(present_ctry)

    bias_min = df_min.set_index("country")["bias_mean"].reindex(present_ctry).values
    bias_max = df_max.set_index("country")["bias_mean"].reindex(present_ctry).values
    rmse_min = df_min.set_index("country")["rmse"].reindex(present_ctry).values
    rmse_max = df_max.set_index("country")["rmse"].reindex(present_ctry).values
    n_vals   = df_min.set_index("country")["n_subjects"].reindex(present_ctry).values

    delta_bias = (bias_min - bias_max).reshape(-1, 1)
    delta_rmse = (rmse_min - rmse_max).reshape(-1, 1)
    n_mat      = n_vals.reshape(-1, 1)

    abs_max_b = max(np.nanpercentile(np.abs(delta_bias), 95), 0.1)
    abs_max_r = max(np.nanpercentile(np.abs(delta_rmse), 95), 0.1)

    row_h = max(6.0, 0.38 * n_ctry + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, row_h))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.08, wspace=0.15)

    for ax, data, abs_max, title, unit in [
        (axes[0], delta_bias, abs_max_b, "Δ Signed bias\n(o_min − o_max, years)", "yr"),
        (axes[1], delta_rmse, abs_max_r, "Δ RMSE\n(o_min − o_max, years)", "yr"),
    ]:
        im = draw_bias_panel(ax, data, n_mat,
                             row_labels=present_ctry if ax is axes[0] else [""] * n_ctry,
                             col_labels=[""],
                             vmin=-abs_max, vmax=abs_max,
                             cmap=plt.cm.RdBu_r,
                             show_row_labels=(ax is axes[0]))
        ax.set_title(title, fontsize=10, pad=6)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                            pad=0.04, fraction=0.06, shrink=0.85)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_ticks([-abs_max, 0, abs_max])
        cbar.set_ticklabels([f"{-abs_max:.2f}", "0", f"+{abs_max:.2f}"])

    fig.suptitle(
        f"{BAG_LABELS[bag]}  —  Δ residuals (o_min − o_max) by country\n"
        "Red = synergistic models have larger error; Blue = smaller error",
        fontsize=10.5, y=0.97)

    out = FIG_DIR / f"resid_delta_{bag}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: 3-panel signed-bias heatmap (one panel per BAG, o_min) ─────────

def plot_combined_bias_heatmap(ctry_order):
    """
    Single figure: 3 panels side by side, one per BAG target.
    Each panel shows country × diagnosis signed bias from the best-syn OOF (o_min).
    No RMSE, no top-20 cluster column — just the bias grid per BAG.
    """
    # Global colour scale: symmetric around 0, driven by 95th-percentile abs bias
    # across all three BAGs so panels are directly comparable.
    all_bias = []
    for bag in BAGS:
        df = pd.read_csv(RES_DIR / f"residuals_ctry_dx_{bag}.csv")
        all_bias.extend(df["bias_mean"].dropna().tolist())
    abs_max = max(np.nanpercentile(np.abs(all_bias), 95), 0.5)

    # Collect present countries per BAG; global order = ctry_order filtered to union
    present_dx = list(DX_ORDER)

    # Figure geometry: one panel per BAG, each panel width proportional to n_dx cols
    n_dx = len(present_dx)
    panel_w = 3.2 + 0.9 * n_dx          # each panel needs its own country labels
    fig_w   = panel_w * 3 + 1.0          # 3 panels + colorbar space
    n_ctry_max = max(
        len([c for c in ctry_order
             if c in pd.read_csv(RES_DIR / f"residuals_ctry_dx_{bag}.csv")["country"].unique()])
        for bag in BAGS
    )
    fig_h = max(7.0, 0.42 * n_ctry_max + 2.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Leave generous top margin so suptitle never overlaps panel titles
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.12, right=0.94,
        top=0.84,   # panel tops at 84 % — suptitle lives in 84–100 %
        bottom=0.09,
        wspace=0.22,
    )

    im_ref = None
    for col_i, bag in enumerate(BAGS):
        df = pd.read_csv(RES_DIR / f"residuals_ctry_dx_{bag}.csv")
        present_ctry = [c for c in ctry_order if c in df["country"].unique()]
        n_ctry = len(present_ctry)

        b_vals, n_vals = pivot_ctry_dx(df, "bias_mean", present_ctry, present_dx)

        ax = fig.add_subplot(gs[0, col_i])
        im = draw_bias_panel(
            ax, b_vals, n_vals,
            row_labels=present_ctry,
            col_labels=present_dx,
            vmin=-abs_max, vmax=abs_max,
            cmap=plt.cm.RdBu_r,
            show_row_labels=True,
        )
        ax.set_title(BAG_LABELS[bag], fontsize=10.5, pad=8)
        if im_ref is None:
            im_ref = im

    # Single shared colourbar below all panels
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.025])
    cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max])
    cbar.set_ticklabels(
        [f"{-abs_max:.1f}", f"{-abs_max/2:.1f}", "0",
         f"+{abs_max/2:.1f}", f"+{abs_max:.1f}"]
    )
    cbar.set_label(
        "Signed BAG bias = predicted − observed  (years)   "
        "Blue = underestimation · Red = overestimation",
        fontsize=8.5,
    )

    fig.suptitle(
        "Regression residuals — signed bias by country × diagnosis (best-syn model, o_min)",
        fontsize=11, y=0.97, va="top",
    )

    out = FIG_DIR / "resid_bias_heatmap_all_bags_omin.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: rung ladder (combined, o_min) ──────────────────────────────────

def plot_rung_comparison():
    df = pd.read_csv(RES_DIR / "residuals_by_rung_combined.csv")
    df = df.set_index("rung_id").reindex(RUNG_ORDER)

    x = np.arange(len(RUNG_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, col, ylabel, color, title in [
        (axes[0], "abs_bias", "Mean |bias| (years)", "#e07b39",
         "Mean absolute bias vs. complexity"),
        (axes[1], "rmse",     "RMSE (years)",         "#6a4c9c",
         "RMSE vs. complexity"),
    ]:
        vals = df[col].values
        ax.plot(x, vals, marker="o", markersize=9, linewidth=2.2, color=color)
        for xi, val in zip(x, vals):
            ax.annotate(f"{val:.3f}", (xi, val), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([RUNG_LABELS[r] for r in RUNG_ORDER], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.35, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - 0.05 * (ylim[1] - ylim[0]),
                    ylim[1] + 0.12 * (ylim[1] - ylim[0]))

    fig.suptitle(
        "Combined BAG  —  Regression error metrics across complexity ladder\n"
        "(o_min top-20, averaged over countries)",
        fontsize=10.5)
    fig.tight_layout()

    out = FIG_DIR / "resid_by_rung_combined.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: residuals vs. continuous age ────────────────────────────────────

def plot_age_continuous():
    """
    Scatter of signed BAG bias (predicted − observed) vs. chronological age,
    coloured by diagnosis. Each panel = one BAG target.

    The BAG method itself has a structural age dependency: at extreme ages,
    the model cannot predict values far outside the feasible range [0, ~120],
    which imposes a regression-to-the-mean / boundary compression on residuals.
    This manifests as a positive slope (young subjects under-predicted, old
    subjects over-predicted) — exactly the pattern this figure is designed to
    surface.

    For each diagnosis we show:
      - individual subject dots (alpha=0.15, small)
      - OLS best-fit line (slope ± 95 % CI band via 2*SE)
      - slope annotation (β yr/yr, p-value)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.13, wspace=0.25)

    for ax_i, bag in enumerate(BAGS):
        ax = axes[ax_i]
        df = pd.read_csv(RES_DIR / f"residuals_subject_{bag}.csv")

        # Age range for x-axis (clip outliers slightly)
        age_lo = df["age"].quantile(0.005)
        age_hi = df["age"].quantile(0.995)
        age_fit = np.linspace(age_lo, age_hi, 200)

        legend_handles = []

        for dx in DX_ORDER:
            sub = df[df["diagnosis"] == dx].copy()
            if len(sub) < 5:
                continue

            color = DX_COLORS[dx]

            # --- scatter (thin, transparent) ---
            ax.scatter(sub["age"], sub["residual"],
                       s=4, alpha=0.15, color=color, linewidths=0, zorder=2)

            # --- OLS fit ---
            slope, intercept, r_val, p_val, se = linregress(sub["age"], sub["residual"])
            y_fit = slope * age_fit + intercept

            # 95 % CI band: ±2·SE_of_residuals / sqrt(n) · t-critical ≈ 2
            # Use propagation: Var(ŷ) = SE_slope² * (x - x̄)² + SE_intercept²
            # Simpler: ± 2*se (of slope) scaled by age deviation
            x_mean = sub["age"].mean()
            n = len(sub)
            s_res = np.sqrt(np.sum((sub["residual"] - (slope * sub["age"] + intercept))**2) / (n - 2))
            sxx = np.sum((sub["age"] - x_mean) ** 2)
            se_fit = s_res * np.sqrt(1/n + (age_fit - x_mean)**2 / sxx)
            ci = 2.0 * se_fit  # ≈ 95 %

            ax.plot(age_fit, y_fit, color=color, linewidth=2.2, zorder=4)
            ax.fill_between(age_fit, y_fit - ci, y_fit + ci,
                            color=color, alpha=0.12, linewidth=0, zorder=3)

            # --- slope annotation ---
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            sign  = "+" if slope >= 0 else ""
            label = (f"{dx}  β={sign}{slope:.3f} yr/yr, {p_str}  (N={n})")
            legend_handles.append(
                mpatches.Patch(color=color, label=label, alpha=0.85))

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_xlabel("Chronological age (years)", fontsize=9.5)
        ax.set_ylabel("BAG bias = predicted − observed  (years)" if ax_i == 0 else "",
                      fontsize=9.5)
        ax.set_title(BAG_LABELS[bag], fontsize=10.5, pad=5)
        ax.set_xlim(age_lo - 1, age_hi + 1)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=legend_handles, fontsize=8.0, framealpha=0.9,
                  loc="upper right", handlelength=1.2)

    fig.suptitle(
        "BAG bias vs. chronological age by diagnosis  (best-syn OOF, o_min)\n"
        "bias > 0 = BAG overestimation  ·  bias < 0 = BAG underestimation\n"
        "Shaded band = ±2 SE of fit  ·  A positive β reflects boundary compression: "
        "BAG predictions are squeezed toward the mean at extreme ages",
        fontsize=10.0, y=1.00)

    out = FIG_DIR / "resid_age_continuous.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: y_true vs y_pred scatter ───────────────────────────────────────

def plot_scatter(bag):
    df = pd.read_csv(RES_DIR / f"residuals_subject_{bag}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    for ax_i, dx in enumerate(DX_ORDER):
        ax = axes[ax_i]
        sub = df[df["diagnosis"] == dx]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        color = DX_COLORS[dx]
        ax.scatter(sub["y_true"], sub["y_pred_full"], s=5, alpha=0.25,
                   color=color, linewidths=0, label=dx)

        # Identity line
        lim = max(abs(sub["y_true"].min()), abs(sub["y_true"].max()),
                  abs(sub["y_pred_full"].min()), abs(sub["y_pred_full"].max()))
        lim = lim * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.0, alpha=0.6)

        # Binned mean prediction curve
        bins = np.linspace(sub["y_true"].quantile(0.01),
                           sub["y_true"].quantile(0.99), 25)
        bmean, bedges, _ = binned_statistic(sub["y_true"], sub["y_pred_full"],
                                            statistic="mean", bins=bins)
        bcount, _, _ = binned_statistic(sub["y_true"], sub["y_pred_full"],
                                        statistic="count", bins=bins)
        bc = 0.5 * (bedges[:-1] + bedges[1:])
        ax.plot(bc[bcount >= 5], bmean[bcount >= 5],
                color="#1a1a2e", linewidth=2.0, zorder=4, label="Binned mean")

        # Stats
        r2 = np.corrcoef(sub["y_true"], sub["y_pred_full"])[0, 1] ** 2
        bias = sub["residual"].mean()
        rmse = np.sqrt((sub["residual"] ** 2).mean())
        ax.text(0.04, 0.95,
                f"R²={r2:.3f}\nBias={bias:+.2f} yr\nRMSE={rmse:.2f} yr\nN={len(sub)}",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True BAG (years)", fontsize=9)
        ax.set_ylabel("Predicted BAG (years)" if ax_i == 0 else "", fontsize=9)
        ax.set_title(f"Diagnosis: {dx}", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{BAG_LABELS[bag]}  —  y_true vs y_pred by diagnosis (best-syn OOF)\n"
        "Black dashed = identity; curve = binned mean prediction",
        fontsize=10.5, y=1.01)
    fig.tight_layout()

    out = FIG_DIR / f"resid_scatter_{bag}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure: signed residual distribution by diagnosis ───────────────────────

def plot_residual_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=False)

    for ax_i, bag in enumerate(BAGS):
        ax = axes[ax_i]
        df = pd.read_csv(RES_DIR / f"residuals_subject_{bag}.csv")

        bins = np.linspace(df["residual"].quantile(0.005),
                           df["residual"].quantile(0.995), 60)

        for dx in DX_ORDER:
            sub = df[df["diagnosis"] == dx]["residual"]
            if len(sub) == 0:
                continue
            counts, edges = np.histogram(sub, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, counts, color=DX_COLORS[dx], linewidth=2.0,
                    label=f"{dx} (N={len(sub)})")
            ax.axvline(sub.mean(), color=DX_COLORS[dx], linewidth=1.2,
                       linestyle="--", alpha=0.7)

        ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
        ax.set_xlabel("BAG bias (predicted − observed, years)", fontsize=9)
        ax.set_ylabel("Density" if ax_i == 0 else "", fontsize=9.5)
        ax.set_title(BAG_LABELS[bag], fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.85)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Distribution of BAG bias (predicted − observed) by diagnosis\n"
        "Positive = overestimation  ·  Negative = underestimation\n"
        "Dashed lines = per-diagnosis mean BAG bias",
        fontsize=10.5, y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "resid_distribution_by_dx.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # Country sort order: structural has most countries, use abs_bias
    df_struct = pd.read_csv(RES_DIR / "residuals_ctry_dx_structural.csv")
    ctry_ord = country_order_from(df_struct, metric="abs_bias")
    print(f"Country order (structural abs_bias, descending):\n  {ctry_ord}\n")

    for bag in BAGS:
        print(f"\n--- {bag.upper()} ---")
        for objective in OBJECTIVES:
            plot_bias_heatmap(bag, objective, ctry_ord)
        plot_delta_heatmap(bag, ctry_ord)
        plot_scatter(bag)

    plot_combined_bias_heatmap(ctry_ord)
    plot_rung_comparison()
    plot_age_continuous()
    plot_residual_distribution()

    print(f"\nAll figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
