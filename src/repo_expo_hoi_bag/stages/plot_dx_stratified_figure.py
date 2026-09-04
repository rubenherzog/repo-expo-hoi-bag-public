#!/usr/bin/env python3
"""
plot_dx_stratified_figure.py
============================
Visualization of pooled-train / dx-stratified-test results.

Layout (2 rows × 3 cols):
  Row 1 – Combined BAG
  Row 2 – Functional BAG

  Col A – Attenuation gradient: median ± 95% CI across all 20 top models,
           per diagnosis, side-by-side synergistic vs redundant.
           Reference line = pooled OOF R² ("full case").

  Col B – Best-performing model only (by pooled R²): violin/strip per dx
           comparing within-fold CN vs {AD, FTD}.

  Col C – Per-country heatmap: median R² (across models) for each
           (fold_country × eval_dx) cell. Shows which geographies drive
           attenuation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from scripts.pipeline_utils import load_stage_config
_STAGE_CFG  = load_stage_config("compute_phase2_stats")
_COMMON_CFG = load_stage_config("common")

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
DATA_ROOT = Path(os.environ.get("V3_PLOT_DATA_ROOT", str(OUTPUT_ROOT)))
EVAL_BASE = DATA_ROOT / "dx_stratified_eval"

FIGURE_OUT = OUTPUT_ROOT / "figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
DPI = 220
FS  = 15        # base fontsize
FS_SMALL = 12
FS_TITLE = 15

DX_ORDER  = ["CN", "AD", "FTD"]
DX_COLORS = {
    "CN":  "#4dac26",   # green
    "AD":  "#d01c8b",   # magenta
    "FTD": "#0571b0",   # blue
}
DX_LABELS = {"CN": "CN", "AD": "AD", "FTD": "FTD"}

GRP_COLORS = {
    "synergistic": "#2166ac",
    "redundant":   "#b2182b",
}
GRP_LABELS = {"synergistic": "Synergistic (o-min)", "redundant": "Redundant (o-max)"}

BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}


MIN_N_TEST = int(_STAGE_CFG.get("min_n_test", 20))  # drop folds with fewer test subjects
_BOOT_SEED = int(_COMMON_CFG.get("bootstrap_rng_seed", 42))
_N_BOOT    = int(_COMMON_CFG.get("n_bootstrap", 2000))

# ── data loading ──────────────────────────────────────────────────────────────
def _dx_files_exist(bag: str) -> bool:
    p = EVAL_BASE / bag
    return (p / "dx_stratified_country_dx.parquet").exists() and (p / "dx_stratified_global.parquet").exists()


def load(bag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = EVAL_BASE / bag
    country_path = p / "dx_stratified_country_dx.parquet"
    global_path = p / "dx_stratified_global.parquet"
    if not country_path.exists() or not global_path.exists():
        missing = []
        if not country_path.exists():
            missing.append(str(country_path))
        if not global_path.exists():
            missing.append(str(global_path))
        raise FileNotFoundError(
            f"Missing dx-stratified files for bag '{bag}': {', '.join(missing)}"
        )

    df_country = pd.read_parquet(country_path)
    df_global  = pd.read_parquet(global_path)
    # Filter out small-sample folds
    n_before = len(df_country)
    df_country = df_country[df_country["n_test"] >= MIN_N_TEST]
    n_after = len(df_country)
    if n_before != n_after:
        print(f"  [{bag}] n_test filter: {n_before} → {n_after} rows "
              f"(dropped {n_before - n_after})")
    return df_country, df_global


def best_model_id(df: pd.DataFrame, group: str) -> str:
    sub = df[df.transfer_group == group]
    return sub.groupby("model_id")["pooled_r2"].first().idxmax()


# ── plot helpers ─────────────────────────────────────────────────────────────

def ci95(x):
    """Median ± 95% CI via bootstrap (1 000 resamples)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(x)
    bs  = np.array([np.median(np.random.choice(x, size=len(x), replace=True))
                    for _ in range(1_000)])
    return med, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def panel_A_attenuation(ax, df: pd.DataFrame, pooled_r2: float, bag: str, show_ylabel=True):
    """
    Grouped bar chart: median R² per (transfer_group × eval_dx).
    95% CI error bars.  Reference line = pooled OOF R².
    """
    rng = np.random.default_rng(_BOOT_SEED)
    n_dx  = len(DX_ORDER)
    n_grp = 2
    groups = ["synergistic", "redundant"]

    bar_w  = 0.32
    gap    = 0.06
    group_w = n_grp * bar_w + gap
    x_centers = np.arange(n_dx) * (group_w + 0.20)

    for gi, grp in enumerate(groups):
        sub = df[df.transfer_group == grp]
        for di, dx in enumerate(DX_ORDER):
            vals = sub[sub.eval_dx == dx]["r2"].dropna().values
            if len(vals) == 0:
                continue
            med, lo, hi = ci95(vals)
            x = x_centers[di] + gi * (bar_w + gap / 2)
            color = GRP_COLORS[grp]
            ax.bar(x, med, width=bar_w, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.5, zorder=2)
            ax.errorbar(x, med, yerr=[[med - lo], [hi - med]],
                        fmt="none", color="#333333", capsize=3, linewidth=1, zorder=3)
            # Individual country dots (median per country across models)
            country_medians = (
                sub[sub.eval_dx == dx]
                .groupby("fold_country")["r2"]
                .median()
                .dropna()
                .values
            )
            if len(country_medians):
                jitter = rng.uniform(-bar_w * 0.35, bar_w * 0.35, len(country_medians))
                ax.scatter(x + jitter, country_medians,
                           s=14, color=color, alpha=0.65, edgecolors="white",
                           linewidths=0.3, zorder=4)
            # Sample size annotation
            n_countries = sub[sub.eval_dx == dx]["fold_country"].nunique()
            n_test_total = sub[sub.eval_dx == dx].groupby("fold_country")["n_test"].first().sum()
            ax.text(x, -0.06, f"{n_countries}c\n{n_test_total:,}",
                    ha="center", va="top", fontsize=8.5, color="#555555")

    # Reference line: pooled OOF R²
    ax.axhline(pooled_r2, color="#555555", ls="--", lw=1.2, zorder=1,
               label=f"Pooled OOF R²={pooled_r2:.3f}")
    ax.axhline(0, color="#999999", ls=":", lw=0.8, zorder=1)

    # DX tick labels (centered between both bars)
    tick_pos = [x_centers[di] + bar_w / 2 + gap / 4 for di in range(n_dx)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([DX_LABELS[dx] for dx in DX_ORDER], fontsize=FS)

    if show_ylabel:
        ax.set_ylabel("Median R²", fontsize=FS)
    ax.set_ylim(-0.30, pooled_r2 + 0.15)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend patches
    patches = [mpatches.Patch(color=GRP_COLORS[g], alpha=0.78, label=GRP_LABELS[g])
               for g in groups]
    patches.append(mpl.lines.Line2D([0], [0], color="#555555", ls="--", lw=1.2,
                                    label=f"Pooled OOF R²={pooled_r2:.3f}"))
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              framealpha=0.85, edgecolor="none")
    ax.set_title(f"A  Top-20 models | {bag.capitalize()}",
                 fontsize=FS_TITLE, fontweight="bold", loc="left")


def panel_B_best_model(ax, df: pd.DataFrame, pooled_r2: float, bag: str, show_ylabel=True):
    """
    For the single best synergistic model: strip + box per dx,
    plus best redundant as a comparison line/dot.
    """
    best_syn = best_model_id(df, "synergistic")
    best_red = best_model_id(df, "redundant")
    syn_r2   = df[df.model_id == best_syn]["pooled_r2"].iloc[0]
    red_r2   = df[df.model_id == best_red]["pooled_r2"].iloc[0]

    rng = np.random.default_rng(_BOOT_SEED)

    for di, dx in enumerate(DX_ORDER):
        # Synergistic – strip
        vals_syn = df[(df.model_id == best_syn) & (df.eval_dx == dx)]["r2"].dropna().values
        if len(vals_syn):
            jitter = rng.uniform(-0.12, 0.12, len(vals_syn))
            ax.scatter(di + jitter, vals_syn,
                       color=DX_COLORS[dx], s=18, alpha=0.55, edgecolors="none", zorder=3)
            med_s, lo_s, hi_s = ci95(vals_syn)
            ax.plot([di - 0.22, di + 0.22], [med_s, med_s],
                    color=DX_COLORS[dx], lw=2.5, zorder=4, solid_capstyle="round")
            ax.errorbar(di, med_s, yerr=[[med_s - lo_s], [hi_s - med_s]],
                        fmt="none", color=DX_COLORS[dx], capsize=4, lw=1.4, zorder=4)

        # Redundant – hollow diamond
        vals_red = df[(df.model_id == best_red) & (df.eval_dx == dx)]["r2"].dropna().values
        if len(vals_red):
            med_r = np.median(vals_red)
            ax.plot(di, med_r, marker="D", ms=6, mfc="none",
                    mec=GRP_COLORS["redundant"], mew=1.5, zorder=5,
                    label="Best redundant (median)" if di == 0 else "_")

    # Reference lines
    ax.axhline(syn_r2, color="#555555", ls="--", lw=1.2, zorder=1)
    ax.axhline(0, color="#999999", ls=":", lw=0.8, zorder=1)

    ax.set_xticks(range(len(DX_ORDER)))
    ax.set_xticklabels([DX_LABELS[dx] for dx in DX_ORDER], fontsize=FS)
    if show_ylabel:
        ax.set_ylabel("Median R²", fontsize=FS)
    ax.set_ylim(-0.45, syn_r2 + 0.15)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    dot_patches = [
        mpatches.Patch(color=DX_COLORS[dx], alpha=0.7, label=dx) for dx in DX_ORDER
    ]
    diamond = mpl.lines.Line2D([0], [0], marker="D", ms=6, mfc="none",
                                mec=GRP_COLORS["redundant"], mew=1.5, ls="none",
                                label=f"Best redundant (pooled R²={red_r2:.3f})")
    ref_line = mpl.lines.Line2D([0], [0], color="#555555", ls="--", lw=1.2,
                                 label=f"Best synergistic pooled R²={syn_r2:.3f}")

    ax.legend(handles=dot_patches + [diamond, ref_line],
              fontsize=8, loc="lower right", framealpha=0.85, edgecolor="none", ncol=2)
    ax.set_title(
        f"B  Best synergistic model (pooled R²={syn_r2:.3f})",
        fontsize=FS_TITLE, fontweight="bold", loc="left")


def panel_C_country_heatmap(ax, df: pd.DataFrame, bag: str, group: str = "synergistic"):
    """
    Heatmap: rows = countries, cols = DX_ORDER.
    Values = median R² over all 20 models for each (country, dx) cell.
    """
    sub = df[df.transfer_group == group]
    pivot = (
        sub.groupby(["fold_country", "eval_dx"])["r2"]
        .median()
        .unstack("eval_dx")
        .reindex(columns=DX_ORDER)
    )
    # Sort countries by CN R² descending
    pivot = pivot.sort_values("CN", ascending=False)

    countries = pivot.index.tolist()
    data = pivot.values

    vmax = 0.55
    vmin = -0.60
    cmap = mpl.cm.RdBu
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Annotate cells
    for ri, country in enumerate(countries):
        for ci, dx in enumerate(DX_ORDER):
            val = data[ri, ci]
            if np.isfinite(val):
                txt_color = "white" if abs(val) > 0.30 else "black"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

    ax.set_xticks(range(len(DX_ORDER)))
    ax.set_xticklabels([DX_LABELS[dx] for dx in DX_ORDER], fontsize=FS)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=7)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.colorbar(im, ax=ax, shrink=0.85, label="Median R²", pad=0.02)
    ax.set_title(
        "C  Country × Dx heatmap (syn)",
        fontsize=FS_TITLE, fontweight="bold", loc="left")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    bags = [bag for bag in ["combined", "functional", "structural"] if _dx_files_exist(bag)]
    for bag in ["combined", "functional", "structural"]:
        if bag not in bags:
            missing_dir = EVAL_BASE / bag
            print(f"  ⚠ {bag}: missing dx-stratified outputs under {missing_dir} — skipping")

    if not bags:
        print("No dx-stratified bags available — nothing to plot.")
        return

    fig, axes = plt.subplots(
        len(bags), 3,
        figsize=(20, 6 * len(bags)),
        gridspec_kw={"width_ratios": [1.1, 1.1, 1.0], "wspace": 0.38, "hspace": 0.16},
    )
    if len(bags) == 1:
        axes = axes[np.newaxis, :]

    for row, bag in enumerate(bags):
        df, glob_df = load(bag)
        best_syn_id = best_model_id(df, "synergistic")
        pooled_r2   = df[df.model_id == best_syn_id]["pooled_r2"].iloc[0]

        panel_A_attenuation(axes[row, 0], df, pooled_r2, bag, show_ylabel=True)
        panel_B_best_model(axes[row, 1], df, pooled_r2, bag, show_ylabel=(row == 0))
        panel_C_country_heatmap(axes[row, 2], df, bag)

    fig.suptitle(
        "Exposome–BAG relationship across diagnostic groups\n"
        "Pooled training · per-diagnosis test (LOCO-country CV)",
        fontsize=FS_TITLE, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(top=0.92)

    out = FIGURE_OUT / "fig_dx_stratified_attenuation.pdf"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    png = FIGURE_OUT / "fig_dx_stratified_attenuation.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {out}\n  {png}")


if __name__ == "__main__":
    rng_seed = 42
    np.random.seed(rng_seed)
    main()
