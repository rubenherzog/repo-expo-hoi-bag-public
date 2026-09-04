"""
Fig 4 – Null-model benchmark: random variable draws vs O-information–guided models.

Layout: 2 rows (combined, functional) × 2 cols (all null, diversity-filtered null)
Each panel: histogram of null R², vertical lines / rug for real top-20 syn & red.

Reads:
  null_model/{bag}/null_global.parquet
  experiments/pooled_oinfo_ladder_{bag}/xgb_tree_d2/xgb_summary_with_calibration_complexity.csv

Writes:
  outputs/fig_null_model_benchmark.png
  outputs/fig_null_model_benchmark.pdf
"""

import os
import pathlib, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scripts.pipeline_utils import load_stage_config
_NULL_CFG = load_stage_config("null_model")

# ── paths ─────────────────────────────────────────────────────────────
OUTPUT_ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    "outputs/variant_a",
))
DATA_ROOT = pathlib.Path(os.environ.get("V3_PLOT_DATA_ROOT", str(OUTPUT_ROOT)))
EXPERIMENTS_ROOT = DATA_ROOT / "experiments"
if not EXPERIMENTS_ROOT.exists():
    EXPERIMENTS_ROOT = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "experiments"
FIG_DIR = OUTPUT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

BAGS = ["combined", "functional", "structural"]
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}
DIV_THRESH = float(_NULL_CFG.get("diversity_max_domain_frac", 0.60))  # max_domain_frac ≤ this → "diverse"

# ── colours ──────────────────────────────────────────────────────────────
C_NULL_ALL  = "#bdbdbd"
C_NULL_DIV  = "#9ecae1"
C_SYN       = "#d62728"   # red-ish for synergistic
C_RED       = "#1f77b4"   # blue for redundant
ALPHA_HIST  = 0.70

# ── load data ────────────────────────────────────────────────────────────
null_dfs, real_dfs = {}, {}
available_bags = []
for bag in BAGS:
    null_path = DATA_ROOT / f"null_model/{bag}/null_global.parquet"
    real_path = EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / "xgb_tree_d2" / "xgb_summary_with_calibration_complexity.csv"
    if not null_path.exists():
        print(f"  ⚠ {bag}: missing null-model file {null_path} — skipping")
        continue
    if not real_path.exists():
        print(f"  ⚠ {bag}: missing experiment summary {real_path} — skipping")
        continue
    null_dfs[bag] = pd.read_parquet(null_path)
    real_dfs[bag] = pd.read_csv(real_path)
    available_bags.append(bag)

if not available_bags:
    print("No bags available — nothing to plot.")
    import sys; sys.exit(0)

# ── figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(available_bags), 2, figsize=(12, 4 * len(available_bags)), sharex="row", sharey="row")
if len(available_bags) == 1:
    axes = axes[np.newaxis, :]

for row, bag in enumerate(available_bags):
    null = null_dfs[bag]
    real = real_dfs[bag]

    syn_top20 = real[real["objective"] == "o_min"].nlargest(20, "global_oof_r2")["global_oof_r2"]
    red_top20 = real[real["objective"] == "o_max"].nlargest(20, "global_oof_r2")["global_oof_r2"]

    null_all = null["global_oof_r2"]
    null_div = null.loc[null["max_domain_frac"] <= DIV_THRESH, "global_oof_r2"]

    for col, (subset, label, color, n_label) in enumerate([
        (null_all, "All null draws", C_NULL_ALL, f"N = {len(null_all):,}"),
        (null_div, f"Diverse null (max domain ≤ {DIV_THRESH:.0%})", C_NULL_DIV, f"N = {len(null_div):,}"),
    ]):
        ax = axes[row, col]

        # histogram
        bins = np.linspace(subset.min() - 0.01, max(subset.max(), syn_top20.max()) + 0.01, 60)
        ax.hist(subset, bins=bins, color=color, alpha=ALPHA_HIST, edgecolor="white",
                linewidth=0.3, label=f"Null ({n_label})", zorder=2)

        # real model markers — rug + vertical spans
        syn_lo, syn_hi = syn_top20.min(), syn_top20.max()
        red_lo, red_hi = red_top20.min(), red_top20.max()

        ax.axvspan(syn_lo, syn_hi, color=C_SYN, alpha=0.18, zorder=1,
                   label=f"Top-20 synergistic [{syn_lo:.3f}–{syn_hi:.3f}]")
        ax.axvspan(red_lo, red_hi, color=C_RED, alpha=0.18, zorder=1,
                   label=f"Top-20 redundant [{red_lo:.3f}–{red_hi:.3f}]")

        # rug ticks
        rug_y = ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 5
        for v in syn_top20:
            ax.plot(v, -rug_y, "|", color=C_SYN, markersize=8, markeredgewidth=1.2)
        for v in red_top20:
            ax.plot(v, -rug_y * 2.5, "|", color=C_RED, markersize=8, markeredgewidth=1.2)

        # percentile annotation
        p99 = np.percentile(subset, 99)
        ax.axvline(p99, ls="--", color="grey", lw=1, zorder=3)
        ax.text(p99, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 100,
                f" p99={p99:.3f}", fontsize=8, color="grey", va="top")

        # p-value annotation
        n_exceed = (subset >= syn_hi).sum()
        p_val = n_exceed / len(subset)
        p_str = f"p < 0.0001" if p_val < 1e-4 else f"p = {p_val:.4f}"
        ax.text(0.97, 0.95, f"P(null ≥ syn ceiling)\n{p_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

        ax.set_xlabel("Out-of-fold R²")
        if col == 0:
            ax.set_ylabel(f"{BAG_LABELS[bag]}\nCount")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# ── tidy ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Fig 4 – Null-model benchmark: random variable draws vs. O-information–guided models",
    fontsize=12, fontweight="bold", y=0.99
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig_null_model_benchmark.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / f'fig_null_model_benchmark.{ext}'}")

plt.close(fig)
print("Done.")
