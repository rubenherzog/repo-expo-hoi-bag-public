"""
Bootstrap stability of the core signature.

For each BAG:
  1. Take the top-20 synergistic + top-20 redundant models (by R²).
  2. Bootstrap-resample (with replacement) 500×.
  3. For each bootstrap, count variable frequency across the 20 models.
  4. Report: mean frequency ± 90% CI per variable.

Output:
  outputs/fig_bootstrap_stability.{png,pdf}
  stats/bootstrap_variable_frequency.csv
"""

from __future__ import annotations
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

from exposome_labels import display_label
from scripts.exposome_domains import load_domain_map
from scripts.pipeline_utils import load_stage_config

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
EXPERIMENTS_ROOT = ROOT / "experiments"
if not EXPERIMENTS_ROOT.exists():
    EXPERIMENTS_ROOT = ROOT / "families" / "pooled_oinfo_ladder" / "experiments"
FIG_DIR   = ROOT / "figures";  FIG_DIR.mkdir(exist_ok=True)
STATS_DIR = ROOT / "stats";    STATS_DIR.mkdir(exist_ok=True)

BAGS = ["combined", "functional", "structural"]
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}

_BOOT_CFG   = load_stage_config("compute_bootstrap_stability")
_COMMON_CFG = load_stage_config("common")
N_BOOT = int(_BOOT_CFG.get("n_bootstrap", 500))
TOP_K  = int(_BOOT_CFG.get("top_k", 20))
RNG    = np.random.default_rng(int(_COMMON_CFG.get("bootstrap_rng_seed", 42)))

# ── domain map (for coloring) ────────────────────────────────────────────
DOMAIN_LABELS_CSV = pathlib.Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    pathlib.Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv",
))
DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)

DOMAIN_COLORS = {
    "Air Pollution": "#969696",
    "Temperature": "#d73027",
    "Precipitation/droughts": "#4393c3",
    "Green space access": "#1a9641",
    "Soil and water quality": "#a65628",
    "Climate disasters": "#ff7f00",
    "Disease-related mortality": "#f768a1",
    "Socioeconomic": "#e6ab02",
    "Democracy": "#7570b3",
    "Migration": "#74c476",
    "Other": "#cccccc",
}

def short(v: str) -> str:
    return display_label(v)


# ── bootstrap ────────────────────────────────────────────────────────────
def bootstrap_variable_frequency(
    var_lists: list[list[str]], n_boot: int, top_k: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    var_lists: list of variable lists (one per model, top_k models).
    Returns DataFrame with columns: variable, mean_freq, lo5, hi95.
    """
    all_vars = sorted({v for vl in var_lists for v in vl})
    boot_freqs = {v: [] for v in all_vars}

    for _ in range(n_boot):
        idx = rng.choice(len(var_lists), size=top_k, replace=True)
        counts = Counter()
        for i in idx:
            counts.update(var_lists[i])
        for v in all_vars:
            boot_freqs[v].append(counts[v] / top_k)

    rows = []
    for v in all_vars:
        arr = np.array(boot_freqs[v])
        rows.append({
            "variable": v,
            "mean_freq": arr.mean(),
            "lo5": np.percentile(arr, 5),
            "hi95": np.percentile(arr, 95),
            "domain": DOMAIN_MAP.get(v, "Other"),
        })
    return pd.DataFrame(rows).sort_values("mean_freq", ascending=False).reset_index(drop=True)


# ── main ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 21))
all_csv_rows = []

for row, bag in enumerate(BAGS):
    exp_path = EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / "xgb_tree_d2" / "xgb_summary_with_calibration_complexity.csv"
    if not exp_path.exists():
        print(f"  ⚠ {bag}: missing experiments — skipping")
        continue
    df = pd.read_csv(exp_path)

    for col, (obj, label, color_key) in enumerate([
        ("o_min", "Synergistic", "#2166ac"),
        ("o_max", "Redundant",   "#b2182b"),
    ]):
        sub = df[df["objective"] == obj].nlargest(TOP_K, "global_oof_r2")
        var_lists = [row_["predictors_identity"].split("|") for _, row_ in sub.iterrows()]

        boot_df = bootstrap_variable_frequency(var_lists, N_BOOT, TOP_K, RNG)
        boot_df["bag"] = bag
        boot_df["group"] = label
        all_csv_rows.append(boot_df)

        # Plot top-30 variables
        ax = axes[row, col]
        plot_df = boot_df.head(30).iloc[::-1]  # reverse for horizontal bar

        colors = [DOMAIN_COLORS.get(d, "#cccccc") for d in plot_df["domain"]]
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["mean_freq"], color=colors, alpha=0.75, height=0.7, zorder=2)
        ax.errorbar(plot_df["mean_freq"], y,
                     xerr=[plot_df["mean_freq"] - plot_df["lo5"],
                           plot_df["hi95"] - plot_df["mean_freq"]],
                     fmt="none", color="#333333", capsize=2, lw=0.8, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels([short(v) for v in plot_df["variable"]], fontsize=8)
        ax.set_xlabel("Frequency in top-20 (bootstrap mean ± 90% CI)", fontsize=9)
        ax.set_title(f"{BAG_LABELS[bag]} — {label} (top-30 vars)", fontsize=11)
        ax.axvline(0.5, ls=":", color="grey", lw=0.8)
        ax.set_xlim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

        # Mark "core" variables (freq > 0.7 with non-overlapping CI from 0.5)
        n_core = (boot_df["lo5"] > 0.5).sum()
        ax.text(0.97, 0.03, f"Core (lo5 > 0.5): {n_core} vars",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

# Domain legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=DOMAIN_COLORS[d], label=d, alpha=0.75)
                  for d in DOMAIN_COLORS if d != "Other"]
fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    f"Bootstrap stability of variable selection (top-{TOP_K} models, {N_BOOT} resamples)",
    fontsize=13, fontweight="bold", y=1.0
)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig_bootstrap_stability.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / f'fig_bootstrap_stability.{ext}'}")
plt.close(fig)

# Save CSV
out_df = pd.concat(all_csv_rows, ignore_index=True)
out_path = STATS_DIR / "bootstrap_variable_frequency.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved {out_path}")

# Print core signature summary
print("\n=== Core signature (lo5 > 0.5) ===")
for bag in BAGS:
    for grp in ["Synergistic", "Redundant"]:
        sub = out_df[(out_df["bag"] == bag) & (out_df["group"] == grp)]
        core = sub[sub["lo5"] > 0.5].sort_values("mean_freq", ascending=False)
        print(f"\n{bag} / {grp}: {len(core)} core variables")
        for _, r in core.iterrows():
            print(f"  {short(r['variable']):25s}  freq={r['mean_freq']:.2f}  "
                  f"[{r['lo5']:.2f}–{r['hi95']:.2f}]  ({r['domain']})")

print("\nDone.")
