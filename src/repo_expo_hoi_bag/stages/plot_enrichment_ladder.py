"""
Tier 2.1 — Threshold-free enrichment curve
Tier 2.2 — Complexity ladder ribbons with effect-size statistics

Layout: 2 rows (combined, functional) × 2 cols (enrichment, ladder)

Panel A (enrichment):
  For each percentile threshold P (1 → 50), what fraction of the
  top-P% models are synergistic?  50% = chance.  Bootstrap 95% CI.

Panel B (ladder):
  Top-20 syn vs top-20 red across 5 complexity rungs.
  Median + IQR ribbon.  Per-rung Cohen's d + permutation p annotated.

Reads:
  canonical/per_experiment/pooled_oinfo_ladder_{bag}/metrics_global_long.parquet

Writes:
  outputs/fig_enrichment_ladder.{png,pdf}
  stats/ladder_effect_sizes.csv
"""

from __future__ import annotations
import os
import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scripts.pipeline_utils import load_stage_config

warnings.filterwarnings("ignore", category=FutureWarning)

_COMMON_CFG = load_stage_config("common")

# ── paths ──────────────────────────────────────────────────────────────
OUTPUT_ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
DATA_ROOT = pathlib.Path(os.environ.get("V3_PLOT_DATA_ROOT", str(OUTPUT_ROOT)))
CANONICAL = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
if not CANONICAL.exists():
    CANONICAL = DATA_ROOT / "canonical" / "per_experiment"
FIG_DIR   = OUTPUT_ROOT / "figures";  FIG_DIR.mkdir(exist_ok=True)
STATS_DIR = OUTPUT_ROOT / "stats";    STATS_DIR.mkdir(exist_ok=True)

BAGS       = ["combined", "functional", "structural"]
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}
RUNG_ORDER = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
RUNG_LABELS = {"ols": "OLS",
               "xgb_tree_d1": "XGB d1", "xgb_tree_d2": "XGB d2",
               "xgb_tree_d3": "XGB d3"}

C_SYN = "#2166ac"
C_RED = "#b2182b"
RNG   = np.random.default_rng(int(_COMMON_CFG.get("bootstrap_rng_seed", 42)))


# ── helpers ──────────────────────────────────────────────────────────────
def enrichment_curve(r2: np.ndarray, is_syn: np.ndarray,
                     percentiles: np.ndarray,
                     n_boot: int = int(_COMMON_CFG.get("n_bootstrap", 2000))) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frac_syn, lo95, hi95) at each percentile threshold."""
    n = len(r2)
    idx_sorted = np.argsort(r2)[::-1]  # best first
    is_syn_sorted = is_syn[idx_sorted]

    fracs, lo, hi = [], [], []
    for p in percentiles:
        k = max(1, int(np.ceil(n * p / 100)))
        top_syn = is_syn_sorted[:k]
        frac = top_syn.mean()
        fracs.append(frac)
        # bootstrap CI
        boots = np.array([RNG.choice(top_syn, size=k, replace=True).mean()
                          for _ in range(n_boot)])
        lo.append(np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5))
    return np.array(fracs), np.array(lo), np.array(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) /
                     (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    """Two-sided permutation test for mean difference."""
    obs = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = RNG.permutation(combined)
        d = perm[:na].mean() - perm[na:].mean()
        if abs(d) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def select_best_rung(df: pd.DataFrame) -> str:
    candidate_rungs = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
    best_rung = max(
        candidate_rungs,
        key=lambda r: (
            df[(df["rung_id"] == r) & (df["objective"] == "o_min")]["full_r2"].max()
            if not df[(df["rung_id"] == r) & (df["objective"] == "o_min")].empty else -999.0
        ),
    )
    return best_rung


# ── build figure ─────────────────────────────────────────────────────────
if not CANONICAL.exists():
    print(f"No canonical root found: {CANONICAL}")
    print("No bags available — nothing to plot.")
    import sys; sys.exit(0)
available_bags = []
for bag in BAGS:
    bag_path = CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if bag_path.exists():
        available_bags.append(bag)
    else:
        print(f"  ⚠ {bag}: missing canonical metrics at {bag_path} — skipping")

if not available_bags:
    print("No bags available — nothing to plot.")
    import sys; sys.exit(0)

fig, axes = plt.subplots(len(available_bags), 2, figsize=(13, 4.5 * len(available_bags)))
if len(available_bags) == 1:
    axes = axes[np.newaxis, :]
ladder_rows = []

for row, bag in enumerate(available_bags):
    df = pd.read_parquet(CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet")
    is_syn = (df["objective"] == "o_min").values

    # ── Panel A: enrichment curve (selected rung) ──────────────────────────
    ax = axes[row, 0]
    selected_rung = select_best_rung(df)
    selected_rung_df = df[df["rung_id"] == selected_rung].copy()
    r2_vals = selected_rung_df["full_r2"].values
    syn_mask = (selected_rung_df["objective"] == "o_min").values

    pcts = np.arange(1, 51)  # top-1% to top-50%
    frac, lo, hi = enrichment_curve(r2_vals, syn_mask, pcts)

    ax.fill_between(pcts, lo * 100, hi * 100, color=C_SYN, alpha=0.15)
    ax.plot(pcts, frac * 100, color=C_SYN, lw=2, label="Synergistic fraction")
    ax.axhline(50, ls="--", color="grey", lw=1, label="Chance (50%)")
    ax.set_xlabel("Top-P% of models (by R²)", fontsize=10)
    ax.set_ylabel("% synergistic in top-P%", fontsize=10)
    ax.set_title(
        f"{BAG_LABELS[bag]} — Synergy enrichment ({RUNG_LABELS[selected_rung]})",
        fontsize=11,
    )
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")

    # Annotate top-5% and top-10%
    for p_ann in [5, 10, 20]:
        idx = np.searchsorted(pcts, p_ann)
        ax.annotate(f"top-{p_ann}%: {frac[idx]*100:.0f}%",
                    xy=(pcts[idx], frac[idx]*100),
                    xytext=(pcts[idx]+3, frac[idx]*100 + 5),
                    fontsize=8, color=C_SYN,
                    arrowprops=dict(arrowstyle="->", color=C_SYN, lw=0.8))

    # ── Panel B: complexity ladder (top-20 syn vs top-20 red) ────────
    ax = axes[row, 1]

    # Identify top-20 model IDs from selected rung for each group
    d2_syn = selected_rung_df[selected_rung_df["objective"] == "o_min"].nlargest(20, "full_r2")
    d2_red = selected_rung_df[selected_rung_df["objective"] == "o_max"].nlargest(20, "full_r2")
    syn_ids = set(d2_syn["candidate_id"])
    red_ids = set(d2_red["candidate_id"])

    x_pos = np.arange(len(RUNG_ORDER))
    syn_medians, syn_q25, syn_q75 = [], [], []
    red_medians, red_q25, red_q75 = [], [], []

    for rung in RUNG_ORDER:
        rdf = df[df["rung_id"] == rung]
        s = rdf[rdf["candidate_id"].isin(syn_ids)]["full_r2"].values
        r = rdf[rdf["candidate_id"].isin(red_ids)]["full_r2"].values

        # Clip extreme OLS negatives for display
        s = np.clip(s, -0.5, None)
        r = np.clip(r, -0.5, None)

        syn_medians.append(np.median(s)); syn_q25.append(np.percentile(s, 25)); syn_q75.append(np.percentile(s, 75))
        red_medians.append(np.median(r)); red_q25.append(np.percentile(r, 25)); red_q75.append(np.percentile(r, 75))

        # Stats
        d_val = cohens_d(s, r)
        p_val = permutation_p(s, r)
        ladder_rows.append({
            "bag": bag, "rung": rung,
            "syn_median": np.median(s), "red_median": np.median(r),
            "cohen_d": d_val, "perm_p": p_val,
            "n_syn": len(s), "n_red": len(r),
        })

        # Annotate d & p above the rung
        ymax = max(np.percentile(s, 75), np.percentile(r, 75))
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(RUNG_ORDER.index(rung), ymax + 0.025,
                f"d={d_val:+.2f} {star}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    syn_medians = np.array(syn_medians)
    syn_q25 = np.array(syn_q25)
    syn_q75 = np.array(syn_q75)
    red_medians = np.array(red_medians)
    red_q25 = np.array(red_q25)
    red_q75 = np.array(red_q75)

    ax.fill_between(x_pos, syn_q25, syn_q75, color=C_SYN, alpha=0.15)
    ax.plot(x_pos, syn_medians, "o-", color=C_SYN, lw=2, markersize=6, label="Top-20 synergistic")
    ax.fill_between(x_pos, red_q25, red_q75, color=C_RED, alpha=0.15)
    ax.plot(x_pos, red_medians, "s-", color=C_RED, lw=2, markersize=6, label="Top-20 redundant")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([RUNG_LABELS[r] for r in RUNG_ORDER], fontsize=9)
    ax.set_ylabel("Out-of-fold R²", fontsize=10)
    ax.set_xlabel("Model complexity →", fontsize=10)
    ax.set_title(f"{BAG_LABELS[bag]} — Complexity ladder", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    y_top = max(syn_q75.max(), red_q75.max()) + 0.08
    ax.set_ylim(bottom=-0.1, top=y_top)

# ── save ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Synergy enrichment & complexity ladder — top-20 models",
    fontsize=13, fontweight="bold", y=0.99
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig_enrichment_ladder.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / f'fig_enrichment_ladder.{ext}'}")
plt.close(fig)

# Save ladder stats
ladder_df = pd.DataFrame(ladder_rows)
ladder_df.to_csv(STATS_DIR / "ladder_effect_sizes.csv", index=False)
print(f"\nSaved {STATS_DIR / 'ladder_effect_sizes.csv'}")
print(ladder_df.to_string(index=False))
print("\nDone.")
