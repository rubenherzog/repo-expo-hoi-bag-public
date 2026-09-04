"""
Fig 2 — Main results figure (composite).

Layout: 2 rows × 3 cols (one set, combined BAG primary)

  A  Synergy enrichment curve (threshold-free, replaces Fisher test)
  B  Complexity ladder (top-20 syn vs red, IQR ribbons, Cohen's d)
  C  Domain diversity × R² scatter (regression line, syn vs red)
  D  Bootstrap variable stability (top-20 syn, horizontal bars)
  E  Null distribution vs real models (histogram, diversity-filtered)
  F  Country meta-regression scatter

Generates one figure per BAG.

Reads existing stats/parquets; no new computation.
"""

from __future__ import annotations
import os
import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from exposome_labels import display_label
from scripts.exposome_domains import load_domain_map, unweighted_domain_stats
from scripts.pipeline_utils import load_stage_config

warnings.filterwarnings("ignore", category=FutureWarning)

_NULL_CFG   = load_stage_config("null_model")
_COMMON_CFG = load_stage_config("common")
_BOOT_CFG   = load_stage_config("compute_bootstrap_stability")

# ── paths ────────────────────────────────────────────────────────────────
OUTPUT_ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
DATA_ROOT = pathlib.Path(os.environ.get("V3_PLOT_DATA_ROOT", str(OUTPUT_ROOT)))
CANONICAL = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
if not CANONICAL.exists():
    CANONICAL = DATA_ROOT / "canonical" / "per_experiment"
EXPERIMENTS_ROOT = DATA_ROOT / "experiments"
if not EXPERIMENTS_ROOT.exists():
    EXPERIMENTS_ROOT = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "experiments"
FIG_DIR   = OUTPUT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
RAW_PATH  = pathlib.Path(os.environ.get(
    "V3_RAW_PATH",
    "data/"
    "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
))

BAGS = ["combined", "functional", "structural"]
BAG_LABELS = {"combined": "Combined BAG", "functional": "Functional BAG", "structural": "Structural BAG"}
BAG_COL    = {"combined": "BAG_comb", "functional": "BAG_func", "structural": "BAG_struc"}
RUNG_ORDER  = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
RUNG_LABELS = {"ols":"OLS", "xgb_tree_d1":"XGB d1",
               "xgb_tree_d2":"XGB d2", "xgb_tree_d3":"XGB d3"}
DIV_THRESH = float(_NULL_CFG.get("diversity_max_domain_frac", 0.60))
TOP_K  = int(_COMMON_CFG.get("top_k_per_order", 20))
N_BOOT = int(_BOOT_CFG.get("n_bootstrap", 500))
RNG    = np.random.default_rng(int(_COMMON_CFG.get("bootstrap_rng_seed", 42)))

C_SYN = "#2166ac"
C_RED = "#b2182b"
C_NULL = "#9ecae1"

# ── domain map ───────────────────────────────────────────────────────────
DOMAIN_LABELS_CSV = pathlib.Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    pathlib.Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv",
))
DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)

DOMAIN_COLORS = {
    "Air Pollution":"#969696", "Temperature":"#d73027", "Precipitation/droughts":"#4393c3",
    "Green space access":"#1a9641", "Soil and water quality":"#a65628",
    "Climate disasters":"#ff7f00", "Disease-related mortality":"#f768a1",
    "Socioeconomic":"#e6ab02", "Democracy":"#7570b3",
    "Migration":"#74c476", "Other":"#cccccc",
}

def short(v):
    return display_label(v)


def select_best_rung(metrics: pd.DataFrame) -> str:
    candidate_rungs = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
    best_rung = max(
        candidate_rungs,
        key=lambda r: (
            metrics[(metrics["rung_id"] == r) & (metrics["objective"] == "o_min")]["full_r2"].max()
            if not metrics[(metrics["rung_id"] == r) & (metrics["objective"] == "o_min")].empty else -999.0
        ),
    )
    return best_rung


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def permutation_p(a, b, n_perm=10000):
    obs = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = RNG.permutation(combined)
        if abs(perm[:na].mean() - perm[na:].mean()) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def domain_stats(preds_str):
    vl = [v.strip() for v in str(preds_str).split("|") if v.strip()]
    stats = unweighted_domain_stats(vl, domain_map=DOMAIN_MAP)
    return stats["shannon_h"], stats["n_domains"]


# ── load data ────────────────────────────────────────────────────────────
raw = pd.read_csv(RAW_PATH, low_memory=False)

# Sub-combination O-info enrichment (frac_neg per candidate per order_k)
# Used by panel A to replace objective-based synergy proxy with a principled metric.
SUBCOMB_ENRICHMENT_CSV = pathlib.Path(os.environ.get(
    "SUBCOMB_ENRICHMENT_CSV",
    "outputs/dedup/subcomb_oinfo/subcomb_enrichment.csv",
))
SUBCOMB_BASELINE_K3 = 0.3323  # fraction of all C(63,3) triplets with O-info < 0
SUBCOMB_BASELINE_K4 = 0.1901  # fraction of all C(63,4) quadruplets with O-info < 0
subcomb_enrich = None
if SUBCOMB_ENRICHMENT_CSV.exists():
    subcomb_enrich = pd.read_csv(SUBCOMB_ENRICHMENT_CSV)
else:
    print(f"  ⚠ subcomb enrichment CSV not found: {SUBCOMB_ENRICHMENT_CSV} — panel A will use legacy objective proxy")

boot_stats_path = DATA_ROOT / "stats" / "bootstrap_variable_frequency.csv"
if not boot_stats_path.exists():
    print(f"No bootstrap stats file found: {boot_stats_path}")
    print("Unable to generate Figure 2 without bootstrap frequency data.")
    import sys; sys.exit(0)
boot_csv = pd.read_csv(boot_stats_path)

available_bags = []
for bag in BAGS:
    metrics_path = CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    country_path = CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
    if not metrics_path.exists():
        print(f"  ⚠ {bag}: missing metrics file {metrics_path} — skipping")
        continue
    if not country_path.exists():
        print(f"  ⚠ {bag}: missing country metrics file {country_path} — skipping")
        continue
    available_bags.append(bag)

if not available_bags:
    print("No bags available — nothing to plot.")
    import sys; sys.exit(0)

for bag in available_bags:
    print(f"\n{'='*60}")
    print(f"  Generating Fig 2 — {bag.upper()}")
    print(f"{'='*60}")

    metrics = pd.read_parquet(CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet")
    selected_rung = select_best_rung(metrics)
    xgb_summary_path = (
        EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / selected_rung
        / "xgb_summary_with_calibration_complexity.csv"
    )
    if not xgb_summary_path.exists():
        print(f"  ⚠ {bag}: missing XGB summary {xgb_summary_path} — skipping")
        continue
    xgb_sum = pd.read_csv(xgb_summary_path)
    null_path = DATA_ROOT / f"null_model/{bag}/null_global.parquet"
    null_df = pd.read_parquet(null_path) if null_path.exists() else None
    country = pd.read_parquet(CANONICAL / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.35})

    # ── A: Synergistic triplet enrichment curve (principled) ─────────
    ax = axes[0, 0]
    selected_rung_df = metrics[metrics["rung_id"] == selected_rung].copy()
    pcts = np.arange(1, 51)

    if subcomb_enrich is not None:
        # Join frac_neg_k3 from enrichment CSV onto the rung's candidate pool.
        # Candidates of order=3 have no triplet sub-combinations; they get NaN
        # and are excluded from the running mean (conservative — they don't
        # inflate the o_max rate either since o_max order-3 candidates also have
        # no sub-combinations).
        enrich_bag = subcomb_enrich[
            (subcomb_enrich["bag"] == bag) &
            (subcomb_enrich["rung_id"] == selected_rung) &
            (subcomb_enrich["order_k"] == 3)
        ][["candidate_id", "frac_neg"]].drop_duplicates("candidate_id")
        enrich_bag = enrich_bag.rename(columns={"frac_neg": "frac_neg_k3"})

        rung_df = selected_rung_df.merge(enrich_bag, on="candidate_id", how="left")

        idx_sorted = np.argsort(rung_df["full_r2"].values)[::-1]
        frac_neg_sorted = rung_df["frac_neg_k3"].values[idx_sorted]
        obj_sorted = rung_df["objective"].values[idx_sorted]

        syn_fracs, syn_lo, syn_hi = [], [], []
        red_fracs, red_lo, red_hi = [], [], []
        for p in pcts:
            k = max(1, int(np.ceil(len(rung_df) * p / 100)))
            top_frac = frac_neg_sorted[:k]
            top_obj  = obj_sorted[:k]
            syn_vals = top_frac[top_obj == "o_min"]
            red_vals = top_frac[top_obj == "o_max"]
            syn_vals = syn_vals[~np.isnan(syn_vals.astype(float))]
            red_vals = red_vals[~np.isnan(red_vals.astype(float))]
            m_syn = syn_vals.mean() if len(syn_vals) else np.nan
            m_red = red_vals.mean() if len(red_vals) else np.nan
            syn_fracs.append(m_syn); red_fracs.append(m_red)
            # Bootstrap CI on the combined top-P pool
            all_vals = top_frac[~np.isnan(top_frac.astype(float))]
            if len(all_vals) >= 2:
                boots = np.array([RNG.choice(all_vals, size=len(all_vals), replace=True).mean()
                                  for _ in range(1000)])
                syn_lo.append(np.percentile(boots, 2.5))
                syn_hi.append(np.percentile(boots, 97.5))
            else:
                syn_lo.append(m_syn); syn_hi.append(m_syn)
            red_lo.append(np.nan); red_hi.append(np.nan)

        syn_fracs = np.array(syn_fracs, dtype=float) * 100
        red_fracs = np.array(red_fracs, dtype=float) * 100
        syn_lo_a  = np.array(syn_lo, dtype=float) * 100
        syn_hi_a  = np.array(syn_hi, dtype=float) * 100

        ax.axhline(SUBCOMB_BASELINE_K3 * 100, ls=":", color="#999", lw=1.2,
                   label=f"Exposome baseline ({SUBCOMB_BASELINE_K3*100:.1f}%)")
        ax.fill_between(pcts, syn_lo_a, syn_hi_a, color=C_SYN, alpha=0.12)
        ax.plot(pcts, syn_fracs, color=C_SYN, lw=2, label="o_min candidates")
        ax.plot(pcts, red_fracs, color=C_RED, lw=2, ls="--", label="o_max candidates")
        ax.set_ylabel("Mean % synergistic triplets (k=3)", fontsize=9)
        ax.set_ylim(0, 100)
        # Annotate at key thresholds
        for pa in [5, 10, 20]:
            idx = np.searchsorted(pcts, pa)
            v = syn_fracs[idx]
            if not np.isnan(v):
                ax.annotate(f"{v:.0f}%", xy=(pcts[idx], v),
                            fontsize=7, color=C_SYN, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(
            f"A  Synergistic triplet enrichment ({RUNG_LABELS[selected_rung]})",
            fontsize=11, fontweight="bold", loc="left",
        )
    else:
        # Fallback: legacy objective-based proxy
        r2_vals = selected_rung_df["full_r2"].values
        syn_mask = (selected_rung_df["objective"] == "o_min").values
        n = len(r2_vals)
        idx_sorted = np.argsort(r2_vals)[::-1]
        is_syn_sorted = syn_mask[idx_sorted]
        fracs, lo, hi = [], [], []
        for p in pcts:
            k = max(1, int(np.ceil(n * p / 100)))
            top = is_syn_sorted[:k]
            fracs.append(top.mean())
            boots = np.array([RNG.choice(top, size=k, replace=True).mean() for _ in range(2000)])
            lo.append(np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5))
        fracs, lo, hi = np.array(fracs), np.array(lo), np.array(hi)
        ax.fill_between(pcts, np.array(lo)*100, np.array(hi)*100, color=C_SYN, alpha=0.15)
        ax.plot(pcts, fracs*100, color=C_SYN, lw=2)
        ax.axhline(50, ls="--", color="grey", lw=1)
        ax.set_ylabel("% synergistic (objective proxy)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(
            f"A  Synergy enrichment — legacy ({RUNG_LABELS[selected_rung]})",
            fontsize=11, fontweight="bold", loc="left",
        )

    ax.set_xlabel("Top-P% of models (by R²)", fontsize=9)
    ax.set_xlim(1, 50)

    # ── B: Complexity ladder ─────────────────────────────────────────
    ax = axes[0, 1]
    selected_rung_syn_ids = set(selected_rung_df[selected_rung_df["objective"]=="o_min"].nlargest(TOP_K, "full_r2")["candidate_id"])
    selected_rung_red_ids = set(selected_rung_df[selected_rung_df["objective"]=="o_max"].nlargest(TOP_K, "full_r2")["candidate_id"])

    x_pos = np.arange(len(RUNG_ORDER))
    syn_meds, syn_q25, syn_q75 = [], [], []
    red_meds, red_q25, red_q75 = [], [], []
    rung_syn_vals, rung_red_vals = [], []

    for rung in RUNG_ORDER:
        r = metrics[metrics["rung_id"]==rung]
        s_vals = np.clip(r[r["candidate_id"].isin(selected_rung_syn_ids)]["full_r2"].values, -0.5, None)
        r_vals = np.clip(r[r["candidate_id"].isin(selected_rung_red_ids)]["full_r2"].values, -0.5, None)
        syn_meds.append(np.median(s_vals)); syn_q25.append(np.percentile(s_vals, 25)); syn_q75.append(np.percentile(s_vals, 75))
        red_meds.append(np.median(r_vals)); red_q25.append(np.percentile(r_vals, 25)); red_q75.append(np.percentile(r_vals, 75))
        rung_syn_vals.append(s_vals)
        rung_red_vals.append(r_vals)

    syn_meds, syn_q25, syn_q75 = np.array(syn_meds), np.array(syn_q25), np.array(syn_q75)
    red_meds, red_q25, red_q75 = np.array(red_meds), np.array(red_q25), np.array(red_q75)

    ax.fill_between(x_pos, syn_q25, syn_q75, color=C_SYN, alpha=0.12)
    ax.plot(x_pos, syn_meds, "o-", color=C_SYN, lw=2, ms=5, label="Top-20 synergistic")
    ax.fill_between(x_pos, red_q25, red_q75, color=C_RED, alpha=0.12)
    ax.plot(x_pos, red_meds, "s-", color=C_RED, lw=2, ms=5, label="Top-20 redundant")

    # Annotate Cohen's d and significance at each rung
    for i, rung in enumerate(RUNG_ORDER):
        d_val = cohens_d(rung_syn_vals[i], rung_red_vals[i])
        p_val = permutation_p(rung_syn_vals[i], rung_red_vals[i])
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ymax = max(syn_q75[i], red_q75[i])
        ax.text(i, ymax + 0.025,
                f"d={d_val:+.2f} {star}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([RUNG_LABELS[r] for r in RUNG_ORDER], fontsize=8)
    ax.set_ylabel("OOF R²", fontsize=9)
    y_top = max(syn_q75.max(), red_q75.max()) + 0.08
    ax.set_ylim(bottom=-0.15, top=y_top)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("B  Complexity ladder", fontsize=11, fontweight="bold", loc="left")

    # ── C: Domain diversity × R² scatter ─────────────────────────────
    ax = axes[0, 2]
    shan = xgb_sum["predictors_identity"].apply(lambda p: domain_stats(p)[0])
    xgb_sum_plot = xgb_sum.copy()
    xgb_sum_plot["shannon_h"] = shan
    xgb_sum_plot["is_syn"] = xgb_sum_plot["objective"] == "o_min"

    for obj, color, label in [("o_min", C_SYN, "Synergistic"), ("o_max", C_RED, "Redundant")]:
        sub = xgb_sum_plot[xgb_sum_plot["objective"]==obj]
        ax.scatter(sub["shannon_h"], sub["global_oof_r2"], c=color, s=12, alpha=0.4, label=label)

    # Regression lines
    import statsmodels.api as sm
    for obj, color in [("o_min", C_SYN), ("o_max", C_RED)]:
        sub = xgb_sum_plot[xgb_sum_plot["objective"]==obj]
        X = sm.add_constant(sub["shannon_h"])
        m = sm.OLS(sub["global_oof_r2"], X).fit()
        xs = np.linspace(sub["shannon_h"].min(), sub["shannon_h"].max(), 50)
        ax.plot(xs, m.predict(sm.add_constant(xs)), color=color, lw=1.5, ls="--")

    ax.set_xlabel("Shannon H (domain diversity)", fontsize=9)
    ax.set_ylabel("OOF R²", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("C  Domain diversity drives R²", fontsize=11, fontweight="bold", loc="left")

    # ── D: Bootstrap variable stability ──────────────────────────────
    ax = axes[1, 0]
    bdf = boot_csv[(boot_csv["bag"]==bag) & (boot_csv["group"]=="Synergistic")].copy()
    bdf = bdf.sort_values("mean_freq", ascending=True).tail(20)
    y = np.arange(len(bdf))
    colors = [DOMAIN_COLORS.get(DOMAIN_MAP.get(v, "Other"), "#ccc") for v in bdf["variable"]]
    ax.barh(y, bdf["mean_freq"], color=colors, alpha=0.75, height=0.7)
    ax.errorbar(bdf["mean_freq"], y,
                xerr=[bdf["mean_freq"]-bdf["lo5"], bdf["hi95"]-bdf["mean_freq"]],
                fmt="none", color="#333", capsize=2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short(v) for v in bdf["variable"]], fontsize=7)
    ax.set_xlabel("Frequency in top-20 (90% CI)", fontsize=9)
    ax.axvline(0.5, ls=":", color="grey", lw=0.8)
    ax.set_xlim(0, 1.05)
    n_core = (bdf["lo5"] > 0.5).sum()
    ax.text(0.97, 0.03, f"Core vars: {n_core}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.set_title("D  Bootstrap stability (synergistic)", fontsize=11, fontweight="bold", loc="left")

    # ── E: Null distribution ─────────────────────────────────────────
    ax = axes[1, 1]
    if null_df is not None:
        null_div = null_df.loc[null_df["max_domain_frac"] <= DIV_THRESH, "global_oof_r2"]
        syn_top20 = xgb_sum[xgb_sum["objective"]=="o_min"].nlargest(TOP_K, "global_oof_r2")["global_oof_r2"]
        red_top20 = xgb_sum[xgb_sum["objective"]=="o_max"].nlargest(TOP_K, "global_oof_r2")["global_oof_r2"]

        bins = np.linspace(null_div.min()-0.01, max(null_div.max(), syn_top20.max())+0.01, 50)
        ax.hist(null_div, bins=bins, color=C_NULL, alpha=0.7, edgecolor="white", lw=0.3,
                label=f"Diverse null (N={len(null_div):,})")
        ax.axvspan(syn_top20.min(), syn_top20.max(), color=C_SYN, alpha=0.18,
                   label=f"Top-20 syn [{syn_top20.min():.3f}–{syn_top20.max():.3f}]")
        ax.axvspan(red_top20.min(), red_top20.max(), color=C_RED, alpha=0.18,
                   label=f"Top-20 red [{red_top20.min():.3f}–{red_top20.max():.3f}]")

        p99 = np.percentile(null_div, 99)
        ax.axvline(p99, ls="--", color="grey", lw=1)
        ax.text(p99, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 50,
                f" p99={p99:.3f}", fontsize=7, color="grey")

        n_exceed = (null_div >= syn_top20.max()).sum()
        p_val = n_exceed / len(null_div)
        p_str = "p < 0.0001" if p_val < 1e-4 else f"p = {p_val:.4f}"
        ax.text(0.97, 0.95, f"P(null ≥ syn ceiling)\n{p_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xlabel("OOF R²", fontsize=9)
        ax.set_title("E  Null benchmark (diverse filter)", fontsize=11, fontweight="bold", loc="left")
    else:
        ax.text(0.5, 0.5, "Null model\nnot yet computed", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="grey")
        ax.set_title("E  Null benchmark (pending)", fontsize=11, fontweight="bold", loc="left")

    # ── F: Country meta-regression scatter ───────────────────────────
    ax = axes[1, 2]
    selected_rung_country = country[country["rung_id"] == selected_rung]
    best_syn_id = selected_rung_country[selected_rung_country["objective"]=="o_min"].groupby("candidate_id")["country_full_r2"].mean().idxmax()
    best_data = selected_rung_country[selected_rung_country["candidate_id"]==best_syn_id][["fold_country","country_full_r2","n_test_scored"]].copy()
    best_data["log_n"] = np.log(best_data["n_test_scored"])

    # BAG variance per country
    bcol = BAG_COL[bag]
    bag_var = raw[raw[bcol].notna()].groupby("country_clean")[bcol].std().reset_index()
    bag_var.columns = ["fold_country", "bag_sd"]
    best_data = best_data.merge(bag_var, on="fold_country", how="left").dropna(subset=["bag_sd"])

    sc = ax.scatter(best_data["log_n"], best_data["country_full_r2"],
                    c=best_data["bag_sd"], cmap="coolwarm", s=50, edgecolors="white", lw=0.5)
    for _, r in best_data.iterrows():
        ax.annotate(r["fold_country"], (r["log_n"], r["country_full_r2"]),
                    fontsize=5.5, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.axhline(0, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("log(n_test)", fontsize=9)
    ax.set_ylabel("Country R²", fontsize=9)
    plt.colorbar(sc, ax=ax, label="BAG SD", shrink=0.8, pad=0.02)
    ax.set_title("F  Country performance drivers", fontsize=11, fontweight="bold", loc="left")

    # ── save fig2 ────────────────────────────────────────────────────
    fig.suptitle(f"Figure 2 — {BAG_LABELS[bag]}: Multi-domain synergy predicts brain-age gap",
                 fontsize=14, fontweight="bold", y=1.01)

    for ext in ("png", "pdf"):
        out = FIG_DIR / f"fig2_main_results_{bag}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)

    # ── Supplementary: k=4 enrichment curve (standalone) ─────────────
    if subcomb_enrich is not None:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 4))

        enrich_bag_k4 = subcomb_enrich[
            (subcomb_enrich["bag"] == bag) &
            (subcomb_enrich["rung_id"] == selected_rung) &
            (subcomb_enrich["order_k"] == 4)
        ][["candidate_id", "frac_neg"]].drop_duplicates("candidate_id")
        enrich_bag_k4 = enrich_bag_k4.rename(columns={"frac_neg": "frac_neg_k4"})

        rung_df_k4 = selected_rung_df.merge(enrich_bag_k4, on="candidate_id", how="left")
        idx_s = np.argsort(rung_df_k4["full_r2"].values)[::-1]
        frac4_sorted = rung_df_k4["frac_neg_k4"].values[idx_s]
        obj4_sorted  = rung_df_k4["objective"].values[idx_s]

        syn4_fracs, red4_fracs = [], []
        for p in pcts:
            k = max(1, int(np.ceil(len(rung_df_k4) * p / 100)))
            top_f = frac4_sorted[:k]
            top_o = obj4_sorted[:k]
            sv = top_f[top_o == "o_min"]; sv = sv[~np.isnan(sv.astype(float))]
            rv = top_f[top_o == "o_max"]; rv = rv[~np.isnan(rv.astype(float))]
            syn4_fracs.append(sv.mean() if len(sv) else np.nan)
            red4_fracs.append(rv.mean() if len(rv) else np.nan)

        syn4_fracs = np.array(syn4_fracs, dtype=float) * 100
        red4_fracs = np.array(red4_fracs, dtype=float) * 100

        ax_s.axhline(SUBCOMB_BASELINE_K4 * 100, ls=":", color="#999", lw=1.2,
                     label=f"Exposome baseline ({SUBCOMB_BASELINE_K4*100:.1f}%)")
        ax_s.plot(pcts, syn4_fracs, color=C_SYN, lw=2, label="o_min candidates")
        ax_s.plot(pcts, red4_fracs, color=C_RED, lw=2, ls="--", label="o_max candidates")
        ax_s.set_xlabel("Top-P% of models (by R²)", fontsize=9)
        ax_s.set_ylabel("Mean % synergistic quadruplets (k=4)", fontsize=9)
        ax_s.set_xlim(1, 50); ax_s.set_ylim(0, 100)
        ax_s.legend(fontsize=8, loc="lower right")
        ax_s.set_title(
            f"Supplementary — Quadruplet enrichment, {BAG_LABELS[bag]} ({RUNG_LABELS[selected_rung]})",
            fontsize=10, fontweight="bold",
        )

        for ext in ("png", "pdf", "svg"):
            out_s = FIG_DIR / f"fig2_subcomb_k4_{bag}.{ext}"
            fig_s.savefig(out_s, dpi=300, bbox_inches="tight")
            print(f"  Saved {out_s}")
        plt.close(fig_s)

print("\nDone.")
