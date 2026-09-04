#!/usr/bin/env python3
"""
make_boss_figure.py
===================
Generate a 6-panel "boss-convincing" figure summarising the exposome–BAG
synergy story.

Panels
------
A  The ceiling belongs to synergy    (scatter: O-info vs metric)
B  Synergistic sets are more efficient (scatter: order vs metric)
C  The recipe: multi-domain composition (horizontal stacked bars)
D  The 7-variable core signature      (binary heatmap of top models)
E  Interactions unlock the signal      (complexity ladder lines)
F  Geographic consistency              (country lollipop/forest plot)

Generates 4 figures (all combinations):
  - combined × R²     combined × f²
  - functional × R²   functional × f²

All data are read from disk; nothing is hardcoded.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import re
from scipy.stats import fisher_exact
from scipy import stats
import re

from exposome_labels import display_label

# ── paths ────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
DATA_ROOT = Path(os.environ.get("V3_PLOT_DATA_ROOT", str(OUTPUT_ROOT)))
CANONICAL_ROOT = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
if not CANONICAL_ROOT.exists():
    CANONICAL_ROOT = DATA_ROOT / "canonical" / "per_experiment"
EXPERIMENTS_ROOT = DATA_ROOT / "families" / "pooled_oinfo_ladder" / "experiments"
if not EXPERIMENTS_ROOT.exists():
    EXPERIMENTS_ROOT = DATA_ROOT / "experiments"
FIGURE_OUT = OUTPUT_ROOT / "figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

DPI = 220
RNG = np.random.default_rng(20260402)
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

# ── domain classification ───────────────────────────────────────────────────
# Taxonomy follows the explicit feature-domain input used by the domain-greedy
# search, so figure entropy and search entropy use the same partition.
DOMAIN_LABELS_CSV = Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    str(Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv"),
))
DOMAIN_DF = pd.read_csv(DOMAIN_LABELS_CSV)
if not {"feature_name", "domain"}.issubset(DOMAIN_DF.columns):
    raise ValueError(f"Domain labels CSV must contain feature_name and domain columns: {DOMAIN_LABELS_CSV}")
DOMAIN_MAP: dict[str, str] = dict(
    zip(DOMAIN_DF["feature_name"].astype(str), DOMAIN_DF["domain"].astype(str))
)

DOMAIN_ORDER = [
    "Air Pollution",
    "Temperature",
    "Precipitation/droughts",
    "Green space access",
    "Soil and water quality",
    "Climate disasters",
    "Disease-related mortality",
    "Socioeconomic",
    "Democracy",
    "Migration",
    "Other",
]
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

def _short(v: str) -> str:
    return display_label(v)

RUNG_ORDER = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
RUNG_LABELS = {
    "ols": "OLS",
    "xgb_tree_d1": "XGB\nd=1",
    "xgb_tree_d2": "XGB\nd=2",
    "xgb_tree_d3": "XGB\nd=3",
}
RUNG_COLORS = {
    "ols": "#2c7fb8",
    "xgb_tree_d1": "#fdbb84",
    "xgb_tree_d2": "#fc8d59",
    "xgb_tree_d3": "#d7301f",
}

OBJ_COLORS = {"o_min": "#2166ac", "o_max": "#b2182b"}
OBJ_LABELS = {"o_min": "Synergy-optimised (o_min)", "o_max": "Redundancy-optimised (o_max)"}

# ── helpers ──────────────────────────────────────────────────────────────────

def _compute_f2(full, base):
    full_arr = np.asarray(full, dtype=float)
    base_arr = np.asarray(base, dtype=float)
    if base_arr.ndim == 0:
        base_arr = np.full(full_arr.shape, base_arr)
    denom = 1.0 - full_arr
    out = np.full(len(full_arr), np.nan)
    ok = np.isfinite(full_arr) & np.isfinite(base_arr) & (np.abs(denom) > 1e-12)
    out[ok] = (full_arr[ok] - base_arr[ok]) / denom[ok]
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.nanvar(a, ddof=1) + (nb - 1) * np.nanvar(b, ddof=1)) / (na + nb - 2))
    if pooled <= 0:
        return np.nan
    return float((np.nanmean(a) - np.nanmean(b)) / pooled)


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if len(a[valid]) + len(b[valid]) == 0:
        return np.nan
    combined = np.concatenate([a, b])
    na = len(a)
    if na == 0 or len(b) == 0:
        return np.nan
    obs = np.nanmean(a) - np.nanmean(b)
    count = 0
    for _ in range(n_perm):
        perm = np.random.default_rng(20260402).permutation(combined)
        if (np.nanmean(perm[:na]) - np.nanmean(perm[na:])) >= abs(obs):
            count += 1
    return float(count / n_perm)


def _classify(v: str) -> str:
    return DOMAIN_MAP.get(v.strip(), "Other")


def _annotate(ax, text, x=0.03, y=0.97, fontsize=8, ha="left", va="top"):
    ax.annotate(
        text, xy=(x, y), xycoords="axes fraction",
        fontsize=fontsize, ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"),
    )


def _order_from_candidate_id(candidate_id: str) -> int:
    m = re.search(r"_ord(\d+)", candidate_id)
    if m:
        return int(m.group(1))
    m = re.search(r"_order(\d+)", candidate_id)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse order from candidate_id={candidate_id}")


def _load_data(bag: str, rung_id: str = "xgb_tree_d2") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (global_d2, country_d2, global_all_rungs) for a BAG target.

    By default this loads xgb_tree_d2, but callers can request another rung.
    """
    exp_dir = f"pooled_oinfo_ladder_{bag}"
    mg = pd.read_parquet(CANONICAL_ROOT / exp_dir / "metrics_global_long.parquet")
    mc = pd.read_parquet(CANONICAL_ROOT / exp_dir / "metrics_country_long.parquet")

    # Add f2 to global
    mg["f2"] = _compute_f2(mg["full_r2"].values, mg["base_r2"].values)

    # rung-specific slice
    d2 = mg[mg["rung_id"] == rung_id].copy()

    # Country d2
    d2c = mc[mc["rung_id"] == rung_id].copy()
    d2c["country_f2"] = _compute_f2(
        d2c["country_full_r2"].values, d2c["country_base_r2"].values
    )

    return d2, d2c, mg


BEST_XGB_RUNGS = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]


def select_best_rung(metrics: pd.DataFrame) -> str:
    """Select the best XGB rung using o_min full R²."""
    candidate_rungs = BEST_XGB_RUNGS
    best_rung = max(
        candidate_rungs,
        key=lambda r: (
            metrics[(metrics["rung_id"] == r) & (metrics["objective"] == "o_min")]["full_r2"].max()
            if not metrics[(metrics["rung_id"] == r) & (metrics["objective"] == "o_min")].empty else -999.0
        ),
    )
    return best_rung


def _load_xgb_summary(bag: str, rung_id: str = "xgb_tree_d2") -> pd.DataFrame:
    """Load the full XGB summary with variable identity for a chosen rung."""
    p = EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / rung_id / \
        "xgb_summary_with_calibration_complexity.csv"
    df = pd.read_csv(p)
    df["vars_list"] = df["predictors_identity"].str.split("|")
    # synergistic = greedy discovery objective (o_min), not the sign of evaluated O-info
    # (consistent with panel_b at line ~285 which uses objective == "o_min").
    df["is_syn"] = df["objective"].astype(str) == "o_min"
    return df


# ── panels ───────────────────────────────────────────────────────────────────

def panel_a_ceiling(ax, d2: pd.DataFrame, metric: str, metric_label: str):
    """Scatter: O-info vs metric, coloured by objective."""
    d2 = d2.copy()
    if "thoi_o" not in d2.columns:
        d2["thoi_o"] = d2["score"]
    else:
        d2["thoi_o"] = d2["thoi_o"].fillna(d2["score"])

    for obj in ["o_max", "o_min"]:
        subset = d2[d2["objective"] == obj]
        ax.scatter(
            subset["thoi_o"], subset[metric],
            c=OBJ_COLORS[obj], s=14, alpha=0.45,
            edgecolors="none", label=OBJ_LABELS[obj], zorder=2,
        )

    # Find threshold for top-model annotation
    best_syn = d2[d2["objective"] == "o_min"]
    best_red = d2[d2["objective"] == "o_max"]
    # Determine a nice ceiling threshold
    all_vals = d2[metric].dropna()
    q95 = all_vals.quantile(0.95)
    above = d2[d2[metric] >= q95]
    n_above = len(above)
    n_syn_above = (above["objective"] == "o_min").sum()
    pct = 100 * n_syn_above / n_above if n_above > 0 else 0

    # Draw horizontal line at q95
    ax.axhline(q95, color="#888888", ls="--", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", ls=":", lw=0.8, zorder=1)

    # Reduce top white space by tightening y-limits after plotting
    all_y = all_vals.values
    if len(all_y):
        ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
        pad = max(0.03 * (ymax - ymin), 0.01)
        ax.set_ylim(ymin - pad, ymax + pad)

    is_syn = d2["objective"] == "o_min"
    is_top = d2[metric] >= q95
    a = (is_syn & is_top).sum()
    b = (~is_syn & is_top).sum()
    c = (is_syn & ~is_top).sum()
    dv = (~is_syn & ~is_top).sum()
    odds, p_fisher = fisher_exact([[a, b], [c, dv]], alternative="greater")

    ax.set_xlabel("O-information", fontsize=TICK_FONTSIZE)
    ax.set_ylabel(metric_label, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_title("A  Synergy dominates the ceiling", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")
    ax.legend(fontsize=LEGEND_FONTSIZE + 2, loc="lower left", framealpha=0.8, edgecolor="none")


def panel_b_parsimony(ax, d2: pd.DataFrame, metric: str, metric_label: str):
    """Scatter: order (n variables) vs metric, coloured by objective."""
    d2 = d2.copy()
    if "thoi_o" not in d2.columns:
        d2["thoi_o"] = d2["score"]
    else:
        d2["thoi_o"] = d2["thoi_o"].fillna(d2["score"])

    if "order" not in d2.columns:
        if "candidate_id" in d2.columns:
            d2["order"] = d2["candidate_id"].map(_order_from_candidate_id)
        else:
            raise ValueError("Cannot derive order because candidate_id is missing")
    else:
        if "candidate_id" in d2.columns:
            d2["order"] = d2["order"].fillna(d2["candidate_id"].map(_order_from_candidate_id))
    d2["order"] = d2["order"].astype(int)

    for obj in ["o_max", "o_min"]:
        subset = d2[d2["objective"] == obj]
        ax.scatter(
            subset["order"], subset[metric],
            c=OBJ_COLORS[obj], s=14, alpha=0.45,
            edgecolors="none", label=OBJ_LABELS[obj], zorder=2,
        )

    # Pareto frontier per objective
    for obj, color in OBJ_COLORS.items():
        sub = d2[d2["objective"] == obj].sort_values("order")
        if sub.empty:
            continue
        orders = sorted(sub["order"].unique())
        frontier_x, frontier_y = [], []
        running_max = -np.inf
        for o in orders:
            best_at_o = sub.loc[sub["order"] == o, metric].max()
            if best_at_o > running_max:
                running_max = best_at_o
            frontier_x.append(o)
            frontier_y.append(running_max)
        ax.plot(frontier_x, frontier_y, color=color, lw=1.5, alpha=0.7, zorder=3)

    # Efficiency annotation
    syn = d2[d2["objective"] == "o_min"]
    red = d2[d2["objective"] == "o_max"]
    eff_syn = (syn[metric] / syn["order"]).median()
    eff_red = (red[metric] / red["order"]).median()
    ratio = eff_syn / eff_red if eff_red > 0 else np.nan

    ax.set_xlabel("Number of variables (order)", fontsize=TICK_FONTSIZE)
    ax.set_ylabel(metric_label, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_title("B  Synergistic sets are more parsimonious", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")


def panel_c_domains(
    ax,
    xgb_summary: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    mg_all: pd.DataFrame | None = None,
    best_rung: str | None = None,
):
    """Horizontal stacked bar: domain composition of 3 model tiers."""
    df = xgb_summary.copy()
    if "candidate_id" not in df.columns and "model_id" in df.columns:
        df["candidate_id"] = df["model_id"]

    def _select_ids(source: pd.DataFrame, objective: str, how: str) -> list[str]:
        src = source[(source["objective"] == objective)]
        if src.empty:
            return []
        if how == "best":
            return src.nlargest(20, "full_r2")["candidate_id"].astype(str).tolist()
        return src.nsmallest(20, "full_r2")["candidate_id"].astype(str).tolist()

    if mg_all is not None and best_rung is not None:
        source = mg_all[(mg_all["rung_id"] == best_rung)].copy()
    else:
        source = df.copy()

    best_syn_ids = _select_ids(source, "o_min", "best")
    worst_syn_ids = _select_ids(source, "o_min", "worst")
    best_red_ids = _select_ids(source, "o_max", "best")

    def _prepare_group(ids: list[str]) -> pd.DataFrame:
        grp = df[df["candidate_id"].astype(str).isin([str(x) for x in ids])].copy()
        if "vars_list" not in grp.columns:
            if "predictors_identity" in grp.columns:
                grp["vars_list"] = grp["predictors_identity"].fillna("").str.split("|")
            else:
                grp["vars_list"] = [[] for _ in range(len(grp))]
        return grp

    best_syn = _prepare_group(best_syn_ids)
    worst_syn = _prepare_group(worst_syn_ids)
    best_red = _prepare_group(best_red_ids)

    tiers = [
        ("Best synergistic\n(top 20)", best_syn),
        ("Worst synergistic\n(bottom 20)", worst_syn),
        ("Best redundant\n(top 20)", best_red),
    ]

    positions = np.arange(len(tiers))
    for i, (label, grp) in enumerate(tiers):
        domain_counts = Counter()
        total = 0
        for vl in grp["vars_list"]:
            if not isinstance(vl, list):
                continue
            for v in vl:
                v2 = v.strip()
                if v2:
                    domain_counts[_classify(v2)] += 1
                    total += 1

        left = 0
        for dom in DOMAIN_ORDER:
            pct = domain_counts.get(dom, 0) / total if total else 0
            if pct > 0:
                ax.barh(i, pct, left=left, height=0.6,
                        color=DOMAIN_COLORS[dom], edgecolor="white", linewidth=0.3)
            left += pct

    ax.set_yticks(positions)
    ax.set_yticklabels([t[0] for t in tiers], fontsize=TICK_FONTSIZE - 2)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE - 2)
    ax.invert_yaxis()


def panel_d_heatmap(ax, xgb_summary: pd.DataFrame, metric_col: str, metric_label: str):
    """Binary heatmap comparing strong and weak synergistic model recipes."""
    df = xgb_summary.copy()
    omin_syn = df[(df["objective"] == "o_min") & (df["is_syn"])]
    top_n = 20
    best = omin_syn.nlargest(top_n, metric_col).copy()
    worst = omin_syn.nsmallest(top_n, metric_col).copy()

    freq_best = Counter()
    freq_worst = Counter()
    for vl in best["vars_list"]:
        for v in vl:
            freq_best[v.strip()] += 1
    for vl in worst["vars_list"]:
        for v in vl:
            freq_worst[v.strip()] += 1

    # Choose variables to show: those that appear most often across best/worst
    combined = Counter()
    combined.update(freq_best)
    combined.update(freq_worst)
    show_vars = [v for v, _ in combined.most_common(50)]
    short_names = [_short(v) for v in show_vars]
    domain_cols = [DOMAIN_COLORS.get(_classify(v), "#333333") for v in show_vars]

    heat = np.zeros((top_n * 2, len(show_vars)), dtype=int)
    for i, vl in enumerate(best["vars_list"]):
        for v in vl:
            v2 = v.strip()
            if v2 in show_vars:
                heat[i, show_vars.index(v2)] = 1
    for i, vl in enumerate(worst["vars_list"]):
        for v in vl:
            v2 = v.strip()
            if v2 in show_vars:
                heat[top_n + i, show_vars.index(v2)] = 1

    cmap = mcolors.ListedColormap(["#f0f0f0", "#2166ac"])  # background, included
    ax.imshow(heat, aspect="auto", interpolation="none", cmap=cmap)

    # Add a divider between best and worst models
    ax.axhline(top_n - 0.5, color="#444444", lw=1.5, ls="--")

    ax.set_xticks(range(len(show_vars)))
    ax.set_xticklabels(short_names, rotation=50, ha="right", fontsize=8)
    # Colour the x-tick labels by domain
    for tick_label, col in zip(ax.get_xticklabels(), domain_cols):
        tick_label.set_color(col)
        tick_label.set_fontweight("bold")

    # Label y-axis by rank with metric value for both blocks
    best_metric_vals = best[metric_col].values
    worst_metric_vals = worst[metric_col].values
    ax.set_yticks([0, top_n - 1, top_n, 2 * top_n - 1])
    ax.set_yticklabels(
        [
            f"Best #1 ({best_metric_vals[0]:.3f})",
            f"Best #{top_n} ({best_metric_vals[-1]:.3f})",
            f"Worst #1 ({worst_metric_vals[0]:.3f})",
            f"Worst #{top_n} ({worst_metric_vals[-1]:.3f})",
        ],
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=8)


def panel_e_ladder(ax, mg_all: pd.DataFrame, metric: str, metric_label: str, rung_id: str = "xgb_tree_d2"):
    """Complexity ladder: OLS → d3 for top models with confidence intervals."""
    d2 = mg_all[mg_all["rung_id"] == rung_id].copy()
    best_syn_ids = d2[d2["objective"] == "o_min"].nlargest(20, "full_r2")["candidate_id"].astype(str).tolist()
    best_red_ids = d2[d2["objective"] == "o_max"].nlargest(20, "full_r2")["candidate_id"].astype(str).tolist()

    syn_medians, syn_q25, syn_q75 = [], [], []
    red_medians, red_q25, red_q75 = [], [], []
    x_pos = np.arange(len(RUNG_ORDER))

    for rung in RUNG_ORDER:
        rung_df = mg_all[mg_all["rung_id"] == rung]
        syn_vals = rung_df[rung_df["candidate_id"].astype(str).isin(best_syn_ids)][metric].dropna().values
        red_vals = rung_df[rung_df["candidate_id"].astype(str).isin(best_red_ids)][metric].dropna().values

        if len(syn_vals) > 0:
            syn_medians.append(np.median(syn_vals))
            syn_q25.append(np.percentile(syn_vals, 25))
            syn_q75.append(np.percentile(syn_vals, 75))
        else:
            syn_medians.append(np.nan)
            syn_q25.append(np.nan)
            syn_q75.append(np.nan)

        if len(red_vals) > 0:
            red_medians.append(np.median(red_vals))
            red_q25.append(np.percentile(red_vals, 25))
            red_q75.append(np.percentile(red_vals, 75))
        else:
            red_medians.append(np.nan)
            red_q25.append(np.nan)
            red_q75.append(np.nan)

        if len(syn_vals) > 1 and len(red_vals) > 1:
            d_val = cohens_d(syn_vals, red_vals)
            y_text = max(np.nanmax(syn_q75), np.nanmax(red_q75)) if len(syn_vals) and len(red_vals) else 0
            ax.text(
                x_pos[RUNG_ORDER.index(rung)],
                y_text + 0.02,
                f"d={d_val:+.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.fill_between(x_pos, syn_q25, syn_q75, color=OBJ_COLORS["o_min"], alpha=0.15)
    ax.plot(x_pos, syn_medians, "o-", color=OBJ_COLORS["o_min"], lw=2, markersize=6, label="Top-20 synergistic")
    ax.fill_between(x_pos, red_q25, red_q75, color=OBJ_COLORS["o_max"], alpha=0.15)
    ax.plot(x_pos, red_medians, "s-", color=OBJ_COLORS["o_max"], lw=2, markersize=6, label="Top-20 redundant")

    ax.axhline(0, color="#cccccc", ls=":", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([RUNG_LABELS[r] for r in RUNG_ORDER], fontsize=TICK_FONTSIZE - 2)
    ax.set_xlabel("Model complexity", fontsize=TICK_FONTSIZE)
    ax.set_ylabel(metric_label, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE - 2)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower right", framealpha=0.8, edgecolor="none")

    all_y = [v for v in syn_q25 + syn_medians + syn_q75 + red_q25 + red_medians + red_q75 if np.isfinite(v)]
    if all_y:
        raw_ymin = min(all_y)
        raw_ymax = max(all_y)
        lower_limit = -0.5
        if raw_ymin < lower_limit:
            ax.axhline(lower_limit, color="#d7301f", ls="--", lw=1.0, zorder=2)
            ax.text(
                0.02, 0.02,
                "Values below -0.5 are clipped",
                transform=ax.transAxes,
                fontsize=TICK_FONTSIZE - 2,
                color="#d7301f",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
            )
            ymin = lower_limit
        else:
            ymin = raw_ymin - max(0.05 * (raw_ymax - raw_ymin), 0.05)
        pad = max(0.05 * (raw_ymax - ymin), 0.05)
        ax.set_ylim(ymin, raw_ymax + pad)


def panel_shannon_entropy(ax, xgb_summary: pd.DataFrame, metric_col: str = "global_oof_r2"):
    """Scatter: Shannon H (domain diversity) vs model metric, separate by objective."""
    # compute shannon H per predictor identity
    def _shannon(predictors_identity: str) -> float:
        if pd.isna(predictors_identity) or predictors_identity == "":
            return 0.0
        vl = [v.strip() for v in predictors_identity.split("|") if v.strip()]
        doms = [_classify(v) for v in vl]
        cnt = Counter(doms)
        total = sum(cnt.values())
        if total == 0:
            return 0.0
        fracs = [c / total for c in cnt.values() if c > 0]
        return -sum(f * np.log2(f) for f in fracs)

    xgb_summary = xgb_summary.copy()
    xgb_summary["shannon_h"] = xgb_summary["predictors_identity"].apply(_shannon)

    for obj in ["o_min", "o_max"]:
        sub = xgb_summary[xgb_summary["objective"] == obj]
        ax.scatter(sub["shannon_h"], sub[metric_col], c=OBJ_COLORS[obj], s=14, alpha=0.45, edgecolors="none",
                   label=OBJ_LABELS[obj])

        if len(sub) >= 2:
            lr = stats.linregress(sub["shannon_h"], sub[metric_col])
            xs = np.linspace(sub["shannon_h"].min(), sub["shannon_h"].max(), 50)
            ys = lr.intercept + lr.slope * xs
            ax.plot(xs, ys, color=OBJ_COLORS[obj], lw=1.5, ls="--")
            y_pos = 0.15 if obj == "o_min" else 0.08
            ax.annotate(
                f"β={lr.slope:.3f}, p={lr.pvalue:.1e}",
                xy=(0.02, y_pos), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=12, color=OBJ_COLORS[obj],
            )



def build_figure(bag: str, use_f2: bool) -> Path:
    """Build and save one 6-panel figure."""
    metric = "f2" if use_f2 else "full_r2"
    metric_label = "f²" if use_f2 else "R²"
    country_metric = "country_f2" if use_f2 else "country_full_r2"
    xgb_metric_col = "global_oof_r2"  # always R² in xgb_summary; f2 computed below

    print(f"  Loading data for {bag} / {'f2' if use_f2 else 'r2'} ...")
    _, _, mg_all = _load_data(bag)
    best_rung = select_best_rung(mg_all)
    d2, d2c, _ = _load_data(bag, rung_id=best_rung)
    xgb_sum = _load_xgb_summary(bag, best_rung)

    # For f2, compute in xgb_summary too
    if use_f2:
        xgb_sum["f2"] = _compute_f2(
            xgb_sum["global_oof_r2"].values,
            xgb_sum["global_oof_r2_base"].values,
        )
        xgb_metric_col = "f2"

    # Reordered layout (2 rows × 4 cols):
    # Row 1 — synergy ceiling, parsimony, null (all), null (diverse)
    # Row 2 — heatmap (best vs worst syn), domain composition, complexity ladder, entropy vs R²
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)

    bag_title = bag.capitalize()
    fig.suptitle(
        f"Exposome synergy & brain-age gap ({bag_title} BAG) — {metric_label}",
        fontsize=16, fontweight="bold",
    )

    # load null draws if present
    null_path = DATA_ROOT / "null_model" / bag / "null_global.parquet"
    null_df = pd.read_parquet(null_path) if null_path.exists() else None

    # Row 1: raw ceiling, parsimony, null (all), null (diverse filtered)
    panel_a_ceiling(axes[0, 0], xgb_sum, xgb_metric_col, metric_label)
    panel_b_parsimony(axes[0, 1], d2, metric, metric_label)
    panel_null_hist(axes[0, 2], null_df, xgb_sum, xgb_metric_col, metric_label, div_thresh=None)
    axes[0, 2].set_title("C  Null benchmark — all draws", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")
    panel_null_hist(axes[0, 3], null_df, xgb_sum, xgb_metric_col, metric_label, div_thresh=0.60)
    axes[0, 3].set_title("D  Null benchmark — diverse draws", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")

    # Row 2: decomposition and complexity
    panel_d_heatmap(axes[1, 0], xgb_sum, xgb_metric_col, metric_label)
    axes[1, 0].set_title("E  Best-vs-worst synergistic recipes", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")
    panel_c_domains(
        axes[1, 1], xgb_sum, xgb_metric_col, metric_label,
        mg_all=mg_all, best_rung=best_rung,
    )
    axes[1, 1].set_title("F  Multi-domain recipe drives performance", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")
    panel_e_ladder(axes[1, 2], mg_all, metric, metric_label, rung_id=best_rung)
    axes[1, 2].set_title("G  Interaction unlocks the signal", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")
    panel_shannon_entropy(axes[1, 3], xgb_sum, xgb_metric_col)
    axes[1, 3].set_title("H  Domain diversity vs performance", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")

    tag = f"{'f2' if use_f2 else 'r2'}_{bag}"
    out_path_png = FIGURE_OUT / f"boss_figure_{tag}.png"
    out_path_pdf = FIGURE_OUT / f"boss_figure_{tag}.pdf"
    fig.savefig(out_path_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_path_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path_png}")
    print(f"  ✓ Saved: {out_path_pdf}")
    return out_path_png


# ── entry ────────────────────────────────────────────────────────────────────
def panel_null_hist(
    ax,
    null_df: pd.DataFrame,
    xgb_summary: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    div_thresh: float | None = None,
):
    """Histogram of null metric with spans for top-20 real models.
    If `div_thresh` is provided, filter null draws by `max_domain_frac <= div_thresh`.
    """
    if null_df is None or null_df.empty:
        ax.text(0.5, 0.5, "Null model\nnot available", ha="center", va="center", color="grey")
        return

    subset = null_df.copy()
    if div_thresh is not None:
        subset = subset.loc[subset["max_domain_frac"] <= div_thresh]

    syn_top20 = xgb_summary[xgb_summary["objective"] == "o_min"].nlargest(20, metric_col)[metric_col]
    red_top20 = xgb_summary[xgb_summary["objective"] == "o_max"].nlargest(20, metric_col)[metric_col]

    if metric_col == "f2":
        base = float(xgb_summary["global_oof_r2_base"].iloc[0])
        vals = _compute_f2(subset["global_oof_r2"].values, base).squeeze()
    else:
        vals = subset["global_oof_r2"].dropna()

    vals = pd.Series(vals).dropna()
    if vals.empty:
        ax.text(0.5, 0.5, "No null draws after filtering", ha="center", va="center", color="grey")
        return

    bins = np.linspace(vals.min() - 0.01, max(vals.max(), syn_top20.max()) + 0.01, 50)
    ax.hist(vals, bins=bins, color="#bdbdbd", alpha=0.7, edgecolor="white", linewidth=0.3)

    # Top-20 spans
    ax.axvspan(syn_top20.min(), syn_top20.max(), color=OBJ_COLORS["o_min"], alpha=0.18)
    ax.axvspan(red_top20.min(), red_top20.max(), color=OBJ_COLORS["o_max"], alpha=0.18)

    p99 = np.percentile(vals, 99)
    ax.axvline(p99, ls="--", color="#888888", lw=0.8)

    ax.set_xlabel(metric_label)
    n_exceed = (vals >= syn_top20.max()).sum()
    p_val = n_exceed / len(vals)
    p_str = "p < 0.0001" if p_val < 1e-4 else f"p = {p_val:.4f}"
    ax.text(0.03, 0.95, f"P(null ≥ syn ceiling)\n{p_str}", transform=ax.transAxes, ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))




if __name__ == "__main__":
    for bag in ["combined", "functional", "structural"]:
        for use_f2 in [False, True]:
            build_figure(bag, use_f2)
    print("\nDone — all 6 figures generated.")


def panel_f_countries(ax, d2c: pd.DataFrame, d2g: pd.DataFrame,
                      metric: str, country_metric: str, metric_label: str):
    """Country lollipop: best syn vs best red per country."""
    best_syn_id = d2g[d2g["objective"] == "o_min"].sort_values(metric, ascending=False).iloc[0]["candidate_id"]
    best_red_id = d2g[d2g["objective"] == "o_max"].sort_values(metric, ascending=False).iloc[0]["candidate_id"]

    syn_c = d2c[d2c["candidate_id"] == best_syn_id][
        ["fold_country", country_metric, "n_test_scored"]
    ].rename(columns={country_metric: "syn"})
    red_c = d2c[d2c["candidate_id"] == best_red_id][
        ["fold_country", country_metric]
    ].rename(columns={country_metric: "red"})

    comp_all = syn_c.merge(red_c, on="fold_country").dropna(subset=["syn", "red"])
    comp_all["delta"] = comp_all["syn"] - comp_all["red"]
    # Exclude extreme outliers (e.g. France with R²≪-0.5) to avoid axis compression
    outlier_thresh = -0.5
    excluded = comp_all[(comp_all["syn"] <= outlier_thresh) | (comp_all["red"] <= outlier_thresh)]
    comp = comp_all[(comp_all["syn"] > outlier_thresh) & (comp_all["red"] > outlier_thresh)].copy()
    comp = comp.sort_values("syn", ascending=True)  # sort by syn performance

    y_pos = np.arange(len(comp))
    for i, (_, row) in enumerate(comp.iterrows()):
        ax.plot([row["syn"], row["red"]], [i, i], color="#cccccc", lw=0.8, zorder=1)
        ax.scatter(row["syn"], i, c=OBJ_COLORS["o_min"], s=30, zorder=3, edgecolors="white", linewidths=0.3)
        ax.scatter(row["red"], i, c=OBJ_COLORS["o_max"], s=30, zorder=3, edgecolors="white", linewidths=0.3)

    labels = [f"{row['fold_country']} (n={row['n_test_scored']:.0f})"
              for _, row in comp.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICK_FONTSIZE)
    ax.axvline(0, color="#cccccc", ls=":", lw=0.8)
    ax.set_xlabel(f"Country-level {metric_label}", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_title("F  Geographic consistency", fontsize=TICK_FONTSIZE, fontweight="bold", loc="left")

    # n_syn_wins = (comp["delta"] > 0).sum()
    # note = f"Synergy wins in {n_syn_wins}/{len(comp)} countries"
    # if len(excluded) > 0:
    #     excl_names = ", ".join(excluded["fold_country"].tolist())
    #     note += f"\n({len(excluded)} excluded: {excl_names}; {metric_label}\u226a0)"
    # _annotate(ax, note, x=0.97, y=0.03, fontsize=7.5, ha="right", va="bottom")

    # Minimal legend
    ax.scatter([], [], c=OBJ_COLORS["o_min"], s=20, label="Best synergistic")
    ax.scatter([], [], c=OBJ_COLORS["o_max"], s=20, label="Best redundant")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower right", framealpha=0.8, edgecolor="none")
