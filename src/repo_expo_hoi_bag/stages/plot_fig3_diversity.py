"""
Figure 3 — Multi-domain diversity as the structural basis of synergistic exposome–BAG prediction.

Layout: 3 rows (BAG) × 4 columns
  Col 1: Shannon H vs R²  scatter  (per-candidate, best rung per objective — Analysis A)
  Col 2: Multi-domain recipe       (stacked-bar domain composition, top-20 syn & red)
  Col 3: Co-occurrence network     REDUNDANCY  (domain level)
  Col 4: Co-occurrence network     SYNERGY     (domain level)

Row order: Functional | Structural | Combined

Outputs (written to paper_figures/):
  fig3_diversity.pdf / .svg / .png
  fig3_diversity_scatter.csv
  fig3_diversity_recipe_top20.csv
  fig3_diversity_regression.csv
  fig3_diversity_cooccurrence_nodes.csv
  fig3_diversity_cooccurrence_edges.csv

Usage:
  V3_OUTPUT_ROOT=outputs/variant_a python -m scripts.plot_fig3_diversity
"""

from __future__ import annotations

import os
import pathlib
import warnings
from collections import Counter

import matplotlib
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
except Exception:
    pass
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from matplotlib.colors import to_rgba

import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection

from scripts.cooccurrence_network_plotting import (
    _build_positions,
    _domain_node_size_from_within_count,
    _edge_color_from_normalized,
    _edge_width_bins_from_edge_frames,
    _edge_color_max_from_edge_frames,
    _edge_width_from_normalized,
    _text_color_for_fill,
    apply_domain_visual_encodings,
    DOMAIN_EDGE_WIDTH_LEVELS,
)
from oinfo_bag_ladder.cooccurrence_network import DomainNetwork

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
STATS       = ROOT / "stats"
PAPER_FIGS  = pathlib.Path(__file__).resolve().parents[1] / "paper_figures"
PAPER_FIGS.mkdir(exist_ok=True)

# ── colours — user-specified domain palette ────────────────────────────────────
DOMAIN_COLORS: dict[str, str] = {
    "Air Pollution":               "#A6BCD3",
    "Climate disasters":           "#F8C695",
    "Democracy":                   "#ACD0A7",
    "Disease-related mortality":   "#F0ABAC",
    "Green space access":          "#BADBD8",
    "Migration":                   "#F6E4A4",
    "Precipitation/droughts":      "#D8BCD0",
    "Socioeconomic":               "#CEBAAF",
    "Soil and water quality":      "#F0BEDE",
    "Temperature":                 "#A4CCCA",
    "Other":                       "#CCCCCC",
}
DOMAIN_ORDER = [d for d in DOMAIN_COLORS if d != "Other"]
DOMAIN_LABELS = {"Precipitation/droughts": "Precipitation"}

SYN_COLOR = "#1B6B2E"   # same as fig2
RED_COLOR = "#4B0082"   # same as fig2
OBJ_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
OBJ_LABEL = {"o_min": "Synergistic", "o_max": "Redundant"}

BAGS      = [
    bag for bag in os.environ.get("PAPER_FIG_BAGS", "functional,structural").split(",")
    if bag in {"functional", "structural"}
]
BAG_LABEL = {
    "functional": "Functional BAG",
    "structural": "Structural BAG",
    "combined":   "Combined BAG",
}

FS    = 22   # same as fig2 base font size + 2
FS_TK = 20   # same as fig2 tick/secondary font size + 2
FS_COL = 23  # column titles, +1 over previous setting

COL_LABELS = [
    "a. BAG prediction vs domain diversity",
    "b. Domain composition",
    "c. Redundancy domain co-ocurrence network",
    "d. Synegy domain co-ocurrence network",
]


# ── patch DOMAIN_COLORS in-place so cooccurrence_network_plotting picks up our palette ─
# Both modules share the same dict object; mutating it propagates everywhere.
import oinfo_bag_ladder.make_boss_figure as _boss
_boss.DOMAIN_COLORS.clear()
_boss.DOMAIN_COLORS.update(DOMAIN_COLORS)


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_p(p: float | None) -> str:
    """Exact p, never a floor bound.

    This is the two-sided p of the OLS slope from scipy.stats.linregress, so it
    has no resolution floor that a "<" bound could legitimately express: the
    former "p<0.001" discarded a value the test actually resolved. Very small
    values render in scientific notation rather than as 0.000.
    """
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 0.001:
        return f"p={p:.2e}"
    return f"p={p:.3f}"


def _edge_color_from_family(value: float, max_normalized: float, family: str) -> tuple[float, float, float, float]:
    """Map normalized edge strength to a family-specific color gradient."""
    max_normalized = max(float(max_normalized), 1.0e-12)
    norm = float(min(max(float(value) / max_normalized, 0.0), 1.0))
    base = SYN_COLOR if family == "synergy" else RED_COLOR
    rgba = to_rgba(base)
    # Keep the same hue family while moving from white to the base color.
    mix = norm
    rgb = tuple((1.0 - mix) * 1.0 + mix * c for c in rgba[:3])
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(0.30 + 0.60 * norm))


def _reg_row(reg_df: pd.DataFrame, bag: str, term: str) -> tuple[float | None, float | None]:
    sub = reg_df[(reg_df["bag"] == bag) & (reg_df["regression"] == "R2_diversity")]
    row = sub[sub["Unnamed: 0"] == term]
    if row.empty:
        return None, None
    return float(row["Coef."].iloc[0]), float(row["P>|t|"].iloc[0])


def _build_domain_network_from_recipe(recipe_df: pd.DataFrame, bag: str, family: str, top_k: int = 20) -> DomainNetwork:
    objective = "o_min" if family == "synergy" else "o_max"
    sub = recipe_df[(recipe_df["bag"] == bag) & (recipe_df["objective"] == objective)].copy()
    if sub.empty:
        empty_models = pd.DataFrame(columns=["model_id", "model_rank", "global_oof_r2", "predictors_identity"])
        empty_nodes = pd.DataFrame(columns=[
            "bag",
            "family",
            "rung",
            "domain",
            "display_label",
            "domain_color",
            "model_presence_topk",
            "model_presence_fraction",
            "within_domain_pair_count",
            "in_top1_model",
            "rank1_model_id",
            "node_size_used",
            "frequency_rank",
        ])
        empty_edges = pd.DataFrame(columns=[
            "bag",
            "family",
            "rung",
            "domain_a",
            "domain_b",
            "cooccurrence_count",
            "normalized_cooccurrence",
            "in_top1_pair",
            "edge_width_used",
            "edge_rank",
        ])
        return DomainNetwork(
            bag=bag,
            family=family,
            rung="",
            top_k=top_k,
            models=empty_models,
            nodes=empty_nodes,
            edges=empty_edges,
            top1_model_id="",
            top1_domains=(),
        )

    model_rows: list[dict[str, object]] = []
    domain_counts: Counter[str] = Counter()
    within_counts: Counter[str] = Counter()
    cross_counts: Counter[tuple[str, str]] = Counter()
    total_pair_opportunities = 0

    for model_id, grp in sub.groupby("model_id", sort=False):
        grp = grp.copy()
        vars_here = [str(v).strip() for v in grp["variable"].tolist() if str(v).strip()]
        domains_here = [str(d).strip() for d in grp["domain"].tolist() if str(d).strip()]
        unique_vars = list(dict.fromkeys(vars_here))
        unique_domains = list(dict.fromkeys([d if d in DOMAIN_ORDER else "Other" for d in domains_here]))
        if not unique_vars:
            continue

        model_rank = pd.to_numeric(grp["model_rank"], errors="coerce").min() if "model_rank" in grp else np.nan
        global_oof_r2 = pd.to_numeric(grp["global_oof_r2"], errors="coerce").iloc[0] if "global_oof_r2" in grp else np.nan
        rung = str(grp["rung"].iloc[0]) if "rung" in grp else ""
        order = pd.to_numeric(grp["order"], errors="coerce").iloc[0] if "order" in grp else np.nan

        model_rows.append(
            {
                "model_id": str(model_id),
                "model_rank": int(model_rank) if pd.notna(model_rank) else np.nan,
                "global_oof_r2": float(global_oof_r2) if pd.notna(global_oof_r2) else np.nan,
                "rung": rung,
                "predictors_identity": "|".join(unique_vars),
                "order": int(order) if pd.notna(order) else np.nan,
                "domain_list": unique_domains,
            }
        )

        total_pair_opportunities += int(len(unique_vars) * (len(unique_vars) - 1) / 2)

        for domain in unique_domains:
            domain_counts[domain] += 1

        by_domain: dict[str, list[str]] = {}
        for domain in domains_here:
            dom = domain if domain in DOMAIN_ORDER else "Other"
            by_domain.setdefault(dom, []).append(domain)

        for domain, members in by_domain.items():
            if len(members) >= 2:
                within_counts[domain] += int(len(members) * (len(members) - 1) / 2)

        domain_keys = sorted(by_domain.keys())
        for i, dom_a in enumerate(domain_keys):
            for dom_b in domain_keys[i + 1:]:
                cross_counts[tuple(sorted((dom_a, dom_b)))] += len(by_domain[dom_a]) * len(by_domain[dom_b])

    models = pd.DataFrame(model_rows)
    models = models.sort_values(
        by=["model_rank", "global_oof_r2", "model_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top1_row = models.iloc[0]
    top1_model_id = str(top1_row["model_id"])
    top1_domains = tuple(dict.fromkeys(
        [str(d).strip() for d in sub[sub["model_id"].astype(str) == top1_model_id]["domain"].tolist() if str(d).strip()]
    ))
    top1_set = set(top1_domains)
    rung = str(top1_row["rung"])
    if total_pair_opportunities <= 0:
        total_pair_opportunities = 1

    node_rows: list[dict[str, object]] = []
    for domain in DOMAIN_ORDER:
        if domain not in domain_counts and domain not in within_counts and domain not in top1_set:
            continue
        node_rows.append(
            {
                "bag": bag,
                "family": family,
                "rung": rung,
                "domain": domain,
                "display_label": domain,
                "domain_color": DOMAIN_COLORS.get(domain, DOMAIN_COLORS.get("Other", "#cccccc")),
                "model_presence_topk": int(domain_counts.get(domain, 0)),
                "model_presence_fraction": float(domain_counts.get(domain, 0) / float(top_k)) if top_k > 0 else np.nan,
                "within_domain_pair_count": int(within_counts.get(domain, 0)),
                "in_top1_model": bool(domain in top1_set),
                "rank1_model_id": top1_model_id if domain in top1_set else "",
                "node_size_used": np.nan,
            }
        )

    edge_rows: list[dict[str, object]] = []
    for (dom_a, dom_b), count in cross_counts.items():
        edge_rows.append(
            {
                "bag": bag,
                "family": family,
                "rung": rung,
                "domain_a": dom_a,
                "domain_b": dom_b,
                "cooccurrence_count": int(count),
                "normalized_cooccurrence": float(count / total_pair_opportunities),
                "in_top1_pair": bool(dom_a in top1_set and dom_b in top1_set),
                "edge_width_used": np.nan,
            }
        )

    node_df = pd.DataFrame(node_rows)
    if not node_df.empty:
        node_df = node_df.sort_values(
            by=["within_domain_pair_count", "model_presence_topk", "domain"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        node_df["frequency_rank"] = np.arange(1, len(node_df) + 1)
    else:
        node_df = pd.DataFrame(columns=[
            "bag",
            "family",
            "rung",
            "domain",
            "display_label",
            "domain_color",
            "model_presence_topk",
            "model_presence_fraction",
            "within_domain_pair_count",
            "in_top1_model",
            "rank1_model_id",
            "node_size_used",
            "frequency_rank",
        ])

    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty:
        edge_df = edge_df.sort_values(
            by=["cooccurrence_count", "domain_a", "domain_b"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        edge_df["edge_rank"] = np.arange(1, len(edge_df) + 1)
    else:
        edge_df = pd.DataFrame(columns=[
            "bag",
            "family",
            "rung",
            "domain_a",
            "domain_b",
            "cooccurrence_count",
            "normalized_cooccurrence",
            "in_top1_pair",
            "edge_width_used",
            "edge_rank",
        ])

    return DomainNetwork(
        bag=bag,
        family=family,
        rung=rung,
        top_k=top_k,
        models=models,
        nodes=node_df,
        edges=edge_df,
        top1_model_id=top1_model_id,
        top1_domains=top1_domains,
    )


# ── panel functions ─────────────────────────────────────────────────────────────

def draw_scatter(ax: plt.Axes, scatter_df: pd.DataFrame, reg_df: pd.DataFrame, bag: str):
    """Shannon H vs R² scatter with bivariate regression lines and annotations per objective."""
    sub = scatter_df[scatter_df["bag"] == bag]

    lr_by_obj: dict[str, object] = {}
    for obj in ["o_max", "o_min"]:
        grp = sub[sub["objective"] == obj]
        ax.scatter(
            grp["shannon_h"], grp["global_oof_r2"],
            c=OBJ_COLOR[obj], alpha=0.20, s=22, linewidths=0,
            rasterized=True, zorder=2,
        )
        if len(grp) >= 10 and grp["shannon_h"].std() > 0:
            lr = scipy_stats.linregress(grp["shannon_h"], grp["global_oof_r2"])
            lr_by_obj[obj] = lr
            xs = np.linspace(grp["shannon_h"].min(), grp["shannon_h"].max(), 100)
            x = grp["shannon_h"].to_numpy(dtype=float)
            y = grp["global_oof_r2"].to_numpy(dtype=float)
            n = len(x)
            x_mean = float(np.mean(x))
            sxx = float(np.sum((x - x_mean) ** 2))
            y_hat = lr.slope * xs + lr.intercept
            if n >= 3 and sxx > 0:
                resid = y - (lr.slope * x + lr.intercept)
                s_err = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
                t_crit = float(scipy_stats.t.ppf(0.975, df=n - 2))
                se_mean = s_err * np.sqrt((1.0 / n) + ((xs - x_mean) ** 2 / sxx))
                ax.fill_between(
                    xs,
                    y_hat - t_crit * se_mean,
                    y_hat + t_crit * se_mean,
                    color=OBJ_COLOR[obj],
                    alpha=0.14,
                    linewidth=0,
                    zorder=2.5,
                )
            ax.plot(xs, y_hat,
                    color=OBJ_COLOR[obj], linewidth=3.6, alpha=0.95, zorder=3)

    # Annotation: bivariate slopes + r + p per objective
    lines = []
    for obj, label in [("o_max", "red"), ("o_min", "syn")]:
        if obj in lr_by_obj:
            lr = lr_by_obj[obj]
            slope_str = f"{lr.slope:.4f}" if lr.slope < 0 else f"{lr.slope:.4f}"
            r_str     = f"{lr.rvalue:.2f}" if lr.rvalue < 0 else f"{lr.rvalue:.2f}"
            lines.append(
                f"β_H {label}={slope_str} (r={r_str}, {_fmt_p(lr.pvalue)})"
            )
    annot = "\n".join(lines)

    ax.text(0.97, 0.03, annot, transform=ax.transAxes,
            fontsize=FS_TK, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    legend_handles = [
        mlines.Line2D([], [], color=OBJ_COLOR["o_min"], marker="o", linestyle="None",
                      markersize=7, label="Synergistic"),
        mlines.Line2D([], [], color=OBJ_COLOR["o_max"], marker="o", linestyle="None",
                      markersize=7, label="Redundant"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=FS_TK - 1,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.2,
    )

    ax.set_xlabel("Shannon domain entropy H (bits)", fontsize=FS)
    ax.set_ylabel("LOCO R²", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.spines[["top", "right"]].set_visible(False)


def draw_recipe(ax: plt.Axes, recipe_df: pd.DataFrame, bag: str):
    """Stacked horizontal bars — one per model in top-20, sorted best→worst R² per block."""
    sub = recipe_df[recipe_df["bag"] == bag].copy()

    records = []
    for (obj, model_id, r2), grp in sub.groupby(
            ["objective", "model_id", "global_oof_r2"], sort=False):
        total = len(grp)
        dom_counts = grp["domain"].value_counts()
        fracs = {d: dom_counts.get(d, 0) / total for d in DOMAIN_ORDER}
        records.append({"objective": obj, "model_id": model_id,
                        "global_oof_r2": r2, **fracs})

    df_wide = pd.DataFrame(records)
    syn = df_wide[df_wide["objective"] == "o_min"].sort_values("global_oof_r2", ascending=True)
    red = df_wide[df_wide["objective"] == "o_max"].sort_values("global_oof_r2", ascending=True)

    n_syn = len(syn)
    n_red = len(red)
    gap   = 1.5

    y_syn = np.arange(n_syn) + n_red + gap
    y_red = np.arange(n_red)
    bar_h = 0.82

    def _plot_block(block_df, ys, label):
        for i, (_, row) in enumerate(block_df.iterrows()):
            left = 0.0
            for dom in DOMAIN_ORDER:
                val = row.get(dom, 0.0)
                if val > 0:
                    ax.barh(ys[i], val, left=left, height=bar_h,
                            color=DOMAIN_COLORS[dom], linewidth=0, zorder=2)
                    left += val
        # label at left edge, inside the axes
        color = SYN_COLOR if label == "Syn" else RED_COLOR
        ax.text(0.01, ys.mean(), label,
                transform=ax.get_yaxis_transform(),
                va="center", ha="left",
                fontsize=FS_TK, color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
                zorder=5)

    _plot_block(syn, y_syn, "Syn")
    _plot_block(red, y_red, "Red")

    ax.axhline(n_red + gap / 2, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n_syn + n_red + gap + 0.7)
    ax.set_xlabel("Domain fraction", fontsize=FS)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=FS_TK)
    ax.tick_params(axis="x", labelsize=FS_TK)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _draw_domain_panel_scaled(
    ax: plt.Axes,
    panel: DomainNetwork,
    *,
    edge_width_bins,
    edge_color_max: float,
    node_size_scale: float = 8.0,
    label_fontsize: float = 9.0,
):
    """Custom domain-network drawing that honours pre-scaled node_size_used.

    Unlike _draw_domain_panel, this function does NOT call apply_domain_visual_encodings
    internally, so any node_size_used values set before calling are preserved.
    node_size_scale is applied on top of the base _domain_node_size_from_within_count.
    """
    import textwrap

    node_df = panel.nodes.copy()
    edge_df = panel.edges.copy()

    # Compute sizes from within_domain_pair_count with our own scale factor
    if not node_df.empty and "within_domain_pair_count" in node_df.columns:
        node_df["node_size_used"] = node_df["within_domain_pair_count"].apply(
            lambda v: _domain_node_size_from_within_count(v, panel.top_k) * node_size_scale
        )
    # Compute edge widths
    if not edge_df.empty and "normalized_cooccurrence" in edge_df.columns:
        edge_df["edge_width_used"] = edge_df["normalized_cooccurrence"].apply(
            lambda v: _edge_width_from_normalized(v, edge_width_bins, DOMAIN_EDGE_WIDTH_LEVELS)
        )

    from oinfo_bag_ladder.make_boss_figure import DOMAIN_ORDER as _DOMAIN_ORDER
    order = [d for d in _DOMAIN_ORDER if d in set(node_df["domain"].astype(str).tolist())]
    positions = _build_positions(order)
    present_set = set(node_df["domain"].astype(str).tolist())
    present_nodes = [d for d in order if d in present_set]
    pos_subset = {d: positions[d] for d in present_nodes if d in positions}

    plot_edges = edge_df[edge_df["cooccurrence_count"] >= 2].copy()
    plot_edges = plot_edges.sort_values(
        by=["cooccurrence_count", "domain_a", "domain_b"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(80)

    ax.axis("off")

    if not plot_edges.empty:
        segments, colors, widths = [], [], []
        for _, edge in plot_edges.iterrows():
            a, b = str(edge["domain_a"]), str(edge["domain_b"])
            if a not in pos_subset or b not in pos_subset:
                continue
            x1, y1, _ = pos_subset[a]
            x2, y2, _ = pos_subset[b]
            segments.append([(x1, y1), (x2, y2)])
            colors.append(_edge_color_from_family(
                edge.get("normalized_cooccurrence", 0.0),
                edge_color_max if edge_color_max is not None else 1.0,
                panel.family,
            ))
            widths.append(float(edge.get("edge_width_used", 1.0)))
        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=1, capstyle="round")
            ax.add_collection(lc)

    if not node_df.empty:
        normal_x, normal_y, normal_sizes, normal_fills = [], [], [], []
        top1_x,   top1_y,   top1_sizes,   top1_fills   = [], [], [], []
        for _, row in node_df.iterrows():
            domain = str(row["domain"])
            if domain not in pos_subset:
                continue
            x, y, _ = pos_subset[domain]
            fill = DOMAIN_COLORS.get(domain, DOMAIN_COLORS.get("Other", "#cccccc"))
            size = float(row["node_size_used"])
            if bool(row.get("in_top1_model", False)):
                top1_x.append(x); top1_y.append(y)
                top1_sizes.append(size); top1_fills.append(fill)
            else:
                normal_x.append(x); normal_y.append(y)
                normal_sizes.append(size); normal_fills.append(fill)
        if normal_x:
            ax.scatter(normal_x, normal_y, s=normal_sizes, c=normal_fills,
                       edgecolors="#666666", linewidths=0.9, alpha=0.96, zorder=3)
        if top1_x:
            ax.scatter(top1_x, top1_y, s=top1_sizes, c=top1_fills,
                       edgecolors="#111111", linewidths=2.4, alpha=0.96, zorder=3)

        for _, row in node_df.iterrows():
            domain = str(row["domain"])
            if domain not in pos_subset:
                continue
            x, y, _ = pos_subset[domain]
            label = textwrap.fill(
                DOMAIN_LABELS.get(domain, domain),
                width=10,
                break_long_words=False,
                break_on_hyphens=True,
            )
            fill = DOMAIN_COLORS.get(domain, "#cccccc")
            txt_color = _text_color_for_fill(fill)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=label_fontsize, color=txt_color, zorder=4)

    # Limits sized to contain nodes + labels without dead space.
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.margins(0)


def draw_network(ax: plt.Axes,
                 panel: DomainNetwork,
                 edge_width_bins, edge_color_max: float):
    """Domain co-occurrence panel using custom scaled drawing."""
    _draw_domain_panel_scaled(
        ax, panel,
        edge_width_bins=edge_width_bins,
        edge_color_max=edge_color_max,
        node_size_scale=7.2,
        label_fontsize=17.0,
    )


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    scatter_df = pd.read_csv(STATS / "per_candidate_diversity_scatter.csv")
    recipe_df  = pd.read_csv(STATS / "per_candidate_recipe_top20.csv")
    reg_df     = pd.read_csv(STATS / "domain_diversity_regression.csv")

    domain_panels = {
        (bag, family): _build_domain_network_from_recipe(recipe_df, bag, family)
        for bag in BAGS
        for family in ("redundancy", "synergy")
    }
    all_edge_frames = [panel.edges for panel in domain_panels.values()]
    edge_width_bins = _edge_width_bins_from_edge_frames(all_edge_frames)
    edge_color_max  = _edge_color_max_from_edge_frames(all_edge_frames)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(32, max(18, 10 * len(BAGS))))
    gs  = gridspec.GridSpec(
        len(BAGS), 4, figure=fig,
        wspace=0.12,
        hspace=0.18,
        width_ratios=[1.0, 0.9, 1.0, 1.0],
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(len(BAGS))]

    for ri, bag in enumerate(BAGS):
        for ci in range(4):
            ax = axes[ri][ci]

            if ci == 0:
                draw_scatter(ax, scatter_df, reg_df, bag)
                ax.text(
                    -0.20,
                    0.5,
                    BAG_LABEL[bag],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=FS,
                )
            elif ci == 1:
                draw_recipe(ax, recipe_df, bag)
            elif ci == 2:
                draw_network(ax, domain_panels[(bag, "redundancy")],
                             edge_width_bins, edge_color_max)
            elif ci == 3:
                draw_network(ax, domain_panels[(bag, "synergy")],
                             edge_width_bins, edge_color_max)

            if ri == 0:
                ax.set_title(
                    COL_LABELS[ci].replace("domain co-ocurrence network", "domain network"),
                    fontsize=FS_COL,
                    loc="left",
                )

    # ── save figure ───────────────────────────────────────────────────────────
    for ext in ("pdf", "svg"):
        out = PAPER_FIGS / f"fig3_diversity.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    fig.savefig(PAPER_FIGS / "fig3_diversity.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {PAPER_FIGS / 'fig3_diversity.png'}")
    plt.close(fig)

    # ── export supporting data ────────────────────────────────────────────────
    scatter_df.to_csv(PAPER_FIGS / "fig3_diversity_scatter.csv", index=False)
    recipe_df.to_csv(PAPER_FIGS / "fig3_diversity_recipe_top20.csv", index=False)
    reg_df[reg_df["regression"] == "R2_diversity"].to_csv(
        PAPER_FIGS / "fig3_diversity_regression.csv", index=False)
    pd.concat([panel.nodes for panel in domain_panels.values()], ignore_index=True).to_csv(
        PAPER_FIGS / "fig3_diversity_cooccurrence_nodes.csv", index=False)
    pd.concat([panel.edges for panel in domain_panels.values()], ignore_index=True).to_csv(
        PAPER_FIGS / "fig3_diversity_cooccurrence_edges.csv", index=False)
    for f in [
        "fig3_diversity_scatter.csv", "fig3_diversity_recipe_top20.csv",
        "fig3_diversity_regression.csv", "fig3_diversity_cooccurrence_nodes.csv",
        "fig3_diversity_cooccurrence_edges.csv",
    ]:
        print(f"Saved: {PAPER_FIGS / f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
