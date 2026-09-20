#!/usr/bin/env python3
"""
Figure 3 v2 — Multi-domain diversity with additional SHAP columns.

This script reuses the exact fig3_diversity layout and adds two SHAP columns:
redundant and synergistic.

Outputs written to paper_figures/:
  fig3_diversity_v2.pdf / .svg / .png
  fig3_diversity_v2_scatter.csv
  fig3_diversity_v2_recipe_top20.csv
  fig3_diversity_v2_regression.csv
  fig3_diversity_v2_cooccurrence_nodes.csv
  fig3_diversity_v2_cooccurrence_edges.csv
  fig3_diversity_v2_shap_selected.csv
"""
from __future__ import annotations

import os
import pathlib
import re
import textwrap
import warnings

import matplotlib
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
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"]
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import numpy as np
import pandas as pd

from scripts.plot_fig3_diversity import (
    BAGS,
    BAG_LABEL,
    DOMAIN_COLORS,
    FS,
    FS_TK,
    _draw_domain_panel_scaled,
    _build_domain_network_from_recipe,
    _edge_color_max_from_edge_frames,
    _edge_width_bins_from_edge_frames,
    draw_network,
    draw_recipe,
    draw_scatter,
)
from exposome_labels import display_label

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data

# Re-assert the fig2 font policy: importing scripts.plot_fig3_diversity above
# executes that module's own top-level rcParams (font.family="Arial", no
# fallback list), which silently overrides the block set earlier in this file.
# This must run LAST (after all imports) to actually win.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"]

ROOT = pathlib.Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
STATS = ROOT / "stats"
PAPER_FIGS = pathlib.Path(__file__).resolve().parents[1] / "paper_figures"
PAPER_FIGS.mkdir(exist_ok=True)

SYN_SHAP_ROOT = ROOT / "shap_oof"
RED_SHAP_ROOT = ROOT / "shap_oof_red"
# Public inputs are versioned at the checkout root, not inside the package.
FEATURE_DOMAINS = pathlib.Path(__file__).resolve().parents[3] / "data" / "metadata" / "exposome_feature_domains.csv"

# Order cap for the paper figure variants. Fig3 is rendered in its capped,
# no-SHAP
# form (scatter, recipe, two co-occurrence networks) directly from the capped
# candidate pool, with all derived tables built in-code.
ORDER_MAX = int(os.environ.get("PAPER_FIG_ORDER_MAX", "30"))
SUFFIX = os.environ.get("PAPER_FIG_SUFFIX", "")
OUT_DIR = pathlib.Path(os.environ["PAPER_FIG_OUTDIR"]) if os.environ.get("PAPER_FIG_OUTDIR") else PAPER_FIGS
CAP_BAGS = [b for b in os.environ.get("PAPER_FIG_BAGS", "").split(",") if b]
# Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): an o_min
# candidate counts as synergistic only if its evaluated O-info (thoi_o) < 0.
_SYN_OINFO_NEGATIVE = os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

BASELINE_FEATURE_PREFIXES = ("age", "year", "sex_", "diag_")
SHAP_POS = "#b2182b"
SHAP_NEG = "#2166ac"
SHAP_TOP_N = 8
SHAP_POINT_MAX = 900
LETTERS = list("ABCDEFGHIJKLMNOPQR")


def _is_baseline_feature(feature: str) -> bool:
    return feature == "__bias__" or any(feature.startswith(prefix) for prefix in BASELINE_FEATURE_PREFIXES)


def _load_shap_bundle(shap_root: pathlib.Path, bag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = shap_root / bag / "shap_summary.parquet"
    values_path = shap_root / bag / "shap_values_oof.parquet"
    if not summary_path.exists() or not values_path.exists():
        raise FileNotFoundError(
            f"Missing SHAP outputs in {shap_root / bag}. Run scripts/run_shap_oof_best_red.py first for redundant SHAP."
        )
    summary_df = pd.read_parquet(summary_path)
    values_df = pd.read_parquet(values_path)
    return summary_df, values_df


def _shap_feature_stats(values_df: pd.DataFrame) -> pd.DataFrame:
    full_prefix = "shap_full__"
    rows = []
    for col in values_df.columns:
        if not col.startswith(full_prefix):
            continue
        feature = col[len(full_prefix):]
        if _is_baseline_feature(feature):
            continue
        vals = pd.to_numeric(values_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_abs = float(np.mean(np.abs(vals)))
        if mean_abs <= 0:
            continue
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": mean_abs,
                "mean_shap": float(np.mean(vals)),
                "median_shap": float(np.median(vals)),
                "n_subjects": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def _select_shap_features(values_df: pd.DataFrame, top_n: int = SHAP_TOP_N) -> tuple[list[str], pd.DataFrame]:
    stats_df = _shap_feature_stats(values_df)
    if stats_df.empty:
        return [], pd.DataFrame()
    selected = stats_df.sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n).copy()
    features = selected["feature"].astype(str).tolist()
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    return features, selected


def _load_feature_domains() -> dict[str, str]:
    df = pd.read_csv(FEATURE_DOMAINS)
    return dict(zip(df["feature_name"].astype(str), df["domain"].astype(str)))


def _fmt_tick(value: float) -> str:
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _format_shap_label(feature: str) -> str:
    label = display_label(feature)
    return label.translate(str.maketrans({"₂": "2", "₃": "3"}))


def _darken_color(color: str, factor: float = 0.62) -> str:
    rgb = np.array(to_rgb(color))
    return to_hex(np.clip(rgb * factor, 0, 1))


def _panel_xlim(values_df: pd.DataFrame, features: list[str]) -> float:
    arrays = [
        pd.to_numeric(values_df[f"shap_full__{feat}"], errors="coerce").dropna().to_numpy(dtype=float)
        for feat in features
        if f"shap_full__{feat}" in values_df.columns
    ]
    if not arrays:
        return 0.05
    vals = np.concatenate(arrays)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.05
    xlim = float(np.nanpercentile(np.abs(vals), 99.5))
    return max(xlim * 1.10 if np.isfinite(xlim) else 0.05, 0.05)


def _draw_shap_panel(
    ax: plt.Axes,
    values_df: pd.DataFrame,
    features: list[str],
    feature_domains: dict[str, str],
    *,
    xlim: float,
    show_ylabel: bool,
    show_xlabel: bool,
    title: str,
) -> None:
    rng = np.random.default_rng(20240517)
    labels = [_format_shap_label(f) for f in features]
    violin_data = []
    violin_pos = []

    for yi, feat in enumerate(features):
        col = f"shap_full__{feat}"
        if col not in values_df.columns:
            continue
        shap_vals = pd.to_numeric(values_df[col], errors="coerce").to_numpy(dtype=float)
        shap_vals = shap_vals[np.isfinite(shap_vals)]
        if shap_vals.size == 0:
            continue
        violin_data.append(shap_vals)
        violin_pos.append(yi)
        if shap_vals.size > SHAP_POINT_MAX:
            keep = rng.choice(np.arange(shap_vals.size), size=SHAP_POINT_MAX, replace=False)
            shap_vals = shap_vals[keep]
        jitter = rng.normal(0, 0.065, size=shap_vals.size)
        jitter = np.clip(jitter, -0.22, 0.22)
        ax.scatter(
            shap_vals,
            yi + jitter,
            c="#111111",
            s=8,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=violin_pos,
            vert=False,
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#9E9E9E")
            body.set_edgecolor("none")
            body.set_alpha(0.5)
            body.set_linewidth(0)
            body.set_zorder(2)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.65, zorder=2)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.6, len(features) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(features)))
    if show_ylabel:
        ax.set_yticklabels(
            [textwrap.fill(label, width=16, break_long_words=False) for label in labels],
            fontsize=FS_TK,
        )
        for tick, feature in zip(ax.get_yticklabels(), features):
            color = DOMAIN_COLORS.get(feature_domains.get(feature, "Other"), "#111111")
            tick.set_color(_darken_color(color))
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ticks = np.linspace(-xlim, xlim, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_tick(v) for v in ticks], fontsize=FS_TK)
        ax.set_xlabel("SHAP value", fontsize=FS)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", length=0, pad=1)
    ax.tick_params(axis="x", labelsize=FS_TK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.12, linewidth=0.6, zorder=1)
    if title:
        ax.set_title(title, fontsize=FS, pad=6)


def _domain_diversity(domains: list[str]) -> tuple[float, int, str]:
    from collections import Counter

    counts = Counter(d for d in domains if d)
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0, "Other"
    fracs = np.array([c / total for c in counts.values()], dtype=float)
    shannon = float(-(fracs * np.log2(fracs)).sum())
    dominant = counts.most_common(1)[0][0]
    return shannon, len(counts), dominant


def _build_cap_tables(bags: list[str], feature_domains: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the per-candidate diversity scatter + top-20 recipe tables on the
    capped pool. The best rung per (bag, objective) is the highest-full-R²
    candidate at order ≤ cap; the scatter uses every candidate of that rung and
    the recipe explodes the top-20 by predictor → domain."""
    scatter_rows, recipe_rows = [], []
    for bag in bags:
        metrics = pd.read_parquet(
            ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
            / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        )
        metrics["order"] = pd.to_numeric(metrics["order"], errors="coerce")
        metrics["full_r2"] = pd.to_numeric(metrics["full_r2"], errors="coerce")
        metrics = metrics[metrics["order"] <= ORDER_MAX]
        for objective in ("o_min", "o_max"):
            sub = metrics[metrics["objective"] == objective]
            # Optional negative-O-info synergy criterion: keep only o_min
            # candidates with evaluated O-info (thoi_o) < 0. o_max untouched.
            if _SYN_OINFO_NEGATIVE and objective == "o_min":
                if "thoi_o" not in sub.columns:
                    raise KeyError("SYN_OINFO_NEGATIVE set but 'thoi_o' absent from metrics.")
                sub = sub[pd.to_numeric(sub["thoi_o"], errors="coerce") < 0]
            if sub.empty:
                continue
            best_rung = str(sub.loc[sub["full_r2"].idxmax(), "rung_id"])
            rung_df = sub[sub["rung_id"] == best_rung]
            for _, row in rung_df.iterrows():
                variables = [v for v in str(row["predictors_identity"]).split("|") if v]
                domains = [feature_domains.get(v, "Other") for v in variables]
                shannon, n_dom, dominant = _domain_diversity(domains)
                scatter_rows.append({
                    "model_id": str(row["candidate_id"]), "objective": objective,
                    "order": int(row["order"]), "global_oof_r2": float(row["full_r2"]),
                    "predictors_identity": str(row["predictors_identity"]), "bag": bag,
                    "rung": best_rung, "shannon_h": shannon, "n_domains": n_dom,
                    "dominant_domain": dominant,
                })
            top = rung_df.nlargest(20, "full_r2").reset_index(drop=True)
            for rank, (_, row) in enumerate(top.iterrows(), start=1):
                for variable in [v for v in str(row["predictors_identity"]).split("|") if v]:
                    recipe_rows.append({
                        "bag": bag, "objective": objective, "rung": best_rung,
                        "model_id": str(row["candidate_id"]), "model_rank": rank,
                        "global_oof_r2": float(row["full_r2"]), "order": int(row["order"]),
                        "variable": variable, "domain": feature_domains.get(variable, "Other"),
                    })
    return pd.DataFrame(scatter_rows), pd.DataFrame(recipe_rows)


CAP_COL_TITLES = (
    "a. BAG prediction vs domain diversity",
    "b. Domain composition",
    "c. Redundancy domain network",
    "d. Synergy domain network",
)
FONT_BUMP = 2   # applied to every font in cols 0-1 (scatter, recipe); networks untouched
TITLE_BUMP = 6  # applied on top of the base column-title fontsize


_BAG_SHORT = {"structural": "struct", "functional": "func", "combined": "comb"}
_ROW_LETTERS = ("a", "b", "c")
_OBJ_GLOSS = ("o_min = synergistic (min O-information) arm; "
              "o_max = redundant (max O-information) arm")


def _build_fig3_panels(scatter_df: pd.DataFrame, recipe_df: pd.DataFrame,
                       nets: dict, bags: list[str]) -> list[Panel]:
    """Per BAG: the scatter points, its fitted slopes, the recipe composition,
    and both domain networks' nodes and edges.

    The regression is recomputed here with the same scipy.stats.linregress call
    draw_scatter uses for its annotation, so the exported slope, r and p are the
    values printed on the panel -- and p is the exact float, never a bound.
    """
    from scipy import stats as _scipy_stats

    panels: list[Panel] = []
    for letter, bag in zip(_ROW_LETTERS, bags):
        short = _BAG_SHORT.get(bag, bag)
        sub = scatter_df[scatter_df["bag"] == bag]

        panels.append(Panel(
            panel_id=f"{letter}1_{short}_diversity_scatter",
            frame=sub[["objective", "rung", "model_id", "order", "global_oof_r2",
                       "shannon_h", "n_domains", "dominant_domain"]].reset_index(drop=True),
            description=(f"{bag.capitalize()} BAG prediction against domain diversity: one point "
                         "per candidate of the best-performing model level."),
            columns={
                "objective": _OBJ_GLOSS,
                "rung": "model level the candidates were drawn from (highest-R² level for that arm)",
                "model_id": "identifier of the exposure set",
                "order": "set size (number of exposures)",
                "global_oof_r2": "held-out LOCO R² -- the y axis",
                "shannon_h": "Shannon domain entropy H (bits) -- the x axis",
                "n_domains": "unique exposome domains in the set",
                "dominant_domain": "most frequent domain in the set",
            },
            notes=("Only candidates of the best model level per arm are plotted. The fitted line "
                   "and its 95% band come from the regression in the companion stats sheet."),
        ))

        stat_rows = []
        for objective, arm in (("o_max", "Max O-info"), ("o_min", "Min O-info")):
            grp = sub[sub["objective"] == objective].dropna(subset=["shannon_h", "global_oof_r2"])
            if len(grp) < 10 or grp["shannon_h"].std() <= 0:
                continue
            lr = _scipy_stats.linregress(grp["shannon_h"], grp["global_oof_r2"])
            stat_rows.append({
                "objective": objective, "arm_label": arm, "n_candidates": int(len(grp)),
                "slope_beta_h": float(lr.slope), "intercept": float(lr.intercept),
                "pearson_r": float(lr.rvalue), "p_value": float(lr.pvalue),
                "stderr": float(lr.stderr),
            })
        panels.append(Panel(
            panel_id=f"{letter}1s_{short}_diversity_stats",
            frame=pd.DataFrame(stat_rows),
            description=(f"{bag.capitalize()} BAG: least-squares fit of held-out LOCO R² on Shannon "
                         "domain entropy H, per arm -- the line and statistics annotated on the scatter."),
            columns={
                "objective": _OBJ_GLOSS, "arm_label": "arm as labelled in the panel legend",
                "n_candidates": "candidates entering the fit",
                "slope_beta_h": "slope on Shannon domain entropy H (R² per bit)",
                "intercept": "fitted intercept", "pearson_r": "Pearson correlation coefficient",
                "p_value": "exact two-sided p-value for the slope",
                "stderr": "standard error of the slope",
            },
            test="Two-sided t-test on the slope of an ordinary least-squares regression (scipy.stats.linregress).",
            notes="p_value is the exact value returned by the test; it is never rounded to or reported as a bound.",
        ))

        panels.append(Panel(
            panel_id=f"{letter}2_{short}_domain_composition",
            frame=recipe_df[recipe_df["bag"] == bag][
                ["objective", "rung", "model_id", "model_rank", "global_oof_r2",
                 "order", "variable", "domain"]].reset_index(drop=True),
            description=(f"{bag.capitalize()} BAG domain composition: every predictor of the top-20 "
                         "candidates per arm, with the domain it belongs to."),
            columns={
                "objective": _OBJ_GLOSS, "rung": "model level the candidates were drawn from",
                "model_id": "identifier of the exposure set",
                "model_rank": "rank of that candidate within its arm, 1 = highest R²",
                "global_oof_r2": "held-out LOCO R² of that candidate",
                "order": "set size (number of exposures)",
                "variable": "exposome predictor name", "domain": "exposome domain of that predictor",
            },
            notes="One row per predictor per candidate, so a candidate of set size k contributes k rows.",
        ))

        for family, col in (("redundancy", "3"), ("synergy", "4")):
            net = nets[(bag, family)]
            panels.append(Panel(
                panel_id=f"{letter}{col}_{short}_{family[:3]}_net_nodes",
                frame=net.nodes.reset_index(drop=True),
                description=(f"{bag.capitalize()} BAG {family} domain network: node sizes, i.e. how "
                             "often each domain appears across the top-20 candidates."),
                columns={"node size reflects the domain's frequency across the plotted candidates": ""},
                notes="Node and edge sheets together define the drawn network.",
            ))
            panels.append(Panel(
                panel_id=f"{letter}{col}_{short}_{family[:3]}_net_edges",
                frame=net.edges.reset_index(drop=True),
                description=(f"{bag.capitalize()} BAG {family} domain network: edge weights, i.e. how "
                             "often each domain pair co-occurs within a candidate."),
                columns={"edge width and colour are binned on the weight, shared across all four networks": ""},
                notes="Edge width bins and the colour maximum are computed across every network in the figure so panels are comparable.",
            ))
    return panels


def _render_capped() -> None:
    """Capped, no-SHAP fig3: scatter | recipe | redundancy network | synergy network."""
    bags_available = CAP_BAGS or list(BAGS)
    bags = [b for b in ("structural", "functional") if b in bags_available] or bags_available
    feature_domains = _load_feature_domains()
    scatter_df, recipe_df = _build_cap_tables(bags, feature_domains)

    nets = {}
    edge_frames = []
    for bag in bags:
        for family in ("synergy", "redundancy"):
            net = _build_domain_network_from_recipe(recipe_df, bag, family)
            nets[(bag, family)] = net
            edge_frames.append(net.edges)
    edge_width_bins = _edge_width_bins_from_edge_frames(edge_frames)
    edge_color_max = _edge_color_max_from_edge_frames(edge_frames)

    nrows = len(bags)
    fig, axes = plt.subplots(nrows, 4, figsize=(26, 6.2 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.28)
    # Shrink ONLY the gap between the two network columns (2, 3), without
    # touching col widths or the col0/col1 spacing: widen col3 to fill the
    # space it gives up by moving its left edge next to col2, keeping its
    # right edge fixed so the rest of the figure is untouched.
    for ri in range(nrows):
        pos2 = axes[ri][2].get_position()
        pos3 = axes[ri][3].get_position()
        gap = 0.015
        new_x0_3 = pos2.x1 + gap
        axes[ri][3].set_position([new_x0_3, pos3.y0, pos3.x1 - new_x0_3, pos3.height])

    empty_reg = pd.DataFrame()
    for ri, bag in enumerate(bags):
        draw_scatter(axes[ri][0], scatter_df, empty_reg, bag)
        # draw_scatter's stats box reads like "β_H red=0.0038 (r=0.10, p=0.01)"
        # per line, one line per objective (red=o_max, syn=o_min). Parse the
        # (r=..., p=...) part per objective, drop the stray text box, and fold
        # each objective's stats into its own point-legend label so there is a
        # single legend that reads e.g. "Min O-info (r=0.52, p<0.001)".
        stats_by_label: dict[str, str] = {}
        for _txt in list(axes[ri][0].texts):
            if "β_H" in _txt.get_text():
                for line in _txt.get_text().split("\n"):
                    m = re.match(r"^β_H\s+(\S+)=\S+\s+(\(r=.*\))$", line)
                    if m:
                        stats_by_label[m.group(1)] = m.group(2)
                _txt.remove()
        legend = axes[ri][0].get_legend()
        if legend is not None:
            handles = legend.legend_handles
            new_labels = []
            for _lbl in legend.get_texts():
                if _lbl.get_text() == "Synergistic":
                    stats = stats_by_label.get("syn", "")
                    new_labels.append(f"Min O-info {stats}".rstrip())
                elif _lbl.get_text() == "Redundant":
                    stats = stats_by_label.get("red", "")
                    new_labels.append(f"Max O-info {stats}".rstrip())
                else:
                    new_labels.append(_lbl.get_text())
            legend.remove()
            axes[ri][0].legend(
                handles=handles, labels=new_labels,
                loc="lower center", frameon=False,
                fontsize=FS_TK - 1 + FONT_BUMP,
                handletextpad=0.4, borderaxespad=0.2, labelspacing=0.2,
            )
        # Make room at the bottom for the relocated legend by lowering ylim.
        ylo, yhi = axes[ri][0].get_ylim()
        axes[ri][0].set_ylim(ylo - 0.12 * (yhi - ylo), yhi)
        axes[ri][0].set_ylabel(f"{BAG_LABEL[bag]}\nLOCO R²", fontsize=FS + FONT_BUMP)
        draw_recipe(axes[ri][1], recipe_df, bag)
        # Rename the recipe panel's block labels to match the panel-a legend
        # vocabulary (Min O-info / Max O-info), leaving everything else as-is.
        for _txt in axes[ri][1].texts:
            if _txt.get_text() == "Syn":
                _txt.set_text("Min O-info")
            elif _txt.get_text() == "Red":
                _txt.set_text("Max O-info")
        _draw_domain_panel_scaled(
            axes[ri][2], nets[(bag, "redundancy")],
            edge_width_bins=edge_width_bins, edge_color_max=edge_color_max,
            node_size_scale=4.8, label_fontsize=17.0,
        )
        _draw_domain_panel_scaled(
            axes[ri][3], nets[(bag, "synergy")],
            edge_width_bins=edge_width_bins, edge_color_max=edge_color_max,
            node_size_scale=4.8, label_fontsize=17.0,
        )
        # Bump every font in the scatter (col 0) and recipe (col 1) panels by
        # FONT_BUMP pt -- axis labels, tick labels, legend text, in-axes text
        # ("Min O-info"/"Max O-info" block labels) -- explicitly excluding the
        # two network panels (cols 2-3), which keep their own fixed sizes.
        for ax in (axes[ri][0], axes[ri][1]):
            for _lbl in (ax.xaxis.label, ax.yaxis.label):
                _lbl.set_fontsize(_lbl.get_fontsize() + FONT_BUMP)
            for _tick in ax.get_xticklabels() + ax.get_yticklabels():
                _tick.set_fontsize(_tick.get_fontsize() + FONT_BUMP)
            for _txt in ax.texts:
                _txt.set_fontsize(_txt.get_fontsize() + FONT_BUMP)
            _leg = ax.get_legend()
            if _leg is not None:
                for _lbl in _leg.get_texts():
                    _lbl.set_fontsize(_lbl.get_fontsize() + FONT_BUMP)
        if ri == 0:
            for ci, title in enumerate(CAP_COL_TITLES):
                # Panel a carries a two-line rotated BAG+"LOCO R²" ylabel that eats
                # into the left margin; pull its title further left so it starts
                # near the figure edge like the other panels' titles do. Panel d's
                # axis sits immediately next to panel c's (tight network gap), so
                # nudge its title right to keep clear of panel c's longer title
                # now that both are larger (TITLE_BUMP).
                if ci == 0:
                    title_x = -0.34
                elif ci == 3:
                    title_x = 0.12
                else:
                    title_x = 0.0
                axes[ri][ci].set_title(title, fontsize=18 + TITLE_BUMP, loc="left", x=title_x)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = OUT_DIR / f"fig3_diversity_v2{SUFFIX}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    for path in write_source_data(
        f"fig3_diversity_v2{SUFFIX}",
        _build_fig3_panels(scatter_df, recipe_df, nets, bags),
        OUT_DIR,
        source_paths=[
            str(ROOT / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
                / "pooled_oinfo_ladder_<bag>" / "metrics_global_long.parquet"),
            str(FEATURE_DOMAINS),
        ],
    ):
        print(f"Saved: {path}")


def main() -> None:
    if ORDER_MAX is not None:
        _render_capped()
        return
    scatter_df = pd.read_csv(STATS / "per_candidate_diversity_scatter.csv")
    recipe_df = pd.read_csv(STATS / "per_candidate_recipe_top20.csv")
    reg_df = pd.read_csv(STATS / "domain_diversity_regression.csv")

    dom_nodes_syn = pd.read_csv(STATS / "domain_cooccurrence_nodes_synergy_rung_xgb_tree_d3.csv")
    dom_edges_syn = pd.read_csv(STATS / "domain_cooccurrence_edges_synergy_rung_xgb_tree_d3.csv")
    dom_nodes_red = pd.read_csv(STATS / "domain_cooccurrence_nodes_redundancy_rung_xgb_tree_d3.csv")
    dom_edges_red = pd.read_csv(STATS / "domain_cooccurrence_edges_redundancy_rung_xgb_tree_d3.csv")

    syn_shap = {}
    red_shap = {}
    syn_values = {}
    red_values = {}
    for bag in BAGS:
        syn_shap[bag], syn_values[bag] = _load_shap_bundle(SYN_SHAP_ROOT, bag)
        red_shap[bag], red_values[bag] = _load_shap_bundle(RED_SHAP_ROOT, bag)
    feature_domains = _load_feature_domains()

    selected_rows = []
    selected_features = {}
    for bag in BAGS:
        red_features, red_selected = _select_shap_features(red_values[bag], top_n=SHAP_TOP_N)
        syn_features, syn_selected = _select_shap_features(syn_values[bag], top_n=SHAP_TOP_N)
        selected_features[bag] = {"red": red_features, "syn": syn_features}
        for family, selected_df in [("redundant", red_selected), ("synergistic", syn_selected)]:
            for _, row in selected_df.iterrows():
                row_dict = row.to_dict()
                selected_rows.append({
                    "bag": bag,
                    "family": family,
                    "domain": feature_domains.get(str(row_dict["feature"]), ""),
                    "display_label": display_label(str(row_dict["feature"])),
                    **row_dict,
                })

    all_edge_frames = [dom_edges_syn, dom_edges_red]
    edge_width_bins = _edge_width_bins_from_edge_frames(all_edge_frames)
    edge_color_max = _edge_color_max_from_edge_frames(all_edge_frames)

    fig = plt.figure(figsize=(62, max(18, 10 * len(BAGS))))
    gs = gridspec.GridSpec(
        len(BAGS),
        8,
        figure=fig,
        wspace=0.04,
        hspace=0.30,
        width_ratios=[1.0, 0.9, 1.0, 1.0, 0.38, 1.35, 0.38, 1.35],
    )
    grid_cols = [0, 1, 2, 3, 5, 7]
    axes = [[fig.add_subplot(gs[r, grid_cols[c]]) for c in range(6)] for r in range(len(BAGS))]

    idx = 0
    for ri, bag in enumerate(BAGS):
        red_features = selected_features[bag]["red"]
        syn_features = selected_features[bag]["syn"]
        red_xlim = _panel_xlim(red_values[bag], red_features)
        syn_xlim = _panel_xlim(syn_values[bag], syn_features)

        for ci in range(6):
            ax = axes[ri][ci]
            ax.text(
                -0.10 if ci in (0, 4) else -0.05,
                1.02,
                LETTERS[idx],
                transform=ax.transAxes,
                fontsize=FS,
                va="bottom",
                ha="left",
            )
            idx += 1

            if ci == 0:
                ax.set_ylabel(BAG_LABEL[bag], fontsize=FS)

            if ci == 0:
                draw_scatter(ax, scatter_df, reg_df, bag)
            elif ci == 1:
                draw_recipe(ax, recipe_df, bag)
            elif ci == 2:
                draw_network(ax, dom_nodes_red, dom_edges_red, bag, "redundancy",
                             edge_width_bins, edge_color_max)
            elif ci == 3:
                draw_network(ax, dom_nodes_syn, dom_edges_syn, bag, "synergy",
                             edge_width_bins, edge_color_max)
            elif ci == 4:
                _draw_shap_panel(
                    ax,
                    red_values[bag],
                    red_features,
                    feature_domains,
                    xlim=red_xlim,
                    show_ylabel=True,
                    show_xlabel=(ri == len(BAGS) - 1),
                    title="Redundant" if ri == 0 else "",
                )
            elif ci == 5:
                _draw_shap_panel(
                    ax,
                    syn_values[bag],
                    syn_features,
                    feature_domains,
                    xlim=syn_xlim,
                    show_ylabel=True,
                    show_xlabel=(ri == len(BAGS) - 1),
                    title="Synergistic" if ri == 0 else "",
                )

            if ci in (0, 1, 2, 3):
                if ri < len(BAGS) - 1:
                    ax.set_xticklabels([])
                elif ci == 0:
                    ax.tick_params(axis="x", labelsize=FS_TK)
            else:
                if ri < len(BAGS) - 1:
                    ax.set_xticklabels([])

    for ext in ("pdf", "svg"):
        out = PAPER_FIGS / f"fig3_diversity_v2.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    fig.savefig(PAPER_FIGS / "fig3_diversity_v2.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {PAPER_FIGS / 'fig3_diversity_v2.png'}")
    plt.close(fig)

    scatter_df.to_csv(PAPER_FIGS / "fig3_diversity_v2_scatter.csv", index=False)
    recipe_df.to_csv(PAPER_FIGS / "fig3_diversity_v2_recipe_top20.csv", index=False)
    reg_df[reg_df["regression"] == "R2_diversity"].to_csv(
        PAPER_FIGS / "fig3_diversity_v2_regression.csv", index=False
    )
    pd.concat([dom_nodes_syn, dom_nodes_red], ignore_index=True).to_csv(
        PAPER_FIGS / "fig3_diversity_v2_cooccurrence_nodes.csv", index=False
    )
    pd.concat([dom_edges_syn, dom_edges_red], ignore_index=True).to_csv(
        PAPER_FIGS / "fig3_diversity_v2_cooccurrence_edges.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(PAPER_FIGS / "fig3_diversity_v2_shap_selected.csv", index=False)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
