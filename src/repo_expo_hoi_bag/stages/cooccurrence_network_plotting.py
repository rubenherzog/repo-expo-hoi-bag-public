from __future__ import annotations

import logging
import math
import textwrap
from collections import Counter
from itertools import combinations
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from exposome_labels import display_label
from oinfo_bag_ladder.config import FIGURE_DPI
from oinfo_bag_ladder.cooccurrence_network import (
    BAG_LABELS,
    DEFAULT_BAG_ORDER,
    DEFAULT_FAMILY_ORDER,
    DomainNetwork,
    FAMILY_OBJECTIVE,
    PanelNetwork,
    build_domain_network,
    build_panel_network,
    load_pooled_metrics,
    resolve_analysis_root,
    select_top_models,
)
from oinfo_bag_ladder.make_boss_figure import DOMAIN_COLORS, DOMAIN_ORDER
from oinfo_bag_ladder.rungs import RUNG_LABELS


LOGGER = logging.getLogger(__name__)

DETAIL_EDGE_WIDTH_LEVELS = (1.0, 1.5, 2.0, 4.0)
DOMAIN_EDGE_WIDTH_LEVELS = (1.0, 3.0, 4.0, 6.0)
EDGE_COLOR_CAP = 0.1


def _safe_display_label(name: str, width: int = 14) -> str:
    label = display_label(name)
    return textwrap.fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _node_size_from_frequency(freq: float, top_k: int) -> float:
    if top_k <= 1:
        return 800.0
    freq = float(freq)
    freq = min(max(freq, 1.0), float(top_k))
    lo, hi = 240.0, 1180.0
    return lo + (freq - 1.0) / (float(top_k) - 1.0) * (hi - lo)


def _edge_width_bins_from_values(values: Iterable[float]) -> tuple[float, float, float, float, float] | None:
    """Build equal-width bins over the observed non-zero normalized edge range."""
    arr = np.asarray([float(v) for v in values if float(v) > 0.0], dtype=float)
    if arr.size == 0:
        return None
    lo = float(arr.min())
    hi = float(arr.max())
    if math.isclose(lo, hi):
        return (lo, hi, hi, hi, hi)
    bins = np.linspace(lo, hi, num=5)
    return tuple(float(v) for v in bins)


def _edge_width_from_normalized(
    value: float,
    bins: tuple[float, float, float, float, float] | None,
    levels: tuple[float, float, float, float],
) -> float:
    """Map normalized co-occurrence to one of four discrete thickness levels."""
    value = float(value)
    if value <= 0.0:
        return 0.0
    if bins is None:
        return levels[-1]
    lo, b1, b2, b3, hi = bins
    if math.isclose(lo, hi):
        return levels[-1]
    if value <= b1:
        return levels[0]
    if value <= b2:
        return levels[1]
    if value <= b3:
        return levels[2]
    return levels[3]


def _edge_width_legend_handles(
    bins: tuple[float, float, float, float, float] | None,
    levels: tuple[float, float, float, float],
) -> list[mlines.Line2D]:
    """Create legend handles for the four discrete edge-width bins."""
    if bins is None:
        return [
            mlines.Line2D([], [], color="#666666", linewidth=level, label=f"{level:.2f}: [n/a, n/a]")
            for level in levels
        ]
    lo, b1, b2, b3, hi = bins
    ranges = [
        (max(0.0, lo), b1),
        (b1, b2),
        (b2, b3),
        (b3, hi),
    ]
    return [
        mlines.Line2D(
            [],
            [],
            color="#666666",
            linewidth=level,
            label=f"{level:.2f}: [{low:.2f}, {high:.2f}]",
        )
        for level, (low, high) in zip(levels, ranges)
    ]


def _edge_width_bins_from_edge_frames(edge_frames: Iterable[pd.DataFrame]) -> tuple[float, float, float, float, float] | None:
    """Derive shared equal-width bins from the observed normalized edge values."""
    values: list[float] = []
    for frame in edge_frames:
        if frame.empty or "normalized_cooccurrence" not in frame.columns:
            continue
        values.extend(float(v) for v in frame["normalized_cooccurrence"].to_numpy() if float(v) > 0.0)
    return _edge_width_bins_from_values(values)


def _edge_color_max_from_edge_frames(
    edge_frames: Iterable[pd.DataFrame],
    *,
    cap: float | None = EDGE_COLOR_CAP,
) -> float:
    """Get the observed max normalized co-occurrence across a figure's edge tables."""
    values: list[float] = []
    for frame in edge_frames:
        if frame.empty or "normalized_cooccurrence" not in frame.columns:
            continue
        values.extend(float(v) for v in frame["normalized_cooccurrence"].to_numpy() if float(v) > 0.0)
    if not values:
        return float(cap) if cap is not None else 1.0
    observed = float(max(values))
    if cap is None:
        return observed
    return float(min(observed, cap))


def _edge_color_from_normalized(value: float, max_normalized: float, family: str) -> tuple[float, float, float, float]:
    """Map normalized co-occurrence to a family-specific color gradient."""
    max_normalized = max(float(max_normalized), 1.0e-12)
    norm = float(min(max(float(value) / max_normalized, 0.0), 1.0))
    cmap_name = "Blues" if family == "synergy" else "Reds"
    rgba = mpl.colormaps[cmap_name](norm)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(0.30 + 0.60 * norm))


def _add_edge_colorbar(ax: plt.Axes, *, family: str, max_normalized: float) -> None:
    """Add a compact normalized co-occurrence colorbar inside an axes without changing layout."""
    cmap_name = "Blues" if family == "synergy" else "Reds"
    max_normalized = max(float(max_normalized), 1.0e-12)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max_normalized)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps[cmap_name])
    sm.set_array([])

    cax = inset_axes(
        ax,
        width="3.0%",
        height="42%",
        loc="center left",
        bbox_to_anchor=(1.08, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cb = plt.colorbar(sm, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=11, length=2, width=0.6)
    cb.set_label("Normalized co-occurrence", fontsize=12)
    cb.set_ticks([0.0, max_normalized / 2.0, max_normalized])


def _domain_node_size_from_within_count(count: float, top_k: int) -> float:
    """Node size for reduced domain networks based on within-domain co-occurrence mass."""
    count = max(float(count), 0.0)
    lo, hi = 220.0, 1400.0
    max_ref = max(1.0, float(top_k) * 12.0)
    scaled = min(count, max_ref) / max_ref
    return lo + scaled * (hi - lo)


def apply_visual_encodings(
    panel_network: PanelNetwork,
    *,
    edge_width_bins: tuple[float, float, float, float, float] | None = None,
    edge_width_levels: tuple[float, float, float, float] = DETAIL_EDGE_WIDTH_LEVELS,
) -> PanelNetwork:
    """Populate node_size_used and edge_width_used columns."""
    nodes = panel_network.nodes.copy()
    edges = panel_network.edges.copy()

    if not nodes.empty:
        nodes["node_size_used"] = nodes["frequency_topk"].apply(lambda v: _node_size_from_frequency(v, panel_network.top_k))
        nodes["node_edgecolor"] = np.where(nodes["in_top1_model"], "#111111", "#666666")
        nodes["node_linewidth"] = np.where(nodes["in_top1_model"], 2.4, 0.9)
    if not edges.empty:
        edges["edge_width_used"] = edges["normalized_cooccurrence"].apply(
            lambda v: _edge_width_from_normalized(v, edge_width_bins, edge_width_levels)
        )
        edges["edge_alpha"] = edges["cooccurrence_count"].apply(
            lambda v: float(
                0.20
                + 0.65
                * (
                    (min(float(v), float(panel_network.top_k)) - 1.0)
                    / (float(panel_network.top_k) - 1.0 if panel_network.top_k > 1 else 1.0)
                )
            )
        )

    return PanelNetwork(
        bag=panel_network.bag,
        family=panel_network.family,
        rung=panel_network.rung,
        top_k=panel_network.top_k,
        models=panel_network.models,
        nodes=nodes,
        edges=edges,
        top1_model_id=panel_network.top1_model_id,
        top1_vars=panel_network.top1_vars,
    )


def apply_domain_visual_encodings(
    domain_network: DomainNetwork,
    *,
    edge_width_bins: tuple[float, float, float, float, float] | None = None,
    edge_width_levels: tuple[float, float, float, float] = DOMAIN_EDGE_WIDTH_LEVELS,
) -> DomainNetwork:
    """Populate node_size_used and edge_width_used columns for reduced domain networks."""
    nodes = domain_network.nodes.copy()
    edges = domain_network.edges.copy()

    if not nodes.empty:
        nodes["node_size_used"] = nodes["within_domain_pair_count"].apply(
            lambda v: _domain_node_size_from_within_count(v, domain_network.top_k)
        )
        nodes["node_edgecolor"] = np.where(nodes["in_top1_model"], "#111111", "#666666")
        nodes["node_linewidth"] = np.where(nodes["in_top1_model"], 2.4, 0.9)
    if not edges.empty:
        edges["edge_width_used"] = edges["normalized_cooccurrence"].apply(
            lambda v: _edge_width_from_normalized(v, edge_width_bins, edge_width_levels)
        )
        edges["edge_alpha"] = edges["cooccurrence_count"].apply(
            lambda v: float(
                0.18
                + 0.72
                * (
                    (min(float(v), float(domain_network.top_k)) - 1.0)
                    / (float(domain_network.top_k) - 1.0 if domain_network.top_k > 1 else 1.0)
                )
            )
        )

    return DomainNetwork(
        bag=domain_network.bag,
        family=domain_network.family,
        rung=domain_network.rung,
        top_k=domain_network.top_k,
        models=domain_network.models,
        nodes=nodes,
        edges=edges,
        top1_model_id=domain_network.top1_model_id,
        top1_domains=domain_network.top1_domains,
    )


def _build_positions(order: list[str]) -> dict[str, tuple[float, float, float]]:
    """Map each variable to an angle and Cartesian position on the unit circle."""
    if not order:
        return {}
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, num=len(order), endpoint=False)
    out: dict[str, tuple[float, float, float]] = {}
    for var, angle in zip(order, angles):
        out[var] = (float(np.cos(angle)), float(np.sin(angle)), float(angle))
    return out


def _text_color_for_fill(fill_color: str) -> str:
    rgba = to_rgba(fill_color)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#111111" if luminance > 0.55 else "#ffffff"


def _format_r2(value: float) -> str:
    return f"R\u00b2={float(value):.3f}"


def _figure_context_from_output_root(output_root: str | Path) -> tuple[str, str]:
    name = Path(output_root).name
    match = re.search(r"(variant_[a-z0-9]+)", name, flags=re.IGNORECASE)
    variant_label = match.group(1).replace("_", " ").title() if match else name.replace("_", " ").title()
    otype = "O & Diversity" if "domain_dual_combined" in name else "O"
    return otype, variant_label


def _best_rung_for_panel(
    analysis_root: str | Path,
    bag: str,
    family: str,
    *,
    rung_override: str | None = None,
) -> str:
    df = load_pooled_metrics(analysis_root, bag, rung="all")
    sub = df[df["bag_target"].astype(str) == str(bag)].copy()
    sub["full_r2"] = pd.to_numeric(sub["full_r2"], errors="coerce")
    sub = sub.dropna(subset=["full_r2"])
    sub = sub[sub["objective"].astype(str) == FAMILY_OBJECTIVE[family]].copy()
    if rung_override is not None:
        sub = sub[sub["rung_id"].astype(str) == str(rung_override)].copy()
    if sub.empty:
        raise ValueError(f"No models available for bag={bag!r}, family={family!r}")
    sub = sub.sort_values(by=["full_r2", "candidate_id"], ascending=[False, True], kind="mergesort")
    return str(sub.iloc[0]["rung_id"])


def _build_best_panel_network(
    analysis_root: str | Path,
    bag: str,
    family: str,
    top_k: int,
    domain_map: dict[str, str],
    *,
    rung_override: str | None = None,
) -> PanelNetwork:
    best_rung = _best_rung_for_panel(analysis_root, bag, family, rung_override=rung_override)
    return build_panel_network(analysis_root, bag, family, best_rung, top_k, domain_map)


def family_variable_order(panel_networks: Iterable[PanelNetwork]) -> list[str]:
    """Return a deterministic order of variables shared across a family figure."""
    rows: list[pd.DataFrame] = [p.nodes for p in panel_networks if not p.nodes.empty]
    if not rows:
        return []
    all_nodes = pd.concat(rows, ignore_index=True)
    agg = (
        all_nodes.groupby("variable", as_index=False, dropna=False)
        .agg(
            domain=("domain", lambda s: str(s.iloc[0]) if len(s) else "Other"),
            frequency_topk=("frequency_topk", "sum"),
            display_label=("display_label", lambda s: str(s.iloc[0]) if len(s) else ""),
        )
        .copy()
    )
    agg["domain_rank"] = agg["domain"].map(lambda d: DOMAIN_ORDER.index(d) if d in DOMAIN_ORDER else len(DOMAIN_ORDER))
    agg = agg.sort_values(
        by=["domain_rank", "frequency_topk", "display_label", "variable"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return agg["variable"].astype(str).tolist()


def _draw_panel(
    ax: plt.Axes,
    panel: PanelNetwork,
    order: list[str],
    *,
    edge_min_count: int,
    edge_max_plot: int | None,
    show_colorbar: bool = False,
    edge_width_bins: tuple[float, float, float, float, float] | None = None,
    edge_color_max: float | None = None,
    panel_title: str | None = None,
) -> dict[str, object]:
    panel = apply_visual_encodings(panel, edge_width_bins=edge_width_bins)
    node_df = panel.nodes.copy()
    edge_df = panel.edges.copy()

    positions = _build_positions(order)
    present_set = set(node_df["variable"].astype(str).tolist())
    present_nodes = [v for v in order if v in present_set]
    pos_subset = {v: positions[v] for v in present_nodes if v in positions}

    plot_edges = edge_df[edge_df["cooccurrence_count"] >= int(edge_min_count)].copy()
    plot_edges = plot_edges.sort_values(
        by=["cooccurrence_count", "var_a", "var_b"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if edge_max_plot is not None and int(edge_max_plot) > 0 and len(plot_edges) > int(edge_max_plot):
        plot_edges = plot_edges.head(int(edge_max_plot)).copy()
    hidden_edge_n = int(len(edge_df) - len(plot_edges))

    ax.set_aspect("equal")
    ax.axis("off")

    if not plot_edges.empty:
        segments = []
        colors = []
        widths = []
        for _, edge in plot_edges.iterrows():
            a = str(edge["var_a"])
            b = str(edge["var_b"])
            if a not in pos_subset or b not in pos_subset:
                continue
            x1, y1, _ = pos_subset[a]
            x2, y2, _ = pos_subset[b]
            segments.append([(x1, y1), (x2, y2)])
            colors.append(
                _edge_color_from_normalized(
                    edge.get("normalized_cooccurrence", 0.0),
                    edge_color_max if edge_color_max is not None else 1.0,
                    panel.family,
                )
            )
            widths.append(float(edge.get("edge_width_used", 1.0)))
        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=1, capstyle="round")
            ax.add_collection(lc)
        if show_colorbar:
            _add_edge_colorbar(ax, family=panel.family, max_normalized=edge_color_max if edge_color_max is not None else 1.0)

    if not node_df.empty:
        normal_x, normal_y, normal_sizes, normal_fills = [], [], [], []
        top1_x, top1_y, top1_sizes, top1_fills = [], [], [], []
        for _, row in node_df.iterrows():
            var = str(row["variable"])
            if var not in pos_subset:
                continue
            x, y, _ = pos_subset[var]
            if bool(row.get("in_top1_model", False)):
                top1_x.append(x)
                top1_y.append(y)
                top1_sizes.append(float(row["node_size_used"]))
                top1_fills.append(DOMAIN_COLORS.get(str(row["domain"]), DOMAIN_COLORS.get("Other", "#cccccc")))
            else:
                normal_x.append(x)
                normal_y.append(y)
                normal_sizes.append(float(row["node_size_used"]))
                normal_fills.append(DOMAIN_COLORS.get(str(row["domain"]), DOMAIN_COLORS.get("Other", "#cccccc")))

        if normal_x:
            ax.scatter(
                normal_x,
                normal_y,
                s=normal_sizes,
                c=normal_fills,
                edgecolors="#666666",
                linewidths=0.9,
                alpha=0.96,
                zorder=3,
            )
        if top1_x:
            ax.scatter(
                top1_x,
                top1_y,
                s=top1_sizes,
                c=top1_fills,
                edgecolors="#111111",
                linewidths=2.4,
                alpha=0.96,
                zorder=3,
            )

        for _, row in node_df.iterrows():
            var = str(row["variable"])
            if var not in pos_subset:
                continue
            x, y, angle = pos_subset[var]
            label = str(row["display_label"])
            font_size = 8.6
            if row["frequency_topk"] <= max(2, panel.top_k // 4):
                font_size = 7.4
            elif row["frequency_topk"] >= max(8, panel.top_k // 2):
                font_size = 9.2
            fill = DOMAIN_COLORS.get(str(row["domain"]), "#cccccc")
            txt_color = _text_color_for_fill(fill)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                color=txt_color,
                zorder=4,
                path_effects=[
                    patheffects.withStroke(
                        linewidth=2.0,
                        foreground="#ffffff" if txt_color == "#111111" else "#111111",
                    )
                ],
            )

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    if panel_title is None:
        panel_title = f"{BAG_LABELS.get(panel.bag, panel.bag.title())} - {panel.family.title()} - Rung {RUNG_LABELS.get(panel.rung, panel.rung)} - Top {panel.top_k}"
    ax.set_title(panel_title, fontsize=19, pad=12)

    if node_df.empty:
        ax.text(0.5, 0.5, "No models", ha="center", va="center", transform=ax.transAxes, fontsize=11)

    stats = {
        "n_nodes": int(len(node_df)),
        "n_edges_all": int(len(edge_df)),
        "n_edges_plotted": int(len(plot_edges)),
        "n_edges_hidden": hidden_edge_n,
        "top1_model_id": panel.top1_model_id,
        "top1_vars": list(panel.top1_vars),
        "top1_model_r2": float(panel.models.iloc[0]["full_r2"]) if not panel.models.empty else np.nan,
    }
    return stats


def _draw_domain_panel(
    ax: plt.Axes,
    panel: DomainNetwork,
    *,
    edge_min_count: int,
    edge_max_plot: int | None,
    show_colorbar: bool = False,
    edge_width_bins: tuple[float, float, float, float, float] | None = None,
    edge_color_max: float | None = None,
    panel_title: str | None = None,
) -> dict[str, object]:
    panel = apply_domain_visual_encodings(panel, edge_width_bins=edge_width_bins)
    node_df = panel.nodes.copy()
    edge_df = panel.edges.copy()

    order = [d for d in DOMAIN_ORDER if d in set(node_df["domain"].astype(str).tolist())]
    positions = _build_positions(order)
    present_set = set(node_df["domain"].astype(str).tolist())
    present_nodes = [d for d in order if d in present_set]
    pos_subset = {d: positions[d] for d in present_nodes if d in positions}

    plot_edges = edge_df[edge_df["cooccurrence_count"] >= int(edge_min_count)].copy()
    plot_edges = plot_edges.sort_values(
        by=["cooccurrence_count", "domain_a", "domain_b"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if edge_max_plot is not None and int(edge_max_plot) > 0 and len(plot_edges) > int(edge_max_plot):
        plot_edges = plot_edges.head(int(edge_max_plot)).copy()
    hidden_edge_n = int(len(edge_df) - len(plot_edges))

    ax.set_aspect("equal")
    ax.axis("off")

    if not plot_edges.empty:
        segments = []
        colors = []
        widths = []
        for _, edge in plot_edges.iterrows():
            a = str(edge["domain_a"])
            b = str(edge["domain_b"])
            if a not in pos_subset or b not in pos_subset:
                continue
            x1, y1, _ = pos_subset[a]
            x2, y2, _ = pos_subset[b]
            segments.append([(x1, y1), (x2, y2)])
            colors.append(
                _edge_color_from_normalized(
                    edge.get("normalized_cooccurrence", 0.0),
                    edge_color_max if edge_color_max is not None else 1.0,
                    panel.family,
                )
            )
            widths.append(float(edge.get("edge_width_used", 1.0)))
        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=1, capstyle="round")
            ax.add_collection(lc)
        if show_colorbar:
            _add_edge_colorbar(ax, family=panel.family, max_normalized=edge_color_max if edge_color_max is not None else 1.0)

    if not node_df.empty:
        normal_x, normal_y, normal_sizes, normal_fills = [], [], [], []
        top1_x, top1_y, top1_sizes, top1_fills = [], [], [], []
        for _, row in node_df.iterrows():
            domain = str(row["domain"])
            if domain not in pos_subset:
                continue
            x, y, _ = pos_subset[domain]
            if bool(row.get("in_top1_model", False)):
                top1_x.append(x)
                top1_y.append(y)
                top1_sizes.append(float(row["node_size_used"]))
                top1_fills.append(DOMAIN_COLORS.get(domain, DOMAIN_COLORS.get("Other", "#cccccc")))
            else:
                normal_x.append(x)
                normal_y.append(y)
                normal_sizes.append(float(row["node_size_used"]))
                normal_fills.append(DOMAIN_COLORS.get(domain, DOMAIN_COLORS.get("Other", "#cccccc")))

        if normal_x:
            ax.scatter(
                normal_x,
                normal_y,
                s=normal_sizes,
                c=normal_fills,
                edgecolors="#666666",
                linewidths=0.9,
                alpha=0.96,
                zorder=3,
            )
        if top1_x:
            ax.scatter(
                top1_x,
                top1_y,
                s=top1_sizes,
                c=top1_fills,
                edgecolors="#111111",
                linewidths=2.4,
                alpha=0.96,
                zorder=3,
            )

        for _, row in node_df.iterrows():
            domain = str(row["domain"])
            if domain not in pos_subset:
                continue
            x, y, _ = pos_subset[domain]
            label = _safe_display_label(domain, width=18)
            font_size = 11.0
            if row["within_domain_pair_count"] == 0:
                font_size = 10.0
            fill = DOMAIN_COLORS.get(domain, "#cccccc")
            txt_color = _text_color_for_fill(fill)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                color=txt_color,
                zorder=4,
                path_effects=[
                    patheffects.withStroke(
                        linewidth=2.0,
                        foreground="#ffffff" if txt_color == "#111111" else "#111111",
                    )
                ],
            )

    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)
    if panel_title is None:
        panel_title = f"{BAG_LABELS.get(panel.bag, panel.bag.title())} - {panel.family.title()} - Rung {RUNG_LABELS.get(panel.rung, panel.rung)} - Top {panel.top_k}"
    ax.set_title(panel_title, fontsize=19, pad=12)

    if node_df.empty:
        ax.text(0.5, 0.5, "No models", ha="center", va="center", transform=ax.transAxes, fontsize=11)

    stats = {
        "n_nodes": int(len(node_df)),
        "n_edges_all": int(len(edge_df)),
        "n_edges_plotted": int(len(plot_edges)),
        "n_edges_hidden": hidden_edge_n,
        "top1_model_id": panel.top1_model_id,
        "top1_domains": list(panel.top1_domains),
        "top1_model_r2": float(panel.models.iloc[0]["full_r2"]) if not panel.models.empty else np.nan,
    }
    return stats


def plot_family_figure(
    panels: dict[str, PanelNetwork],
    *,
    family: str,
    rung: str,
    top_k: int,
    output_root: str | Path,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
) -> dict[str, Path]:
    """Plot a three-panel BAG figure for one family."""
    root = Path(output_root)
    figure_dir = root
    figure_dir.mkdir(parents=True, exist_ok=True)

    ordered_panels = [panels[bag] for bag in DEFAULT_BAG_ORDER if bag in panels]
    order = family_variable_order(ordered_panels)
    if not order:
        raise ValueError(f"No variables to plot for family={family!r}, rung={rung!r}")
    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in ordered_panels])
    edge_color_max = _edge_color_max_from_edge_frames([panel.edges for panel in ordered_panels], cap=None)

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.6), constrained_layout=False)
    if len(ordered_panels) == 1:
        axes = np.asarray([axes])

    panel_stats = []
    for ax, bag in zip(axes, DEFAULT_BAG_ORDER):
        if bag not in panels:
            ax.axis("off")
            continue
        stats = _draw_panel(
            ax,
            panels[bag],
            order,
            edge_min_count=edge_min_count,
            edge_max_plot=edge_max_plot,
            show_colorbar=(bag == DEFAULT_BAG_ORDER[-1]),
            edge_width_bins=edge_width_bins,
            edge_color_max=edge_color_max,
        )
        panel_stats.append((bag, stats))

    family_title = "Synergy" if family == "synergy" else "Redundancy"
    fig.suptitle(
        f"Top-model co-occurrence network - {family_title} - Rung {RUNG_LABELS.get(rung, rung)} - Top {top_k}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    domain_handles = [
        mpatches.Patch(facecolor=DOMAIN_COLORS.get(domain, "#cccccc"), edgecolor="#111111", label=domain)
        for domain in DOMAIN_ORDER
        if domain in DOMAIN_COLORS
    ]
    freq_small = max(1, int(round(top_k * 0.10)))
    edge_handles = _edge_width_legend_handles(edge_width_bins, DETAIL_EDGE_WIDTH_LEVELS)
    node_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=math.sqrt(_node_size_from_frequency(freq_small, top_k)) / 1.3,
            markerfacecolor="#d9d9d9",
            markeredgecolor="#666666",
            label=f"Node size: frequency in top-{top_k}",
        ),
    ]
    border_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="#d9d9d9",
        markeredgecolor="#111111",
        markeredgewidth=2.4,
        label="Thick border: variable is in the best model",
    )

    legend_handles = [*domain_handles, *node_handles, *edge_handles, border_handle]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("#444444")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])

    png_path = figure_dir / f"top_model_cooccurrence_network_{family}_rung_{rung}.png"
    pdf_path = figure_dir / f"top_model_cooccurrence_network_{family}_rung_{rung}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def plot_domain_family_figure(
    panels: dict[str, DomainNetwork],
    *,
    family: str,
    rung: str,
    top_k: int,
    output_root: str | Path,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
) -> dict[str, Path]:
    """Plot a three-panel BAG figure for the reduced domain network."""
    root = Path(output_root)
    figure_dir = root
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 7.1), constrained_layout=False)
    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in panels.values()])
    edge_color_max = _edge_color_max_from_edge_frames([panel.edges for panel in panels.values()])

    for ax, bag in zip(axes, DEFAULT_BAG_ORDER):
        if bag not in panels:
            ax.axis("off")
            continue
        _draw_domain_panel(
            ax,
            panels[bag],
            edge_min_count=edge_min_count,
            edge_max_plot=edge_max_plot,
            show_colorbar=(bag == DEFAULT_BAG_ORDER[-1]),
            edge_width_bins=edge_width_bins,
            edge_color_max=edge_color_max,
        )

    family_title = "Synergy" if family == "synergy" else "Redundancy"
    fig.suptitle(
        f"Domain co-occurrence network - {family_title} - Rung {RUNG_LABELS.get(rung, rung)} - Top {top_k}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    node_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor="#d9d9d9",
            markeredgecolor="#666666",
            label="Node size: within-domain co-occurrence mass",
        ),
    ]
    edge_handles = _edge_width_legend_handles(edge_width_bins, DOMAIN_EDGE_WIDTH_LEVELS)
    border_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="#d9d9d9",
        markeredgecolor="#111111",
        markeredgewidth=2.4,
        label="Thick border: domain appears in the best model",
    )

    legend_handles = [*node_handles, border_handle, *edge_handles, ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("#444444")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])

    png_path = figure_dir / f"top_model_domain_cooccurrence_network_{family}_rung_{rung}.png"
    pdf_path = figure_dir / f"top_model_domain_cooccurrence_network_{family}_rung_{rung}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def _row_label_axes(fig: plt.Figure, axes: np.ndarray, row: int, label: str) -> None:
    row_axes = axes[row, :]
    left = min(ax.get_position().x0 for ax in row_axes)
    y0 = min(ax.get_position().y0 for ax in row_axes)
    y1 = max(ax.get_position().y1 for ax in row_axes)
    fig.text(
        left - 0.025,
        (y0 + y1) / 2.0,
        label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )


def _bag_panel_label(bag: str) -> str:
    label = BAG_LABELS.get(bag, bag.title())
    return label.replace(" BAG", "")


def plot_combined_variable_figure(
    panels_by_family: dict[str, dict[str, PanelNetwork]],
    *,
    output_root: str | Path,
    variant_label: str,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
) -> dict[str, Path]:
    """Plot one detailed figure per variant with redundancy on top and synergy below."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    row_order = ["redundancy", "synergy"]
    panel_order = DEFAULT_BAG_ORDER
    all_panels = [
        panels_by_family[family][bag]
        for family in row_order
        for bag in panel_order
        if family in panels_by_family and bag in panels_by_family[family]
    ]
    top_k_ref = all_panels[0].top_k
    rung_label = RUNG_LABELS.get(all_panels[0].rung, all_panels[0].rung)
    order = family_variable_order(all_panels)
    if not order:
        raise ValueError("No variables to plot for combined detailed figure")

    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in all_panels])
    edge_color_max = _edge_color_max_from_edge_frames([panel.edges for panel in all_panels], cap=None)

    fig, axes = plt.subplots(2, 3, figsize=(20.8, 12.6), constrained_layout=False)
    panel_stats: list[tuple[str, str, dict[str, object]]] = []

    for row_idx, family in enumerate(row_order):
        if family not in panels_by_family:
            for ax in axes[row_idx, :]:
                ax.axis("off")
            continue
        for col_idx, bag in enumerate(panel_order):
            if bag not in panels_by_family[family]:
                axes[row_idx, col_idx].axis("off")
                continue
            panel = panels_by_family[family][bag]
            r2 = float(panel.models.iloc[0]["full_r2"]) if not panel.models.empty else np.nan
            if row_idx == 0:
                panel_title = f"{_bag_panel_label(bag)}\n{rung_label} · {_format_r2(r2)}"
            else:
                panel_title = f"{rung_label} · {_format_r2(r2)}"
            stats = _draw_panel(
                axes[row_idx, col_idx],
                panel,
                order,
                edge_min_count=edge_min_count,
                edge_max_plot=edge_max_plot,
                show_colorbar=(col_idx == len(panel_order) - 1),
                edge_width_bins=edge_width_bins,
                edge_color_max=edge_color_max,
                panel_title=panel_title,
            )
            panel_stats.append((family, bag, stats))

    otype, variant_name = _figure_context_from_output_root(output_root)
    fig.suptitle(f"{otype} - {variant_name}", fontsize=16, fontweight="bold", y=0.985)

    domain_handles = [
        mpatches.Patch(facecolor=DOMAIN_COLORS.get(domain, "#cccccc"), edgecolor="#111111", label=domain)
        for domain in DOMAIN_ORDER
        if domain in DOMAIN_COLORS
    ]
    freq_small = max(1, int(round(top_k_ref * 0.10)))
    edge_handles = _edge_width_legend_handles(edge_width_bins, DETAIL_EDGE_WIDTH_LEVELS)
    node_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=math.sqrt(_node_size_from_frequency(freq_small, top_k_ref)) / 1.3,
            markerfacecolor="#d9d9d9",
            markeredgecolor="#666666",
            label=f"Node size: frequency in top-{top_k_ref}",
        ),
    ]
    border_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="#d9d9d9",
        markeredgecolor="#111111",
        markeredgewidth=2.4,
        label="Thick border: variable is in the best model",
    )

    legend_handles = [*domain_handles, *node_handles, *edge_handles, border_handle]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=11.5,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("#444444")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0.05, 0.03, 1.0, 0.98])
    _row_label_axes(fig, axes, 0, "Redundancy")
    _row_label_axes(fig, axes, 1, "Synergy")
    png_path = root / f"top_model_cooccurrence_network_detailed_{Path(output_root).name}.png"
    pdf_path = root / f"top_model_cooccurrence_network_detailed_{Path(output_root).name}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_combined_domain_figure(
    panels_by_family: dict[str, dict[str, DomainNetwork]],
    *,
    output_root: str | Path,
    variant_label: str,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
) -> dict[str, Path]:
    """Plot one domain figure per variant with redundancy on top and synergy below."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    row_order = ["redundancy", "synergy"]
    panel_order = DEFAULT_BAG_ORDER
    all_panels = [
        panels_by_family[family][bag]
        for family in row_order
        for bag in panel_order
        if family in panels_by_family and bag in panels_by_family[family]
    ]
    top_k_ref = all_panels[0].top_k
    rung_label = RUNG_LABELS.get(all_panels[0].rung, all_panels[0].rung)
    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in all_panels])
    edge_color_max = _edge_color_max_from_edge_frames([panel.edges for panel in all_panels])

    fig, axes = plt.subplots(2, 3, figsize=(18.8, 11.8), constrained_layout=False)

    for row_idx, family in enumerate(row_order):
        if family not in panels_by_family:
            for ax in axes[row_idx, :]:
                ax.axis("off")
            continue
        for col_idx, bag in enumerate(panel_order):
            if bag not in panels_by_family[family]:
                axes[row_idx, col_idx].axis("off")
                continue
            panel = panels_by_family[family][bag]
            r2 = float(panel.models.iloc[0]["full_r2"]) if not panel.models.empty else np.nan
            if row_idx == 0:
                panel_title = f"{_bag_panel_label(bag)}\n{rung_label} · {_format_r2(r2)}"
            else:
                panel_title = f"{rung_label} · {_format_r2(r2)}"
            _draw_domain_panel(
                axes[row_idx, col_idx],
                panel,
                edge_min_count=edge_min_count,
                edge_max_plot=edge_max_plot,
                show_colorbar=(col_idx == len(panel_order) - 1),
                edge_width_bins=edge_width_bins,
                edge_color_max=edge_color_max,
                panel_title=panel_title,
            )

    otype, variant_name = _figure_context_from_output_root(output_root)
    fig.suptitle(f"{otype} - {variant_name}", fontsize=16, fontweight="bold", y=0.985)

    node_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor="#d9d9d9",
            markeredgecolor="#666666",
            label="Node size: within-domain co-occurrence mass",
        ),
    ]
    edge_handles = _edge_width_legend_handles(edge_width_bins, DOMAIN_EDGE_WIDTH_LEVELS)
    border_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="#d9d9d9",
        markeredgecolor="#111111",
        markeredgewidth=2.4,
        label="Thick border: domain appears in the best model",
    )

    legend_handles = [*node_handles, border_handle, *edge_handles]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=11.5,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("#444444")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0.05, 0.03, 1.0, 0.98])
    _row_label_axes(fig, axes, 0, "Redundancy")
    _row_label_axes(fig, axes, 1, "Synergy")
    png_path = root / f"top_model_domain_cooccurrence_network_{Path(output_root).name}.png"
    pdf_path = root / f"top_model_domain_cooccurrence_network_{Path(output_root).name}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def build_family_outputs(
    analysis_root: str | Path,
    output_root: str | Path,
    *,
    rung: str | None,
    family: str,
    top_k: int,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
    domain_map: dict[str, str] | None = None,
    write_figure: bool = True,
) -> tuple[dict[str, PanelNetwork], dict[str, Path]]:
    if domain_map is None:
        from scripts.exposome_domains import load_domain_map

        domain_map = load_domain_map()

    panels: dict[str, PanelNetwork] = {}

    for bag in DEFAULT_BAG_ORDER:
        panel = _build_best_panel_network(analysis_root, bag, family, top_k, domain_map, rung_override=rung)
        panels[bag] = panel

    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in panels.values()])
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for bag in DEFAULT_BAG_ORDER:
        panel = apply_visual_encodings(panels[bag], edge_width_bins=edge_width_bins, edge_width_levels=DETAIL_EDGE_WIDTH_LEVELS)
        panels[bag] = panel
        node_frames.append(panel.nodes.copy())
        edge_frames.append(panel.edges.copy())

    root = Path(output_root)
    stats_dir = root / "stats"
    reports_dir = root / "reports"
    stats_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rung_label = str(rung) if rung is not None else "auto"

    nodes_df = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    edges_df = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()

    node_path = stats_dir / f"cooccurrence_nodes_{family}_rung_{rung_label}.csv"
    edge_path = stats_dir / f"cooccurrence_edges_{family}_rung_{rung_label}.csv"
    nodes_df.to_csv(node_path, index=False)
    edges_df.to_csv(edge_path, index=False)

    figure_paths: dict[str, Path] = {}
    if write_figure:
        figure_paths = plot_family_figure(
            panels,
            family=family,
            rung=rung,
            top_k=top_k,
            output_root=root,
            edge_min_count=edge_min_count,
            edge_max_plot=edge_max_plot,
        )

    return panels, {
        "nodes": node_path,
        "edges": edge_path,
        **figure_paths,
    }


def build_domain_family_outputs(
    analysis_root: str | Path,
    output_root: str | Path,
    *,
    rung: str | None,
    family: str,
    top_k: int,
    edge_min_count: int = 2,
    edge_max_plot: int | None = 80,
    domain_map: dict[str, str] | None = None,
    write_figure: bool = True,
) -> tuple[dict[str, DomainNetwork], dict[str, Path]]:
    """Build and write the reduced domain-level outputs for one family."""
    if domain_map is None:
        from scripts.exposome_domains import load_domain_map

        domain_map = load_domain_map()

    variable_panels: dict[str, PanelNetwork] = {}
    domain_panels: dict[str, DomainNetwork] = {}

    for bag in DEFAULT_BAG_ORDER:
        panel = _build_best_panel_network(analysis_root, bag, family, top_k, domain_map, rung_override=rung)
        variable_panels[bag] = panel
        domain_panels[bag] = build_domain_network(panel, domain_map)

    edge_width_bins = _edge_width_bins_from_edge_frames([panel.edges for panel in domain_panels.values()])
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for bag in DEFAULT_BAG_ORDER:
        variable_panels[bag] = apply_visual_encodings(
            variable_panels[bag],
            edge_width_bins=edge_width_bins,
            edge_width_levels=DETAIL_EDGE_WIDTH_LEVELS,
        )
        domain_panels[bag] = apply_domain_visual_encodings(
            domain_panels[bag],
            edge_width_bins=edge_width_bins,
            edge_width_levels=DOMAIN_EDGE_WIDTH_LEVELS,
        )
        node_frames.append(domain_panels[bag].nodes.copy())
        edge_frames.append(domain_panels[bag].edges.copy())

    root = Path(output_root)
    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    rung_label = str(rung) if rung is not None else "auto"

    nodes_df = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    edges_df = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()

    node_path = stats_dir / f"domain_cooccurrence_nodes_{family}_rung_{rung_label}.csv"
    edge_path = stats_dir / f"domain_cooccurrence_edges_{family}_rung_{rung_label}.csv"
    nodes_df.to_csv(node_path, index=False)
    edges_df.to_csv(edge_path, index=False)

    figure_paths: dict[str, Path] = {}
    if write_figure:
        figure_paths = plot_domain_family_figure(
            domain_panels,
            family=family,
            rung=rung,
            top_k=top_k,
            output_root=root,
            edge_min_count=edge_min_count,
            edge_max_plot=edge_max_plot,
        )

    return domain_panels, {
        "nodes": node_path,
        "edges": edge_path,
        **figure_paths,
    }


def write_markdown_summary(
    *,
    summary_path: str | Path,
    analysis_root: str | Path,
    rung: str,
    top_k_by_family: dict[str, int],
    panels_by_family: dict[str, dict[str, PanelNetwork]],
    domain_panels_by_family: dict[str, dict[str, DomainNetwork]] | None = None,
    edge_min_count: int,
    edge_max_plot: int | None,
) -> Path:
    root = Path(summary_path)
    root.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Co-occurrence network summary for rung {rung}")
    lines.append("")
    lines.append(f"- Analysis root: `{Path(analysis_root)}`")
    lines.append(f"- Edge filter for figures: co-occurrence_count >= {edge_min_count}" + (f", capped at strongest {edge_max_plot}" if edge_max_plot is not None else ""))
    lines.append("")

    selected_families = [f for f in DEFAULT_FAMILY_ORDER if f in panels_by_family]

    for family in selected_families:
        top_k = top_k_by_family[family]
        lines.append(f"## {family.title()}")
        lines.append("")
        for bag in DEFAULT_BAG_ORDER:
            panel = panels_by_family[family][bag]
            node_df = panel.nodes.copy()
            edge_df = panel.edges.copy()
            bag_label = BAG_LABELS.get(bag, bag.title())
            lines.append(f"### {bag_label}")
            lines.append("")
            lines.append(f"- Best rung: `{panel.rung}`")
            lines.append(f"- Unique variables in top-{top_k}: {len(node_df)}")
            lines.append(f"- Unique co-occurring pairs in top-{top_k}: {len(edge_df)}")
            lines.append(f"- Best model ID: `{panel.top1_model_id}`")
            lines.append(f"- Variables in best model: {', '.join([display_label(v) for v in panel.top1_vars]) if panel.top1_vars else 'none'}")
            if not node_df.empty:
                top_nodes = node_df.sort_values(
                    by=["frequency_topk", "variable"],
                    ascending=[False, True],
                    kind="mergesort",
                ).head(8)
                lines.append("")
                lines.append("Top variables by frequency:")
                for _, row in top_nodes.iterrows():
                    marker = "*" if bool(row["in_top1_model"]) else "-"
                    lines.append(
                        f"{marker} {display_label(row['variable'])} "
                        f"({row['domain']}): {int(row['frequency_topk'])}/{top_k}"
                    )
            if not edge_df.empty:
                top_edges = edge_df.head(8)
                lines.append("")
                lines.append("Top edges by co-occurrence:")
                for _, row in top_edges.iterrows():
                    lines.append(
                        f"- {display_label(row['var_a'])} + {display_label(row['var_b'])}: "
                        f"{int(row['cooccurrence_count'])}/{top_k}"
                    )
            lines.append("")

        family_counts = [len(panels_by_family[family][bag].nodes) for bag in DEFAULT_BAG_ORDER]
        family_edges = [len(panels_by_family[family][bag].edges) for bag in DEFAULT_BAG_ORDER]
        lines.append("### Family-level note")
        lines.append("")
        lines.append(
            f"- Mean unique variables across bags: {np.mean(family_counts):.1f}; "
            f"mean unique edges: {np.mean(family_edges):.1f}"
        )
        lines.append("")

    if "synergy" in panels_by_family and "redundancy" in panels_by_family:
        syn_counts = [len(panels_by_family["synergy"][bag].nodes) for bag in DEFAULT_BAG_ORDER]
        red_counts = [len(panels_by_family["redundancy"][bag].nodes) for bag in DEFAULT_BAG_ORDER]
        syn_edges = [len(panels_by_family["synergy"][bag].edges) for bag in DEFAULT_BAG_ORDER]
        red_edges = [len(panels_by_family["redundancy"][bag].edges) for bag in DEFAULT_BAG_ORDER]
        lines.append("## Synergy vs redundancy")
        lines.append("")
        lines.append(
            f"- Average unique variables: synergy {np.mean(syn_counts):.1f} vs redundancy {np.mean(red_counts):.1f}"
        )
        lines.append(
            f"- Average unique pairs: synergy {np.mean(syn_edges):.1f} vs redundancy {np.mean(red_edges):.1f}"
        )
        if np.mean(red_counts) > np.mean(syn_counts):
            lines.append("- Redundancy is the more complex family here, with a larger average node count.")
        elif np.mean(syn_counts) > np.mean(red_counts):
            lines.append("- Synergy is the more complex family here, with a larger average node count.")
        else:
            lines.append("- The two families have similar average node counts for this rung.")
        if np.mean(red_edges) > np.mean(syn_edges):
            lines.append("- Redundancy also carries a larger edge burden, which is why the figure uses edge filtering.")
        lines.append("")

    if domain_panels_by_family:
        lines.append("## Reduced domain networks")
        lines.append("")
        lines.append(
            "These panels collapse variables into exposome domains. Node size reflects within-domain co-occurrence mass, "
            "while edges reflect cross-domain co-occurrence mass."
        )
        lines.append("")
        for family in selected_families:
            lines.append(f"### {family.title()}")
            lines.append("")
            for bag in DEFAULT_BAG_ORDER:
                panel = domain_panels_by_family[family][bag]
                node_df = panel.nodes.copy()
                edge_df = panel.edges.copy()
                lines.append(f"- {BAG_LABELS.get(bag, bag.title())}: {len(node_df)} domains, {len(edge_df)} domain pairs")
            lines.append("")

    root.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return root
