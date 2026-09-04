from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oinfo_bag_ladder.config import CANONICAL_ROOT, FIGURE_DPI, FIGURES_ROOT, OBJECTIVE_ORDER
from oinfo_bag_ladder.io_utils import read_parquet
from oinfo_bag_ladder.rungs import RUNG_LABELS, RUNG_ORDER
from oinfo_bag_ladder.tables import _fit_slope, _spearman_stats


RUNG_COLORS = {
    "ols": "#2c7fb8",
    "xgb_tree_d1": "#fdbb84",
    "xgb_tree_d2": "#fc8d59",
    "xgb_tree_d3": "#d7301f",
}


def _compute_f2(full: pd.Series | np.ndarray, base: pd.Series | np.ndarray) -> np.ndarray:
    full_arr = pd.to_numeric(pd.Series(full), errors="coerce").to_numpy(dtype=float)
    base_arr = pd.to_numeric(pd.Series(base), errors="coerce").to_numpy(dtype=float)
    denom = 1.0 - full_arr
    out = np.full(len(full_arr), np.nan, dtype=float)
    ok = np.isfinite(full_arr) & np.isfinite(base_arr) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    out[ok] = (full_arr[ok] - base_arr[ok]) / denom[ok]
    return out


def _prepare_metric_frames(
    metrics_global_long: pd.DataFrame,
    metrics_country_long: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = metrics_global_long.copy()
    c = metrics_country_long.copy()
    if metric == "f2":
        if not g.empty:
            g["plot_metric"] = _compute_f2(g.get("full_r2"), g.get("base_r2"))
        if not c.empty:
            c["plot_metric"] = _compute_f2(c.get("country_full_r2"), c.get("country_base_r2"))
    else:
        if not g.empty:
            g["plot_metric"] = pd.to_numeric(g.get("full_r2"), errors="coerce")
        if not c.empty:
            c["plot_metric"] = pd.to_numeric(c.get("country_full_r2"), errors="coerce")
    return g, c


def _compute_metric_frontier_summary(metrics_global_long: pd.DataFrame, metric_col: str = "plot_metric") -> pd.DataFrame:
    base = metrics_global_long.copy()
    if base.empty:
        return pd.DataFrame()
    base = base[base["objective"].isin(["o_max", "o_min"])].copy()
    base["thoi_o"] = pd.to_numeric(base["thoi_o"], errors="coerce")
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")
    out = []
    group_cols = ["experiment_id", "bag_target", "objective", "rung_id", "rung_label"]
    for group_key, group in base.groupby(group_cols, dropna=False, observed=True):
        group = group.dropna(subset=["thoi_o", metric_col]).sort_values("thoi_o").reset_index(drop=True)
        if group.empty:
            continue
        n = len(group)
        n_bins = max(1, min(40, n // 25 if n >= 25 else 1))
        if n_bins <= 1:
            group["bin_idx"] = 0
        else:
            ranked = group["thoi_o"].rank(method="first")
            group["bin_idx"] = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
            group["bin_idx"] = pd.to_numeric(group["bin_idx"], errors="coerce").fillna(0).astype(int)
        binned = (
            group.groupby("bin_idx", dropna=False)
            .agg(
                bin_n=("candidate_id", "size"),
                thoi_o_median=("thoi_o", "median"),
                plot_metric_q50=(metric_col, lambda s: s.quantile(0.50)),
                plot_metric_q95=(metric_col, lambda s: s.quantile(0.95)),
            )
            .reset_index()
        )
        rho, pvalue = _spearman_stats(group["thoi_o"], group[metric_col])
        q95_slope = _fit_slope(binned["thoi_o_median"], binned["plot_metric_q95"])
        for col_name, value in zip(group_cols, group_key):
            binned[col_name] = value
        binned["candidate_n"] = n
        binned["q95_slope"] = q95_slope
        binned["spearman_rho"] = rho
        binned["spearman_pvalue"] = pvalue
        out.append(binned)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _compute_metric_top_tail(metrics_global_long: pd.DataFrame, metric_col: str = "plot_metric") -> pd.DataFrame:
    base = metrics_global_long.copy()
    if base.empty:
        return pd.DataFrame()
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")
    out = []
    group_cols = ["experiment_id", "bag_target", "objective", "rung_id", "rung_label"]
    for group_key, group in base.groupby(group_cols, dropna=False, observed=True):
        group = group.sort_values(
            by=[metric_col, "delta_r2_vs_base", "thoi_o", "candidate_id"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        n_total = len(group)
        if n_total == 0:
            continue
        n_selected = min(n_total, max(10, int(np.ceil(0.10 * n_total))))
        tail = group.head(n_selected).copy()
        tail["tail_rank"] = np.arange(1, len(tail) + 1)
        tail["tail_n_total"] = n_total
        tail["tail_n_selected"] = n_selected
        tail["tail_fraction"] = 0.10
        tail["tail_threshold_plot_metric"] = pd.to_numeric(tail[metric_col], errors="coerce").min()
        out.append(tail)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_canonical_tables(canonical_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = CANONICAL_ROOT if canonical_root is None else Path(canonical_root)
    table_names = [
        "experiment_registry",
        "candidate_registry",
        "metrics_global_long",
        "metrics_country_long",
        "ladder_contrasts_long",
        "frontier_summary_long",
        "top_tail_summary_long",
        "selection_long",
        "portability_comparison_global",
        "portability_comparison_country",
        "transfer_comparison_global",
        "transfer_comparison_country",
    ]
    out = {}
    for name in table_names:
        path = root / f"{name}.parquet"
        out[name] = read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def _annotate(
    ax,
    lines: list[str],
    fontsize: int = 8,
    x: float = 0.03,
    y: float = 0.97,
    ha: str = "left",
    va: str = "top",
) -> None:
    text = "\n".join([line for line in lines if line])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.5},
    )


def _plot_candidate_panel(
    ax,
    df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    rung_id: str,
    title: str | None = None,
    y_col: str = "full_r2",
    y_label: str = "Full LOCO R²",
) -> None:
    color = RUNG_COLORS.get(rung_id, "#4c4c4c")
    x = pd.to_numeric(df.get("thoi_o"), errors="coerce")
    y = pd.to_numeric(df.get(y_col), errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) >= 300:
        ax.hexbin(x, y, gridsize=28, mincnt=1, cmap="Greys", linewidths=0.0)
    elif len(x) > 0:
        ax.scatter(x, y, s=30, alpha=0.45, color=color, edgecolors="none")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    if frontier_df is not None and not frontier_df.empty:
        frontier_df = frontier_df.sort_values("thoi_o_median")
        ax.plot(
            frontier_df["thoi_o_median"],
            frontier_df["plot_metric_q95"],
            color="#111111",
            linewidth=2.0,
        )
        ax.plot(
            frontier_df["thoi_o_median"],
            frontier_df["plot_metric_q50"],
            color="#555555",
            linewidth=1.2,
            linestyle="--",
        )
        first = frontier_df.iloc[0]
        _annotate(
            ax,
            [
                f"n={int(first.get('candidate_n', len(df)))}",
                f"rho={first.get('spearman_rho', np.nan):.3f}" if pd.notna(first.get("spearman_rho")) else "rho=NA",
            ],
            fontsize=9.5,
            x=0.97,
            y=0.03,
            ha="right",
            va="bottom",
        )
    else:
        _annotate(ax, [f"n={len(df)}"], fontsize=9.5, x=0.97, y=0.03, ha="right", va="bottom")

    ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.set_xlabel("O-information score")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize=10)


def _safe_violin(
    ax,
    data_by_label: list[tuple[str, np.ndarray]],
    colors: dict[str, str],
    ylabel: str,
    title: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    cleaned = []
    for label, vals in data_by_label:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            cleaned.append((label, arr))
    values = [vals for _, vals in cleaned]
    labels = [label for label, _ in cleaned]
    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=10)
        return []

    positions = np.arange(1, len(values) + 1)
    parts = ax.violinplot(values, positions=positions, showmeans=False, showextrema=False, widths=0.85)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(colors.get(label, "#777777"))
        body.set_edgecolor("#222222")
        body.set_alpha(0.75)
    rng = np.random.default_rng(20260319)
    for pos, vals in zip(positions, values):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=9,
            color="#111111",
            alpha=0.30,
            edgecolors="none",
            zorder=2,
        )
    medians = [float(np.nanmedian(v)) for v in values]
    ax.scatter(positions, medians, color="#111111", s=30, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([RUNG_LABELS.get(label, label) for label in labels], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    return cleaned


def _plot_summary_candidate_row(
    ax,
    tail_df: pd.DataFrame,
    objective: str,
    fixed_ylim: tuple[float, float] | None = None,
    value_col: str = "full_r2",
    ylabel: str = "Top-tail full R²",
    title: str = "Top 10% ceiling",
) -> None:
    row = tail_df[tail_df["objective"] == objective].copy()
    data_by_label = []
    for rung_id in RUNG_ORDER:
        vals = pd.to_numeric(row.loc[row["rung_id"] == rung_id, value_col], errors="coerce").dropna().to_numpy()
        data_by_label.append((rung_id, vals))
    cleaned = _safe_violin(ax, data_by_label, RUNG_COLORS, ylabel=ylabel, title=title)
    if not cleaned:
        return

    if fixed_ylim is None:
        ymin0, ymax0 = ax.get_ylim()
        yrange0 = max(ymax0 - ymin0, 1e-6)
        data_low = min(float(np.nanmin(arr)) for _, arr in cleaned)
        data_high = max(float(np.nanmax(arr)) for _, arr in cleaned)
        ax.set_ylim(min(ymin0, data_low - 0.18 * yrange0), max(ymax0, data_high + 0.18 * yrange0))
    else:
        ax.set_ylim(*fixed_ylim)
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-6)
    margin = 0.025 * y_range

    for idx, (rung_id, vals) in enumerate(cleaned, start=1):
        sub = row[row["rung_id"] == rung_id]
        if sub.empty:
            continue
        med_delta = pd.to_numeric(sub["delta_r2_vs_base"], errors="coerce").median()
        med_order = pd.to_numeric(sub["order"], errors="coerce").median()
        med_oinfo = pd.to_numeric(sub["thoi_o"], errors="coerce").median()
        violin_low = float(np.nanmin(vals))
        violin_high = float(np.nanmax(vals))
        space_above = y_max - violin_high
        space_below = violin_low - y_min
        if space_above >= space_below:
            y_text = violin_high + margin
            va = "bottom"
        else:
            y_text = violin_low - margin
            va = "top"
        ax.text(
            idx,
            y_text,
            f"Δ={med_delta:.3f}\no={med_oinfo:.2f}\nk={med_order:.0f}",
            ha="center",
            va=va,
            fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.8},
        )


def _shared_ylim_from_series(*series_like: pd.Series, pad_frac: float = 0.05) -> tuple[float, float] | None:
    arrays = []
    for s in series_like:
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            arrays.append(vals)
    if not arrays:
        return None
    all_vals = np.concatenate(arrays)
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None
    if ymax <= 0:
        ymax = 0.0
    pad = max(abs(ymax) * pad_frac, 0.02)
    return -0.5, ymax + pad


def _shared_country_xlim_from_series(*series_like: pd.Series, pad_frac: float = 0.05) -> tuple[float, float] | None:
    arrays = []
    for s in series_like:
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            arrays.append(vals)
    if not arrays:
        return None
    all_vals = np.concatenate(arrays)
    xmax = float(np.nanmax(all_vals))
    if not np.isfinite(xmax):
        return None
    if xmax <= 0:
        xmax = 0.0
    pad = max(abs(xmax) * pad_frac, 0.02)
    return -1.5, xmax + pad


def _build_country_distribution(metrics_country_long: pd.DataFrame, top_tail_summary_long: pd.DataFrame, bag_target: str, objective: str) -> pd.DataFrame:
    tail = top_tail_summary_long[
        (top_tail_summary_long["bag_target"] == bag_target) & (top_tail_summary_long["objective"] == objective)
    ][["experiment_id", "bag_target", "objective", "rung_id", "candidate_id"]].drop_duplicates()
    if tail.empty or metrics_country_long.empty:
        return pd.DataFrame()

    merged = metrics_country_long.merge(
        tail,
        on=["experiment_id", "bag_target", "objective", "rung_id", "candidate_id"],
        how="inner",
    )
    return merged.reset_index(drop=True)


def _plot_country_panel(
    ax,
    country_dist: pd.DataFrame,
    objective: str,
    rung_id: str,
    country_order: list[str],
    show_y_labels: bool,
    fixed_xlim: tuple[float, float] | None = None,
    value_col: str = "country_full_r2",
    xlabel: str = "Country full LOCO R²",
) -> None:
    sub = country_dist[
        (country_dist["objective"] == objective) & (country_dist["rung_id"] == rung_id)
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No country data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    sub["fold_country"] = sub["fold_country"].astype(str)
    arrays = []
    kept_countries = []
    med_support = []
    for country in country_order:
        vals = pd.to_numeric(
            sub.loc[sub["fold_country"] == country, value_col],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        arrays.append(vals)
        kept_countries.append(country)
        med_support.append(
            pd.to_numeric(
                sub.loc[sub["fold_country"] == country, "n_test_scored"],
                errors="coerce",
            ).median()
        )
    if not arrays:
        ax.text(0.5, 0.5, "No country data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    positions = np.arange(1, len(arrays) + 1)
    parts = ax.violinplot(arrays, positions=positions, vert=False, showmeans=False, showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(RUNG_COLORS.get(rung_id, "#777777"))
        body.set_edgecolor("#222222")
        body.set_alpha(0.65)

    rng = np.random.default_rng(20260319)
    for pos, vals in zip(positions, arrays):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(vals, np.full(len(vals), pos) + jitter, s=10, color="#111111", alpha=0.35, edgecolors="none")

    ax.set_yticks(positions)
    if show_y_labels:
        ax.set_yticklabels(kept_countries, fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(xlabel)
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    else:
        x_left, x_right = ax.get_xlim()
        ax.set_xlim(left=-1.5, right=x_right)
    ax.axvline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")
    med_support_all = pd.Series(med_support, dtype=float).median()


def _plot_summary_country_row(
    ax,
    country_dist: pd.DataFrame,
    objective: str,
    fixed_ylim: tuple[float, float] | None = None,
    value_col: str = "country_full_r2",
    ylabel: str = "Country full R²",
    title: str = "Country tail summary",
) -> None:
    row = country_dist[country_dist["objective"] == objective].copy()
    data_by_label = []
    for rung_id in RUNG_ORDER:
        vals = pd.to_numeric(row.loc[row["rung_id"] == rung_id, value_col], errors="coerce").dropna().to_numpy()
        data_by_label.append((rung_id, vals))
    cleaned = _safe_violin(ax, data_by_label, RUNG_COLORS, ylabel=ylabel, title=title)
    if not cleaned:
        return
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
        return

    arrays = [vals for _, vals in cleaned]
    ymax = float(np.nanmax(np.concatenate(arrays)))
    if ymax <= 0:
        ymax = 0.0
    pad = max(abs(ymax) * 0.05, 0.02)
    ax.set_ylim(-1.5, ymax + pad)


def plot_canonical_bag_figure(
    bag_target: str,
    metrics_global_long: pd.DataFrame,
    metrics_country_long: pd.DataFrame,
    frontier_summary_long: pd.DataFrame,
    top_tail_summary_long: pd.DataFrame,
    figsize: tuple[float, float] = (22, 15),
    metric: str = "r2",
) -> plt.Figure:
    metrics_global_long, metrics_country_long = _prepare_metric_frames(metrics_global_long, metrics_country_long, metric)
    bag_global = metrics_global_long[metrics_global_long["bag_target"] == bag_target].copy()
    if metric == "f2":
        bag_frontier = _compute_metric_frontier_summary(bag_global, metric_col="plot_metric")
        bag_tail = _compute_metric_top_tail(bag_global, metric_col="plot_metric")
    else:
        bag_frontier = frontier_summary_long[frontier_summary_long["bag_target"] == bag_target].copy()
        if not bag_frontier.empty:
            bag_frontier = bag_frontier.rename(columns={"full_r2_q50": "plot_metric_q50", "full_r2_q95": "plot_metric_q95"})
        bag_tail = top_tail_summary_long[top_tail_summary_long["bag_target"] == bag_target].copy()
        if not bag_tail.empty:
            bag_tail["plot_metric"] = pd.to_numeric(bag_tail.get("full_r2"), errors="coerce")
    bag_country_dist = _build_country_distribution(metrics_country_long, bag_tail, bag_target, "o_max")
    bag_country_dist = pd.concat(
        [
            bag_country_dist,
            _build_country_distribution(metrics_country_long, bag_tail, bag_target, "o_min"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    metric_title = "F²" if metric == "f2" else "R²"
    metric_label = "Full LOCO F²" if metric == "f2" else "Full LOCO R²"
    tail_label = "Top-tail full F²" if metric == "f2" else "Top-tail full R²"
    country_label = "Country full LOCO F²" if metric == "f2" else "Country full LOCO R²"
    country_summary_label = "Country full F²" if metric == "f2" else "Country full R²"

    fig, axes = plt.subplots(4, 6, figsize=figsize, constrained_layout=True)
    shared_candidate_ylim = _shared_ylim_from_series(
        bag_global.get("plot_metric", pd.Series(dtype=float)),
        bag_tail.get("plot_metric", pd.Series(dtype=float)),
    )

    for row_idx, objective in enumerate(OBJECTIVE_ORDER):
        obj_global = bag_global[bag_global["objective"] == objective].copy()
        for col_idx, rung_id in enumerate(RUNG_ORDER):
            ax = axes[row_idx, col_idx]
            panel_df = obj_global[obj_global["rung_id"] == rung_id].copy()
            panel_frontier = bag_frontier[
                (bag_frontier["objective"] == objective) & (bag_frontier["rung_id"] == rung_id)
            ].copy()
            title = RUNG_LABELS.get(rung_id, rung_id) if row_idx == 0 else None
            _plot_candidate_panel(ax, panel_df, panel_frontier, rung_id=rung_id, title=title, y_col="plot_metric", y_label=metric_label)
            if shared_candidate_ylim is not None:
                ax.set_ylim(*shared_candidate_ylim)
            if col_idx > 0:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(RUNG_LABELS.get(rung_id, rung_id), fontsize=10)
        _plot_summary_candidate_row(
            axes[row_idx, 5],
            bag_tail,
            objective,
            fixed_ylim=shared_candidate_ylim,
            value_col="plot_metric",
            ylabel=tail_label,
            title="Top 10% ceiling",
        )
        if row_idx == 0:
            axes[row_idx, 5].set_title("Summary", fontsize=10)

    for row_offset, objective in enumerate(OBJECTIVE_ORDER, start=2):
        obj_country = bag_country_dist[bag_country_dist["objective"] == objective].copy()
        shared_country_xlim = _shared_country_xlim_from_series(obj_country.get("plot_metric", pd.Series(dtype=float)))
        country_order = (
            obj_country.groupby("fold_country", observed=True)["plot_metric"]
            .median()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        for col_idx, rung_id in enumerate(RUNG_ORDER):
            ax = axes[row_offset, col_idx]
            _plot_country_panel(
                ax,
                bag_country_dist,
                objective=objective,
                rung_id=rung_id,
                country_order=country_order,
                show_y_labels=(col_idx == 0),
                fixed_xlim=shared_country_xlim,
                value_col="plot_metric",
                xlabel=country_label,
            )
            if row_offset == 2:
                ax.set_title(RUNG_LABELS.get(rung_id, rung_id), fontsize=10)
        _plot_summary_country_row(
            axes[row_offset, 5],
            bag_country_dist,
            objective,
            fixed_ylim=shared_country_xlim,
            value_col="plot_metric",
            ylabel=country_summary_label,
            title="Country tail summary",
        )
        if row_offset == 2:
            axes[row_offset, 5].set_title("Summary", fontsize=10)

    row_labels = ["o_max: candidate", "o_min: candidate", "o_max: country", "o_min: country"]
    for i, label in enumerate(row_labels):
        axes[i, 0].annotate(
            label,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(f"{bag_target.title()} BAG: O-information ladder ({metric_title})", fontsize=16, y=1.01)
    return fig


def save_all_bag_figures(
    metrics_global_long: pd.DataFrame,
    metrics_country_long: pd.DataFrame,
    frontier_summary_long: pd.DataFrame,
    top_tail_summary_long: pd.DataFrame,
    output_dir: Path | None = None,
    dpi: int = FIGURE_DPI,
    metric: str = "r2",
) -> list[Path]:
    outdir = FIGURES_ROOT if output_dir is None else Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bag_targets = (
        metrics_global_long["bag_target"].dropna().astype(str).drop_duplicates().tolist()
        if not metrics_global_long.empty
        else []
    )
    written = []
    for bag_target in bag_targets:
        fig = plot_canonical_bag_figure(
            bag_target=bag_target,
            metrics_global_long=metrics_global_long,
            metrics_country_long=metrics_country_long,
            frontier_summary_long=frontier_summary_long,
            top_tail_summary_long=top_tail_summary_long,
            metric=metric,
        )
        suffix = "" if metric == "r2" else f"_{metric}"
        out_path = outdir / f"canonical_bag_{bag_target}{suffix}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def _diag_rows(df: pd.DataFrame, col: str = "test_diagnosis_group") -> list[str]:
    order = ["CN", "AD", "MCI", "FTD"]
    vals = df[col].dropna().astype(str).unique().tolist() if col in df.columns else []
    return [x for x in order if x in vals]


def _plot_distribution_panel(ax, vals: np.ndarray, color: str, ylabel: str = "", title: str | None = None) -> None:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=10)
        return
    parts = ax.violinplot([arr], positions=[1], showmeans=False, showextrema=False, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.75)
    rng = np.random.default_rng(20260323)
    jitter = rng.uniform(-0.08, 0.08, size=len(arr))
    ax.scatter(np.full(len(arr), 1.0) + jitter, arr, s=18, color="#111111", alpha=0.35, edgecolors="none")
    ax.scatter([1], [float(np.nanmedian(arr))], s=28, color="#111111", zorder=3)
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")


def _plot_shift_summary_panel(ax, df: pd.DataFrame, value_col: str, title: str | None = None) -> None:
    data_by_label = []
    for rung_id in RUNG_ORDER:
        vals = pd.to_numeric(df.loc[df["rung_id"] == rung_id, value_col], errors="coerce").dropna().to_numpy()
        data_by_label.append((rung_id, vals))
    _safe_violin(ax, data_by_label, RUNG_COLORS, ylabel=value_col.replace("_", " "), title=title)
    ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")


def plot_portability_bag_figure(
    bag_target: str,
    portability_global_df: pd.DataFrame,
    figsize: tuple[float, float] = (22, 14),
) -> plt.Figure:
    bag_df = portability_global_df[portability_global_df["bag_target"] == bag_target].copy()
    diagnoses = _diag_rows(bag_df)
    fig, axes = plt.subplots(len(diagnoses), 6, figsize=figsize, constrained_layout=True)
    if len(diagnoses) == 1:
        axes = np.asarray([axes])

    yvals = pd.to_numeric(bag_df.get("target_full_r2"), errors="coerce")
    yvals = yvals[np.isfinite(yvals)]
    ymax = float(np.nanmax(yvals)) if len(yvals) else 0.0
    ylim = (-0.5, ymax + max(abs(ymax) * 0.05, 0.02))

    for row_idx, diagnosis in enumerate(diagnoses):
        row = bag_df[bag_df["test_diagnosis_group"].astype(str) == diagnosis].copy()
        for col_idx, rung_id in enumerate(RUNG_ORDER):
            ax = axes[row_idx, col_idx]
            vals = pd.to_numeric(row.loc[row["rung_id"] == rung_id, "target_full_r2"], errors="coerce").dropna().to_numpy()
            _plot_distribution_panel(
                ax,
                vals,
                color=RUNG_COLORS.get(rung_id, "#777777"),
                ylabel="Full LOCO R²" if col_idx == 0 else "",
                title=RUNG_LABELS.get(rung_id, rung_id) if row_idx == 0 else None,
            )
            ax.set_ylim(*ylim)
        _plot_shift_summary_panel(
            axes[row_idx, 5],
            row,
            value_col="shift_full_r2_vs_pooled",
            title="Summary" if row_idx == 0 else None,
        )
        axes[row_idx, 5].set_ylim(*ylim)
        axes[row_idx, 0].annotate(
            diagnosis,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(f"{bag_target.title()} BAG: portability", fontsize=16, y=1.01)
    return fig


def save_all_portability_figures(
    portability_global_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = FIGURE_DPI,
) -> list[Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bag_targets = portability_global_df["bag_target"].dropna().astype(str).drop_duplicates().tolist() if not portability_global_df.empty else []
    written = []
    for bag_target in bag_targets:
        fig = plot_portability_bag_figure(bag_target, portability_global_df)
        out_path = outdir / f"portability_bag_{bag_target}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_transfer_bag_figure(
    bag_target: str,
    transfer_global_df: pd.DataFrame,
    figsize: tuple[float, float] = (22, 14),
) -> plt.Figure:
    bag_df = transfer_global_df[transfer_global_df["bag_target"] == bag_target].copy()
    diagnoses = _diag_rows(bag_df)
    fig, axes = plt.subplots(len(diagnoses), 6, figsize=figsize, constrained_layout=True)
    if len(diagnoses) == 1:
        axes = np.asarray([axes])

    all_vals = pd.concat(
        [
            pd.to_numeric(bag_df.get("transfer_full_r2"), errors="coerce"),
            pd.to_numeric(bag_df.get("dx_refit_full_r2"), errors="coerce"),
        ],
        ignore_index=True,
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    ymax = float(np.nanmax(all_vals)) if len(all_vals) else 0.0
    ylim = (-0.5, ymax + max(abs(ymax) * 0.05, 0.02))

    for row_idx, diagnosis in enumerate(diagnoses):
        row = bag_df[bag_df["test_diagnosis_group"].astype(str) == diagnosis].copy()
        for col_idx, rung_id in enumerate(RUNG_ORDER):
            ax = axes[row_idx, col_idx]
            sub = row[row["rung_id"] == rung_id].copy()
            transfer_vals = pd.to_numeric(sub["transfer_full_r2"], errors="coerce").dropna().to_numpy()
            refit_vals = pd.to_numeric(sub["dx_refit_full_r2"], errors="coerce").dropna().to_numpy()
            if len(transfer_vals) == 0 and len(refit_vals) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
            else:
                data = []
                pos = []
                colors = []
                labels = []
                if len(transfer_vals):
                    data.append(transfer_vals)
                    pos.append(0.9)
                    colors.append("#444444")
                    labels.append("CN→DX")
                if len(refit_vals):
                    data.append(refit_vals)
                    pos.append(1.1)
                    colors.append(RUNG_COLORS.get(rung_id, "#777777"))
                    labels.append("DX refit")
                parts = ax.violinplot(data, positions=pos, showmeans=False, showextrema=False, widths=0.18)
                for body, color in zip(parts["bodies"], colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("#222222")
                    body.set_alpha(0.75)
                rng = np.random.default_rng(20260323)
                for p, vals in zip(pos, data):
                    jitter = rng.uniform(-0.03, 0.03, size=len(vals))
                    ax.scatter(np.full(len(vals), p) + jitter, vals, s=15, color="#111111", alpha=0.35, edgecolors="none")
                ax.set_xticks(pos)
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            ax.set_ylim(*ylim)
            ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle=":")
            if col_idx == 0:
                ax.set_ylabel("Full LOCO R²")
            if row_idx == 0:
                ax.set_title(RUNG_LABELS.get(rung_id, rung_id), fontsize=10)
        _plot_shift_summary_panel(
            axes[row_idx, 5],
            row,
            value_col="shift_full_r2_transfer_minus_refit",
            title="Summary" if row_idx == 0 else None,
        )
        axes[row_idx, 5].set_ylim(*ylim)
        axes[row_idx, 0].annotate(
            diagnosis,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(f"{bag_target.title()} BAG: CN transfer", fontsize=16, y=1.01)
    return fig


def save_all_transfer_figures(
    transfer_global_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = FIGURE_DPI,
) -> list[Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bag_targets = transfer_global_df["bag_target"].dropna().astype(str).drop_duplicates().tolist() if not transfer_global_df.empty else []
    written = []
    for bag_target in bag_targets:
        fig = plot_transfer_bag_figure(bag_target, transfer_global_df)
        out_path = outdir / f"transfer_bag_{bag_target}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written
