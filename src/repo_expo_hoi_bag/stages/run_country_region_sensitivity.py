#!/usr/bin/env python3
"""Country (LOCO) and region (LORO) sensitivity for the exposome-BAG analysis.

Two rows of evidence, both showing top-20 synergy / top-20 redundancy as in Fig.2:

- Country LOCO (row 1) is *extracted* from the on-disk canonical metrics
  (`metrics_country_long.parquet` + `metrics_global_long.parquet`) — no model is
  refit. Top-20 syn/red are chosen independently within each rung by that rung's
  global LOCO R2 (intra-rung rule), then their per-held-out-country R2 is read off.
- Region LORO (row 2) is a fresh leave-one-region-out evaluation of the same Fig.2
  candidate pool, splitting on `subregion_name` instead of `country_clean`. The
  same main-analysis country/diagnosis exclusions are applied first.

Lightweight stats -> outputs/sensitivity/country_region/. Heavy LORO eval folds ->
bundle via the shared evaluator's resume cache.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
# Publication style (Arial, fonttype 42, editable SVG text) must be imported
# before pyplot so its rcParams apply to every figure this stage draws.
from repo_expo_hoi_bag.figures.style import (  # noqa: E402
    BAG_LABELS,
    BAG_ROW_ORDER,
    BAG_SHORT,
    GRID_FS_TK,
    LEVEL_LABELS,
    NULL_COLOR,
    RED_COLOR,
    ROW_LETTERS,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.sensitivity_common import (
    ACTIVE_RUNGS,
    RUNG_LABELS,
    active_rungs,
    analysis_cfg_from_config,
    apply_order_cap,
    baseline_candidate_df,
    build_original_model_df,
    bundle_sensitivity_root,
    canonical_root,
    cap_split_subdir,
    copy_tree_contents,
    env_bool,
    evaluate_candidates_by_rung,
    filter_syn_pool,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    selected_bags,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

TOP_K = 20
CV_LEVEL_LABEL = {"country_loco": "Country LOCO", "region_loro": "Region LORO"}
SERIES_STYLE = {
    "top20_syn": {"color": SYN_COLOR, "offset": -0.15, "label": "Min O-info"},
    "top20_red": {"color": RED_COLOR, "offset": 0.15, "label": "Max O-info"},
}
POINT_AREA = 54


def _shared_row_ylim(perf: pd.DataFrame, cv_level: str, rungs: list[str]) -> tuple[float, float]:
    """Return one y-axis range for every model-level panel in a display row."""
    row = perf[
        perf["cv_level"].eq(cv_level)
        & perf["rung_id"].astype(str).isin(rungs)
    ]
    values = pd.concat(
        [pd.to_numeric(row[column], errors="coerce") for column in
         ("min_r2", "median_r2", "max_r2", "baseline_r2")],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    values = pd.concat([values, pd.Series([0.0])], ignore_index=True)
    low, high = float(values.min()), float(values.max())
    span = high - low
    pad = 0.05 * span if span > 0 else max(abs(high) * 0.05, 0.01)
    return low - pad, high + pad


def _top_ids_by_rung(global_df: pd.DataFrame, rungs: list[str]) -> dict:
    """{rung: {"syn": set(ids), "red": set(ids)}} top-20 intra-rung by full_r2."""
    out: dict = {}
    # Restrict to candidates of order <= cap (cap21 vs complete) before taking the
    # intra-rung top-20, so the plotted top-20 syn/red sets match the capped pool.
    g = apply_order_cap(global_df.copy())
    g["full_r2"] = pd.to_numeric(g["full_r2"], errors="coerce")
    for rung in rungs:
        rr = g[g["rung_id"].astype(str) == rung]
        syn_pool = filter_syn_pool(rr[rr["objective"].astype(str) == "o_min"], "o_min", "thoi_o")
        syn = set(syn_pool.nlargest(TOP_K, "full_r2")["candidate_id"].astype(str))
        red = set(rr[rr["objective"].astype(str) == "o_max"].nlargest(TOP_K, "full_r2")["candidate_id"].astype(str))
        out[rung] = {"syn": syn, "red": red}
    return out


def _per_fold_from_country_long(
    country_df: pd.DataFrame,
    top_ids: dict,
    rungs: list[str],
    fold_col: str,
    r2_col: str,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-fold top-20 syn/red ranges, with a per-(rung, fold) baseline merged in.

    baseline_df: columns [rung_id, fold, baseline_r2].
    """
    rows = []
    c = country_df.copy()
    c[r2_col] = pd.to_numeric(c[r2_col], errors="coerce")
    base_lkp = {(str(r["rung_id"]), str(r["fold"])): float(r["baseline_r2"]) for _, r in baseline_df.iterrows()}
    for rung in rungs:
        rr = c[c["rung_id"].astype(str) == rung]
        for series, ids in [("top20_syn", top_ids[rung]["syn"]), ("top20_red", top_ids[rung]["red"])]:
            sub = rr[rr["candidate_id"].astype(str).isin(ids)]
            for fold, ff in sub.groupby(fold_col):
                vals = ff[r2_col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                rows.append(
                    {
                        "rung_id": rung,
                        "series": series,
                        "fold": str(fold),
                        "median_r2": float(np.median(vals)),
                        "min_r2": float(np.min(vals)),
                        "max_r2": float(np.max(vals)),
                        "n_models": int(len(vals)),
                        "baseline_r2": base_lkp.get((rung, str(fold)), np.nan),
                        "n_test": int(np.nanmedian(pd.to_numeric(ff.get("n_test_total", np.nan), errors="coerce"))) if "n_test_total" in ff else -1,
                    }
                )
    return pd.DataFrame(rows)


def _baseline_from_col(country_df: pd.DataFrame, rungs: list[str], fold_col: str, base_col: str) -> pd.DataFrame:
    c = country_df.copy()
    c[base_col] = pd.to_numeric(c[base_col], errors="coerce")
    rows = []
    for rung in rungs:
        rr = c[c["rung_id"].astype(str) == rung]
        for fold, ff in rr.groupby(fold_col):
            rows.append({"rung_id": rung, "fold": str(fold), "baseline_r2": float(np.nanmedian(ff[base_col].to_numpy(dtype=float)))})
    return pd.DataFrame(rows)


def _baseline_from_candidate(country_df: pd.DataFrame, rungs: list[str], fold_col: str, r2_col: str) -> pd.DataFrame:
    """Per-(rung, fold) baseline = LOCO R2 of the __baseline__ empty-predictor model."""
    c = country_df[country_df["candidate_id"].astype(str) == "__baseline__"].copy()
    c[r2_col] = pd.to_numeric(c[r2_col], errors="coerce")
    rows = []
    for rung in rungs:
        rr = c[c["rung_id"].astype(str) == rung]
        for fold, ff in rr.groupby(fold_col):
            rows.append({"rung_id": rung, "fold": str(fold), "baseline_r2": float(np.nanmedian(ff[r2_col].to_numpy(dtype=float)))})
    return pd.DataFrame(rows)


def _extract_country_loco(bag: str, rungs: list[str]) -> pd.DataFrame:
    base = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}"
    gl = pd.read_parquet((base / "metrics_global_long.parquet"))
    co = pd.read_parquet((base / "metrics_country_long.parquet"))
    for df in (gl, co):
        if "bag_target" in df.columns:
            df.drop(df[df["bag_target"].astype(str) != bag].index, inplace=True)
    gl = gl[gl["rung_id"].astype(str).isin(rungs)].copy()
    co = co[co["rung_id"].astype(str).isin(rungs)].copy()
    top_ids = _top_ids_by_rung(gl, rungs)
    baseline_df = _baseline_from_col(co, rungs, "fold_country", "country_base_r2")
    out = _per_fold_from_country_long(
        co, top_ids, rungs, fold_col="fold_country", r2_col="country_full_r2", baseline_df=baseline_df
    )
    out["cv_level"] = "country_loco"
    out["bag"] = bag
    return out


def _run_region_loro(
    bag: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    analysis_cfg: dict,
    rungs: list[str],
    eval_dir: Path,
    n_jobs: int,
    max_candidates: int | None,
) -> pd.DataFrame:
    # Cap the Fig.2 pool first (smoke), then always append the baseline so the
    # head() cap inside the evaluator can never drop it.
    pool = load_fig2_candidate_pool(bag)
    if max_candidates is not None:
        pool = pool.head(int(max_candidates)).copy()
    pool = pd.concat([pool, baseline_candidate_df()], ignore_index=True)
    summary, country = evaluate_candidates_by_rung(
        model_df=model_df,
        candidate_df=pool,
        exposome_cols=feature_names,
        bag=bag,
        rungs=rungs,
        analysis_cfg=analysis_cfg,
        outdir=eval_dir,
        n_jobs=n_jobs,
        max_candidates=None,
        cv_override={"split_col": "subregion_name"},
    )
    summary["score"] = pd.to_numeric(summary["score"], errors="coerce")
    summary["global_oof_r2"] = pd.to_numeric(summary["global_oof_r2"], errors="coerce")
    models = summary[summary["candidate_id"].astype(str) != "__baseline__"]
    # Cap by interaction order (cap21 vs complete) before taking the top-20 per
    # rung; the LORO eval itself stays over the full pool (cap-invariant, reused).
    models = apply_order_cap(models)
    top_ids: dict = {}
    for rung in rungs:
        rr = models[models["rung_id"].astype(str) == rung]
        # syn/red are the greedy discovery objectives (o_min/o_max), NOT the sign of
        # the evaluated O-info — a candidate found via o_min can evaluate to score>0
        # and still IS synergistic-route. (Matches the o_min/o_max split at line ~69.)
        syn_pool = filter_syn_pool(rr[rr["objective"].astype(str) == "o_min"], "o_min", "score")
        syn = set(syn_pool.nlargest(TOP_K, "global_oof_r2")["candidate_id"].astype(str))
        red = set(rr[rr["objective"].astype(str) == "o_max"].nlargest(TOP_K, "global_oof_r2")["candidate_id"].astype(str))
        top_ids[rung] = {"syn": syn, "red": red}
    baseline_df = _baseline_from_candidate(country, rungs, "fold_country", "r2")
    out = _per_fold_from_country_long(
        country, top_ids, rungs, fold_col="fold_country", r2_col="r2", baseline_df=baseline_df
    )
    out["cv_level"] = "region_loro"
    out["bag"] = bag
    return out


def _plot_country_region(perf_by_bag: dict[str, pd.DataFrame], rungs: list[str], outdir: Path) -> None:
    """One figure for both BAGs: structural rows first, then functional.

    Each BAG contributes its country-LOCO and region-LORO rows, so the merged
    figure is 4 rows (structural LOCO, structural LORO, functional LOCO,
    functional LORO). Row labels follow the plot_fig2_grid_v3 convention: the
    first panel of each row carries "<letter>. <row description>" left-aligned,
    and there is no suptitle.
    """
    rows: list[tuple[str, str, pd.DataFrame]] = []
    for bag in BAG_ROW_ORDER:
        perf = perf_by_bag.get(bag)
        if perf is None or perf.empty:
            continue
        for level in ("country_loco", "region_loro"):
            if level in set(perf["cv_level"]):
                rows.append((bag, level, perf))
    if not rows:
        return

    fig, axes = plt.subplots(
        len(rows), len(rungs), figsize=(3.6 * len(rungs), 3.6 * len(rows)), squeeze=False,
        # hspace doubled (0.45 -> 0.90): the rotated country/region tick labels
        # under each row need the extra vertical room.
        gridspec_kw={"wspace": 0.28, "hspace": 0.90},
    )
    for ri, (bag, lv, perf) in enumerate(rows):
        row_ylim = _shared_row_ylim(perf, lv, rungs)
        for ci, rung in enumerate(rungs):
            ax = axes[ri, ci]
            cell = perf[(perf["cv_level"] == lv) & (perf["rung_id"].astype(str) == rung)]
            folds = sorted(cell["fold"].unique())
            xmap = {f: i for i, f in enumerate(folds)}
            for series, display in SERIES_STYLE.items():
                ss = cell[cell["series"] == series]
                xs = [xmap[f] + display["offset"] for f in ss["fold"]]
                ax.errorbar(
                    xs, ss["median_r2"],
                    yerr=[ss["median_r2"] - ss["min_r2"], ss["max_r2"] - ss["median_r2"]],
                    fmt="o", ms=np.sqrt(POINT_AREA), lw=1.0,
                    color=display["color"], alpha=0.55,
                    markeredgewidth=0, zorder=3, label=display["label"],
                )
            bl = cell.dropna(subset=["baseline_r2"]).drop_duplicates("fold")
            if not bl.empty:
                ax.scatter(
                    [xmap[f] for f in bl["fold"]], bl["baseline_r2"],
                    marker="_", s=110, linewidths=2.0, color="black", zorder=4,
                    label="Baseline",
                )
            ax.axhline(0, color=NULL_COLOR, lw=0.8, ls=":", zorder=1)
            ax.set_ylim(row_ylim)
            ax.set_xticks(range(len(folds)))
            ax.set_xticklabels(folds, rotation=90, fontsize=GRID_FS_TK - 4)
            ax.tick_params(axis="y", labelsize=GRID_FS_TK)
            ax.set_ylabel(LEVEL_LABELS[rung], fontsize=GRID_FS_TK)
            if ci == 0:
                ax.set_title(
                    f"{ROW_LETTERS[ri]}. {BAG_LABELS[bag]} {CV_LEVEL_LABEL[lv]} — R² LOCO",
                    fontsize=GRID_FS_TK,
                    loc="left",
                )
            if ri == len(rows) - 1 and ci == len(rungs) - 1:
                handles = [
                    mlines.Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=np.sqrt(POINT_AREA),
                        markerfacecolor=SYN_COLOR, markeredgewidth=0, alpha=0.55, label="Min O-info",
                    ),
                    mlines.Line2D(
                        [0], [0], marker="o", linestyle="none", markersize=np.sqrt(POINT_AREA),
                        markerfacecolor=RED_COLOR, markeredgewidth=0, alpha=0.55, label="Max O-info",
                    ),
                    mlines.Line2D([0], [0], color="black", lw=2.0, label="Baseline"),
                ]
                ax.legend(
                    handles=handles,
                    fontsize=GRID_FS_TK - 5,
                    loc="lower left",
                    ncol=3,
                    framealpha=0.7,
                    handlelength=1.0,
                    handletextpad=0.3,
                    columnspacing=0.7,
                    borderpad=0.3,
                )
            style_axis(ax)
    stem = "country_region_sensitivity"
    save_figure(fig, stem, outdir)
    plt.close(fig)

    # Source Data: one sheet per drawn panel (row x rung), carrying exactly the
    # per-fold values the errorbars and baseline marks are drawn from.
    panels = []
    for ri, (bag, lv, perf) in enumerate(rows):
        for ci, rung in enumerate(rungs):
            cell = perf[(perf["cv_level"] == lv) & (perf["rung_id"].astype(str) == rung)]
            if cell.empty:
                continue
            frame = cell[
                [c for c in ["fold", "series", "median_r2", "min_r2", "max_r2", "baseline_r2"] if c in cell.columns]
            ].sort_values(["series", "fold"]).reset_index(drop=True)
            panels.append(
                Panel(
                    # Kept short so the xlsx 31-char sheet-name cap cannot
                    # truncate the rung (struct/func + loco/loro + rung).
                    panel_id=(
                        f"{ROW_LETTERS[ri]}{ci + 1}_{BAG_SHORT[bag]}_"
                        f"{'loco' if lv == 'country_loco' else 'loro'}_"
                        f"{'ols' if rung == 'ols' else rung.replace('xgb_tree_', '')}"
                    ),
                    frame=frame,
                    description=(
                        f"{BAG_LABELS[bag]} {CV_LEVEL_LABEL[lv]} held-out-fold LOCO R2 for the "
                        f"top-20 candidates in the synergy arm and redundancy arm at model level "
                        f"{LEVEL_LABELS[rung]}."
                    ),
                    columns={
                        "fold": "held-out country (country_loco) or subregion (region_loro)",
                        "series": "top20_syn = top 20 in the minimum-O-information (synergy) arm; top20_red = top 20 in the maximum-O-information (redundancy) arm",
                        "median_r2": "median held-out R² across the 20 candidates in that series",
                        "min_r2": "minimum held-out R² across those candidates (lower errorbar)",
                        "max_r2": "maximum held-out R² across those candidates (upper errorbar)",
                        "baseline_r2": "covariate-only baseline R² for that fold (horizontal dash)",
                    },
                    notes=(
                        "Top-20 sets are selected within each model level by that level's global LOCO R². "
                        "Values are held-out R²; no model is refit for the plot."
                    ),
                )
            )
    write_source_data(stem, panels, outdir)


def _plot_country_region_rotated(
    perf_by_bag: dict[str, pd.DataFrame], rungs: list[str], outdir: Path
) -> None:
    """Horizontal alternative to the canonical country/region figure.

    The panel grid and data contract are unchanged. Within each panel, held-out
    folds move to the y-axis and R² to the x-axis; error ranges, baseline marks,
    model labels and the legend rotate with that coordinate system.
    """
    rows: list[tuple[str, str, pd.DataFrame]] = []
    for bag in BAG_ROW_ORDER:
        perf = perf_by_bag.get(bag)
        if perf is None or perf.empty:
            continue
        for level in ("country_loco", "region_loro"):
            if level in set(perf["cv_level"]):
                rows.append((bag, level, perf))
    if not rows:
        return

    fig, axes = plt.subplots(
        len(rows), len(rungs),
        figsize=(4.6 * len(rungs), 3.6 * len(rows)),
        squeeze=False,
        gridspec_kw={"wspace": 0.65, "hspace": 0.55},
    )
    for ri, (bag, lv, perf) in enumerate(rows):
        row_xlim = _shared_row_ylim(perf, lv, rungs)
        for ci, rung in enumerate(rungs):
            ax = axes[ri, ci]
            cell = perf[(perf["cv_level"] == lv) & (perf["rung_id"].astype(str) == rung)]
            folds = sorted(cell["fold"].unique())
            ymap = {fold: i for i, fold in enumerate(folds)}
            for series, display in SERIES_STYLE.items():
                subset = cell[cell["series"] == series]
                ys = [ymap[fold] + display["offset"] for fold in subset["fold"]]
                ax.errorbar(
                    subset["median_r2"],
                    ys,
                    xerr=[
                        subset["median_r2"] - subset["min_r2"],
                        subset["max_r2"] - subset["median_r2"],
                    ],
                    fmt="o",
                    ms=np.sqrt(POINT_AREA),
                    lw=1.0,
                    color=display["color"],
                    alpha=0.55,
                    markeredgewidth=0,
                    zorder=3,
                    label=display["label"],
                )
            baseline = cell.dropna(subset=["baseline_r2"]).drop_duplicates("fold")
            if not baseline.empty:
                ax.scatter(
                    baseline["baseline_r2"],
                    [ymap[fold] for fold in baseline["fold"]],
                    marker="|",
                    s=110,
                    linewidths=2.0,
                    color="black",
                    zorder=4,
                    label="Baseline",
                )
            ax.axvline(0, color=NULL_COLOR, lw=0.8, ls=":", zorder=1)
            ax.set_xlim(row_xlim)
            ax.set_yticks(range(len(folds)))
            ax.set_yticklabels(folds, rotation=0, fontsize=GRID_FS_TK - 4)
            ax.set_ylim(len(folds) - 0.5, -0.5)
            ax.tick_params(axis="x", labelsize=GRID_FS_TK)
            ax.set_xlabel(LEVEL_LABELS[rung], fontsize=GRID_FS_TK)
            if ci == 0:
                ax.set_title(
                    f"{ROW_LETTERS[ri]}. {BAG_LABELS[bag]} {CV_LEVEL_LABEL[lv]} — R² LOCO",
                    fontsize=GRID_FS_TK,
                    loc="left",
                )
            if ri == len(rows) - 1 and ci == len(rungs) - 1:
                handles = [
                    mlines.Line2D(
                        [0], [0], marker="o", linestyle="none",
                        markersize=np.sqrt(POINT_AREA), markerfacecolor=SYN_COLOR,
                        markeredgewidth=0, alpha=0.55, label="Min O-info",
                    ),
                    mlines.Line2D(
                        [0], [0], marker="o", linestyle="none",
                        markersize=np.sqrt(POINT_AREA), markerfacecolor=RED_COLOR,
                        markeredgewidth=0, alpha=0.55, label="Max O-info",
                    ),
                    mlines.Line2D([0], [0], color="black", lw=2.0, label="Baseline"),
                ]
                ax.legend(
                    handles=handles,
                    fontsize=GRID_FS_TK - 5,
                    loc="lower left",
                    ncol=1,
                    framealpha=0.7,
                    handlelength=1.0,
                    handletextpad=0.3,
                    borderpad=0.3,
                )
            style_axis(ax)

    stem = "country_region_sensitivity_rotated"
    save_figure(fig, stem, outdir)
    plt.close(fig)

    panels = []
    for ri, (bag, lv, perf) in enumerate(rows):
        for ci, rung in enumerate(rungs):
            cell = perf[(perf["cv_level"] == lv) & (perf["rung_id"].astype(str) == rung)]
            if cell.empty:
                continue
            frame = cell[
                [column for column in
                 ("fold", "series", "median_r2", "min_r2", "max_r2", "baseline_r2")
                 if column in cell.columns]
            ].sort_values(["series", "fold"]).reset_index(drop=True)
            panels.append(
                Panel(
                    panel_id=(
                        f"{ROW_LETTERS[ri]}{ci + 1}_{BAG_SHORT[bag]}_"
                        f"{'loco' if lv == 'country_loco' else 'loro'}_"
                        f"{'ols' if rung == 'ols' else rung.replace('xgb_tree_', '')}"
                    ),
                    frame=frame,
                    description=(
                        f"{BAG_LABELS[bag]} {CV_LEVEL_LABEL[lv]} held-out-fold LOCO R2 for "
                        f"the top-20 candidates in the synergy arm and redundancy arm at "
                        f"model level {LEVEL_LABELS[rung]}; horizontal display."
                    ),
                    columns={
                        "fold": "held-out country (country_loco) or subregion (region_loro)",
                        "series": "top20_syn = top 20 in the minimum-O-information (synergy) arm; top20_red = top 20 in the maximum-O-information (redundancy) arm",
                        "median_r2": "median held-out R² across the 20 candidates in that series",
                        "min_r2": "minimum held-out R² across those candidates (left errorbar)",
                        "max_r2": "maximum held-out R² across those candidates (right errorbar)",
                        "baseline_r2": "covariate-only baseline R² for that fold (vertical dash)",
                    },
                    notes=(
                        "Top-20 sets are selected within each model level by that level's global "
                        "LOCO R². Values and panel layout match the canonical figure; only the "
                        "within-panel axes are rotated."
                    ),
                )
            )
    write_source_data(stem, panels, outdir)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = env_bool("SMOKE_TEST") or env_bool("SENSITIVITY_SMOKE")
    include_combined = env_bool("SENSITIVITY_INCLUDE_COMBINED", default=False)
    skip_loro = env_bool("SKIP_LORO", default=False)
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = None
    if smoke:
        max_candidates = int(os.environ.get("SENSITIVITY_MAX_CANDIDATES", "24"))

    analysis_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / "country_region"
    local_root.mkdir(parents=True, exist_ok=True)

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    # Region split: merge subregion_name (kept out of build_model_df) by N_MEGA, then
    # pre-exclude main-analysis countries (prepare_analysis_table ties country
    # exclusion to split_col, which becomes subregion_name for LORO).
    original_model = build_original_model_df(raw, feature_names, analysis_cfg)
    sub_map = raw.assign(_k=raw["N_MEGA"].astype(str).str.strip()).set_index("_k")["subregion_name"]
    original_model["subregion_name"] = original_model["N_MEGA"].astype(str).str.strip().map(sub_map)
    excl_countries = set(analysis_cfg.get("exclude_countries", []))
    region_model = original_model[~original_model["country_clean"].isin(excl_countries)].copy()
    region_cfg = dict(analysis_cfg)
    region_cfg["exclude_countries"] = []

    # Both BAGs share one figure folder and one merged figure; cap split stays
    # (applied inside repo_sensitivity_figures_root).
    fig_dir = repo_sensitivity_figures_root(cfg, "country_region")
    fig_dir.mkdir(parents=True, exist_ok=True)
    perf_by_bag: dict[str, pd.DataFrame] = {}

    for bag in selected_bags(cfg, include_combined=include_combined):
        print(f"\n=== Country/region sensitivity | bag={bag} | smoke={smoke} ===")
        bag_dir = local_root / bag
        eval_dir = bag_dir / "eval_loro"  # heavy LORO eval: cap-invariant, reused
        # Per-cap leaf for the cap-dependent per-fold tables (top-20 sets change
        # with the order cap); eval_dir stays shared so the LORO eval is reused.
        out_dir = bag_dir / cap_split_subdir() if cap_split_subdir() else bag_dir
        for d in [bag_dir, eval_dir, out_dir, fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        country_perf = _extract_country_loco(bag, rungs)
        country_perf.to_csv(out_dir / "country_loco_per_fold.csv", index=False)

        parts = [country_perf]
        if not skip_loro:
            region_perf = _run_region_loro(
                bag, region_model, feature_names, region_cfg, rungs, eval_dir, n_jobs, max_candidates
            )
            region_perf.to_csv(out_dir / "region_loro_per_fold.csv", index=False)
            parts.append(region_perf)

        perf = pd.concat(parts, ignore_index=True)
        perf.to_csv(out_dir / "country_region_per_fold.csv", index=False)
        perf_by_bag[bag] = perf

    # One figure for both BAGs (structural rows first), drawn after the loop.
    _plot_country_region(perf_by_bag, rungs, fig_dir)
    _plot_country_region_rotated(perf_by_bag, rungs, fig_dir)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "country_region")
    copy_tree_contents(repo_sensitivity_figures_root(cfg, "country_region"), bundle_sensitivity_root(cfg) / "country_region")


if __name__ == "__main__":
    main()
