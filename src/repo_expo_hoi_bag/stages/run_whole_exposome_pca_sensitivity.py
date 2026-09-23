#!/usr/bin/env python3
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
    LEVEL_LABELS,
    ROW_LETTERS,
    save_figure,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.sensitivity_common import (
    RUNG_COLORS,
    RUNG_LABELS,
    active_rungs,
    analysis_cfg_from_config,
    build_original_model_df,
    build_whole_pca_model_df,
    bundle_sensitivity_root,
    cap_split_subdir,
    copy_tree_contents,
    env_bool,
    evaluate_candidates_by_rung,
    filtered_rows_for_bag,
    incremental_pc_candidate_table,
    load_existing_baselines,
    load_original_complete_best,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    selected_bags,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def country_balanced_summary(summary: pd.DataFrame, country: pd.DataFrame) -> pd.DataFrame:
    """Use the same unweighted country-mean R² as the reference lines."""
    keys = ["candidate_id", "rung_id"]
    means = (
        country.assign(r2=pd.to_numeric(country["r2"], errors="coerce"))
        .groupby(keys, observed=True)["r2"]
        .mean()
        .rename("country_balanced_r2")
        .reset_index()
    )
    out = summary.merge(means, on=keys, how="left", validate="one_to_one")
    if out["country_balanced_r2"].isna().any():
        raise ValueError("Country-balanced PCA R² is missing for one or more candidates")
    out = out.rename(columns={"global_oof_r2": "participant_global_oof_r2"})
    out["global_oof_r2"] = out["country_balanced_r2"]
    return out


def _draw_pca_rows(
    bag: str,
    variance: pd.DataFrame,
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    original_best: pd.DataFrame,
    rungs: list[str],
    ax_var,
    ax_perf,
    row_letter: str,
    show_legend: bool,
) -> list[Panel]:
    """Draw one BAG as a single row of two columns (variance, PC performance).

    The row label follows the plot_fig2_grid_v3 convention: "<letter>. <BAG> ..."
    left-aligned on the row's first column. Returns that BAG's Source Data panels.
    """
    ax_cum = ax_var.twinx()
    x = variance["pc_n"].to_numpy(dtype=int)
    ax_var.plot(x, variance["explained_variance_ratio"], color="#2c7fb8", marker="o", lw=1.8, label="Explained variance")
    ax_cum.plot(
        x,
        variance["cumulative_explained_variance_ratio"],
        color="#f03b20",
        marker="s",
        lw=1.8,
        label="Cumulative explained variance",
    )
    ax_var.set_xlabel("Number of PCs")
    ax_var.set_ylabel("Explained variance ratio")
    ax_cum.set_ylabel("Cumulative explained variance ratio")
    ax_var.set_title(
        f"{row_letter}. {BAG_LABELS[bag]} — whole-exposome PCA variance",
        loc="left", fontsize=11,
    )
    ax_var.set_xticks(x)
    if show_legend:
        lines = ax_var.get_lines() + ax_cum.get_lines()
        ax_var.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")
    # Only the top spine is dropped here: this row carries a twin y-axis on the
    # right, so style_axis's right-spine removal would strip that axis's frame.
    ax_var.spines["top"].set_visible(False)
    ax_cum.spines["top"].set_visible(False)

    base_map = dict(zip(baselines["rung_id"], baselines["baseline_r2"]))
    best_map = (
        dict(zip(original_best["rung_id"], original_best["best_model_r2"]))
        if original_best is not None and not original_best.empty
        else {}
    )
    for rung in rungs:
        rr = summary[summary["rung_id"].astype(str) == rung].copy()
        rr["pc_n"] = pd.to_numeric(rr["pc_n"], errors="coerce")
        rr = rr.sort_values("pc_n")
        ax_perf.plot(
            rr["pc_n"],
            rr["global_oof_r2"],
            color=RUNG_COLORS[rung],
            marker="o",
            lw=1.8,
            label=LEVEL_LABELS[rung],
        )
        if rung in base_map:
            ax_perf.axhline(base_map[rung], color=RUNG_COLORS[rung], ls="--", lw=1.0, alpha=0.65)
        if rung in best_map:
            ax_perf.axhline(best_map[rung], color=RUNG_COLORS[rung], ls="-", lw=1.4, alpha=0.85)
    ax_perf.set_xlabel("Number of PCs")
    ax_perf.set_ylabel("R² LOCO")
    ax_perf.set_title(
        "Incremental PCA predictors by model level",
        loc="left", fontsize=11,
    )
    ax_perf.set_xticks(x)
    if show_legend:
        handles, labels = ax_perf.get_legend_handles_labels()
        handles.extend(
            [
                mpl.lines.Line2D([0], [0], color="#666666", ls="--", lw=1.0),
                mpl.lines.Line2D([0], [0], color="#666666", ls="-", lw=1.4),
            ]
        )
        labels.extend(["Level baseline (dashed)", "Original best model (solid)"])
        ax_perf.legend(handles, labels, fontsize=7, ncol=2)
    style_axis(ax_perf)

    # Source Data: one panel is the variance curve pair, the other the per-level
    # incremental-PC performance plus the two reference lines drawn on it.
    perf_frame = (
        summary[summary["rung_id"].astype(str).isin(rungs)]
        .assign(pc_n=lambda d: pd.to_numeric(d["pc_n"], errors="coerce"))
        .sort_values(["rung_id", "pc_n"])[["rung_id", "pc_n", "global_oof_r2"]]
        .reset_index(drop=True)
    )
    perf_frame["level_baseline_r2"] = perf_frame["rung_id"].map(base_map)
    perf_frame["original_best_model_r2"] = perf_frame["rung_id"].map(best_map)
    perf_frame = perf_frame.rename(columns={"rung_id": "model_level"})
    perf_frame["model_level"] = perf_frame["model_level"].map(LEVEL_LABELS)

    performance_description = (
        "unweighted mean of held-out country-level LOCO R² for that level and PC count"
        if os.environ.get("WHOLE_PCA_PERFORMANCE_ESTIMATOR", "").strip() == "country-balanced"
        else "global out-of-fold LOCO R² for that level and PC count"
    )
    return [
            Panel(
                panel_id=f"{row_letter}1_{BAG_SHORT[bag]}_pca_variance",
                frame=variance[
                    ["pc_n", "explained_variance_ratio", "cumulative_explained_variance_ratio"]
                ].reset_index(drop=True),
                description=(
                    f"Whole-exposome PCA variance by number of components, {BAG_LABELS[bag]}."
                ),
                columns={
                    "pc_n": "number of principal components",
                    "explained_variance_ratio": "variance explained by that component (left axis)",
                    "cumulative_explained_variance_ratio": "cumulative variance explained (right axis)",
                },
            ),
            Panel(
                panel_id=f"{row_letter}2_{BAG_SHORT[bag]}_incremental_pcs",
                frame=perf_frame,
                description=(
                    f"Held-out LOCO R² of incremental whole-exposome PCA predictors by model "
                    f"level, {BAG_LABELS[bag]}."
                ),
                columns={
                    "model_level": "model level (OLS, d1, d2, d3)",
                    "pc_n": "number of leading PCs entered as predictors",
                    "global_oof_r2": performance_description,
                    "level_baseline_r2": "covariate-only baseline for that level (dashed line)",
                    "original_best_model_r2": "original best model for that level (solid line)",
                },
                notes=(
                    "PCA loadings are fit per LOCO fold on training countries only; scores are "
                    "projected for all subject rows, so evaluation stays subject-level."
                ),
            ),
    ]


def _plot_pca(
    bag_inputs: dict[str, dict],
    outdir: Path,
    rungs: list[str],
    *,
    write_rendered_source_data: bool = True,
) -> None:
    """One 2x2 figure: structural row first, then functional.

    Each BAG is one row of two columns (PCA variance, incremental-PC
    performance). No suptitle.
    """
    bags = [b for b in BAG_ROW_ORDER if b in bag_inputs]
    if not bags:
        return
    fig, axes = plt.subplots(
        len(bags), 2, figsize=(19.0, 4.6 * len(bags)),
        squeeze=False, gridspec_kw={"hspace": 0.34, "wspace": 0.26},
    )
    panels: list[Panel] = []
    for i, bag in enumerate(bags):
        d = bag_inputs[bag]
        panels.extend(
            _draw_pca_rows(
                bag, d["variance"], d["summary"], d["baselines"], d["original_best"],
                rungs, axes[i][0], axes[i][1], row_letter=ROW_LETTERS[i], show_legend=(i == 0),
            )
        )
    stem = "whole_exposome_pca_sensitivity"
    save_figure(fig, stem, outdir)
    plt.close(fig)
    if write_rendered_source_data:
        write_source_data(stem, panels, outdir)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = env_bool("SMOKE_TEST") or env_bool("SENSITIVITY_SMOKE")
    include_combined = env_bool("SENSITIVITY_INCLUDE_COMBINED", default=False)
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    max_pcs = int(os.environ.get("SENSITIVITY_MAX_PCS", cfg["whole_exposome_pca"].get("max_pcs", 10)))
    if smoke:
        max_pcs = min(max_pcs, 3)
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))

    analysis_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / "whole_exposome_pca"
    local_root.mkdir(parents=True, exist_ok=True)

    if env_bool("WHOLE_PCA_REBUILD_FROM_SUMMARIES"):
        bag_inputs: dict[str, dict] = {}
        for bag in selected_bags(cfg, include_combined=include_combined):
            bag_dir = local_root / bag
            summary = pd.read_csv(bag_dir / "whole_exposome_pca_global_all.csv")
            country = pd.read_csv(bag_dir / "whole_exposome_pca_country_all.csv")
            if os.environ.get("WHOLE_PCA_PERFORMANCE_ESTIMATOR", "").strip() == "country-balanced":
                summary = country_balanced_summary(summary, country)
            bag_inputs[bag] = {
                "variance": pd.read_csv(bag_dir / "whole_exposome_pca_variance.csv"),
                "summary": summary,
                "baselines": pd.read_csv(bag_dir / "existing_baselines_by_rung.csv"),
                "original_best": pd.read_csv(bag_dir / "original_complete_best_by_rung.csv"),
            }
        fig_dir = repo_sensitivity_figures_root(cfg, "whole_exposome_pca")
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_pca(bag_inputs, fig_dir, rungs)
        copy_tree_contents(fig_dir, bundle_sensitivity_root(cfg) / "whole_exposome_pca")
        return

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    original_model = build_original_model_df(raw, feature_names, analysis_cfg)

    # Both BAGs share one figure folder and one merged figure; cap split stays
    # (applied inside repo_sensitivity_figures_root).
    fig_dir = repo_sensitivity_figures_root(cfg, "whole_exposome_pca")
    fig_dir.mkdir(parents=True, exist_ok=True)
    bag_inputs: dict[str, dict] = {}

    for bag in selected_bags(cfg, include_combined=include_combined):
        print(f"\n=== Whole-exposome PCA sensitivity | bag={bag} | smoke={smoke} ===")
        fit_rows = filtered_rows_for_bag(original_model, bag, analysis_cfg)
        bag_dir = local_root / bag
        eval_dir = bag_dir / "eval"
        for d in [eval_dir, fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        pca_model, variance, pc_names = build_whole_pca_model_df(original_model, fit_rows.index, feature_names, max_pcs)
        candidates = incremental_pc_candidate_table(pc_names)
        variance.to_csv(bag_dir / "whole_exposome_pca_variance.csv", index=False)
        candidates.to_csv(bag_dir / "whole_exposome_pca_candidates.csv", index=False)

        # Carry the raw exposome features alongside the (descriptive) full-cohort PC
        # scores so the evaluator can refit PCA per LOCO fold on training rows only.
        eval_df = pca_model.join(original_model[feature_names])
        summary, country = evaluate_candidates_by_rung(
            model_df=eval_df,
            candidate_df=candidates,
            exposome_cols=pc_names,
            bag=bag,
            rungs=rungs,
            analysis_cfg=analysis_cfg,
            outdir=eval_dir,
            n_jobs=n_jobs,
            max_candidates=None,
            fold_pca={"mode": "whole", "raw_cols": list(feature_names), "max_pcs": max_pcs},
        )
        summary["bag"] = bag
        country["bag"] = bag
        summary.to_csv(bag_dir / "whole_exposome_pca_global_all.csv", index=False)
        country.to_csv(bag_dir / "whole_exposome_pca_country_all.csv", index=False)
        baselines = load_existing_baselines(bag)
        baselines.to_csv(bag_dir / "existing_baselines_by_rung.csv", index=False)
        # Cap-dependent reference (cap21 vs complete) -> per-cap leaf so the two
        # variants do not overwrite; PCA eval + baselines stay shared in bag_dir.
        original_best = load_original_complete_best(bag)
        ref_dir = bag_dir / cap_split_subdir() if cap_split_subdir() else bag_dir
        ref_dir.mkdir(parents=True, exist_ok=True)
        original_best.to_csv(ref_dir / "original_complete_best_by_rung.csv", index=False)
        bag_inputs[bag] = {
            "variance": variance,
            "summary": summary,
            "baselines": baselines,
            "original_best": original_best,
        }

    # One figure for both BAGs (structural rows first), drawn after the loop.
    _plot_pca(bag_inputs, fig_dir, rungs)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "whole_exposome_pca")
    copy_tree_contents(repo_sensitivity_figures_root(cfg, "whole_exposome_pca"), bundle_sensitivity_root(cfg) / "whole_exposome_pca")


if __name__ == "__main__":
    main()
