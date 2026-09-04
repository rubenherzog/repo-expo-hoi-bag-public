#!/usr/bin/env python3
"""Healthy-controls-only (CN) sensitivity for the exposome-BAG analysis.

Audit finding: scripts/run_dx_stratified_eval.py trains on the POOLED cohort
(CN+AD+MCI+FTD) and only stratifies the test by diagnosis, so its CN R2 is not a
CN-only model. This runner instead trains AND tests on CN subjects only and drops
diagnosis as a baseline covariate (it is constant), giving a true healthy-controls
exposome-BAG association. The original Fig.2 greedy candidate pool is re-evaluated
under standard country LOCO.

Lightweight stats -> outputs/sensitivity/hc_only/. Heavy eval folds -> bundle via
the shared evaluator's resume cache.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.sensitivity_common import (
    RUNG_COLORS,
    RUNG_LABELS,
    active_rungs,
    analysis_cfg_from_config,
    baseline_candidate_df,
    build_original_model_df,
    bundle_sensitivity_root,
    copy_tree_contents,
    evaluate_candidates_by_rung,
    filter_syn_pool,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

C_SYN = "#2166ac"
C_RED = "#b2182b"
C_BEST = "#238b45"
C_BASE = "#666666"
TOP_K = 20


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "y"} if val else default


def _global_perf(summary: pd.DataFrame, rungs: list[str]) -> tuple[pd.DataFrame, dict]:
    s = summary.copy()
    s["score"] = pd.to_numeric(s["score"], errors="coerce")
    s["global_oof_r2"] = pd.to_numeric(s["global_oof_r2"], errors="coerce")
    base_map = {}
    rows = []
    top_ids: dict = {}
    for rung in rungs:
        rr = s[s["rung_id"].astype(str) == rung]
        bl = rr[rr["candidate_id"].astype(str) == "__baseline__"]["global_oof_r2"]
        base_map[rung] = float(bl.iloc[0]) if len(bl) else np.nan
        models = rr[rr["candidate_id"].astype(str) != "__baseline__"]
        # syn/red are the greedy discovery objectives (o_min/o_max), NOT the sign of
        # the evaluated O-info (a candidate found via o_min can evaluate to score>0 and
        # still IS synergistic-route).
        syn = filter_syn_pool(
            models[models["objective"].astype(str) == "o_min"], "o_min", "score"
        ).nlargest(TOP_K, "global_oof_r2")
        red = models[models["objective"].astype(str) == "o_max"].nlargest(TOP_K, "global_oof_r2")
        top_ids[rung] = {"syn": set(syn["candidate_id"].astype(str)), "red": set(red["candidate_id"].astype(str))}
        for label, sub in [("top20_syn", syn), ("top20_red", red)]:
            v = sub["global_oof_r2"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rows.append({
                "rung_id": rung, "series": label,
                "median_r2": float(np.median(v)) if len(v) else np.nan,
                "min_r2": float(np.min(v)) if len(v) else np.nan,
                "max_r2": float(np.max(v)) if len(v) else np.nan,
                "n": int(len(v)),
            })
        allv = models["global_oof_r2"].to_numpy(dtype=float)
        allv = allv[np.isfinite(allv)]
        rows.append({"rung_id": rung, "series": "best", "median_r2": float(np.max(allv)) if len(allv) else np.nan,
                     "min_r2": np.nan, "max_r2": np.nan, "n": int(len(allv))})
        rows.append({"rung_id": rung, "series": "baseline", "median_r2": base_map[rung],
                     "min_r2": np.nan, "max_r2": np.nan, "n": 1})
    return pd.DataFrame(rows), top_ids


def _plot_hc(bag: str, gperf: pd.DataFrame, country: pd.DataFrame, top_ids: dict,
             rungs: list[str], outdir: Path) -> None:
    fig = plt.figure(figsize=(3.6 * len(rungs), 7.2))
    gs = fig.add_gridspec(2, len(rungs), hspace=0.5, wspace=0.3)

    # Row 0: global rung panel spanning all columns.
    axg = fig.add_subplot(gs[0, :])
    x = np.arange(len(rungs))
    HALF = 0.3
    for i, rung in enumerate(rungs):
        rr = gperf[gperf["rung_id"].astype(str) == rung]
        for series, color, x0, x1 in [("top20_syn", C_SYN, i - HALF, i), ("top20_red", C_RED, i, i + HALF)]:
            row = rr[rr["series"] == series]
            if row.empty or not np.isfinite(row["min_r2"].iloc[0]):
                continue
            lo, hi = float(row["min_r2"].iloc[0]), float(row["max_r2"].iloc[0])
            axg.add_patch(mpl.patches.Rectangle((x0, lo), x1 - x0, hi - lo, facecolor=color, alpha=0.55, lw=0))
        base = rr[rr["series"] == "baseline"]["median_r2"]
        best = rr[rr["series"] == "best"]["median_r2"]
        if len(base) and np.isfinite(base.iloc[0]):
            axg.plot([i - HALF, i + HALF], [base.iloc[0]] * 2, color="black", lw=3, ls="-")
        if len(best) and np.isfinite(best.iloc[0]):
            axg.plot([i - HALF, i + HALF], [best.iloc[0]] * 2, color=C_BEST, lw=3, ls="--")
    axg.set_xticks(x)
    axg.set_xticklabels([RUNG_LABELS[r] for r in rungs])
    axg.set_ylabel("R² LOCO (CN-only)")
    axg.set_title(f"Healthy-controls-only global rung performance: {bag} BAG", fontsize=11, fontweight="bold", loc="left")
    axg.legend(handles=[
        mpl.patches.Patch(color=C_SYN, alpha=0.55, label="Top-20 syn"),
        mpl.patches.Patch(color=C_RED, alpha=0.55, label="Top-20 red"),
        mpl.lines.Line2D([0], [0], color="black", lw=3, label="Baseline"),
        mpl.lines.Line2D([0], [0], color=C_BEST, lw=3, ls="--", label="Best model"),
    ], fontsize=7, loc="upper left", ncol=2)
    axg.grid(axis="y", alpha=0.3)

    # Row 1: one per-country panel per rung.
    c = country.copy()
    c["r2"] = pd.to_numeric(c["r2"], errors="coerce")
    for ci, rung in enumerate(rungs):
        ax = fig.add_subplot(gs[1, ci])
        rr = c[c["rung_id"].astype(str) == rung]
        folds = sorted(rr["fold_country"].dropna().astype(str).unique())
        xmap = {f: i for i, f in enumerate(folds)}
        for series, color, dx, ids in [("syn", C_SYN, -0.15, top_ids[rung]["syn"]),
                                       ("red", C_RED, 0.15, top_ids[rung]["red"])]:
            sub = rr[rr["candidate_id"].astype(str).isin(ids)]
            agg = sub.groupby(sub["fold_country"].astype(str))["r2"].median()
            ax.scatter([xmap[f] + dx for f in agg.index], agg.values, s=12, color=color, alpha=0.8,
                       label={"syn": "Top-20 syn", "red": "Top-20 red"}[series])
        # Per-country baseline (covariate-only) from the __baseline__ model on the CN sample.
        bl = rr[rr["candidate_id"].astype(str) == "__baseline__"]
        if not bl.empty:
            bagg = bl.groupby(bl["fold_country"].astype(str))["r2"].median()
            ax.scatter([xmap[f] for f in bagg.index if f in xmap],
                       [bagg[f] for f in bagg.index if f in xmap],
                       marker="_", s=80, color=C_BASE, zorder=1,
                       label="Baseline" if ci == len(rungs) - 1 else None)
        ax.axhline(0, color="#bbbbbb", lw=0.7, ls=":")
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels(folds, rotation=90, fontsize=5)
        ax.set_title(RUNG_LABELS[rung], fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Per-country R² LOCO")
        if ci == len(rungs) - 1:
            ax.legend(fontsize=6)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"hc_only_sensitivity_{bag}.png", dpi=210, bbox_inches="tight")
    fig.savefig(outdir / f"hc_only_sensitivity_{bag}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = _env_bool("SMOKE_TEST") or _env_bool("SENSITIVITY_SMOKE")
    include_combined = _env_bool("SENSITIVITY_INCLUDE_COMBINED", default=False)
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = int(os.environ.get("SENSITIVITY_MAX_CANDIDATES", "24")) if smoke else None

    analysis_cfg = analysis_cfg_from_config(cfg)
    # CN-only: keep CN subjects, drop diagnosis as a covariate (constant).
    hc_cfg = dict(analysis_cfg)
    hc_cfg["include_diagnosis"] = False
    hc_cfg["exclude_diagnosis"] = []

    local_root = repo_sensitivity_root(cfg) / "hc_only"
    local_root.mkdir(parents=True, exist_ok=True)

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = build_original_model_df(raw, feature_names, analysis_cfg)
    cn_model = full_model[full_model["Diagnosis"].astype(str) == "CN"].copy()

    for bag in selected_bags(cfg, include_combined=include_combined):
        print(f"\n=== HC-only (CN) sensitivity | bag={bag} | smoke={smoke} | n_cn={len(cn_model)} ===")
        bag_dir = local_root / bag
        eval_dir = bag_dir / "eval"
        fig_dir = bag_dir / "figures"
        for d in [bag_dir, eval_dir, fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        pool = load_fig2_candidate_pool(bag)
        if max_candidates is not None:
            pool = pool.head(max_candidates).copy()
        pool = pd.concat([pool, baseline_candidate_df()], ignore_index=True)

        summary, country = evaluate_candidates_by_rung(
            model_df=cn_model,
            candidate_df=pool,
            exposome_cols=feature_names,
            bag=bag,
            rungs=rungs,
            analysis_cfg=hc_cfg,
            outdir=eval_dir,
            n_jobs=n_jobs,
            max_candidates=None,
        )
        summary["bag"] = bag
        country["bag"] = bag
        summary.to_csv(bag_dir / "hc_only_global_all.csv", index=False)
        country.to_csv(bag_dir / "hc_only_country_all.csv", index=False)

        gperf, top_ids = _global_perf(summary, rungs)
        gperf.to_csv(bag_dir / "hc_only_global_perf.csv", index=False)
        _plot_hc(bag, gperf, country, top_ids, rungs, fig_dir)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "hc_only")


if __name__ == "__main__":
    main()
