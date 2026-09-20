#!/usr/bin/env python3
"""Residualized-BAG sensitivity: evaluate baseline, single-exposure, and the
order-capped multi-exposure candidate pool directly under a BAG target that
already had age + sex + diagnosis regressed out.

ONE pass only -- everything is evaluated under the RESIDUALIZED target. There
is no separate evaluation under the original target here; that comparison was
already done and is not repeated by this script.

Candidate set per bag:
  - baseline: covariate-only (zero exposome predictors).
  - all 63 single-exposure candidates (one exposome feature each).
  - the multi-exposure candidate pool (order_min<=order<=PAPER_FIG_ORDER_MAX,
    default cap 30; set PAPER_FIG_ORDER_MAX to override.
    Sourced from load_fig2_candidate_pool, the only existing pool of
    multi-exposure o_min/o_max candidates in this repo -- reused here purely
    as a candidate source, with no comparison against the original target.

Residualization is LEAKAGE-FREE and per-fold: evaluate_candidates_by_rung
(fold_residualize=True) refits BAG ~ age + sex + diagnosis on each LOCO
fold's training countries only (sensitivity_common.build_fold_residualized_y
/ _residualize_train_fit) and applies it to every row, so a held-out test
country's target is never informed by a model that saw that same country's
data. This mirrors the existing fold_pca leakage-free pattern.

Country is deliberately NOT in the residualization formula: country is the
LOCO grouping variable for every candidate (including the baseline), not a
confound to strip out of BAG. Removing it from the target would change what
every candidate is scored against and conflate "control for country" with the
LOCO fold structure the main analysis already uses.

The standalone compute_residualized_bag_target.py script still produces a
pooled-fit residual CSV, but only as an informational sanity statistic
(overall residual SD per bag) -- its output is NOT fed into this evaluation
and must not be re-wired in, since that pooled fit would leak each fold's
held-out country into its own target.

The O-information synergy/redundancy labels (o_min/o_max) are a property of
the exposome matrix only, independent of which BAG target is regressed.

Lightweight stats -> outputs/sensitivity/residualized_bag/. Heavy eval folds
-> bundle via the shared evaluator's resume cache.

Cost note: with PAPER_FIG_ORDER_MAX=30 the multi-exposure pool is the complete
candidates per bag; together with the 63 single-exposure candidates and the
baseline that is ~864 models refit per rung per bag, across every active LOCO
fold. This is the dominant cost driver for external execution.
"""
from __future__ import annotations

import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_ROW_ORDER,
    LEVEL_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    SINGLE_COLOR,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from scripts.sensitivity_common import (
    active_rungs,
    analysis_cfg_from_config,
    baseline_candidate_df,
    build_original_model_df,
    bundle_sensitivity_root,
    combo_candidate_table,
    copy_tree_contents,
    evaluate_candidates_by_rung,
    filter_syn_pool,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    selected_bags,
    sensitivity_eval_work_root,
)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "y"} if val else default


# Same env-var contract as the paper-figure generators; the public default is
# the cap-30 multi-exposure pool.
ORDER_MAX = int(os.environ["PAPER_FIG_ORDER_MAX"]) if os.environ.get("PAPER_FIG_ORDER_MAX") else 30


def _cap_by_order(df: pd.DataFrame) -> pd.DataFrame:
    return df if ORDER_MAX is None else df[df["order"] <= ORDER_MAX]


def _best_rows_for_role(frame: pd.DataFrame, role: str, score_column: str = "r2") -> pd.DataFrame:
    selected = frame.loc[frame["candidate_role"].astype(str).eq(role)].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.sort_values(["rung_id", score_column], ascending=[True, False], kind="stable")
        .groupby("rung_id", sort=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )


def _build_summary_table(frame: pd.DataFrame, bag: str, rungs: list[str]) -> pd.DataFrame:
    rows = []
    for rung in rungs:
        sub = frame.loc[frame["rung_id"].eq(rung)].copy()
        if sub.empty:
            continue

        baseline = sub.loc[sub["candidate_role"].eq("baseline")].sort_values("r2", ascending=False).head(1)
        singles = sub.loc[sub["candidate_role"].eq("single_exposure")].sort_values("r2", ascending=False).head(1)
        best_syn = filter_syn_pool(
            sub.loc[sub["objective"].eq("o_min")], "o_min", "score"
        ).sort_values("r2", ascending=False).head(1)
        best_red = sub.loc[sub["objective"].eq("o_max")].sort_values("r2", ascending=False).head(1)

        base_r2 = float(baseline["r2"].iloc[0]) if len(baseline) else pd.NA
        single_r2 = float(singles["r2"].iloc[0]) if len(singles) else pd.NA
        syn_r2 = float(best_syn["r2"].iloc[0]) if len(best_syn) else pd.NA
        red_r2 = float(best_red["r2"].iloc[0]) if len(best_red) else pd.NA

        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "baseline_r2": base_r2,
                "best_single_r2": single_r2,
                "best_synergy_r2": syn_r2,
                "best_redundancy_r2": red_r2,
                "delta_single_vs_baseline": single_r2 - base_r2 if pd.notna(base_r2) and pd.notna(single_r2) else pd.NA,
                "delta_synergy_vs_baseline": syn_r2 - base_r2 if pd.notna(base_r2) and pd.notna(syn_r2) else pd.NA,
                "delta_redundancy_vs_baseline": red_r2 - base_r2 if pd.notna(base_r2) and pd.notna(red_r2) else pd.NA,
                "best_single_candidate_id": singles["candidate_id"].iloc[0] if len(singles) else pd.NA,
                "best_synergy_candidate_id": best_syn["candidate_id"].iloc[0] if len(best_syn) else pd.NA,
                "best_redundancy_candidate_id": best_red["candidate_id"].iloc[0] if len(best_red) else pd.NA,
                "n_candidates": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _draw_residualized_row(
    summary: pd.DataFrame,
    bag: str,
    ax,
    row_letter: str,
    *,
    r2_estimand: str = "global out-of-fold LOCO R²",
) -> list[Panel]:
    """Draw one BAG's row and return its Source Data panel.

    The row label follows the plot_fig2_grid_v3 convention: "<letter>. <BAG>"
    left-aligned on the row's axis.
    """
    if summary.empty:
        return []

    x = list(range(len(summary)))

    ax.plot(x, summary["baseline_r2"], color="black", lw=2.2, marker="s", ms=5, label="Baseline")

    has_single = summary["best_single_r2"].notna().any()
    has_syn = summary["best_synergy_r2"].notna().any()
    has_red = summary["best_redundancy_r2"].notna().any()

    if has_single:
        ax.plot(x, summary["best_single_r2"], color=SINGLE_COLOR, lw=1.9, marker="o", ms=5, label="Best single exposure")
    if has_syn:
        ax.plot(x, summary["best_synergy_r2"], color=SYN_COLOR, lw=1.9, marker="o", ms=5, label="Best synergy (o_min)")
    if has_red:
        ax.plot(x, summary["best_redundancy_r2"], color=RED_COLOR, lw=1.9, marker="o", ms=5, label="Best redundancy (o_max)")

    ax.set_xticks(x)
    # "xgb_tree_d1" -> "d1": strip the prefix only (replacing it WITH "d" yielded "dd1").
    ax.set_xticklabels([r.replace("xgb_tree_", "") if r != "ols" else "OLS" for r in summary["rung_id"]])
    ax.set_xlabel("Model level")
    ax.set_ylabel("R² LOCO")
    ax.set_title(f"{row_letter}. {BAG_LABELS[bag]}", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    style_axis(ax)

    handles = [
        mpatches.Patch(color="black", label="Baseline"),
    ]
    if has_single:
        handles.append(mpatches.Patch(color=SINGLE_COLOR, label="Best single exposure"))
    if has_syn:
        handles.append(mpatches.Patch(color=SYN_COLOR, label="Best synergy (o_min)"))
    if has_red:
        handles.append(mpatches.Patch(color=RED_COLOR, label="Best redundancy (o_max)"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")

    frame = summary[
        [
            c
            for c in [
                "rung_id",
                "baseline_r2",
                "best_single_r2",
                "best_synergy_r2",
                "best_redundancy_r2",
            ]
            if c in summary.columns
        ]
    ].reset_index(drop=True)
    frame = frame.rename(columns={"rung_id": "model_level"})
    frame["model_level"] = frame["model_level"].map(LEVEL_LABELS)
    return [
            Panel(
                panel_id=f"{row_letter}_{bag}_residualized_r2",
                frame=frame,
                description=(
                    f"Held-out LOCO R² by model level under a {BAG_LABELS[bag]} target with age, sex and "
                    "diagnosis already regressed out."
                ),
                columns={
                    "model_level": "model level (OLS, d1, d2, d3)",
                    "baseline_r2": "covariate-only baseline, zero exposome predictors",
                    "best_single_r2": "best of the 63 single-exposure candidates",
                    "best_synergy_r2": "best synergistic (o_min) multi-exposure candidate",
                    "best_redundancy_r2": "best redundant (o_max) multi-exposure candidate",
                },
                notes=(
                    "Residualization is per-fold and leakage-free: BAG ~ age + sex + diagnosis "
                    "is refit on each LOCO fold's training countries only. Country is "
                    f"deliberately not in the residualization formula. R² estimand: {r2_estimand}."
                ),
            )
    ]


def _plot_residualized(
    summary: pd.DataFrame,
    bags: list[str],
    outdir,
    stem: str,
    *,
    r2_estimand: str = "global out-of-fold LOCO R²",
) -> None:
    """One figure with a COLUMN per BAG, structural leftmost. No suptitle.

    Each BAG is a single panel, so they sit side by side rather than stacked;
    the shared y-axis makes the two modalities directly comparable.
    """
    present = [b for b in bags if not summary.loc[summary["bag"].eq(b)].empty]
    if not present:
        return
    fig, axes = plt.subplots(
        1, len(present), figsize=(7.2 * len(present), 4.6), squeeze=False,
        gridspec_kw={"wspace": 0.22},
    )
    panels: list[Panel] = []
    for i, bag in enumerate(present):
        panels.extend(
            _draw_residualized_row(
                summary.loc[summary["bag"].eq(bag)].copy(),
                bag,
                axes[0][i],
                ROW_LETTERS[i],
                r2_estimand=r2_estimand,
            )
        )
    save_figure(fig, stem, outdir)
    plt.close(fig)
    write_source_data(stem, panels, outdir)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = _env_bool("SMOKE_TEST") or _env_bool("SENSITIVITY_SMOKE")
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = int(os.environ.get("SENSITIVITY_MAX_CANDIDATES", "24")) if smoke else None

    base_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / "residualized_bag"
    local_root.mkdir(parents=True, exist_ok=True)
    # Heavy per-fold LOCO eval artifacts stay in the external runtime; only summary
    # tables go to local_root (the checkout).
    eval_root = sensitivity_eval_work_root(cfg, "residualized_bag")

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = build_original_model_df(raw, feature_names, base_cfg)

    all_rows = []
    for bag in selected_bags(cfg, include_combined=True):
        print(f"\n=== residualized-bag sensitivity | bag={bag} | smoke={smoke} | order_max={ORDER_MAX} ===")
        bag_dir = local_root / bag
        bag_dir.mkdir(parents=True, exist_ok=True)
        eval_dir = eval_root / bag
        eval_dir.mkdir(parents=True, exist_ok=True)

        multi_pool = _cap_by_order(load_fig2_candidate_pool(bag))
        if max_candidates is not None:
            # Sample per objective (not a plain head()) so smoke mode keeps
            # both o_min and o_max candidates.
            per_objective = max(1, max_candidates // max(1, multi_pool["objective"].nunique()))
            multi_pool = (
                multi_pool.groupby("objective", group_keys=False)
                .head(per_objective)
                .reset_index(drop=True)
            )

        # All single-exposure candidates (one exposome feature each).
        single_pool = combo_candidate_table(
            feature_names,
            family="single_exposure",
            source_label="single_exposure",
            order_min=1,
            order_max=1,
            prefix=f"single_{bag}",
            extra={"objective": "single_exposure"},
        )
        # combo_candidate_table sets feature_id to the generated candidate_id;
        # overwrite with the actual exposome feature name for readability.
        single_pool["feature_id"] = single_pool["nplet_vars"].apply(lambda v: v[0])

        pool = pd.concat([multi_pool, single_pool, baseline_candidate_df()], ignore_index=True)
        print(
            f"  candidate pool: {len(pool)} total "
            f"(multi_exposure={len(multi_pool)}, single_exposure={len(single_pool)}, baseline=1)"
        )

        bag_rows = []
        for rung in rungs:
            print(f"  rung={rung}: evaluating {len(pool)} candidates under the residualized target")
            summary_resid, _ = evaluate_candidates_by_rung(
                model_df=full_model,
                candidate_df=pool,
                exposome_cols=feature_names,
                bag=bag,
                rungs=[rung],
                analysis_cfg=base_cfg,
                outdir=eval_dir / f"residualized_eval_{rung}",
                n_jobs=n_jobs,
                max_candidates=None,
                fold_residualize=True,
            )
            resid_rows = summary_resid.copy()
            # candidate_role = objective (o_min=synergistic, o_max=redundant,
            # single_exposure, baseline) -- already present via the
            # candidate_df merge inside evaluate_candidates_by_rung.
            resid_rows["candidate_role"] = resid_rows["objective"].astype(str)
            resid_rows = resid_rows.rename(columns={"global_oof_r2": "r2"})
            # Carry the evaluated O-info (`score`) through for the optional
            # negative-O-info synergy criterion (SYN_OINFO_NEGATIVE).
            keep_cols = ["rung_id", "candidate_role", "objective", "candidate_id", "feature_id", "r2"]
            if "score" in resid_rows.columns:
                keep_cols.append("score")
            bag_rows.append(resid_rows[keep_cols])

        bag_combined = pd.concat(bag_rows, ignore_index=True)
        bag_combined["bag"] = bag
        bag_combined.to_csv(bag_dir / f"{bag}_global_all_rungs.csv", index=False)

        baseline_compare = bag_combined[bag_combined["candidate_role"] == "baseline"]
        print("  Baseline under residualized target:")
        print(baseline_compare[["rung_id", "r2"]].to_string(index=False))

        all_rows.append(bag_combined)

    # Main-k10 jobs may be scheduled one BAG at a time.  The paper contract is
    # nevertheless one Structural+Functional figure, exactly as in the parent
    # implementation when both bags are selected together.  Re-read any
    # already-completed sibling BAG summaries before assembling the delivery so
    # the job that finishes second writes that single complete figure without
    # repeating a refit.
    persisted_rows: list[pd.DataFrame] = []
    for persisted_bag in BAG_ROW_ORDER:
        persisted = local_root / persisted_bag / f"{persisted_bag}_global_all_rungs.csv"
        if persisted.is_file():
            persisted_rows.append(pd.read_csv(persisted))
    combined = pd.concat(persisted_rows, ignore_index=True) if persisted_rows else pd.DataFrame()
    combined.to_csv(local_root / "global_all_rungs.csv", index=False)

    summary_rows = []
    for bag in BAG_ROW_ORDER:
        bag_frame = combined.loc[combined["bag"].eq(bag)].copy()
        if not bag_frame.empty:
            summary_rows.append(_build_summary_table(bag_frame, bag, rungs))
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary.to_csv(local_root / "residualized_bag_best_by_rung.csv", index=False)

    fig_root = repo_sensitivity_figures_root(cfg, "residualized_bag")
    # Structural and functional merge into one two-row figure; the optional
    # combined BAG is not part of that pair and keeps its own single-row figure.
    _plot_residualized(summary, list(BAG_ROW_ORDER), fig_root, "residualized_bag")
    if not summary.loc[summary["bag"].eq("combined")].empty:
        _plot_residualized(summary, ["combined"], fig_root / "combined", "residualized_bag_combined")

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "residualized_bag")
    copy_tree_contents(
        fig_root,
        bundle_sensitivity_root(cfg) / "residualized_bag",
    )


if __name__ == "__main__":
    main()
