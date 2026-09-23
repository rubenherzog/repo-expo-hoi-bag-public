#!/usr/bin/env python3
"""Education + scanner covariate sensitivity for the exposome-BAG analysis.

Re-evaluates the full Fig.2 candidate pool per rung under the SAME baseline
covariate set used everywhere else in the pipeline (age + year + sex +
diagnosis), plus three covariate ADD-ONS, all on the SAME complete-case
subsample (rows where Edu and scanner_id are both resolvable) so that R2
differences are attributable to the added covariates, not to a change in
sample:

- plus_education:           baseline + years of education (Edu, continuous)
- plus_scanner:              baseline + scanner identity (resonador -> scanner_id, dummy)
- plus_education_scanner:    baseline + education + scanner

The baseline arm is always (re-)evaluated too -- it is the reference line each
variant is compared against, not a fourth variant. Select a subset of the
three add-ons via EDU_SCANNER_VARIANTS (comma-separated labels from the list
above); defaults to all three. The join NaN rate against the scanner table is
reported explicitly rather than imputed.

Figure follows the same convention as plot_fig2_grid_v3.py's panel B: per model
level a jittered strip of the top-20 candidates (synergistic left, redundant
right) with an IQR box, median bar and 1.5x-IQR whiskers over it, plus a solid
black baseline line. One figure with a row per BAG (structural first) and one
panel per covariate add-on within the row.

Lightweight stats -> outputs/sensitivity/education_scanner_baseline/. Heavy
eval folds -> bundle via the shared evaluator's resume cache.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from repo_expo_hoi_bag.analysis.selection import select_baseline_per_rung, select_best_per_rung
from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_ROW_ORDER,
    BAG_SHORT,
    LEVEL_LABELS,
    OBJECTIVE_ARM_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    RUNG_ORDER,
    SYN_COLOR,
    SINGLE_COLOR,
    save_figure,
    style_axis,
)
from scripts.exposome_domains import unweighted_domain_stats
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from scripts.sensitivity_common import (
    CHECKOUT_ROOT,
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
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    selected_bags,
    sensitivity_eval_work_root,
)

SCANNER_SHEET = "Sheet 1"

BASELINE_LABEL = "baseline_covariates"

# The three covariate add-ons (NOT including the baseline itself -- that's the
# reference every panel is compared against). Order matters: it's also the
# figure's panel order.
COVARIATE_VARIANTS: dict[str, dict[str, bool]] = {
    "plus_education": {"include_education": True},
    "plus_scanner": {"include_scanner": True},
    "plus_education_scanner": {"include_education": True, "include_scanner": True},
}
# Compact variant tags for Source Data sheet names (xlsx caps names at 31 chars).
_VARIANT_SHORT = {
    "plus_education": "edu",
    "plus_scanner": "scan",
    "plus_education_scanner": "eduscan",
}


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "y"} if val else default


def _load_scanner_table() -> pd.DataFrame:
    raw_path = os.environ.get("SCANNER_XLSX_PATH", "").strip()
    if not raw_path:
        raise RuntimeError(
            "SCANNER_XLSX_PATH is not set. Point it at the scanner metadata workbook "
            "(N_MEGA + resonador columns, sheet 'Sheet 1') before running "
            "education-scanner-baseline."
        )
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scanner metadata file not found at {path}. Set SCANNER_XLSX_PATH to override."
        )
    df = pd.read_excel(path, sheet_name=SCANNER_SHEET)
    if "resonador" not in df.columns or "N_MEGA" not in df.columns:
        raise ValueError(
            f"Expected columns 'N_MEGA' and 'resonador' (scanner id) in {path}; "
            f"found: {list(df.columns)[:20]}"
        )
    out = df[["N_MEGA", "resonador"]].copy()
    out["N_MEGA"] = out["N_MEGA"].astype(str).str.strip()
    out = out.dropna(subset=["N_MEGA"]).drop_duplicates("N_MEGA", keep="first")
    out = out.rename(columns={"resonador": "scanner_id"})
    out["scanner_id"] = out["scanner_id"].astype(str).str.strip()
    return out


def _selected_variants() -> dict[str, dict[str, bool]]:
    requested = os.environ.get("EDU_SCANNER_VARIANTS", "").strip()
    if not requested:
        return dict(COVARIATE_VARIANTS)
    labels = [v.strip() for v in requested.split(",") if v.strip()]
    bad = [v for v in labels if v not in COVARIATE_VARIANTS]
    if bad:
        raise ValueError(f"Unknown EDU_SCANNER_VARIANTS entry(ies) {bad}; choices: {list(COVARIATE_VARIANTS)}")
    return {label: COVARIATE_VARIANTS[label] for label in labels}


def _attach_education_and_scanner(full_model: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw_edu = raw[["N_MEGA", "Edu"]].copy()
    raw_edu["N_MEGA"] = raw_edu["N_MEGA"].astype(str).str.strip()
    raw_edu = raw_edu.drop_duplicates("N_MEGA", keep="first")

    out = full_model.merge(raw_edu, on="N_MEGA", how="left")
    out = out.merge(_load_scanner_table(), on="N_MEGA", how="left")
    return out


TOP_K = 20

RUNG_LABEL = {"ols": "OLS", "xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}

_VARIANT_TITLE: dict[str, str] = {
    "plus_education": "+ Education",
    "plus_scanner": "+ Scanner",
    "plus_education_scanner": "+ Education + scanner",
}


def _rung_topk(df_global: pd.DataFrame, rung: str, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same selection rule as plot_fig2_grid_v3.py's _rung_top50, but top-K
    (our pool is far smaller than the full Fig.2 greedy pool, so K=50 would
    just take everything)."""
    r = df_global[df_global["rung_id"] == rung]
    syn = filter_syn_pool(r[r["objective"] == "o_min"], "o_min", "score").nlargest(top_k, "delta_r2_vs_base")
    red = r[r["objective"] == "o_max"].nlargest(top_k, "delta_r2_vs_base")
    return syn, red


def _draw_split_boxes(ax, df_global: pd.DataFrame, rungs_present: list[str], top_k: int = TOP_K) -> list[float]:
    """Direct port of plot_fig2_grid_v3.py::_draw_split_boxes: per model level a
    jittered strip of the top-K candidates -- left=synergistic, right=redundant --
    with an IQR box, median bar and 1.5x-IQR whiskers over it, plus a solid black
    baseline line. Same visual contract (and geometry) as the main figure."""
    HALF = 0.32        # half-width of each objective strip
    BW = HALF * 0.5    # box half-width
    rng_jitter = np.random.default_rng(0)
    all_vals: list[float] = []
    for i, rung in enumerate(rungs_present):
        syn, red = _rung_topk(df_global, rung, top_k)
        for vals, color, x_center in [
            (syn["full_r2"].to_numpy(dtype=float), SYN_COLOR, i - HALF * 0.5),
            (red["full_r2"].to_numpy(dtype=float), RED_COLOR, i + HALF * 0.5),
        ]:
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            all_vals.extend(vals.tolist())

            # jitter strip
            jitter = rng_jitter.uniform(-BW * 0.9, BW * 0.9, size=len(vals))
            ax.scatter(x_center + jitter, vals, c=color, s=54, alpha=0.55,
                       edgecolors="none", zorder=3)

            # boxplot overlay
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            iqr = q75 - q25
            w_lo = max(vals.min(), q25 - 1.5 * iqr)
            w_hi = min(vals.max(), q75 + 1.5 * iqr)
            ax.add_patch(mpatches.Rectangle(
                (x_center - BW, q25), 2 * BW, iqr,
                facecolor=color, alpha=0.40, edgecolor=color, lw=1.2, zorder=4,
            ))
            ax.plot([x_center - BW, x_center + BW], [med, med], color=color, lw=2.0, zorder=5)
            ax.plot([x_center, x_center], [w_lo, q25], color=color, lw=1.0, zorder=4)
            ax.plot([x_center, x_center], [q75, w_hi], color=color, lw=1.0, zorder=4)

        base_rows = df_global.loc[df_global["rung_id"] == rung, "base_r2"]
        base_val = pd.to_numeric(base_rows, errors="coerce").dropna()
        if not base_val.empty and np.isfinite(base_val.iloc[0]):
            base_val = float(base_val.iloc[0])
            ax.plot([i - HALF, i + HALF], [base_val, base_val], color="black", lw=2.5, ls="-", zorder=6)
            all_vals.append(base_val)

    ax.set_xticks(range(len(rungs_present)))
    ax.set_xticklabels([LEVEL_LABELS.get(r, r) for r in rungs_present], fontsize=9)
    ax.set_xlim(-0.5, len(rungs_present) - 0.5)
    return all_vals


def _draw_best_single_lines(ax, single_summary: pd.DataFrame, rungs_present: list[str]) -> list[float]:
    """Draw the best education-adjusted single exposure as an orange line.

    A short line (rather than a star) retains the visual language used for the
    baseline and makes it readable over the split top-20 distributions.
    """
    values: list[float] = []
    for i, rung in enumerate(rungs_present):
        rows = single_summary[single_summary["rung_id"].astype(str) == rung].copy()
        if rows.empty:
            continue
        score = pd.to_numeric(rows["global_oof_r2"], errors="coerce")
        if not score.notna().any():
            continue
        value = float(score.max())
        ax.plot([i - 0.13, i + 0.13], [value, value], color=SINGLE_COLOR, lw=3.0, zorder=7)
        values.append(value)
    return values


def _entropy_scatter_frame(summary: pd.DataFrame, domain_map: dict[str, str], rung: str) -> pd.DataFrame:
    """Return the exact d3 greedy-model points used in the entropy scatter."""
    frame = summary[
        (summary["rung_id"].astype(str) == rung)
        & (summary["objective"].astype(str).isin(["o_min", "o_max"]))
    ].copy()
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        raw_predictors = str(getattr(row, "predictors_identity"))
        # These sensitivity CSVs serialise predictor sets as a pipe-delimited
        # identity string (not a Python literal).  Split it directly so the
        # entropy represents the domains actually used by every model.
        predictors = [item.strip() for item in raw_predictors.split("|") if item.strip()]
        stats = unweighted_domain_stats(predictors, domain_map=domain_map)
        rows.append(
            {
                "candidate_id": str(getattr(row, "candidate_id")),
                "objective": str(getattr(row, "objective")),
                "rung_id": rung,
                "order": int(getattr(row, "order")),
                "global_oof_r2": float(getattr(row, "global_oof_r2")),
                "shannon_h": float(stats["shannon_h"]),
                "n_domains": int(stats["n_domains"]),
                "predictors_identity": "|".join(predictors),
            }
        )
    return pd.DataFrame(rows)


def _plot_education_no_scanner(
    bag_inputs: dict[str, dict[str, pd.DataFrame]],
    rungs: list[str],
    outdir: Path,
    *,
    source_paths: list[str] | None = None,
    partial_set_size: bool = False,
    stem: str = "education_scanner_baseline_no_scanner",
) -> None:
    """Render the requested education-only two-column supplementary figure.

    The first column is the established Fig.2-style education comparison.  The
    second column shows every d3 greedy candidate against Shannon domain
    entropy; when requested, both variables are residualized by set size.
    """
    rungs_present = [r for r in RUNG_ORDER if r in set(rungs)]
    bags = [b for b in BAG_ROW_ORDER if b in bag_inputs]
    if not bags or not rungs_present:
        raise ValueError("Education-only figure requires both BAGs and at least one model level")
    domain_csv = CHECKOUT_ROOT / "data" / "metadata" / "exposome_feature_domains.csv"
    domain_map = pd.read_csv(domain_csv).set_index("feature_name")["domain"].astype(str).to_dict()
    deployed_rung = "xgb_tree_d3" if "xgb_tree_d3" in rungs_present else rungs_present[-1]

    fig, axes = plt.subplots(
        len(bags), 2, figsize=(10.2, 8.8), squeeze=False,
        gridspec_kw={"hspace": 0.34, "wspace": 0.32},
    )
    panels: list[Panel] = []
    for row_i, bag in enumerate(bags):
        education = bag_inputs[bag]["education"]
        singles = bag_inputs[bag].get("singles", pd.DataFrame())
        baseline = select_baseline_per_rung(
            education, baseline_label="baseline", score_column="global_oof_r2"
        ).set_index("rung_id")["global_oof_r2"]
        left = _variant_global_frame(education, baseline)
        ax_left, ax_right = axes[row_i]
        left_values = _draw_split_boxes(ax_left, left, rungs_present)
        if not singles.empty:
            left_values.extend(_draw_best_single_lines(ax_left, singles, rungs_present))
        if left_values:
            lo, hi = min(left_values), max(left_values)
            pad = (hi - lo) * 0.10 if hi > lo else 0.02
            ax_left.set_ylim(lo - pad, hi + pad)
        ax_left.set_title(
            f"{ROW_LETTERS[row_i]}. {BAG_LABELS[bag]} — + Education", fontsize=10, loc="left"
        )
        ax_left.set_ylabel("R² LOCO")
        style_axis(ax_left)

        top_left = []
        for rung in rungs_present:
            syn, red = _rung_topk(left[left["rung_id"] == rung], rung, TOP_K)
            top_left.extend([syn, red])
        left_source = pd.concat(top_left, ignore_index=True)
        left_source = left_source[["rung_id", "objective", "candidate_id", "full_r2", "base_r2", "delta_r2_vs_base"]]
        left_source = left_source.rename(columns={"candidate_id": "candidate_id"})
        if not singles.empty:
            best_single = (
                singles.sort_values("global_oof_r2", ascending=False).groupby("rung_id", as_index=False).head(1)
                [["rung_id", "candidate_id", "predictors_identity", "global_oof_r2"]]
                .rename(columns={"candidate_id": "best_single_candidate_id", "global_oof_r2": "best_single_r2"})
            )
            left_source = left_source.merge(best_single, on="rung_id", how="left")
        panels.append(Panel(
            panel_id=f"{ROW_LETTERS[row_i]}1_{BAG_SHORT[bag]}_edu",
            frame=left_source,
            description=f"{BAG_LABELS[bag]} education-adjusted top-{TOP_K} models." if singles.empty else f"{BAG_LABELS[bag]} education-adjusted top-{TOP_K} models and best single exposure.",
            columns={"rung_id": "model level", "objective": "o_min = Min O-info; o_max = Max O-info",
                     "full_r2": "country-balanced LOCO R² for plotted top-20 point",
                     "base_r2": "education-adjusted covariate baseline R² (black line)"},
            notes="Values are copied from the delivered + Education panel of the original figure.",
        ))

        scatter = _entropy_scatter_frame(education, domain_map, deployed_rung)
        x_column, y_column = "shannon_h", "global_oof_r2"
        if partial_set_size:
            scatter["entropy_residual"] = np.nan
            scatter["r2_residual"] = np.nan
            for objective, points in scatter.groupby("objective", observed=True):
                valid = points[["shannon_h", "global_oof_r2", "order"]].apply(
                    pd.to_numeric, errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < 4 or valid["order"].nunique() < 2:
                    continue
                design = np.column_stack([np.ones(len(valid)), valid["order"].to_numpy(dtype=float)])
                entropy = valid["shannon_h"].to_numpy(dtype=float)
                r2 = valid["global_oof_r2"].to_numpy(dtype=float)
                scatter.loc[valid.index, "entropy_residual"] = entropy - design @ np.linalg.lstsq(design, entropy, rcond=None)[0]
                scatter.loc[valid.index, "r2_residual"] = r2 - design @ np.linalg.lstsq(design, r2, rcond=None)[0]
            x_column, y_column = "entropy_residual", "r2_residual"
        for objective, color in (("o_min", SYN_COLOR), ("o_max", RED_COLOR)):
            points = scatter[scatter["objective"] == objective].dropna(subset=[x_column, y_column])
            ax_right.scatter(points[x_column], points[y_column], s=20, color=color,
                             alpha=0.20, linewidths=0, zorder=2)
            if len(points) > 2 and points[x_column].nunique() > 1:
                fit = scipy_stats.linregress(points[x_column], points[y_column])
                xs = np.linspace(points[x_column].min(), points[x_column].max(), 100)
                ax_right.plot(xs, fit.intercept + fit.slope * xs, color=color, lw=2.2, zorder=3)
        if partial_set_size:
            ax_right.set_title(f"Partial R² LOCO vs entropy ({LEVEL_LABELS[deployed_rung]})", fontsize=10)
            ax_right.set_xlabel("Entropy residual (adjusted for set size)")
            ax_right.set_ylabel("R² LOCO residual\n(adjusted for set size)")
        else:
            ax_right.set_title(f"R² LOCO vs domain entropy ({LEVEL_LABELS[deployed_rung]})", fontsize=10)
            ax_right.set_xlabel("Shannon domain entropy H (bits)")
            ax_right.set_ylabel("R² LOCO")
        style_axis(ax_right)
        panels.append(Panel(
            panel_id=f"{ROW_LETTERS[row_i]}2_{BAG_SHORT[bag]}_entropy",
            frame=scatter,
            description=(
                f"{BAG_LABELS[bag]} education-adjusted {LEVEL_LABELS[deployed_rung]} candidates: "
                "partial LOCO R²–entropy scatter after residualizing both variables by set size."
                if partial_set_size else
                f"{BAG_LABELS[bag]} education-adjusted {LEVEL_LABELS[deployed_rung]} candidates by domain entropy."
            ),
            columns={"shannon_h": "ordinary Shannon entropy of predictor domains (bits)",
                     "global_oof_r2": "LOCO R²", "objective": "Min O-info or Max O-info",
                     "order": "set size (number of exposures)",
                     "n_domains": "number of distinct predictor domains",
                     **({"entropy_residual": "entropy residual after linear adjustment for set size -- plotted x axis",
                         "r2_residual": "LOCO R² residual after linear adjustment for set size -- plotted y axis"} if partial_set_size else {})},
            notes=("Both LOCO R² and entropy are residualized linearly on set size within each BAG and arm. "
                   "No single-exposure point is plotted in this diversity panel." if partial_set_size else
                   "No single-exposure point is plotted in this diversity panel."),
        ))

    legend = [
        mpatches.Patch(color=SYN_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_min']}"),
        mpatches.Patch(color=RED_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_max']}"),
        mlines.Line2D([0], [0], color="black", lw=2.5, label="Baseline + Education"),
    ]
    if any(not bag_inputs[bag].get("singles", pd.DataFrame()).empty for bag in bags):
        legend.append(mlines.Line2D([0], [0], color=SINGLE_COLOR, lw=3.0, label="Best single + Education"))
    axes[0][0].legend(handles=legend, fontsize=7, loc="best")
    save_figure(fig, stem, outdir)
    plt.close(fig)
    write_source_data(stem, panels, outdir, source_paths=source_paths)


def _variant_global_frame(variant_summary: pd.DataFrame, baseline_by_rung: pd.Series) -> pd.DataFrame:
    """Build the df_global-equivalent frame _draw_split_boxes expects:
    full_r2 (this variant), base_r2 (baseline arm, same rung), delta_r2_vs_base."""
    df = variant_summary[variant_summary["objective"].isin(["o_min", "o_max"])].copy()
    df["full_r2"] = pd.to_numeric(df["global_oof_r2"], errors="coerce")
    df["base_r2"] = df["rung_id"].map(baseline_by_rung)
    df["delta_r2_vs_base"] = df["full_r2"] - df["base_r2"]
    return df


def _plot_comparison(
    bag_inputs: dict[str, dict],
    rungs: list[str],
    outdir: Path,
    *,
    r2_estimand: str = "global out-of-fold LOCO R²",
    write_rendered_source_data: bool = True,
) -> None:
    """One figure with a row per BAG, structural first. No suptitle.

    Within a row there is one Fig2-style split-box panel per covariate add-on,
    all sharing the same baseline reference line (the baseline arm, not a
    further variant). Rows share a y-axis within a BAG so the add-ons are
    directly comparable; the two BAGs keep independent scales because their R²
    ranges differ.
    """
    rungs_present = [r for r in RUNG_ORDER if r in set(rungs)]
    bags = [b for b in BAG_ROW_ORDER if b in bag_inputs]
    if not bags:
        return
    variants_present = [
        v for v in COVARIATE_VARIANTS if v in bag_inputs[bags[0]]["dist_by_variant"]
    ]
    if not variants_present:
        return

    fig, axes = plt.subplots(
        len(bags), len(variants_present),
        figsize=(4.6 * len(variants_present), 4.4 * len(bags)),
        squeeze=False, gridspec_kw={"hspace": 0.34, "wspace": 0.18},
    )

    panels: list[Panel] = []
    for row_i, bag in enumerate(bags):
        d = bag_inputs[bag]
        baseline_rows = select_baseline_per_rung(
            d["baseline_summary"], baseline_label="baseline", score_column="global_oof_r2"
        )
        baseline_by_rung = baseline_rows.set_index("rung_id")["global_oof_r2"]

        row_vals: list[float] = []
        for col_i, variant in enumerate(variants_present):
            ax = axes[row_i][col_i]
            df_global = _variant_global_frame(d["dist_by_variant"][variant], baseline_by_rung)
            row_vals.extend(_draw_split_boxes(ax, df_global, rungs_present))
            title = _VARIANT_TITLE[variant]
            if col_i == 0:
                title = f"{ROW_LETTERS[row_i]}. {BAG_LABELS[bag]} — {title}"
            ax.set_title(title, fontsize=10, loc="left")
            style_axis(ax)

            frame = df_global[
                [c for c in ["rung_id", "objective", "candidate_id", "full_r2", "base_r2",
                             "delta_r2_vs_base"] if c in df_global.columns]
            ].copy()
            frame = frame.rename(columns={"rung_id": "model_level"})
            frame["model_level"] = frame["model_level"].map(LEVEL_LABELS)
            panels.append(
                Panel(
                    panel_id=f"{ROW_LETTERS[row_i]}{col_i + 1}_{BAG_SHORT[bag]}_{_VARIANT_SHORT[variant]}",
                    frame=frame.sort_values(["objective", "model_level"]).reset_index(drop=True),
                    description=(
                        f"{BAG_LABELS[bag]}: held-out LOCO R² of the top-{TOP_K} synergistic and "
                        f"top-{TOP_K} redundant candidates per model level with "
                        f"{_VARIANT_TITLE[variant].lower()}."
                    ),
                    columns={
                        "model_level": "model level (OLS, d1, d2, d3)",
                        "objective": "o_min = synergistic arm, o_max = redundant arm",
                        "candidate_id": "candidate identifier",
                        "full_r2": f"{r2_estimand} for that candidate (plotted point)",
                        "base_r2": "covariate baseline R² for that level (black line)",
                        "delta_r2_vs_base": "full_r2 minus base_r2",
                    },
                    notes=(
                        "Complete-case sample carrying both education and scanner identity. "
                        f"R² estimand: {r2_estimand}. Box is the IQR, centre bar the median, "
                        "whiskers 1.5x IQR."
                    ),
                )
            )

        if row_vals:
            lo, hi = min(row_vals), max(row_vals)
            pad = (hi - lo) * 0.10 if hi > lo else 0.02
            for col_i in range(len(variants_present)):
                axes[row_i][col_i].set_ylim(lo - pad, hi + pad)
                if col_i:
                    axes[row_i][col_i].tick_params(labelleft=False)
        axes[row_i][0].set_ylabel("R² LOCO")

    patches = [
        mpatches.Patch(color=SYN_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_min']}"),
        mpatches.Patch(color=RED_COLOR, alpha=0.55, label=f"Top-{TOP_K} {OBJECTIVE_ARM_LABELS['o_max']}"),
        mlines.Line2D([0], [0], color="black", lw=2.5, ls="-", label="Baseline covariates"),
    ]
    axes[0][-1].legend(handles=patches, fontsize=7, loc="best")
    stem = "education_scanner_baseline"
    save_figure(fig, stem, outdir)
    plt.close(fig)
    if write_rendered_source_data:
        write_source_data(stem, panels, outdir)


def _rebuild_from_cached_summaries(
    *,
    eval_root: Path,
    local_root: Path,
    fig_dir: Path,
    bags: list[str],
    rungs: list[str],
    variants: dict[str, dict[str, bool]],
) -> None:
    """Rebuild lightweight combined outputs after per-BAG cluster jobs.

    The model summaries are already complete in the run-scoped external cache;
    this path performs no fitting and prevents concurrently launched BAG jobs
    from leaving a last-writer-wins combined figure or CSV.
    """
    labels = [BASELINE_LABEL, *variants]
    all_rows: list[pd.DataFrame] = []
    bag_inputs: dict[str, dict] = {}
    for bag in bags:
        dist_by_covset: dict[str, pd.DataFrame] = {}
        bag_rows: list[pd.DataFrame] = []
        for label in labels:
            path = eval_root / bag / label / f"{bag}_global_all_rungs.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cached education/scanner summary: {path}")
            summary = pd.read_csv(path)
            summary["bag"] = bag
            summary["covariate_set"] = label
            dist_by_covset[label] = summary
            summary_syn = pd.concat(
                [
                    filter_syn_pool(summary[summary["objective"] == "o_min"], "o_min", "score"),
                    summary[summary["objective"] != "o_min"],
                ],
                ignore_index=True,
            )
            best_syn = select_best_per_rung(summary_syn, objective="o_min", score_column="global_oof_r2")
            best_red = select_best_per_rung(summary, objective="o_max", score_column="global_oof_r2")
            baseline = select_baseline_per_rung(summary, baseline_label="baseline", score_column="global_oof_r2")
            best_syn["candidate_role"] = "best_synergy"
            best_red["candidate_role"] = "best_redundancy"
            baseline["candidate_role"] = "baseline"
            bag_rows.append(pd.concat([best_syn, best_red, baseline], ignore_index=True))
        bag_combined = pd.concat(bag_rows, ignore_index=True)
        bag_combined.to_csv(local_root / f"{bag}_global_all_rungs.csv", index=False)
        all_rows.append(bag_combined)
        bag_inputs[bag] = {
            "baseline_summary": dist_by_covset[BASELINE_LABEL],
            "dist_by_variant": {key: value for key, value in dist_by_covset.items() if key != BASELINE_LABEL},
        }
    _plot_comparison(bag_inputs, rungs, fig_dir)
    pd.concat(all_rows, ignore_index=True).to_csv(local_root / "global_all_rungs.csv", index=False)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = _env_bool("SMOKE_TEST") or _env_bool("SENSITIVITY_SMOKE")
    include_combined = _env_bool("SENSITIVITY_INCLUDE_COMBINED", default=False)
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = int(os.environ.get("SENSITIVITY_MAX_CANDIDATES", "24")) if smoke else None

    base_cfg = analysis_cfg_from_config(cfg)
    variants = _selected_variants()
    # Baseline always runs first -- it is the reference line every variant
    # panel is compared against, not a fourth variant.
    run_cfgs = {BASELINE_LABEL: dict(base_cfg)}
    run_cfgs.update({label: {**base_cfg, **flags} for label, flags in variants.items()})

    local_root = repo_sensitivity_root(cfg) / "education_scanner_baseline"
    local_root.mkdir(parents=True, exist_ok=True)
    # Heavy per-fold LOCO eval artifacts stay on the cluster; only summary
    # tables + figures go to local_root (the checkout).
    eval_root = sensitivity_eval_work_root(cfg, "education_scanner_baseline")

    fig_dir = repo_sensitivity_figures_root(cfg, "education_scanner_baseline")
    fig_dir.mkdir(parents=True, exist_ok=True)
    bags = selected_bags(cfg, include_combined=include_combined)
    if _env_bool("EDU_SCANNER_REBUILD_FROM_CACHE"):
        _rebuild_from_cached_summaries(
            eval_root=eval_root,
            local_root=local_root,
            fig_dir=fig_dir,
            bags=bags,
            rungs=rungs,
            variants=variants,
        )
        return

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = build_original_model_df(raw, feature_names, base_cfg)
    full_model = _attach_education_and_scanner(full_model, raw)

    n_total = len(full_model)
    n_no_edu = int(full_model["Edu"].isna().sum())
    n_no_scanner = int(full_model["scanner_id"].isna().sum())
    complete_case = full_model[full_model["Edu"].notna() & full_model["scanner_id"].notna()].copy()
    n_complete = len(complete_case)
    join_report = pd.DataFrame([{
        "n_total_rows": n_total,
        "n_missing_education": n_no_edu,
        "pct_missing_education": float(n_no_edu / max(1, n_total)),
        "n_missing_scanner_after_join": n_no_scanner,
        "pct_missing_scanner_after_join": float(n_no_scanner / max(1, n_total)),
        "n_complete_case_education_and_scanner": n_complete,
        "pct_complete_case_education_and_scanner": float(n_complete / max(1, n_total)),
    }])
    join_report.to_csv(local_root / "education_scanner_join_summary.csv", index=False)
    print(
        f"=== education-scanner-baseline join report: n_total={n_total} "
        f"missing_education={n_no_edu} ({n_no_edu / max(1, n_total):.2%}) "
        f"missing_scanner_after_join={n_no_scanner} ({n_no_scanner / max(1, n_total):.2%}) "
        f"complete_case_n={n_complete} ({n_complete / max(1, n_total):.2%}) ==="
    )

    all_rows = []
    # Both BAGs share one figure folder and one merged figure.
    bag_inputs: dict[str, dict] = {}

    for bag in bags:
        print(f"\n=== education-scanner-baseline sensitivity | bag={bag} | smoke={smoke} | n={n_complete} ===")
        bag_dir = local_root / bag
        bag_dir.mkdir(parents=True, exist_ok=True)
        eval_bag_dir = eval_root / bag
        eval_bag_dir.mkdir(parents=True, exist_ok=True)

        pool = load_fig2_candidate_pool(bag)
        if max_candidates is not None:
            # Stratify per objective: pool is sorted by candidate_id, NOT
            # interleaved by objective, so a plain head() can silently drop
            # o_min or o_max entirely and break select_best_per_rung downstream.
            pool = pool.groupby("objective", group_keys=False).head(max_candidates).copy()
        pool = pd.concat([pool, baseline_candidate_df()], ignore_index=True)

        bag_rows = []
        dist_by_covset: dict[str, pd.DataFrame] = {}
        for label, run_cfg in run_cfgs.items():
            eval_dir = eval_bag_dir / label
            summary, country = evaluate_candidates_by_rung(
                model_df=complete_case,
                candidate_df=pool,
                exposome_cols=feature_names,
                bag=bag,
                rungs=rungs,
                analysis_cfg=run_cfg,
                outdir=eval_dir,
                n_jobs=n_jobs,
                max_candidates=None,
            )
            summary["bag"] = bag
            summary["covariate_set"] = label
            country["bag"] = bag
            country["covariate_set"] = label
            dist_by_covset[label] = summary

            # Optional negative-O-info synergy criterion: restrict o_min to
            # evaluated O-info (score) < 0 before the best-per-rung pick. No-op off.
            summary_syn = pd.concat(
                [
                    filter_syn_pool(summary[summary["objective"] == "o_min"], "o_min", "score"),
                    summary[summary["objective"] != "o_min"],
                ],
                ignore_index=True,
            )
            best_syn = select_best_per_rung(summary_syn, objective="o_min", score_column="global_oof_r2")
            best_red = select_best_per_rung(summary, objective="o_max", score_column="global_oof_r2")
            baseline_row = select_baseline_per_rung(summary, baseline_label="baseline", score_column="global_oof_r2")
            best_syn["candidate_role"] = "best_synergy"
            best_red["candidate_role"] = "best_redundancy"
            baseline_row["candidate_role"] = "baseline"
            bag_rows.append(pd.concat([best_syn, best_red, baseline_row], ignore_index=True))

        bag_combined = pd.concat(bag_rows, ignore_index=True)
        bag_inputs[bag] = {
            "baseline_summary": dist_by_covset[BASELINE_LABEL],
            "dist_by_variant": {k: v for k, v in dist_by_covset.items() if k != BASELINE_LABEL},
        }
        all_rows.append(bag_combined)

    # One figure for both BAGs (structural row first), drawn after the loop.
    _plot_comparison(bag_inputs, rungs, fig_dir)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(local_root / "global_all_rungs.csv", index=False)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "education_scanner_baseline")
    copy_tree_contents(
        repo_sensitivity_figures_root(cfg, "education_scanner_baseline"),
        bundle_sensitivity_root(cfg) / "education_scanner_baseline",
    )


if __name__ == "__main__":
    main()
