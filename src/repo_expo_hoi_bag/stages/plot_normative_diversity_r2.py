#!/usr/bin/env python3
"""Normative diversity vs LOCO R2 figure.

Builds a 2 x 8 grid using the same pooled/normative data and conditions as
``plot_normative_transfer_grid.py``. Only lightweight plotting tables are
written locally; heavy evaluation outputs remain in the external runtime.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib
import os

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.exposome_domains import load_domain_map, unweighted_domain_stats
from scripts.plot_normative_transfer_grid import (
    BAGS,
    CONDITION_ORDER,
    GRID_FS,
    GRID_FS_TK,
    ORDER_MAX,
    PAPER_FIGURES_DIR,
    RED_COLOR,
    RUNG_ORDER,
    RUNG_DISPLAY,
    SYN_COLOR,
    load_plot_data,
)

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data


LOCAL_STATS_DIR = Path(
    os.environ.get(
        "NORM_DIVERSITY_STATS_DIR",
        str(REPO_ROOT / "outputs" / "variant_a" / "stats" / "normative_diversity_r2"),
    )
)
FIG_SUFFIX = os.environ.get("NORM_FIG_SUFFIX", "").strip()
FIG_SUFFIX_PART = f"_{FIG_SUFFIX}" if FIG_SUFFIX else ""
FIG_STEM = os.environ.get("NORM_DIVERSITY_FIG_STEM", "").strip() or "fig_normative_diversity_r2"
DOMAIN_LABELS_CSV = Path(
    os.environ.get(
        "NORM_DOMAIN_LABELS_CSV",
        str(Path(__file__).resolve().parents[3] / "data" / "metadata" / "exposome_feature_domains.csv"),
    )
)
DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)

OBJ_COLOR = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
OBJ_DISPLAY = {"o_min": "Min O-info", "o_max": "Max O-info"}
BAG_LABEL = {"functional": "Functional", "structural": "Structural"}
BAG_ROW_ORDER = ["structural", "functional"]
CONDITION_DISPLAY = {
    "Pooled": "Pooled",
    "CN->CN": "HC→HC",
    "AD->AD": "AD→AD",
    "AD->FTD": "AD→FTLD",
    "FTD->FTD": "FTLD→FTLD",
    "FTD->AD": "FTLD→AD",
    "AD+FTD->AD": "AD+FTLD→AD",
    "AD+FTD->FTD": "AD+FTLD→FTLD",
}
DIVERSITY_RUNG = os.environ.get("NORM_DIVERSITY_RUNG", "").strip()
# Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): an o_min
# candidate counts as synergistic only if its evaluated O-info (oinfo) < 0.
_SYN_OINFO_NEGATIVE = os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)


def _fmt_p(p: float | None) -> str:
    """Exact p, never a floor bound.

    This is the two-sided p of an OLS coefficient, so it has no resolution floor
    that a "<" bound could legitimately express: the former "p<0.001" discarded a
    value the test actually resolved. Very small values render in scientific
    notation rather than as 0.000.
    """
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 0.001:
        return f"p={p:.2e}"
    return f"p={p:.3f}"


def _domain_stats(predictors_identity: str) -> dict[str, object]:
    vars_list = [v.strip() for v in str(predictors_identity).split("|") if v.strip()]
    stats = unweighted_domain_stats(vars_list, domain_map=DOMAIN_MAP)
    return {
        "shannon_h": stats["shannon_h"],
        "n_domains": stats["n_domains"],
        "dominant_domain": stats["dominant_domain"],
    }


def add_domain_diversity(df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame([_domain_stats(p) for p in df["predictors_identity"]])
    out = pd.concat([df.reset_index(drop=True), stats], axis=1)
    out["is_synergistic"] = out["objective"].eq("o_min").astype(int)
    return out


def select_best_rung_by_objective(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rung = os.environ.get("NORM_POOLED_RUNG", "").strip()
    rows = []
    selected = []
    for bag in BAGS:
        bag_df = df[df["bag"].eq(bag)]
        for condition in CONDITION_ORDER:
            cond_df = bag_df[bag_df["condition"].astype(str).eq(condition)]
            for objective in ["o_min", "o_max"]:
                obj_df = cond_df[cond_df["objective"].eq(objective)].copy()
                if obj_df.empty:
                    continue
                by_rung = (
                    obj_df.groupby("rung_id", observed=True)["r2"]
                    .max()
                    .reindex(RUNG_ORDER)
                    .dropna()
                )
                if by_rung.empty:
                    continue
                best_rung = DIVERSITY_RUNG or (
                    pooled_rung
                    if condition == "Pooled" and pooled_rung
                    else str(by_rung.idxmax())
                )
                if best_rung not in by_rung.index.astype(str):
                    raise ValueError(
                        f"Configured pooled rung {best_rung!r} is unavailable for {bag}/{objective}"
                    )
                best_r2 = float(by_rung.loc[best_rung])
                keep = obj_df[obj_df["rung_id"].astype(str).eq(best_rung)].copy()
                rows.append(
                    {
                        "bag": bag,
                        "condition": condition,
                        "objective": objective,
                        "best_rung": best_rung,
                        "best_rung_label": RUNG_DISPLAY.get(best_rung, best_rung),
                        "best_r2": best_r2,
                        "n_models": int(len(keep)),
                    }
                )
                selected.append(keep)
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    best_df = pd.DataFrame(rows)
    return selected_df, best_df


def fit_panel_model(panel_df: pd.DataFrame) -> dict[str, object]:
    fit_df = panel_df[["r2", "shannon_h", "is_synergistic", "order"]].copy()
    for col in ["r2", "shannon_h", "is_synergistic", "order"]:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce").astype(float)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, object] = {
        "n_obs": int(len(fit_df)),
        "beta_h_red": np.nan,
        "beta_h_syn": np.nan,
        "p_h_red": np.nan,
        "p_h_syn_interaction": np.nan,
        "model_r2": np.nan,
    }
    if len(fit_df) < 12 or fit_df["shannon_h"].nunique() < 2 or fit_df["is_synergistic"].nunique() < 2:
        return result
    try:
        model = smf.ols("r2 ~ shannon_h * is_synergistic + order", data=fit_df).fit()
    except Exception:
        return result
    beta_h = float(model.params.get("shannon_h", np.nan))
    beta_int = float(model.params.get("shannon_h:is_synergistic", np.nan))
    result.update(
        {
            "beta_h_red": beta_h,
            "beta_h_syn": beta_h + beta_int,
            "p_h_red": float(model.pvalues.get("shannon_h", np.nan)),
            "p_h_syn_interaction": float(model.pvalues.get("shannon_h:is_synergistic", np.nan)),
            "model_r2": float(model.rsquared),
        }
    )
    return result


def build_panel_models(scatter_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bag, condition), panel_df in scatter_df.groupby(["bag", "condition"], observed=True):
        stats = fit_panel_model(panel_df)
        rows.append({"bag": bag, "condition": condition, **stats})
    return pd.DataFrame(rows)


def build_panel_correlations(scatter_df: pd.DataFrame) -> pd.DataFrame:
    """Return the arm-specific bivariate statistics drawn in every panel."""
    rows = []
    for (bag, condition, objective), group in scatter_df.groupby(
        ["bag", "condition", "objective"], observed=True
    ):
        fit = group[["shannon_h", "r2"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(fit) < 10 or fit["shannon_h"].std() <= 0:
            continue
        result = scipy_stats.linregress(fit["shannon_h"], fit["r2"])
        rows.append(
            {
                "bag": bag,
                "condition": condition,
                "objective": objective,
                "arm_label": OBJ_DISPLAY[objective],
                "n_candidates": int(len(fit)),
                "pearson_r": float(result.rvalue),
                "p_value": float(result.pvalue),
                "slope_beta_h": float(result.slope),
                "intercept": float(result.intercept),
                "stderr": float(result.stderr),
            }
        )
    return pd.DataFrame(rows)


def draw_panel(
    ax: plt.Axes,
    scatter_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    bag: str,
    condition: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    show_title: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    sub = scatter_df[scatter_df["bag"].eq(bag) & scatter_df["condition"].astype(str).eq(condition)].copy()
    lr_by_obj = {}
    for objective in ["o_max", "o_min"]:
        grp = sub[sub["objective"].eq(objective)].dropna(subset=["shannon_h", "r2"])
        if grp.empty:
            continue
        ax.scatter(
            grp["shannon_h"],
            grp["r2"],
            c=OBJ_COLOR[objective],
            alpha=0.20,
            s=22,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        if len(grp) >= 10 and grp["shannon_h"].std() > 0:
            lr = scipy_stats.linregress(grp["shannon_h"], grp["r2"])
            lr_by_obj[objective] = lr
            xs = np.linspace(float(grp["shannon_h"].min()), float(grp["shannon_h"].max()), 100)
            x = grp["shannon_h"].to_numpy(dtype=float)
            y = grp["r2"].to_numpy(dtype=float)
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
                    color=OBJ_COLOR[objective],
                    alpha=0.14,
                    linewidth=0,
                    zorder=2.5,
                )
            ax.plot(xs, y_hat, color=OBJ_COLOR[objective], linewidth=2.3, alpha=0.95, zorder=3)

    for line_index, objective in enumerate(("o_min", "o_max")):
        row = correlation_df[
            correlation_df["bag"].eq(bag)
            & correlation_df["condition"].astype(str).eq(condition)
            & correlation_df["objective"].eq(objective)
        ]
        stats = "r=n/a, p=n/a"
        if not row.empty:
            stats = (
                f"r={float(row['pearson_r'].iloc[0]):.2f}, "
                f"{_fmt_p(float(row['p_value'].iloc[0]))}"
            )
        ax.text(
            0.03,
            0.090 - 0.055 * line_index,
            f"{OBJ_DISPLAY[objective]} ({stats})",
            transform=ax.transAxes,
            color=OBJ_COLOR[objective],
            fontsize=max(GRID_FS_TK - 3, 5),
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.72),
            zorder=5,
        )

    if show_title:
        ax.set_title(CONDITION_DISPLAY.get(condition, condition), fontsize=GRID_FS, pad=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show_xlabel:
        ax.set_xlabel("")
        ax.set_xticks([0, 2])
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(f"{BAG_LABEL[bag]} BAG\nLOCO R²", fontsize=GRID_FS)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=GRID_FS_TK)
    ax.spines[["top", "right"]].set_visible(False)


def _shared_limits(values: pd.Series, pad_frac: float = 0.05) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    if np.isclose(lo, hi):
        pad = max(abs(hi) * pad_frac, 0.01)
    else:
        pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def save_tables(scatter_df: pd.DataFrame, best_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    stats_dir = LOCAL_STATS_DIR / FIG_SUFFIX if FIG_SUFFIX else LOCAL_STATS_DIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    compact_cols = [
        "bag",
        "condition",
        "objective",
        "rung_id",
        "candidate_id",
        "order",
        "r2",
        "shannon_h",
        "n_domains",
        "dominant_domain",
        "source",
    ]
    scatter_df[compact_cols].to_csv(stats_dir / "normative_diversity_r2_scatter.csv", index=False)
    best_df.to_csv(stats_dir / "normative_diversity_r2_best_rungs.csv", index=False)
    model_df.to_csv(stats_dir / "normative_diversity_r2_panel_models.csv", index=False)


_BAG_SHORT = {"structural": "struct", "functional": "func"}
_ROW_LETTERS = ("a", "b")
_OBJ_GLOSS = ("o_min = synergistic (min O-information) arm; "
              "o_max = redundant (max O-information) arm")


def _build_diversity_panels(scatter_df: pd.DataFrame, correlation_df: pd.DataFrame,
                            best_df: pd.DataFrame) -> list[Panel]:
    """One scatter panel and one arm-specific correlation panel per BAG row.

    Each grid cell is a (BAG, condition) pair, so the condition column indexes
    the columns of the row rather than a separate sheet.
    """
    panels: list[Panel] = []
    for letter, bag in zip(_ROW_LETTERS, BAG_ROW_ORDER):
        short = _BAG_SHORT.get(bag, bag)
        sub = scatter_df[scatter_df["bag"].eq(bag)]
        panels.append(Panel(
            panel_id=f"{letter}1_{short}_diversity_scatter",
            frame=sub[["condition", "objective", "rung_id", "candidate_id", "order",
                       "r2", "shannon_h", "n_domains", "dominant_domain"]].reset_index(drop=True),
            description=(f"{BAG_LABEL[bag]} BAG: held-out LOCO R² against Shannon domain entropy H, "
                         "one point per candidate, in each transfer condition."),
            columns={
                "condition": "transfer condition: the normative model's training group -> the evaluated group",
                "objective": _OBJ_GLOSS,
                "rung_id": "model level the candidates were drawn from",
                "candidate_id": "identifier of the exposure set",
                "order": "set size (number of exposures)",
                "r2": "held-out LOCO R² -- the y axis",
                "shannon_h": "Shannon domain entropy H (bits) -- the x axis",
                "n_domains": "unique exposome domains in the set",
                "dominant_domain": "most frequent domain in the set",
            },
            notes=("Only candidates of the configured common model level are plotted; all finite "
                   "country-balanced R² values, including negative values, enter the fit. "
                   "The selected model level per cell is in the companion stats sheet."),
        ))
        panels.append(Panel(
            panel_id=f"{letter}2_{short}_diversity_stats",
            frame=correlation_df[correlation_df["bag"].eq(bag)].reset_index(drop=True),
            description=(f"{BAG_LABEL[bag]} BAG: arm-specific Pearson correlations and "
                         "least-squares lines reported in each panel."),
            columns={
                "condition": "transfer condition",
                "objective": _OBJ_GLOSS,
                "arm_label": "arm label printed in the panel legend",
                "n_candidates": "candidates entering the correlation and line fit",
                "pearson_r": "Pearson correlation coefficient printed in the legend",
                "p_value": "exact two-sided p-value for the Pearson correlation",
                "slope_beta_h": "least-squares slope on Shannon domain entropy H (R² per bit)",
                "intercept": "least-squares intercept",
                "stderr": "standard error of the slope",
            },
            test="Two-sided Pearson correlation test (scipy.stats.linregress).",
            notes=("The line and its 95% confidence band use the same arm-specific bivariate fit. "
                   "Adjusted diversity-by-arm models controlling for set size are reported in "
                   "Supplementary Table 15 rather than annotated on the figure."),
        ))
    panels.append(Panel(
        panel_id="model_level_selection",
        frame=best_df.reset_index(drop=True),
        description="The model level selected for each BAG, condition and arm, whose candidates the scatter shows.",
        columns={
            "bag": "brain-age-gap modality", "condition": "transfer condition",
            "objective": _OBJ_GLOSS, "best_rung": "selected model level",
            "best_rung_label": "that level as printed in the panel annotation",
            "best_r2": "highest held-out LOCO R² at that level",
            "n_models": "candidates carried into the scatter from that level",
        },
        notes=("The main-k10 contextual comparison fixes the deployed d3 level in every cell so "
               "pooled and diagnosis-specific associations compare the same model class."),
    ))
    return panels


def main() -> None:
    frames = []
    for bag in BAGS:
        frames.append(load_plot_data(bag))
    df = pd.concat(frames, ignore_index=True)
    df = add_domain_diversity(df)
    # Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): drop
    # o_min candidates whose evaluated O-info (oinfo) is not < 0 before best-rung
    # selection and the r2~shannon_h*is_synergistic+order beta fit. o_max kept.
    if _SYN_OINFO_NEGATIVE:
        if "oinfo" not in df.columns:
            raise KeyError("SYN_OINFO_NEGATIVE set but 'oinfo' absent from diversity pool.")
        o = pd.to_numeric(df["oinfo"], errors="coerce")
        df = df[(df["objective"] != "o_min") | (o < 0)].copy()
    scatter_all_df, best_df = select_best_rung_by_objective(df)
    r2 = pd.to_numeric(scatter_all_df["r2"], errors="coerce")
    scatter_df = scatter_all_df[np.isfinite(r2)].copy()
    model_df = build_panel_models(scatter_df)
    correlation_df = build_panel_correlations(scatter_df)
    save_tables(scatter_df, best_df, model_df)

    xlim = _shared_limits(scatter_df["shannon_h"], pad_frac=0.04)
    ylim_by_bag = {
        bag: _shared_limits(
            scatter_df[scatter_df["bag"].eq(bag)]["r2"], pad_frac=0.05
        )
        for bag in BAG_ROW_ORDER
    }

    fig, axes = plt.subplots(2, len(CONDITION_ORDER), figsize=(18, 6.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.02, wspace=0.005, hspace=0.02)
    for row, bag in enumerate(BAG_ROW_ORDER):
        for col, condition in enumerate(CONDITION_ORDER):
            draw_panel(
                axes[row, col],
                scatter_df,
                correlation_df,
                bag,
                condition,
                show_ylabel=(col == 0),
                show_xlabel=(row == len(BAGS) - 1),
                show_title=(row == 0),
                xlim=xlim,
                ylim=ylim_by_bag[bag],
            )
    fig.supxlabel("Shannon domain entropy H (bits)", fontsize=GRID_FS)

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = PAPER_FIGURES_DIR / f"{FIG_STEM}{FIG_SUFFIX_PART}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    for path in write_source_data(
        f"{FIG_STEM}{FIG_SUFFIX_PART}",
        _build_diversity_panels(scatter_df, correlation_df, best_df),
        PAPER_FIGURES_DIR,
        source_paths=[str(LOCAL_STATS_DIR), str(DOMAIN_LABELS_CSV)],
    ):
        print(f"Saved: {path}")

    audit = best_df.groupby(["bag", "condition"], observed=True)["objective"].nunique().reset_index(name="n_objectives")
    missing = audit[audit["n_objectives"] < 2]
    if not missing.empty:
        print("WARNING: panels missing syn/red best-rung selection:")
        print(missing.to_string(index=False))
    stats_dir = LOCAL_STATS_DIR / FIG_SUFFIX if FIG_SUFFIX else LOCAL_STATS_DIR
    print(f"Saved tables to {stats_dir}")
    print(f"Shared xlim: {xlim}")
    print(f"Shared ylims by row: {ylim_by_bag}")
    print("R2 plotting/model filter: all finite country-balanced R2 values")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
