#!/usr/bin/env python3
"""Render the no-SHAP, order-capped Figure 3 from the new k10 HPO results.

The immutable candidate registry supplies only candidate identity, set size and
O-information arm.  Every plotted performance value is recalculated from the
new paper-reanalysis LOCO country metrics, reduced as an unweighted mean over
held-out countries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("warning")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from repo_expo_hoi_bag.config.models import load_historical_paper_reference
from repo_expo_hoi_bag.figures.source_data import write_source_data

ROOT = Path(__file__).resolve().parents[3]
BAGS = ("structural", "functional")
OBJECTIVES = ("o_min", "o_max")
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
ORDER_MAX = 30
TOP_K = 20


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--hpo-set", choices=("k10", "k63"), default="k10")
    parser.add_argument(
        "--analysis-run-id",
        help="Analysis-run directory supplying the evaluated metrics; defaults to paper_reanalysis_<hpo-set>.",
    )
    parser.add_argument(
        "--candidate-scope",
        help="Metrics subdirectory for candidate sets; defaults to --hpo-set.",
    )
    parser.add_argument(
        "--output-label",
        help="Label used only in delivered HPO figure paths and filenames; defaults to --hpo-set.",
    )
    parser.add_argument("--r2-estimator", choices=("country-balanced", "global-oof"), default="country-balanced")
    parser.add_argument("--output-directory", type=Path, help="Exact delivery directory; defaults to <output-root>/<label>/paper/complete.")
    parser.add_argument("--model-rung", choices=("xgb_tree_d2", "xgb_tree_d3"), default="xgb_tree_d3")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "figures" / "HPO"
    )
    return parser.parse_args()


def _reference_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return load_historical_paper_reference(
        ROOT / "config" / "paper_reference.yaml"
    ).root.resolve()


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return path


def _metrics_path(
    repro_root: Path,
    analysis_run_id: str,
    candidate_scope: str | None = None,
    bag: str | None = None,
    rung: str | None = None,
) -> Path:
    if bag is None and rung is None:
        bag, rung = analysis_run_id, str(candidate_scope)
        analysis_run_id, candidate_scope = "paper_reanalysis_k10", "k10"
    assert bag is not None and rung is not None and candidate_scope is not None
    run_root = repro_root / "results" / "analysis_runs" / analysis_run_id
    if rung == "ols":
        path = run_root / "ols" / bag / "ols" / "ols" / "metrics_country.csv"
        if path.is_file():
            return path
        return repro_root / "results" / "analysis_runs" / "paper_reanalysis_k10" / "ols" / bag / "ols" / "ols" / "metrics_country.csv"
    return run_root / "xgb" / bag / rung / candidate_scope / "metrics_country.csv"


def _r2_metrics(path: Path, estimator: str) -> pd.DataFrame:
    """Return one new, equally-country-weighted R² per candidate."""
    if estimator == "global-oof":
        global_path = path.with_name("metrics_global.csv")
        frame = pd.read_csv(_require(global_path))
        if "candidate_id" not in frame or "global_oof_r2" not in frame:
            raise ValueError(f"{global_path} lacks candidate_id/global_oof_r2")
        return frame[["candidate_id", "global_oof_r2"]].rename(columns={"global_oof_r2": "country_balanced_r2"})
    frame = pd.read_csv(_require(path))
    required = {"candidate_id", "n_test", "r2"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    valid = frame[pd.to_numeric(frame["n_test"], errors="coerce") > 0].copy()
    valid["r2"] = pd.to_numeric(valid["r2"], errors="raise")
    result = valid.groupby("candidate_id", as_index=False, observed=True)["r2"].mean()
    return result.rename(columns={"r2": "country_balanced_r2"})


def _country_balanced(path: Path) -> pd.DataFrame:
    """Backward-compatible name for the main country-balanced estimator."""
    return _r2_metrics(path, "country-balanced")


def _registry_for_bag(registry: pd.DataFrame, bag: str) -> pd.DataFrame:
    selected = registry.loc[
        registry["experiment_id"].eq(f"pooled_oinfo_ladder_{bag}"),
        ["candidate_id", "objective", "order", "thoi_o", "predictors_identity"],
    ].copy()
    if len(selected) != 1120 or selected["candidate_id"].duplicated().any():
        raise ValueError(f"Invalid official candidate registry for bag={bag!r}")
    return selected


def _load_metrics(
    repro_root: Path, analysis_run_id: str, candidate_scope: str, estimator: str, bag: str, registry: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for rung in RUNGS:
        path = _metrics_path(repro_root, analysis_run_id, candidate_scope, bag, rung)
        metrics = _r2_metrics(path, estimator)
        merged = metrics.merge(registry, on="candidate_id", how="inner", validate="one_to_one")
        if len(merged) != len(registry):
            raise ValueError(f"Candidate registry mismatch in {path}")
        merged["bag"] = bag
        merged["rung"] = rung
        frames.append(merged)
        sources.append(str(path.with_name("metrics_global.csv") if estimator == "global-oof" else path))
    return pd.concat(frames, ignore_index=True), sources


def _domain_diversity(predictors: str, feature_domains: dict[str, str]) -> tuple[float, int, str]:
    domains = [feature_domains.get(feature, "Other") for feature in predictors.split("|") if feature]
    counts = Counter(domains)
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0, "Other"
    fractions = np.asarray([count / total for count in counts.values()], dtype=float)
    return float(-(fractions * np.log2(fractions)).sum()), len(counts), counts.most_common(1)[0][0]


def _build_tables(
    metrics: pd.DataFrame,
    feature_domains: dict[str, str],
    model_rung: str = "xgb_tree_d3",
    estimator: str = "country-balanced",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the historical capped Figure-3 selection to new metrics."""
    capped = metrics[pd.to_numeric(metrics["order"], errors="raise") <= ORDER_MAX].copy()
    scatter_rows: list[dict[str, object]] = []
    recipe_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for bag in BAGS:
        for objective in OBJECTIVES:
            arm = capped[(capped["bag"] == bag) & (capped["objective"] == objective)].copy()
            if arm.empty:
                raise ValueError(f"No capped candidates for {bag}/{objective}")
            arm = arm[arm["rung"].eq(model_rung)].copy()
            ranked = arm.sort_values(
                ["country_balanced_r2", "rung", "candidate_id"],
                ascending=[False, True, True], kind="mergesort",
            )
            winner = ranked.iloc[0]
            rung = str(winner["rung"])
            winner_metric = "winner_global_oof_r2" if estimator == "global-oof" else "winner_country_balanced_r2"
            selection_rule = (
                "maximum global pooled OOF R2 among candidates of order <= 30"
                if estimator == "global-oof"
                else "maximum country-balanced R2 among candidates of order <= 30"
            )
            selected_rows.append({
                "bag": bag, "objective": objective, "selected_rung": rung,
                "winner_candidate_id": str(winner["candidate_id"]),
                winner_metric: float(winner["country_balanced_r2"]),
                "selection_rule": selection_rule,
            })
            rung_frame = arm[arm["rung"].eq(rung)].copy()
            for row in rung_frame.itertuples(index=False):
                shannon, n_domains, dominant = _domain_diversity(str(row.predictors_identity), feature_domains)
                scatter_rows.append({
                    "model_id": str(row.candidate_id), "objective": objective, "order": int(row.order),
                    "global_oof_r2": float(row.country_balanced_r2), "country_balanced_r2": float(row.country_balanced_r2),
                    "predictors_identity": str(row.predictors_identity), "bag": bag, "rung": rung,
                    "shannon_h": shannon, "n_domains": n_domains, "dominant_domain": dominant,
                })
            top = rung_frame.sort_values(
                ["country_balanced_r2", "candidate_id"], ascending=[False, True], kind="mergesort"
            ).head(TOP_K)
            for rank, row in enumerate(top.itertuples(index=False), start=1):
                for variable in str(row.predictors_identity).split("|"):
                    if variable:
                        recipe_rows.append({
                            "bag": bag, "objective": objective, "rung": rung,
                            "model_id": str(row.candidate_id), "model_rank": rank,
                            "global_oof_r2": float(row.country_balanced_r2),
                            "country_balanced_r2": float(row.country_balanced_r2),
                            "order": int(row.order), "variable": variable,
                            "domain": feature_domains.get(variable, "Other"),
                        })
    return pd.DataFrame(scatter_rows), pd.DataFrame(recipe_rows), pd.DataFrame(selected_rows)


def _legacy_figure_module() -> Any:
    """Load the established no-SHAP drawing primitives from this checkout only."""
    core = ROOT / "src" / "repo_expo_hoi_bag" / "core"
    stages = ROOT / "src" / "repo_expo_hoi_bag" / "stages"
    for path in (core, stages):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    legacy_scripts = types.ModuleType("scripts")
    legacy_scripts.__path__ = [str(stages)]
    sys.modules["scripts"] = legacy_scripts
    # The retained drawing primitive predates this public checkout's package
    # layout.  Supply its documented configuration override rather than making
    # a compatibility directory or depending on another checkout.
    os.environ.setdefault(
        "V3_EXPOSOME_DOMAIN_LABELS_CSV",
        str(ROOT / "data" / "metadata" / "exposome_feature_domains.csv"),
    )
    from scripts import plot_fig3_diversity_v2

    return plot_fig3_diversity_v2


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _regression_table(scatter: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import linregress

    rows: list[dict[str, object]] = []
    for (bag, objective), group in scatter.groupby(["bag", "objective"], observed=True):
        group = group.dropna(subset=["shannon_h", "country_balanced_r2"])
        if len(group) < 10 or group["shannon_h"].std() <= 0:
            continue
        fit = linregress(group["shannon_h"], group["country_balanced_r2"])
        rows.append({
            "bag": bag, "objective": objective, "n_candidates": int(len(group)),
            "slope_beta_h": float(fit.slope), "intercept": float(fit.intercept),
            "pearson_r": float(fit.rvalue), "p_value": float(fit.pvalue), "stderr": float(fit.stderr),
        })
    return pd.DataFrame(rows)


def _render(scatter: pd.DataFrame, recipe: pd.DataFrame, output_dir: Path, sources: list[str], hpo_set: str, model_rung: str, estimator: str) -> None:
    figure = _legacy_figure_module()
    networks: dict[tuple[str, str], Any] = {}
    edge_frames: list[pd.DataFrame] = []
    for bag in BAGS:
        for family in ("synergy", "redundancy"):
            network = figure._build_domain_network_from_recipe(recipe, bag, family)
            networks[(bag, family)] = network
            edge_frames.append(network.edges)
    width_bins = figure._edge_width_bins_from_edge_frames(edge_frames)
    color_max = figure._edge_color_max_from_edge_frames(edge_frames)

    fig, axes = plt.subplots(len(BAGS), 4, figsize=(26, 6.2 * len(BAGS)), squeeze=False)
    fig.subplots_adjust(hspace=0.28)
    for row_index in range(len(BAGS)):
        position_2 = axes[row_index][2].get_position()
        position_3 = axes[row_index][3].get_position()
        new_x0 = position_2.x1 + 0.015
        axes[row_index][3].set_position([new_x0, position_3.y0, position_3.x1 - new_x0, position_3.height])

    for row_index, bag in enumerate(BAGS):
        figure.draw_scatter(axes[row_index][0], scatter, pd.DataFrame(), bag)
        # The historical capped panel places its exact regressions in the legend.
        labels: dict[str, str] = {}
        for text in list(axes[row_index][0].texts):
            if "β_H" not in text.get_text():
                continue
            for line in text.get_text().split("\n"):
                match = figure.re.match(r"^β_H\s+(\S+)=\S+\s+(\(r=.*\))$", line)
                if match:
                    labels[match.group(1)] = match.group(2)
            text.remove()
        legend = axes[row_index][0].get_legend()
        if legend is not None:
            handles = legend.legend_handles
            new_labels = [
                f"Min O-info {labels.get('syn', '')}".rstrip() if item.get_text() == "Synergistic" else
                f"Max O-info {labels.get('red', '')}".rstrip() if item.get_text() == "Redundant" else item.get_text()
                for item in legend.get_texts()
            ]
            legend.remove()
            axes[row_index][0].legend(handles=handles, labels=new_labels, loc="lower center", frameon=False,
                                      fontsize=figure.FS_TK - 1 + figure.FONT_BUMP, handletextpad=0.4,
                                      borderaxespad=0.2, labelspacing=0.2)
        low, high = axes[row_index][0].get_ylim()
        axes[row_index][0].set_ylim(low - 0.12 * (high - low), high)
        axes[row_index][0].set_ylabel(f"{figure.BAG_LABEL[bag]}\nLOCO R²", fontsize=figure.FS + figure.FONT_BUMP)
        figure.draw_recipe(axes[row_index][1], recipe, bag)
        for text in axes[row_index][1].texts:
            if text.get_text() == "Syn":
                text.set_text("Min O-info")
            elif text.get_text() == "Red":
                text.set_text("Max O-info")
        figure._draw_domain_panel_scaled(axes[row_index][2], networks[(bag, "redundancy")], edge_width_bins=width_bins,
                                         edge_color_max=color_max, node_size_scale=4.8, label_fontsize=17.0)
        figure._draw_domain_panel_scaled(axes[row_index][3], networks[(bag, "synergy")], edge_width_bins=width_bins,
                                         edge_color_max=color_max, node_size_scale=4.8, label_fontsize=17.0)
        for axis in (axes[row_index][0], axes[row_index][1]):
            for label in (axis.xaxis.label, axis.yaxis.label):
                label.set_fontsize(label.get_fontsize() + figure.FONT_BUMP)
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontsize(tick.get_fontsize() + figure.FONT_BUMP)
            for text in axis.texts:
                text.set_fontsize(text.get_fontsize() + figure.FONT_BUMP)
            if axis.get_legend() is not None:
                for text in axis.get_legend().get_texts():
                    text.set_fontsize(text.get_fontsize() + figure.FONT_BUMP)
        if row_index == 0:
            for column, title in enumerate(figure.CAP_COL_TITLES):
                title_x = -0.34 if column == 0 else 0.12 if column == 3 else 0.0
                axes[row_index][column].set_title(title, fontsize=18 + figure.TITLE_BUMP, loc="left", x=title_x)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig3_diversity_v2_max_30_hpo_{hpo_set}_{model_rung.removeprefix('xgb_tree_')}"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    panels = figure._build_fig3_panels(scatter, recipe, networks, list(BAGS))
    for panel in panels:
        if "global_oof_r2" in panel.columns:
            panel.columns["global_oof_r2"] = (
                ("global pooled held-out LOCO R² -- the y axis" if estimator == "global-oof" else "country-balanced held-out LOCO R² (unweighted mean over countries) -- the y axis")
            )
        if "country_balanced_r2" in panel.frame.columns:
            panel.columns["country_balanced_r2"] = (
                ("global pooled held-out LOCO R²" if estimator == "global-oof" else "country-balanced held-out LOCO R² (unweighted mean over countries)")
            )
        panel.notes = (panel.notes + " " if panel.notes else "") + (
            ("All displayed performance values are newly calculated global pooled LOCO R²." if estimator == "global-oof" else "All displayed performance values are newly calculated country-balanced LOCO R².")
        )
    write_source_data(stem, panels, output_dir, source_paths=sources)


def main() -> None:
    args = _args()
    repro_root = args.repro_data_root.resolve()
    analysis_run_id = args.analysis_run_id or f"paper_reanalysis_{args.hpo_set}"
    candidate_scope = args.candidate_scope or args.hpo_set
    output_label = args.output_label or args.hpo_set
    reference_root = _reference_root(args.reference_root)
    registry_path = reference_root / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "candidate_registry.parquet"
    domains_path = ROOT / "data" / "metadata" / "exposome_feature_domains.csv"
    registry = pd.read_parquet(_require(registry_path))
    domains = pd.read_csv(_require(domains_path)).set_index("feature_name")["domain"].astype(str).to_dict()
    all_metrics: list[pd.DataFrame] = []
    sources = [str(registry_path), str(domains_path)]
    for bag in BAGS:
        metrics, paths = _load_metrics(repro_root, analysis_run_id, candidate_scope, args.r2_estimator, bag, _registry_for_bag(registry, bag))
        all_metrics.append(metrics)
        sources.extend(paths)
    metrics = pd.concat(all_metrics, ignore_index=True)
    scatter, recipe, selected = _build_tables(metrics, domains, args.model_rung, args.r2_estimator)
    regressions = _regression_table(scatter)
    suffix = "" if args.r2_estimator == "country-balanced" else "_global"
    stats_dir = repro_root / "results" / "analysis_runs" / analysis_run_id / "main_statistics" / f"fig3_diversity_{args.model_rung.removeprefix('xgb_tree_')}{suffix}"
    metrics_name = "candidate_global_oof_metrics.csv" if args.r2_estimator == "global-oof" else "candidate_country_balanced_metrics.csv"
    _atomic_csv(metrics, stats_dir / metrics_name)
    _atomic_csv(selected, stats_dir / "selected_rungs.csv")
    _atomic_csv(scatter, stats_dir / "per_candidate_diversity_scatter.csv")
    _atomic_csv(recipe, stats_dir / "per_candidate_recipe_top20.csv")
    _atomic_csv(regressions, stats_dir / "domain_diversity_regression.csv")
    manifest = {"hpo_set": args.hpo_set, "analysis_run_id": analysis_run_id, "candidate_scope": candidate_scope, "output_label": output_label, "model_rung": args.model_rung, "performance_estimator": args.r2_estimator, "order_max": ORDER_MAX,
                "top_k": TOP_K, "sources": sources}
    (stats_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_dir = args.output_directory.resolve() if args.output_directory else args.output_root.resolve() / output_label / "paper" / "complete"
    _render(scatter, recipe, output_dir, sources, output_label, args.model_rung, args.r2_estimator)
    print(f"Saved Figure 3 statistics: {stats_dir}")


if __name__ == "__main__":
    main()
