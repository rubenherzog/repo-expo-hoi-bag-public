#!/usr/bin/env python3
"""Compose the validated Figure 2B and Figure 3 scatter panels across HPO sets.

This figure contains no recalculation: each cell is redrawn from the delivered
source-data tables of the already validated global-OOF figures, using
the same Figure 2 and no-SHAP Figure 3 drawing primitives.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from repo_expo_hoi_bag.config.models import load_historical_paper_reference

ROOT = Path(__file__).resolve().parents[3]
SPEC_LABELS = ("Historical", "k1 (single)", "k10", "k63")
ROWS = (("structural", "b1_struct_r2", "a1_struct_diversity_scatter"),
        ("functional", "c1_func_r2", "b1_func_diversity_scatter"))
RUNG_MAP = {"OLS": "ols", "d1": "xgb_tree_d1", "d2": "xgb_tree_d2", "d3": "xgb_tree_d3"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "outputs/figures/HPO/comparative")
    parser.add_argument("--r2-estimator", choices=("global-oof", "country-balanced"), default="global-oof")
    return parser.parse_args()


def _legacy_modules() -> tuple[object, object]:
    stages = ROOT / "src/repo_expo_hoi_bag/stages"
    core = ROOT / "src/repo_expo_hoi_bag/core"
    for path in (core, stages):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    scripts = types.ModuleType("scripts")
    scripts.__path__ = [str(stages)]
    sys.modules["scripts"] = scripts
    os.environ.setdefault("V3_EXPOSOME_DOMAIN_LABELS_CSV", str(ROOT / "data/metadata/exposome_feature_domains.csv"))
    from scripts import plot_fig3_diversity
    from repo_expo_hoi_bag.stages import plot_fig2_grid_v3
    return plot_fig2_grid_v3, plot_fig3_diversity


def _source(directory: Path, token: str) -> Path:
    matches = sorted(directory.glob(f"*_{token}.csv"))
    d3_matches = [path for path in matches if "_d3_source_data_" in path.name]
    if len(d3_matches) == 1:
        return d3_matches[0]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source-data table for {token} in {directory}; found {matches}")
    return matches[0]


def _source_directories(estimator: str) -> tuple[Path, Path, Path, Path]:
    suffix = "global/paper/complete/source_data" if estimator == "global-oof" else "paper/complete/source_data"
    return (
        ROOT / "outputs/figures/HPO/historical_validation/paper/complete/source_data",
        ROOT / f"outputs/figures/HPO/single/{suffix}",
        ROOT / f"outputs/figures/HPO/k10/{suffix}",
        ROOT / f"outputs/figures/HPO/k63/{suffix}",
    )


def _reference_root() -> Path:
    return load_historical_paper_reference(
        ROOT / "config" / "paper_reference.yaml"
    ).root


def _historical_cb_r2(bag: str) -> tuple[pd.DataFrame, dict[str, float], list[Path]]:
    """Exact country-mean analogue of the delivered historical Figure 2B."""
    ref = _reference_root() / "results/variant_a"
    metrics_path = ref / "families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}/metrics_country_long.parquet"
    metrics = pd.read_parquet(metrics_path)
    frame = metrics.groupby(["candidate_id", "objective", "order", "predictors_identity", "rung_id"], observed=True, as_index=False).agg(
        full_r2=("country_full_r2", "mean"), base_r2=("country_base_r2", "mean")
    )
    frame = frame[frame["order"].le(30)].copy()
    frame["delta_r2_vs_base"] = frame["full_r2"] - frame["base_r2"]
    singles: dict[str, float] = {}
    sources = [metrics_path]
    for rung in RUNG_MAP.values():
        folder = ref / f"single_exposure_eval_{rung}" / bag
        path = folder / "single_exposure_country.csv"
        single = pd.read_csv(path)
        singles[rung] = float(single[single["n_test"] > 0].groupby("feature_name", observed=True)["r2"].mean().max())
        sources.append(path)
    return frame, singles, sources


def _historical_cb_scatter(path: Path, bag: str) -> tuple[pd.DataFrame, Path]:
    """Reuse Fig. 3 geometry and replace only its y values by country means."""
    ref = _reference_root() / "results/variant_a"
    metrics_path = ref / "families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}/metrics_country_long.parquet"
    scores = pd.read_parquet(metrics_path)
    scores = scores[scores["rung_id"].eq("xgb_tree_d3")].groupby("candidate_id", observed=True, as_index=False).country_full_r2.mean()
    frame = pd.read_csv(path).merge(scores, left_on="model_id", right_on="candidate_id", validate="one_to_one")
    frame["global_oof_r2"] = frame.pop("country_full_r2")
    frame["bag"] = bag
    return frame, metrics_path


def _fig2_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"baseline_r2": "base_r2"})
    frame["rung_id"] = frame["model_level"].map(RUNG_MAP)
    frame["delta_r2_vs_base"] = frame["full_r2"] - frame["base_r2"]
    if frame["rung_id"].isna().any():
        raise ValueError(f"Unknown Figure 2 model level in {path}")
    return frame


def _scatter_frame(path: Path, bag: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["bag"] = bag
    return frame


def _single_values(frame: pd.DataFrame) -> dict[str, float]:
    return frame.groupby("rung_id", observed=True)["best_single_r2"].first().to_dict()


def _correlation_annotation(axis: plt.Axes) -> None:
    """Keep the existing Fig. 3 r/p annotation, omitting its slope prefix."""
    for text in axis.texts:
        lines = []
        for line in text.get_text().splitlines():
            match = re.match(r"^β_H\s+(\S+)=\S+\s+\(r=([^,]+),\s*(.+)\)$", line)
            lines.append(f"r {match.group(1)}={match.group(2)} ({match.group(3)})" if match else line)
        text.set_text("\n".join(lines))


def _shared_y_limits(axes: list[plt.Axes]) -> None:
    low = min(axis.get_ylim()[0] for axis in axes)
    high = max(axis.get_ylim()[1] for axis in axes)
    for axis in axes:
        axis.set_ylim(low, high)


def _partial_correlation_title(frame: pd.DataFrame) -> str:
    """Partial Pearson r(H, R² | set size), separately for the two arms."""
    lines = []
    for objective, label in (("o_max", "Red"), ("o_min", "Syn")):
        subset = frame[frame["objective"].eq(objective)].dropna(subset=["shannon_h", "global_oof_r2", "order"])
        design = np.column_stack([np.ones(len(subset)), subset["order"].to_numpy(float)])
        h = subset["shannon_h"].to_numpy(float)
        r2 = subset["global_oof_r2"].to_numpy(float)
        h_res = h - design @ np.linalg.lstsq(design, h, rcond=None)[0]
        r2_res = r2 - design @ np.linalg.lstsq(design, r2, rcond=None)[0]
        result = stats.pearsonr(h_res, r2_res)
        lines.append(f"{label}: partial r={result.statistic:.2f}, p={result.pvalue:.2g}")
    return "\n".join(lines)


def main() -> None:
    args = _args(); fig2, fig3 = _legacy_modules()
    # Exact repository-wide font contract, reasserted after the legacy Fig. 3
    # import which historically changes matplotlib's process-wide defaults.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"]
    figure, axes = plt.subplots(4, 4, figsize=(23, 20), squeeze=False)
    figure.subplots_adjust(left=.075, right=.985, top=.94, bottom=.055, hspace=.42, wspace=.28)
    source_rows: list[dict[str, str]] = []
    directories = _source_directories(args.r2_estimator)
    for column, (label, directory) in enumerate(zip(SPEC_LABELS, directories)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing validated source-data directory: {directory}")
        axes[0, column].set_title(label, fontsize=fig2.FS)
        for bag_index, (bag, r2_token, scatter_token) in enumerate(ROWS):
            r2_path, scatter_path = _source(directory, r2_token), _source(directory, scatter_token)
            r2_axis, scatter_axis = axes[2 * bag_index, column], axes[2 * bag_index + 1, column]
            # OLS was never tuned.  Therefore every parameter-set column must
            # use the immutable historical OLS candidates, baseline and best
            # single reference; only d1--d3 vary by parameter set.
            if args.r2_estimator == "country-balanced":
                historical_r2, historical_singles, cb_sources = _historical_cb_r2(bag)
            else:
                historical_path = _source(directories[0], r2_token)
                historical_r2 = _fig2_frame(historical_path)
                historical_singles = _single_values(historical_r2)
                cb_sources = [historical_path]
            if args.r2_estimator == "country-balanced" and column == 0:
                r2_frame, singles = historical_r2, historical_singles
                scatter_frame, scatter_source = _historical_cb_scatter(scatter_path, bag)
                fig2.draw_panel_b(r2_axis, r2_frame, "", show_legend=column == 0, single_exp_r2=singles)
                fig3.draw_scatter(scatter_axis, scatter_frame, pd.DataFrame(), bag)
                source_rows.extend({"parameter_set": label, "bag": bag, "panel": "historical_country_balanced_input", "source": str(source)} for source in [*cb_sources, scatter_source])
            else:
                r2_frame = _fig2_frame(r2_path)
                singles = _single_values(r2_frame)
                if column:
                    r2_frame = pd.concat([r2_frame[r2_frame["rung_id"].ne("ols")], historical_r2[historical_r2["rung_id"].eq("ols")]], ignore_index=True)
                    singles["ols"] = historical_singles["ols"]
                    source_rows.extend({"parameter_set": label, "bag": bag, "panel": "historical_ols_shared_reference", "source": str(source)} for source in cb_sources)
                fig2.draw_panel_b(r2_axis, r2_frame, "", show_legend=column == 0, single_exp_r2=singles)
                fig3.draw_scatter(scatter_axis, _scatter_frame(scatter_path, bag), pd.DataFrame(), bag)
            _correlation_annotation(scatter_axis)
            scatter_data = scatter_frame if args.r2_estimator == "country-balanced" and column == 0 else _scatter_frame(scatter_path, bag)
            scatter_axis.set_title(_partial_correlation_title(scatter_data), fontsize=fig3.FS_TK, loc="center")
            if column:
                r2_axis.set_ylabel("")
                scatter_axis.set_ylabel("")
            else:
                bag_label = "Structural BAG" if bag == "structural" else "Functional BAG"
                estimator_label = "Global LOCO R²" if args.r2_estimator == "global-oof" else "Country-balanced LOCO R²"
                r2_axis.set_ylabel(f"{bag_label}\n{estimator_label}", fontsize=fig2.FS)
                scatter_axis.set_ylabel(f"{bag_label}\n{estimator_label}", fontsize=fig2.FS)
            if column:
                legend = r2_axis.get_legend()
                if legend is not None: legend.remove()
                legend = scatter_axis.get_legend()
                if legend is not None: legend.remove()
            source_rows.extend((
                {"parameter_set": label, "bag": bag, "panel": "fig2_loco_r2_by_rung", "source": str(r2_path)},
                {"parameter_set": label, "bag": bag, "panel": "fig3_diversity_vs_r2", "source": str(scatter_path)},
            ))
    for row in range(4):
        _shared_y_limits(list(axes[row, :]))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"fig2_fig3_parameter_comparison_{args.r2_estimator.replace('-', '_')}"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.output_directory / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(source_rows).to_csv(args.output_directory / f"{stem}_source_manifest.csv", index=False)
    print(f"Saved: {args.output_directory / (stem + '.png')}")


if __name__ == "__main__":
    main()
