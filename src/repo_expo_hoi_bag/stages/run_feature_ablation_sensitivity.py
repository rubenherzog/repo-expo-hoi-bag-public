#!/usr/bin/env python3
"""Leave-one-exposure-out refits for the published top-20 BAG models.

The paper's pooled canonical metrics select the top 20 candidates separately
within BAG, model level, and O-information arm. This stage removes each
exposure from those fixed parent sets, reuses an already-canonical fit whenever
the resulting predictor identity exists, and refits only unseen identities
under the unchanged subject-level LOCO procedure.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from repo_expo_hoi_bag.analysis.feature_ablation import (  # noqa: E402
    ablation_candidate_table,
    arm_contrasts,
    attach_ablation_metrics,
    build_ablation_requests,
    cloud_summary,
    parent_ablation_summary,
    select_ranked_top_models,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402
from repo_expo_hoi_bag.figures.style import (  # noqa: E402
    BAG_LABELS,
    BAG_ROW_ORDER,
    LEVEL_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    SYN_COLOR,
    save_figure,
    style_axis,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.sensitivity_common import (  # noqa: E402
    active_rungs,
    analysis_cfg_from_config,
    build_original_model_df,
    bundle_root,
    bundle_sensitivity_root,
    copy_tree_contents,
    env_bool,
    evaluate_candidates_by_rung,
    load_raw_and_domains,
    load_sensitivity_config,
    log_msg,
    repo_sensitivity_root,
    selected_bags,
    sensitivity_eval_work_root,
)


STEM = "feature_ablation"
ARM_LABELS = {"o_min": "Synergy arm", "o_max": "Redundancy arm"}
ARM_COLORS = {"o_min": SYN_COLOR, "o_max": RED_COLOR}


def _feature_ablation_config(cfg: dict) -> dict:
    try:
        values = dict(cfg["feature_ablation"])
        values["top_k"] = int(values["top_k"])
        values["order_max"] = int(values["order_max"])
        values["bootstrap_draws"] = int(values["bootstrap_draws"])
        values["permutation_draws"] = int(values["permutation_draws"])
        values["random_seed"] = int(values["random_seed"])
        values["jitter_width"] = float(values["jitter_width"])
        values["arm_offset"] = float(values["arm_offset"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid feature_ablation sensitivity configuration") from exc
    if values["top_k"] < 1 or values["order_max"] < 2:
        raise ValueError("feature_ablation.top_k must be positive and order_max must be at least 2")
    if values["bootstrap_draws"] < 1 or values["permutation_draws"] < 1:
        raise ValueError("feature_ablation resampling draw counts must be positive")
    return values


def _canonical_metrics_path() -> Path:
    root = bundle_root()
    candidates = (
        root / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet",
        root / "runs" / "oinfo_only" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Feature ablation requires pooled canonical metrics. Checked: " + ", ".join(map(str, candidates))
    )


def _canonical_identity_metrics(metrics: pd.DataFrame, *, bag: str, rung: str) -> pd.DataFrame:
    subset = metrics.loc[
        (metrics["bag_target"].astype(str) == bag) & (metrics["rung_id"].astype(str) == rung),
        ["predictors_identity", "full_r2"],
    ].copy()
    subset["predictors_identity"] = subset["predictors_identity"].astype(str)
    subset["full_r2"] = pd.to_numeric(subset["full_r2"], errors="coerce")
    subset = subset.dropna(subset=["full_r2"])
    duplicate_spread = subset.groupby("predictors_identity")["full_r2"].agg(lambda values: values.max() - values.min())
    inconsistent = duplicate_spread[duplicate_spread > 1e-12]
    if not inconsistent.empty:
        raise ValueError(
            f"Canonical predictor identities disagree on R² for bag={bag}, rung={rung}: "
            f"{inconsistent.index[:3].tolist()}"
        )
    out = subset.drop_duplicates("predictors_identity", keep="first").rename(
        columns={"predictors_identity": "ablation_predictors_identity", "full_r2": "ablated_full_r2"}
    )
    out["provenance"] = "canonical_reuse"
    return out


def _cache_paths(eval_dir: Path) -> tuple[Path, Path]:
    return eval_dir / "ablation_metric_cache.csv", eval_dir / "ablation_metric_cache.json"


def _cache_signature(*, bag: str, rung: str, config: dict) -> str:
    payload = {
        "bag": bag,
        "rung": rung,
        "order_max": config["order_max"],
        "selection_top_k": config["top_k"],
        "method": "subject_level_loco_leave_one_exposure_out_v1",
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_refit_cache(eval_dir: Path, *, bag: str, rung: str, config: dict) -> pd.DataFrame:
    cache_path, manifest_path = _cache_paths(eval_dir)
    if not cache_path.exists() and not manifest_path.exists():
        return pd.DataFrame(columns=["ablation_predictors_identity", "ablated_full_r2", "provenance"])
    if not cache_path.exists() or not manifest_path.exists():
        raise ValueError(f"Incomplete ablation cache at {eval_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != _cache_signature(bag=bag, rung=rung, config=config):
        raise ValueError(f"Ablation cache signature mismatch at {eval_dir}; use a new cache directory")
    cached = pd.read_csv(cache_path)
    required = {"ablation_predictors_identity", "ablated_full_r2", "provenance"}
    missing = required.difference(cached.columns)
    if missing:
        raise ValueError(f"Ablation cache missing columns at {eval_dir}: {sorted(missing)}")
    cached["ablated_full_r2"] = pd.to_numeric(cached["ablated_full_r2"], errors="coerce")
    return cached.dropna(subset=["ablated_full_r2"]).drop_duplicates("ablation_predictors_identity")


def _write_refit_cache(cache: pd.DataFrame, eval_dir: Path, *, bag: str, rung: str, config: dict) -> None:
    cache_path, manifest_path = _cache_paths(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache.sort_values("ablation_predictors_identity", kind="stable").to_csv(cache_path, index=False)
    manifest_path.write_text(
        json.dumps({"signature": _cache_signature(bag=bag, rung=rung, config=config)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_metrics_for_cell(
    requests: pd.DataFrame,
    canonical_metrics: pd.DataFrame,
    *,
    bag: str,
    rung: str,
    model_df: pd.DataFrame,
    feature_names: list[str],
    analysis_cfg: dict,
    eval_dir: Path,
    n_jobs: int,
    config: dict,
) -> pd.DataFrame:
    unique_requests = requests.drop_duplicates("ablation_predictors_identity", keep="first").copy()
    canonical = _canonical_identity_metrics(canonical_metrics, bag=bag, rung=rung)
    cache = _load_refit_cache(eval_dir, bag=bag, rung=rung, config=config)
    resolved = pd.concat([canonical, cache], ignore_index=True).drop_duplicates(
        "ablation_predictors_identity", keep="first"
    )
    known = set(resolved["ablation_predictors_identity"].astype(str))
    missing = unique_requests.loc[
        ~unique_requests["ablation_predictors_identity"].astype(str).isin(known)
    ].copy()
    if not missing.empty:
        candidates = ablation_candidate_table(missing)
        batch_digest = sha256(
            "\n".join(sorted(candidates["predictors_identity"].astype(str))).encode("utf-8")
        ).hexdigest()[:16]
        batch_dir = eval_dir / "batches" / batch_digest
        log_msg(
            f"Feature ablation refit bag={bag} rung={rung}: {len(candidates)} unseen predictor identities"
        )
        summary, _country = evaluate_candidates_by_rung(
            model_df=model_df,
            candidate_df=candidates,
            exposome_cols=feature_names,
            bag=bag,
            rungs=[rung],
            analysis_cfg=analysis_cfg,
            outdir=batch_dir,
            n_jobs=n_jobs,
        )
        refitted = summary[["predictors_identity", "global_oof_r2"]].rename(
            columns={"predictors_identity": "ablation_predictors_identity", "global_oof_r2": "ablated_full_r2"}
        )
        refitted["ablated_full_r2"] = pd.to_numeric(refitted["ablated_full_r2"], errors="coerce")
        if refitted["ablated_full_r2"].isna().any() or len(refitted) != len(candidates):
            raise ValueError(f"Refit failed to produce finite R² for every ablation in bag={bag}, rung={rung}")
        refitted["provenance"] = "new_refit"
        cache = pd.concat([cache, refitted], ignore_index=True).drop_duplicates(
            "ablation_predictors_identity", keep="last"
        )
        _write_refit_cache(cache, eval_dir, bag=bag, rung=rung, config=config)
        resolved = pd.concat([canonical, cache], ignore_index=True).drop_duplicates(
            "ablation_predictors_identity", keep="first"
        )
    out = resolved.loc[
        resolved["ablation_predictors_identity"].astype(str).isin(
            set(unique_requests["ablation_predictors_identity"].astype(str))
        )
    ].copy()
    out["bag"] = bag
    out["rung_id"] = rung
    return out


def _plot_results(
    results: pd.DataFrame,
    clouds: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    bags: list[str],
    rungs: list[str],
    config: dict,
    outdir: Path,
    stem: str = STEM,
    value_column: str = "r2_loss",
    value_label: str = "ΔR² after leave-one-exposure-out refit",
    value_description: str = "Leave-one-exposure-out R² losses",
    source_value_description: str = "parent_full_r2 minus ablated_full_r2; positive values indicate loss after removal",
    source_paths: list[str] | None = None,
) -> None:
    present_bags = [bag for bag in BAG_ROW_ORDER if bag in bags]
    figure, axes = plt.subplots(
        len(present_bags), len(rungs), figsize=(4.5 * len(rungs), 4.6 * len(present_bags)), squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.34},
    )
    bag_y_limits: dict[str, tuple[float, float]] = {}
    for bag in present_bags:
        values = pd.to_numeric(results.loc[results["bag"] == bag, value_column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Cannot determine y-axis limits for {bag}: no finite {value_column} values")
        upper = max(float(values.max()), 0.01)
        lower = min(float(values.min()), 0.0)
        bag_y_limits[bag] = (lower - 0.06 * upper, 1.08 * upper)
    panels: list[Panel] = []
    for row_index, bag in enumerate(present_bags):
        for column_index, rung in enumerate(rungs):
            axis = axes[row_index, column_index]
            cell = results.loc[(results["bag"] == bag) & (results["rung_id"] == rung)].copy()
            cloud = clouds.loc[(clouds["bag"] == bag) & (clouds["rung_id"] == rung)].copy()
            for arm, offset in (("o_min", -config["arm_offset"]), ("o_max", config["arm_offset"])):
                arm_rows = cell.loc[cell["arm"] == arm].copy()
                if arm_rows.empty:
                    continue
                stable = pd.util.hash_pandas_object(
                    arm_rows[["parent_candidate_id", "removed_feature"]], index=False
                ).to_numpy(dtype=np.uint64)
                jitter = ((stable % 1_000_003) / 1_000_003 - 0.5) * 2 * config["jitter_width"]
                x = arm_rows["rank_within_arm"].to_numpy(dtype=float) + offset + jitter
                axis.scatter(x, arm_rows[value_column], s=10, color=ARM_COLORS[arm], alpha=0.42, linewidths=0)
                marks = cloud.loc[cloud["arm"] == arm]
                for mark in marks.itertuples(index=False):
                    xpos = float(mark.rank_within_arm) + offset
                    axis.vlines(xpos, mark.q25_r2_loss, mark.q75_r2_loss, color=ARM_COLORS[arm], lw=1.8, zorder=3)
                    axis.hlines(mark.median_r2_loss, xpos - 0.09, xpos + 0.09, color=ARM_COLORS[arm], lw=2.4, zorder=4)
            axis.axhline(0, color="#888888", lw=0.8, zorder=0)
            axis.set_xlim(0.35, 20.65)
            # The four model levels within a BAG share a scale; structural and
            # functional BAG retain their own range so each cloud is legible.
            axis.set_ylim(*bag_y_limits[bag])
            axis.set_xticks([1, 5, 10, 15, 20])
            axis.set_xlabel("Top-20 rank by complete-model R²" if row_index == len(present_bags) - 1 else "")
            if row_index == 0 and column_index == 0:
                axis.set_ylabel(value_label)
                axis.set_title(
                    f"{ROW_LETTERS[row_index]}. {BAG_LABELS[bag]} — {LEVEL_LABELS[rung]}",
                    loc="left",
                    fontsize=13,
                )
            elif row_index == 0:
                axis.set_title(LEVEL_LABELS[rung], fontsize=13)
            elif column_index == 0:
                axis.set_ylabel(value_label)
                axis.set_title(f"{ROW_LETTERS[row_index]}. {BAG_LABELS[bag]}", loc="left", fontsize=13)
            style_axis(axis)
            panel_frame = cell.merge(
                cloud[
                    [
                        "arm",
                        "rank_within_arm",
                        "parent_candidate_id",
                        "n_ablations",
                        "q25_r2_loss",
                        "median_r2_loss",
                        "q75_r2_loss",
                    ]
                ],
                on=["arm", "rank_within_arm", "parent_candidate_id"],
                how="left",
                validate="many_to_one",
            )
            panel_frame["model_level"] = LEVEL_LABELS[rung]
            panel_frame["arm_label"] = panel_frame["arm"].map(ARM_LABELS)
            value_fields = ["r2_loss"] if value_column == "r2_loss" else ["r2_loss", value_column]
            panel_columns = [
                "model_level",
                "arm",
                "arm_label",
                "rank_within_arm",
                "parent_candidate_id",
                "parent_full_r2",
                "removed_feature",
                "ablated_full_r2",
                *value_fields,
                "provenance",
                "n_ablations",
                "q25_r2_loss",
                "median_r2_loss",
                "q75_r2_loss",
            ]
            source_columns = {
                "rank_within_arm": "rank within arm by complete-model held-out LOCO R²; 1 is highest",
                "parent_full_r2": "complete parent model held-out LOCO R²",
                "ablated_full_r2": "held-out LOCO R² after removing removed_feature and refitting",
                "r2_loss": "parent_full_r2 minus ablated_full_r2; positive values indicate loss after removal",
                "provenance": "canonical_reuse if an identical predictor set was already evaluated; new_refit otherwise",
                "q25_r2_loss": "lower quartile of the rank-specific displayed point cloud",
                "median_r2_loss": "median mark drawn for the rank-specific displayed point cloud",
                "q75_r2_loss": "upper quartile of the rank-specific displayed point cloud",
            }
            if value_column != "r2_loss":
                source_columns[value_column] = source_value_description
            panels.append(
                Panel(
                    panel_id=f"{ROW_LETTERS[row_index]}_{bag[:6]}_{LEVEL_LABELS[rung]}",
                    frame=panel_frame[panel_columns],
                    description=(
                        f"{value_description} for the top-20 {ARM_LABELS['o_min'].lower()} and "
                        f"{ARM_LABELS['o_max'].lower()} models in the {BAG_LABELS[bag]} at {LEVEL_LABELS[rung]}."
                    ),
                    columns=source_columns,
                    notes="Ranks are calculated independently within the synergy and redundancy arms.",
                )
            )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SYN_COLOR, label="Synergy arm", markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=RED_COLOR, label="Redundancy arm", markersize=6),
    ]
    axes[0, -1].legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
    save_figure(figure, stem, outdir)
    plt.close(figure)
    write_source_data(stem, panels, outdir, source_paths=source_paths or ["feature_ablation_results.csv"])


def main() -> None:
    cfg = load_sensitivity_config()
    config = _feature_ablation_config(cfg)
    smoke = env_bool("SENSITIVITY_SMOKE") or env_bool("SMOKE_TEST")
    bags = selected_bags(cfg, include_combined=False)
    rungs = active_rungs(cfg)
    top_k = config["top_k"]
    if smoke:
        bags = bags[:1]
        rungs = rungs[:2]
        top_k = min(2, top_k)
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    canonical_path = _canonical_metrics_path()
    canonical_metrics = pd.read_parquet(canonical_path)
    parents = select_ranked_top_models(
        canonical_metrics, bags=bags, rungs=rungs, top_k=top_k, order_max=config["order_max"]
    )
    requests = build_ablation_requests(parents)
    summary_root = repo_sensitivity_root(cfg) / "feature_ablation"
    summary_root.mkdir(parents=True, exist_ok=True)
    parents.to_csv(summary_root / "feature_ablation_parent_selection.csv", index=False)
    requests.to_csv(summary_root / "feature_ablation_requests.csv", index=False)

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, analysis_cfg_from_config(cfg))
    analysis_cfg = analysis_cfg_from_config(cfg)
    eval_root = sensitivity_eval_work_root(cfg, "feature_ablation")
    cell_metrics: list[pd.DataFrame] = []
    for bag in bags:
        for rung in rungs:
            cell_requests = requests.loc[(requests["bag"] == bag) & (requests["rung_id"] == rung)].copy()
            metrics = _resolve_metrics_for_cell(
                cell_requests,
                canonical_metrics,
                bag=bag,
                rung=rung,
                model_df=model_df,
                feature_names=feature_names,
                analysis_cfg=analysis_cfg,
                eval_dir=eval_root / bag / rung,
                n_jobs=n_jobs,
                config=config,
            )
            cell_metrics.append(metrics)
    ablation_metrics = pd.concat(cell_metrics, ignore_index=True)
    results = attach_ablation_metrics(requests, ablation_metrics)
    parent_summary = parent_ablation_summary(results)
    clouds = cloud_summary(results)
    contrasts = arm_contrasts(
        parent_summary,
        bootstrap_draws=config["bootstrap_draws"],
        permutation_draws=config["permutation_draws"],
        seed=config["random_seed"],
    )
    results.to_csv(summary_root / "feature_ablation_results.csv", index=False)
    parent_summary.to_csv(summary_root / "feature_ablation_parent_summary.csv", index=False)
    clouds.to_csv(summary_root / "feature_ablation_cloud_summary.csv", index=False)
    contrasts.to_csv(summary_root / "feature_ablation_arm_contrasts.csv", index=False)
    reuse_audit = (
        results.groupby(["bag", "rung_id", "provenance"], as_index=False)
        .agg(ablation_rows=("r2_loss", "size"), physical_identities=("ablation_predictors_identity", "nunique"))
        .sort_values(["bag", "rung_id", "provenance"], kind="stable")
    )
    reuse_audit.to_csv(summary_root / "feature_ablation_reuse_audit.csv", index=False)
    log_msg(
        f"Feature ablation complete: requested_rows={len(results)} "
        f"physical_identities={results.drop_duplicates(['bag', 'rung_id', 'ablation_predictors_identity']).shape[0]}"
    )
    # Sensitivity figures live beside their result tables so a standalone
    # sensitivity bundle contains both the analysis and its render artefacts.
    figure_root = summary_root / "figures"
    _plot_results(results, clouds, contrasts, bags=bags, rungs=rungs, config=config, outdir=figure_root)
    bundle_destination = bundle_sensitivity_root(cfg) / "feature_ablation"
    copy_tree_contents(summary_root, bundle_destination)
    copy_tree_contents(figure_root, bundle_destination / "figures")


if __name__ == "__main__":
    main()
