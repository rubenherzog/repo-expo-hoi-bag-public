from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import gc
from pathlib import Path

import pandas as pd

from loco_fusion_matrix_engine import build_model_df, load_artifacts_greedy_only
from oinfo_bag_ladder.candidate_sources import (
    build_cn_top_tail_candidates,
    build_greedy_o_only_candidates,
    build_pooled_top_tail_candidates,
)
from oinfo_bag_ladder.config import (
    ANALYSIS_CFG,
    CANONICAL_ROOT,
    CV_CFG,
    DATA_PATHS,
    DEFAULT_ANALYSIS_FAMILY,
    EARLY_STOP_CFG,
    FIGURES_ROOT,
    MATRIX_CFG,
    PERF_CFG_OLS,
    PERF_CFG_XGB,
    REFERENCE_SNAPSHOT_CANONICAL_ROOT,
    canonical_root,
    experiments_root,
    figures_root,
    per_experiment_canonical_dir,
    rung_dir,
)
from oinfo_bag_ladder.experiments import build_experiment_registry
from oinfo_bag_ladder.io_utils import read_parquet, write_parquet
from oinfo_bag_ladder.plots import save_all_bag_figures
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
from oinfo_bag_ladder.stages_ols import run_or_load_ols_rung
from oinfo_bag_ladder.stages_xgb import run_or_load_xgb_rung
from oinfo_bag_ladder.tables import (
    canonicalize_country_metrics,
    canonicalize_global_metrics,
    compute_frontier_summary,
    compute_ladder_contrasts,
    compute_portability_comparison_country,
    compute_portability_comparison_global,
    compute_top_tail_summary,
    compute_transfer_comparison_country,
    compute_transfer_comparison_global,
    sort_for_reporting,
)


def _int_or_default(value, fallback: int) -> int:
    return fallback if value is None else int(value)


DEFAULT_RUN_SETTINGS = {
    "analysis_family": DEFAULT_ANALYSIS_FAMILY,
    "bag_target_mode": "all",
    "diagnosis_mode": "all",
    "candidate_limit": 0,
    "order_min": _int_or_default(ANALYSIS_CFG.get("order_min"), 3),
    "order_max": _int_or_default(ANALYSIS_CFG.get("order_max"), 30),
    "force": False,
    "render_figures": True,
    "ols_n_jobs": _int_or_default(PERF_CFG_OLS.get("n_jobs"), 40),
    "ols_chunk_size": _int_or_default(PERF_CFG_OLS.get("chunk_size_models"), 100),
    "xgb_n_jobs": _int_or_default(PERF_CFG_XGB.get("n_jobs"), 40),
    "xgb_chunk_size": _int_or_default(PERF_CFG_XGB.get("chunk_size_models"), 100),
    "pooled_reference_canonical_root": str(REFERENCE_SNAPSHOT_CANONICAL_ROOT),
    "cn_reference_canonical_root": str(canonical_root("single_dx_oinfo_ladder")),
}

RUN_SETTINGS = {
    **DEFAULT_RUN_SETTINGS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run O-information BAG ladder experiments.")
    parser.add_argument("--analysis-family", default=RUN_SETTINGS.get("analysis_family", DEFAULT_ANALYSIS_FAMILY))
    parser.add_argument("--bag-target-mode", default=RUN_SETTINGS.get("bag_target_mode", "all"))
    parser.add_argument("--diagnosis-mode", default=RUN_SETTINGS.get("diagnosis_mode", "all"))
    parser.add_argument("--force", action="store_true", default=bool(RUN_SETTINGS.get("force", False)))
    parser.add_argument("--render-figures", action="store_true", default=bool(RUN_SETTINGS.get("render_figures", True)))
    parser.add_argument("--candidate-limit", type=int, default=int(RUN_SETTINGS.get("candidate_limit", 0)))
    parser.add_argument("--order-min", type=int, default=int(RUN_SETTINGS.get("order_min", 3)))
    parser.add_argument("--order-max", type=int, default=int(RUN_SETTINGS.get("order_max", 30)))
    parser.add_argument("--ols-n-jobs", type=int, default=int(RUN_SETTINGS.get("ols_n_jobs", 40)))
    parser.add_argument("--ols-chunk-size", type=int, default=int(RUN_SETTINGS.get("ols_chunk_size", 100)))
    parser.add_argument("--xgb-n-jobs", type=int, default=int(RUN_SETTINGS.get("xgb_n_jobs", 40)))
    parser.add_argument("--xgb-chunk-size", type=int, default=int(RUN_SETTINGS.get("xgb_chunk_size", 100)))
    parser.add_argument(
        "--pooled-reference-canonical-root",
        type=Path,
        default=Path(str(RUN_SETTINGS.get("pooled_reference_canonical_root", REFERENCE_SNAPSHOT_CANONICAL_ROOT))),
    )
    parser.add_argument(
        "--cn-reference-canonical-root",
        type=Path,
        default=Path(str(RUN_SETTINGS.get("cn_reference_canonical_root", canonical_root("single_dx_oinfo_ladder")))),
    )
    return parser.parse_args()


def _ensure_dirs(analysis_family: str) -> tuple[Path, Path, Path]:
    exp_root = experiments_root(analysis_family)
    can_root = canonical_root(analysis_family)
    fig_root = figures_root(analysis_family)
    for path in [exp_root, can_root, fig_root, can_root / "per_experiment"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    return exp_root, can_root, fig_root


def _limit_candidates(
    candidate_df: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    candidate_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(candidate_limit) <= 0 or candidate_df.empty:
        return candidate_df.reset_index(drop=True), candidate_registry.reset_index(drop=True)
    ordered = (
        candidate_df.sort_values(
            by=["objective", "order", "rank", "feature_id"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        .groupby(["objective", "order"], as_index=False, group_keys=False)
        .head(int(candidate_limit))
        .copy()
    )
    keep_ids = set(ordered["candidate_id"].astype(str).tolist())
    registry = candidate_registry[candidate_registry["candidate_id"].astype(str).isin(keep_ids)].copy()
    return ordered.reset_index(drop=True), registry.reset_index(drop=True)


def _limit_selection_by_rung(
    candidate_df: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    selection_df: pd.DataFrame,
    candidate_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if int(candidate_limit) <= 0 or selection_df is None or selection_df.empty:
        return candidate_df.reset_index(drop=True), candidate_registry.reset_index(drop=True), selection_df.reset_index(drop=True)

    trimmed = (
        selection_df.sort_values(
            by=["source_rung_id", "objective", "order", "source_tail_rank", "candidate_id"],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
        .groupby(["source_rung_id", "objective", "order"], as_index=False, group_keys=False)
        .head(int(candidate_limit))
        .copy()
    )
    keep_ids = set(trimmed["candidate_id"].astype(str).tolist())
    candidate_df = candidate_df[candidate_df["candidate_id"].astype(str).isin(keep_ids)].copy()
    candidate_registry = candidate_registry[candidate_registry["candidate_id"].astype(str).isin(keep_ids)].copy()
    return candidate_df.reset_index(drop=True), candidate_registry.reset_index(drop=True), trimmed.reset_index(drop=True)


def _merge_per_experiment_df(new_df: pd.DataFrame, path: Path, id_cols: list[str]) -> pd.DataFrame:
    """Merge new_df with an existing per-experiment parquet, keeping new rows for any
    (id_cols) combination already present and retaining all other existing rows.
    This allows incremental runs to accumulate candidates without data loss.
    """
    if not path.exists() or new_df.empty:
        return new_df
    try:
        existing = read_parquet(path)
    except Exception:
        return new_df
    if existing.empty:
        return new_df
    actual_keys = [c for c in id_cols if c in existing.columns and c in new_df.columns]
    combined = pd.concat([new_df, existing], ignore_index=True)
    if actual_keys:
        return combined.drop_duplicates(subset=actual_keys, keep="first").reset_index(drop=True)
    return combined.drop_duplicates().reset_index(drop=True)


def _write_per_experiment_tables(
    analysis_family: str,
    experiment_id: str,
    global_df: pd.DataFrame,
    country_df: pd.DataFrame,
    selection_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    outdir = per_experiment_canonical_dir(analysis_family, experiment_id)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics_global_long": outdir / "metrics_global_long.parquet",
        "metrics_country_long": outdir / "metrics_country_long.parquet",
        "ladder_contrasts_long": outdir / "ladder_contrasts_long.parquet",
        "frontier_summary_long": outdir / "frontier_summary_long.parquet",
        "top_tail_summary_long": outdir / "top_tail_summary_long.parquet",
    }

    # Merge with any existing per-experiment data to accumulate candidates across runs
    global_df = sort_for_reporting(
        _merge_per_experiment_df(global_df, paths["metrics_global_long"],
                                 ["bag_target", "rung_id", "candidate_id"])
    )
    country_df = sort_for_reporting(
        _merge_per_experiment_df(country_df, paths["metrics_country_long"],
                                 ["bag_target", "rung_id", "candidate_id", "fold_country"])
    )

    # Recompute derived tables from the full merged data
    contrasts_df = sort_for_reporting(compute_ladder_contrasts(global_df))
    frontier_df = sort_for_reporting(compute_frontier_summary(global_df))
    top_tail_df = sort_for_reporting(compute_top_tail_summary(global_df))

    write_parquet(global_df, paths["metrics_global_long"])
    write_parquet(country_df, paths["metrics_country_long"])
    write_parquet(contrasts_df, paths["ladder_contrasts_long"])
    write_parquet(frontier_df, paths["frontier_summary_long"])
    write_parquet(top_tail_df, paths["top_tail_summary_long"])
    if selection_df is not None and not selection_df.empty:
        paths["selection_long"] = outdir / "selection_long.parquet"
        write_parquet(sort_for_reporting(selection_df), paths["selection_long"])
    return paths


def _aggregate_tables(analysis_family: str, experiment_ids: list[str], table_name: str) -> pd.DataFrame:
    parts = []
    for experiment_id in experiment_ids:
        path = per_experiment_canonical_dir(analysis_family, experiment_id) / f"{table_name}.parquet"
        if path.exists():
            parts.append(read_parquet(path))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new_df with an existing parquet at path.

    For tables that contain a candidate_id column, deduplication is done at
    the (bag_target, rung_id, candidate_id) level so that incremental runs
    accumulate candidates instead of replacing them.  For tables without
    candidate_id, the original bag_target-level replacement is used to keep
    parallel per-BAG runs safe.
    """
    if not path.exists() or new_df.empty:
        return new_df
    try:
        existing = read_parquet(path)
    except Exception:
        return new_df
    if existing.empty:
        return new_df
    if "bag_target" not in existing.columns or "bag_target" not in new_df.columns:
        # No bag_target column (e.g. candidate_registry) — deduplicate by all cols
        combined = pd.concat([existing, new_df], ignore_index=True)
        return combined.drop_duplicates().reset_index(drop=True)
    if "candidate_id" in existing.columns and "candidate_id" in new_df.columns:
        # Fine-grained merge: new rows take priority; existing candidates not in new_df are kept
        id_cols = [c for c in ["bag_target", "rung_id", "candidate_id"]
                   if c in existing.columns and c in new_df.columns]
        combined = pd.concat([new_df, existing], ignore_index=True)
        return combined.drop_duplicates(subset=id_cols, keep="first").reset_index(drop=True)
    # No candidate_id: drop existing rows for this BAG and replace (parallel-safe)
    new_bags = set(new_df["bag_target"].astype(str).unique())
    surviving = existing[~existing["bag_target"].astype(str).isin(new_bags)].copy()
    return pd.concat([surviving, new_df], ignore_index=True)


def _subset_model_df(model_df: pd.DataFrame, experiment_row: dict) -> pd.DataFrame:
    family = str(experiment_row["analysis_family"])
    if family in {"pooled_oinfo_ladder"}:
        return model_df.copy()
    if family in {"single_dx_oinfo_ladder", "single_dx_portability"}:
        dx = str(experiment_row["test_diagnosis_group"])
        return model_df[model_df["Diagnosis"].astype(str) == dx].copy().reset_index(drop=True)
    if family == "cn_normative_transfer":
        train_dx = str(experiment_row["train_diagnosis_group"])
        test_dx = str(experiment_row["test_diagnosis_group"])
        keep = {train_dx, test_dx}
        return model_df[model_df["Diagnosis"].astype(str).isin(keep)].copy().reset_index(drop=True)
    raise ValueError(f"Unsupported analysis family for model subset: {family}")


def _analysis_cfg_for_experiment(base_cfg: dict, experiment_row: dict) -> dict:
    cfg = dict(base_cfg)
    family = str(experiment_row["analysis_family"])
    if family in {"single_dx_oinfo_ladder", "single_dx_portability", "cn_normative_transfer"}:
        cfg["include_diagnosis"] = False
        cfg["exclude_diagnosis"] = []
    return cfg


def _cv_cfg_for_experiment(base_cfg: dict, experiment_row: dict) -> dict:
    cfg = dict(base_cfg)
    family = str(experiment_row["analysis_family"])
    if family == "cn_normative_transfer":
        cfg["train_diagnosis_group"] = str(experiment_row["train_diagnosis_group"])
        cfg["test_diagnosis_group"] = str(experiment_row["test_diagnosis_group"])
        cfg["shared_country_only"] = True
    return cfg


def _resolve_candidate_source(
    experiment_row: dict,
    *,
    filtered_candidates: pd.DataFrame,
    filtered_registry: pd.DataFrame,
    all_candidates: pd.DataFrame,
    all_registry: pd.DataFrame,
    pooled_reference_canonical_root: Path,
    cn_reference_canonical_root: Path,
    candidate_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    source = str(experiment_row["candidate_source"])
    bag_target = str(experiment_row["bag_target"])
    if source == "greedy_o_only":
        cand, reg = _limit_candidates(filtered_candidates, filtered_registry, candidate_limit)
        return cand, reg, None
    if source == "pooled_top_tail_per_rung":
        cand, reg, sel = build_pooled_top_tail_candidates(
            all_candidates,
            all_registry,
            bag_target=bag_target,
            canonical_root=pooled_reference_canonical_root,
        )
        return _limit_selection_by_rung(cand, reg, sel, candidate_limit)
    if source == "cn_top_tail_per_rung":
        cand, reg, sel = build_cn_top_tail_candidates(
            all_candidates,
            all_registry,
            bag_target=bag_target,
            canonical_root=cn_reference_canonical_root,
        )
        return _limit_selection_by_rung(cand, reg, sel, candidate_limit)
    raise ValueError(f"Unsupported candidate_source: {source}")


def _candidate_df_for_rung(candidate_df: pd.DataFrame, selection_df: pd.DataFrame | None, rung_id: str) -> pd.DataFrame:
    if selection_df is None or selection_df.empty:
        return candidate_df.reset_index(drop=True)
    keep_ids = set(selection_df.loc[selection_df["source_rung_id"].astype(str) == str(rung_id), "candidate_id"].astype(str))
    out = candidate_df[candidate_df["candidate_id"].astype(str).isin(keep_ids)].copy()
    return out.reset_index(drop=True)


def _filter_country_to_top_tail(metrics_country_df: pd.DataFrame, top_tail_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_country_df is None or metrics_country_df.empty or top_tail_df is None or top_tail_df.empty:
        return pd.DataFrame()
    keys = [c for c in ["experiment_id", "bag_target", "objective", "rung_id", "candidate_id"] if c in metrics_country_df.columns and c in top_tail_df.columns]
    tail = top_tail_df[keys].drop_duplicates()
    return metrics_country_df.merge(tail, on=keys, how="inner")


def _write_comparison_tables(
    analysis_family: str,
    family_can_root: Path,
    aggregated_global: pd.DataFrame,
    aggregated_country: pd.DataFrame,
    pooled_reference_canonical_root: Path,
    cn_reference_canonical_root: Path,
) -> None:
    if analysis_family == "single_dx_portability":
        pooled_top_tail = read_parquet(Path(pooled_reference_canonical_root) / "top_tail_summary_long.parquet")
        pooled_country = read_parquet(Path(pooled_reference_canonical_root) / "metrics_country_long.parquet")
        pooled_country = _filter_country_to_top_tail(pooled_country, pooled_top_tail)
        write_parquet(
            sort_for_reporting(compute_portability_comparison_global(pooled_top_tail, aggregated_global)),
            family_can_root / "portability_comparison_global.parquet",
        )
        write_parquet(
            sort_for_reporting(compute_portability_comparison_country(pooled_country, aggregated_country)),
            family_can_root / "portability_comparison_country.parquet",
        )
    elif analysis_family == "cn_normative_transfer":
        dx_global = read_parquet(Path(cn_reference_canonical_root) / "metrics_global_long.parquet")
        dx_country = read_parquet(Path(cn_reference_canonical_root) / "metrics_country_long.parquet")
        write_parquet(
            sort_for_reporting(compute_transfer_comparison_global(aggregated_global, dx_global)),
            family_can_root / "transfer_comparison_global.parquet",
        )
        write_parquet(
            sort_for_reporting(compute_transfer_comparison_country(aggregated_country, dx_country)),
            family_can_root / "transfer_comparison_country.parquet",
        )


def main() -> None:
    args = parse_args()
    analysis_family = str(args.analysis_family)
    _exp_root, family_can_root, family_fig_root = _ensure_dirs(analysis_family)

    analysis_cfg_base = dict(ANALYSIS_CFG)
    analysis_cfg_base["order_min"] = args.order_min
    analysis_cfg_base["order_max"] = args.order_max

    perf_cfg_ols = dict(PERF_CFG_OLS)
    perf_cfg_ols["n_jobs"] = args.ols_n_jobs
    perf_cfg_ols["chunk_size_models"] = args.ols_chunk_size

    perf_cfg_xgb = dict(PERF_CFG_XGB)
    perf_cfg_xgb["n_jobs"] = args.xgb_n_jobs
    perf_cfg_xgb["chunk_size_models"] = args.xgb_chunk_size

    raw_complete_df, exposome_df, filtered_candidate_df, candidate_all_df = load_artifacts_greedy_only(DATA_PATHS, analysis_cfg_base)
    model_df = build_model_df(raw_complete_df, exposome_df, analysis_cfg_base)
    exposome_cols = exposome_df.columns.astype(str).tolist()

    filtered_o_candidate_df, filtered_candidate_registry = build_greedy_o_only_candidates(filtered_candidate_df, exposome_cols)
    all_o_candidate_df, all_candidate_registry = build_greedy_o_only_candidates(candidate_all_df, exposome_cols)

    experiment_registry = build_experiment_registry(
        bag_target_mode=args.bag_target_mode,
        analysis_family=analysis_family,
        diagnosis_mode=args.diagnosis_mode,
    )
    er_path = family_can_root / "experiment_registry.parquet"
    merged_er = _merge_with_existing(sort_for_reporting(experiment_registry), er_path)
    write_parquet(sort_for_reporting(merged_er), er_path)

    rung_specs = get_rung_specs()
    experiment_ids: list[str] = []
    candidate_registry_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []

    for experiment_row in experiment_registry.to_dict(orient="records"):
        experiment_id = str(experiment_row["experiment_id"])
        experiment_ids.append(experiment_id)

        candidate_df_exp, candidate_registry_exp, selection_df = _resolve_candidate_source(
            experiment_row,
            filtered_candidates=filtered_o_candidate_df,
            filtered_registry=filtered_candidate_registry,
            all_candidates=all_o_candidate_df,
            all_registry=all_candidate_registry,
            pooled_reference_canonical_root=args.pooled_reference_canonical_root,
            cn_reference_canonical_root=args.cn_reference_canonical_root,
            candidate_limit=args.candidate_limit,
        )
        candidate_registry_parts.append(candidate_registry_exp.assign(experiment_id=experiment_id))
        if selection_df is not None and not selection_df.empty:
            selection_parts.append(selection_df.assign(experiment_id=experiment_id))

        run_model_df = _subset_model_df(model_df, experiment_row)
        exp_analysis_cfg = _analysis_cfg_for_experiment(analysis_cfg_base, experiment_row)
        exp_cv_cfg = _cv_cfg_for_experiment(CV_CFG, experiment_row)

        global_parts = []
        country_parts = []
        for rung_spec in rung_specs:
            rung_candidate_df = _candidate_df_for_rung(candidate_df_exp, selection_df, rung_spec["rung_id"])
            if rung_candidate_df.empty:
                continue
            stage_dir = rung_dir(analysis_family, experiment_id, rung_spec["rung_id"])
            if rung_spec["model_family"] == "ols":
                stage_data = run_or_load_ols_rung(
                    experiment_row=experiment_row,
                    rung_spec=rung_spec,
                    stage_dir=stage_dir,
                    model_df=run_model_df,
                    candidate_df=rung_candidate_df,
                    exposome_cols=exposome_cols,
                    analysis_cfg=exp_analysis_cfg,
                    cv_cfg=exp_cv_cfg,
                    perf_cfg=perf_cfg_ols,
                    matrix_cfg=MATRIX_CFG,
                    force_recompute=args.force,
                )
            else:
                stage_data = run_or_load_xgb_rung(
                    experiment_row=experiment_row,
                    rung_spec=rung_spec,
                    stage_dir=stage_dir,
                    model_df=run_model_df,
                    candidate_df=rung_candidate_df,
                    exposome_cols=exposome_cols,
                    analysis_cfg=exp_analysis_cfg,
                    cv_cfg=exp_cv_cfg,
                    perf_cfg=perf_cfg_xgb,
                    xgb_cfg=build_xgb_cfg_for_rung(rung_spec),
                    early_stop_cfg=EARLY_STOP_CFG,
                    force_recompute=args.force,
                )

            global_parts.append(
                canonicalize_global_metrics(
                    global_df=stage_data["global_df"],
                    experiment_row=experiment_row,
                    rung_spec=rung_spec,
                    candidate_registry=candidate_registry_exp,
                )
            )
            country_parts.append(
                canonicalize_country_metrics(
                    country_df=stage_data["country_df"],
                    experiment_row=experiment_row,
                    rung_spec=rung_spec,
                    candidate_registry=candidate_registry_exp,
                )
            )
            del stage_data
            gc.collect()

        exp_global = pd.concat(global_parts, ignore_index=True) if global_parts else pd.DataFrame()
        exp_country = pd.concat(country_parts, ignore_index=True) if country_parts else pd.DataFrame()
        _write_per_experiment_tables(
            analysis_family=analysis_family,
            experiment_id=experiment_id,
            global_df=exp_global,
            country_df=exp_country,
            selection_df=selection_df,
        )
        del global_parts, country_parts, exp_global, exp_country, run_model_df, candidate_df_exp, candidate_registry_exp, selection_df
        gc.collect()

    for table_name in [
        "metrics_global_long",
        "metrics_country_long",
        "ladder_contrasts_long",
        "frontier_summary_long",
        "top_tail_summary_long",
    ]:
        table_df = _aggregate_tables(analysis_family, experiment_ids, table_name)
        out_path = family_can_root / f"{table_name}.parquet"
        merged = _merge_with_existing(sort_for_reporting(table_df), out_path)
        write_parquet(sort_for_reporting(merged), out_path)
        del table_df, merged
        gc.collect()

    if candidate_registry_parts:
        candidate_registry_df = pd.concat(candidate_registry_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
        out_path = family_can_root / "candidate_registry.parquet"
        merged_cr = _merge_with_existing(sort_for_reporting(candidate_registry_df), out_path)
        write_parquet(sort_for_reporting(merged_cr), out_path)
    if selection_parts:
        selection_df_all = pd.concat(selection_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
        out_path = family_can_root / "selection_long.parquet"
        merged_sel = _merge_with_existing(sort_for_reporting(selection_df_all), out_path)
        write_parquet(sort_for_reporting(merged_sel), out_path)

    aggregated_global = read_parquet(family_can_root / "metrics_global_long.parquet")
    aggregated_country = read_parquet(family_can_root / "metrics_country_long.parquet")
    _write_comparison_tables(
        analysis_family=analysis_family,
        family_can_root=family_can_root,
        aggregated_global=aggregated_global,
        aggregated_country=aggregated_country,
        pooled_reference_canonical_root=args.pooled_reference_canonical_root,
        cn_reference_canonical_root=args.cn_reference_canonical_root,
    )

    if args.render_figures and analysis_family == DEFAULT_ANALYSIS_FAMILY:
        aggregated_frontier = read_parquet(family_can_root / "frontier_summary_long.parquet")
        aggregated_tail = read_parquet(family_can_root / "top_tail_summary_long.parquet")
        save_all_bag_figures(
            metrics_global_long=aggregated_global,
            metrics_country_long=aggregated_country,
            frontier_summary_long=aggregated_frontier,
            top_tail_summary_long=aggregated_tail,
            output_dir=family_fig_root,
        )


if __name__ == "__main__":
    main()
