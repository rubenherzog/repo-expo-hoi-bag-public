from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from loco_fusion_matrix_engine import pin_blas_threads
from oinfo_bag_ladder.io_utils import load_json, now_iso, save_json, stable_hash
from xgb_loco_engine import require_xgboost, run_xgb_loco_stage


KEEP_XGB_FILES = {
    "xgb_model_vs_baseline_global.csv",
    "xgb_model_vs_baseline_country.csv",
    "xgb_summary_with_calibration_complexity.csv",
    "stage_manifest.json",
}


def _cleanup_xgb_stage_dir(stage_dir: Path) -> None:
    for path in stage_dir.iterdir():
        if path.is_file() and path.name not in KEEP_XGB_FILES:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _required_paths(stage_dir: Path) -> dict[str, Path]:
    return {
        "summary": stage_dir / "xgb_summary_with_calibration_complexity.csv",
        "global": stage_dir / "xgb_model_vs_baseline_global.csv",
        "country": stage_dir / "xgb_model_vs_baseline_country.csv",
        "manifest": stage_dir / "stage_manifest.json",
    }


def run_or_load_xgb_rung(
    experiment_row: dict,
    rung_spec: dict,
    stage_dir: Path,
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    analysis_cfg: dict,
    cv_cfg: dict,
    perf_cfg: dict,
    xgb_cfg: dict,
    early_stop_cfg: dict,
    force_recompute: bool = False,
) -> dict[str, pd.DataFrame]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths = _required_paths(stage_dir)
    stage_manifest = load_json(paths["manifest"], {})
    stage_hash = stable_hash(
        {
            "experiment": experiment_row,
            "rung": rung_spec,
            "analysis_cfg": analysis_cfg,
            "cv_cfg": cv_cfg,
            "perf_cfg": perf_cfg,
            "xgb_cfg": xgb_cfg,
            "early_stop_cfg": early_stop_cfg,
            "candidate_ids": candidate_df["feature_id"].astype(str).tolist(),
        }
    )
    files_exist = all(paths[key].exists() for key in ("summary", "global", "country"))
    cache_hit = (not force_recompute) and files_exist and stage_manifest.get("stage_hash") == stage_hash

    if cache_hit:
        return {
            "summary_df": pd.read_csv(paths["summary"]),
            "global_df": pd.read_csv(paths["global"]),
            "country_df": pd.read_csv(paths["country"]),
        }

    storage_cfg = {
        "write_xgb_base_predictions": False,
        "write_xgb_model_predictions": False,
        "keep_xgb_base_predictions_in_memory": False,
        "keep_xgb_model_predictions_in_memory": False,
    }
    compare_cfg = {
        "compute_baseline_oof": True,
        "compute_paired_delta": False,
        "compute_paired_after_all_bags": False,
        "save_thin_paired_predictions": False,
    }

    # Snapshot existing CSV data before re-running so we can merge afterwards
    existing_summary: pd.DataFrame | None = None
    existing_global: pd.DataFrame | None = None
    existing_country: pd.DataFrame | None = None
    if not force_recompute and files_exist:
        try:
            existing_summary = pd.read_csv(paths["summary"])
            existing_global = pd.read_csv(paths["global"])
            existing_country = pd.read_csv(paths["country"])
        except Exception:
            existing_summary = existing_global = existing_country = None

    require_xgboost()
    pin_blas_threads(int(perf_cfg.get("blas_threads", 1)))
    data = run_xgb_loco_stage(
        model_df=model_df,
        candidate_df=candidate_df,
        exposome_cols=exposome_cols,
        bag_targets={experiment_row["bag_target"]: experiment_row["bag_target_column"]},
        outdir=stage_dir,
        analysis_cfg=analysis_cfg,
        cv_cfg=cv_cfg,
        perf_cfg=perf_cfg,
        xgb_cfg=xgb_cfg,
        early_stop_cfg=early_stop_cfg,
        enable_progress=True,
        storage_cfg=storage_cfg,
        compare_cfg=compare_cfg,
    )

    _cleanup_xgb_stage_dir(stage_dir)

    new_summary = data.get("summary_with_calib_df", pd.DataFrame()).copy()
    new_global = data.get("model_vs_base_global_df", pd.DataFrame()).copy()
    new_country = data.get("model_vs_base_country_df", pd.DataFrame()).copy()
    del data
    gc.collect()

    # Merge new results with existing CSV data to accumulate candidates across runs
    def _merge_csv(new_df: pd.DataFrame, existing_df: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame:
        if existing_df is None or existing_df.empty:
            return new_df
        if new_df.empty:
            return existing_df
        combined = pd.concat([new_df, existing_df], ignore_index=True)
        actual_keys = [k for k in keys if k in combined.columns]
        if actual_keys:
            return combined.drop_duplicates(subset=actual_keys, keep="first").reset_index(drop=True)
        return combined.drop_duplicates().reset_index(drop=True)

    merged_summary = _merge_csv(new_summary, existing_summary, ["bag_target", "model_id"])
    merged_global = _merge_csv(new_global, existing_global, ["bag_target", "model_id"])
    merged_country = _merge_csv(new_country, existing_country, ["bag_target", "model_id", "fold_country"])

    merged_summary.to_csv(paths["summary"], index=False)
    merged_global.to_csv(paths["global"], index=False)
    merged_country.to_csv(paths["country"], index=False)

    manifest = {
        "timestamp_utc": now_iso(),
        "experiment_id": experiment_row["experiment_id"],
        "rung_id": rung_spec["rung_id"],
        "stage_hash": stage_hash,
        "candidate_count": int(len(merged_summary)),
        "files": {key: str(path) for key, path in paths.items()},
    }
    save_json(paths["manifest"], manifest)

    return {
        "summary_df": merged_summary,
        "global_df": merged_global,
        "country_df": merged_country,
    }
