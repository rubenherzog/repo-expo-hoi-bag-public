from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from loco_fusion_matrix_engine import pin_blas_threads, run_loco_stage
from oinfo_bag_ladder.io_utils import load_json, now_iso, save_json, stable_hash


KEEP_OLS_FILES = {
    "loco_model_vs_baseline_global.csv",
    "loco_model_vs_baseline_country.csv",
    "loco_oof_predictions.csv",
    "loco_summary_with_calibration_complexity.csv",
    "stage_manifest.json",
}


def _cleanup_ols_stage_dir(stage_dir: Path) -> None:
    for path in stage_dir.iterdir():
        if path.is_file() and path.name not in KEEP_OLS_FILES:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _required_paths(stage_dir: Path) -> dict[str, Path]:
    return {
        "summary": stage_dir / "loco_summary_with_calibration_complexity.csv",
        "global": stage_dir / "loco_model_vs_baseline_global.csv",
        "country": stage_dir / "loco_model_vs_baseline_country.csv",
        "manifest": stage_dir / "stage_manifest.json",
    }


def run_or_load_ols_rung(
    experiment_row: dict,
    rung_spec: dict,
    stage_dir: Path,
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    analysis_cfg: dict,
    cv_cfg: dict,
    perf_cfg: dict,
    matrix_cfg: dict,
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
            "matrix_cfg": matrix_cfg,
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

    perf_cfg_local = dict(perf_cfg)
    perf_cfg_local["enable_resume"] = False
    storage_cfg = {
        "write_checkpoint_files": False,
        "write_per_bag_prediction_checkpoints": bool(matrix_cfg.get("write_loco_predictions", False)),
        "write_per_bag_identity_map": False,
        "write_loco_predictions": bool(matrix_cfg.get("write_loco_predictions", False)),
        "write_compat_loco_predictions": False,
        "keep_base_predictions_in_memory": bool(matrix_cfg.get("write_loco_predictions", False)),
        "keep_baseline_predictions_in_memory": False,
        "write_baseline_prediction_checkpoints": False,
        "write_baseline_predictions": False,
    }
    compare_cfg = {
        "compute_baseline_oof": True,
        "compute_paired_delta": False,
        "compute_paired_after_all_bags": False,
        "save_thin_paired_predictions": False,
    }

    pin_blas_threads(int(perf_cfg_local.get("blas_threads", 1)))
    data = run_loco_stage(
        model_df=model_df,
        candidate_df=candidate_df,
        exposome_cols=exposome_cols,
        bag_targets={experiment_row["bag_target"]: experiment_row["bag_target_column"]},
        outdir=stage_dir,
        analysis_cfg=analysis_cfg,
        cv_cfg=cv_cfg,
        perf_cfg=perf_cfg_local,
        matrix_cfg=matrix_cfg,
        enable_progress=True,
        storage_cfg=storage_cfg,
        compare_cfg=compare_cfg,
    )

    _cleanup_ols_stage_dir(stage_dir)
    manifest = {
        "timestamp_utc": now_iso(),
        "experiment_id": experiment_row["experiment_id"],
        "rung_id": rung_spec["rung_id"],
        "stage_hash": stage_hash,
        "candidate_count": int(len(candidate_df)),
        "files": {key: str(path) for key, path in paths.items()},
    }
    save_json(paths["manifest"], manifest)

    out = {
        "summary_df": data.get("loco_summary_with_calibration_complexity_df", pd.DataFrame()).copy(),
        "global_df": data.get("model_vs_base_global_df", pd.DataFrame()).copy(),
        "country_df": data.get("model_vs_base_country_df", pd.DataFrame()).copy(),
    }
    del data
    gc.collect()
    return out
