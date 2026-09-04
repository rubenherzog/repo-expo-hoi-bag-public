from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Union

from scripts.pipeline_config import load_pipeline_config, load_smoke_config, merge_dicts


# XGBoost max_depth values for each tree-complexity rung.
RUNG_DEPTH: dict[str, int] = {
    "xgb_tree_d1": 1,
    "xgb_tree_d2": 2,
    "xgb_tree_d3": 3,
}


def env_bool(env_key: str, default: bool = False) -> bool:
    raw = os.environ.get(env_key, "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes")
    return default


def env_int(env_key: str, default: int | None = None) -> int | None:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"Environment variable {env_key} must be an integer: {raw!r}")
    return default


def env_list(env_key: str, default: Union[str, Sequence[str], None] = None) -> list[str]:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return [tok.strip() for tok in raw.split(",") if tok.strip()]
    if isinstance(default, str):
        return [tok.strip() for tok in default.split(",") if tok.strip()]
    if isinstance(default, Sequence):
        return [tok for tok in default if isinstance(tok, str) and tok.strip()]
    return []


def default_cv_cfg() -> Dict[str, Any]:
    return {
        "split_col": "country_clean",
        "min_country_size_test": 0,
        "unseen_diag_policy": "drop_test_rows",
    }


def default_xgb_cfg() -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 15,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "gamma": 0.5,
        "n_estimators": 4000,
        "early_stopping_rounds": 100,
        "random_state": 20260304,
        "nthread": 1,
        "missing": float("nan"),
    }


def normalize_xgb_cfg(xgb_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(xgb_cfg)
    if "missing" in cfg:
        missing = cfg["missing"]
        if missing is None:
            cfg.pop("missing", None)
        elif isinstance(missing, str):
            if missing.strip().lower() in {"nan", "none", ""}:
                cfg.pop("missing", None)
            else:
                try:
                    cfg["missing"] = float(missing)
                except ValueError:
                    raise ValueError(
                        f"Invalid XGBoost missing value in config: {missing!r}. "
                        "Use a numeric value, None, or omit the parameter."
                    )
    return cfg


def default_perf_cfg(stage_cfg: Dict[str, Any], env_jobs_key: str = "N_JOBS", default_n_jobs: int = 8) -> Dict[str, Any]:
    perf_cfg = dict(stage_cfg.get("perf_cfg", {}))
    perf_cfg["n_jobs"] = env_int(env_jobs_key, int(perf_cfg.get("n_jobs", default_n_jobs)))
    perf_cfg.setdefault("backend", "loky")
    perf_cfg.setdefault("blas_threads", 1)
    return perf_cfg


def default_paths(stage_cfg: Dict[str, Any]) -> Dict[str, str]:
    greedy_root = Path(os.environ.get("V3_GREEDY_ROOT", stage_cfg.get("greedy_root", "outputs/greedy")))
    return {
        "input_csv": os.environ.get(
            "V3_RAW_PATH",
            stage_cfg.get("input_csv", "data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"),
        ),
        "greedy_topk_csv": str(greedy_root / "greedy_topk_by_objective_order.csv"),
        "greedy_feature_names_csv": str(greedy_root / "exposome_feature_names.csv"),
    }


def build_analysis_cfg(stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    analysis_cfg = dict(stage_cfg.get("analysis_cfg", {}))
    analysis_cfg["exclude_diagnosis"] = env_list(
        "V3_EXCLUDE_DIAGNOSIS",
        analysis_cfg.get("exclude_diagnosis", "Other,AFM"),
    )
    analysis_cfg["exclude_countries"] = env_list(
        "V3_EXCLUDE_COUNTRIES",
        analysis_cfg.get("exclude_countries", "North Macedonia,New Zeland,Belgium"),
    )
    return analysis_cfg


def load_stage_config(
    stage_name: str,
    smoke: bool = False,
    config_path: Optional[Union[Path, str]] = None,
    smoke_config_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    pipeline_cfg = load_pipeline_config(config_path)
    stage_cfg = merge_dicts(pipeline_cfg.get("common", {}), pipeline_cfg.get(stage_name, {}))
    if smoke:
        smoke_cfg = load_smoke_config(smoke_config_path)
        stage_cfg = merge_dicts(stage_cfg, smoke_cfg.get("defaults", {}))
        stage_cfg = merge_dicts(stage_cfg, smoke_cfg.get(stage_name, {}))

    if "xgb_cfg" in stage_cfg and isinstance(stage_cfg["xgb_cfg"], dict):
        stage_cfg["xgb_cfg"] = normalize_xgb_cfg(stage_cfg["xgb_cfg"])

    return stage_cfg


def load_pipeline_driver_config(
    config_path: Optional[Union[Path, str]] = None,
    smoke_config_path: Optional[Union[Path, str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load pipeline and smoke config files for a driver script.

    Returns:
        pipeline_config, smoke_config, pipeline_defaults, smoke_defaults
    """
    pipeline_cfg = load_pipeline_config(config_path)
    smoke_cfg = load_smoke_config(smoke_config_path)
    return (
        pipeline_cfg,
        smoke_cfg,
        pipeline_cfg.get("defaults", {}),
        smoke_cfg.get("defaults", {}),
    )
