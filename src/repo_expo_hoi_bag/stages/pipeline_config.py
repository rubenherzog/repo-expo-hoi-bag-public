from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
DEFAULT_SMOKE_CONFIG = PROJECT_ROOT / "config" / "smoke.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the root.")
    return data


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def env_list(env_key: str, default: Union[str, list[str], None] = None) -> list[str]:
    raw = os.environ.get(env_key, "")
    if raw.strip():
        return [tok.strip() for tok in raw.split(",") if tok.strip()]
    if isinstance(default, list):
        return default
    if isinstance(default, str):
        return [tok.strip() for tok in default.split(",") if tok.strip()]
    return []


def load_pipeline_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else Path(os.environ.get("V3_PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG))
    return _load_yaml(path)


def load_smoke_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else Path(os.environ.get("V3_SMOKE_CONFIG", DEFAULT_SMOKE_CONFIG))
    return _load_yaml(path)


def load_stage_config(
    stage_name: str,
    config_path: Optional[Union[Path, str]] = None,
    smoke: bool = False,
    smoke_config_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    pipeline_cfg = load_pipeline_config(config_path)
    stage_cfg = merge_dicts(pipeline_cfg.get("common", {}), pipeline_cfg.get(stage_name, {}))
    if smoke:
        smoke_cfg = load_smoke_config(smoke_config_path)
        stage_cfg = merge_dicts(stage_cfg, smoke_cfg.get("defaults", {}))
        stage_cfg = merge_dicts(stage_cfg, smoke_cfg.get(stage_name, {}))
    return stage_cfg
