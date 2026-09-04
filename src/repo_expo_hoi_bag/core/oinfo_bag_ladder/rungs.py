from __future__ import annotations

from copy import deepcopy

from oinfo_bag_ladder.config import BASE_XGB_CFG


RUNG_SPECS = [
    {
        "rung_id": "ols",
        "display_label": "OLS",
        "model_family": "ols",
    },
    {
        "rung_id": "xgb_tree_d1",
        "display_label": "XGB d1",
        "model_family": "xgb",
        "xgb_overrides": {
            "booster": "gbtree",
            "max_depth": 1,
        },
    },
    {
        "rung_id": "xgb_tree_d2",
        "display_label": "XGB d2",
        "model_family": "xgb",
        "xgb_overrides": {
            "booster": "gbtree",
            "max_depth": 2,
        },
    },
    {
        "rung_id": "xgb_tree_d3",
        "display_label": "XGB d3",
        "model_family": "xgb",
        "xgb_overrides": {
            "booster": "gbtree",
            "max_depth": 3,
        },
    },
]

RUNG_ORDER = [spec["rung_id"] for spec in RUNG_SPECS]
RUNG_LABELS = {spec["rung_id"]: spec["display_label"] for spec in RUNG_SPECS}


def get_rung_specs() -> list[dict]:
    return [dict(spec) for spec in RUNG_SPECS]


def build_xgb_cfg_for_rung(rung_spec: dict) -> dict:
    cfg = deepcopy(BASE_XGB_CFG)
    cfg.update(rung_spec.get("xgb_overrides", {}))

    if cfg.get("booster") == "gblinear":
        for key in ["max_depth", "min_child_weight", "subsample", "colsample_bytree", "gamma", "tree_method"]:
            cfg.pop(key, None)
    else:
        cfg.setdefault("booster", "gbtree")
        cfg.setdefault("tree_method", "hist")
        if cfg.get("missing", None) is None:
            cfg.pop("missing", None)
    return cfg
