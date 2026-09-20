#!/usr/bin/env python3
"""Cross-evaluate completed global XGBoost HPO configurations at three model sizes."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
import yaml

from loco_fusion_matrix_engine import align_exposome_to_greedy_features, build_model_df, load_greedy_feature_names
from loco_fusion_matrix_engine import prepare_bag_context
from oinfo_bag_ladder.config import ANALYSIS_CFG, BAG_TARGETS, CV_CFG, DATA_PATHS, EARLY_STOP_CFG
from oinfo_bag_ladder.io_utils import stable_hash
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
from xgb_nested_loco_tuning import build_global_loco_fold_designs, deterministic_seed, _evaluate_trial


STAGE_NAME = "xgb_hpo_cross_test"
SCOPES = ("baseline", "single_exposure", "full_exposome")
PRIMARY_BAGS = ("structural", "functional")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def resolve_cross_artifacts(checkout: Path, raw: str) -> list[Path]:
    """Resolve distinct completed selected-config artifacts beneath local outputs."""
    root = (Path(checkout).resolve() / "outputs" / "xgb_nested_loco_tuning").resolve()
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError("XGB_HPO_CROSS_ARTIFACTS must contain exactly three selected-config paths")
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        path = path if path.is_absolute() else Path(checkout) / path
        path = path.resolve()
        if path.parent.parent != root or path.name != "selected_xgb_configs.json" or not path.is_file():
            raise ValueError(f"Cross-test artifact must be a completed local tuning selected-config file: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("XGB_HPO_CROSS_ARTIFACTS must contain three distinct artifacts")
    return paths


def _policy(checkout: Path, variant: str) -> dict:
    config = yaml.safe_load((checkout / "config" / "country_exclusions.yaml").read_text())
    item = config["bag"]["variants"][variant]
    return {"variant": variant, "exclude_countries": list(item["exclude_countries"]), "exclude_diagnosis": list(item["exclude_diagnosis"])}


def _load_source(path: Path, policy: dict, rung_id: str) -> dict:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    observed = manifest["two_phase_country_config"]
    if (observed.get("variant"), list(observed.get("bag_exclude_countries", [])), list(observed.get("bag_exclude_diagnosis", []))) != (
        policy["variant"], policy["exclude_countries"], policy["exclude_diagnosis"]
    ):
        raise ValueError(f"Artifact country policy does not match requested cross-test policy: {path}")
    payload = json.loads(path.read_text())
    selected = {row["bag_target"]: row for row in payload["selected"] if row["rung_id"] == rung_id}
    if set(selected) != set(PRIMARY_BAGS):
        raise ValueError(f"Artifact must provide one {rung_id} global configuration for each BAG: {path}")
    if any(row["outer_country"] != "__global__" for row in selected.values()):
        raise ValueError(f"Artifact must be global LOCO tuning: {path}")
    # Completed full-exposome studies predate the explicit feature_scope field;
    # absence is therefore the backwards-compatible canonical full panel.
    return {"label": str(manifest.get("feature_scope", "full_exposome")), "artifact": str(path), "sha256": _sha256(path), "selected": selected}


def main() -> None:
    runtime = Path(os.environ["REPRO_DATA_ROOT"]).resolve()
    checkout = Path(os.environ["REPO_CHECKOUT_ROOT"]).resolve()
    paths = resolve_cross_artifacts(checkout, os.environ.get("XGB_HPO_CROSS_ARTIFACTS", ""))
    variant = os.environ.get("XGB_HPO_CROSS_VARIANT", "a").strip() or "a"
    rung_id = os.environ.get("XGB_HPO_CROSS_RUNG", "xgb_tree_d3").strip()
    n_jobs = int(os.environ.get("XGB_HPO_CROSS_N_JOBS", "20"))
    if n_jobs != 20:
        raise ValueError("XGB_HPO_CROSS_N_JOBS must be exactly 20 for this controlled local cross-test")
    policy = _policy(checkout, variant)
    sources = [_load_source(path, policy, rung_id) for path in paths]
    if set(source["label"] for source in sources) != set(SCOPES):
        raise ValueError("Cross-test artifacts must be exactly the baseline, single_exposure, and full_exposome studies")
    rung = next(spec for spec in get_rung_specs() if spec["rung_id"] == rung_id)
    raw = pd.read_csv(DATA_PATHS["input_csv"], low_memory=False)
    raw["N_MEGA"] = raw["N_MEGA"].astype(str).str.strip()
    raw = raw.drop_duplicates(subset="N_MEGA", keep="first").reset_index(drop=True)
    features = load_greedy_feature_names(Path("data/exposome_feature_names.csv"))
    exposome = align_exposome_to_greedy_features(raw, features).apply(pd.to_numeric, errors="coerce")
    if exposome.isna().any().any():
        raise ValueError("Cross-test complete exposome contains missing values")
    analysis_cfg = {**ANALYSIS_CFG, "exclude_countries": policy["exclude_countries"], "exclude_diagnosis": policy["exclude_diagnosis"]}
    model_df = build_model_df(raw, exposome, analysis_cfg)
    feature_names = exposome.columns.astype(str).tolist()
    contexts = {
        bag: dict(prepare_bag_context(model_df, BAG_TARGETS[bag], bag, analysis_cfg, dict(CV_CFG), feature_names), exposome_features=feature_names)
        for bag in PRIMARY_BAGS
    }
    provenance = {"stage": STAGE_NAME, "rung_id": rung_id, "policy": policy, "sources": [{k: v for k, v in s.items() if k != "selected"} for s in sources], "n_jobs": n_jobs, "stage_sha256": _sha256(Path(__file__))}
    root = runtime / "results" / "sensitivity" / STAGE_NAME
    outdir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + stable_hash(provenance)[:16])
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    def evaluate(source: dict, target_scope: str, bag: str) -> dict:
        context = contexts[bag]
        designs = build_global_loco_fold_designs(context, analysis_cfg, dict(EARLY_STOP_CFG), deterministic_seed(20260304, bag, "__global__"))
        result = _evaluate_trial(context, designs, build_xgb_cfg_for_rung(rung), source["selected"][bag]["params"], deterministic_seed(20260304, bag, "__global__", rung_id), "mean_country_mse", fold_n_jobs=5, xgb_nthread=2, feature_scope=target_scope)
        return {"source_scope": source["label"], "target_scope": target_scope, "bag_target": bag, "rung_id": rung_id, "source_artifact": source["artifact"], **result}

    tasks = [(source, target, bag) for source in sources for target in SCOPES for bag in PRIMARY_BAGS]
    # Every task consumes exactly 10 cores (five country folds × two XGBoost
    # threads); two concurrent tasks therefore exhaust, but never exceed, 20.
    rows = Parallel(n_jobs=2, backend="loky", verbose=10)(delayed(evaluate)(*task) for task in tasks)
    table = pd.DataFrame(rows)
    table.to_parquet(outdir / "cross_test_diagnostics.parquet", index=False)
    table[["source_scope", "target_scope", "bag_target", "rung_id", "mean_country_mse", "median_country_mse", "inner_global_oof_r2", "inner_country_balanced_oof_r2", "complete_country_coverage"]].to_csv(outdir / "cross_test_summary.csv", index=False)
    print(f"Completed {STAGE_NAME}: {outdir}", flush=True)


if __name__ == "__main__":
    main()
