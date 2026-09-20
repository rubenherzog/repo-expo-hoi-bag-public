#!/usr/bin/env python3
"""Run one resumable main-paper re-analysis scheduler unit.

There is deliberately no HPO code here: all XGBoost parameters are read from
the four released, immutable selection artifacts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import types

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "src" / "repo_expo_hoi_bag" / "core", ROOT / "src" / "repo_expo_hoi_bag" / "stages"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# Legacy numerical helpers import ``scripts.*`` because they normally run from
# the copied compatibility layout. Expose the active local stages directory as
# that package without making any source checkout a runtime dependency.
legacy_scripts = types.ModuleType("scripts")
legacy_scripts.__path__ = [str(ROOT / "src" / "repo_expo_hoi_bag" / "stages")]
sys.modules["scripts"] = legacy_scripts

from repo_expo_hoi_bag.analysis.paper_reanalysis import (  # noqa: E402
    PRIMARY_BAGS, XGB_RUNGS, checkpoint_identity, load_frozen_hpo, load_main_policy,
    sha256_file, validate_candidate_pool,
)
from repo_expo_hoi_bag.analysis.xgb_frozen_top50 import baseline_candidate, single_exposure_candidates  # noqa: E402
from loco_fusion_matrix_engine import align_exposome_to_greedy_features, build_model_df, load_greedy_feature_names, prepare_bag_context, target_map  # noqa: E402
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs  # noqa: E402
from sensitivity_common import _fit_candidate  # noqa: E402
from xgb_loco_engine import _build_base_fold_mats  # noqa: E402
from xgb_nested_loco_tuning import resolve_tuned_fold_configs  # noqa: E402


ARTIFACTS = {
    "baseline": ("PAPER_REANALYSIS_BASELINE_HPO", "baseline"),
    "single": ("PAPER_REANALYSIS_SINGLE_HPO", "single_exposure"),
    "k10": ("PAPER_REANALYSIS_K10_HPO", "domain_balanced_k10"),
    "k63": ("PAPER_REANALYSIS_K63_HPO", "full_exposome"),
}
ANALYSIS_CFG = {
    "exclude_countries": ["France", "Italy", "Egypt", "Greece", "Poland"],
    "exclude_diagnosis": ["Other", "AFM", "MCI"],
    "include_sex": True, "include_diagnosis": True, "include_year": True,
    "min_n_obs_for_metrics": 5,
}
CV_CFG = {"split_col": "country_clean", "min_country_size_test": 0, "unseen_diag_policy": "drop_test_rows"}
EARLY_STOP_CFG = {"val_country_frac": 0.20, "val_country_min": 1}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("xgb", "ols"), required=True)
    p.add_argument("--bags", nargs="+", choices=PRIMARY_BAGS, required=True)
    p.add_argument("--rung", choices=XGB_RUNGS)
    p.add_argument("--run-id", required=True)
    p.add_argument("--n-jobs", type=int, default=40)
    p.add_argument("--chunk-size", type=int, default=40)
    p.add_argument("--preflight", action="store_true", help="Validate inputs and folds without fitting or writing.")
    p.add_argument("--accept-legacy-checkpoints", action="store_true",
                   help="One-time migration for verified pre-40-model chunk markers.")
    return p.parse_args()


def _runtime(run_id: str) -> tuple[Path, Path]:
    root = Path(os.environ["REPRO_DATA_ROOT"]).resolve()
    results = root / "results" / "analysis_runs" / run_id
    work = root / "work" / "analysis_runs" / run_id
    if results.exists() or work.exists():
        # Existing namespaces are allowed only for genuine resumptions; their
        # per-unit manifest below validates the exact scientific identity.
        return results, work
    results.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    return results, work


def _load_model() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv", low_memory=False)
    names = load_greedy_feature_names(ROOT / "data" / "metadata" / "exposome_feature_names.csv")
    exposome = align_exposome_to_greedy_features(raw, names).apply(pd.to_numeric, errors="coerce")
    if exposome.isna().any().any():
        raise ValueError("The public complete-case cohort has missing exposome values")
    return build_model_df(raw, exposome.reset_index(drop=True), ANALYSIS_CFG), names


def _candidate_frame(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["nplet_vars"] = out["nplet_vars_str"].fillna("").astype(str).map(lambda value: [v for v in value.split("|") if v])
    out["candidate_family"] = "official_deduplicated_pool"
    out["source_label"] = "paper_reference"
    return out


def _build_context(model_df: pd.DataFrame, bag: str, features: list[str]) -> tuple[dict, dict]:
    context = prepare_bag_context(model_df, target_map(bag)[bag], bag, ANALYSIS_CFG, CV_CFG, features)
    folds = {country: _build_base_fold_mats(context, country, ANALYSIS_CFG, EARLY_STOP_CFG, seed=20260304 + index)
             for index, country in enumerate(context["countries"])}
    return context, folds


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _completed_chunks(work_dir: Path, identity: str, expected_ids: set[str], *, accept_legacy: bool) -> dict[str, dict]:
    """Load completed chunks from any prior chunk size without rerunning them."""
    completed: dict[str, dict] = {}
    for marker in sorted(work_dir.glob("chunk-*.json")):
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt checkpoint marker: {marker}") from exc
        legacy_marker = bool(re.fullmatch(r"chunk-\d{4}\.json", marker.name))
        if saved.get("identity") != identity and not (accept_legacy and legacy_marker):
            raise ValueError(f"Incompatible checkpoint identity: {marker}")
        if saved.get("status") != "complete":
            continue
        members = [str(value) for value in saved.get("candidate_ids", [])]
        if not members or not set(members).issubset(expected_ids):
            raise ValueError(f"Invalid checkpoint candidate membership: {marker}")
        if any(member in completed for member in members):
            raise ValueError(f"Duplicate completed candidate in checkpoints: {marker}")
        files = saved.get("files", {})
        if not all(Path(files.get(key, "")).is_file() for key in ("global", "country")):
            raise ValueError(f"Checkpoint data files missing for {marker}")
        scored_ids = pd.read_csv(files["global"])["candidate_id"].astype(str).tolist()
        if set(scored_ids) != set(members) or len(scored_ids) != len(set(scored_ids)):
            raise ValueError(f"Checkpoint result membership mismatch: {marker}")
        for member in members:
            completed[member] = saved
    return completed


def _write_resized_chunk(work_dir: Path, chunk_size: int, index: int, identity: str,
                         candidate_ids: list[str], files: dict[str, str]) -> Path:
    """Use size-qualified markers so a resumed run never collides with old chunks."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"chunk-{chunk_size:04d}-{index:04d}.json"
    _atomic_json({"status": "complete", "identity": identity, "candidate_ids": candidate_ids,
                  "files": files}, path)
    return path


def _evaluate_arm(*, arm: str, candidates: pd.DataFrame, context: dict, folds: dict, features: list[str],
                  rung: str, cfg: dict | None, fold_cfg: dict | None, result_dir: Path, work_dir: Path,
                  identity: str, n_jobs: int, chunk_size: int, accept_legacy: bool) -> None:
    records = candidates.drop_duplicates("predictors_identity", keep="first").to_dict("records")
    ids = [str(row["candidate_id"]) for row in records]
    completed = _completed_chunks(work_dir, identity, set(ids), accept_legacy=accept_legacy)
    pending_rows = [row for row in records if str(row["candidate_id"]) not in completed]
    for chunk_index, start in enumerate(range(0, len(pending_rows), chunk_size)):
        rows = pending_rows[start:start + chunk_size]
        member_ids = [str(row["candidate_id"]) for row in rows]
        fitted = Parallel(n_jobs=min(n_jobs, len(rows)), backend="loky")(
            delayed(_fit_candidate)(row, context=context, fold_designs=folds, exposome_cols=features,
                                    rung_id=rung, xgb_cfg=cfg, fold_xgb_cfgs=fold_cfg)
            for row in rows
        ) if len(rows) > 1 else [
            _fit_candidate(rows[0], context=context, fold_designs=folds, exposome_cols=features,
                           rung_id=rung, xgb_cfg=cfg, fold_xgb_cfgs=fold_cfg)
        ]
        global_path = result_dir / "chunks" / f"chunk-{chunk_size:04d}-{chunk_index:04d}-global.csv"
        country_path = result_dir / "chunks" / f"chunk-{chunk_size:04d}-{chunk_index:04d}-country.csv"
        _atomic_csv(pd.DataFrame([item[0] for item in fitted]), global_path)
        _atomic_csv(pd.concat([item[1] for item in fitted], ignore_index=True), country_path)
        _write_resized_chunk(work_dir, chunk_size, chunk_index, identity, member_ids,
                             {"global": str(global_path), "country": str(country_path)})
        print(f"[{arm}] completed chunk {chunk_index} ({len(member_ids)} models)", flush=True)
    markers = sorted(work_dir.glob("chunk-*.json"))
    global_parts, country_parts = [], []
    for marker in markers:
        saved = json.loads(marker.read_text())
        legacy_marker = bool(re.fullmatch(r"chunk-\d{4}\.json", marker.name))
        if (saved.get("identity") != identity and not (accept_legacy and legacy_marker)) or saved.get("status") != "complete":
            raise ValueError(f"Incompatible checkpoint marker: {marker}")
        global_parts.append(pd.read_csv(saved["files"]["global"]))
        country_parts.append(pd.read_csv(saved["files"]["country"]))
    merged_global = pd.concat(global_parts, ignore_index=True)
    if set(merged_global["candidate_id"].astype(str)) != set(ids) or merged_global["candidate_id"].astype(str).duplicated().any():
        raise ValueError(f"Incomplete or duplicate completed candidate set for {arm}")
    _atomic_csv(merged_global, result_dir / "metrics_global.csv")
    _atomic_csv(pd.concat(country_parts, ignore_index=True), result_dir / "metrics_country.csv")


def _run_bag(mode: str, bag: str, rung: str, artifacts: dict, results: Path, work: Path, n_jobs: int, chunk_size: int,
             accept_legacy: bool) -> None:
    reference = Path(os.environ["PAPER_REANALYSIS_REFERENCE_ROOT"]).resolve()
    registry_path = reference / "candidate_registry.parquet"
    pool = validate_candidate_pool(pd.read_parquet(registry_path), bag)
    model_df, features = _load_model()
    context, folds = _build_context(model_df, bag, features)
    pool_hash = sha256_file(registry_path)
    code_hash = sha256_file(Path(__file__))
    arms = {"baseline": baseline_candidate(), "single": single_exposure_candidates(features), "candidates": _candidate_frame(pool)}
    rung_specs = {spec["rung_id"]: spec for spec in get_rung_specs()}
    for arm, candidates in arms.items():
        label = arm if mode == "ols" else ("k10" if arm == "candidates" else arm)
        configs = [(label, None, None)] if mode == "ols" else (
            [(label, artifacts[label]["rows"][(bag, rung, "__global__")], artifacts[label]["sha256"])] if arm != "candidates" else
            [("k10", artifacts["k10"]["rows"][(bag, rung, "__global__")], artifacts["k10"]["sha256"]),
             ("k63", artifacts["k63"]["rows"][(bag, rung, "__global__")], artifacts["k63"]["sha256"])])
        for config_label, selected, hpo_hash in configs:
            base_cfg = None if mode == "ols" else build_xgb_cfg_for_rung(rung_specs[rung])
            fold_cfg = None if selected is None else {country: {**base_cfg, **dict(selected["params"])} for country in context["countries"]}
            # k10 and k63 are immutable delivery namespaces. Baseline/single
            # are evaluated once and delivered with k10; k63 records the same
            # provenance in its unit manifest rather than refitting them.
            delivery_root = results
            delivery_work = work
            if mode == "xgb" and config_label == "k63":
                delivery_root = results.parent / results.name.replace("_k10", "_k63")
                delivery_work = work.parent / work.name.replace("_k10", "_k63")
                delivery_root.mkdir(parents=True, exist_ok=True)
                delivery_work.mkdir(parents=True, exist_ok=True)
            # Previous OLS workers wrote candidate chunks to an ``ols`` arm.
            # Preserve that exact location solely to reuse those completed
            # candidates; new baseline/single arms are isolated.
            storage_label = "ols" if mode == "ols" and config_label == "candidates" else config_label
            arm_result = delivery_root / mode / bag / ("ols" if mode == "ols" else rung) / storage_label
            arm_work = delivery_work / mode / bag / ("ols" if mode == "ols" else rung) / storage_label
            identity = checkpoint_identity(code_hash=code_hash, pool_hash=pool_hash, policy_hash=artifacts.get("policy_hash", "ols"),
                                           hpo_hash=hpo_hash or "ols", bag=bag, rung=("ols" if mode == "ols" else rung))
            _evaluate_arm(arm=config_label, candidates=candidates, context=context, folds=folds, features=features,
                          rung=("ols" if mode == "ols" else rung), cfg=base_cfg, fold_cfg=fold_cfg,
                          result_dir=arm_result, work_dir=arm_work, identity=identity, n_jobs=n_jobs, chunk_size=chunk_size,
                          accept_legacy=accept_legacy)
            _atomic_json({
                "run_id": delivery_root.name, "mode": mode, "bag": bag,
                "rung": "ols" if mode == "ols" else rung, "arm": config_label,
                "checkpoint_identity": identity, "candidate_pool": str(registry_path),
                "candidate_pool_sha256": pool_hash, "country_policy_sha256": artifacts.get("policy_hash"),
                "frozen_hpo": None if selected is None else {
                    "path": artifacts[config_label]["path"], "sha256": hpo_hash,
                    "selected_params": selected["params"],
                },
                "countries": context["countries"], "n_candidates": int(len(candidates)),
                "chunk_size": chunk_size, "n_jobs": n_jobs,
            }, arm_result / "manifest.json")


def _preflight_bag(bag: str) -> None:
    """Exercise all setup dependencies without fitting a model or writing state."""
    reference = Path(os.environ["PAPER_REANALYSIS_REFERENCE_ROOT"]).resolve()
    pool = validate_candidate_pool(pd.read_parquet(reference / "candidate_registry.parquet"), bag)
    model_df, features = _load_model()
    context, folds = _build_context(model_df, bag, features)
    if not context["countries"] or len(folds) != len(context["countries"]):
        raise ValueError(f"Invalid LOCO fold construction for {bag}")
    print(f"[preflight] bag={bag} candidates={len(pool)} countries={len(context['countries'])} folds={len(folds)}", flush=True)


def _worker_import_probe() -> str:
    """Executed in a fresh loky process to validate worker-side imports."""
    from sensitivity_common import _fit_candidate as probe  # noqa: F401
    return "ok"


def _preflight_workers() -> None:
    checks = Parallel(n_jobs=2, backend="loky")(delayed(_worker_import_probe)() for _ in range(2))
    if checks != ["ok", "ok"]:
        raise ValueError("Worker import preflight did not complete")
    print("[preflight] loky worker imports=ok", flush=True)


def main() -> None:
    args = _args()
    if args.mode == "xgb" and not args.rung:
        raise ValueError("--rung is required in xgb mode")
    if args.mode == "ols" and args.rung:
        raise ValueError("OLS has no XGBoost rung")
    policy, policy_hash = load_main_policy(ROOT / "config" / "country_exclusions.yaml")
    artifacts = {"policy_hash": policy_hash}
    if args.mode == "xgb":
        for label, (env_name, scope) in ARTIFACTS.items():
            # A deliberate frozen-parameter sensitivity may apply the completed
            # single-exposure HPO vector to candidate sets. The artifact still
            # has to declare its true scope; only this explicit boundary permits
            # the cross-scope reuse.
            if label in {"k10", "k63"} and os.environ.get("PAPER_REANALYSIS_SET_HPO_SOURCE") == "single":
                scope = "single_exposure"
            artifacts[label] = load_frozen_hpo(Path(os.environ[env_name]), feature_scope=scope, country_policy=policy)
    if args.preflight:
        _preflight_workers()
        for bag in args.bags:
            _preflight_bag(bag)
        return
    results, work = _runtime(args.run_id)
    for bag in args.bags:
        _run_bag(args.mode, bag, args.rung or "ols", artifacts, results, work, args.n_jobs, args.chunk_size,
                 args.accept_legacy_checkpoints)


if __name__ == "__main__":
    main()
