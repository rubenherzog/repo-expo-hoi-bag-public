"""Contracts for the main-paper XGBoost re-analysis.

This module deliberately does not discover candidates or tune models.  It
validates immutable inputs before a scheduler worker is allowed to evaluate a
single XGBoost chunk, and keeps the unit of resumption small and auditable.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PRIMARY_BAGS = ("structural", "functional")
XGB_RUNGS = ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")


def sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_hpo(artifact: Path, *, feature_scope: str, country_policy: dict[str, Any]) -> dict[str, Any]:
    """Validate a released selection artifact, never a tuning checkpoint."""
    artifact = Path(artifact)
    manifest_path = artifact.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "paper_final_hpo_selection":
        raise ValueError(f"Not a final HPO selection artifact: {artifact}")
    if sha256_file(artifact) != manifest.get("selected_sha256"):
        raise ValueError(f"Frozen HPO hash mismatch: {artifact}")
    data = json.loads(artifact.read_text(encoding="utf-8"))
    provenance = data.get("provenance", {})
    if provenance.get("feature_scope") != feature_scope:
        raise ValueError(f"HPO feature scope mismatch: expected {feature_scope}")
    observed = provenance.get("two_phase_country_config", {})
    for key in ("variant", "bag_exclude_countries", "bag_exclude_diagnosis"):
        if observed.get(key) != country_policy.get(key):
            raise ValueError(f"HPO country policy mismatch for {key}")
    rows = {(str(r.get("bag_target")), str(r.get("rung_id")), str(r.get("outer_country"))): r
            for r in data.get("selected", [])}
    expected = {(bag, rung, "__global__") for bag in PRIMARY_BAGS for rung in XGB_RUNGS}
    if set(rows) != expected:
        raise ValueError("Frozen HPO must cover exactly structural/functional × d1/d2/d3 globally")
    return {"path": str(artifact.resolve()), "sha256": sha256_file(artifact), "feature_scope": feature_scope,
            "rows": rows, "manifest_sha256": sha256_file(manifest_path)}


def validate_candidate_pool(registry: pd.DataFrame, bag: str) -> pd.DataFrame:
    """Return one BAG's official deduplicated pool or fail closed."""
    rows = registry[registry["experiment_id"].astype(str) == f"pooled_oinfo_ladder_{bag}"].copy()
    if len(rows) != 1120 or rows["candidate_id"].astype(str).nunique() != 1120:
        raise ValueError(f"Official {bag} pool must contain exactly 1,120 unique candidates")
    grouped = rows.groupby(["objective", "order"], dropna=False, observed=True).size()
    expected = {(objective, order) for objective in ("o_max", "o_min") for order in range(3, 31)}
    if set(grouped.index) != expected or not (grouped == 20).all():
        raise ValueError("Official pool must be top-20 for o_max/o_min at every order 3–30")
    return rows.sort_values(["objective", "order", "rank", "candidate_id"]).reset_index(drop=True)


def checkpoint_identity(*, code_hash: str, pool_hash: str, policy_hash: str, hpo_hash: str, bag: str, rung: str) -> str:
    payload = {"code_hash": code_hash, "pool_hash": pool_hash, "policy_hash": policy_hash,
               "hpo_hash": hpo_hash, "bag": bag, "rung": rung}
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pending_chunks(work_dir: Path, identity: str, candidate_ids: list[str], chunk_size: int) -> list[tuple[int, list[str]]]:
    """Find only completed chunks whose identity and candidate membership match."""
    pending: list[tuple[int, list[str]]] = []
    for index, start in enumerate(range(0, len(candidate_ids), chunk_size)):
        members = candidate_ids[start:start + chunk_size]
        marker = work_dir / f"chunk-{index:04d}.json"
        complete = False
        if marker.is_file():
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
                complete = saved.get("identity") == identity and saved.get("candidate_ids") == members and saved.get("status") == "complete"
            except json.JSONDecodeError:
                complete = False
        if not complete:
            pending.append((index, members))
    return pending


def write_chunk_complete(work_dir: Path, index: int, identity: str, candidate_ids: list[str], files: dict[str, str]) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"chunk-{index:04d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": "complete", "identity": identity, "candidate_ids": candidate_ids,
                                     "files": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_main_policy(path: Path) -> tuple[dict[str, Any], str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    source = dict(raw["bag"]["variants"]["historical_a"])
    policy = {
        "variant": "historical_a",
        "bag_exclude_countries": list(source["exclude_countries"]),
        "bag_exclude_diagnosis": list(source["exclude_diagnosis"]),
    }
    return policy, sha256_file(Path(path))


def validate_ols_reuse_manifest(manifest: dict[str, Any], *, cohort_sha256: str, policy_sha256: str,
                                pool_sha256: str, contract_sha256: str) -> None:
    """Fail closed unless the immutable OLS provenance proves all reuse criteria."""
    expected = {
        "cohort_sha256": cohort_sha256,
        "country_policy_sha256": policy_sha256,
        "candidate_pool_sha256": pool_sha256,
        "ols_contract_sha256": contract_sha256,
    }
    missing = sorted(set(expected).difference(manifest))
    if missing:
        raise ValueError(f"OLS reuse provenance is incomplete: missing {missing}")
    mismatch = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatch:
        raise ValueError(f"OLS reuse provenance mismatch: {mismatch}")


def country_balanced_ols_r2(country_metrics: pd.DataFrame) -> pd.DataFrame:
    """Simple mean of valid LOCO country R² values for each candidate."""
    required = {"candidate_id", "country_full_r2", "n_test_scored"}
    if not required.issubset(country_metrics.columns):
        raise ValueError(f"OLS country metrics missing {sorted(required.difference(country_metrics.columns))}")
    valid = country_metrics[pd.to_numeric(country_metrics["n_test_scored"], errors="coerce") > 0].copy()
    valid["country_full_r2"] = pd.to_numeric(valid["country_full_r2"], errors="coerce")
    return valid.groupby("candidate_id", as_index=False)["country_full_r2"].mean().rename(
        columns={"country_full_r2": "country_balanced_r2"}
    )
