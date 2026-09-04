#!/usr/bin/env python3
"""Single-exposure normative LOCO pipeline for OLS and XGBoost.

This runner reuses the already validated normative-transfer evaluators:

- ``scripts.run_ols_normative_loco_pipeline``
- ``scripts.run_xgb_normative_loco_pipeline``

The only change is the candidate pool: every exposome variable is evaluated as
one single-exposure candidate. No baseline candidate is included here because
the corresponding normative baselines are already available from the full
normative runs.

Heavy outputs are written to the external runtime under ``REPRO_DATA_ROOT``.
Local outputs are limited to lightweight global summaries and manifests.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.ridge_common import build_ridge_model_df
from scripts.sensitivity_common import bundle_root, log_msg
from scripts.run_ols_normative_loco_pipeline import (
    OLSNormFamily,
    ols_norm_families,
    ols_norm_test_dx,
    _evaluate_train_family as evaluate_ols_train_family,
)
from scripts.run_xgb_normative_loco_pipeline import (
    XGBNormFamily,
    xgb_norm_families,
    xgb_norm_rungs,
    xgb_norm_test_dx,
    _evaluate_train_family as evaluate_xgb_train_family,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAGS = ["functional", "structural"]


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y"}


def selected_bags() -> list[str]:
    requested = os.environ.get("SINGLE_NORM_BAGS", "").strip()
    bags = [b.strip() for b in requested.split(",") if b.strip()] if requested else list(DEFAULT_BAGS)
    bad = [b for b in bags if b not in {"functional", "structural", "combined"}]
    if bad:
        raise ValueError(f"Unsupported SINGLE_NORM_BAGS: {bad}")
    return [b for b in ["functional", "structural", "combined"] if b in set(bags)]


def selected_models() -> list[str]:
    requested = os.environ.get("SINGLE_NORM_MODELS", "").strip()
    models = [m.strip().lower() for m in requested.split(",") if m.strip()] if requested else ["ols", "xgb"]
    bad = [m for m in models if m not in {"ols", "xgb"}]
    if bad:
        raise ValueError(f"Unsupported SINGLE_NORM_MODELS: {bad}; allowed=ols,xgb")
    return [m for m in ["ols", "xgb"] if m in set(models)]


def local_root() -> Path:
    root = Path(os.environ.get("SINGLE_NORM_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "single_exposure_normative")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def eval_root() -> Path:
    override = os.environ.get("SINGLE_NORM_EVAL_ROOT", "").strip()
    if override:
        root = Path(override)
    else:
        if not os.environ.get("REPRO_DATA_ROOT", "").strip():
            raise EnvironmentError(
                "REPRO_DATA_ROOT is required. Heavy single-exposure normative outputs "
                "must be written to the external runtime, not local outputs/."
            )
        subdir = os.environ.get("SINGLE_NORM_BUNDLE_SUBDIR", "single_exposure_normative").strip()
        root = bundle_root() / (subdir or "single_exposure_normative")
    root.mkdir(parents=True, exist_ok=True)
    return root


def single_exposure_candidate_pool(feature_names: list[str], *, smoke: bool = False) -> pd.DataFrame:
    names = feature_names[:5] if smoke else feature_names
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "candidate_id": f"single__{name}",
                "feature_id": f"single__{name}",
                "objective": "single_exposure",
                "order": 1,
                "rank": i + 1,
                "score": pd.NA,
                "nplet_vars": [name],
                "predictors_identity": name,
                "predictors_identity_n": 1,
                "candidate_family": "single_exposure",
                "source_label": "single_exposure",
                "feature_name": name,
                "feature_idx": i,
            }
        )
    return pd.DataFrame(rows)


def write_manifest(root: Path, *, eval_dir: Path, bags: list[str], models: list[str], n_jobs: int, smoke: bool) -> None:
    rows = [
        {"key": "local_summary_root", "value": str(root)},
        {"key": "external_eval_root", "value": str(eval_dir)},
        {"key": "bags", "value": ",".join(bags)},
        {"key": "models", "value": ",".join(models)},
        {"key": "n_jobs", "value": str(n_jobs)},
        {"key": "smoke", "value": str(bool(smoke))},
        {"key": "contains_baseline_candidate", "value": "false"},
        {"key": "fit_once_per_train_group", "value": "true"},
        {"key": "SINGLE_NORM_EVAL_CHUNK_SIZE", "value": os.environ.get("SINGLE_NORM_EVAL_CHUNK_SIZE", "63")},
    ]
    pd.DataFrame(rows).to_csv(root / "single_exposure_normative_manifest.csv", index=False)


def _ols_families() -> list[OLSNormFamily]:
    return ols_norm_families()


def _xgb_families() -> list[XGBNormFamily]:
    return xgb_norm_families()


def main() -> None:
    smoke = _env_bool("SINGLE_NORM_SMOKE") or _env_bool("SMOKE_TEST")
    bags = selected_bags()
    models = selected_models()
    n_jobs = int(os.environ.get("SINGLE_NORM_N_JOBS", os.environ.get("N_JOBS", "20")))
    lroot = local_root()
    eroot = eval_root()

    # Keep chunking explicit and small enough to start logging progress per model/family.
    os.environ.setdefault("OLS_NORM_EVAL_CHUNK_SIZE", os.environ.get("SINGLE_NORM_EVAL_CHUNK_SIZE", "63"))
    os.environ.setdefault("XGB_NORM_EVAL_CHUNK_SIZE", os.environ.get("SINGLE_NORM_EVAL_CHUNK_SIZE", "63"))
    os.environ.setdefault("OLS_NORM_FAMILIES", "cn_norm,ad_norm,ftd_norm,adftd_norm")
    os.environ.setdefault("XGB_NORM_FAMILIES", "cn_norm,ad_norm,ftd_norm,adftd_norm")
    os.environ.setdefault("OLS_NORM_TEST_DX", "CN,AD,FTD")
    os.environ.setdefault("XGB_NORM_TEST_DX", "CN,AD,FTD")

    write_manifest(lroot, eval_dir=eroot, bags=bags, models=models, n_jobs=n_jobs, smoke=smoke)
    write_manifest(eroot, eval_dir=eroot, bags=bags, models=models, n_jobs=n_jobs, smoke=smoke)

    model_df, feature_names = build_ridge_model_df()
    pool = single_exposure_candidate_pool(feature_names, smoke=smoke)
    pool.to_csv(lroot / "single_exposure_candidate_pool.csv", index=False)
    pool.to_csv(eroot / "single_exposure_candidate_pool.csv", index=False)
    log_msg(f"Single-exposure normative candidate pool: rows={len(pool)} smoke={smoke}")

    all_summary = []
    all_country = []

    for bag in bags:
        log_msg(f"Single-exposure normative bag start: {bag}")

        if "ols" in models:
            test_dx = ols_norm_test_dx()
            for family in _ols_families():
                outdir = eroot / "ols" / bag / family.family_id / f"train_{family.train_label}"
                summary, country = evaluate_ols_train_family(
                    model_df=model_df,
                    candidate_df=pool,
                    exposome_cols=feature_names,
                    bag=bag,
                    family=family,
                    test_dx=test_dx,
                    outdir=outdir,
                    n_jobs=n_jobs,
                )
                for df in (summary, country):
                    if not df.empty:
                        df["bag"] = bag
                        df["model_family"] = "ols"
                all_summary.append(summary)
                all_country.append(country)

        if "xgb" in models:
            test_dx = xgb_norm_test_dx()
            rungs = xgb_norm_rungs()
            for family in _xgb_families():
                outdir = eroot / "xgb" / bag / family.family_id / f"train_{family.train_label}"
                summary, country = evaluate_xgb_train_family(
                    model_df=model_df,
                    candidate_df=pool,
                    exposome_cols=feature_names,
                    bag=bag,
                    family=family,
                    test_dx=test_dx,
                    rungs=rungs,
                    outdir=outdir,
                    n_jobs=n_jobs,
                )
                for df in (summary, country):
                    if not df.empty:
                        df["bag"] = bag
                        df["model_family"] = "xgb"
                all_summary.append(summary)
                all_country.append(country)

        log_msg(f"Single-exposure normative bag complete: {bag}")

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        summary_all.to_csv(lroot / "single_exposure_normative_global_all.csv", index=False)
        summary_all.to_csv(eroot / "single_exposure_normative_global_all.csv", index=False)
    if all_country:
        pd.concat(all_country, ignore_index=True).to_csv(eroot / "single_exposure_normative_country_all.csv", index=False)

    log_msg(f"Single-exposure normative pipeline complete: local summaries={lroot} detailed eval={eroot}")


if __name__ == "__main__":
    main()
