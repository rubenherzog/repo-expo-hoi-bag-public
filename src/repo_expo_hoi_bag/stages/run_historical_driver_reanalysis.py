#!/usr/bin/env python3
"""Paper re-analysis through the historical XGBoost stage.

This is intentionally a thin adapter.  It delegates every fit to
``run_xgb_loco_stage`` (the engine used by the prior-paper driver), and only
adds immutable-parameter validation plus isolated delivery namespaces.
"""
from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "src" / "repo_expo_hoi_bag" / "core", ROOT / "src" / "repo_expo_hoi_bag" / "stages"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from repo_expo_hoi_bag.analysis.paper_reanalysis import (  # noqa: E402
    PRIMARY_BAGS, XGB_RUNGS, load_frozen_hpo, load_main_policy, sha256_file,
    validate_candidate_pool,
)
from repo_expo_hoi_bag.analysis.xgb_frozen_top50 import baseline_candidate, single_exposure_candidates  # noqa: E402
from repo_expo_hoi_bag.stages.run_paper_reanalysis import ANALYSIS_CFG, CV_CFG, EARLY_STOP_CFG, _candidate_frame, _load_model  # noqa: E402
from loco_fusion_matrix_engine import pin_blas_threads, target_map  # noqa: E402
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs  # noqa: E402
from xgb_loco_engine import run_xgb_loco_stage  # noqa: E402


SET_SCOPES = {"k10": "domain_balanced_k10", "k63": "full_exposome", "single": "single_exposure"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bags", nargs="+", required=True, choices=PRIMARY_BAGS)
    parser.add_argument("--rung", required=True, choices=XGB_RUNGS)
    parser.add_argument("--set", dest="sets", nargs="+", choices=("historical", *SET_SCOPES), required=True)
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--validate-reference", action="store_true")
    return parser.parse_args()


def _artifact(label: str, policy: dict) -> dict | None:
    if label == "historical":
        return None
    path = os.environ[f"PAPER_REANALYSIS_{label.upper()}_HPO"]
    return load_frozen_hpo(Path(path), feature_scope=SET_SCOPES[label], country_policy=policy)


def _normalize(data: dict, rung: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_df = data["model_vs_base_global_df"].copy().rename(columns={"model_id": "candidate_id"})
    country_df = data["model_vs_base_country_df"].copy().rename(columns={"model_id": "candidate_id"})
    global_df["rung_id"] = rung
    country_df["rung_id"] = rung
    return global_df, country_df


def _reference_check(global_result: pd.DataFrame, country_result: pd.DataFrame, reference: Path, bag: str, rung: str) -> None:
    """Fail unless every shared historical metric reproduces to 1e-12."""
    stage = reference.parent / "experiments" / f"pooled_oinfo_ladder_{bag}" / rung
    pairs = ((global_result, pd.read_csv(stage / "xgb_model_vs_baseline_global.csv"), ["candidate_id"]),
             (country_result, pd.read_csv(stage / "xgb_model_vs_baseline_country.csv"), ["candidate_id", "fold_country"]))
    for observed, expected, keys in pairs:
        expected = expected.rename(columns={"model_id": "candidate_id"})
        observed = observed.drop(columns=["rung_id"], errors="ignore")
        common = [column for column in expected.columns if column in observed.columns]
        left = observed[common].sort_values(keys).reset_index(drop=True)
        right = expected[common].sort_values(keys).reset_index(drop=True)
        if left.shape != right.shape or not left[keys].equals(right[keys]):
            raise ValueError(f"Historical validation keys differ for {bag}/{rung}")
        for column in common:
            if column in keys:
                continue
            lhs, rhs = left[column], right[column]
            if pd.api.types.is_numeric_dtype(lhs) and pd.api.types.is_numeric_dtype(rhs):
                delta = (lhs - rhs).abs()
                if not (delta.fillna(0.0) <= 1e-12).all():
                    raise ValueError(f"Historical validation failed for {bag}/{rung}/{column}: max delta={delta.max()}")
            elif not lhs.fillna("").astype(str).equals(rhs.fillna("").astype(str)):
                raise ValueError(f"Historical validation failed for {bag}/{rung}/{column}")


def main() -> None:
    args = _args()
    if args.validate_reference and args.sets != ["historical"]:
        raise ValueError("--validate-reference is only valid with --set historical")
    runtime = Path(os.environ["REPRO_DATA_ROOT"]).resolve()
    root = runtime / "results" / "analysis_runs" / args.run_id
    reference = Path(os.environ["PAPER_REANALYSIS_REFERENCE_ROOT"]).resolve()
    policy, policy_hash = load_main_policy(ROOT / "config" / "country_exclusions.yaml")
    artifacts = {label: _artifact(label, policy) for label in args.sets}
    raw_model, features = _load_model()
    specs = {row["rung_id"]: row for row in get_rung_specs()}
    base_cfg = build_xgb_cfg_for_rung(specs[args.rung])
    # This is the only numerical setup done by the old stages_xgb wrapper
    # before it delegates to run_xgb_loco_stage.  Do not allow an incidental
    # baseline cache to substitute models during the equivalence check.
    pin_blas_threads(1)
    os.environ.pop("XGB_BASELINE_CACHE_DIR", None)
    for bag in args.bags:
        pool = validate_candidate_pool(pd.read_parquet(reference / "candidate_registry.parquet"), bag)
        candidates = _candidate_frame(pool)
        for label in args.sets:
            selected = None if artifacts[label] is None else artifacts[label]["rows"][(bag, args.rung, "__global__")]
            params = dict(base_cfg) if selected is None else {**base_cfg, **dict(selected["params"])}
            # The frozen selection is global by contract: broadcast the same
            # immutable vector to all outer LOCO countries.
            outdir = root / "xgb" / bag / args.rung / label
            if outdir.exists():
                raise FileExistsError(
                    f"Refusing to overwrite an existing historical-driver result: {outdir}"
                )
            data = run_xgb_loco_stage(
                model_df=raw_model, candidate_df=candidates, exposome_cols=features,
                bag_targets={bag: target_map(bag)[bag]}, outdir=outdir / "historical_engine",
                analysis_cfg=ANALYSIS_CFG, cv_cfg=CV_CFG,
                perf_cfg={"n_jobs": args.n_jobs, "backend": "loky", "blas_threads": 1,
                          "chunk_size_models": args.chunk_size, "enable_identity_prune": True},
                xgb_cfg=params, early_stop_cfg=EARLY_STOP_CFG,
                enable_progress=True,
                storage_cfg={"write_xgb_base_predictions": False, "write_xgb_model_predictions": False,
                             "keep_xgb_base_predictions_in_memory": False, "keep_xgb_model_predictions_in_memory": False},
                compare_cfg={"compute_baseline_oof": True, "compute_paired_delta": False,
                             "compute_paired_after_all_bags": False, "save_thin_paired_predictions": False},
            )
            global_df, country_df = _normalize(data, args.rung)
            if args.validate_reference:
                _reference_check(global_df, country_df, reference, bag, args.rung)
            outdir.mkdir(parents=True, exist_ok=True)
            global_df.to_csv(outdir / "metrics_global.csv", index=False)
            country_df.to_csv(outdir / "metrics_country.csv", index=False)
            manifest = {"engine": "historical_run_xgb_loco_stage", "bag": bag, "rung": args.rung,
                        "parameter_set": label, "historical_validation": args.validate_reference,
                        "candidate_pool_sha256": sha256_file(reference / "candidate_registry.parquet"),
                        "country_policy_sha256": policy_hash,
                        "frozen_hpo": None if selected is None else {"path": artifacts[label]["path"], "sha256": artifacts[label]["sha256"], "params": selected["params"]}}
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"[{bag}/{args.rung}/{label}] complete", flush=True)


if __name__ == "__main__":
    main()
