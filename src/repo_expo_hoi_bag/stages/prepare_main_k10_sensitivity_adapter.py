#!/usr/bin/env python3
"""Materialize the parent sensitivity metric contract from main k10 outputs.

The parent sensitivity stages consume ``metrics_global_long.parquet`` and
``metrics_country_long.parquet``.  This adapter changes only their data source:
it joins the immutable candidate/O-information identities to the completed main
k10 XGBoost metrics.  It never evaluates a model, computes O-information, or
reads previous BAG metrics.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


BAGS = ("structural", "functional")
RUNGS = ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
EFFECTIVE_EXPOSOME_NAME = "effective_exposome_country_year.csv"
DOMAIN_PC1_OINFO_NAME = "domain_pc1_oinfo_scores.csv"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--adapter-run-id", required=True)
    parser.add_argument("--effective-exposome-source", type=Path, required=True)
    parser.add_argument("--feature-domains", type=Path, required=True)
    parser.add_argument("--domain-pc1-oinfo-reference", type=Path, required=True)
    parser.add_argument(
        "--complete-existing-adapter",
        action="store_true",
        help="Add the effective exposome input to an existing assigned adapter.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--r2-estimator",
        choices=("country-balanced", "global-oof"),
        default="country-balanced",
        help=(
            "Estimand written into the parent selection columns full_r2/base_r2. "
            "Both estimands are always materialized as country_balanced_r2 and "
            "global_oof_r2; this only chooses which one downstream stages select on."
        ),
    )
    return parser.parse_args()


def _registry_by_bag(path: Path, bag: str) -> pd.DataFrame:
    registry = pd.read_parquet(path)
    rows = registry[registry["experiment_id"].astype(str).eq(f"pooled_oinfo_ladder_{bag}")].copy()
    if len(rows) != 1120 or rows["candidate_id"].astype(str).nunique() != 1120:
        raise ValueError(f"Expected the complete 1,120-candidate registry for {bag}")
    required = {"candidate_id", "objective", "order", "score", "thoi_o", "predictors_identity"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Candidate registry lacks {sorted(missing)}")
    return rows[list(required)].copy()


def _materialize_effective_exposome(
    source: Path,
    feature_domains: Path,
    destination: Path,
) -> dict[str, object]:
    """Write one row per unique exposome signature, matching the parent method."""
    raw = pd.read_csv(source, low_memory=False)
    domains = pd.read_csv(feature_domains)
    feature_names = domains["feature_name"].astype(str).tolist()
    required = ["country_clean", *feature_names]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Effective-exposome source lacks columns: {missing[:10]}")

    signature_country_counts = raw.groupby(feature_names, dropna=False)[
        "country_clean"
    ].nunique(dropna=False)
    if signature_country_counts.gt(1).any():
        raise ValueError("An exposome signature maps to more than one country")

    effective = raw.drop_duplicates(subset=feature_names, keep="first").copy()
    if effective[feature_names].isna().any().any():
        raise ValueError("Effective exposome contains missing feature values")
    if len(effective) != 259:
        raise ValueError(f"Expected 259 unique exposome signatures, found {len(effective)}")
    if destination.exists():
        existing = pd.read_csv(destination, low_memory=False)
        if not existing[required].reset_index(drop=True).equals(
            effective[required].reset_index(drop=True)
        ):
            raise FileExistsError(
                f"Existing effective exposome conflicts with public input: {destination}"
            )
    else:
        effective.to_csv(destination, index=False)
    return {
        "relative_path": destination.name,
        "sha256": _sha256(destination),
        "source_sha256": _sha256(source),
        "feature_domains_sha256": _sha256(feature_domains),
        "rows": int(len(effective)),
        "features": int(len(feature_names)),
        "unit": "unique_exposome_signature",
    }


def _materialize_domain_pc1_oinfo(reference: Path, destination: Path) -> dict[str, object]:
    frame = pd.read_csv(reference)
    required = {"predictors_identity", "order", "score", "thoi_o"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Domain-PC1 O-information reference lacks {sorted(missing)}")
    scores = frame[list(sorted(required))].drop_duplicates("predictors_identity").copy()
    if len(scores) != 968 or scores["predictors_identity"].nunique() != 968:
        raise ValueError("Expected 968 unique precomputed domain-PC1 candidates")
    if scores[["score", "thoi_o"]].isna().any().any():
        raise ValueError("Domain-PC1 O-information reference contains missing scores")
    if destination.exists():
        existing = pd.read_csv(destination)
        if not existing.equals(scores):
            raise FileExistsError(f"Existing domain-PC1 reference conflicts: {destination}")
    else:
        scores.to_csv(destination, index=False)
    return {
        "relative_path": destination.name,
        "sha256": _sha256(destination),
        "source_sha256": _sha256(reference),
        "rows": int(len(scores)),
        "reuse_key": "predictors_identity",
    }


def _metrics_root(runtime: Path, source_run_id: str, bag: str, rung: str, arm: str) -> Path:
    return runtime / "results" / "analysis_runs" / source_run_id / "xgb" / bag / rung / arm


def _build_bag(runtime: Path, source_run_id: str, registry: Path, destination: Path, bag: str, estimator: str = "country-balanced") -> dict[str, object]:
    identity = _registry_by_bag(registry, bag)
    global_oof = estimator == "global-oof"
    global_parts: list[pd.DataFrame] = []
    country_parts: list[pd.DataFrame] = []
    source_files: list[Path] = []
    for rung in RUNGS:
        candidate_root = _metrics_root(runtime, source_run_id, bag, rung, "k10")
        baseline_root = _metrics_root(runtime, source_run_id, bag, rung, "baseline")
        global_path = candidate_root / "metrics_global.csv"
        country_path = candidate_root / "metrics_country.csv"
        baseline_country_path = baseline_root / "metrics_country.csv"
        for path in (global_path, country_path, baseline_country_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required main k10 input is missing: {path}")
        source_files.extend((global_path, country_path, baseline_country_path))
        global_metrics = pd.read_csv(global_path)
        country_metrics = pd.read_csv(country_path)
        baseline_country = pd.read_csv(baseline_country_path)
        if set(global_metrics["candidate_id"].astype(str)) != set(identity["candidate_id"].astype(str)):
            raise ValueError(f"k10 global candidate membership mismatch for {bag}/{rung}")
        if set(country_metrics["candidate_id"].astype(str)) != set(identity["candidate_id"].astype(str)):
            raise ValueError(f"k10 country candidate membership mismatch for {bag}/{rung}")
        baseline = baseline_country[["fold_country", "r2"]].rename(columns={"r2": "country_base_r2"})
        if global_oof:
            baseline_global = pd.read_csv(baseline_country_path.with_name("metrics_global.csv"))
            baseline_r2 = float(pd.to_numeric(baseline_global["global_oof_r2"], errors="coerce").iloc[0])
            source_files.append(baseline_country_path.with_name("metrics_global.csv"))
        else:
            baseline_r2 = float(
                pd.to_numeric(baseline_country.loc[
                    pd.to_numeric(baseline_country["n_test"], errors="coerce").gt(0), "r2"
                ], errors="coerce").mean()
            )
        if not pd.notna(baseline_r2):
            raise ValueError(f"Missing {estimator} baseline R² for {bag}/{rung}")
        global_frame = identity.merge(global_metrics, on="candidate_id", how="inner", validate="one_to_one")
        country_balanced = (
            country_metrics[pd.to_numeric(country_metrics["n_test"], errors="coerce").gt(0)]
            .groupby("candidate_id", as_index=False, observed=True)["r2"]
            .mean()
            .rename(columns={"r2": "country_balanced_r2"})
        )
        global_frame = global_frame.merge(country_balanced, on="candidate_id", how="inner", validate="one_to_one")
        # ``full_r2`` is the historical parent-stage selection column.  Both
        # estimands are materialized above; this only chooses which one the
        # parent stages select and plot on.  They are never mixed within a run.
        selection_column = "global_oof_r2" if global_oof else "country_balanced_r2"
        global_frame["full_r2"] = pd.to_numeric(global_frame[selection_column], errors="coerce")
        # ``base_r2`` is the unchanged column name required by the parent PCA
        # sensitivity.  Its value is the k10 baseline under the same estimand
        # that is used for ``full_r2``.
        global_frame["base_r2"] = baseline_r2
        global_frame["r2_mode"] = "global_oof" if global_oof else "country_balanced"
        global_frame["bag_target"] = bag
        country_frame = country_metrics.merge(identity, on="candidate_id", how="inner", validate="many_to_one")
        country_frame = country_frame.merge(baseline, on="fold_country", how="left", validate="many_to_one")
        country_frame["country_full_r2"] = pd.to_numeric(country_frame["r2"], errors="coerce")
        country_frame["n_test_scored"] = pd.to_numeric(country_frame["n_test"], errors="coerce")
        country_frame["bag_target"] = bag
        global_parts.append(global_frame)
        country_parts.append(country_frame)
    root = destination / "per_experiment" / f"pooled_oinfo_ladder_{bag}"
    root.mkdir(parents=True, exist_ok=False)
    pd.concat(global_parts, ignore_index=True).to_parquet(root / "metrics_global_long.parquet", index=False)
    pd.concat(country_parts, ignore_index=True).to_parquet(root / "metrics_country_long.parquet", index=False)
    return {
        "bag": bag,
        "candidate_registry_sha256": _sha256(registry),
        "input_sha256": {str(path): _sha256(path) for path in source_files},
        "n_global_rows": int(sum(len(frame) for frame in global_parts)),
        "n_country_rows": int(sum(len(frame) for frame in country_parts)),
    }


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.resolve()
    registry = args.candidate_registry.resolve()
    effective_source = args.effective_exposome_source.resolve()
    feature_domains = args.feature_domains.resolve()
    domain_pc1_reference = args.domain_pc1_oinfo_reference.resolve()
    destination = runtime / "results" / "analysis_runs" / args.adapter_run_id / "input_adapter"
    if args.smoke_test:
        for bag in BAGS:
            _registry_by_bag(registry, bag)
            for rung in RUNGS:
                for arm in ("k10", "baseline"):
                    root = _metrics_root(runtime, args.source_run_id, bag, rung, arm)
                    if not (root / "metrics_country.csv").is_file():
                        raise FileNotFoundError(root / "metrics_country.csv")
                baseline = pd.read_csv(_metrics_root(runtime, args.source_run_id, bag, rung, "baseline") / "metrics_country.csv")
                if "n_test" not in baseline or "r2" not in baseline:
                    raise ValueError(f"Baseline metrics lack n_test/r2 for {bag}/{rung}")
                if not pd.to_numeric(baseline.loc[pd.to_numeric(baseline["n_test"], errors="coerce").gt(0), "r2"], errors="coerce").notna().any():
                    raise ValueError(f"Baseline metrics have no valid country-balanced R² for {bag}/{rung}")
        print("Smoke test passed: main-k10 sensitivity inputs are complete.")
        return
    if args.complete_existing_adapter:
        manifest_path = destination / "main_k10_adapter_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Existing adapter manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["effective_exposome"] = _materialize_effective_exposome(
            effective_source,
            feature_domains,
            destination / EFFECTIVE_EXPOSOME_NAME,
        )
        manifest["domain_pc1_oinfo"] = _materialize_domain_pc1_oinfo(
            domain_pc1_reference,
            destination / DOMAIN_PC1_OINFO_NAME,
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Completed main-k10 sensitivity adapter: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing adapter: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "analysis_label": "main",
        "source_run_id": args.source_run_id,
        "model_family": "xgboost_k10_only",
        "ols_policy": "reused_reference_not_materialized",
        "country_exclusions": ["France", "Italy", "Egypt", "Greece", "Poland"],
        "rungs": list(RUNGS),
        "bags": list(BAGS),
        "r2_mode": "global_oof" if args.r2_estimator == "global-oof" else "country_balanced",
        "selection_column": "global_oof_r2" if args.r2_estimator == "global-oof" else "country_balanced_r2",
        "bags_detail": [_build_bag(runtime, args.source_run_id, registry, destination, bag, args.r2_estimator) for bag in BAGS],
        "effective_exposome": _materialize_effective_exposome(
            effective_source,
            feature_domains,
            destination / EFFECTIVE_EXPOSOME_NAME,
        ),
        "domain_pc1_oinfo": _materialize_domain_pc1_oinfo(
            domain_pc1_reference,
            destination / DOMAIN_PC1_OINFO_NAME,
        ),
    }
    (destination / "main_k10_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared main-k10 sensitivity adapter: {destination}")


if __name__ == "__main__":
    main()
