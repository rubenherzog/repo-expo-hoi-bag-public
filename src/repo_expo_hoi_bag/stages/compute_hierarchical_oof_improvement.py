#!/usr/bin/env python3
"""Hierarchical test of individual OOF prediction improvement over the baseline.

The question
------------
Within the observed BAG-association landscape, is the OOF prediction
improvement of the high-performing subset region consistently positive at the
participant level, after accounting for subjects being clustered in countries?

Estimand
--------
For participant i and candidate j::

    improvement_ij = (y_i - yhat_base_i)^2 - (y_i - yhat_cand_ij)^2

Candidate dependence is collapsed *before* any model is fitted: the subject-level
outcome is ``mean_improvement_i,K = mean_j improvement_ij`` over the candidates
of region K. Subject x candidate rows are never treated as independent.

Model, per BAG x arm x rung x region::

    mean_improvement_i,K = beta0 + u_country[c(i)] + eps_i

fitted by REML. ``beta0`` is the population-level mean improvement. The
sensitivity model adds a country-year intercept nested in country.

Interpretation limits (section 8)
---------------------------------
The Top-K regions are defined by high LOCO R2 on the same countries used for
evaluation. These tests are therefore **conditional on the observed candidate
landscape**: they are not selection-unbiased, are not nested-selection
estimates, and are not expected deployment performance in a new country. The
country-cross-selected analysis remains the separate sensitivity analysis about
selection stability and transportability.

``beta0`` is on the squared-error scale, not the R2 scale. It is never called
a delta-R2; observed candidate delta-R2 is reported separately from the
country-balanced landscape metrics.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.run_hierarchical_oof_predictions import (
    NESTED_K,
    PRIMARY_K,
    SELECTION_NOTE,
    region_members,
)
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import _sha256

ROOT = frontier.ROOT

BAGS = ("structural", "functional")
ARMS = ("o_min", "o_max")
# OLS and d1 are descriptive levels; PRIMARY_RUNGS alone form the Holm family.
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
PRIMARY_RUNGS = ("xgb_tree_d2", "xgb_tree_d3")
PRIMARY_REGION = f"Top{PRIMARY_K}"
REGION_ORDER = (*[f"Top{k}" for k in NESTED_K], "ALL")

CONDITIONAL_CAVEAT = (
    "Top-K regions are conditional on the observed candidate landscape: candidates "
    "enter because they had high LOCO R2 on these same countries. These tests are "
    "not selection-unbiased, are not nested-selection estimates, and do not "
    "estimate deployment performance in an unseen country."
)
SCALE_NOTE = (
    "beta0 is a mean improvement in squared prediction error, not a delta-R2. It is "
    "not normalised to the R2 scale. Observed candidate delta-R2 is reported "
    "separately in the landscape summary."
)
DEPENDENCE_NOTE = (
    "Candidates within a region overlap heavily. Improvement is averaged across the "
    "region within each participant before the mixed model is fitted, so subject x "
    "candidate rows never enter as independent observations. Countries enter as a "
    "random intercept."
)


def _fit_mixed(
    frame: pd.DataFrame, groups: str, *, nested: str | None = None
) -> dict[str, object]:
    """REML random-intercept fit of mean improvement on an intercept only."""
    data = frame.dropna(subset=["mean_improvement", groups]).copy()

    # The formula interface is used for both models: the array interface does not
    # attach a random-intercept design here and returns a singular random-effects
    # covariance, which would silently discard the country variance component.
    if nested is not None:
        # Country-year nested in country: a variance component within the group.
        model = MixedLM.from_formula(
            "mean_improvement ~ 1",
            groups=groups,
            vc_formula={"country_year": f"0 + C({nested})"},
            re_formula="1",
            data=data,
        )
    else:
        model = MixedLM.from_formula(
            "mean_improvement ~ 1", groups=groups, re_formula="1", data=data
        )

    # ``lbfgs`` inverts a singular information matrix on these intercept-only
    # designs. The remaining optimisers agree with each other to the printed
    # precision, so the first that converges is used and the choice is recorded.
    result = None
    messages: list[str] = []
    for method in ("bfgs", "cg", "powell", "nm"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(reml=True, method=method)
            except (np.linalg.LinAlgError, ValueError) as exc:
                messages.append(f"{method}: {type(exc).__name__}: {exc}")
                continue
            messages.extend(sorted({str(item.message)[:160] for item in caught}))
        optimiser = method
        break
    if result is None:
        raise RuntimeError(f"No optimiser converged; attempts: {messages}")

    beta = float(np.asarray(result.fe_params)[0])
    se = float(np.sqrt(np.asarray(result.cov_params())[0, 0]))
    z = beta / se if se > 0 else float("nan")
    p_two = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    # Directional secondary quantity only; never replaces the two-sided test.
    p_positive = float(stats.norm.sf(z)) if np.isfinite(z) else float("nan")

    group_var = float(np.asarray(result.cov_re)[0, 0]) if result.cov_re.size else float("nan")
    vc_var = float(result.vcomp[0]) if nested is not None and len(result.vcomp) else float("nan")
    resid = float(result.scale)
    total = group_var + resid + (vc_var if np.isfinite(vc_var) else 0.0)

    return {
        "beta_intercept": beta,
        "se": se,
        "ci_lo": beta - 1.959963984540054 * se,
        "ci_hi": beta + 1.959963984540054 * se,
        "z_statistic": z,
        "p_two_sided": p_two,
        "p_positive_tail": p_positive,
        "country_var": group_var,
        "country_year_var": vc_var,
        "residual_var": resid,
        "icc_country": group_var / total if total > 0 else float("nan"),
        "n_subjects": int(len(data)),
        "n_countries": int(data[groups].nunique()),
        "n_country_years": int(data[nested].nunique()) if nested is not None else pd.NA,
        "converged": bool(result.converged),
        "optimiser": optimiser,
        "convergence_warnings": "; ".join(messages) if messages else "",
    }


def model_results(subjects: pd.DataFrame, *, nested: bool = False) -> pd.DataFrame:
    """One fitted row per BAG x arm x rung x region."""
    rows: list[dict[str, object]] = []
    keys = ["bag", "arm", "rung", "region"]
    for (bag, arm, rung, region), block in subjects.groupby(keys, sort=False):
        fit = _fit_mixed(
            block, "country", nested="country_year" if nested else None
        )
        rows.append(
            {
                "bag": bag,
                "arm": arm,
                "rung": rung,
                "region": region,
                "model": "country + country_year" if nested else "country",
                "n_candidate_predictions": int(block["n_candidate_predictions"].iloc[0]),
                **fit,
            }
        )
    frame = pd.DataFrame(rows)
    order = frame["region"].map({name: index for index, name in enumerate(REGION_ORDER)})
    return frame.assign(_order=order).sort_values(
        ["bag", "rung", "arm", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def apply_holm(results: pd.DataFrame) -> pd.DataFrame:
    """Holm within each BAG across exactly the four primary Top-20 tests.

    The family is the four synergy/redundancy x d2/d3 comparisons. OLS and d1
    are descriptive levels and stay outside it, so adding them never inflates
    the family. Top10/Top50 receive a separate within-K correction, and ALL is a
    control that is never folded in.
    """
    frame = results.copy()
    frame["holm_p"] = np.nan
    frame["multiplicity_family"] = pd.NA

    corrected = frame["region"].ne("ALL") & frame["rung"].isin(PRIMARY_RUNGS)
    for (bag, region), block in frame[corrected].groupby(["bag", "region"], sort=False):
        family = f"{bag}:{region}"
        values = block["p_two_sided"].to_numpy(float)
        frame.loc[block.index, "multiplicity_family"] = family
        frame.loc[block.index, "holm_p"] = multipletests(values, method="holm")[1]

    # Descriptive levels first, then ALL, so the control label wins wherever the
    # two overlap and every ALL row carries the same family regardless of level.
    frame.loc[~frame["rung"].isin(PRIMARY_RUNGS), "multiplicity_family"] = (
        "descriptive level (not corrected)"
    )
    frame.loc[frame["region"].eq("ALL"), "multiplicity_family"] = "control (not corrected)"
    return frame


def validate_families(results: pd.DataFrame) -> None:
    """The primary family is exactly four tests per BAG; ALL is excluded.

    OLS and d1 are descriptive levels, so they are excluded here exactly as they
    are excluded from the correction itself.
    """
    primary = results[
        results["region"].eq(PRIMARY_REGION) & results["rung"].isin(PRIMARY_RUNGS)
    ]
    sizes = primary.groupby("multiplicity_family").size()
    if sizes.empty:
        raise ValueError("No primary Top-20 tests were produced")
    for family, size in sizes.items():
        if size != 4:
            raise ValueError(f"Primary Holm family {family!r} has {size} tests, expected 4")
    expected = {f"{bag}:{PRIMARY_REGION}" for bag in BAGS}
    if set(sizes.index) != expected:
        raise ValueError(f"Primary families {sorted(sizes.index)} != {sorted(expected)}")
    if results.loc[results["region"].eq("ALL"), "holm_p"].notna().any():
        raise ValueError("The ALL control must not carry a primary Holm-adjusted P")


def landscape_summary(
    runtime: Path, source_run_id: str, membership: dict
) -> pd.DataFrame:
    """Observed country-balanced candidate R2 versus baseline, per region.

    These are descriptive landscape quantities on the R2 scale, kept entirely
    separate from the subject-level hierarchical model.
    """
    registry = frontier._registry(frontier._registry_path(runtime, source_run_id))
    rows: list[dict[str, object]] = []
    for bag in BAGS:
        inputs = frontier.load_bag(runtime, source_run_id, bag, registry)
        for rung in RUNGS:
            wide = inputs.multivariate[rung]
            baseline = float(inputs.baseline[rung].mean())
            for arm in ARMS:
                regions = membership[(bag, arm, rung)]
                for region, members in regions.items():
                    scores = wide.loc[wide.index.astype(str).isin(members)].mean(axis=1)
                    delta = (scores - baseline).to_numpy(float)
                    rows.append(
                        {
                            "bag": bag,
                            "arm": arm,
                            "rung": rung,
                            "region": region,
                            "n_candidates": int(len(delta)),
                            "mean_candidate_r2": float(scores.mean()),
                            "baseline_r2": baseline,
                            "mean_delta_r2": float(delta.mean()),
                            "median_delta_r2": float(np.median(delta)),
                            "q1_delta_r2": float(np.percentile(delta, 25)),
                            "q3_delta_r2": float(np.percentile(delta, 75)),
                            "min_delta_r2": float(delta.min()),
                            "max_delta_r2": float(delta.max()),
                            "n_candidates_delta_positive": int((delta > 0).sum()),
                            "fraction_delta_positive": float((delta > 0).mean()),
                        }
                    )
    frame = pd.DataFrame(rows)
    order = frame["region"].map({name: index for index, name in enumerate(REGION_ORDER)})
    return frame.assign(_order=order).sort_values(
        ["bag", "rung", "arm", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def load_subjects(cache_dir: Path, country_year: pd.DataFrame | None) -> pd.DataFrame:
    """Concatenate the per-cell loss accumulators and attach country-year.

    Prefers the ``all``-scope file for a cell when present, because it carries
    every region including the ALL control; falls back to the ``topk`` file.
    """
    frames: list[pd.DataFrame] = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                stem = f"{bag}__{arm}__{rung}.parquet"
                path = cache_dir / f"all__{stem}"
                if not path.is_file():
                    path = cache_dir / f"topk__{stem}"
                if not path.is_file():
                    raise FileNotFoundError(f"No loss accumulator for {bag}/{arm}/{rung}")
                frames.append(pd.read_parquet(path))
    subjects = pd.concat(frames, ignore_index=True)

    recomputed = subjects["baseline_loss"] - subjects["mean_candidate_loss"]
    if not np.allclose(recomputed, subjects["mean_improvement"], rtol=0, atol=1e-12):
        raise ValueError("mean_improvement is not baseline_loss - mean_candidate_loss")

    if country_year is not None:
        subjects = subjects.merge(country_year, on="row_id", how="left", validate="many_to_one")
        if subjects["country_year"].isna().any():
            raise ValueError("country_year is missing for some participants")
    return subjects


def membership_table(membership: dict) -> pd.DataFrame:
    rows = [
        {
            "bag": bag,
            "arm": arm,
            "rung": rung,
            "region": region,
            "global_rank": rank,
            "candidate_id": candidate,
        }
        for (bag, arm, rung), regions in membership.items()
        for region, members in regions.items()
        for rank, candidate in enumerate(members, start=1)
    ]
    return pd.DataFrame(rows).sort_values(
        ["bag", "rung", "arm", "region", "global_rank"], kind="stable"
    ).reset_index(drop=True)


def country_year_map(bag: str) -> pd.DataFrame:
    """row_id -> country_year label, from the production model frame."""
    from repo_expo_hoi_bag.stages.run_paper_reanalysis import _build_context, _load_model

    model_df, features = _load_model()
    context, _ = _build_context(model_df, bag, features)
    frame = pd.DataFrame(
        {
            "row_id": context["row_id"],
            "country": np.asarray(context["country"]).astype(str),
        }
    )
    if "exposome_year" not in model_df.columns:
        raise ValueError("exposome_year is unavailable in the model frame")
    frame["exposome_year"] = frame["row_id"].map(model_df["exposome_year"])
    if frame["exposome_year"].isna().any():
        raise ValueError("exposome_year is missing for some participants")
    frame["country_year"] = (
        frame["country"] + "_" + frame["exposome_year"].astype("Int64").astype(str)
    )
    return frame[["row_id", "country_year", "exposome_year"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Loss accumulators from the fitting stage. Defaults to the external "
            "runtime, where the heavy per-subject caches are written."
        ),
    )
    parser.add_argument("--run-id", default="hierarchical_oof_main_k10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10",
    )
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    cache_dir = args.cache_dir or (
        runtime / "work/analysis_runs" / args.run_id / "hierarchical_oof/loss_cache"
    )
    registry = frontier._registry(frontier._registry_path(runtime, args.source_run_id))
    membership: dict = {}
    for bag in BAGS:
        inputs = frontier.load_bag(runtime, args.source_run_id, bag, registry)
        for rung in RUNGS:
            for arm in ARMS:
                regions, everything = region_members(inputs, arm, rung)
                membership[(bag, arm, rung)] = {**regions, "ALL": everything}

    years = pd.concat([country_year_map(bag) for bag in BAGS], ignore_index=True)
    years = years.drop_duplicates("row_id")
    subjects = load_subjects(cache_dir, years)

    results = apply_holm(model_results(subjects))
    validate_families(results)

    nested = model_results(subjects, nested=True)
    landscape = landscape_summary(runtime, args.source_run_id, membership)
    members = membership_table(membership)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects.to_parquet(args.output_dir / "hierarchical_oof_subject_scores.parquet", index=False)
    results.to_csv(args.output_dir / "hierarchical_oof_model_results.csv", index=False)
    nested.to_csv(
        args.output_dir / "hierarchical_oof_country_year_sensitivity.csv", index=False
    )
    landscape.to_csv(args.output_dir / "hierarchical_oof_landscape_summary.csv", index=False)
    members.to_csv(args.output_dir / "hierarchical_oof_candidate_membership.csv", index=False)

    inputs_hashed = {
        key: _sha256(Path(path))
        for bag in BAGS
        for key, path in frontier.load_bag(
            runtime, args.source_run_id, bag, registry
        ).paths.items()
    }
    manifest = {
        "stage": "compute_hierarchical_oof_improvement",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": args.source_run_id,
        "repro_data_root": str(runtime),
        "inputs_sha256": inputs_hashed,
        "candidate_ranking_rule": SELECTION_NOTE,
        "regions": list(REGION_ORDER),
        "primary_region": PRIMARY_REGION,
        "model_formula": "mean_improvement_i ~ 1 + (1 | country), REML",
        "sensitivity_formula": "mean_improvement_i ~ 1 + (1 | country) + (1 | country_year), REML",
        "multiplicity_families": {
            str(family): int(size)
            for family, size in results.groupby("multiplicity_family").size().items()
        },
        "prediction_cache": str(cache_dir),
        "convergence": {
            "primary_all_converged": bool(results["converged"].all()),
            "sensitivity_all_converged": bool(nested["converged"].all()),
        },
        "software": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "caveats": {
            "conditional_on_observed_landscape": CONDITIONAL_CAVEAT,
            "scale": SCALE_NOTE,
            "dependence": DEPENDENCE_NOTE,
        },
    }
    (args.output_dir / "hierarchical_oof_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
