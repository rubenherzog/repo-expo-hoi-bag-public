#!/usr/bin/env python3
"""Hierarchical bootstrap of the Top-K region's incremental R2 over baseline.

The mixed model reports an intercept on the squared-error scale. This stage
answers the same question directly on the R2 scale, by resampling the nested
data structure instead of assuming a variance decomposition.

Estimand
--------
Within a held-out country ``c``::

    dR2_c = (SSE_baseline_c - mean_SSE_TopK_c) / SST_c

and the canonical country-balanced quantity is ``dR2 = mean_c(dR2_c)``, the same
unweighted-mean-over-countries estimand the paper uses elsewhere.

Resampling (three nested levels, countries held fixed)
------------------------------------------------------
1. the observed countries are kept, not resampled: they are the strata;
2. within a country, country-years are resampled with replacement;
3. within a sampled country-year, subjects are resampled with replacement.

This propagates both the between-country-year and the within-country-year
components, which matters because the exposome is measured at country-year
resolution and subjects inside one country-year share an exposure vector.

No model is refitted. Every quantity is re-aggregated from the stored
per-subject ``y_true``, ``baseline_loss`` and ``mean_candidate_loss``, which are
sums over subjects and therefore exactly recomputable under resampling.

Multiplicity
------------
Top20 is the prespecified primary region and Holm is applied across exactly the
four synergy/redundancy x d2/d3 tests within each BAG. The remaining K are a
sensitivity grid describing how the effect decays as the region widens; they are
not a menu to select K from, and they do not enter the primary family.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import (
    BAGS,
    ARMS,
    RUNGS,
    PRIMARY_REGION,
    PRIMARY_RUNGS,
    country_year_map,
    load_subjects,
)

ROOT = frontier.ROOT

DRAWS = 10_000
SEED = 20260922

RESAMPLING_NOTE = (
    "Countries are held fixed as strata. Within each country, country-years are "
    "resampled with replacement; within each sampled country-year, subjects are "
    "resampled with replacement. SSE and SST are re-aggregated from the stored "
    "per-subject losses, so no model is refitted."
)
K_GRID_NOTE = (
    "Top20 is the prespecified primary region. The remaining K are a sensitivity "
    "grid showing how the effect decays as the region widens; they are not used "
    "to select K and are excluded from the primary Holm family."
)
SCALE_NOTE = (
    "This estimand is on the R2 scale and is directly comparable to the observed "
    "candidate delta-R2. The mixed-model beta0 is on the squared-error scale; the "
    "two agree in sign because SST is constant within a resampled country."
)


def _country_year_blocks(frame: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    """Row positions of each country-year, and the country each one belongs to."""
    blocks: list[np.ndarray] = []
    owners: list[str] = []
    for (country, year), index in frame.groupby(
        ["country", "country_year"], sort=True
    ).indices.items():
        blocks.append(np.asarray(index, dtype=np.int64))
        owners.append(str(country))
    return blocks, np.asarray(owners)


def observed_delta_r2(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-country incremental R2 of the region over the baseline."""
    rows = []
    for country, block in frame.groupby("country", sort=True):
        y = block["y_true"].to_numpy(float)
        sst = float(((y - y.mean()) ** 2).sum())
        sse_base = float(block["baseline_loss"].sum())
        sse_cand = float(block["mean_candidate_loss"].sum())
        rows.append(
            {
                "country": str(country),
                "n_subjects": int(len(block)),
                "sst": sst,
                "sse_baseline": sse_base,
                "sse_candidate": sse_cand,
                "delta_r2_country": (sse_base - sse_cand) / sst if sst > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_cell(
    frame: pd.DataFrame, *, draws: int = DRAWS, seed: int = SEED, n_jobs: int = 1
) -> tuple[np.ndarray, pd.DataFrame]:
    """Country-balanced delta-R2 across ``draws`` nested-resampling replicates."""
    frame = frame.reset_index(drop=True)
    blocks, owners = _country_year_blocks(frame)
    countries = np.unique(owners)
    # Country-year blocks belonging to each country, as positions into ``blocks``.
    by_country = {country: np.flatnonzero(owners == country) for country in countries}

    y = frame["y_true"].to_numpy(float)
    loss_base = frame["baseline_loss"].to_numpy(float)
    loss_cand = frame["mean_candidate_loss"].to_numpy(float)

    # One independent seed per replicate, spawned once up front. The seed of
    # replicate i does not depend on how the replicates are grouped, so results
    # are identical for any n_jobs and reproducible for a given seed.
    seeds = np.random.SeedSequence(seed).spawn(draws)

    def chunk(offset: int, size: int) -> np.ndarray:
        """Replicates ``offset`` .. ``offset + size``, each with its own seed."""
        out = np.empty(size, dtype=float)
        for draw in range(size):
            rng = np.random.default_rng(seeds[offset + draw])
            per_country = np.empty(len(countries), dtype=float)
            for position, country in enumerate(countries):
                candidates = by_country[country]
                chosen = rng.choice(candidates, size=len(candidates), replace=True)
                # Resample subjects within each drawn country-year, with replacement.
                picks = [
                    rng.choice(blocks[index], size=len(blocks[index]), replace=True)
                    for index in chosen
                ]
                rows = np.concatenate(picks)
                values = y[rows]
                sst = float(((values - values.mean()) ** 2).sum())
                if sst <= 0:
                    per_country[position] = np.nan
                    continue
                per_country[position] = (
                    float(loss_base[rows].sum()) - float(loss_cand[rows].sum())
                ) / sst
            out[draw] = np.nanmean(per_country)
        return out

    if n_jobs <= 1:
        estimates = chunk(0, draws)
    else:
        edges = np.linspace(0, draws, n_jobs + 1).astype(int)
        parts = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
            delayed(chunk)(int(start), int(stop - start))
            for start, stop in zip(edges[:-1], edges[1:])
            if stop > start
        )
        estimates = np.concatenate(parts)

    return estimates, observed_delta_r2(frame)


def summarise(estimates: np.ndarray, observed: pd.DataFrame) -> dict[str, object]:
    positive = float(np.mean(estimates > 0))
    return {
        "observed_delta_r2": float(observed["delta_r2_country"].mean()),
        "bootstrap_median": float(np.median(estimates)),
        "ci_lo": float(np.percentile(estimates, 2.5)),
        "ci_hi": float(np.percentile(estimates, 97.5)),
        "fraction_bootstrap_positive": positive,
        # One-sided bootstrap P for H0: delta-R2 <= 0, with the usual +1/(B+1)
        # correction so it can never be exactly zero.
        "p_one_sided": float((np.sum(estimates <= 0) + 1) / (len(estimates) + 1)),
        "n_countries": int(len(observed)),
        "n_subjects": int(observed["n_subjects"].sum()),
    }


def apply_holm(results: pd.DataFrame) -> pd.DataFrame:
    """Holm across exactly the four primary Top20 tests within each BAG."""
    frame = results.copy()
    frame["holm_p"] = np.nan
    frame["multiplicity_family"] = pd.NA

    # OLS and d1 are descriptive levels and never enter the primary family.
    primary = frame[frame["region"].eq(PRIMARY_REGION) & frame["rung"].isin(PRIMARY_RUNGS)]
    for bag, block in primary.groupby("bag", sort=False):
        if len(block) != 4:
            raise ValueError(f"Primary family for {bag} has {len(block)} tests, expected 4")
        frame.loc[block.index, "multiplicity_family"] = f"{bag}:{PRIMARY_REGION}"
        frame.loc[block.index, "holm_p"] = multipletests(
            block["p_one_sided"].to_numpy(float), method="holm"
        )[1]

    frame.loc[~frame["rung"].isin(PRIMARY_RUNGS), "multiplicity_family"] = (
        "descriptive level (not corrected)"
    )
    frame.loc[frame["region"].ne(PRIMARY_REGION), "multiplicity_family"] = (
        "sensitivity grid (not corrected)"
    )
    return frame


def _region_sort_key(region: str) -> tuple[int, int]:
    return (1, 0) if region == "ALL" else (0, int(region.removeprefix("Top")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--run-id", default="hierarchical_oof_kgrid")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap",
    )
    args = parser.parse_args()

    runtime = args.repro_data_root.resolve()
    cache_dir = args.cache_dir or (
        runtime / "work/analysis_runs" / args.run_id / "hierarchical_oof/loss_cache"
    )

    years = pd.concat([country_year_map(bag) for bag in BAGS], ignore_index=True)
    years = years.drop_duplicates("row_id")
    subjects = load_subjects(cache_dir, years)

    regions = sorted(subjects["region"].unique(), key=_region_sort_key)
    print(f"{len(regions)} regions: {regions}", flush=True)

    summaries: list[dict[str, object]] = []
    country_rows: list[pd.DataFrame] = []

    for bag in BAGS:
        for rung in RUNGS:
            for arm in ARMS:
                for region in regions:
                    block = subjects[
                        subjects["bag"].eq(bag)
                        & subjects["arm"].eq(arm)
                        & subjects["rung"].eq(rung)
                        & subjects["region"].eq(region)
                    ]
                    if block.empty:
                        continue
                    estimates, observed = bootstrap_cell(
                        block, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs
                    )
                    summaries.append(
                        {
                            "bag": bag,
                            "arm": arm,
                            "rung": rung,
                            "region": region,
                            "K": block["n_candidate_predictions"].iloc[0],
                            **summarise(estimates, observed),
                        }
                    )
                    country_rows.append(
                        observed.assign(bag=bag, arm=arm, rung=rung, region=region)
                    )
                    print(
                        f"  {bag}/{arm}/{rung}/{region}: "
                        f"dR2={summaries[-1]['observed_delta_r2']:+.4f} "
                        f"CI[{summaries[-1]['ci_lo']:+.4f},{summaries[-1]['ci_hi']:+.4f}] "
                        f"P={summaries[-1]['p_one_sided']:.4f}",
                        flush=True,
                    )

    results = apply_holm(pd.DataFrame(summaries))
    countries = pd.concat(country_rows, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "hierarchical_oof_bootstrap_results.csv", index=False)
    countries.to_csv(args.output_dir / "hierarchical_oof_bootstrap_by_country.csv", index=False)

    manifest = {
        "stage": "compute_hierarchical_oof_bootstrap",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "draws": args.draws,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "prediction_cache": str(cache_dir),
        "estimand": "mean over countries of (SSE_baseline - mean_SSE_TopK) / SST",
        "resampling": RESAMPLING_NOTE,
        "k_grid": K_GRID_NOTE,
        "scale": SCALE_NOTE,
        "primary_region": PRIMARY_REGION,
        "refitting": "none; derived entirely from stored per-subject losses",
        "software": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    (args.output_dir / "hierarchical_oof_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
