#!/usr/bin/env python3
"""Paired hierarchical-bootstrap comparisons from stored subject-level OOF losses.

Every comparison uses the same country-stratified hierarchical resample for
both members of the pair: observed countries are fixed, country-years are
sampled within country, and subjects are sampled within the selected
country-year.  R² is recomputed in every replicate from SSE and SST, then
averaged with equal country weight.  No model is fitted by this stage.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages.compute_hierarchical_oof_bootstrap import (
    DRAWS,
    SEED,
    _country_year_blocks,
)
from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import (
    ARMS,
    BAGS,
    PRIMARY_REGION,
    RUNGS,
    country_year_map,
    load_subjects,
)

LEVEL_PAIRS = (
    ("xgb_tree_d1", "ols"),
    ("xgb_tree_d2", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d2"),
    ("xgb_tree_d2", "ols"),
    ("xgb_tree_d3", "xgb_tree_d1"),
    ("xgb_tree_d3", "ols"),
)
ALIGNMENT_COLUMNS = ("row_id", "country", "country_year", "y_true")

RESAMPLING_NOTE = (
    "Observed countries are fixed. Within each country, country-years are "
    "resampled with replacement; within each sampled country-year, subjects "
    "are resampled with replacement. The identical sampled rows are used for "
    "both models in every paired contrast."
)


@dataclass(frozen=True)
class ContrastSpec:
    """One independent paired OOF contrast, evaluated by one outer worker."""

    family: str | None
    question: str
    bag: str
    arm: str | None
    rung: str | None
    left_label: str
    right_label: str
    left: pd.DataFrame
    right: pd.DataFrame
    region: str = PRIMARY_REGION
    left_loss: str = "mean_candidate_loss"
    right_loss: str = "mean_candidate_loss"
    right_candidate_id: str | None = None


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort and validate two OOF score frames before a paired contrast."""
    left = left.sort_values("row_id", kind="stable").reset_index(drop=True)
    right = right.sort_values("row_id", kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"Paired OOF frames have different lengths: {len(left)} != {len(right)}")
    if left["row_id"].duplicated().any() or right["row_id"].duplicated().any():
        raise ValueError("A paired OOF frame has duplicate row_id values")
    for column in ALIGNMENT_COLUMNS:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        if column == "y_true":
            matches = np.array_equal(a, b)
        else:
            matches = np.array_equal(a.astype(str), b.astype(str))
        if not matches:
            raise ValueError(f"Paired OOF frames disagree on {column}")
    return left, right


def observed_paired_r2(
    left: pd.DataFrame, right: pd.DataFrame, *, left_loss: str, right_loss: str
) -> pd.DataFrame:
    """Country-specific R² for two models and their signed paired difference."""
    left, right = _aligned(left, right)
    rows: list[dict[str, object]] = []
    for country, block in left.groupby("country", sort=True):
        positions = block.index.to_numpy()
        y = block["y_true"].to_numpy(float)
        sst = float(np.square(y - y.mean()).sum())
        sse_left = float(left.loc[positions, left_loss].sum())
        sse_right = float(right.loc[positions, right_loss].sum())
        r2_left = 1.0 - sse_left / sst if sst > 0 else np.nan
        r2_right = 1.0 - sse_right / sst if sst > 0 else np.nan
        rows.append(
            {
                "country": str(country),
                "n_subjects": int(len(positions)),
                "sst": sst,
                "sse_left": sse_left,
                "sse_right": sse_right,
                "r2_left": r2_left,
                "r2_right": r2_right,
                "delta_r2_country": r2_left - r2_right,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired_r2(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_loss: str = "mean_candidate_loss",
    right_loss: str = "mean_candidate_loss",
    draws: int = DRAWS,
    seed: int = SEED,
    n_jobs: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return paired country-balanced ΔR² draws and observed country values.

    ``left - right`` is the reported direction.  The function draws row
    indices once per replicate and uses those *same indices* to calculate both
    SSEs.  Therefore the returned distribution is a paired contrast, rather
    than a difference between independently bootstrapped summaries.
    """
    left, right = _aligned(left, right)
    blocks, owners = _country_year_blocks(left)
    countries = np.unique(owners)
    by_country = {country: np.flatnonzero(owners == country) for country in countries}
    y = left["y_true"].to_numpy(float)
    loss_left = left[left_loss].to_numpy(float)
    loss_right = right[right_loss].to_numpy(float)
    seeds = np.random.SeedSequence(seed).spawn(draws)

    def chunk(offset: int, size: int) -> np.ndarray:
        estimates = np.empty(size, dtype=float)
        for draw in range(size):
            rng = np.random.default_rng(seeds[offset + draw])
            per_country = np.empty(len(countries), dtype=float)
            for position, country in enumerate(countries):
                country_blocks = by_country[country]
                selected_blocks = rng.choice(
                    country_blocks, size=len(country_blocks), replace=True
                )
                sampled_rows = np.concatenate(
                    [
                        rng.choice(blocks[index], size=len(blocks[index]), replace=True)
                        for index in selected_blocks
                    ]
                )
                values = y[sampled_rows]
                sst = float(np.square(values - values.mean()).sum())
                if sst <= 0:
                    per_country[position] = np.nan
                    continue
                # Both SSEs deliberately use the same sampled_rows vector.
                r2_left = 1.0 - float(loss_left[sampled_rows].sum()) / sst
                r2_right = 1.0 - float(loss_right[sampled_rows].sum()) / sst
                per_country[position] = r2_left - r2_right
            estimates[draw] = np.nanmean(per_country)
        return estimates

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
    return estimates, observed_paired_r2(left, right, left_loss=left_loss, right_loss=right_loss)


def summarise_paired(estimates: np.ndarray, observed: pd.DataFrame) -> dict[str, object]:
    """Percentile interval and two-sided paired bootstrap P value."""
    p_lower = (np.sum(estimates <= 0.0) + 1) / (len(estimates) + 1)
    p_upper = (np.sum(estimates >= 0.0) + 1) / (len(estimates) + 1)
    return {
        "observed_delta_r2": float(observed["delta_r2_country"].mean()),
        "bootstrap_median": float(np.median(estimates)),
        "ci_lo": float(np.percentile(estimates, 2.5)),
        "ci_hi": float(np.percentile(estimates, 97.5)),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "n_countries": int(len(observed)),
        "n_subjects": int(observed["n_subjects"].sum()),
    }


def _top20(subjects: pd.DataFrame, bag: str, arm: str, rung: str) -> pd.DataFrame:
    return _region(subjects, bag, arm, rung, PRIMARY_REGION)


def _region(subjects: pd.DataFrame, bag: str, arm: str, rung: str, region: str) -> pd.DataFrame:
    frame = subjects[
        subjects["bag"].eq(bag)
        & subjects["arm"].eq(arm)
        & subjects["rung"].eq(rung)
        & subjects["region"].eq(region)
    ]
    if frame.empty:
        raise ValueError(f"Missing {region} OOF scores for {bag}/{arm}/{rung}")
    return frame


def _queue_contrast(specs: list[ContrastSpec], **kwargs: object) -> None:
    """Queue an independent contrast; execution is deliberately outer-parallel."""
    # Call sites share their former _add_contrast signature; execution options
    # belong to the single outer pool rather than individual specifications.
    for key in ("draws", "seed", "n_jobs"):
        kwargs.pop(key, None)
    specs.append(ContrastSpec(**kwargs))


def _evaluate_contrast(spec: ContrastSpec, *, draws: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one contrast sequentially inside its allocated outer worker."""
    estimates, observed = bootstrap_paired_r2(
        spec.left, spec.right, left_loss=spec.left_loss, right_loss=spec.right_loss,
        draws=draws, seed=seed, n_jobs=1,
    )
    row = {
        "question": spec.question, "multiplicity_family": spec.family, "bag": spec.bag,
        "arm": spec.arm, "rung": spec.rung, "left_model": spec.left_label,
        "right_model": spec.right_label, "right_candidate_id": spec.right_candidate_id,
        "comparison_direction": f"{spec.left_label} minus {spec.right_label}",
        "region": spec.region, **summarise_paired(estimates, observed),
    }
    countries = observed.assign(
        question=spec.question, multiplicity_family=spec.family, bag=spec.bag, arm=spec.arm,
        rung=spec.rung, left_model=spec.left_label, right_model=spec.right_label,
        right_candidate_id=spec.right_candidate_id, region=spec.region,
    )
    return row, countries


def _execute_contrasts(
    specs: list[ContrastSpec], *, draws: int, seed: int, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run independent contrasts in one reusable outer worker pool.

    Parallelising whole contrasts keeps one worker busy for all its bootstrap
    draws.  It avoids repeatedly splitting 10,000 draws into tiny 250-draw
    chunks and launching a new inner pool for every contrast.
    """
    if n_jobs <= 1:
        completed = [_evaluate_contrast(spec, draws=draws, seed=seed) for spec in specs]
    else:
        completed = Parallel(n_jobs=min(n_jobs, len(specs)), backend="loky", max_nbytes="1M")(
            delayed(_evaluate_contrast)(spec, draws=draws, seed=seed) for spec in specs
        )
    rows, countries = zip(*completed)
    return pd.DataFrame(rows), pd.concat(countries, ignore_index=True)


def apply_holm(results: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust P values separately in each prespecified scientific family."""
    results = results.copy()
    results["holm_p"] = np.nan
    for family, block in results.dropna(subset=["multiplicity_family"]).groupby("multiplicity_family", sort=False):
        results.loc[block.index, "holm_p"] = multipletests(
            block["p_two_sided"].to_numpy(float), method="holm"
        )[1]
    return results


def compute(
    subjects: pd.DataFrame, best_single: dict[tuple[str, str], pd.DataFrame], *, draws: int,
    seed: int, n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the primary Top20 paired OOF comparison families."""
    specs: list[ContrastSpec] = []
    for bag in BAGS:
        # Top20 against the matched baseline: eight closely related comparisons per BAG.
        family = f"top20_vs_baseline:{bag}"
        for arm in ARMS:
            for rung in RUNGS:
                frame = _top20(subjects, bag, arm, rung)
                _queue_contrast(
                    specs, family=family, question="top20_vs_baseline", bag=bag,
                    arm=arm, left_label="Top20 set", right_label="matched baseline",
                    left=frame, right=frame, left_loss="mean_candidate_loss",
                    right_loss="baseline_loss", rung=rung, draws=draws, seed=seed, n_jobs=n_jobs,
                )
        # The globally selected single-exposure OOF score is an already-stored
        # model prediction, converted to its per-subject squared loss here. It
        # is not cross-selected and no model is fitted by this contrast.
        family = f"top20_vs_best_single:{bag}"
        for arm in ARMS:
            for rung in RUNGS:
                single = best_single[(bag, rung)]
                _queue_contrast(
                    specs, family=family, question="top20_vs_best_single", bag=bag,
                    arm=arm, left_label="Top20 set", right_label="best single exposure",
                    left=_top20(subjects, bag, arm, rung), right=single, rung=rung,
                    draws=draws, seed=seed, n_jobs=n_jobs,
                    right_candidate_id=str(single["candidate_id"].iloc[0]),
                )
        # Four paired arm comparisons per BAG, one at each model level.
        family = f"synergy_vs_redundancy:{bag}"
        for rung in RUNGS:
            _queue_contrast(
                specs, family=family, question="synergy_vs_redundancy", bag=bag,
                arm=None, left_label="synergy Top20 set", right_label="redundancy Top20 set",
                left=_top20(subjects, bag, "o_min", rung),
                right=_top20(subjects, bag, "o_max", rung), rung=rung, draws=draws, seed=seed, n_jobs=n_jobs,
            )
        # Six pairwise levels are a small family within each BAG x arm trajectory.
        for arm in ARMS:
            family = f"model_complexity:{bag}:{arm}"
            for high, low in LEVEL_PAIRS:
                _queue_contrast(
                    specs, family=family, question="model_complexity", bag=bag,
                    arm=arm, left_label=high, right_label=low,
                    left=_top20(subjects, bag, arm, high),
                    right=_top20(subjects, bag, arm, low), rung=None,
                    draws=draws, seed=seed, n_jobs=n_jobs,
                )
    results, countries = _execute_contrasts(specs, draws=draws, seed=seed, n_jobs=n_jobs)
    return apply_holm(results), countries


def _load_cell(cache: Path, years: pd.DataFrame, bag: str, arm: str, rung: str) -> pd.DataFrame:
    path = cache / f"all__{bag}__{arm}__{rung}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF loss accumulator: {path}")
    frame = pd.read_parquet(path)
    frame = frame.merge(years, on="row_id", how="left", validate="many_to_one")
    if frame["country_year"].isna().any():
        raise ValueError(f"country_year is missing in {path}")
    return frame


def load_top1(cache: Path, years: pd.DataFrame) -> pd.DataFrame:
    """Recover exact Top1 OOF losses from the persisted rank-ordered sums.

    The fitting stage stores cumulative squared losses in global performance-rank
    order.  The first cumulative row therefore is Top1 exactly, with no fitting
    or approximation required.
    """
    frames: list[pd.DataFrame] = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                stem = f"all__{bag}__{arm}__{rung}"
                path = cache / f"{stem}__cumulative.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing Top1 cumulative OOF cache: {path}")
                with np.load(path, allow_pickle=True) as cached:
                    cumulative = cached["cumulative"]
                    candidate_order = cached["candidate_order"]
                    if cumulative.ndim != 2 or cumulative.shape[0] < 1:
                        raise ValueError(f"Invalid cumulative loss matrix: {path}")
                    if len(candidate_order) < 1:
                        raise ValueError(f"Missing candidate order in: {path}")
                    frame = pd.DataFrame({
                        "row_id": cached["row_id"],
                        "country": cached["country"].astype(str),
                        "y_true": cached["y_true"],
                        "baseline_loss": cached["baseline_loss"],
                        "mean_candidate_loss": cumulative[0],
                        "region": "Top1",
                        "bag": bag,
                        "arm": arm,
                        "rung": rung,
                        "candidate_id": str(candidate_order[0]),
                    })
                frame = frame.merge(years, on="row_id", how="left", validate="one_to_one")
                if frame["country_year"].isna().any():
                    raise ValueError(f"country_year is missing for Top1 cache: {path}")
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_top20(cache: Path, years: pd.DataFrame) -> pd.DataFrame:
    """Load only the primary region, avoiding a 1.46m-row in-memory table."""
    frames = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                frame = _load_cell(cache, years, bag, arm, rung)
                frames.append(frame[frame["region"].eq(PRIMARY_REGION)].copy())
    return pd.concat(frames, ignore_index=True)


def load_best_single_oof(runtime: Path, years: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Load already-generated globally selected best-single OOF predictions.

    The selected single is defined by the existing country-balanced metrics;
    its stored subject predictions are converted to squared loss only.  This
    performs no selection and no model fitting.
    """
    root = runtime / "results/analysis_runs/paper_reanalysis_k10/main_statistics/model_comparison/oof/level_best_single"
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for bag in BAGS:
        for rung in RUNGS:
            path = root / bag / f"oof_{rung}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing best-single OOF predictions: {path}")
            frame = pd.read_parquet(path)
            needed = {"row_id", "country", "y_true", "y_pred_full", "candidate_id"}
            missing = needed.difference(frame.columns)
            if missing:
                raise ValueError(f"Best-single OOF {path} lacks columns: {sorted(missing)}")
            frame = frame.merge(years, on="row_id", how="left", validate="one_to_one")
            if frame["country_year"].isna().any():
                raise ValueError(f"country_year is missing in best-single OOF {path}")
            if frame["candidate_id"].nunique() != 1:
                raise ValueError(f"Best-single OOF {path} has multiple candidate IDs")
            frame = frame.assign(
                mean_candidate_loss=np.square(
                    frame["y_pred_full"].to_numpy(float) - frame["y_true"].to_numpy(float)
                )
            )
            if not np.isfinite(frame["mean_candidate_loss"].to_numpy(float)).all():
                raise ValueError(f"Best-single OOF {path} has non-finite squared losses")
            frames[(bag, rung)] = frame
    return frames


def compute_frontier(
    cache: Path, years: pd.DataFrame, *, draws: int, seed: int, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream non-primary K-grid cells; no unneeded OOF region is retained."""
    specs: list[ContrastSpec] = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                cell = _load_cell(cache, years, bag, arm, rung)
                for region, frame in cell.groupby("region", sort=False):
                    if region == PRIMARY_REGION:
                        continue
                    _queue_contrast(
                        specs, family=None, question="topk_frontier_sensitivity", bag=bag,
                        arm=arm, left_label=f"{region} set", right_label="matched baseline",
                        left=frame, right=frame, left_loss="mean_candidate_loss",
                        right_loss="baseline_loss", rung=rung, region=region,
                        draws=draws, seed=seed, n_jobs=n_jobs,
                    )
    return _execute_contrasts(specs, draws=draws, seed=seed, n_jobs=n_jobs)


def compute_frontier_vs_best_single(
    cache: Path,
    years: pd.DataFrame,
    best_single: dict[tuple[str, str], pd.DataFrame],
    *,
    draws: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive non-Top20 K-grid contrasts against stored best-single OOF.

    Top20 itself is already present in the primary comparison table and is not
    recalculated here.  ``ALL`` is intentionally excluded because it is not
    plotted in the K-frontier figures.
    """
    specs: list[ContrastSpec] = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                single = best_single[(bag, rung)]
                cell = _load_cell(cache, years, bag, arm, rung)
                for region, frame in cell.groupby("region", sort=False):
                    if region in {PRIMARY_REGION, "ALL"}:
                        continue
                    _queue_contrast(
                        specs, family=None, question="topk_vs_best_single_sensitivity",
                        bag=bag, arm=arm, rung=rung, region=region,
                        left_label=f"{region} set", right_label="best single exposure",
                        left=frame, right=single, right_candidate_id=str(single["candidate_id"].iloc[0]),
                        draws=draws, seed=seed, n_jobs=n_jobs,
                    )
    return _execute_contrasts(specs, draws=draws, seed=seed, n_jobs=n_jobs)


def compute_top1_comparisons(
    top1: pd.DataFrame,
    best_single: dict[tuple[str, str], pd.DataFrame],
    *,
    draws: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top1 sensitivity contrasts versus baseline and best single, without refitting."""
    specs: list[ContrastSpec] = []
    for bag in BAGS:
        for arm in ARMS:
            for rung in RUNGS:
                frame = _region(top1, bag, arm, rung, "Top1")
                _queue_contrast(
                    specs, family=None, question="top1_vs_baseline_sensitivity",
                    bag=bag, arm=arm, rung=rung, region="Top1",
                    left_label="Top1 set", right_label="matched baseline", left=frame, right=frame,
                    left_loss="mean_candidate_loss", right_loss="baseline_loss",
                    draws=draws, seed=seed, n_jobs=n_jobs,
                )
                single = best_single[(bag, rung)]
                _queue_contrast(
                    specs, family=None, question="top1_vs_best_single_sensitivity",
                    bag=bag, arm=arm, rung=rung, region="Top1",
                    left_label="Top1 set", right_label="best single exposure",
                    left=frame, right=single, right_candidate_id=str(single["candidate_id"].iloc[0]),
                    draws=draws, seed=seed, n_jobs=n_jobs,
                )
    return _execute_contrasts(specs, draws=draws, seed=seed, n_jobs=n_jobs)


def compute_top1_complexity_and_arm(
    top1: pd.DataFrame, *, draws: int, seed: int, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top1 model-level and Min O versus Max O contrasts, without refitting."""
    specs: list[ContrastSpec] = []
    for bag in BAGS:
        family = f"top1_min_o_vs_max_o:{bag}"
        for rung in RUNGS:
            _queue_contrast(
                specs, family=family, question="top1_min_o_vs_max_o", bag=bag,
                arm=None, rung=rung, region="Top1", left_label="Min O Top1 set",
                right_label="Max O Top1 set", left=_region(top1, bag, "o_min", rung, "Top1"),
                right=_region(top1, bag, "o_max", rung, "Top1"),
                draws=draws, seed=seed, n_jobs=n_jobs,
            )
        for arm in ARMS:
            family = f"top1_model_complexity:{bag}:{arm}"
            for high, low in LEVEL_PAIRS:
                _queue_contrast(
                    specs, family=family, question="top1_model_complexity", bag=bag,
                    arm=arm, rung=None, region="Top1", left_label=high, right_label=low,
                    left=_region(top1, bag, arm, high, "Top1"),
                    right=_region(top1, bag, arm, low, "Top1"),
                    draws=draws, seed=seed, n_jobs=n_jobs,
                )
    results, by_country = _execute_contrasts(specs, draws=draws, seed=seed, n_jobs=n_jobs)
    return apply_holm(results), by_country


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--run-id", default="hierarchical_oof_kgrid")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--best-single-frontier-only", action="store_true",
        help="Compute only non-Top20 K-grid contrasts against stored best-single OOF.",
    )
    parser.add_argument(
        "--top1-only", action="store_true",
        help="Compute only Top1 sensitivity contrasts from persisted cumulative OOF losses.",
    )
    parser.add_argument(
        "--top1-complexity-only", action="store_true",
        help="Compute only Top1 model-complexity and Min O versus Max O contrasts.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[3]
        / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/paired_bootstrap",
    )
    args = parser.parse_args()
    cache = args.cache_dir or (
        args.repro_data_root.resolve() / "work/analysis_runs" / args.run_id / "hierarchical_oof/loss_cache"
    )
    years = pd.concat([country_year_map(bag) for bag in BAGS], ignore_index=True).drop_duplicates("row_id")
    best_single = load_best_single_oof(args.repro_data_root.resolve(), years)
    if args.top1_complexity_only:
        top1 = load_top1(cache, years)
        results, by_country = compute_top1_complexity_and_arm(
            top1, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "paired_hierarchical_oof_top1_complexity.csv", index=False)
        by_country.to_csv(args.output_dir / "paired_hierarchical_oof_top1_complexity_by_country.csv", index=False)
        manifest = {
            "stage": "compute_hierarchical_oof_top1_complexity",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "draws": args.draws, "seed": args.seed, "prediction_cache": str(cache),
            "resampling": RESAMPLING_NOTE,
            "estimands": ["mean over countries of paired Top1 model-level R2 differences",
                           "mean over countries of paired (R2_MinO - R2_MaxO)"],
            "refitting": "none; Top1 is recovered exactly from persisted cumulative OOF losses",
            "families": {"Top1 model complexity": "6 tests within BAG x arm",
                         "Top1 Min O versus Max O": "4 tests within BAG"},
        }
        (args.output_dir / "paired_hierarchical_oof_top1_complexity_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"Saved: {args.output_dir}")
        return
    if args.top1_only:
        top1 = load_top1(cache, years)
        results, by_country = compute_top1_comparisons(
            top1, best_single, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "paired_hierarchical_oof_top1_sensitivity.csv", index=False)
        by_country.to_csv(args.output_dir / "paired_hierarchical_oof_top1_sensitivity_by_country.csv", index=False)
        manifest = {
            "stage": "compute_hierarchical_oof_top1_sensitivity",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "draws": args.draws, "seed": args.seed, "prediction_cache": str(cache),
            "resampling": RESAMPLING_NOTE,
            "estimands": ["mean over countries of paired (R2_Top1 - R2_baseline)",
                           "mean over countries of paired (R2_Top1 - R2_best_single)"],
            "refitting": "none; Top1 is recovered exactly from persisted cumulative OOF losses",
            "scope": "Top1 descriptive sensitivity; no multiplicity correction",
        }
        (args.output_dir / "paired_hierarchical_oof_top1_sensitivity_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"Saved: {args.output_dir}")
        return
    if args.best_single_frontier_only:
        results, by_country = compute_frontier_vs_best_single(
            cache, years, best_single, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "paired_hierarchical_oof_best_single_frontier_sensitivity.csv", index=False)
        by_country.to_csv(args.output_dir / "paired_hierarchical_oof_best_single_frontier_sensitivity_by_country.csv", index=False)
        manifest = {
            "stage": "compute_hierarchical_oof_best_single_frontier_sensitivity",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "draws": args.draws, "seed": args.seed, "prediction_cache": str(cache),
            "resampling": RESAMPLING_NOTE,
            "estimand": "mean over countries of paired (R2_TopK - R2_best_single)",
            "refitting": "none; derived entirely from stored per-subject OOF losses",
            "scope": "K=5,10,15,30,40,50,100; Top20 is reused from the primary table; ALL excluded",
        }
        (args.output_dir / "paired_hierarchical_oof_best_single_frontier_sensitivity_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"Saved: {args.output_dir}")
        return
    subjects = load_top20(cache, years)
    primary_results, primary_by_country = compute(subjects, best_single, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs)
    frontier_results, frontier_by_country = compute_frontier(cache, years, draws=args.draws, seed=args.seed, n_jobs=args.n_jobs)
    results = pd.concat([primary_results, frontier_results], ignore_index=True)
    by_country = pd.concat([primary_by_country, frontier_by_country], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "paired_hierarchical_oof_comparisons.csv", index=False)
    by_country.to_csv(args.output_dir / "paired_hierarchical_oof_comparisons_by_country.csv", index=False)
    manifest = {
        "stage": "compute_hierarchical_oof_comparisons", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "draws": args.draws, "seed": args.seed, "prediction_cache": str(cache),
        "resampling": RESAMPLING_NOTE, "estimand": "mean over countries of paired (R2_left - R2_right)",
        "refitting": "none; derived entirely from stored per-subject OOF losses",
        "families": {"top20_vs_baseline": "8 tests within BAG", "top20_vs_best_single": "8 tests within BAG", "synergy_vs_redundancy": "4 tests within BAG", "model_complexity": "6 tests within BAG x arm", "topk_frontier_sensitivity": "descriptive; no adjusted P"},
        "best_single_source": "Stored level_best_single OOF predictions; per-subject squared losses are derived without refitting.",
    }
    (args.output_dir / "paired_hierarchical_oof_comparisons_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
