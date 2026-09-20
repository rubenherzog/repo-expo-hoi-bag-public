"""Compute dependency-aware analyses requested during manuscript revision.

The analyses in this module use existing out-of-fold predictions and candidate
metrics. They do not refit predictive models.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.publication_stats.model_comparisons import (
    _r2_pooled,
    country_cluster_bootstrap,
    load_pair,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs" / "dedup" / "model_comparison" / "revision_inference"
N_DRAWS = 10_000
SEED = 20260801


def canonical_root() -> Path:
    explicit = os.environ.get("MANUSCRIPT_CANONICAL_ROOT", "").strip()
    bundle = Path(os.environ.get("REPRO_DATA_ROOT", ROOT / "REPRO_DATA_ROOT"))
    candidates = [
        Path(explicit) if explicit else None,
        bundle / "results/variant_a/families/pooled_oinfo_ladder/canonical/per_experiment",
        bundle / "runs/oinfo_only/variant_a/families/pooled_oinfo_ladder/canonical/per_experiment",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Canonical per-experiment metrics were not found. Set REPRO_DATA_ROOT to the "
        "analysis bundle or MANUSCRIPT_CANONICAL_ROOT to canonical/per_experiment."
    )


def model_complexity_summary() -> pd.DataFrame:
    canonical = canonical_root()
    rows: list[pd.DataFrame] = []
    for bag in ("structural", "functional"):
        path = canonical / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        frame = pd.read_parquet(path)
        frame = frame.loc[:, ["rung_id", "objective", "candidate_id", "order", "full_r2"]]
        selected = frame.loc[
            frame.groupby(["rung_id", "objective"], observed=True)["full_r2"].idxmax()
        ].copy()
        selected.insert(0, "bag", bag)
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    order = pd.CategoricalDtype(["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"], ordered=True)
    result["rung_id"] = result["rung_id"].astype(order)
    return result.sort_values(["bag", "objective", "rung_id"]).reset_index(drop=True)


def single_exposure_complexity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise and compare existing best-single OOF predictions by level.

    Each level selected its best single exposure independently.  The inferential
    contrast uses the paired OOF predictions of the selected d3 and OLS models and
    resamples whole countries, matching the manuscript's other predictive comparisons.
    It compares the selected model families rather than isolating complexity for a
    fixed exposure.  No predictive model is fitted here.
    """
    levels = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
    level_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for bag_index, bag in enumerate(("structural", "functional")):
        paths = {
            level: ROOT
            / "outputs"
            / "dedup"
            / "model_comparison"
            / "oof"
            / "best_single"
            / bag
            / f"oof_{level}.parquet"
            for level in levels
        }
        for level, path in paths.items():
            frame = pd.read_parquet(path)
            candidate_ids = frame["candidate_id"].dropna().astype(str).unique()
            if len(candidate_ids) != 1:
                raise ValueError(f"Expected one selected exposure in {path}, found {candidate_ids}")
            level_rows.append(
                {
                    "bag": bag,
                    "rung_id": level,
                    "selected_exposure": candidate_ids[0].removeprefix("best_single::"),
                    "full_r2": _r2_pooled(
                        frame["y_true"].to_numpy(float), frame["y_pred_full"].to_numpy(float)
                    ),
                    "n_subjects": len(frame),
                    "n_countries": frame["country"].nunique(),
                }
            )

        truth, pred_d3, pred_ols, countries = load_pair(
            paths["xgb_tree_d3"], paths["ols"]
        )
        result = country_cluster_bootstrap(
            truth,
            pred_d3,
            pred_ols,
            countries,
            "sq",
            N_DRAWS,
            np.random.default_rng(SEED + bag_index),
        )
        comparison_rows.append(
            {
                "bag": bag,
                "comparison": "Best single exposure at d3 minus best single exposure at OLS",
                "n_subjects": len(truth),
                "n_countries": len(np.unique(countries)),
                "r2_d3": _r2_pooled(truth, pred_d3),
                "r2_ols": _r2_pooled(truth, pred_ols),
                "delta_r2": result["delta_r2"],
                "delta_r2_ci_lo": result["delta_r2_ci_lo"],
                "delta_r2_ci_hi": result["delta_r2_ci_hi"],
                "country_cluster_bootstrap_p": result["delta_r2_p"],
                "bootstrap_draws": N_DRAWS,
                "bootstrap_unit": "country",
            }
        )
    return pd.DataFrame(level_rows), pd.DataFrame(comparison_rows)


def diversity_performance_ranges() -> pd.DataFrame:
    """Summarise the existing Fig. 3 candidate-level source data for Table ST6."""
    rows: list[dict[str, object]] = []
    source_root = (
        ROOT / "outputs" / "figures" / "dedup" / "paper" / "complete" / "source_data"
    )
    for bag, panel in (("structural", "a1_struct"), ("functional", "b1_func")):
        path = source_root / (
            f"fig3_diversity_v2_max_30_dedup_source_data_{panel}_diversity_scatter.csv"
        )
        candidates = pd.read_csv(path)
        for objective, group in candidates.groupby("objective", sort=False):
            rows.append(
                {
                    "bag": bag,
                    "objective": objective,
                    "n_candidates": len(group),
                    "minimum_loco_r2": group["global_oof_r2"].min(),
                    "maximum_loco_r2": group["global_oof_r2"].max(),
                }
            )
    return pd.DataFrame(rows)


def country_crossfit_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each arm's d3 candidate without using the evaluation country."""
    canonical = canonical_root()
    fold_rows: list[dict[str, object]] = []
    for bag in ("structural", "functional"):
        path = canonical / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["rung_id"] == "xgb_tree_d3"].copy()
        for objective, arm in (("o_min", "Synergy arm"), ("o_max", "Redundancy arm")):
            arm_frame = frame[frame["objective"] == objective]
            for country in sorted(arm_frame["fold_country"].unique()):
                training = arm_frame[arm_frame["fold_country"] != country].copy()
                training["sse"] = training["country_full_rmse"].pow(2) * training["n_test_scored"]
                ranking = training.groupby("candidate_id", as_index=False).agg(
                    training_sse=("sse", "sum"), training_n=("n_test_scored", "sum")
                )
                ranking["training_rmse"] = np.sqrt(ranking["training_sse"] / ranking["training_n"])
                winner = ranking.loc[ranking["training_rmse"].idxmin()]
                held_out = arm_frame[
                    (arm_frame["fold_country"] == country)
                    & (arm_frame["candidate_id"] == winner["candidate_id"])
                ].iloc[0]
                fold_rows.append(
                    {
                        "bag": bag,
                        "arm": arm,
                        "held_out_country": country,
                        "selected_candidate": winner["candidate_id"],
                        "selected_set_size": int(held_out["order"]),
                        "training_country_rmse": winner["training_rmse"],
                        "held_out_country_r2": held_out["country_full_r2"],
                        "held_out_country_rmse": held_out["country_full_rmse"],
                        "held_out_n": int(held_out["n_test_scored"]),
                    }
                )
    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for (bag, arm), group in folds.groupby(["bag", "arm"]):
        counts = group["selected_candidate"].value_counts()
        summary_rows.append(
            {
                "bag": bag,
                "arm": arm,
                "held_out_countries": len(group),
                "unique_selected_candidates": group["selected_candidate"].nunique(),
                "most_frequent_candidate": counts.index[0],
                "most_frequent_candidate_fraction": counts.iloc[0] / len(group),
                "median_held_out_country_r2": group["held_out_country_r2"].median(),
                "q1_held_out_country_r2": group["held_out_country_r2"].quantile(0.25),
                "q3_held_out_country_r2": group["held_out_country_r2"].quantile(0.75),
                "positive_r2_countries": int((group["held_out_country_r2"] > 0).sum()),
            }
        )
    return folds, pd.DataFrame(summary_rows)


def best_multivariate_vs_single() -> pd.DataFrame:
    """Compare each BAG's overall d3 winner with its best single exposure."""
    rows: list[dict[str, object]] = []
    for bag_index, (bag, family, arm) in enumerate(
        (
            ("structural", "best_red", "Redundancy arm"),
            ("functional", "best_syn", "Synergy arm"),
        )
    ):
        oof_root = ROOT / "outputs" / "dedup" / "model_comparison" / "oof"
        truth, pred_multivariate, pred_single, countries = load_pair(
            oof_root / family / bag / "oof_xgb_tree_d3.parquet",
            oof_root / "best_single" / bag / "oof_xgb_tree_d3.parquet",
        )
        result = country_cluster_bootstrap(
            truth,
            pred_multivariate,
            pred_single,
            countries,
            "sq",
            N_DRAWS,
            np.random.default_rng(SEED + bag_index),
        )
        rows.append(
            {
                "bag": bag,
                "selected_arm": arm,
                "n_subjects": len(truth),
                "n_countries": len(np.unique(countries)),
                "r2_multivariate": _r2_pooled(truth, pred_multivariate),
                "r2_single": _r2_pooled(truth, pred_single),
                "delta_r2": result["delta_r2"],
                "delta_r2_ci_lo": result["delta_r2_ci_lo"],
                "delta_r2_ci_hi": result["delta_r2_ci_hi"],
                "country_cluster_bootstrap_p": result["delta_r2_p"],
                "bootstrap_draws": N_DRAWS,
                "bootstrap_unit": "country",
            }
        )
    return pd.DataFrame(rows)


def overlap_aware_top50() -> pd.DataFrame:
    """Within-order arm-label randomisation retaining the full overlap structure."""
    canonical = canonical_root()
    rng = np.random.default_rng(SEED)
    subcomb = pd.read_csv(ROOT / "outputs/dedup/subcomb_oinfo/subcomb_enrichment.csv")
    rows = []
    for bag in ("structural", "functional"):
        path = canonical / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
        triplets = subcomb[
            (subcomb["bag"] == bag)
            & (subcomb["rung_id"] == "xgb_tree_d3")
            & (subcomb["order_k"] == 3)
        ][["candidate_id", "objective", "frac_neg"]]
        frame = pd.read_parquet(path)
        frame = frame[
            (frame["rung_id"] == "xgb_tree_d3")
            & frame["candidate_id"].isin(triplets["candidate_id"])
        ].copy()
        frame = frame.merge(
            triplets[["candidate_id", "frac_neg"]], on="candidate_id", validate="one_to_one"
        )
        if len(frame) != 100 or frame["objective"].value_counts().to_dict() != {"o_min": 50, "o_max": 50}:
            raise ValueError(f"Expected the selected top 50 per arm for {bag}")

        def statistics(labels: np.ndarray) -> tuple[float, float]:
            work = frame.assign(permuted_arm=labels)
            syn = work[work["permuted_arm"] == "o_min"]
            red = work[work["permuted_arm"] == "o_max"]
            return float(syn["order"].median() - red["order"].median()), float(
                syn["frac_neg"].median() - red["frac_neg"].median()
            )

        labels = frame["objective"].to_numpy(copy=True)
        observed = statistics(labels)
        null = np.empty((N_DRAWS, 2))
        for draw in range(N_DRAWS):
            # A single arm-label randomisation retains the selected candidates,
            # their feature overlap and the association between set size and
            # triplet content under the null.
            null[draw] = statistics(rng.permutation(labels))
        for metric_index, (metric, value) in enumerate(
            (("Median set-size difference", observed[0]), ("Median negative-Omega triplet-fraction difference", observed[1]))
        ):
            exceedances = int(np.sum(np.abs(null[:, metric_index]) >= abs(value)))
            rows.append(
                {
                    "bag": bag,
                    "contrast": "Synergy arm minus redundancy arm among top 50 per arm",
                    "metric": metric,
                    "observed_difference": value,
                    "null_mean": null[:, metric_index].mean(),
                    "null_sd": null[:, metric_index].std(ddof=1),
                    "permutation_p_two_sided": (exceedances + 1) / (N_DRAWS + 1),
                    "permutation_exceedances": exceedances,
                    "permutations": N_DRAWS,
                    "stratification": "none; complete selected candidate pool retained",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_complexity_summary().to_csv(OUTPUT / "model_complexity_best_per_level.csv", index=False)
    single_levels, single_comparison = single_exposure_complexity()
    single_levels.to_csv(OUTPUT / "single_exposure_complexity_per_level.csv", index=False)
    single_comparison.to_csv(OUTPUT / "single_exposure_complexity_paired.csv", index=False)
    diversity_performance_ranges().to_csv(
        OUTPUT / "diversity_performance_ranges_d3.csv", index=False
    )
    folds, summary = country_crossfit_selection()
    folds.to_csv(OUTPUT / "country_crossfit_selection_folds.csv", index=False)
    summary.to_csv(OUTPUT / "country_crossfit_selection_summary.csv", index=False)
    best_multivariate_vs_single().to_csv(
        OUTPUT / "best_multivariate_vs_single.csv", index=False
    )
    overlap_aware_top50().to_csv(OUTPUT / "overlap_aware_top50_permutation.csv", index=False)
    print(f"Wrote revision analyses to {OUTPUT}")


if __name__ == "__main__":
    main()
