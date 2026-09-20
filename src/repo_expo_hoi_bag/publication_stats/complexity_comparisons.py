"""Matched model-level comparisons for the manuscript complexity claim.

The primary block compares the best model selected independently at each level
within each BAG and discovery arm, the covariate-only baseline, and the best
single exposure.  A second block holds each deployed depth-3 arm candidate fixed
across levels, which isolates model complexity from candidate selection.

All contrasts use existing subject-level LOCO out-of-fold predictions.  The
country-cluster bootstrap resamples whole countries and recomputes pooled R².
Raw two-sided p-values and Holm adjustments are retained so the publication
table can show the inferential family chosen during manuscript review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.publication_stats.paths import model_comparison_output_dir


OUTPUT_DIR = model_comparison_output_dir()
OOF_DIR = OUTPUT_DIR / "oof"
LEVELS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
LEVEL_LABELS = {
    "ols": "OLS",
    "xgb_tree_d1": "d1",
    "xgb_tree_d2": "d2",
    "xgb_tree_d3": "d3",
}
PAIRS = (
    ("xgb_tree_d1", "ols"),
    ("xgb_tree_d2", "ols"),
    ("xgb_tree_d3", "ols"),
    ("xgb_tree_d2", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d1"),
    ("xgb_tree_d3", "xgb_tree_d2"),
)
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260805


def _load_prediction(path: Path, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["row_id", "country", "y_true", prediction_column, "candidate_id"],
    ).rename(columns={prediction_column: "prediction"})
    finite = np.isfinite(frame["y_true"]) & np.isfinite(frame["prediction"])
    return frame.loc[finite].copy()


def _align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    merged = a.merge(b, on="row_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not np.allclose(merged["y_true_a"], merged["y_true_b"]):
        raise ValueError("The paired OOF files do not contain matching outcomes")
    if not (merged["country_a"].astype(str) == merged["country_b"].astype(str)).all():
        raise ValueError("The paired OOF files do not contain matching countries")
    return merged


def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
    sst = np.sum(np.square(y_true - y_true.mean()))
    return float(1.0 - np.sum(np.square(y_true - prediction)) / sst)


def _cluster_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "country": frame["country_a"].astype(str),
            "y": frame["y_true_a"].astype(float),
            "sse_a": np.square(frame["y_true_a"] - frame["prediction_a"]),
            "sse_b": np.square(frame["y_true_a"] - frame["prediction_b"]),
        }
    )
    work["y2"] = np.square(work["y"])
    return work.groupby("country", sort=True).agg(
        n=("y", "size"),
        sum_y=("y", "sum"),
        sum_y2=("y2", "sum"),
        sse_a=("sse_a", "sum"),
        sse_b=("sse_b", "sum"),
    )


def _bootstrap_delta_r2(
    frame: pd.DataFrame,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    stats = _cluster_sufficient_statistics(frame)
    n_countries = len(stats)
    counts = rng.multinomial(
        n_countries,
        np.repeat(1.0 / n_countries, n_countries),
        size=draws,
    )
    n = counts @ stats["n"].to_numpy(float)
    sum_y = counts @ stats["sum_y"].to_numpy(float)
    sum_y2 = counts @ stats["sum_y2"].to_numpy(float)
    sst = sum_y2 - np.square(sum_y) / n
    delta = (
        counts @ stats["sse_b"].to_numpy(float)
        - counts @ stats["sse_a"].to_numpy(float)
    ) / sst
    lower, upper = np.percentile(delta, [2.5, 97.5])
    p_value = 2.0 * min(np.mean(delta <= 0.0), np.mean(delta >= 0.0))
    p_value = min(1.0, max(float(p_value), 1.0 / draws))
    return float(lower), float(upper), p_value


def _candidate_identity(frame: pd.DataFrame) -> str:
    values = frame["candidate_id"].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"Expected one candidate identity, found {values.tolist()}")
    return str(values[0])


def _trajectory_paths(
    bag: str,
    family: str,
) -> tuple[dict[str, tuple[Path, str]], str]:
    if family == "covariate_baseline":
        paths = {
            level: (OOF_DIR / "best_syn" / bag / f"oof_{level}.parquet", "y_pred_base")
            for level in LEVELS
        }
        return paths, "Covariate baseline"
    labels = {
        "level_best_syn": "Level-selected synergy arm",
        "level_best_red": "Level-selected redundancy arm",
        "best_single": "Best single exposure",
        "best_syn": "Fixed deployed synergy-arm candidate",
        "best_red": "Fixed deployed redundancy-arm candidate",
    }
    paths = {
        level: (OOF_DIR / family / bag / f"oof_{level}.parquet", "y_pred_full")
        for level in LEVELS
    }
    return paths, labels[family]


def _validate_baselines(bag: str) -> None:
    """Prove that choosing one source family does not change baseline predictions."""
    for level in LEVELS:
        reference = _load_prediction(
            OOF_DIR / "best_syn" / bag / f"oof_{level}.parquet", "y_pred_base"
        ).sort_values("row_id")
        for family in ("best_red", "best_single"):
            other = _load_prediction(
                OOF_DIR / family / bag / f"oof_{level}.parquet", "y_pred_base"
            ).sort_values("row_id")
            if not reference["row_id"].reset_index(drop=True).equals(
                other["row_id"].reset_index(drop=True)
            ) or not np.array_equal(
                reference["prediction"].to_numpy(), other["prediction"].to_numpy()
            ):
                raise ValueError(f"Baseline mismatch for {bag}, {level}, {family}")


def generate(*, draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("level_selected", "level_best_syn"),
        ("level_selected", "level_best_red"),
        ("level_selected", "covariate_baseline"),
        ("level_selected", "best_single"),
        ("fixed_candidate", "best_syn"),
        ("fixed_candidate", "best_red"),
    )
    comparison_index = 0
    for bag in ("structural", "functional"):
        _validate_baselines(bag)
        for analysis_scope, family in specifications:
            paths, trajectory = _trajectory_paths(bag, family)
            loaded = {
                level: _load_prediction(path, column)
                for level, (path, column) in paths.items()
            }
            identities = {
                level: "covariates_only"
                if family == "covariate_baseline"
                else _candidate_identity(frame)
                for level, frame in loaded.items()
            }
            for model_a, model_b in PAIRS:
                paired = _align(loaded[model_a], loaded[model_b])
                rng = np.random.default_rng(seed + comparison_index)
                comparison_index += 1
                ci_lower, ci_upper, raw_p = _bootstrap_delta_r2(
                    paired, draws=draws, rng=rng
                )
                y_true = paired["y_true_a"].to_numpy(float)
                prediction_a = paired["prediction_a"].to_numpy(float)
                prediction_b = paired["prediction_b"].to_numpy(float)
                r2_a = _r2(y_true, prediction_a)
                r2_b = _r2(y_true, prediction_b)
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "bag": bag,
                        "trajectory": trajectory,
                        "model_a": LEVEL_LABELS[model_a],
                        "model_b": LEVEL_LABELS[model_b],
                        "comparison_a_vs_b": (
                            f"{LEVEL_LABELS[model_a]} vs {LEVEL_LABELS[model_b]}"
                        ),
                        "candidate_a": identities[model_a],
                        "candidate_b": identities[model_b],
                        "same_predictor_identity": identities[model_a] == identities[model_b],
                        "n_subjects": len(paired),
                        "n_countries": paired["country_a"].nunique(),
                        "r2_a": r2_a,
                        "r2_b": r2_b,
                        "delta_r2_a_minus_b": r2_a - r2_b,
                        "delta_r2_ci_lower": ci_lower,
                        "delta_r2_ci_upper": ci_upper,
                        "country_cluster_bootstrap_p_raw": raw_p,
                        "bootstrap_draws": draws,
                        "bootstrap_seed": seed + comparison_index - 1,
                    }
                )

    result = pd.DataFrame(rows)
    result["holm_p_within_bag_trajectory"] = np.nan
    for _, indices in result.groupby(
        ["analysis_scope", "bag", "trajectory"], sort=False
    ).groups.items():
        result.loc[indices, "holm_p_within_bag_trajectory"] = multipletests(
            result.loc[indices, "country_cluster_bootstrap_p_raw"], method="holm"
        )[1]
    result["holm_p_d3_claim_within_bag_trajectory"] = np.nan
    d3_rows = result["model_a"].eq("d3")
    for _, indices in result.loc[d3_rows].groupby(
        ["analysis_scope", "bag", "trajectory"], sort=False
    ).groups.items():
        result.loc[indices, "holm_p_d3_claim_within_bag_trajectory"] = multipletests(
            result.loc[indices, "country_cluster_bootstrap_p_raw"], method="holm"
        )[1]
    result["holm_p_within_analysis_scope"] = np.nan
    for _, indices in result.groupby("analysis_scope", sort=False).groups.items():
        result.loc[indices, "holm_p_within_analysis_scope"] = multipletests(
            result.loc[indices, "country_cluster_bootstrap_p_raw"], method="holm"
        )[1]
    result["holm_p_all_comparisons"] = multipletests(
        result["country_cluster_bootstrap_p_raw"], method="holm"
    )[1]
    return result


def _write_markdown(result: pd.DataFrame, path: Path) -> None:
    columns = [
        "analysis_scope",
        "bag",
        "trajectory",
        "comparison_a_vs_b",
        "same_predictor_identity",
        "r2_a",
        "r2_b",
        "delta_r2_a_minus_b",
        "delta_r2_ci_lower",
        "delta_r2_ci_upper",
        "country_cluster_bootstrap_p_raw",
        "holm_p_within_bag_trajectory",
        "holm_p_d3_claim_within_bag_trajectory",
        "holm_p_within_analysis_scope",
    ]
    rendered = result.loc[:, columns].copy()
    for column in (
        "r2_a",
        "r2_b",
        "delta_r2_a_minus_b",
        "delta_r2_ci_lower",
        "delta_r2_ci_upper",
        "country_cluster_bootstrap_p_raw",
        "holm_p_within_bag_trajectory",
        "holm_p_d3_claim_within_bag_trajectory",
        "holm_p_within_analysis_scope",
    ):
        rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
    path.write_text(
        "# Matched model-level comparisons\n\n"
        "Model A is the more complex level; positive delta R2 favours model A. "
        "Raw p-values are two-sided country-cluster bootstrap values.\n\n"
        + rendered.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate(draws=args.draws, seed=args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "complexity_pairwise_comparisons.csv"
    md_path = OUTPUT_DIR / "complexity_pairwise_comparisons.md"
    result.to_csv(csv_path, index=False)
    _write_markdown(result, md_path)
    print(f"Wrote {len(result)} comparisons to {csv_path}")
    print(f"Rendered report to {md_path}")


if __name__ == "__main__":
    main()
