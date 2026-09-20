"""Pairwise model-level comparisons with the d2 arm winners held fixed."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.publication_stats.complexity_comparisons import (
    DEFAULT_DRAWS,
    LEVEL_LABELS,
    LEVELS,
    OOF_DIR,
    PAIRS,
    _align,
    _bootstrap_delta_r2,
    _candidate_identity,
    _load_prediction,
    _r2,
)
from repo_expo_hoi_bag.publication_stats.paths import model_comparison_output_dir


OUTPUT_DIR = model_comparison_output_dir()
DEFAULT_SEED = 20260920
FAMILIES = {
    "d2_best_syn": "Fixed d2 synergy-arm candidate",
    "d2_best_red": "Fixed d2 redundancy-arm candidate",
}


def generate(*, draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for bag in ("structural", "functional"):
        for family, trajectory in FAMILIES.items():
            loaded = {
                level: _load_prediction(
                    OOF_DIR / family / bag / f"oof_{level}.parquet", "y_pred_full"
                )
                for level in LEVELS
            }
            identities = {
                level: _candidate_identity(frame) for level, frame in loaded.items()
            }
            if len(set(identities.values())) != 1:
                raise ValueError(
                    f"The d2 candidate is not fixed for {bag}, {family}: {identities}"
                )
            for model_a, model_b in PAIRS:
                paired = _align(loaded[model_a], loaded[model_b])
                rng = np.random.default_rng(seed + comparison_index)
                comparison_index += 1
                lower, upper, raw_p = _bootstrap_delta_r2(
                    paired, draws=draws, rng=rng
                )
                y_true = paired["y_true_a"].to_numpy(float)
                prediction_a = paired["prediction_a"].to_numpy(float)
                prediction_b = paired["prediction_b"].to_numpy(float)
                rows.append(
                    {
                        "analysis_scope": "fixed_d2_candidate",
                        "bag": bag,
                        "trajectory": trajectory,
                        "model_a": LEVEL_LABELS[model_a],
                        "model_b": LEVEL_LABELS[model_b],
                        "comparison_a_vs_b": (
                            f"{LEVEL_LABELS[model_a]} vs {LEVEL_LABELS[model_b]}"
                        ),
                        "candidate_a": identities[model_a],
                        "candidate_b": identities[model_b],
                        "same_predictor_identity": True,
                        "n_subjects": len(paired),
                        "n_countries": paired["country_a"].nunique(),
                        "r2_a": _r2(y_true, prediction_a),
                        "r2_b": _r2(y_true, prediction_b),
                        "delta_r2_a_minus_b": (
                            _r2(y_true, prediction_a) - _r2(y_true, prediction_b)
                        ),
                        "delta_r2_ci_lower": lower,
                        "delta_r2_ci_upper": upper,
                        "country_cluster_bootstrap_p_raw": raw_p,
                        "bootstrap_draws": draws,
                        "bootstrap_seed": seed + comparison_index - 1,
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_p_within_bag_trajectory"] = np.nan
    for _, indices in result.groupby(["bag", "trajectory"], sort=False).groups.items():
        result.loc[indices, "holm_p_within_bag_trajectory"] = multipletests(
            result.loc[indices, "country_cluster_bootstrap_p_raw"], method="holm"
        )[1]
    return result


def _write_markdown(result: pd.DataFrame) -> None:
    columns = [
        "bag",
        "trajectory",
        "comparison_a_vs_b",
        "r2_a",
        "r2_b",
        "delta_r2_a_minus_b",
        "delta_r2_ci_lower",
        "delta_r2_ci_upper",
        "country_cluster_bootstrap_p_raw",
        "holm_p_within_bag_trajectory",
    ]
    (OUTPUT_DIR / "complexity_d2_fixed_comparisons.md").write_text(
        "# Model-level comparisons with the d2 winner held fixed\n\n"
        "Positive delta R2 favours the more complex model level. Raw p-values "
        "use a two-sided country-cluster bootstrap with 10,000 draws; Holm "
        "adjustment is across the six level comparisons within each BAG measure "
        "and arm.\n\n"
        + result[columns].to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate(draws=args.draws, seed=args.seed)
    output = OUTPUT_DIR / "complexity_d2_fixed_comparisons.csv"
    result.to_csv(output, index=False)
    _write_markdown(result)
    print(f"Wrote {len(result)} d2-fixed comparisons to {output}")


if __name__ == "__main__":
    main()
