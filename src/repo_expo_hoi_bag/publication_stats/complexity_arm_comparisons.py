"""Within-level synergy-versus-redundancy comparisons for the complexity figure.

The fixed-candidate contrasts already exist in ``comparison_results.csv``.  The
level-selected contrasts pair the independently selected synergy- and
redundancy-arm OOF predictions generated for Supplementary Table 3.  This
module places both selections in one compact file and applies Holm correction
across the four model levels within each BAG measure and selection rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.publication_stats.complexity_comparisons import (
    LEVEL_LABELS,
    LEVELS,
    OOF_DIR,
    _align,
    _bootstrap_delta_r2,
    _candidate_identity,
    _load_prediction,
    _r2,
)
from repo_expo_hoi_bag.publication_stats.paths import model_comparison_output_dir


OUTPUT_DIR = model_comparison_output_dir()
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20260877
D2_FIXED_SEED = 20260960


def _fixed_rows() -> list[dict[str, object]]:
    source_path = OUTPUT_DIR / "comparison_results.csv"
    source = pd.read_csv(source_path)
    source = source[
        source["comparison"].str.contains(r"\|syn_vs_red\|")
        & source["loss"].eq("sq")
    ].copy()
    rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        bag, _, level = str(row["comparison"]).split("|")
        rows.append(
            {
                "selection_rule": "fixed_d3_candidates",
                "bag": bag,
                "model_level": LEVEL_LABELS[level],
                "synergy_candidate": _candidate_identity(
                    _load_prediction(
                        OOF_DIR / "best_syn" / bag / f"oof_{level}.parquet",
                        "y_pred_full",
                    )
                ),
                "redundancy_candidate": _candidate_identity(
                    _load_prediction(
                        OOF_DIR / "best_red" / bag / f"oof_{level}.parquet",
                        "y_pred_full",
                    )
                ),
                "n_subjects": int(row["n_subjects"]),
                "n_countries": int(row["n_countries"]),
                "r2_synergy": float(row["r2_a"]),
                "r2_redundancy": float(row["r2_b"]),
                "delta_r2_synergy_minus_redundancy": float(row["delta_r2"]),
                "delta_r2_ci_lower": float(row["delta_r2_ci_lo"]),
                "delta_r2_ci_upper": float(row["delta_r2_ci_hi"]),
                "country_cluster_bootstrap_p_raw": float(row["boot_p_delta_r2"]),
                "bootstrap_draws": 10_000,
                "bootstrap_seed": 20260624,
                "bootstrap_seed_role": "shared RNG stream start",
                "source": str(source_path),
            }
        )
    return rows


def _level_selected_rows(*, draws: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for bag in ("structural", "functional"):
        for level in LEVELS:
            synergy = _load_prediction(
                OOF_DIR / "level_best_syn" / bag / f"oof_{level}.parquet",
                "y_pred_full",
            )
            redundancy = _load_prediction(
                OOF_DIR / "level_best_red" / bag / f"oof_{level}.parquet",
                "y_pred_full",
            )
            paired = _align(synergy, redundancy)
            rng = np.random.default_rng(seed + comparison_index)
            comparison_index += 1
            lower, upper, raw_p = _bootstrap_delta_r2(
                paired, draws=draws, rng=rng
            )
            y_true = paired["y_true_a"].to_numpy(float)
            pred_syn = paired["prediction_a"].to_numpy(float)
            pred_red = paired["prediction_b"].to_numpy(float)
            rows.append(
                {
                    "selection_rule": "winner_at_each_level",
                    "bag": bag,
                    "model_level": LEVEL_LABELS[level],
                    "synergy_candidate": _candidate_identity(synergy),
                    "redundancy_candidate": _candidate_identity(redundancy),
                    "n_subjects": len(paired),
                    "n_countries": paired["country_a"].nunique(),
                    "r2_synergy": _r2(y_true, pred_syn),
                    "r2_redundancy": _r2(y_true, pred_red),
                    "delta_r2_synergy_minus_redundancy": (
                        _r2(y_true, pred_syn) - _r2(y_true, pred_red)
                    ),
                    "delta_r2_ci_lower": lower,
                    "delta_r2_ci_upper": upper,
                    "country_cluster_bootstrap_p_raw": raw_p,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed + comparison_index - 1,
                    "bootstrap_seed_role": "per-comparison seed",
                    "source": "paired level_best_syn and level_best_red OOF",
                }
            )
    return rows


def _d2_fixed_rows(*, draws: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison_index, bag in enumerate(("structural", "functional")):
        # Load the same fixed candidates at every model level.
        for level_index, level in enumerate(LEVELS):
            synergy = _load_prediction(
                OOF_DIR / "d2_best_syn" / bag / f"oof_{level}.parquet",
                "y_pred_full",
            )
            redundancy = _load_prediction(
                OOF_DIR / "d2_best_red" / bag / f"oof_{level}.parquet",
                "y_pred_full",
            )
            paired = _align(synergy, redundancy)
            row_seed = seed + comparison_index * len(LEVELS) + level_index
            lower, upper, raw_p = _bootstrap_delta_r2(
                paired, draws=draws, rng=np.random.default_rng(row_seed)
            )
            y_true = paired["y_true_a"].to_numpy(float)
            pred_syn = paired["prediction_a"].to_numpy(float)
            pred_red = paired["prediction_b"].to_numpy(float)
            rows.append(
                {
                    "selection_rule": "fixed_d2_candidates",
                    "bag": bag,
                    "model_level": LEVEL_LABELS[level],
                    "synergy_candidate": _candidate_identity(synergy),
                    "redundancy_candidate": _candidate_identity(redundancy),
                    "n_subjects": len(paired),
                    "n_countries": paired["country_a"].nunique(),
                    "r2_synergy": _r2(y_true, pred_syn),
                    "r2_redundancy": _r2(y_true, pred_red),
                    "delta_r2_synergy_minus_redundancy": (
                        _r2(y_true, pred_syn) - _r2(y_true, pred_red)
                    ),
                    "delta_r2_ci_lower": lower,
                    "delta_r2_ci_upper": upper,
                    "country_cluster_bootstrap_p_raw": raw_p,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": row_seed,
                    "bootstrap_seed_role": "per-comparison seed",
                    "source": "paired d2_best_syn and d2_best_red OOF",
                }
            )
    return rows


def generate(*, draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            *_fixed_rows(),
            *_d2_fixed_rows(draws=draws, seed=D2_FIXED_SEED),
            *_level_selected_rows(draws=draws, seed=seed),
        ]
    )
    statistic_columns = [
        "n_subjects",
        "n_countries",
        "r2_synergy",
        "r2_redundancy",
        "delta_r2_synergy_minus_redundancy",
        "delta_r2_ci_lower",
        "delta_r2_ci_upper",
        "country_cluster_bootstrap_p_raw",
        "bootstrap_draws",
        "bootstrap_seed",
        "bootstrap_seed_role",
        "source",
    ]
    # The d3 winners coincide across the two selection rules. Reuse the
    # already-published comparison instead of introducing Monte Carlo drift.
    selected_indices = result.index[result["selection_rule"].eq("winner_at_each_level")]
    for selected_index in selected_indices:
        selected = result.loc[selected_index]
        match = result[
            result["selection_rule"].eq("fixed_d3_candidates")
            & result["bag"].eq(selected["bag"])
            & result["model_level"].eq(selected["model_level"])
            & result["synergy_candidate"].eq(selected["synergy_candidate"])
            & result["redundancy_candidate"].eq(selected["redundancy_candidate"])
        ]
        if len(match) == 1:
            result.loc[selected_index, statistic_columns] = match.iloc[0][
                statistic_columns
            ].to_numpy()
    result["holm_p_across_four_levels"] = np.nan
    for _, indices in result.groupby(
        ["selection_rule", "bag"], sort=False
    ).groups.items():
        result.loc[indices, "holm_p_across_four_levels"] = multipletests(
            result.loc[indices, "country_cluster_bootstrap_p_raw"], method="holm"
        )[1]
    level_order = {label: index for index, label in enumerate(LEVEL_LABELS.values())}
    result["_level_order"] = result["model_level"].map(level_order)
    return result.sort_values(
        ["selection_rule", "bag", "_level_order"]
    ).drop(columns="_level_order").reset_index(drop=True)


def _write_markdown(result: pd.DataFrame, path: Path) -> None:
    rendered = result[
        [
            "selection_rule",
            "bag",
            "model_level",
            "r2_synergy",
            "r2_redundancy",
            "delta_r2_synergy_minus_redundancy",
            "delta_r2_ci_lower",
            "delta_r2_ci_upper",
            "country_cluster_bootstrap_p_raw",
            "holm_p_across_four_levels",
        ]
    ].copy()
    path.write_text(
        "# Within-level arm comparisons for the model-complexity figure\n\n"
        "Delta R2 is synergy minus redundancy. Raw p-values use a two-sided "
        "country-cluster bootstrap with 10,000 draws; Holm adjustment is across "
        "the four model levels within each BAG measure and selection rule.\n\n"
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
    output = OUTPUT_DIR / "complexity_arm_comparisons.csv"
    result.to_csv(output, index=False)
    _write_markdown(result, OUTPUT_DIR / "complexity_arm_comparisons.md")
    print(f"Wrote {len(result)} within-level arm comparisons to {output}")


if __name__ == "__main__":
    main()
