"""Matched inference for the selected domain-balanced representations.

The descriptive sensitivity table selects the best subset separately for each
representation and BAG.  This module compares those exact selected subsets with
the deployed depth-3 synergy-arm model using aligned subject-level OOF
predictions and the paper's country-cluster bootstrap on delta R-squared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.publication_stats.model_comparisons import compare_one
from repo_expo_hoi_bag.publication_stats.paths import REPOSITORY_ROOT


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs/sensitivity/dedup/domain_imbalance/manuscript_matched_inference.csv"
)
OOF_ROOT = REPOSITORY_ROOT / "outputs/dedup/model_comparison/oof"
SUMMARY = (
    REPOSITORY_ROOT
    / "outputs/sensitivity/dedup/domain_imbalance/manuscript_main_comparison.csv"
)
FAMILY_MAP = {
    "best_single_per_domain": "selected_best_single_per_domain",
    "within_domain_pc1": "selected_within_domain_pc1",
}


def build(n_boot: int = 10_000, seed: int = 20260624) -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for record in summary.to_dict("records"):
        bag = str(record["bag"])
        representation = str(record["candidate_family"])
        oof_family = FAMILY_MAP[representation]
        main_path = OOF_ROOT / "best_syn" / bag / "oof_xgb_tree_d3.parquet"
        alternative_path = OOF_ROOT / oof_family / bag / "oof_xgb_tree_d3.parquet"
        results = compare_one(
            f"{bag}|syn_vs_{oof_family}|xgb_tree_d3",
            main_path,
            alternative_path,
            "y_pred_full",
            "y_pred_full",
            "country",
            n_boot,
            rng,
        )
        result = next(item for item in results if item["loss"] == "sq")
        expected = float(record["sensitivity_best_r2"])
        if abs(float(result["r2_b"]) - expected) > 0.005:
            raise ValueError(
                f"OOF reproduction gate failed for {bag}/{representation}: "
                f"paired={result['r2_b']:.6f}, selected={expected:.6f}"
            )
        rows.append(
            {
                "bag": bag,
                "candidate_family": representation,
                "oof_family": oof_family,
                "candidate_id": pd.read_parquet(alternative_path, columns=["candidate_id"])[
                    "candidate_id"
                ].iloc[0],
                "n_subjects": result["n_subjects"],
                "n_countries": result["n_countries"],
                "r2_main": result["r2_a"],
                "r2_alternative": result["r2_b"],
                "selected_summary_r2": expected,
                "delta_r2_main_minus_alternative": result["delta_r2"],
                "delta_r2_ci_lo": result["delta_r2_ci_lo"],
                "delta_r2_ci_hi": result["delta_r2_ci_hi"],
                "country_cluster_bootstrap_p": result["boot_p_delta_r2"],
                "bootstrap_draws": n_boot,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.n_boot, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
