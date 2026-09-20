"""Fill the inferential fields used by publication Supplementary Table 15."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from repo_expo_hoi_bag.publication_stats.paths import (
    REPOSITORY_ROOT,
    required_dedup_root,
)


SENSITIVITY_ROOT = REPOSITORY_ROOT / "outputs/sensitivity/dedup"
CONTEXT_SOURCE = (
    SENSITIVITY_ROOT / "normative_context/manuscript_transfer_by_context.csv"
)
CONTEXT_OUTPUT = (
    SENSITIVITY_ROOT / "normative_context/manuscript_selected_model_inference.csv"
)
WEIGHTING_SOURCE = (
    SENSITIVITY_ROOT / "diagnosis_balance/manuscript_weighting_comparison.csv"
)
WEIGHTING_MANIFEST = (
    SENSITIVITY_ROOT / "diagnosis_balance/diagnosis_balance_model_manifest.csv"
)
WEIGHTING_OUTPUT = (
    SENSITIVITY_ROOT / "diagnosis_balance/manuscript_weighting_inference.csv"
)
COMPARISONS = (
    REPOSITORY_ROOT / "outputs/dedup/model_comparison/comparison_results.csv"
)
CONFIG = (
    REPOSITORY_ROOT
    / "src/repo_expo_hoi_bag/stages/resources/sensitivity.yaml"
)


def _context_country_path(root: Path, row: dict[str, object]) -> tuple[Path, str, str]:
    train, test = str(row["condition"]).split("->")
    token = {"CN": "cn", "AD": "ad", "FTD": "ftd", "AD+FTD": "adftd"}[train]
    family = f"{token}_norm"
    rung = "ols" if row["model_level"] == "OLS" else f"xgb_tree_{row['model_level']}"
    model_family = "ols" if rung == "ols" else "xgb"
    path = (
        root
        / f"{model_family}_normative_loco"
        / str(row["bag"])
        / family
        / f"train_{train}"
        / f"{row['bag']}_{family}_{train}_{rung}_country.csv"
    )
    return path, test, rung


def selected_model_inference(root: Path) -> pd.DataFrame:
    source = pd.read_csv(CONTEXT_SOURCE)
    comparisons = pd.read_csv(COMPARISONS)
    weighting = pd.read_csv(WEIGHTING_SOURCE)
    diagnosis_counts = {
        (bag, diagnosis): int(group["weighted_n"].iloc[0])
        for (bag, diagnosis), group in weighting.groupby(
            ["bag", "diagnosis"], observed=True
        )
    }

    def cohort_n(bag: str, diagnosis: str) -> int:
        if diagnosis == "all":
            return sum(
                diagnosis_counts[(bag, item)] for item in ("CN", "AD", "FTD")
            )
        if diagnosis == "AD+FTD":
            return diagnosis_counts[(bag, "AD")] + diagnosis_counts[(bag, "FTD")]
        return diagnosis_counts[(bag, diagnosis)]

    rows: list[dict[str, object]] = []
    for row in source.to_dict("records"):
        if row["condition"] == "Pooled":
            arm = "syn" if row["objective"] == "o_min" else "red"
            comparison = (
                f"{row['bag']}|best_{arm}_vs_baseline|xgb_tree_d3"
            )
            match = comparisons[
                comparisons["comparison"].eq(comparison)
                & comparisons["loss"].eq("sq")
            ].iloc[0]
            n_subjects = int(match["n_subjects"])
            n_countries = int(match["n_countries"])
            p_value = float(match["wilcoxon_country_p"])
            n_train = cohort_n(str(row["bag"]), "all")
        else:
            path, test, _rung = _context_country_path(root, row)
            country = pd.read_csv(path)
            country = country[
                country["test_dx"].eq(test)
                & country["candidate_id"].isin(
                    [str(row["candidate_id"]), "__baseline__"]
                )
            ]
            paired = country.pivot(
                index="fold_country", columns="candidate_id", values="r2"
            ).dropna()
            difference = (
                paired[str(row["candidate_id"])] - paired["__baseline__"]
            )
            p_value = float(stats.wilcoxon(difference).pvalue)
            selected = country[
                country["candidate_id"].eq(str(row["candidate_id"]))
            ]
            n_subjects = int(selected["n_test"].sum())
            n_countries = int(len(paired))
            train, _test = str(row["condition"]).split("->")
            n_train = cohort_n(str(row["bag"]), train)
        rows.append(
            {
                "bag": row["bag"],
                "condition": row["condition"],
                "n_train": n_train,
                "n_test": n_subjects,
                "n_countries": n_countries,
                "country_wilcoxon_p": p_value,
                "test": "two-sided paired Wilcoxon across countries",
            }
        )
    return pd.DataFrame(rows)


def _weighting_p_values(n_boot: int, seed: int) -> pd.DataFrame:
    source = pd.read_csv(WEIGHTING_SOURCE)
    manifest = pd.read_csv(WEIGHTING_MANIFEST)
    rows: list[dict[str, object]] = []
    for row in source.to_dict("records"):
        spec = manifest[
            manifest["bag"].eq(row["bag"])
            & manifest["objective"].eq(row["objective"])
        ].iloc[0]
        balanced = pd.read_parquet(Path(spec["balanced_oof_output"]))
        unweighted = pd.read_parquet(Path(spec["unweighted_oof_source"]))
        diagnosis = str(row["diagnosis"])
        unweighted = unweighted[
            unweighted["diagnosis"].isin(["CN", "AD", "FTD"])
        ]
        keys = ["row_id", "N_MEGA", "country", "diagnosis", "y_true"]
        merged = balanced[keys + ["y_pred_full"]].merge(
            unweighted[keys + ["y_pred_full"]],
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_balanced", "_unweighted"),
        )
        merged = merged[merged["diagnosis"].eq(diagnosis)].copy()
        merged["balanced_error"] = (
            merged["y_pred_full_balanced"] - merged["y_true"]
        )
        merged["unweighted_error"] = (
            merged["y_pred_full_unweighted"] - merged["y_true"]
        )
        country = (
            merged.groupby(merged["country"].astype(str), observed=True)
            .agg(
                n=("y_true", "size"),
                balanced_error=("balanced_error", "sum"),
                unweighted_error=("unweighted_error", "sum"),
            )
            .sort_index()
        )
        values = country.to_numpy(dtype=float)
        rng = np.random.default_rng(
            seed
            + sum(
                ord(char)
                for char in (
                    f"{row['bag']}|{row['objective']}|full|{diagnosis}"
                )
            )
        )
        draw = rng.integers(0, len(values), size=(n_boot, len(values)))
        sampled = values[draw].sum(axis=1)
        delta = sampled[:, 1] / sampled[:, 0] - sampled[:, 2] / sampled[:, 0]
        p_value = 2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0)))
        p_value = min(1.0, max(p_value, 1.0 / n_boot))
        rows.append(
            {
                "bag": row["bag"],
                "objective": row["objective"],
                "diagnosis": diagnosis,
                "n_countries": int(len(values)),
                "country_cluster_bootstrap_p": p_value,
                "bootstrap_draws": n_boot,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = required_dedup_root()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    balance = config["paper_analysis"]["diagnosis_balance"]
    context = selected_model_inference(root)
    weighting = _weighting_p_values(
        int(balance["bootstrap_draws"]), int(balance["bootstrap_seed"])
    )
    CONTEXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(CONTEXT_OUTPUT, index=False)
    weighting.to_csv(WEIGHTING_OUTPUT, index=False)
    print(context.to_string(index=False))
    print(weighting.to_string(index=False))


if __name__ == "__main__":
    main()
