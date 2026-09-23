"""Finalize the corrected main-k10 supplementary workbook in place.

This stage preserves the validated publication sheets from the prepared
17-sheet workbook and rebuilds only the tables affected by the corrected HPO
routing: ST11, ST15, and ST17. It performs no model fitting.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats
from statsmodels.stats.multitest import multipletests

from repo_expo_hoi_bag.stages.build_publication_supplementary_tables import (
    PublicationTable,
    _write_table,
)


ROOT = Path(__file__).resolve().parents[3]
# The delivered country-balanced package is the default in every path below.
# A parallel estimand delivery (R2_MODE=global_oof) overrides the run and the
# delivery root explicitly; it never writes into the country-balanced tree.
RUN_ID = os.environ.get("MAIN_K10_DELIVERY_RUN_ID", "").strip() or "main_k10_release_20260916"
RUN_ROOT = ROOT / "outputs" / "main" / RUN_ID / "sensitivity"
PREPARED = ROOT / "results" / "main" / "Supplementary_Tables_main_k10_awaiting_diagnosis_balance.xlsx"
DELIVERY_ROOT = Path(
    os.environ.get("MAIN_K10_DELIVERY_ROOT", "").strip()
    or str(ROOT / "outputs" / "main" / "paper" / "complete")
)
DELIVERY = DELIVERY_ROOT / "tables" / "Supplementary_Tables_main_k10.xlsx"
PCA_SOURCE_ROOT = DELIVERY_ROOT / "figures" / "supplementary" / "source_data"
TABLE_SOURCE_ROOT = DELIVERY_ROOT / "tables" / "source_data"

# ---------------------------------------------------------------------------
# R2 estimand vocabulary
# ---------------------------------------------------------------------------
# Every reported R2 in this workbook is one of two estimands over the *same*
# OOF predictions, folds, cohort and exclusions.  R2_MODE selects which one;
# the labels, column headers, guard values and note wording below all follow
# it, so a workbook can never mix the two or mislabel one as the other.
GLOBAL_OOF_MODE = os.environ.get("R2_MODE", "").strip() == "global_oof"
R2_MODE_VALUE = "global_oof" if GLOBAL_OOF_MODE else "country_balanced"
# Value stamped by the upstream statistics stages.
R2_ESTIMAND_TOKEN = "global_oof_r2" if GLOBAL_OOF_MODE else "unweighted_mean_country_r2"
# Human-readable estimand name used in headers, the "R² estimand" column and notes.
R2_ESTIMAND_LABEL = (
    "Global pooled out-of-fold R²" if GLOBAL_OOF_MODE else "Unweighted mean held-out-country R²"
)
R2_ESTIMAND_PHRASE = (
    "global pooled out-of-fold R²" if GLOBAL_OOF_MODE else "unweighted mean held-out-country R²"
)
R2_ESTIMAND_PHRASE_LOCO = (
    "global pooled out-of-fold LOCO R²"
    if GLOBAL_OOF_MODE
    else "unweighted mean held-out-country LOCO R²"
)
# Column header for the per-row point estimate.
R2_COLUMN_LABEL = "Global OOF R²" if GLOBAL_OOF_MODE else "Country-balanced R²"
# Heavy covariate/residualised-target fits are estimand-invariant: the global
# delivery reuses the completed evaluations rather than refitting them.
SENSITIVITY_EVAL_RUN_ID = (
    os.environ.get("MAIN_K10_SENSITIVITY_EVAL_RUN_ID", "").strip() or RUN_ID
)

SHEETS = [
    "ST01_HigherOrder", "ST02_ModelComplexity", "ST03_BestModels",
    "ST04_CountryBlockNull", "ST05_TopComposition", "ST06_DomainDiversity",
    "ST07_DomainComposition", "ST08_Cooccurrence", "ST09_RecurrentTriplets",
    "ST10_ResidualBias", "ST11_NormativeTransfer", "ST12_MixedModels",
    "ST13_AltRepresentations", "ST14_NegativeOmega", "ST15_DiagnosticContext",
    "ST16_Geography", "ST17_CovariatesTargets", "ST18_SetSizeCaps",
]

BAG_LABEL = {"structural": "Structural", "functional": "Functional"}
ARM_LABEL = {"o_min": "Synergy arm", "o_max": "Redundancy arm"}
DX_LABEL = {"CN": "HC", "FTD": "FTLD", "all": "All"}
LEVEL_LABEL = {"xgb_tree_d1": "d1", "xgb_tree_d2": "d2", "xgb_tree_d3": "d3"}
ACTIVE_REPRO_DATA_ROOT: Path | None = None


def _runtime_root() -> Path:
    if ACTIVE_REPRO_DATA_ROOT is None:
        raise RuntimeError("The canonical --repro-data-root has not been configured")
    return ACTIVE_REPRO_DATA_ROOT


def _candidate_country_scores(frame: pd.DataFrame, global_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-candidate score under the active estimand.

    The returned column keeps the historical name ``country_balanced_r2`` so
    every caller's selection code is unchanged; under R2_MODE=global_oof it
    carries the stored pooled global OOF R2 taken from ``global_frame``."""
    if GLOBAL_OOF_MODE and global_frame is not None:
        if "global_oof_r2" not in global_frame.columns:
            raise ValueError("Global metrics frame lacks global_oof_r2")
        data = global_frame.copy()
        data["global_oof_r2"] = pd.to_numeric(data["global_oof_r2"], errors="coerce")
        return (
            data.groupby("candidate_id", observed=True)["global_oof_r2"]
            .max()
            .rename("country_balanced_r2")
            .reset_index()
        )
    data = frame.copy()
    data["r2"] = pd.to_numeric(data["r2"], errors="coerce")
    return (
        data[data["n_test"].gt(0)]
        .groupby("candidate_id", observed=True)["r2"]
        .mean()
        .rename("country_balanced_r2")
        .reset_index()
    )


def _country_vector(frame: pd.DataFrame, candidate_id: str) -> pd.Series:
    selected = frame[frame["candidate_id"].astype(str).eq(str(candidate_id))].copy()
    selected["r2"] = pd.to_numeric(selected["r2"], errors="coerce")
    return selected.set_index("fold_country")["r2"].dropna()


def _paired_wilcoxon(current: pd.Series, reference: pd.Series) -> tuple[int, float]:
    paired = pd.concat(
        [current.rename("current"), reference.rename("reference")], axis=1
    ).dropna()
    if len(paired) < 3:
        return len(paired), np.nan
    difference = paired["current"] - paired["reference"]
    if np.allclose(difference, 0):
        return len(paired), 1.0
    return len(paired), float(stats.wilcoxon(difference, alternative="two-sided").pvalue)


def _paired_country_bootstrap(
    main: pd.Series, alternative: pd.Series, *, seed: int, draws: int = 10_000
) -> dict[str, float]:
    paired = pd.concat(
        [main.rename("main"), alternative.rename("alternative")], axis=1
    ).dropna()
    difference = (paired["main"] - paired["alternative"]).to_numpy(float)
    if difference.size < 3:
        raise ValueError("At least three paired countries are required")
    rng = np.random.default_rng(seed)
    sampled = difference[
        rng.integers(0, difference.size, size=(draws, difference.size))
    ].mean(axis=1)
    p_value = min(
        1.0,
        max(
            2 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0))),
            1 / draws,
        ),
    )
    return {
        "n_countries": int(difference.size),
        "delta": float(difference.mean()),
        "ci_lo": float(np.percentile(sampled, 2.5)),
        "ci_hi": float(np.percentile(sampled, 97.5)),
        "p": p_value,
    }


def _diagnosis_weighting_pvalues(draws: int = 10_000, seed: int = 20260731) -> pd.DataFrame:
    """Reproduce the parent country-cluster test from saved paired OOF predictions."""
    manifest = pd.read_csv(
        RUN_ROOT / "diagnosis_balance/diagnosis_balance_model_manifest.csv"
    )
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        balanced = pd.read_parquet(record["balanced_oof_output"])
        unweighted = pd.read_parquet(record["unweighted_oof_source"])
        merged = balanced[
            ["row_id", "country", "diagnosis", "y_true", "y_pred_full"]
        ].merge(
            unweighted[["row_id", "y_true", "y_pred_full"]],
            on="row_id",
            suffixes=("_balanced", "_unweighted"),
            validate="one_to_one",
        )
        for diagnosis in ("all", "CN", "AD", "FTD"):
            subset = merged if diagnosis == "all" else merged[merged["diagnosis"].eq(diagnosis)]
            subset = subset.dropna(
                subset=["country", "y_true_balanced", "y_pred_full_balanced", "y_pred_full_unweighted"]
            ).copy()
            subset["error_balanced"] = subset["y_pred_full_balanced"] - subset["y_true_balanced"]
            subset["error_unweighted"] = subset["y_pred_full_unweighted"] - subset["y_true_balanced"]
            countries = sorted(subset["country"].astype(str).unique())
            values = (
                subset.groupby(subset["country"].astype(str), observed=True)
                .agg(
                    n=("y_true_balanced", "size"),
                    balanced=("error_balanced", "sum"),
                    unweighted=("error_unweighted", "sum"),
                )
                .reindex(countries)
                .to_numpy(float)
            )
            rng = np.random.default_rng(
                seed + sum(ord(char) for char in f"{record['bag']}|{record['objective']}|full|{diagnosis}")
            )
            sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].sum(axis=1)
            delta = sampled[:, 1] / sampled[:, 0] - sampled[:, 2] / sampled[:, 0]
            p_value = min(
                1.0,
                max(2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0))), 1 / draws),
            )
            rows.append(
                {
                    "bag": record["bag"],
                    "objective": record["objective"],
                    "diagnosis": diagnosis,
                    "country_cluster_bootstrap_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def _holm_within(frame: pd.DataFrame, column: str, groups: list[str]) -> pd.Series:
    adjusted = pd.Series(index=frame.index, dtype=float)
    for indices in frame.groupby(groups, sort=False).groups.values():
        values = pd.to_numeric(frame.loc[indices, column], errors="coerce")
        valid = values.notna()
        if valid.any():
            adjusted.loc[values.index[valid]] = multipletests(values[valid], method="holm")[1]
    return adjusted


def _table_11() -> PublicationTable:
    source = RUN_ROOT / "normative_transfer" / "stats" / "normative_transfer_results.csv"
    data = pd.read_csv(source)
    if len(data) != 14 or set(data["bag"]) != set(BAG_LABEL):
        raise ValueError(f"Unexpected normative-transfer summary in {source}: rows={len(data)}")
    frame = pd.DataFrame(
        {
            "BAG": data["bag"].map(BAG_LABEL),
            "Train → test": data["transfer"].str.replace("CN", "HC", regex=False).str.replace("FTD", "FTLD", regex=False),
            "Participants": data["n_scored"],
            "Countries": data["bestVbase_n_countries"],
            "Baseline R²": data["r2_baseline"],
            "Best arm": data["best_arm"],
            "Best R²": data["r2_best"],
            "ΔR², best − baseline": data["delta_best_vs_base"],
            "Wilcoxon p": data["bestVbase_wilcoxon_p"],
            "Sign-test p": data["bestVbase_sign_p"],
        }
    )
    primary = frame["Train → test"].isin(
        ("HC->HC", "AD->AD", "FTLD->FTLD", "AD+FTLD->AD", "AD+FTLD->FTLD")
    )
    frame["Primary comparison"] = primary
    frame["Holm-adjusted Wilcoxon p"] = np.nan
    frame["Holm-adjusted sign-test p"] = np.nan
    frame.loc[primary, "Holm-adjusted Wilcoxon p"] = _holm_within(
        frame.loc[primary], "Wilcoxon p", ["BAG"]
    )
    frame.loc[primary, "Holm-adjusted sign-test p"] = _holm_within(
        frame.loc[primary], "Sign-test p", ["BAG"]
    )
    return PublicationTable(
        11,
        "Diagnostic train-to-test performance",
        frame,
        "R² is the unweighted mean of held-out-country R². Depth-3 models were selected independently within each train-to-test setting. Across-country tests were paired and two-sided; P values were Holm-adjusted across the five prespecified primary settings within each BAG measure. The two cross-diagnosis settings are shown for completeness and are not included in multiplicity adjustment.",
    )


def _table_01() -> PublicationTable:
    source = TABLE_SOURCE_ROOT / "table_adapters/ST01_greedy_oinfo_by_order.csv"
    data = pd.read_csv(source)
    frame = data.rename(
        columns={
            "objective": "Discovery arm",
            "order": "Set size",
            "n_candidates": "Greedy candidates",
            "median_omega": "Median Ω (nats)",
            "min_omega": "Minimum Ω (nats)",
            "max_omega": "Maximum Ω (nats)",
        }
    )
    frame["Discovery arm"] = frame["Discovery arm"].map(ARM_LABEL)
    return PublicationTable(
        1,
        "Greedy O-information envelope by set size",
        frame[["Discovery arm", "Set size", "Greedy candidates", "Median Ω (nats)", "Minimum Ω (nats)", "Maximum Ω (nats)"]],
        "The candidate population and extrema are exactly those used for the O-information panel of Figure 2.",
    )


def _table_03() -> PublicationTable:
    source = TABLE_SOURCE_ROOT / "table_adapters/ST03_best_multivariate_vs_single.csv"
    data = pd.read_csv(source)
    frame = pd.DataFrame(
        {
            "BAG": data["bag"].map(BAG_LABEL),
            "Discovery arm": data["objective"].map(ARM_LABEL),
            "Countries": data["n_countries"],
            "Participants": data["n_subjects"],
            "Multivariate R²": data["r2_a"],
            "Single-exposure R²": data["r2_b"],
            "ΔR²": data["delta_r2"],
            "95% CI, lower": data["ci_lo"],
            "95% CI, upper": data["ci_hi"],
            "Two-sided p": data["p_raw"],
            "Holm-adjusted p": data["holm_p_across_four_comparisons"],
        }
    )
    return PublicationTable(
        3,
        "Best multivariate versus single-exposure models",
        frame,
        "R² is the unweighted mean of held-out-country R², matching Figure 2. Confidence intervals and P values use a two-sided country bootstrap with 10,000 draws.",
    )


def _table_04() -> PublicationTable:
    root = RUN_ROOT / "country_block_null"
    data = pd.concat(
        [pd.read_csv(root / f"{bag}_winner_pvalues.csv") for bag in ("structural", "functional")],
        ignore_index=True,
    )
    data = data[data["rung_id"].eq("xgb_tree_d3")].copy()
    if set(data["r2_estimand"].astype(str)) != {R2_MODE_VALUE}:
        raise ValueError("ST04 requires completed country-balanced country-block-null outputs")
    destination = TABLE_SOURCE_ROOT / "country_block_null"
    destination.mkdir(parents=True, exist_ok=True)
    for bag in ("structural", "functional"):
        pd.read_csv(root / f"{bag}_winner_pvalues.csv").to_csv(
            destination / f"{bag}_winner_pvalues.csv", index=False
        )
    frame = pd.DataFrame(
        {
            "BAG": data["bag"].map(BAG_LABEL),
            "Model": data["label"].replace({"best_syn": "Best synergy arm", "best_red": "Best redundancy arm"}),
            "Set size": data["order"],
            f"Observed {R2_COLUMN_LABEL}": data["observed_r2"],
            "Null mean R²": data["null_mean"],
            "Null SD": data["null_std"],
            "z versus null": data["cohen_d_vs_null"],
            "Empirical upper-tail p": data["p_value"],
            "Permutations": data["n_perm"],
        }
    )
    return PublicationTable(
        4,
        "Country-block permutation null",
        frame,
        f"Observed and permuted statistics are the {R2_ESTIMAND_PHRASE} used in Figure 2. Country-year exposure signatures are permuted as blocks; empirical p=(k+1)/(n+1).",
    )


def _table_08() -> PublicationTable:
    stats = pd.read_csv(TABLE_SOURCE_ROOT / "local_analyses/network_stats_d3.csv")
    tests = pd.read_csv(TABLE_SOURCE_ROOT / "local_analyses/network_permutation_d3.csv")
    observed = stats.pivot(index=["bag"], columns="objective", values=["n_domains", "n_edges", "density", "mean_weighted_degree"])
    rows = []
    for row in tests.itertuples(index=False):
        rows.append(
            {
                "BAG": BAG_LABEL[row.bag],
                "Network statistic": row.metric,
                "Synergy arm": observed.loc[row.bag, (row.metric, "o_min")],
                "Redundancy arm": observed.loc[row.bag, (row.metric, "o_max")],
                "Synergy − redundancy": row.syn_minus_red,
                "Two-sided permutation p": row.permutation_p_two_sided,
                "Holm-adjusted p": row.holm_p_within_bag,
                "Permutations": row.permutations,
            }
        )
    return PublicationTable(
        8,
        "Domain co-occurrence network architecture",
        pd.DataFrame(rows),
        "Edge weights count exposure-pair co-occurrences aggregated to domain pairs, exactly as in Figure 3. Arm labels are permuted across the overlapping top-20 candidate pool.",
    )


def _table_02() -> PublicationTable:
    source = (
        _runtime_root()
        / "results/analysis_runs"
        / (RUN_ID if GLOBAL_OOF_MODE else "paper_reanalysis_k10/main_statistics")
        / "model_comparison"
        / "complexity_and_arm_comparisons.csv"
    )
    frame = pd.read_csv(source)
    if set(frame["r2_estimand"].dropna().astype(str)) != {R2_ESTIMAND_TOKEN}:
        raise ValueError(f"ST02 requires {R2_MODE_VALUE} complexity comparisons")
    frame["BAG"] = frame["bag"].map(BAG_LABEL)
    frame["Discovery arm"] = frame["objective"].map(ARM_LABEL).fillna(frame["objective"])
    columns = [
        "BAG", "comparison_type", "Discovery arm", "model_a", "model_b",
        "n_subjects", "n_countries", "r2_a", "r2_b", "delta_r2", "ci_lo",
        "ci_hi", "p_raw", "holm_p_within_bag_trajectory",
        "holm_p_d3_claim_within_bag_trajectory", "r2_estimand",
    ]
    return PublicationTable(
        2,
        "Model-complexity and arm comparisons",
        frame[columns],
        "All R² values are unweighted means of held-out-country R². Differences, confidence intervals and two-sided P values use a 10,000-draw paired country bootstrap; Holm families match the parent model-complexity analysis. Participant- and country-loss Wilcoxon sensitivity columns remain in the runtime audit CSV.",
    )


def _table_10() -> PublicationTable:
    subject = pd.read_csv(TABLE_SOURCE_ROOT / "derived/main_residual_subject_summary.csv")
    summary = pd.DataFrame(
        {
            "Analysis": "Residual summary",
            "BAG": subject["bag"].map(BAG_LABEL),
            "Discovery arm": subject["objective"].map(ARM_LABEL),
            "Diagnosis": subject["diagnosis"].map(DX_LABEL).fillna(subject["diagnosis"]),
            "Participants": subject["n_subjects"],
            "Countries": np.nan,
            R2_COLUMN_LABEL: np.nan,
            f"Baseline {R2_COLUMN_LABEL}": np.nan,
            "ΔR²": np.nan,
            "95% CI, lower": np.nan,
            "95% CI, upper": np.nan,
            "P value": np.nan,
            "Mean bias (years)": subject["bias_mean"],
            "MAE (years)": subject["mae"],
            "RMSE (years)": subject["rmse"],
        }
    )
    oof_root = (
        _runtime_root()
        / "results/analysis_runs/paper_reanalysis_k10/main_statistics/model_comparison"
        / ("oof_global_oof" if GLOBAL_OOF_MODE else "oof")
    )
    tests: list[dict[str, object]] = []
    for bag_index, bag in enumerate(("structural", "functional")):
        for objective_index, (objective, family) in enumerate(
            (("o_min", "level_best_syn"), ("o_max", "level_best_red"))
        ):
            oof = pd.read_parquet(
                oof_root / family / bag / "oof_xgb_tree_d3.parquet",
                columns=["country", "y_true", "y_pred_full", "y_pred_base"],
            ).dropna()
            full = pd.Series(
                {
                    str(country): 1 - np.square(group.y_true - group.y_pred_full).sum()
                    / np.square(group.y_true - group.y_true.mean()).sum()
                    for country, group in oof.groupby("country", sort=True)
                }
            )
            baseline = pd.Series(
                {
                    str(country): 1 - np.square(group.y_true - group.y_pred_base).sum()
                    / np.square(group.y_true - group.y_true.mean()).sum()
                    for country, group in oof.groupby("country", sort=True)
                }
            )
            inference = _paired_country_bootstrap(
                full, baseline, seed=20260910 + 2 * bag_index + objective_index
            )
            tests.append(
                {
                    "Analysis": "Model versus covariate baseline",
                    "BAG": BAG_LABEL[bag],
                    "Discovery arm": ARM_LABEL[objective],
                    "Diagnosis": "All",
                    "Participants": len(oof),
                    "Countries": inference["n_countries"],
                    R2_COLUMN_LABEL: float(full.mean()),
                    f"Baseline {R2_COLUMN_LABEL}": float(baseline.mean()),
                    "ΔR²": inference["delta"],
                    "95% CI, lower": inference["ci_lo"],
                    "95% CI, upper": inference["ci_hi"],
                    "P value": inference["p"],
                    "Mean bias (years)": np.nan,
                    "MAE (years)": np.nan,
                    "RMSE (years)": np.nan,
                }
            )
    tests_frame = pd.DataFrame(tests)
    tests_frame["Holm-adjusted p"] = multipletests(tests_frame["P value"], method="holm")[1]
    summary["Holm-adjusted p"] = np.nan
    frame = pd.concat([summary, tests_frame], ignore_index=True)
    frame["R² estimand"] = frame[R2_COLUMN_LABEL].notna().map(
        {True: R2_ESTIMAND_LABEL, False: "Not applicable"}
    )
    return PublicationTable(
        10,
        "Residual bias by diagnosis and country",
        frame,
        f"Bias is predicted minus observed BAG in years. Model-versus-baseline R² is the {R2_ESTIMAND_PHRASE}; confidence intervals and two-sided P values use a 10,000-draw paired country bootstrap and are Holm-adjusted across the four deployed comparisons.",
    )


def _table_12() -> PublicationTable:
    root = TABLE_SOURCE_ROOT / "residual_confounds"
    estimates = pd.read_csv(root / "lme_estimates_all.csv")
    variance = pd.read_csv(root / "lme_variance_all.csv")
    fixed = pd.DataFrame(
        {
            "BAG": estimates["bag"].map(BAG_LABEL),
            "Discovery arm": estimates["objective"].map(ARM_LABEL),
            "Analysis": "Fixed effect",
            "Term": estimates["term"].str.replace("FTD", "FTLD", regex=False),
            "Estimate": estimates["coef"],
            "SE": estimates["se"],
            "95% CI, lower": estimates["ci_lo"],
            "95% CI, upper": estimates["ci_hi"],
            "P value": estimates["p_value"],
            "Participants": estimates["n_obs"],
            "Countries": estimates["n_countries"],
        }
    )
    random = pd.DataFrame(
        {
            "BAG": variance["bag"].map(BAG_LABEL),
            "Discovery arm": variance["objective"].map(ARM_LABEL),
            "Analysis": "Country random intercept",
            "Term": "Country",
            "Estimate": variance["adjusted_country_variance"],
            "SE": np.nan,
            "95% CI, lower": np.nan,
            "95% CI, upper": np.nan,
            "P value": variance["country_random_intercept_bootstrap_p"],
            "Participants": variance["n_obs"],
            "Countries": variance["n_countries"],
        }
    )
    return PublicationTable(
        12,
        "Residual-confound linear mixed-effects models",
        pd.concat([fixed, random], ignore_index=True),
        "These are the exact standardized full models used in the residual-confounds figure. Female and HC are reference levels; random-intercept P values use the configured parametric bootstrap.",
    )


def _table_15() -> PublicationTable:
    transfer = pd.read_csv(
        RUN_ROOT / "normative_transfer/stats/normative_transfer_results.csv"
    )
    selected_rows = pd.DataFrame(
        {
            "Analysis": "Selected normative model",
            "BAG": transfer["bag"].map(BAG_LABEL),
            "Setting": transfer["transfer"].str.replace("CN", "HC", regex=False).str.replace("FTD", "FTLD", regex=False),
            "Model or arm": transfer["best_arm"],
            "N": transfer["n_scored"],
            "Countries": transfer["bestVbase_n_countries"],
            "Primary estimate": transfer["r2_best"],
            "Comparator estimate": transfer["r2_baseline"],
            "Difference or interaction": transfer["delta_best_vs_base"],
            "P value": transfer["bestVbase_wilcoxon_p"],
            "Estimand": R2_ESTIMAND_LABEL,
            "95% CI": np.nan,
            "Secondary estimate": np.nan,
            "Secondary comparator": np.nan,
            "Secondary difference": np.nan,
            "Secondary 95% CI": np.nan,
        }
    )

    diversity = pd.read_csv(
        RUN_ROOT / "normative_diversity/normative_diversity_r2_panel_models.csv"
    )
    diversity_rows = pd.DataFrame(
        {
            "Analysis": "Normative diversity × arm",
            "BAG": diversity["bag"].map(BAG_LABEL),
            "Setting": diversity["condition"].astype(str).str.replace("CN", "HC", regex=False).str.replace("FTD", "FTLD", regex=False),
            "Model or arm": "Redundancy versus synergy arm",
            "N": diversity["n_obs"],
            "Countries": np.nan,
            "Primary estimate": diversity["beta_h_red"],
            "Comparator estimate": diversity["beta_h_syn"],
            "Difference or interaction": diversity["beta_h_syn"] - diversity["beta_h_red"],
            "P value": diversity["p_h_syn_interaction"],
            "Estimand": f"Candidate R² is {R2_ESTIMAND_PHRASE}",
            "95% CI": np.nan,
            "Secondary estimate": np.nan,
            "Secondary comparator": np.nan,
            "Secondary difference": np.nan,
            "Secondary 95% CI": np.nan,
        }
    )

    root = RUN_ROOT / "diagnosis_balance"
    metrics = pd.read_csv(root / "diagnosis_balance_metrics.csv")
    bootstrap = pd.read_csv(root / "diagnosis_balance_country_bootstrap.csv")
    keys = ["bag", "objective", "diagnosis"]
    weighted = metrics[
        metrics["training_scheme"].eq("equal_diagnosis_weight") & metrics["model"].eq("full")
    ].copy()
    unweighted = metrics[
        metrics["training_scheme"].eq("unweighted") & metrics["model"].eq("full")
    ].copy()
    comparison = weighted.merge(unweighted, on=keys, suffixes=("_weighted", "_unweighted"), validate="one_to_one")
    comparison = comparison.merge(bootstrap[bootstrap["model"].eq("full")], on=keys, validate="one_to_one")
    comparison = comparison.merge(
        _diagnosis_weighting_pvalues(), on=keys, validate="one_to_one"
    )
    weighting_rows = pd.DataFrame(
        {
            "Analysis": "Equal diagnosis weighting",
            "BAG": comparison["bag"].map(BAG_LABEL),
            "Setting": comparison["diagnosis"].map(DX_LABEL).fillna(comparison["diagnosis"]),
            "Model or arm": comparison["objective"].map(ARM_LABEL),
            "N": comparison["n_subjects"],
            "Countries": comparison["n_countries"],
            "Primary estimate": comparison["bias_mean_weighted"],
            "Comparator estimate": comparison["bias_mean_unweighted"],
            "Difference or interaction": comparison["delta_bias_balanced_minus_unweighted"],
            "P value": comparison["country_cluster_bootstrap_p"],
            "Estimand": "Mean signed BAG bias (years)",
            "95% CI": comparison.apply(
                lambda row: f"{row['delta_bias_ci_low']:.3f} to {row['delta_bias_ci_high']:.3f}",
                axis=1,
            ),
            "Secondary estimate": comparison["mae_weighted"],
            "Secondary comparator": comparison["mae_unweighted"],
            "Secondary difference": comparison["delta_mae_balanced_minus_unweighted"],
            "Secondary 95% CI": comparison.apply(
                lambda row: f"{row['delta_mae_ci_low']:.3f} to {row['delta_mae_ci_high']:.3f}",
                axis=1,
            ),
        }
    )
    frame = pd.concat([selected_rows, diversity_rows, weighting_rows], ignore_index=True)
    frame["Holm-adjusted p"] = _holm_within(frame, "P value", ["Analysis", "BAG"])
    return PublicationTable(
        15,
        "Diagnostic-context and diagnosis-weighting sensitivity",
        frame.reset_index(drop=True),
        f"Normative selected-model R² and every R² entering the diversity-by-arm models use the {R2_ESTIMAND_PHRASE}. Selected-model tests are paired two-sided Wilcoxon tests across countries. Diversity rows report the set-size-adjusted arm interaction from the parent analysis. Diagnosis-weighting rows retain the parent bias estimand and 10,000-draw country-cluster confidence intervals. Holm adjustment is within analysis and BAG.",
    )


def _table_13() -> PublicationTable:
    rows: list[dict[str, object]] = []
    work = _runtime_root() / "work/analysis_runs" / SENSITIVITY_EVAL_RUN_ID
    # Heavy domain-representative evaluations are estimand-invariant; the
    # global delivery reads the completed ones and re-selects on its estimand.
    sensitivity = ROOT / "outputs" / "main" / SENSITIVITY_EVAL_RUN_ID / "sensitivity"
    seed = 20260624
    for bag_index, bag in enumerate(("structural", "functional")):
        main_path = (
            work / "normative_transfer" / bag / bag / "pooled" / "train_all"
            / f"{bag}_pooled_all_xgb_tree_d3_country.csv"
        )
        main_country = pd.read_csv(main_path)
        main_country = main_country[
            main_country["candidate_id"].astype(str).str.contains("_o_min_", regex=False)
        ]
        main_global = pd.read_csv(str(main_path).replace("_country.csv", "_global.csv"))
        main_global = main_global[
            main_global["candidate_id"].astype(str).str.contains("_o_min_", regex=False)
        ]
        main_scores = _candidate_country_scores(main_country, main_global)
        main_id = str(main_scores.loc[main_scores["country_balanced_r2"].idxmax(), "candidate_id"])
        main_vector = _country_vector(main_country, main_id)
        main_r2 = float(main_vector.mean())

        domain_country = pd.read_csv(
            sensitivity / "domain_imbalance" / bag / "domain_imbalance_country_all.csv",
            low_memory=False,
        )
        domain_global = pd.read_csv(
            sensitivity / "domain_imbalance" / bag / "domain_imbalance_global_all.csv",
            low_memory=False,
        )
        for family_index, (family, label) in enumerate(
            (("best_single_per_domain", "Best single exposure per domain"),
             ("within_domain_pc1", "Within-domain PC1"))
        ):
            subset = domain_country[
                domain_country["rung_id"].eq("xgb_tree_d3")
                & domain_country["candidate_family"].eq(family)
            ].copy()
            subset_global = domain_global[
                domain_global["rung_id"].eq("xgb_tree_d3")
                & domain_global["candidate_family"].eq(family)
            ].copy()
            scores = _candidate_country_scores(subset, subset_global)
            best = scores.loc[scores["country_balanced_r2"].idxmax()]
            alternative_id = str(best["candidate_id"])
            metadata = subset[subset["candidate_id"].astype(str).eq(alternative_id)].iloc[0]
            alternative = _country_vector(subset, alternative_id)
            inference = _paired_country_bootstrap(
                main_vector, alternative, seed=seed + 10 * bag_index + family_index
            )
            rows.append(
                {
                    "BAG": BAG_LABEL[bag],
                    "Representation": label,
                    "Components": int(metadata["order"]),
                    R2_COLUMN_LABEL: float(alternative.mean()),
                    f"Main synergy-arm {R2_COLUMN_LABEL}": main_r2,
                    "ΔR², alternative − main": float(alternative.mean() - main_r2),
                    "Main candidate": main_id,
                    "Alternative candidate": alternative_id,
                    "Countries": inference["n_countries"],
                    "Paired ΔR², main − alternative": inference["delta"],
                    "95% CI, lower": inference["ci_lo"],
                    "95% CI, upper": inference["ci_hi"],
                    "Country-bootstrap p": inference["p"],
                }
            )

        pca_country = pd.read_csv(
            sensitivity / "whole_exposome_pca" / bag / "whole_exposome_pca_country_all.csv"
        )
        pca = pca_country[
            pca_country["rung_id"].eq("xgb_tree_d3")
            & pca_country["candidate_id"].eq("whole_pca_pc10")
        ].copy()
        alternative = _country_vector(pca, "whole_pca_pc10")
        inference = _paired_country_bootstrap(
            main_vector, alternative, seed=seed + 10 * bag_index + 2
        )
        rows.append(
            {
                "BAG": BAG_LABEL[bag],
                "Representation": "Whole-exposome PCA",
                "Components": 10,
                R2_COLUMN_LABEL: float(alternative.mean()),
                f"Main synergy-arm {R2_COLUMN_LABEL}": main_r2,
                "ΔR², alternative − main": float(alternative.mean() - main_r2),
                "Main candidate": main_id,
                "Alternative candidate": "whole_pca_pc10",
                "Countries": inference["n_countries"],
                "Paired ΔR², main − alternative": inference["delta"],
                "95% CI, lower": inference["ci_lo"],
                "95% CI, upper": inference["ci_hi"],
                "Country-bootstrap p": inference["p"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["Holm-adjusted p"] = _holm_within(frame, "Country-bootstrap p", ["BAG"])
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    return PublicationTable(
        13,
        "Alternative exposome representations",
        frame,
        f"The three prespecified alternatives and deployed depth-3 synergy-arm model are all summarized by the {R2_ESTIMAND_PHRASE}. Whole-exposome PCA uses the prespecified ten-component comparison. Inference uses paired two-sided country bootstraps on mean country R² (10,000 draws), with Holm adjustment across the three representations within each BAG measure.",
    )


def _table_14() -> PublicationTable:
    selection = pd.read_csv(
        TABLE_SOURCE_ROOT / "selection_sensitivities/ST14_negative_omega_selection.csv"
    )
    work = _runtime_root() / "work/analysis_runs" / SENSITIVITY_EVAL_RUN_ID / "normative_transfer"
    rows: list[dict[str, object]] = []
    for bag in ("structural", "functional"):
        country = pd.read_csv(
            work / bag / bag / "pooled" / "train_all"
            / f"{bag}_pooled_all_xgb_tree_d3_country.csv"
        )
        chosen = selection[selection["bag"].eq(bag)].set_index("objective")
        synergy_id = str(chosen.loc["o_min", "candidate_id"])
        redundancy_id = str(chosen.loc["o_max", "candidate_id"])
        synergy = _country_vector(country, synergy_id)
        redundancy = _country_vector(country, redundancy_id)
        n_countries, p_value = _paired_wilcoxon(synergy, redundancy)
        rows.append(
            {
                "BAG": BAG_LABEL[bag],
                "Restricted synergy candidate": synergy_id,
                "Synergy set size": int(chosen.loc["o_min", "order"]),
                f"Synergy {R2_COLUMN_LABEL}": float(synergy.mean()),
                "Redundancy candidate": redundancy_id,
                "Redundancy set size": int(chosen.loc["o_max", "order"]),
                f"Redundancy {R2_COLUMN_LABEL}": float(redundancy.mean()),
                "Synergy − redundancy R²": float(synergy.mean() - redundancy.mean()),
                "Countries": n_countries,
                "Paired-country Wilcoxon p": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame["Holm-adjusted p"] = multipletests(
        frame["Paired-country Wilcoxon p"], method="holm"
    )[1]
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    return PublicationTable(
        14,
        "Negative-O-information arm-definition sensitivity",
        frame,
        f"The restricted synergy arm additionally requires evaluated O-information below zero. R² is the {R2_ESTIMAND_PHRASE}. Synergy and redundancy are compared with two-sided paired Wilcoxon tests across countries; P values are Holm-adjusted across the two BAG measures, matching the parent analysis.",
    )


def _table_18() -> PublicationTable:
    selection = pd.read_csv(
        TABLE_SOURCE_ROOT / "selection_sensitivities/ST18_set_size_cap_selection.csv"
    )
    caps = {5, 10, 15, 20, 21, 22, 25, 30}
    selection = selection[selection["maximum_set_size"].isin(caps)]
    score_column = "global_oof_r2" if GLOBAL_OOF_MODE else "country_balanced_r2"
    if score_column not in selection.columns:
        raise ValueError(f"ST18 source lacks the active estimand column {score_column!r}")
    selection = selection.rename(columns={score_column: "country_balanced_r2"})
    selected = (
        selection.sort_values("country_balanced_r2", ascending=False)
        .groupby(["bag", "maximum_set_size"], observed=True)
        .first()
        .reset_index()
    )
    betas = pd.read_csv(
        (RUN_ROOT / "order_cap" / "diversity_r2_betas_by_cap.csv")
        if GLOBAL_OOF_MODE
        else (ROOT / "outputs/sensitivity/dedup/order_cap/diversity_r2_betas_by_cap.csv")
    )
    condition_column = "analysis" if "analysis" in betas.columns else "condition"
    betas = betas[
        betas[condition_column].eq("Pooled")
        & betas["rung_id"].eq("xgb_tree_d3")
        & betas["order_cap"].isin(caps)
    ].copy()
    frame = selected.merge(
        betas,
        left_on=["bag", "maximum_set_size"],
        right_on=["bag", "order_cap"],
        validate="one_to_one",
    )
    frame = frame.rename(
        columns={
            "bag": "BAG",
            "maximum_set_size": "Maximum set size",
            "country_balanced_r2": f"Best {R2_COLUMN_LABEL}",
            "selected_order": "Selected set size",
            "objective": "Selected arm",
            "beta_h_red": "Redundancy-arm slope",
            "beta_h_syn": "Synergy-arm slope",
            "beta_interaction": "Arm interaction",
            "p_h_syn_interaction_perm": "Within-order permutation p",
        }
    )
    frame["BAG"] = frame["BAG"].map(BAG_LABEL)
    frame["Selected arm"] = frame["Selected arm"].map(ARM_LABEL)
    frame["Holm-adjusted permutation p"] = _holm_within(
        frame, "Within-order permutation p", ["BAG"]
    )
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    columns = [
        "BAG", "Maximum set size", f"Best {R2_COLUMN_LABEL}", "Selected set size",
        "Selected arm", "Redundancy-arm slope", "Synergy-arm slope", "Arm interaction",
        "Within-order permutation p", "Holm-adjusted permutation p", "R² estimand",
    ]
    return PublicationTable(
        18,
        "Maximum candidate-set-size sensitivity",
        frame[columns],
        f"Representative caps include the prespecified cap 21, the sign-change boundary at cap 22 and the final cap 30. Candidate R² is the {R2_ESTIMAND_PHRASE}. Within-order arm-label permutation P values are Holm-adjusted across the eight displayed caps within each BAG measure, matching the parent analysis.",
    )


def _table_17() -> PublicationTable:
    work = _runtime_root() / "work/analysis_runs" / SENSITIVITY_EVAL_RUN_ID / "sensitivity_eval"
    rows: list[dict[str, object]] = []
    specifications = (
        "baseline_covariates", "plus_education", "plus_scanner", "plus_education_scanner"
    )
    for bag in ("structural", "functional"):
        selected: dict[str, tuple[str, pd.Series, int]] = {}
        for specification in specifications:
            folder = work / "education_scanner_baseline" / bag / specification
            country = pd.read_csv(folder / f"{bag}_xgb_tree_d3_country.csv")
            global_frame = pd.read_csv(folder / f"{bag}_xgb_tree_d3_global.csv")
            synergy_ids = global_frame.loc[
                global_frame["objective"].astype(str).eq("o_min"), "candidate_id"
            ].astype(str)
            country = country[country["candidate_id"].astype(str).isin(synergy_ids)].copy()
            scores = _candidate_country_scores(
                country, global_frame[global_frame["candidate_id"].astype(str).isin(synergy_ids)]
            )
            best = scores.loc[scores["country_balanced_r2"].idxmax()]
            candidate_id = str(best["candidate_id"])
            participants = int(
                global_frame.loc[
                    global_frame["candidate_id"].astype(str).eq(candidate_id), "n_scored"
                ].iloc[0]
            )
            selected[specification] = (
                candidate_id, _country_vector(country, candidate_id), participants
            )
        reference = selected["baseline_covariates"][1]
        reference_r2 = float(reference.mean())
        for specification in specifications:
            candidate_id, vector, participants = selected[specification]
            n_countries, p_value = (
                (int(reference.notna().sum()), np.nan)
                if specification == "baseline_covariates"
                else _paired_wilcoxon(vector, reference)
            )
            rows.append(
                {
                    "Analysis": "Additional covariates",
                    "BAG": BAG_LABEL[bag],
                    "Model level": "d3",
                    "Specification or model": specification,
                    "Candidate": candidate_id,
                    "Participants": participants,
                    "Countries": n_countries,
                    "Countries with R² > 0": int(vector.gt(0).sum()),
                    R2_COLUMN_LABEL: float(vector.mean()),
                    "Reference R²": reference_r2,
                    "ΔR²": float(vector.mean() - reference_r2),
                    "Paired-country Wilcoxon p": p_value,
                }
            )

        country = pd.read_csv(
            work / "residualized_bag" / bag / "residualized_eval_xgb_tree_d3"
            / f"{bag}_xgb_tree_d3_country.csv"
        )
        global_frame = pd.read_csv(
            work / "residualized_bag" / bag / "residualized_eval_xgb_tree_d3"
            / f"{bag}_xgb_tree_d3_global.csv"
        )
        roles = {
            "Covariate baseline": global_frame["candidate_id"].astype(str).eq("__baseline__"),
            "Best single exposure": global_frame["candidate_family"].astype(str).eq("single_exposure"),
            "Best synergy-arm model": global_frame["objective"].astype(str).eq("o_min"),
            "Best redundancy-arm model": global_frame["objective"].astype(str).eq("o_max"),
        }
        chosen: dict[str, tuple[str, pd.Series]] = {}
        for role, mask in roles.items():
            ids = global_frame.loc[mask, "candidate_id"].astype(str)
            scores = _candidate_country_scores(
                country[country["candidate_id"].astype(str).isin(ids)],
                global_frame[global_frame["candidate_id"].astype(str).isin(ids)],
            )
            best = scores.loc[scores["country_balanced_r2"].idxmax()]
            candidate_id = str(best["candidate_id"])
            chosen[role] = (candidate_id, _country_vector(country, candidate_id))
        reference = chosen["Covariate baseline"][1]
        reference_r2 = float(reference.mean())
        for role, (candidate_id, vector) in chosen.items():
            n_countries, p_value = (
                (int(reference.notna().sum()), np.nan)
                if role == "Covariate baseline"
                else _paired_wilcoxon(vector, reference)
            )
            rows.append(
                {
                    "Analysis": "Residualised BAG target",
                    "BAG": BAG_LABEL[bag],
                    "Model level": "d3",
                    "Specification or model": role,
                    "Candidate": candidate_id,
                    "Participants": None,
                    "Countries": n_countries,
                    "Countries with R² > 0": int(vector.gt(0).sum()),
                    R2_COLUMN_LABEL: float(vector.mean()),
                    "Reference R²": reference_r2,
                    "ΔR²": float(vector.mean() - reference_r2),
                    "Paired-country Wilcoxon p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["Holm-adjusted p"] = _holm_within(
        frame, "Paired-country Wilcoxon p", ["Analysis", "BAG"]
    )
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    return PublicationTable(
        17,
        "Covariate and BAG-target sensitivity analyses",
        frame,
        f"Additional-covariate models use the same complete-case sample carrying education and scanner identity. Residualised-target models remove age, sex and diagnosis effects within each training fold. Every R² is the {R2_ESTIMAND_PHRASE}; the positive-country count makes negative aggregate values auditable. Non-reference specifications are compared with the displayed reference using two-sided paired Wilcoxon tests across countries, with Holm adjustment across the three comparisons within each analysis and BAG measure.",
    )


def _table_05() -> PublicationTable:
    """Top-50 composition; selection and statistic both use the active estimand."""
    frame = pd.read_csv(TABLE_SOURCE_ROOT / "table_adapters/ST05_top50_composition.csv")
    frame = frame.rename(columns={"median_country_balanced_r2": f"Median {R2_COLUMN_LABEL}"})
    frame["bag"] = frame["bag"].map(BAG_LABEL)
    frame["objective"] = frame["objective"].map(ARM_LABEL)
    frame = frame.rename(
        columns={
            "bag": "BAG", "objective": "Discovery arm", "n": "Candidates",
            "median_set_size": "Median set size", "mean_set_size": "Mean set size",
            "label_permutation_p_two_sided": "Arm-label permutation p",
            "permutations": "Permutations", "holm_p": "Holm-adjusted p",
        }
    )
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    return PublicationTable(
        5,
        "Top-50 composition",
        frame,
        f"The top 50 candidates per arm are selected on the {R2_ESTIMAND_PHRASE}. Set-size differences between arms use a two-sided arm-label permutation test, Holm-adjusted across the two BAG measures.",
    )


def _table_06() -> PublicationTable:
    """Domain-diversity permutation on the deployed depth-3 models."""
    frame = pd.read_csv(TABLE_SOURCE_ROOT / "local_analyses/diversity_permutation_d3.csv")
    frame["bag"] = frame["bag"].map(BAG_LABEL)
    frame = frame.rename(
        columns={
            "bag": "BAG",
            "beta_interaction_syn_minus_red": "Diversity slope difference (synergy − redundancy)",
            "permutation_p_two_sided": "Arm-label permutation p",
            "n_candidates": "Candidates", "permutations": "Permutations",
            "holm_p": "Holm-adjusted p",
        }
    )
    frame["R² estimand"] = R2_ESTIMAND_LABEL
    return PublicationTable(
        6,
        "Domain-diversity permutation",
        frame,
        f"Each arm's set-size-adjusted diversity slope is fitted on the {R2_ESTIMAND_PHRASE} of every depth-3 candidate. The arm difference is tested by two-sided arm-label permutation and Holm-adjusted across the two BAG measures.",
    )


def _table_07() -> PublicationTable:
    """Top-20 domain composition of the deployed depth-3 models."""
    frame = pd.read_csv(TABLE_SOURCE_ROOT / "local_analyses/domain_composition_top20_d3.csv")
    return PublicationTable(
        7,
        "Top-20 domain composition",
        frame,
        f"Composition of the 20 highest-scoring candidates per arm and model level, selected on the {R2_ESTIMAND_PHRASE}. Both estimands are retained as columns so the selection can be audited.",
    )


def _table_09() -> PublicationTable:
    """Recurrent depth-3 triplets among the selected candidates."""
    frame = pd.read_csv(TABLE_SOURCE_ROOT / "local_analyses/recurrent_triplets_top20_d3.csv")
    return PublicationTable(
        9,
        "Depth-3 recurrent triplets",
        frame,
        f"Triplet prevalence within the top-20 candidates of each arm, selected on the {R2_ESTIMAND_PHRASE}. O-information is read only from the approved triplet index; no model is refitted.",
    )


def _table_16() -> PublicationTable:
    """Country-LOCO and region-LORO sensitivity, plus the country meta-regression."""
    source = DELIVERY_ROOT / "figures/supplementary/source_data"
    parts = []
    for path in sorted(source.glob("country_region_sensitivity_source_data_*.csv")):
        if path.name.endswith("_README.csv"):
            continue
        part = pd.read_csv(path)
        part.insert(0, "source_file", path.name)
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No country/region source data under {source}")
    frame = pd.concat(parts, ignore_index=True)
    # Second component: the country meta-regression coefficients.
    meta_path = TABLE_SOURCE_ROOT / "country_meta_regression/country_meta_regression.csv"
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        meta.insert(0, "source_file", "country_meta_regression.csv")
        frame = pd.concat([frame, meta], ignore_index=True)
    return PublicationTable(
        16,
        "Country and region sensitivity",
        frame,
        f"Held-out country (LOCO) and held-out region (LORO) R² of the top-20 candidates per arm and level, selected on the {R2_ESTIMAND_PHRASE}. Plotted values are held-out fold R²; no model is refitted for the table.",
    )


def _replace_sheet(workbook, table: PublicationTable, destination_name: str) -> None:
    if destination_name in workbook.sheetnames:
        del workbook[destination_name]
    _write_table(workbook, table)
    workbook[table.sheet_name].title = destination_name


def _refresh_table_13_pca(workbook) -> None:
    """Replace only stale PCA values with the corrected country-balanced curves."""
    sheet = workbook["ST13_AltRepresentations"]
    panels = {
        "whole_exposome_pca_sensitivity_source_data_a2_struct_incremental_pcs.csv":
            PCA_SOURCE_ROOT / "whole_exposome_pca_sensitivity_source_data_a2_struct_incremental_pcs.csv",
        "whole_exposome_pca_sensitivity_source_data_b2_func_incremental_pcs.csv":
            PCA_SOURCE_ROOT / "whole_exposome_pca_sensitivity_source_data_b2_func_incremental_pcs.csv",
    }
    lookup: dict[tuple[str, str, int], tuple[float, float, float]] = {}
    for source_name, path in panels.items():
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            lookup[(source_name, str(row.model_level), int(row.pc_n))] = (
                float(row.global_oof_r2),
                float(row.level_baseline_r2),
                float(row.original_best_model_r2),
            )
    replaced = 0
    for row in range(1, sheet.max_row + 1):
        source_name = sheet.cell(row, 1).value
        if source_name in panels:
            key = (str(source_name), str(sheet.cell(row, 10).value), int(sheet.cell(row, 7).value))
            performance, baseline, original = lookup[key]
            sheet.cell(row, 11, performance)
            sheet.cell(row, 12, baseline)
            sheet.cell(row, 13, original)
            replaced += 1
        elif (
            source_name == "whole_exposome_pca_sensitivity_source_data_README.csv"
            and sheet.cell(row, 5).value == "column: global_oof_r2"
        ):
            sheet.cell(
                row,
                6,
                "unweighted mean of held-out country-level LOCO R² for that level and PC count",
            )
    if replaced != 60:
        raise ValueError(f"Expected to refresh 60 PCA rows in ST13, refreshed {replaced}")


def refresh_table_figure_alignment_sources(repro_data_root: Path) -> None:
    """Refresh lightweight ST01/ST03/ST08/ST12 sources from their figure data."""
    from repo_expo_hoi_bag.stages.run_main_local_analyses import _network
    from repo_expo_hoi_bag.stages.run_main_table_adapters import _country_balanced_vs_single

    fig2 = DELIVERY_ROOT / "figures/main/source_data/fig2_grid_v3_max_30_hpo_k10_source_data_a1_oinfo_by_set_size.csv"
    envelope = pd.read_csv(fig2).rename(
        columns={"set_size": "order", "omega_min": "min_omega", "omega_max": "max_omega"}
    )
    st01_path = TABLE_SOURCE_ROOT / "table_adapters/ST01_greedy_oinfo_by_order.csv"
    summary = pd.read_csv(st01_path)[
        ["objective", "order", "n_candidates", "median_omega"]
    ]
    envelope = envelope.merge(summary, on=["objective", "order"], validate="one_to_one")
    envelope["source_file"] = str(fig2)
    envelope = envelope[["objective", "order", "n_candidates", "median_omega", "min_omega", "max_omega", "source_file"]]
    envelope.to_csv(st01_path, index=False)

    oof_dirname = "oof_global_oof" if GLOBAL_OOF_MODE else "oof"
    oof_root = repro_data_root / "results/analysis_runs/paper_reanalysis_k10/main_statistics/model_comparison" / oof_dirname
    comparisons = pd.DataFrame(
        [
            _country_balanced_vs_single(oof_root, bag, objective, 10_000, 20261200 + 2 * bag_index + objective_index)
            for bag_index, bag in enumerate(("structural", "functional"))
            for objective_index, objective in enumerate(("o_min", "o_max"))
        ]
    )
    comparisons["holm_p_across_four_comparisons"] = multipletests(comparisons["p_raw"], method="holm")[1]
    comparisons.to_csv(TABLE_SOURCE_ROOT / "table_adapters/ST03_best_multivariate_vs_single.csv", index=False)

    fig3_dirname = "fig3_diversity_d3_global" if GLOBAL_OOF_MODE else "fig3_diversity_d3"
    scatter = pd.read_csv(
        repro_data_root / "results/analysis_runs/paper_reanalysis_k10/main_statistics" / fig3_dirname / "per_candidate_diversity_scatter.csv"
    )
    domain_frame = pd.read_csv(ROOT / "data/metadata/exposome_feature_domains.csv")
    domains = dict(zip(domain_frame["feature_name"].astype(str), domain_frame["domain"].astype(str)))
    observed_rows = []
    test_rows = []
    for bag_index, bag in enumerate(("structural", "functional")):
        bag_frame = scatter[scatter["bag"].eq(bag)]
        groups = {
            objective: bag_frame[bag_frame["objective"].eq(objective)].nlargest(20, "country_balanced_r2")
            for objective in ("o_min", "o_max")
        }
        observed = {objective: _network(group, domains) for objective, group in groups.items()}
        for objective in ("o_min", "o_max"):
            observed_rows.append({"bag": bag, "objective": objective, **observed[objective]})
        combined = pd.concat([groups["o_min"], groups["o_max"]], ignore_index=True)
        rng = np.random.default_rng(20260930 + bag_index)
        null = {metric: np.empty(10_000) for metric in observed["o_min"]}
        for draw in range(10_000):
            chosen = rng.permutation(len(combined))
            left = _network(combined.iloc[chosen[:20]], domains)
            right = _network(combined.iloc[chosen[20:]], domains)
            for metric in null:
                null[metric][draw] = left[metric] - right[metric]
        for metric in null:
            delta = observed["o_min"][metric] - observed["o_max"][metric]
            test_rows.append(
                {
                    "bag": bag,
                    "metric": metric,
                    "syn_minus_red": delta,
                    "permutation_p_two_sided": (1 + int(np.sum(np.abs(null[metric]) >= abs(delta)))) / 10_001,
                    "permutations": 10_000,
                }
            )
    network_tests = pd.DataFrame(test_rows)
    network_tests["holm_p_within_bag"] = network_tests.groupby("bag", observed=True)["permutation_p_two_sided"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    pd.DataFrame(observed_rows).to_csv(TABLE_SOURCE_ROOT / "local_analyses/network_stats_d3.csv", index=False)
    network_tests.to_csv(TABLE_SOURCE_ROOT / "local_analyses/network_permutation_d3.csv", index=False)

    residual_root = TABLE_SOURCE_ROOT / "residual_confounds"
    estimates = pd.concat(
        [pd.read_csv(residual_root / f"{bag}_{suffix}_lme_estimates.csv") for bag in ("structural", "functional") for suffix in ("syn", "red")],
        ignore_index=True,
    )
    variance = pd.concat(
        [pd.read_csv(residual_root / f"{bag}_{suffix}_lme_variance.csv") for bag in ("structural", "functional") for suffix in ("syn", "red")],
        ignore_index=True,
    )
    estimates.to_csv(residual_root / "lme_estimates_all.csv", index=False)
    variance.to_csv(residual_root / "lme_variance_all.csv", index=False)


def refresh_country_balanced_sensitivity_figures(repro_data_root: Path) -> None:
    """Redraw existing sensitivity figures from their saved country-level evaluations."""
    from repo_expo_hoi_bag.stages.run_domain_imbalance_sensitivity import (
        _plot_domain_imbalance,
    )
    from repo_expo_hoi_bag.stages.run_education_scanner_baseline_sensitivity import (
        BASELINE_LABEL,
        _plot_comparison,
    )
    from repo_expo_hoi_bag.stages.run_residualized_bag_sensitivity import (
        _build_summary_table,
        _plot_residualized,
    )
    from repo_expo_hoi_bag.stages.run_whole_exposome_pca_sensitivity import (
        _plot_pca,
        country_balanced_summary,
    )

    figures = DELIVERY_ROOT / "figures/supplementary"
    rungs = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
    all_rungs = ["ols", *rungs]
    main_run = repro_data_root / "results/analysis_runs/paper_reanalysis_k10"
    work = repro_data_root / "work/analysis_runs" / RUN_ID / "sensitivity_eval"

    domain_inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for bag in ("structural", "functional"):
        bag_root = RUN_ROOT / "domain_imbalance" / bag
        country = pd.read_csv(
            bag_root / "domain_imbalance_country_all.csv", low_memory=False
        )
        keys = [
            "candidate_id", "candidate_family", "source_label", "order", "score",
            "predictors_identity", "fit_candidate_id", "rung_id", "bag",
        ]
        metadata = country[keys].drop_duplicates(["candidate_id", "rung_id"])
        balanced = (
            country[country["n_test"].gt(0)]
            .groupby(["candidate_id", "rung_id"], observed=True)["r2"]
            .mean()
            .rename("global_oof_r2")
            .reset_index()
        )
        summary = metadata.merge(
            balanced,
            on=["candidate_id", "rung_id"],
            validate="one_to_one",
        )
        best_single = pd.read_csv(bag_root / "candidates/best_single_by_rung_source.csv")
        baseline_rows = []
        for rung in rungs:
            family = "ols" if rung == "ols" else "xgb"
            rung_root = main_run / family / bag / rung
            single_country = pd.read_csv(rung_root / "single/metrics_country.csv")
            single_scores = _candidate_country_scores(single_country).set_index("candidate_id")[
                "country_balanced_r2"
            ]
            mask = best_single["source_rung"].astype(str).eq(rung)
            best_single.loc[mask, "global_oof_r2"] = best_single.loc[
                mask, "candidate_id"
            ].astype(str).map(single_scores)
            baseline_country = pd.read_csv(rung_root / "baseline/metrics_country.csv")
            baseline_rows.append(
                {
                    "rung_id": rung,
                    "baseline_r2": float(
                        _candidate_country_scores(baseline_country)["country_balanced_r2"].iloc[0]
                    ),
                }
            )
        domain_inputs[bag] = {
            "summary": summary,
            "candidate_scores": pd.read_csv(bag_root / "domain_imbalance_candidate_scores.csv"),
            "best_single": best_single,
            "baselines": pd.DataFrame(baseline_rows),
            "original_best": pd.read_csv(bag_root / "original_complete_best_by_rung.csv"),
        }
    _plot_domain_imbalance(domain_inputs, figures, rungs, 20)

    pca_inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for bag in ("structural", "functional"):
        bag_root = RUN_ROOT / "whole_exposome_pca" / bag
        pca_inputs[bag] = {
            "variance": pd.read_csv(bag_root / "whole_exposome_pca_variance.csv"),
            "summary": country_balanced_summary(
                pd.read_csv(bag_root / "whole_exposome_pca_global_all.csv"),
                pd.read_csv(bag_root / "whole_exposome_pca_country_all.csv"),
            ),
            "baselines": pd.read_csv(bag_root / "existing_baselines_by_rung.csv"),
            "original_best": pd.read_csv(
                bag_root / "original_complete_best_by_rung.csv"
            ),
        }
    previous_estimator = os.environ.get("WHOLE_PCA_PERFORMANCE_ESTIMATOR")
    os.environ["WHOLE_PCA_PERFORMANCE_ESTIMATOR"] = "country-balanced"
    try:
        _plot_pca(pca_inputs, figures, rungs)
    finally:
        if previous_estimator is None:
            os.environ.pop("WHOLE_PCA_PERFORMANCE_ESTIMATOR", None)
        else:
            os.environ["WHOLE_PCA_PERFORMANCE_ESTIMATOR"] = previous_estimator

    specifications = (
        "baseline_covariates", "plus_education", "plus_scanner", "plus_education_scanner"
    )
    education_rungs = [
        rung
        for rung in all_rungs
        if all(
            (
                work
                / "education_scanner_baseline"
                / bag
                / specification
                / f"{bag}_{rung}_country.csv"
            ).is_file()
            for bag in ("structural", "functional")
            for specification in specifications
        )
    ]
    education_inputs: dict[str, dict[str, object]] = {}
    for bag in ("structural", "functional"):
        summaries: dict[str, pd.DataFrame] = {}
        for specification in specifications:
            pieces = []
            for rung in education_rungs:
                folder = work / "education_scanner_baseline" / bag / specification
                global_frame = pd.read_csv(folder / f"{bag}_{rung}_global.csv")
                country = pd.read_csv(folder / f"{bag}_{rung}_country.csv")
                scores = _candidate_country_scores(country).set_index("candidate_id")[
                    "country_balanced_r2"
                ]
                global_frame["global_oof_r2"] = global_frame["candidate_id"].astype(str).map(scores)
                pieces.append(global_frame)
            summaries[specification] = pd.concat(pieces, ignore_index=True)
        education_inputs[bag] = {
            "baseline_summary": summaries[BASELINE_LABEL],
            "dist_by_variant": {
                key: value for key, value in summaries.items() if key != BASELINE_LABEL
            },
        }
    _plot_comparison(
        education_inputs,
        education_rungs,
        figures,
        r2_estimand=R2_ESTIMAND_PHRASE_LOCO,
    )

    residual_rungs = [
        rung
        for rung in all_rungs
        if all(
            (
                work
                / "residualized_bag"
                / bag
                / f"residualized_eval_{rung}"
                / f"{bag}_{rung}_country.csv"
            ).is_file()
            for bag in ("structural", "functional")
        )
    ]
    residual_rows = []
    for bag in ("structural", "functional"):
        bag_rows = []
        for rung in residual_rungs:
            folder = work / "residualized_bag" / bag / f"residualized_eval_{rung}"
            global_frame = pd.read_csv(folder / f"{bag}_{rung}_global.csv")
            country = pd.read_csv(folder / f"{bag}_{rung}_country.csv")
            scores = _candidate_country_scores(country).set_index("candidate_id")[
                "country_balanced_r2"
            ]
            global_frame["r2"] = global_frame["candidate_id"].astype(str).map(scores)
            global_frame["candidate_role"] = global_frame["objective"].astype(str)
            bag_rows.append(global_frame)
        residual_rows.append(
            _build_summary_table(
                pd.concat(bag_rows, ignore_index=True), bag, residual_rungs
            )
        )
    residual_summary = pd.concat(residual_rows, ignore_index=True)
    residual_summary.to_csv(
        RUN_ROOT / "residualized_bag/residualized_bag_best_by_rung.csv", index=False
    )
    _plot_residualized(
        residual_summary,
        ["structural", "functional"],
        figures,
        "residualized_bag",
        r2_estimand=R2_ESTIMAND_PHRASE_LOCO,
    )


def refresh_country_balanced_order_cap_tests(repro_data_root: Path) -> None:
    """Refresh the parent set-size analysis and its figure with main-k10 R²."""
    os.environ.setdefault(
        "NORM_XGB_ROOT",
        str(repro_data_root / "work/analysis_runs" / RUN_ID / "normative_transfer"),
    )
    os.environ.setdefault("NORM_OLS_ROOT", str(repro_data_root / "ols_normative_loco"))
    os.environ.setdefault(
        "NORM_SINGLE_NORMATIVE_ROOT",
        str(repro_data_root / "work/analysis_runs" / RUN_ID / "normative_transfer"),
    )
    os.environ.setdefault(
        "NORM_POOLED_CANONICAL_ROOT",
        str(repro_data_root / "results/analysis_runs" / RUN_ID / "input_adapter"),
    )
    os.environ.setdefault(
        "NORM_POOLED_OLS_CANONICAL_ROOT",
        str(repro_data_root / "results/variant_a/paper_dedup_max30/canonical"),
    )
    os.environ.setdefault(
        "NORM_POOLED_MAIN_RUN_ROOT",
        str(repro_data_root / "results/analysis_runs/paper_reanalysis_k10"),
    )
    os.environ["NORM_TRANSFER_RUNGS"] = "xgb_tree_d3"
    from repo_expo_hoi_bag.stages import compute_order_cap_reconciliation as stage
    from repo_expo_hoi_bag.stages.plot_normative_transfer_grid import load_plot_data

    caps = list(range(5, 31))
    stage.ORDER_CAPS = caps
    stage.OUT_DIR = ROOT / "outputs/sensitivity/dedup/order_cap"
    stage.FIGURES_DIR = DELIVERY_ROOT / "figures/supplementary"
    frames: dict[str, pd.DataFrame] = {}
    for bag in ("structural", "functional"):
        data = load_plot_data(bag)
        pooled = data[
            data["condition"].astype(str).eq("Pooled")
            & data["rung_id"].astype(str).eq("xgb_tree_d3")
        ].copy()
        sources = set(pooled["source"].dropna().astype(str))
        if sources != {"main_fig2_global_oof" if GLOBAL_OOF_MODE else "main_fig2_country_balanced"}:
            raise ValueError(
                f"Set-size analysis must use the exact main Fig. 2 country-balanced "
                f"scores for {bag}; found sources={sorted(sources)}"
            )
        frames[bag] = pooled

    recon = stage.reconciliation_long(frames, {})
    plotted_selection = pd.concat(
        [
            recon[["bag", "order_cap", "best_syn_order", "best_syn_r2"]].rename(
                columns={
                    "order_cap": "maximum_set_size",
                    "best_syn_order": "selected_order",
                    "best_syn_r2": "country_balanced_r2",
                }
            ).assign(objective="o_min"),
            recon[["bag", "order_cap", "best_red_order", "best_red_r2"]].rename(
                columns={
                    "order_cap": "maximum_set_size",
                    "best_red_order": "selected_order",
                    "best_red_r2": "country_balanced_r2",
                }
            ).assign(objective="o_max"),
        ],
        ignore_index=True,
    )
    expected_selection = pd.read_csv(
        TABLE_SOURCE_ROOT / "selection_sensitivities/ST18_set_size_cap_selection.csv"
    )
    keys = ["bag", "objective", "maximum_set_size"]
    comparison = expected_selection.merge(
        plotted_selection,
        on=keys,
        suffixes=("_table", "_figure"),
        validate="one_to_one",
    )
    if len(comparison) != len(expected_selection) or not (
        comparison["selected_order_table"].astype(int)
        .eq(comparison["selected_order_figure"].astype(int))
        .all()
        and np.allclose(
            comparison["country_balanced_r2_table"],
            comparison["country_balanced_r2_figure"],
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Set-size figure winners do not match the main-k10 ST18 source")

    betas = stage.diversity_betas_by_cap(frames)
    by_order = stage.r2_by_order(frames)
    stage.OUT_DIR.mkdir(parents=True, exist_ok=True)
    recon.to_csv(stage.OUT_DIR / stage.RECON_CSV_NAME, index=False)
    by_order.to_csv(stage.OUT_DIR / "r2_by_order.csv", index=False)
    betas.to_csv(stage.OUT_DIR / "diversity_r2_betas_by_cap.csv", index=False)
    stage.plot_s10_set_size_sensitivity(recon, betas)


def finalize(prepared: Path, destination: Path, *, include_country_block_null: bool = True) -> Path:
    workbook = load_workbook(prepared)
    replacements = [
        (_table_01(), "ST01_HigherOrder"),
        (_table_02(), "ST02_ModelComplexity"),
        (_table_03(), "ST03_BestModels"),
        (_table_05(), "ST05_TopComposition"),
        (_table_06(), "ST06_DomainDiversity"),
        (_table_07(), "ST07_DomainComposition"),
        (_table_09(), "ST09_RecurrentTriplets"),
        (_table_16(), "ST16_Geography"),
        (_table_08(), "ST08_Cooccurrence"),
        (_table_10(), "ST10_ResidualBias"),
        (_table_11(), "ST11_NormativeTransfer"),
        (_table_12(), "ST12_MixedModels"),
        (_table_13(), "ST13_AltRepresentations"),
        (_table_14(), "ST14_NegativeOmega"),
        (_table_15(), "ST15_DiagnosticContext"),
        (_table_17(), "ST17_CovariatesTargets"),
        (_table_18(), "ST18_SetSizeCaps"),
    ]
    if include_country_block_null:
        replacements.append((_table_04(), "ST04_CountryBlockNull"))
    for table, name in replacements:
        _replace_sheet(workbook, table, name)
    missing = [name for name in SHEETS if name not in workbook.sheetnames]
    extras = [name for name in workbook.sheetnames if name not in SHEETS]
    if missing or extras:
        raise ValueError(f"Workbook inventory mismatch: missing={missing}, extras={extras}")
    workbook._sheets = [workbook[name] for name in SHEETS]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    workbook.save(temporary)
    check = load_workbook(temporary, read_only=True, data_only=True)
    if check.sheetnames != SHEETS or any(check[name].max_row < 4 for name in SHEETS):
        raise ValueError("Temporary workbook failed final sheet/row validation")
    temporary.replace(destination)
    return destination


def main() -> None:
    global ACTIVE_REPRO_DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--destination", type=Path, default=DELIVERY)
    parser.add_argument(
        "--repro-data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--defer-country-block-null",
        action="store_true",
        help="Refresh all local corrections while retaining ST04 until the cluster jobs finish.",
    )
    args = parser.parse_args()
    ACTIVE_REPRO_DATA_ROOT = args.repro_data_root.resolve()
    if GLOBAL_OOF_MODE:
        # Under the global estimand the set-size analysis and the four
        # estimand-dependent sensitivity figures are produced by their own
        # stages (render_global_oof_set_size, render_global_oof_sensitivity_
        # figures) before this workbook is assembled, so the country-balanced
        # delivery-time conversions must not run here.
        print("R2_MODE=global_oof: skipping the country-balanced figure refreshes")
    else:
        refresh_country_balanced_order_cap_tests(ACTIVE_REPRO_DATA_ROOT)
        refresh_country_balanced_sensitivity_figures(ACTIVE_REPRO_DATA_ROOT)
    refresh_table_figure_alignment_sources(ACTIVE_REPRO_DATA_ROOT)
    print(finalize(args.prepared, args.destination, include_country_block_null=not args.defer_country_block_null))


if __name__ == "__main__":
    main()
