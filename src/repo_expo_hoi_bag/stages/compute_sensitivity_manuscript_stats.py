#!/usr/bin/env python3
"""Generate every statistic used by the Results sensitivity subsection.

The stage is additive: it reads existing deduplicated sensitivity summaries and
figure Source Data, performs no model fitting, and writes claim-level comparison
tables beside the analysis that produced each result. It also writes an index
under ``outputs/sensitivity/<namespace>/manuscript_claims/`` so manuscript values
can be audited without recomputing a comparison during writing.

Analysis choices (BAG order, deployed rung, complete cap, reference cap,
diagnoses, objective and geographic series) come from ``sensitivity.yaml``.
Set ``REPO_CHECKOUT_ROOT`` to aggregate another checkout and ``DEDUP_NS`` to
select a parallel sensitivity namespace; both default to the active checkout
and ``dedup``.
"""
from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

import pandas as pd
import yaml


REPOSITORY_ROOT = Path(
    os.environ.get("REPO_CHECKOUT_ROOT", "").strip()
    or Path(__file__).resolve().parents[3]
).resolve()
CONFIG_PATH = Path(__file__).resolve().parent / "resources" / "sensitivity.yaml"


def _load_config() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        paper = config["paper_analysis"]
        manuscript = config["manuscript_sensitivity"]
        required = {
            "deployed_rung",
            "order_max",
            "primary_diagnoses",
            "diagnosis_labels",
            "objectives",
        }
        missing = required.difference(paper)
        if missing:
            raise KeyError(f"paper_analysis missing {sorted(missing)}")
        required_manuscript = {
            "default_namespace",
            "negative_o_namespace",
            "reference_order_cap",
            "primary_objective",
            "arm_comparison_country_metric",
            "country_region_series",
            "country_region_cap_split",
        }
        missing_manuscript = required_manuscript.difference(manuscript)
        if missing_manuscript:
            raise KeyError(
                f"manuscript_sensitivity missing {sorted(missing_manuscript)}"
            )
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete manuscript sensitivity configuration") from exc
    return config, manuscript


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required sensitivity input is absent: {path}")
    return pd.read_csv(path)


def _unique_glob(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Source Data file matching {pattern!r} under {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _require_nonempty(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No rows resolved for {label}")
    return frame


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _rung_label(rung_id: str) -> str:
    if rung_id == "ols":
        return "OLS"
    prefix = "xgb_tree_"
    if not rung_id.startswith(prefix):
        raise ValueError(f"Unsupported deployed rung label: {rung_id}")
    return rung_id.removeprefix(prefix)


def _bag_order(frame: pd.DataFrame, bags: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    out["_bag_order"] = pd.Categorical(out["bag"], categories=bags, ordered=True)
    sort_columns = ["_bag_order"]
    sort_columns.extend(
        column
        for column in ("condition", "objective", "candidate_family", "covariate_set")
        if column in out.columns
    )
    return out.sort_values(sort_columns).drop(columns="_bag_order").reset_index(drop=True)


def _summarize_order_cap(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
    reference_cap: int,
    complete_cap: int,
) -> list[Path]:
    output_root = sensitivity_root / "order_cap"
    summary_path = output_root / "order_cap_summary_by_cap_bag_rung.csv"
    summary = _require_csv(summary_path)
    selected = summary[
        summary["bag"].isin(bags)
        & summary["analysis"].eq("Pooled")
        & summary["rung_id"].eq(deployed_rung)
        & summary["order_cap"].isin((reference_cap, complete_cap))
    ].copy()
    selected = _require_nonempty(selected, "pooled order-cap comparison")
    selected["source_file"] = str(summary_path.relative_to(REPOSITORY_ROOT))
    order_columns = [
        "bag",
        "order_cap",
        "rung_id",
        "best_r2",
        "best_order",
        "best_objective",
        "best_syn_r2",
        "best_red_r2",
        "syn_minus_red",
        "baseline_r2",
        "best_minus_baseline",
        "source_file",
    ]
    order_output = _bag_order(selected[order_columns], bags)

    diversity_path = output_root / "diversity_r2_betas_by_cap.csv"
    diversity = _require_csv(diversity_path)
    diversity_selected = diversity[
        diversity["bag"].isin(bags)
        & diversity["condition"].eq("Pooled")
        & diversity["rung_id"].eq(deployed_rung)
        & diversity["order_cap"].isin((reference_cap, complete_cap))
    ].copy()
    diversity_selected = _require_nonempty(
        diversity_selected, "pooled diversity order-cap comparison"
    )
    diversity_selected["source_file"] = str(diversity_path.relative_to(REPOSITORY_ROOT))
    diversity_output = _bag_order(diversity_selected, bags)

    return [
        _write(order_output, output_root / "manuscript_order_cap_comparison.csv"),
        _write(
            diversity_output,
            output_root / "manuscript_diversity_cap_comparison.csv",
        ),
    ]


def _summarize_domain_imbalance(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
    cap_split: str,
) -> Path:
    output_root = sensitivity_root / "domain_imbalance"
    rows: list[dict[str, object]] = []
    for bag in bags:
        family_path = output_root / bag / "domain_imbalance_best_by_order.csv"
        families = _require_csv(family_path)
        families = _require_nonempty(
            families[families["rung_id"].eq(deployed_rung)].copy(),
            f"{bag} domain-balanced candidates at {deployed_rung}",
        )
        main_path = output_root / bag / cap_split / "original_complete_best_by_rung.csv"
        main = _require_csv(main_path)
        main = _require_nonempty(
            main[main["rung_id"].eq(deployed_rung)].copy(),
            f"{bag} main comparison at {deployed_rung}",
        ).iloc[0]
        for family, family_rows in families.groupby("candidate_family", sort=True):
            best = family_rows.loc[family_rows["global_oof_r2"].idxmax()]
            score = float(best["score"])
            rows.append(
                {
                    "bag": bag,
                    "candidate_family": family,
                    "rung_id": deployed_rung,
                    "sensitivity_best_r2": float(best["global_oof_r2"]),
                    "sensitivity_best_order": int(best["order"]),
                    "sensitivity_best_score": score,
                    "score_defined_arm": "synergy" if score < 0 else "redundancy",
                    "main_best_r2": float(main["original_complete_best_r2"]),
                    "delta_r2_sensitivity_minus_main": (
                        float(best["global_oof_r2"])
                        - float(main["original_complete_best_r2"])
                    ),
                    "source_file": str(family_path.relative_to(REPOSITORY_ROOT)),
                    "main_source_file": str(main_path.relative_to(REPOSITORY_ROOT)),
                }
            )
    return _write(
        _bag_order(pd.DataFrame(rows), bags),
        output_root / "manuscript_main_comparison.csv",
    )


def _summarize_pca(
    sensitivity_root: Path,
    figures_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
    cap_split: str,
) -> Path:
    output_root = sensitivity_root / "whole_exposome_pca"
    source_root = figures_root / "sensitivity" / "whole_exposome_pca" / cap_split / "source_data"
    deployed_label = _rung_label(deployed_rung)
    rows: list[dict[str, object]] = []
    panel_tokens = {"structural": "struct", "functional": "func"}
    for bag in bags:
        token = panel_tokens.get(bag, bag)
        performance_path = _unique_glob(source_root, f"*_{token}_incremental_pcs.csv")
        performance = _require_csv(performance_path)
        performance = _require_nonempty(
            performance[performance["model_level"].eq(deployed_label)].copy(),
            f"{bag} PCA performance at {deployed_label}",
        )
        best = performance.loc[performance["global_oof_r2"].idxmax()]
        variance_path = output_root / bag / "whole_exposome_pca_variance.csv"
        variance = _require_csv(variance_path).sort_values("pc_n")
        variance = _require_nonempty(variance, f"{bag} PCA variance")
        rows.append(
            {
                "bag": bag,
                "rung_id": deployed_rung,
                "best_pc_count": int(best["pc_n"]),
                "best_pca_r2": float(best["global_oof_r2"]),
                "main_best_r2": float(best["original_best_model_r2"]),
                "delta_r2_pca_minus_main": (
                    float(best["global_oof_r2"])
                    - float(best["original_best_model_r2"])
                ),
                "pc1_variance_fraction": float(variance.iloc[0]["explained_variance_ratio"]),
                "cumulative_variance_fraction": float(
                    variance.iloc[-1]["cumulative_explained_variance_ratio"]
                ),
                "n_components": int(variance.iloc[-1]["pc_n"]),
                "source_file": str(performance_path.relative_to(REPOSITORY_ROOT)),
                "variance_source_file": str(variance_path.relative_to(REPOSITORY_ROOT)),
            }
        )
    return _write(
        _bag_order(pd.DataFrame(rows), bags),
        output_root / "manuscript_main_comparison.csv",
    )


def _summarize_covariates(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
) -> Path:
    output_root = sensitivity_root / "education_scanner_baseline"
    source_path = output_root / "global_all_rungs.csv"
    source = _require_csv(source_path)
    selected = source[
        source["bag"].isin(bags)
        & source["rung_id"].eq(deployed_rung)
        & source["candidate_role"].eq("best_synergy")
    ].copy()
    selected = _require_nonempty(selected, "education/scanner comparison")
    reference = (
        selected[selected["covariate_set"].eq("baseline_covariates")]
        .set_index("bag")["global_oof_r2"]
    )
    selected["reference_r2"] = selected["bag"].map(reference)
    if selected["reference_r2"].isna().any():
        raise ValueError("A BAG is missing its complete-case baseline-covariate reference")
    selected["delta_r2_vs_reference"] = (
        selected["global_oof_r2"] - selected["reference_r2"]
    )
    selected["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
    columns = [
        "bag",
        "covariate_set",
        "rung_id",
        "global_oof_r2",
        "reference_r2",
        "delta_r2_vs_reference",
        "n_scored",
        "source_file",
    ]
    return _write(
        _bag_order(selected[columns], bags),
        output_root / "manuscript_covariate_comparison.csv",
    )


def _summarize_residualized_target(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
) -> Path:
    output_root = sensitivity_root / "residualized_bag"
    source_path = output_root / "residualized_bag_best_by_rung.csv"
    source = _require_csv(source_path)
    selected = source[
        source["bag"].isin(bags) & source["rung_id"].eq(deployed_rung)
    ].copy()
    selected = _require_nonempty(selected, "residualized BAG comparison")
    selected["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
    return _write(
        _bag_order(selected, bags),
        output_root / "manuscript_target_comparison.csv",
    )


def _summarize_diagnosis_balance(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    diagnoses: tuple[str, ...],
    diagnosis_labels: dict[str, str],
    primary_objective: str,
) -> Path:
    output_root = sensitivity_root / "diagnosis_balance"
    source_path = output_root / "diagnosis_balance_metrics.csv"
    source = _require_csv(source_path)
    selected = source[
        source["bag"].isin(bags)
        & source["objective"].eq(primary_objective)
        & source["diagnosis"].isin(diagnoses)
        & source["model"].eq("full")
    ].copy()
    selected = _require_nonempty(selected, "diagnosis-balanced comparison")
    index_columns = ["bag", "objective", "diagnosis"]
    equal = selected[selected["training_scheme"].eq("equal_diagnosis_weight")].set_index(
        index_columns
    )
    unweighted = selected[selected["training_scheme"].eq("unweighted")].set_index(
        index_columns
    )
    if not equal.index.equals(unweighted.index):
        raise ValueError("Weighted and unweighted diagnosis rows do not align")
    output = equal[["n", "bias_mean", "mae", "rmse", "r2"]].rename(
        columns=lambda column: f"weighted_{column}"
    )
    for column in ("bias_mean", "mae", "rmse", "r2"):
        output[f"unweighted_{column}"] = unweighted[column]
        output[f"delta_{column}_weighted_minus_unweighted"] = (
            equal[column] - unweighted[column]
        )
    output = output.reset_index()
    output["diagnosis_label"] = output["diagnosis"].map(diagnosis_labels)
    output["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
    return _write(
        _bag_order(output, bags),
        output_root / "manuscript_weighting_comparison.csv",
    )


def _summarize_geographic_generalization(
    sensitivity_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
    series: str,
    cap_split: str,
) -> Path:
    output_root = sensitivity_root / "country_region"
    rows: list[dict[str, object]] = []
    input_names = {
        "country": "country_loco_per_fold.csv",
        "region": "region_loro_per_fold.csv",
    }
    for bag in bags:
        for geographic_level, filename in input_names.items():
            source_path = output_root / bag / cap_split / filename
            source = _require_csv(source_path)
            selected = source[
                source["rung_id"].eq(deployed_rung) & source["series"].eq(series)
            ].copy()
            selected = _require_nonempty(
                selected,
                f"{bag} {geographic_level} generalization at {deployed_rung}",
            )
            nonpositive = selected.loc[selected["median_r2"] <= 0, "fold"].astype(str)
            rows.append(
                {
                    "bag": bag,
                    "geographic_level": geographic_level,
                    "rung_id": deployed_rung,
                    "series": series,
                    "n_positive_median_r2": int((selected["median_r2"] > 0).sum()),
                    "n_folds": int(len(selected)),
                    "fraction_positive_median_r2": float(
                        (selected["median_r2"] > 0).mean()
                    ),
                    "nonpositive_folds": "|".join(nonpositive),
                    "minimum_median_r2": float(selected["median_r2"].min()),
                    "maximum_median_r2": float(selected["median_r2"].max()),
                    "source_file": str(source_path.relative_to(REPOSITORY_ROOT)),
                }
            )
    return _write(
        _bag_order(pd.DataFrame(rows), bags),
        output_root / "manuscript_generalization_summary.csv",
    )


def _summarize_country_meta(
    sensitivity_root: Path,
    *,
    namespace: str,
    bags: tuple[str, ...],
) -> Path:
    output_root = sensitivity_root / "country_meta_regression"
    source_path = (
        REPOSITORY_ROOT / "outputs" / namespace / "model_comparison" / "country_meta_regression.csv"
    )
    source = _require_csv(source_path)
    unnamed = [column for column in source if column.startswith("Unnamed:")]
    if unnamed:
        source = source.rename(columns={unnamed[0]: "term"})
    source = _require_nonempty(source[source["bag"].isin(bags)].copy(), "country meta-regression")
    source["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
    return _write(
        _bag_order(source, bags),
        output_root / "manuscript_country_meta_regression.csv",
    )


def _summarize_normative_context(
    sensitivity_root: Path,
    figures_root: Path,
    *,
    bags: tuple[str, ...],
) -> list[Path]:
    output_root = sensitivity_root / "normative_context"
    source_root = figures_root / "normative" / "source_data"
    transfer_rows: list[pd.DataFrame] = []
    for bag in bags:
        source_path = _unique_glob(source_root, f"*transfer_grid_{bag}_*_a_{bag}_loco_r2.csv")
        source = _require_csv(source_path)
        best_by_arm = source.loc[source.groupby(["condition", "objective"])["r2"].idxmax()]
        best = best_by_arm.loc[best_by_arm.groupby("condition")["r2"].idxmax()].copy()
        best.insert(0, "bag", bag)
        best["delta_r2_vs_baseline"] = best["r2"] - best["baseline_r2"]
        best["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
        transfer_rows.append(best)
    transfer = _bag_order(pd.concat(transfer_rows, ignore_index=True), bags)
    transfer_path = _write(transfer, output_root / "manuscript_transfer_by_context.csv")

    aggregates: list[dict[str, object]] = []
    for bag, rows in transfer.groupby("bag", sort=False):
        level_counts = rows["model_level"].value_counts().sort_index()
        aggregates.append(
            {
                "bag": bag,
                "n_contexts": int(len(rows)),
                "n_exposome_above_baseline": int((rows["delta_r2_vs_baseline"] > 0).sum()),
                "n_synergy_wins": int(rows["objective"].eq("o_min").sum()),
                "n_redundancy_wins": int(rows["objective"].eq("o_max").sum()),
                "winning_model_levels": "|".join(
                    f"{level}:{int(count)}" for level, count in level_counts.items()
                ),
                "minimum_delta_r2_vs_baseline": float(rows["delta_r2_vs_baseline"].min()),
                "maximum_delta_r2_vs_baseline": float(rows["delta_r2_vs_baseline"].max()),
                "source_file": "|".join(sorted(rows["source_file"].unique())),
            }
        )
    aggregate_path = _write(
        _bag_order(pd.DataFrame(aggregates), bags),
        output_root / "manuscript_transfer_summary.csv",
    )

    diversity_rows: list[pd.DataFrame] = []
    for bag in bags:
        token = "struct" if bag == "structural" else "func" if bag == "functional" else bag
        source_path = _unique_glob(source_root, f"*_{token}_diversity_stats.csv")
        source = _require_csv(source_path)
        source["source_file"] = str(source_path.relative_to(REPOSITORY_ROOT))
        diversity_rows.append(source)
    diversity = _bag_order(pd.concat(diversity_rows, ignore_index=True), bags)
    diversity_path = _write(diversity, output_root / "manuscript_diversity_by_context.csv")

    diversity_aggregates: list[dict[str, object]] = []
    for bag, rows in diversity.groupby("bag", sort=False):
        diversity_aggregates.append(
            {
                "bag": bag,
                "n_contexts": int(len(rows)),
                "n_positive_redundancy_slopes": int((rows["beta_h_red"] > 0).sum()),
                "n_negative_redundancy_slopes": int((rows["beta_h_red"] < 0).sum()),
                "n_positive_synergy_slopes": int((rows["beta_h_syn"] > 0).sum()),
                "n_negative_synergy_slopes": int((rows["beta_h_syn"] < 0).sum()),
                "minimum_redundancy_slope": float(rows["beta_h_red"].min()),
                "maximum_redundancy_slope": float(rows["beta_h_red"].max()),
                "minimum_synergy_slope": float(rows["beta_h_syn"].min()),
                "maximum_synergy_slope": float(rows["beta_h_syn"].max()),
                "source_file": "|".join(sorted(rows["source_file"].unique())),
            }
        )
    diversity_aggregate_path = _write(
        _bag_order(pd.DataFrame(diversity_aggregates), bags),
        output_root / "manuscript_diversity_summary.csv",
    )
    return [transfer_path, aggregate_path, diversity_path, diversity_aggregate_path]


def _summarize_arm_definition(
    sensitivity_root: Path,
    alternative_root: Path,
    *,
    bags: tuple[str, ...],
    deployed_rung: str,
    complete_cap: int,
) -> list[Path]:
    main_order_path = sensitivity_root / "order_cap" / "order_cap_summary_by_cap_bag_rung.csv"
    alt_order_path = alternative_root / "order_cap" / "order_cap_summary_by_cap_bag_rung.csv"
    main_order = _require_csv(main_order_path)
    alt_order = _require_csv(alt_order_path)

    filters = lambda frame: frame[
        frame["bag"].isin(bags)
        & frame["analysis"].eq("Pooled")
        & frame["rung_id"].eq(deployed_rung)
        & frame["order_cap"].eq(complete_cap)
    ].copy()
    main_selected = _require_nonempty(filters(main_order), "path-only arm definition")
    alt_selected = _require_nonempty(filters(alt_order), "negative-O arm definition")
    columns = ["bag", "best_syn_r2", "best_red_r2", "best_r2", "best_objective", "best_order"]
    output = main_selected[columns].merge(
        alt_selected[columns],
        on="bag",
        suffixes=("_path_only", "_negative_o"),
        validate="one_to_one",
    )
    for column in ("best_syn_r2", "best_red_r2", "best_r2"):
        output[f"delta_{column}_negative_o_minus_path_only"] = (
            output[f"{column}_negative_o"] - output[f"{column}_path_only"]
        )

    main_diversity_path = sensitivity_root / "order_cap" / "diversity_r2_betas_by_cap.csv"
    alt_diversity_path = alternative_root / "order_cap" / "diversity_r2_betas_by_cap.csv"
    main_diversity = _require_csv(main_diversity_path)
    alt_diversity = _require_csv(alt_diversity_path)
    diversity_filters = lambda frame: frame[
        frame["bag"].isin(bags)
        & frame["condition"].eq("Pooled")
        & frame["rung_id"].eq(deployed_rung)
        & frame["order_cap"].eq(complete_cap)
    ].copy()
    main_diversity = _require_nonempty(
        diversity_filters(main_diversity), "path-only diversity definition"
    )
    alt_diversity = _require_nonempty(
        diversity_filters(alt_diversity), "negative-O diversity definition"
    )
    diversity_columns = ["bag", "n_obs", "beta_h_red", "beta_h_syn", "beta_interaction"]
    diversity = main_diversity[diversity_columns].merge(
        alt_diversity[diversity_columns],
        on="bag",
        suffixes=("_path_only", "_negative_o"),
        validate="one_to_one",
    )
    output = output.merge(diversity, on="bag", validate="one_to_one")
    output["n_candidates_removed"] = (
        output["n_obs_path_only"] - output["n_obs_negative_o"]
    )
    output["delta_beta_interaction_negative_o_minus_path_only"] = (
        output["beta_interaction_negative_o"]
        - output["beta_interaction_path_only"]
    )
    output["path_only_source_file"] = str(main_order_path.relative_to(REPOSITORY_ROOT))
    output["negative_o_source_file"] = str(alt_order_path.relative_to(REPOSITORY_ROOT))
    output["path_only_diversity_source_file"] = str(
        main_diversity_path.relative_to(REPOSITORY_ROOT)
    )
    output["negative_o_diversity_source_file"] = str(
        alt_diversity_path.relative_to(REPOSITORY_ROOT)
    )
    definition_path = _write(
        _bag_order(output, bags),
        alternative_root / "manuscript_comparison" / "negative_o_vs_path_only.csv",
    )
    comparison_root = alternative_root / "manuscript_comparison"
    arm_summary_path = comparison_root / "negative_o_arm_comparison_summary.csv"
    arm_country_path = comparison_root / "negative_o_arm_comparison_by_country.csv"
    _require_csv(arm_summary_path)
    _require_csv(arm_country_path)
    return [definition_path, arm_summary_path, arm_country_path]


def _write_index(paths: Iterable[Path], output_root: Path) -> Path:
    rows = []
    for path in paths:
        frame = _require_csv(path)
        rows.append(
            {
                "table": path.name,
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "n_rows": int(len(frame)),
                "columns": "|".join(frame.columns),
            }
        )
    return _write(pd.DataFrame(rows), output_root / "manuscript_claim_table_index.csv")


def main() -> None:
    config, manuscript = _load_config()
    paper = config["paper_analysis"]
    bags = tuple(str(bag) for bag in config["defaults"]["primary_bags"])
    diagnoses = tuple(str(diagnosis) for diagnosis in paper["primary_diagnoses"])
    diagnosis_labels = {
        str(diagnosis): str(label)
        for diagnosis, label in paper["diagnosis_labels"].items()
    }
    deployed_rung = str(paper["deployed_rung"])
    complete_cap = int(paper["order_max"])
    reference_cap = int(manuscript["reference_order_cap"])
    primary_objective = str(manuscript["primary_objective"])
    series = str(manuscript["country_region_series"])
    cap_split = str(manuscript["country_region_cap_split"])
    namespace = (
        os.environ.get("DEDUP_NS", "").strip()
        or str(manuscript["default_namespace"])
    )

    sensitivity_base = REPOSITORY_ROOT / "outputs" / "sensitivity"
    sensitivity_root = sensitivity_base / namespace
    figures_root = REPOSITORY_ROOT / "outputs" / "figures" / namespace
    alternative_namespace = str(manuscript["negative_o_namespace"])
    alternative_root = sensitivity_base / alternative_namespace

    written: list[Path] = []
    written.extend(
        _summarize_order_cap(
            sensitivity_root,
            bags=bags,
            deployed_rung=deployed_rung,
            reference_cap=reference_cap,
            complete_cap=complete_cap,
        )
    )
    written.append(
        _summarize_domain_imbalance(
            sensitivity_root,
            bags=bags,
            deployed_rung=deployed_rung,
            cap_split=cap_split,
        )
    )
    written.append(
        _summarize_pca(
            sensitivity_root,
            figures_root,
            bags=bags,
            deployed_rung=deployed_rung,
            cap_split=cap_split,
        )
    )
    written.append(
        _summarize_covariates(
            sensitivity_root,
            bags=bags,
            deployed_rung=deployed_rung,
        )
    )
    written.append(
        _summarize_residualized_target(
            sensitivity_root,
            bags=bags,
            deployed_rung=deployed_rung,
        )
    )
    written.append(
        _summarize_diagnosis_balance(
            sensitivity_root,
            bags=bags,
            diagnoses=diagnoses,
            diagnosis_labels=diagnosis_labels,
            primary_objective=primary_objective,
        )
    )
    written.append(
        _summarize_geographic_generalization(
            sensitivity_root,
            bags=bags,
            deployed_rung=deployed_rung,
            series=series,
            cap_split=cap_split,
        )
    )
    written.append(
        _summarize_country_meta(
            sensitivity_root,
            namespace=namespace,
            bags=bags,
        )
    )
    written.extend(
        _summarize_normative_context(
            sensitivity_root,
            figures_root,
            bags=bags,
        )
    )
    written.extend(
        _summarize_arm_definition(
            sensitivity_root,
            alternative_root,
            bags=bags,
            deployed_rung=deployed_rung,
            complete_cap=complete_cap,
        )
    )
    index_path = _write_index(
        written,
        sensitivity_root / "manuscript_claims",
    )
    print(f"Wrote {len(written)} claim tables and index {index_path}")


if __name__ == "__main__":
    main()
