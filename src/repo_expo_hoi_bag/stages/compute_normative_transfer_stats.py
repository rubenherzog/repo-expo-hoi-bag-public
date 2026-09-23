#!/usr/bin/env python3
"""Summarize diagnosis-transfer performance from the configured dedup bundle.

This stage is aggregation-only: it selects the best pooled model for every
configured train-to-test diagnosis transfer and computes paired country-level
contrasts. It does not refit a model. The bundle marker, transfer grid, BAGs,
deployed rung, objectives, and interaction-order cap all come from the shared
paper-analysis configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.sensitivity_common import (
    CHECKOUT_ROOT,
    CONFIG_PATH,
    bundle_root,
    load_sensitivity_config,
    paper_analysis_config,
)


def _global_oof_mode() -> bool:
    """Whether the global pooled OOF estimand is active."""
    return os.environ.get("R2_MODE", "").strip() == "global_oof"


def _best_pooled(
    global_metrics: pd.DataFrame,
    *,
    train_diagnosis: str,
    test_diagnosis: str,
    objective: str,
    rung_id: str,
    order_max: int,
) -> pd.Series | None:
    pool = global_metrics[
        (global_metrics["train_dx"].astype(str) == train_diagnosis)
        & (global_metrics["test_dx"].astype(str) == test_diagnosis)
        & (global_metrics["rung_id"].astype(str) == rung_id)
        & (global_metrics["objective"].astype(str) == objective)
    ].copy()
    if objective != "baseline":
        if "order" not in pool.columns:
            raise KeyError("Normative-transfer metrics do not contain candidate order")
        pool = pool[pd.to_numeric(pool["order"], errors="coerce") <= order_max]
    pool = pool[np.isfinite(pd.to_numeric(pool["country_balanced_r2"], errors="coerce"))]
    if pool.empty:
        return None
    return pool.loc[pd.to_numeric(pool["country_balanced_r2"]).idxmax()]


def _country_vector(
    country_metrics: pd.DataFrame,
    *,
    candidate_id: str,
    train_diagnosis: str,
    test_diagnosis: str,
    rung_id: str,
) -> pd.Series:
    pool = country_metrics[
        (country_metrics["candidate_id"].astype(str) == candidate_id)
        & (country_metrics["train_dx"].astype(str) == train_diagnosis)
        & (country_metrics["test_dx"].astype(str) == test_diagnosis)
        & (country_metrics["rung_id"].astype(str) == rung_id)
    ]
    return pool.set_index("fold_country")["r2"]


def _paired_country(first: pd.Series, second: pd.Series) -> dict[str, object]:
    common = first.index.intersection(second.index)
    first_values = first.loc[common].to_numpy(dtype=float)
    second_values = second.loc[common].to_numpy(dtype=float)
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    difference = first_values[finite] - second_values[finite]
    n_countries = int(difference.size)
    if n_countries < 3:
        return {
            "n_countries": n_countries,
            "median_delta": np.nan,
            "wilcoxon_p": np.nan,
            "sign_p": np.nan,
            "wins": np.nan,
        }
    try:
        wilcoxon_p = float(
            stats.wilcoxon(difference, zero_method="wilcox").pvalue
        )
    except ValueError:
        wilcoxon_p = np.nan
    wins = int((difference > 0).sum())
    return {
        "n_countries": n_countries,
        "median_delta": float(np.median(difference)),
        "wilcoxon_p": wilcoxon_p,
        "sign_p": float(stats.binomtest(wins, n_countries, 0.5).pvalue),
        "wins": f"{wins}/{n_countries}",
    }


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def main() -> None:
    cfg = load_sensitivity_config()
    paper_cfg = paper_analysis_config(cfg)
    transfer_cfg = paper_cfg.normative_transfer
    source_root = bundle_root().resolve()
    required_marker = source_root / str(transfer_cfg["required_bundle_marker"])
    configured_source = os.environ.get("NORMATIVE_TRANSFER_SOURCE_DIR", "").strip()
    if not configured_source and not required_marker.is_file():
        raise FileNotFoundError(
            "The configured dedup-bundle marker is absent; refusing to summarize "
            f"normative transfer from {source_root}. Missing: {required_marker}"
        )

    family = str(transfer_cfg["model_family"])
    configured_transfers = transfer_cfg["transfers"]
    if not configured_transfers or any(
        len(values) != 2 for values in configured_transfers
    ):
        raise ValueError("normative_transfer.transfers must contain diagnosis pairs")
    transfers = [
        (str(values[0]), str(values[1])) for values in configured_transfers
    ]

    source_dir = (
        Path(configured_source)
        if configured_source
        else source_root / f"{family}_normative_loco"
    )
    configured_output = os.environ.get("NORMATIVE_TRANSFER_OUTPUT_DIR", "").strip()
    output_dir = (
        Path(configured_output)
        if configured_output
        else CHECKOUT_ROOT / "outputs" / "dedup" / "model_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for bag in paper_cfg.primary_bags:
        active_marker = (
            source_dir / bag / f"{family}_norm_run_manifest.csv"
            if configured_source
            else required_marker
        )
        global_path = source_dir / bag / f"{family}_norm_global_all.csv"
        country_path = source_dir / bag / f"{family}_norm_country_all.csv"
        missing = [
            path
            for path in (active_marker, global_path, country_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing normative-transfer inputs for {bag}: {missing}"
            )
        global_metrics = pd.read_csv(global_path)
        country_metrics = pd.read_csv(country_path)
        country_metrics["r2"] = pd.to_numeric(country_metrics["r2"], errors="coerce")
        if "n_test" in country_metrics.columns:
            country_metrics = country_metrics[
                pd.to_numeric(country_metrics["n_test"], errors="coerce").gt(0)
            ].copy()
        country_metrics = country_metrics[np.isfinite(country_metrics["r2"])].copy()
        transfer_keys = ["candidate_id", "rung_id", "family_id", "train_dx", "test_dx"]
        if _global_oof_mode():
            # The normative evaluation already stores the pooled global OOF R2
            # for each transfer cell; per-country values are not averaged.
            if "global_oof_r2" not in global_metrics.columns:
                raise ValueError(f"{global_path} lacks global_oof_r2")
            global_metrics["country_balanced_r2"] = pd.to_numeric(
                global_metrics["global_oof_r2"], errors="coerce"
            )
        else:
            balanced = (
                country_metrics.groupby(transfer_keys, observed=True)["r2"]
                .mean()
                .rename("country_balanced_r2")
                .reset_index()
            )
            global_metrics = global_metrics.merge(
                balanced, on=transfer_keys, how="left", validate="one_to_one"
            )
        if global_metrics["country_balanced_r2"].isna().any():
            raise ValueError(f"Missing normative scores in {country_path}")

        for train_diagnosis, test_diagnosis in transfers:
            selected = {
                objective: _best_pooled(
                    global_metrics,
                    train_diagnosis=train_diagnosis,
                    test_diagnosis=test_diagnosis,
                    objective=objective,
                    rung_id=paper_cfg.deployed_rung,
                    order_max=paper_cfg.order_max,
                )
                for objective in (*paper_cfg.objectives, "baseline")
            }
            if any(selected[objective] is None for objective in paper_cfg.objectives):
                continue
            synergy = selected["o_min"]
            redundancy = selected["o_max"]
            baseline = selected["baseline"]
            assert synergy is not None and redundancy is not None

            r2_synergy = float(synergy["country_balanced_r2"])
            r2_redundancy = float(redundancy["country_balanced_r2"])
            r2_baseline = (
                float(baseline["country_balanced_r2"])
                if baseline is not None
                else np.nan
            )
            vectors = {
                objective: _country_vector(
                    country_metrics,
                    candidate_id=str(row["candidate_id"]),
                    train_diagnosis=train_diagnosis,
                    test_diagnosis=test_diagnosis,
                    rung_id=paper_cfg.deployed_rung,
                )
                for objective, row in (("o_min", synergy), ("o_max", redundancy))
            }
            baseline_vector = (
                _country_vector(
                    country_metrics,
                    candidate_id=str(baseline["candidate_id"]),
                    train_diagnosis=train_diagnosis,
                    test_diagnosis=test_diagnosis,
                    rung_id=paper_cfg.deployed_rung,
                )
                if baseline is not None
                else pd.Series(dtype=float)
            )
            best_objective = (
                "o_min" if r2_synergy >= r2_redundancy else "o_max"
            )
            best_r2 = max(r2_synergy, r2_redundancy)
            synergy_vs_redundancy = _paired_country(
                vectors["o_min"], vectors["o_max"]
            )
            best_vs_baseline = (
                _paired_country(vectors[best_objective], baseline_vector)
                if not baseline_vector.empty
                else {}
            )
            rows.append(
                {
                    "bag": bag,
                    "train_dx": train_diagnosis,
                    "test_dx": test_diagnosis,
                    "transfer": f"{train_diagnosis}->{test_diagnosis}",
                    "n_scored": int(synergy["n_scored"]),
                    "r2_baseline": round(r2_baseline, 4),
                    "r2_best_syn": round(r2_synergy, 4),
                    "r2_best_red": round(r2_redundancy, 4),
                    "best_objective": best_objective,
                    "best_arm": paper_cfg.objective_metadata[best_objective][
                        "arm_label"
                    ],
                    "r2_best": round(best_r2, 4),
                    "delta_best_vs_base": (
                        round(best_r2 - r2_baseline, 4)
                        if np.isfinite(r2_baseline)
                        else np.nan
                    ),
                    "syn_id": str(synergy["candidate_id"]),
                    "red_id": str(redundancy["candidate_id"]),
                    "source_bundle": str(source_dir.resolve()),
                    "source_bundle_marker": str(active_marker.resolve()),
                    "rung_id": paper_cfg.deployed_rung,
                    "order_max": paper_cfg.order_max,
                    "r2_estimand": "global_oof_r2" if _global_oof_mode() else "unweighted_mean_country_r2",
                    "configuration_source": str(CONFIG_PATH.resolve()),
                    "synVred_n_countries": synergy_vs_redundancy["n_countries"],
                    "synVred_median_dR2": synergy_vs_redundancy["median_delta"],
                    "synVred_wins": synergy_vs_redundancy["wins"],
                    "synVred_wilcoxon_p": synergy_vs_redundancy["wilcoxon_p"],
                    "synVred_sign_p": synergy_vs_redundancy["sign_p"],
                    "bestVbase_n_countries": best_vs_baseline.get(
                        "n_countries", np.nan
                    ),
                    "bestVbase_median_dR2": best_vs_baseline.get(
                        "median_delta", np.nan
                    ),
                    "bestVbase_wins": best_vs_baseline.get("wins", np.nan),
                    "bestVbase_wilcoxon_p": best_vs_baseline.get(
                        "wilcoxon_p", np.nan
                    ),
                    "bestVbase_sign_p": best_vs_baseline.get("sign_p", np.nan),
                }
            )

    results = pd.DataFrame(rows)
    csv_path = output_dir / "normative_transfer_results.csv"
    results.to_csv(csv_path, index=False)

    lines = ["# Normative-transfer results\n"]
    lines.append(
        f"Source bundle: `{source_root}`. Rung: `{paper_cfg.deployed_rung}`; "
        f"set size <= {paper_cfg.order_max}. Country-level tests use country as "
        "the paired cluster unit.\n"
    )
    for bag in paper_cfg.primary_bags:
        subset = results[results["bag"] == bag]
        if subset.empty:
            continue
        lines.extend(
            [
                f"\n## {bag.capitalize()} BAG\n",
                "| Transfer | n | R² base | R² syn | R² red | best | ΔR² best−base | best−base sign P |",
                "|---|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.transfer} | {row.n_scored} | {row.r2_baseline} | "
                f"{row.r2_best_syn} | {row.r2_best_red} | {row.best_objective} | "
                f"{row.delta_best_vs_base} | {_format_p_value(row.bestVbase_sign_p)} |"
            )
    markdown_path = output_dir / "normative_transfer_results.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
