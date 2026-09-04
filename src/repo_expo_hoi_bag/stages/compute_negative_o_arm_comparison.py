#!/usr/bin/env python3
"""Compare the best negative-O synergy and redundancy models by LOCO fold.

This aggregation-only stage selects the best model in each arm from the
canonical global metrics, then compares their matched country-level held-out
R-squared values with a two-sided Wilcoxon signed-rank test. It performs no
model fitting and writes both the country-level contrasts and the claim-level
summary beside the negative-O sensitivity analysis.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml


REPOSITORY_ROOT = Path(
    os.environ.get("REPRO_DATA_ROOT", "").strip()
    or os.environ.get("REPO_CHECKOUT_ROOT", "").strip()
    or Path(__file__).resolve().parents[3]
).resolve()
CONFIG_PATH = Path(__file__).resolve().parent / "resources" / "sensitivity.yaml"


def _load_configuration() -> tuple[dict[str, object], dict[str, object]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        paper = config["paper_analysis"]
        comparison = config["arm_comparison"]
        required_paper = {"objectives", "deployed_rung", "order_max"}
        required_comparison = {
            "negative_o_namespace",
            "primary_objective",
            "arm_comparison_country_metric",
        }
        if missing := required_paper.difference(paper):
            raise KeyError(f"paper_analysis missing {sorted(missing)}")
        if missing := required_comparison.difference(comparison):
            raise KeyError(f"arm_comparison missing {sorted(missing)}")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete negative-O arm-comparison configuration") from exc
    return paper, comparison


def _canonical_root() -> Path:
    root = os.environ.get("REPRO_DATA_ROOT", "").strip()
    if not root:
        raise EnvironmentError(
            "REPRO_DATA_ROOT must point to the deduplicated analysis bundle"
        )
    return (
        Path(root)
        / "runs"
        / "oinfo_only"
        / "variant_a"
        / "families"
        / "pooled_oinfo_ladder"
        / "canonical"
    )


def _metrics_paths(bag: str) -> tuple[Path, Path]:
    root = _canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}"
    return root / "metrics_global_long.parquet", root / "metrics_country_long.parquet"


def _require_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical metric table is absent: {path}")
    return pd.read_parquet(path)


def _best_model(
    metrics: pd.DataFrame,
    *,
    objective: str,
    rung_id: str,
    order_max: int,
    require_negative_o: bool,
) -> pd.Series:
    required = {"candidate_id", "objective", "rung_id", "order", "full_r2", "thoi_o"}
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"Canonical global metrics missing columns: {sorted(missing)}")
    pool = metrics[
        metrics["objective"].astype(str).eq(objective)
        & metrics["rung_id"].astype(str).eq(rung_id)
        & pd.to_numeric(metrics["order"], errors="coerce").le(order_max)
    ].copy()
    if require_negative_o:
        pool = pool[pd.to_numeric(pool["thoi_o"], errors="coerce").lt(0)]
    pool["full_r2"] = pd.to_numeric(pool["full_r2"], errors="coerce")
    pool = pool[np.isfinite(pool["full_r2"])]
    if pool.empty:
        criterion = " with negative evaluated O-information" if require_negative_o else ""
        raise ValueError(
            f"No {objective} candidates at {rung_id}, order <= {order_max}{criterion}"
        )
    return pool.loc[pool["full_r2"].idxmax()]


def _rank_biserial(difference: pd.Series) -> float:
    nonzero = difference[difference.ne(0)].astype(float)
    if nonzero.empty:
        return 0.0
    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero.gt(0)].sum())
    negative = float(ranks[nonzero.lt(0)].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def _compare_bag(
    bag: str,
    *,
    rung_id: str,
    order_max: int,
    synergy_objective: str,
    redundancy_objective: str,
    country_metric: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    global_path, country_path = _metrics_paths(bag)
    global_metrics = _require_parquet(global_path)
    country_metrics = _require_parquet(country_path)
    synergy = _best_model(
        global_metrics,
        objective=synergy_objective,
        rung_id=rung_id,
        order_max=order_max,
        require_negative_o=True,
    )
    redundancy = _best_model(
        global_metrics,
        objective=redundancy_objective,
        rung_id=rung_id,
        order_max=order_max,
        require_negative_o=False,
    )

    required_country = {"candidate_id", "rung_id", "fold_country", country_metric}
    missing = required_country.difference(country_metrics.columns)
    if missing:
        raise KeyError(f"Canonical country metrics missing columns: {sorted(missing)}")
    candidate_ids = {
        "synergy": str(synergy["candidate_id"]),
        "redundancy": str(redundancy["candidate_id"]),
    }
    selected = country_metrics[
        country_metrics["rung_id"].astype(str).eq(rung_id)
        & country_metrics["candidate_id"].astype(str).isin(candidate_ids.values())
    ][["fold_country", "candidate_id", country_metric]].copy()
    selected[country_metric] = pd.to_numeric(selected[country_metric], errors="coerce")
    wide = selected.pivot(
        index="fold_country",
        columns="candidate_id",
        values=country_metric,
    )
    wide = wide[[candidate_ids["synergy"], candidate_ids["redundancy"]]].dropna()
    if len(wide) < 3:
        raise ValueError(f"Fewer than three matched country folds for {bag}")
    wide.columns = ["synergy_country_r2", "redundancy_country_r2"]
    wide = wide.reset_index()
    wide.insert(0, "bag", bag)
    wide["delta_country_r2_redundancy_minus_synergy"] = (
        wide["redundancy_country_r2"] - wide["synergy_country_r2"]
    )
    difference = wide["delta_country_r2_redundancy_minus_synergy"]
    test = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    global_synergy = float(synergy["full_r2"])
    global_redundancy = float(redundancy["full_r2"])
    summary = {
        "bag": bag,
        "rung_id": rung_id,
        "order_max": order_max,
        "synergy_candidate_id": candidate_ids["synergy"],
        "redundancy_candidate_id": candidate_ids["redundancy"],
        "global_r2_synergy": global_synergy,
        "global_r2_redundancy": global_redundancy,
        "delta_global_r2_redundancy_minus_synergy": global_redundancy - global_synergy,
        "n_paired_countries": int(len(wide)),
        "median_country_delta_r2_redundancy_minus_synergy": float(difference.median()),
        "mean_country_delta_r2_redundancy_minus_synergy": float(difference.mean()),
        "redundancy_country_wins": int(difference.gt(0).sum()),
        "synergy_country_wins": int(difference.lt(0).sum()),
        "country_ties": int(difference.eq(0).sum()),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_two_sided": float(test.pvalue),
        "rank_biserial_redundancy_minus_synergy": _rank_biserial(difference),
        "country_metric": country_metric,
        "global_source_file": str(global_path),
        "country_source_file": str(country_path),
    }
    return wide, summary


def main() -> None:
    paper, comparison = _load_configuration()
    namespace = str(comparison["negative_o_namespace"])
    synergy_objective = str(comparison["primary_objective"])
    country_metric = str(comparison["arm_comparison_country_metric"])
    objectives = tuple(str(value) for value in paper["objectives"])
    alternatives = [value for value in objectives if value != synergy_objective]
    if synergy_objective not in objectives or len(alternatives) != 1:
        raise ValueError("The configured synergy objective must identify one of two arms")
    redundancy_objective = alternatives[0]
    bags = tuple(str(value) for value in paper["residual_bias_bags"])
    rung_id = str(paper["deployed_rung"])
    order_max = int(paper["order_max"])

    country_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for bag in bags:
        country, summary = _compare_bag(
            bag,
            rung_id=rung_id,
            order_max=order_max,
            synergy_objective=synergy_objective,
            redundancy_objective=redundancy_objective,
            country_metric=country_metric,
        )
        country_frames.append(country)
        summary_rows.append(summary)

    output_root = (
        REPOSITORY_ROOT
        / "outputs"
        / "sensitivity"
        / namespace
        / "comparison"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    country_path = output_root / "negative_o_arm_comparison_by_country.csv"
    summary_path = output_root / "negative_o_arm_comparison_summary.csv"
    pd.concat(country_frames, ignore_index=True).to_csv(country_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {country_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
