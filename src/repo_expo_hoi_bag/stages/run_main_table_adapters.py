#!/usr/bin/env python3
"""Finish locally derivable ST01/ST03/ST05/ST09 from k10 metrics and allowed Ω."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


BAGS = ("structural", "functional"); RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"); OBJECTIVES = ("o_min", "o_max")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-data-root", type=Path, required=True); p.add_argument("--source-run-id", default="paper_reanalysis_k10"); p.add_argument("--main-run-id", default="main")
    p.add_argument("--candidate-registry", type=Path, required=True); p.add_argument("--triplet-oinfo-dir", type=Path, required=True); p.add_argument("--historical-root", type=Path, required=True); p.add_argument("--xgb-comparison-run-id", required=True); p.add_argument("--output-subdir", default="table_adapters"); p.add_argument("--draws", type=int, default=10_000); p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def _metrics(root: Path, historical: Path, bag: str, rung: str) -> pd.DataFrame:
    if rung == "ols":
        path = historical / "results/variant_a/families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
        x = pd.read_parquet(path, filters=[("rung_id", "=", "ols")])
        score = x.groupby("candidate_id", observed=True).country_full_r2.mean().rename("country_balanced_r2").reset_index()
        score["bag"], score["rung"] = bag, rung
        return score
    directory = root / ("ols" if rung == "ols" else "xgb") / bag / ("ols" if rung == "ols" else rung) / ("ols" if rung == "ols" else "k10")
    country = pd.read_csv(directory / "metrics_country.csv")
    score = country[country.n_test > 0].groupby("candidate_id", observed=True).r2.mean().rename("country_balanced_r2").reset_index()
    score["bag"], score["rung"] = bag, rung
    return score


def _permutation_order(frame: pd.DataFrame, draws: int, seed: int) -> float:
    observed = frame.loc[frame.objective.eq("o_min"), "order"].median() - frame.loc[frame.objective.eq("o_max"), "order"].median()
    rng = np.random.default_rng(seed); labels = frame.objective.to_numpy(); null = np.empty(draws)
    for draw in range(draws):
        shuffled = rng.permutation(labels); null[draw] = frame.loc[shuffled == "o_min", "order"].median() - frame.loc[shuffled == "o_max", "order"].median()
    return (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (draws + 1)


def _country_r2(group: pd.DataFrame, prediction: str) -> float:
    y = group["y_true"].to_numpy(float)
    denominator = np.square(y - y.mean()).sum()
    if denominator <= 0:
        return np.nan
    return float(1 - np.square(y - group[prediction].to_numpy(float)).sum() / denominator)


def _country_balanced_vs_single(
    oof_root: Path,
    bag: str,
    objective: str,
    draws: int,
    seed: int,
) -> dict[str, object]:
    family = "level_best_syn" if objective == "o_min" else "level_best_red"
    columns = ["row_id", "country", "y_true", "y_pred_full", "candidate_id"]
    multivariate = pd.read_parquet(
        oof_root / family / bag / "oof_xgb_tree_d3.parquet", columns=columns
    )
    single = pd.read_parquet(
        oof_root / "level_best_single" / bag / "oof_xgb_tree_d3.parquet", columns=columns
    )
    paired = multivariate.merge(
        single[["row_id", "y_true", "y_pred_full", "candidate_id"]],
        on="row_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if not np.allclose(paired["y_true_a"], paired["y_true_b"]):
        raise ValueError(f"OOF outcomes differ for {bag}/{objective} multivariate versus single")
    paired = paired.rename(columns={"y_true_a": "y_true", "y_pred_full_a": "prediction_a", "y_pred_full_b": "prediction_b"})
    valid = np.isfinite(paired["y_true"]) & np.isfinite(paired["prediction_a"]) & np.isfinite(paired["prediction_b"])
    paired = paired[valid].copy()
    country = paired.groupby("country", sort=True).apply(
        lambda group: pd.Series(
            {
                "r2_a": _country_r2(group, "prediction_a"),
                "r2_b": _country_r2(group, "prediction_b"),
            }
        )
    ).dropna()
    country["delta_r2"] = country["r2_a"] - country["r2_b"]
    rng = np.random.default_rng(seed)
    values = country["delta_r2"].to_numpy(float)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    observed = float(values.mean())
    p_value = min(1.0, max(2 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0))), 1 / draws))
    return {
        "comparison_type": "vs_single",
        "bag": bag,
        "objective": objective,
        "model_a": "xgb_tree_d3",
        "model_b": "best_single",
        "candidate_a": str(paired["candidate_id_a"].iloc[0]),
        "candidate_b": str(paired["candidate_id_b"].iloc[0]),
        "n_subjects": len(paired),
        "n_countries": len(country),
        "r2_a": float(country["r2_a"].mean()),
        "r2_b": float(country["r2_b"].mean()),
        "delta_r2": observed,
        "ci_lo": float(np.percentile(sampled, 2.5)),
        "ci_hi": float(np.percentile(sampled, 97.5)),
        "p_raw": p_value,
        "r2_estimand": "unweighted_mean_country_r2",
        "bootstrap_unit": "country",
        "bootstrap_draws": draws,
    }


def main() -> None:
    a = _args(); runtime = a.repro_data_root.resolve(); source = runtime / "results/analysis_runs" / a.source_run_id
    registry = pd.read_parquet(a.candidate_registry); registry = registry[registry.order.le(30)].copy()
    registry["bag"] = registry.experiment_id.astype(str).str.removeprefix("pooled_oinfo_ladder_")
    scores = pd.concat([_metrics(source, a.historical_root.resolve(), bag, rung) for bag in BAGS for rung in RUNGS], ignore_index=True).merge(registry[["candidate_id", "bag", "experiment_id", "objective", "order", "thoi_o", "predictors_identity"]], on=["candidate_id", "bag"], validate="many_to_one")
    if a.smoke_test:
        print(f"Smoke test passed: {len(scores)} k10/OLS metric rows and {len(registry)} greedy candidates")
        return
    out = runtime / "results/analysis_runs" / a.main_run_id / a.output_subdir
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    # ST01: use the exact greedy catalogue and envelope displayed in Figure 2.
    greedy_path = a.historical_root.resolve() / "work/greedy/greedy_topk_by_objective_order.csv"
    greedy = pd.read_csv(greedy_path)
    greedy = greedy[greedy.objective.isin(OBJECTIVES) & pd.to_numeric(greedy.order, errors="coerce").le(30)].copy()
    greedy["omega"] = pd.to_numeric(greedy["thoi_o"], errors="raise")
    st01 = greedy.groupby(["objective", "order"], observed=True).agg(
        n_candidates=("omega", "size"), median_omega=("omega", "median"),
        min_omega=("omega", "min"), max_omega=("omega", "max")
    ).reset_index()
    st01["source_file"] = str(greedy_path)
    st01.to_csv(out / "ST01_greedy_oinfo_by_order.csv", index=False)
    # ST03: compare the same country-balanced d3 values displayed in Figure 2.
    oof_root = source / "main_statistics/model_comparison/oof"
    st03 = pd.DataFrame(
        [
            _country_balanced_vs_single(oof_root, bag, objective, a.draws, 20261200 + 2 * BAGS.index(bag) + OBJECTIVES.index(objective))
            for bag in BAGS for objective in OBJECTIVES
        ]
    )
    st03["holm_p_across_four_comparisons"] = multipletests(st03["p_raw"], method="holm")[1]
    st03.to_csv(out / "ST03_best_multivariate_vs_single.csv", index=False)
    # ST05: top-50 composition; selection and statistic both use country-balanced k10/OLS values.
    rows = []
    for bag in BAGS:
        level = scores[(scores.bag.eq(bag)) & (scores.rung.eq("xgb_tree_d3"))]
        selected = pd.concat([level[level.objective.eq(obj)].nlargest(50, "country_balanced_r2") for obj in OBJECTIVES], ignore_index=True)
        p = _permutation_order(selected, a.draws, 20261000 + BAGS.index(bag))
        for obj in OBJECTIVES:
            group = selected[selected.objective.eq(obj)]
            rows.append({"bag": bag, "objective": obj, "n": len(group), "median_set_size": group.order.median(), "mean_set_size": group.order.mean(), "median_country_balanced_r2": group.country_balanced_r2.median(), "label_permutation_p_two_sided": p, "permutations": a.draws})
    st05 = pd.DataFrame(rows); st05["holm_p"] = multipletests(st05.groupby("bag").label_permutation_p_two_sided.transform("first").drop_duplicates(), method="holm")[1].repeat(2)[:len(st05)]; st05.to_csv(out / "ST05_top50_composition.csv", index=False)
    # ST09: union of top-20 candidates at every level, deduplicated by identity; Ω is read only from its approved index.
    triplet_rows = []
    for bag in BAGS:
        chosen = pd.concat([scores[(scores.bag.eq(bag)) & (scores.rung.eq(rung)) & (scores.objective.eq(obj))].nlargest(20, "country_balanced_r2") for rung in RUNGS for obj in OBJECTIVES], ignore_index=True).drop_duplicates(["candidate_id", "objective"])
        detail = pd.read_parquet(a.triplet_oinfo_dir / f"subcomb_oinfo_{bag}.parquet", filters=[("order_k", "=", 3)])
        detail = detail[detail.candidate_id.astype(str).isin(set(chosen.candidate_id.astype(str)))].merge(chosen[["candidate_id", "objective"]], on="candidate_id", validate="many_to_one")
        for obj in OBJECTIVES:
            pool = chosen[chosen.objective.eq(obj)].candidate_id.nunique(); part = detail[detail.objective.eq(obj)].copy(); part["triplet"] = part.nplet_cols.map(lambda x: " | ".join(sorted(str(x).split("|"))))
            counts = part.groupby("triplet", observed=True).candidate_id.nunique().rename("n_candidates").reset_index(); counts["prevalence_pct"] = 100 * counts.n_candidates / pool; counts["bag"], counts["objective"], counts["n_candidates_in_arm"] = bag, obj, pool; triplet_rows.append(counts)
    st09 = pd.concat(triplet_rows, ignore_index=True).sort_values(["bag", "objective", "prevalence_pct", "triplet"], ascending=[True, True, False, True]); st09.to_csv(out / "ST09_recurrent_triplets_all_levels.csv", index=False)
    print(f"Saved local table adapters: {out}")


if __name__ == "__main__": main()
