#!/usr/bin/env python3
"""Country-cluster OOF comparisons for the HPO level winners.

Local port of the paper's ``complexity_comparisons.py`` and
``complexity_arm_comparisons.py``.  It consumes only newly regenerated OOF
predictions and applies the same 10,000-draw country bootstrap and Holm rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

LEVELS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
PAIRS = (("xgb_tree_d1", "ols"), ("xgb_tree_d2", "ols"), ("xgb_tree_d3", "ols"),
         ("xgb_tree_d2", "xgb_tree_d1"), ("xgb_tree_d3", "xgb_tree_d1"), ("xgb_tree_d3", "xgb_tree_d2"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--hpo-set", choices=("k1", "k10", "k63"), default="k10")
    parser.add_argument("--source-run-id", help="Immutable k10 OOF source; defaults to paper_reanalysis_<hpo-set>.")
    parser.add_argument("--output-run-id", help="Separate analysis namespace for derived statistics.")
    parser.add_argument("--exclude-ols", action="store_true", help="Use only k10 XGBoost OOF; required when OLS is reused from historical metrics.")
    return parser.parse_args()


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum())


def _country_r2(frame: pd.DataFrame, prediction: str) -> pd.Series:
    """Held-out-country R² values underlying the paper's balanced estimand."""
    return pd.Series(
        {
            str(country): _r2(
                group["y_true"].to_numpy(float), group[prediction].to_numpy(float)
            )
            for country, group in frame.groupby("country", sort=True)
        },
        dtype=float,
    )


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["row_id", "country", "y_true", "y_pred_full", "y_pred_base", "candidate_id"])
    return frame[np.isfinite(frame.y_true) & np.isfinite(frame.y_pred_full)].copy()


def _bootstrap(frame: pd.DataFrame, draws: int, seed: int, column_a: str, column_b: str) -> dict[str, float]:
    country_a = _country_r2(frame, column_a)
    country_b = _country_r2(frame, column_b)
    paired = pd.concat([country_a.rename("a"), country_b.rename("b")], axis=1).dropna()
    if paired.empty:
        raise ValueError("No finite paired country R² values")
    differences = (paired["a"] - paired["b"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(differences), np.repeat(1 / len(differences), len(differences)), size=draws
    )
    delta = counts @ differences / len(differences)
    p = min(1.0, max(2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0))), 1 / draws))
    return {"delta_r2": float(differences.mean()),
            "ci_lo": float(np.percentile(delta, 2.5)), "ci_hi": float(np.percentile(delta, 97.5)), "p_raw": p}


def _wilcoxon(values: np.ndarray, minimum: int) -> tuple[float, float, int, float]:
    nonzero = values[values != 0]
    if len(nonzero) < minimum:
        return np.nan, np.nan, int(len(nonzero)), float(np.median(values))
    statistic, p_value = stats.wilcoxon(nonzero, alternative="two-sided")
    return float(statistic), float(p_value), int(len(nonzero)), float(np.median(values))


def _sensitivity_tests(frame: pd.DataFrame) -> dict[str, float]:
    """Exact sensitivity tests from the paper's model_comparisons module."""
    y = frame.y_true.to_numpy(float)
    result: dict[str, float] = {}
    for loss_name, loss in (("sq", np.square), ("abs", np.abs)):
        difference = loss(y - frame.prediction_a.to_numpy(float)) - loss(y - frame.prediction_b.to_numpy(float))
        subject_stat, subject_p, subject_n, subject_median = _wilcoxon(difference, 10)
        country_means = pd.DataFrame({"country": frame.country.astype(str), "difference": difference}).groupby("country", sort=True).difference.mean().to_numpy(float)
        country_stat, country_p, country_n, country_median = _wilcoxon(country_means, 6)
        result.update({
            f"wilcoxon_subject_stat_{loss_name}": subject_stat,
            f"wilcoxon_subject_p_{loss_name}": subject_p,
            f"wilcoxon_subject_n_{loss_name}": subject_n,
            f"wilcoxon_subject_median_loss_diff_{loss_name}": subject_median,
            f"wilcoxon_country_stat_{loss_name}": country_stat,
            f"wilcoxon_country_p_{loss_name}": country_p,
            f"wilcoxon_country_n_{loss_name}": country_n,
            f"wilcoxon_country_median_loss_diff_{loss_name}": country_median,
        })
    return result


def _as_result(comparison_type: str, bag: str, objective: str, model_a: str, model_b: str, frame: pd.DataFrame, draws: int, seed: int, **extra: object) -> dict[str, object]:
    country_a = _country_r2(frame, "prediction_a")
    country_b = _country_r2(frame, "prediction_b")
    return {
        "comparison_type": comparison_type, "bag": bag, "objective": objective,
        "model_a": model_a, "model_b": model_b, "n_subjects": len(frame),
        "n_countries": frame.country.nunique(), "r2_a": float(country_a.mean()),
        "r2_b": float(country_b.mean()),
        "r2_estimand": "unweighted_mean_country_r2",
        **_bootstrap(frame, draws, seed, "prediction_a", "prediction_b"),
        **_sensitivity_tests(frame), **extra,
    }


def _comparison(root: Path, bag: str, objective: str, a: str, b: str, draws: int, seed: int) -> dict[str, object]:
    family = "level_best_syn" if objective == "o_min" else "level_best_red"
    left, right = _load(root / "oof" / family / bag / f"oof_{a}.parquet"), _load(root / "oof" / family / bag / f"oof_{b}.parquet")
    merged = left[["row_id", "country", "y_true", "y_pred_full"]].merge(right[["row_id", "y_true", "y_pred_full"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not np.allclose(merged.y_true_a, merged.y_true_b): raise ValueError("OOF outcomes differ")
    merged = merged.rename(columns={"country": "country", "y_true_a": "y_true", "y_pred_full_a": "prediction_a", "y_pred_full_b": "prediction_b"})
    return _as_result("level_selected", bag, objective, a, b, merged, draws, seed)


def _arm_comparison(root: Path, bag: str, rung: str, draws: int, seed: int) -> dict[str, object]:
    syn, red = _load(root / "oof/level_best_syn" / bag / f"oof_{rung}.parquet"), _load(root / "oof/level_best_red" / bag / f"oof_{rung}.parquet")
    merged = syn[["row_id", "country", "y_true", "y_pred_full"]].merge(red[["row_id", "y_true", "y_pred_full"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one").rename(columns={"y_true_a":"y_true","y_pred_full_a":"prediction_a","y_pred_full_b":"prediction_b"})
    return _as_result("arm_level_selected", bag, "o_min_minus_o_max", rung, rung, merged, draws, seed)


def _baseline_comparison(root: Path, bag: str, objective: str, rung: str, draws: int, seed: int) -> dict[str, object]:
    family = "level_best_syn" if objective == "o_min" else "level_best_red"
    frame = _load(root / "oof" / family / bag / f"oof_{rung}.parquet").rename(columns={"y_pred_full": "prediction_a", "y_pred_base": "prediction_b"})
    return _as_result("vs_baseline", bag, objective, rung, "baseline", frame, draws, seed)


def _single_comparison(root: Path, bag: str, objective: str, draws: int, seed: int) -> dict[str, object]:
    family = "level_best_syn" if objective == "o_min" else "level_best_red"
    multi, single = _load(root / "oof" / family / bag / "oof_xgb_tree_d3.parquet"), _load(root / "oof/level_best_single" / bag / "oof_xgb_tree_d3.parquet")
    merged = multi[["row_id","country","y_true","y_pred_full"]].merge(single[["row_id","y_true","y_pred_full"]], on="row_id", suffixes=("_a","_b"), validate="one_to_one").rename(columns={"y_true_a":"y_true","y_pred_full_a":"prediction_a","y_pred_full_b":"prediction_b"})
    return _as_result("vs_single", bag, objective, "xgb_tree_d3", "best_single", merged, draws, seed)


def _fixed_comparison(root: Path, bag: str, objective: str, source_level: str, a: str, b: str, draws: int, seed: int) -> dict[str, object]:
    family = f"fixed_{source_level}_{'syn' if objective == 'o_min' else 'red'}"
    left = _load(root / "oof" / family / bag / f"oof_{a}.parquet")
    right = _load(root / "oof" / family / bag / f"oof_{b}.parquet")
    merged = left[["row_id", "country", "y_true", "y_pred_full", "candidate_id"]].merge(
        right[["row_id", "y_true", "y_pred_full", "candidate_id"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one"
    ).rename(columns={"y_true_a": "y_true", "y_pred_full_a": "prediction_a", "y_pred_full_b": "prediction_b"})
    if not np.allclose(merged.y_true, merged.y_true_b):
        raise ValueError("OOF outcomes differ for fixed trajectory")
    same_identity = bool((merged.candidate_id_a == merged.candidate_id_b).all())
    return _as_result(f"fixed_{source_level}_candidate", bag, objective, a, b, merged, draws, seed, same_predictor_identity=same_identity, candidate_a=str(merged.candidate_id_a.iloc[0]), candidate_b=str(merged.candidate_id_b.iloc[0]))


def _fixed_arm_comparison(root: Path, bag: str, source_level: str, rung: str, draws: int, seed: int) -> dict[str, object]:
    syn = _load(root / f"oof/fixed_{source_level}_syn" / bag / f"oof_{rung}.parquet")
    red = _load(root / f"oof/fixed_{source_level}_red" / bag / f"oof_{rung}.parquet")
    merged = syn[["row_id", "country", "y_true", "y_pred_full", "candidate_id"]].merge(
        red[["row_id", "y_true", "y_pred_full", "candidate_id"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one"
    ).rename(columns={"y_true_a": "y_true", "y_pred_full_a": "prediction_a", "y_pred_full_b": "prediction_b"})
    return _as_result(f"arm_fixed_{source_level}_candidate", bag, "o_min_minus_o_max", rung, rung, merged, draws, seed, candidate_a=str(merged.candidate_id_a.iloc[0]), candidate_b=str(merged.candidate_id_b.iloc[0]))


def _reference_trajectory(root: Path, bag: str, family: str, a: str, b: str, draws: int, seed: int) -> dict[str, object]:
    left = _load(root / "oof" / family / bag / f"oof_{a}.parquet")
    right = _load(root / "oof" / family / bag / f"oof_{b}.parquet")
    merged = left[["row_id", "country", "y_true", "y_pred_full", "y_pred_base", "candidate_id"]].merge(
        right[["row_id", "y_true", "y_pred_full", "y_pred_base", "candidate_id"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one"
    ).rename(columns={"y_true_a": "y_true"})
    if family == "level_best_single":
        merged = merged.rename(columns={"y_pred_full_a": "prediction_a", "y_pred_full_b": "prediction_b"})
        kind, candidate_a, candidate_b = "single_trajectory", str(merged.candidate_id_a.iloc[0]), str(merged.candidate_id_b.iloc[0])
    else:
        merged = merged.rename(columns={"y_pred_base_a": "prediction_a", "y_pred_base_b": "prediction_b"})
        kind, candidate_a, candidate_b = "baseline_trajectory", "covariates_only", "covariates_only"
    return _as_result(kind, bag, "reference", a, b, merged, draws, seed, candidate_a=candidate_a, candidate_b=candidate_b, same_predictor_identity=candidate_a == candidate_b)


def main() -> None:
    args = _args()
    source_run_id = args.source_run_id or f"paper_reanalysis_{args.hpo_set}"
    source_run = args.repro_data_root.resolve() / "results/analysis_runs" / source_run_id
    base = source_run / "main_statistics/model_comparison"
    output = base if not args.output_run_id else args.repro_data_root.resolve() / "results/analysis_runs" / args.output_run_id / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(level for level in LEVELS if not args.exclude_ols or level != "ols")
    pairs = tuple((a, b) for a, b in PAIRS if a in levels and b in levels)
    jobs = [("level", bag, obj, a, b) for bag in ("structural","functional") for obj in ("o_min","o_max") for a,b in pairs]
    jobs += [("arm", bag, "", rung, rung) for bag in ("structural","functional") for rung in levels]
    jobs += [("baseline", bag, objective, rung, "") for bag in ("structural","functional") for objective in ("o_min","o_max") for rung in levels]
    jobs += [("single", bag, objective, "", "") for bag in ("structural","functional") for objective in ("o_min","o_max")]
    jobs += [(f"fixed_{source}", bag, objective, a, b) for source in ("d2", "d3") for bag in ("structural", "functional") for objective in ("o_min", "o_max") for a, b in pairs]
    jobs += [(f"fixed_arm_{source}", bag, "", rung, "") for source in ("d2", "d3") for bag in ("structural", "functional") for rung in levels]
    jobs += [("reference", bag, family, a, b) for bag in ("structural", "functional") for family in ("level_best_single", "level_best_syn") for a, b in pairs]
    def run(index: int, job: tuple[str,str,str,str,str]) -> dict[str,object]:
        kind, bag, objective, a, b = job
        if kind == "level": return _comparison(base, bag, objective, a, b, args.draws, 20260805 + index)
        if kind == "arm": return _arm_comparison(base, bag, a, args.draws, 20260877 + index)
        if kind == "single": return _single_comparison(base, bag, objective, args.draws, 20260950 + index)
        if kind.startswith("fixed_arm_"): return _fixed_arm_comparison(base, bag, kind.removeprefix("fixed_arm_"), a, args.draws, 20261090 + index)
        if kind.startswith("fixed_"): return _fixed_comparison(base, bag, objective, kind.removeprefix("fixed_"), a, b, args.draws, 20261010 + index)
        if kind == "reference": return _reference_trajectory(base, bag, objective, a, b, args.draws, 20261150 + index)
        return _baseline_comparison(base, bag, objective, a, args.draws, 20260910 + index)
    result = pd.DataFrame(Parallel(n_jobs=min(args.n_jobs,len(jobs)), backend="loky")(delayed(run)(i, job) for i,job in enumerate(jobs)))
    result["holm_p_within_bag_trajectory"] = np.nan
    complexity = result.comparison_type.isin(["level_selected", "fixed_d2_candidate", "fixed_d3_candidate", "single_trajectory", "baseline_trajectory"])
    for _, indices in result.loc[complexity].groupby(["comparison_type", "bag", "objective"], sort=False).groups.items():
        result.loc[indices, "holm_p_within_bag_trajectory"] = multipletests(result.loc[indices, "p_raw"], method="holm")[1]
    result["holm_p_d3_claim_within_bag_trajectory"] = np.nan
    d3_claim = complexity & result.model_a.eq("xgb_tree_d3")
    for _, indices in result.loc[d3_claim].groupby(["comparison_type", "bag", "objective"], sort=False).groups.items():
        result.loc[indices, "holm_p_d3_claim_within_bag_trajectory"] = multipletests(result.loc[indices, "p_raw"], method="holm")[1]
    result["holm_p_across_four_levels"] = np.nan
    arm = result.comparison_type.isin(["arm_level_selected", "arm_fixed_d2_candidate", "arm_fixed_d3_candidate"])
    for _, indices in result.loc[arm].groupby(["comparison_type", "bag"], sort=False).groups.items():
        result.loc[indices, "holm_p_across_four_levels"] = multipletests(result.loc[indices, "p_raw"], method="holm")[1]
    result["holm_p_within_analysis_scope"] = np.nan
    for _, indices in result.loc[complexity].groupby("comparison_type", sort=False).groups.items():
        result.loc[indices, "holm_p_within_analysis_scope"] = multipletests(result.loc[indices, "p_raw"], method="holm")[1]
    result["holm_p_all_complexity_comparisons"] = np.nan
    result.loc[complexity, "holm_p_all_complexity_comparisons"] = multipletests(result.loc[complexity, "p_raw"], method="holm")[1]
    winners = pd.read_csv(base / "level_winners.csv")
    if args.exclude_ols:
        winners = winners[winners.rung.ne("ols")].copy()
    base_rows = []
    for row in winners.itertuples(index=False):
        run_root = source_run
        root = run_root / "ols" / row.bag / "ols" if row.rung == "ols" else run_root / "xgb" / row.bag / row.rung
        model_scope = "ols" if row.rung == "ols" and row.scope in {"k10", "k63"} else row.scope
        if not (root / model_scope / "metrics_country.csv").is_file():
            root = source_run / ("ols" if row.rung == "ols" else "xgb") / row.bag / ("ols" if row.rung == "ols" else row.rung)
        country = pd.read_csv(root / model_scope / "metrics_country.csv")
        model_r2 = country.loc[(country.candidate_id.astype(str) == str(row.candidate_id)) & (country.n_test > 0), "r2"].mean()
        baseline_root = root if (root / "baseline" / "metrics_country.csv").is_file() else source_run / ("ols" if row.rung == "ols" else "xgb") / row.bag / ("ols" if row.rung == "ols" else row.rung)
        baseline = pd.read_csv(baseline_root / "baseline" / "metrics_country.csv")
        baseline_r2 = baseline.loc[baseline.n_test > 0].groupby("candidate_id").r2.mean().iloc[0]
        f2 = (model_r2 - baseline_r2) / (1 - model_r2)
        base_rows.append({"family":row.family,"bag":row.bag,"objective":row.objective,"rung":row.rung,"candidate_id":row.candidate_id,"winner_country_balanced_r2":model_r2,"baseline_country_balanced_r2":baseline_r2,"cohen_f2":f2})
    result.to_csv(output / "complexity_and_arm_comparisons.csv", index=False); pd.DataFrame(base_rows).to_csv(output / "level_winner_cohen_f2.csv", index=False)
    print(f"Saved: {output / 'complexity_and_arm_comparisons.csv'}")


if __name__ == "__main__": main()
