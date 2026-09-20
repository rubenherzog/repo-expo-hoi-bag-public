#!/usr/bin/env python3
"""Definitive four-specification robustness tables (historical, k1, k10, k63).

This is deliberately a read-only synthesis of completed candidate evaluations
and regenerated OOF comparisons.  Candidate winners are selected separately
for global OOF R² and the simple country-balanced R², exactly as requested;
no preliminary winner is carried from one estimator to the other.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
BAGS = ("structural", "functional")
RUNGS = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
OBJECTIVES = ("o_min", "o_max")
SPECS = ("historical", "k1", "k10", "k63")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-data-root", type=Path, required=True)
    p.add_argument("--historical-reference-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _domains() -> dict[str, str]:
    frame = pd.read_csv(ROOT / "data/metadata/exposome_feature_domains.csv")
    return dict(zip(frame.feature_name.astype(str), frame.domain.astype(str)))


def _entropy(value: str, domains: dict[str, str]) -> float:
    counts = Counter(domains.get(v, "UNKNOWN") for v in str(value).split("|") if v)
    n = sum(counts.values())
    return -sum((x / n) * np.log2(x / n) for x in counts.values()) if n else np.nan


def _hpo_metrics(
    root: Path, reference_root: Path, spec: str, estimator: str
) -> pd.DataFrame:
    run_id = "paper_reanalysis_single_set_params_k10" if spec == "k1" else f"paper_reanalysis_{spec}"
    scope = "k10" if spec == "k1" else spec
    parts: list[pd.DataFrame] = []
    registry = pd.read_parquet(
        reference_root
        / "results/variant_a/families/pooled_oinfo_ladder/canonical/candidate_registry.parquet"
    )
    for bag in BAGS:
        meta = registry[registry.experiment_id.eq(f"pooled_oinfo_ladder_{bag}")][["candidate_id", "objective", "order", "predictors_identity"]]
        for rung in RUNGS:
            folder = root / "results/analysis_runs" / run_id / ("ols" if rung == "ols" else "xgb") / bag / ("ols" if rung == "ols" else rung) / ("ols" if rung == "ols" else scope)
            file = folder / ("metrics_global.csv" if estimator == "global_oof_r2" else "metrics_country.csv")
            if not file.is_file():
                folder = root / "results/analysis_runs/paper_reanalysis_k10" / ("ols" if rung == "ols" else "xgb") / bag / ("ols" if rung == "ols" else rung) / ("ols" if rung == "ols" else scope)
                file = folder / ("metrics_global.csv" if estimator == "global_oof_r2" else "metrics_country.csv")
            score = pd.read_csv(file)
            if estimator == "country_balanced_r2":
                score = score[score.n_test > 0].groupby("candidate_id", as_index=False).r2.mean()
            else:
                score = score.rename(columns={"global_oof_r2": "r2"})
            score = score[["candidate_id", "r2"]].merge(meta, on="candidate_id", validate="one_to_one")
            score["bag"], score["rung"] = bag, rung
            parts.append(score)
    return pd.concat(parts, ignore_index=True).rename(columns={"r2": "r2"})


def _historical_metrics(reference_root: Path, estimator: str) -> pd.DataFrame:
    ref = reference_root
    parts: list[pd.DataFrame] = []
    for bag in BAGS:
        path = ref / "results/variant_a/families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}" / ("metrics_global_long.parquet" if estimator == "global_oof_r2" else "metrics_country_long.parquet")
        frame = pd.read_parquet(path)
        if estimator == "global_oof_r2":
            take = frame[["candidate_id", "objective", "order", "predictors_identity", "rung_id", "full_r2"]].rename(columns={"rung_id":"rung", "full_r2":"r2"})
        else:
            # Parquet dictionary columns arrive as categoricals.  observed=True
            # is essential here: the historical table otherwise materializes
            # the full Cartesian product of candidate categories.
            take = frame.groupby(["candidate_id", "objective", "order", "predictors_identity", "rung_id"], as_index=False, observed=True).country_full_r2.mean().rename(columns={"rung_id":"rung", "country_full_r2":"r2"})
        take["bag"] = bag; parts.append(take)
    return pd.concat(parts, ignore_index=True)


def _reference_scores(
    root: Path, reference_root: Path, spec: str, estimator: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return best-single and baseline score per bag/rung under one estimator."""
    if spec == "historical":
        ref = reference_root / "results/variant_a"
        parts: list[pd.DataFrame] = []
        for bag in BAGS:
            for rung in RUNGS:
                folder = ref / f"single_exposure_eval_{'ols' if rung == 'ols' else rung}" / bag
                file = folder / ("single_exposure_global.csv" if estimator == "global_oof_r2" else "single_exposure_country.csv")
                x = pd.read_csv(file)
                if estimator != "global_oof_r2":
                    x = x[x.n_test > 0].groupby("feature_name", as_index=False).r2.mean()
                    value = x.r2.max()
                else:
                    value = x.global_oof_r2.max()
                parts.append(pd.DataFrame({"bag":[bag], "rung":[rung], "best_single_r2":[value]}))
        singles = pd.concat(parts, ignore_index=True)
        candidate = _historical_metrics(reference_root, estimator)
        baseline = candidate.groupby(["bag", "rung"], as_index=False).base_r2.mean() if "base_r2" in candidate else pd.DataFrame()
        # country historical records retain the base value under its country name.
        if estimator != "global_oof_r2":
            baseparts=[]
            for bag in BAGS:
                p=ref / "families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_country_long.parquet"
                x=pd.read_parquet(p).groupby("rung_id",as_index=False, observed=True).country_base_r2.mean(); x["bag"]=bag; baseparts.append(x)
            baseline=pd.concat(baseparts,ignore_index=True).rename(columns={"rung_id":"rung","country_base_r2":"baseline_r2"})
        else:
            # recover the duplicated global base columns before their selection-only projection
            baseparts=[]
            for bag in BAGS:
                p=ref / "families/pooled_oinfo_ladder/canonical/per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
                x=pd.read_parquet(p).groupby("rung_id",as_index=False, observed=True).base_r2.mean(); x["bag"]=bag; baseparts.append(x)
            baseline=pd.concat(baseparts,ignore_index=True).rename(columns={"rung_id":"rung","base_r2":"baseline_r2"})
        return singles, baseline
    run_id = "paper_reanalysis_single_set_params_k10" if spec == "k1" else f"paper_reanalysis_{spec}"
    parts=[]; bases=[]
    for bag in BAGS:
        for rung in RUNGS:
            folder=root / "results/analysis_runs" / run_id / ("ols" if rung=="ols" else "xgb") / bag / ("ols" if rung=="ols" else rung)
            if not (folder / "single").is_dir():
                folder=root / "results/analysis_runs/paper_reanalysis_k10" / ("ols" if rung=="ols" else "xgb") / bag / ("ols" if rung=="ols" else rung)
            def get(scope: str) -> pd.DataFrame:
                x=pd.read_csv(folder / scope / ("metrics_global.csv" if estimator=="global_oof_r2" else "metrics_country.csv"))
                return x.rename(columns={"global_oof_r2":"r2"})[["candidate_id","r2"]] if estimator=="global_oof_r2" else x[x.n_test>0].groupby("candidate_id",as_index=False).r2.mean()
            parts.append({"bag":bag,"rung":rung,"best_single_r2":get("single").r2.max()})
            bases.append({"bag":bag,"rung":rung,"baseline_r2":get("baseline").r2.mean()})
    return pd.DataFrame(parts), pd.DataFrame(bases)


def _performance(
    root: Path, reference_root: Path, spec: str, estimator: str
) -> pd.DataFrame:
    scores = (
        _historical_metrics(reference_root, estimator)
        if spec == "historical"
        else _hpo_metrics(root, reference_root, spec, estimator)
    )
    scores = scores[pd.to_numeric(scores.order, errors="raise") <= 30]
    winners = scores.sort_values(["r2", "candidate_id"], ascending=[False, True], kind="mergesort").groupby(["bag", "objective", "rung"], as_index=False, observed=True).first()
    singles, baseline = _reference_scores(root, reference_root, spec, estimator)
    out=winners.merge(singles,on=["bag","rung"],validate="many_to_one").merge(baseline,on=["bag","rung"],validate="many_to_one")
    out.insert(0,"specification",spec); out.insert(1,"estimator",estimator)
    out["delta_vs_best_single"]=out.r2-out.best_single_r2; out["delta_vs_baseline"]=out.r2-out.baseline_r2
    return out


def _diversity(
    root: Path,
    reference_root: Path,
    spec: str,
    estimator: str,
    domains: dict[str, str],
) -> pd.DataFrame:
    scores = (
        _historical_metrics(reference_root, estimator)
        if spec == "historical"
        else _hpo_metrics(root, reference_root, spec, estimator)
    )
    scores=scores[(scores.order<=30)&scores.rung.isin(["xgb_tree_d2","xgb_tree_d3"])].copy(); scores["entropy"]=[_entropy(x,domains) for x in scores.predictors_identity]
    rows=[]
    for (bag,rung,obj), x in scores.groupby(["bag","rung","objective"],sort=True):
        raw=stats.pearsonr(x.entropy,x.r2)
        design=np.column_stack([np.ones(len(x)),x.order.to_numpy(float)])
        er=x.entropy-np.linalg.lstsq(design,x.entropy,rcond=None)[0]@design.T
        rr=x.r2-np.linalg.lstsq(design,x.r2,rcond=None)[0]@design.T
        partial=stats.pearsonr(er,rr)
        rows.append({"specification":spec,"estimator":estimator,"bag":bag,"rung":rung,"objective":obj,"n":len(x),"raw_pearson_r":raw.statistic,"raw_p":raw.pvalue,"partial_r_given_set_size":partial.statistic,"partial_p_given_set_size":partial.pvalue})
    return pd.DataFrame(rows)


def main() -> None:
    a = _args()
    root = a.repro_data_root.resolve()
    reference_root = a.historical_reference_root.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    performance = pd.concat(
        [
            _performance(root, reference_root, s, e)
            for s in SPECS
            for e in ("global_oof_r2", "country_balanced_r2")
        ],
        ignore_index=True,
    )
    diversity = pd.concat(
        [
            _diversity(root, reference_root, s, e, _domains())
            for s in SPECS
            for e in ("global_oof_r2", "country_balanced_r2")
        ],
        ignore_index=True,
    )
    performance.to_csv(out/"performance_winners_all_specs.csv",index=False); diversity.to_csv(out/"diversity_correlations_all_specs.csv",index=False)
    # Preserve the exact 10k country-bootstrap/global-OOF comparison tables
    # generated by the historical method for every HPO specification.
    comp=[]
    historic = pd.read_csv(ROOT / "outputs/dedup/model_comparison/comparison_results.csv")
    historic.insert(0, "specification", "historical")
    historic.insert(1, "test_family", "historical_country_cluster_bootstrap")
    comp.append(historic)
    for s in ("k1","k10","k63"):
        p=root/f"results/analysis_runs/paper_reanalysis_{s}/main_statistics/model_comparison/complexity_and_arm_comparisons.csv"
        x=pd.read_csv(p); x.insert(0,"specification",s); x.insert(1,"test_family","hpo_country_cluster_bootstrap"); comp.append(x)
    pd.concat(comp,ignore_index=True, sort=False).to_csv(out/"global_oof_country_bootstrap_tests_all_specs.csv",index=False)
    print(f"Saved definitive tables: {out}")

if __name__ == "__main__": main()
