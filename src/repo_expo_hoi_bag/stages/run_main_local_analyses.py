#!/usr/bin/env python3
"""Run every non-fitting main analysis derivable from completed k10 outputs.

Allowed inputs are the new k10 run, the public feature-domain metadata and,
only for recurrent triplets, the immutable O-information index.  The stage
never reads the previous delivered analysis outputs and never fits a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


BAGS = ("structural", "functional")
OBJECTIVES = ("o_min", "o_max")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--main-run-id", default="main")
    parser.add_argument("--triplet-oinfo-index", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _entropy_domains(value: str, domains: dict[str, str]) -> float:
    counts = Counter(domains[feature] for feature in value.split("|") if feature)
    total = sum(counts.values())
    return float(-sum((n / total) * np.log2(n / total) for n in counts.values())) if total else np.nan


def _network(frame: pd.DataFrame, domains: dict[str, str]) -> dict[str, float]:
    nodes: Counter[str] = Counter(); edges: Counter[tuple[str, str]] = Counter()
    for predictors in frame.predictors_identity.astype(str):
        feature_domains = [domains[feature] for feature in predictors.split("|") if feature]
        nodes.update(set(feature_domains))
        edges.update(
            tuple(sorted((left, right)))
            for left, right in combinations(feature_domains, 2)
            if left != right
        )
    n_nodes, n_edges = len(nodes), len(edges)
    degrees = Counter()
    for (left, right), weight in edges.items(): degrees[left] += weight; degrees[right] += weight
    return {"n_domains": float(n_nodes), "n_edges": float(n_edges), "density": n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else np.nan, "mean_weighted_degree": float(np.mean(list(degrees.values()))) if degrees else np.nan}


def _diversity(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, object]:
    def interaction(labels: np.ndarray) -> float:
        syn = (labels == "o_min").astype(float)
        h = frame.shannon_h.to_numpy(float)
        design = np.column_stack([np.ones(len(frame)), h, syn, h * syn, frame.order.to_numpy(float)])
        return float(np.linalg.lstsq(design, frame.country_balanced_r2.to_numpy(float), rcond=None)[0][3])
    observed = interaction(frame.objective.to_numpy())
    rng = np.random.default_rng(seed); null = np.empty(draws)
    for index in range(draws):
        labels = frame.objective.to_numpy().copy()
        for indices in frame.groupby("order", observed=True).indices.values():
            labels[indices] = rng.permutation(labels[indices])
        null[index] = interaction(labels)
    p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (draws + 1)
    return {"beta_interaction_syn_minus_red": observed, "permutation_p_two_sided": p, "n_candidates": len(frame), "permutations": draws}


def _triplets(top: pd.DataFrame, index: Path, bag: str, objective: str) -> pd.DataFrame:
    allowed = set(top.model_id.astype(str))
    if index.is_dir():
        index = index / f"subcomb_oinfo_{bag}.parquet"
    if not index.is_file():
        raise FileNotFoundError(f"Missing permitted O-information triplet index for {bag}: {index}")
    source = pd.read_parquet(index, filters=[("bag", "=", bag)])
    source = source[(source.candidate_id.astype(str).isin(allowed)) & (source.order_k == 3)].copy()
    source["triplet"] = source.nplet_cols.astype(str).map(lambda x: " | ".join(sorted(x.split("|"))))
    result = source.groupby("triplet", observed=True).agg(n_candidates=("candidate_id", "nunique"), omega=("o_info", "first")).reset_index()
    result["frequency_pct"] = 100 * result.n_candidates / len(allowed)
    result["bag"], result["objective"] = bag, objective
    return result.sort_values(["frequency_pct", "triplet"], ascending=[False, True], kind="mergesort")


def main() -> None:
    args = _args(); runtime = args.repro_data_root.resolve()
    # Each estimand has its own Figure 3 statistics directory; the scatter's
    # ``country_balanced_r2`` column is the carrier for whichever estimand was
    # selected there, so only the source directory changes.
    global_oof = os.environ.get("R2_MODE", "").strip() == "global_oof"
    source = (
        runtime / "results" / "analysis_runs" / args.source_run_id / "main_statistics"
        / ("fig3_diversity_d3_global" if global_oof else "fig3_diversity_d3")
    )
    scatter_path, recipe_path = source / "per_candidate_diversity_scatter.csv", source / "per_candidate_recipe_top20.csv"
    if not scatter_path.is_file() or not recipe_path.is_file() or not args.triplet_oinfo_index.exists():
        raise FileNotFoundError("Required k10 scatter/recipe or permitted O-information triplet index is missing")
    scatter, recipe = pd.read_csv(scatter_path), pd.read_csv(recipe_path)
    required = {"model_id", "objective", "order", "country_balanced_r2", "predictors_identity", "bag", "shannon_h"}
    if required.difference(scatter.columns): raise ValueError(f"k10 scatter columns missing {sorted(required.difference(scatter.columns))}")
    if args.smoke_test:
        print(f"Smoke test passed: {len(scatter)} k10 d3 candidates; {len(recipe)} top-20 feature rows; triplets={args.triplet_oinfo_index}")
        return
    out = runtime / "results" / "analysis_runs" / args.main_run_id / "local_analyses"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite existing local analysis: {out}")
    out.mkdir(parents=True)
    domain_frame = pd.read_csv(Path(__file__).resolve().parents[3] / "data/metadata/exposome_feature_domains.csv")
    domains = dict(zip(domain_frame.feature_name.astype(str), domain_frame.domain.astype(str)))
    diversity_rows: list[dict[str, object]] = []; network_rows: list[dict[str, object]] = []; permutation_rows: list[dict[str, object]] = []; triplet_parts: list[pd.DataFrame] = []
    for bag_index, bag in enumerate(BAGS):
        bag_frame = scatter[scatter.bag.eq(bag)].copy()
        diversity_rows.append({"bag": bag, **_diversity(bag_frame, args.draws, 20260920 + bag_index)})
        groups = {objective: bag_frame[bag_frame.objective.eq(objective)].nlargest(20, "country_balanced_r2") for objective in OBJECTIVES}
        observed = {objective: _network(group, domains) for objective, group in groups.items()}
        for objective in OBJECTIVES: network_rows.append({"bag": bag, "objective": objective, **observed[objective]})
        combined = pd.concat([groups["o_min"], groups["o_max"]], ignore_index=True)
        rng = np.random.default_rng(20260930 + bag_index)
        metrics = tuple(observed["o_min"])
        null = {metric: np.empty(args.draws) for metric in metrics}
        for draw in range(args.draws):
            chosen = rng.permutation(len(combined))
            left, right = _network(combined.iloc[chosen[:20]], domains), _network(combined.iloc[chosen[20:]], domains)
            for metric in metrics: null[metric][draw] = left[metric] - right[metric]
        for metric in metrics:
            delta = observed["o_min"][metric] - observed["o_max"][metric]
            permutation_rows.append({"bag": bag, "metric": metric, "syn_minus_red": delta, "permutation_p_two_sided": (1 + int(np.sum(np.abs(null[metric]) >= abs(delta)))) / (args.draws + 1), "permutations": args.draws})
        for objective in OBJECTIVES: triplet_parts.append(_triplets(groups[objective], args.triplet_oinfo_index, bag, objective))
    diversity = pd.DataFrame(diversity_rows); diversity["holm_p"] = multipletests(diversity.permutation_p_two_sided, method="holm")[1]
    pd.DataFrame(network_rows).to_csv(out / "network_stats_d3.csv", index=False)
    network_perm = pd.DataFrame(permutation_rows); network_perm["holm_p_within_bag"] = network_perm.groupby("bag", observed=True).permutation_p_two_sided.transform(lambda x: multipletests(x, method="holm")[1]); network_perm.to_csv(out / "network_permutation_d3.csv", index=False)
    diversity.to_csv(out / "diversity_permutation_d3.csv", index=False)
    recipe.to_csv(out / "domain_composition_top20_d3.csv", index=False)
    pd.concat(triplet_parts, ignore_index=True).to_csv(out / "recurrent_triplets_top20_d3.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"analysis_label": "main", "source_k10": str(source), "r2_mode": "global_oof" if global_oof else "country_balanced", "performance_estimator": "global_oof_r2" if global_oof else "country_balanced_r2", "allowed_oinfo_triplet_index": str(args.triplet_oinfo_index), "forbidden_previous_output_inputs": True, "draws": args.draws}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved local main analyses: {out}")


if __name__ == "__main__": main()
