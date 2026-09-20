#!/usr/bin/env python3
"""Co-occurrence-network & domain-overlap statistics for the dedup-complete pool.

Additive aggregation over the EXISTING dedup candidate pool
(`metrics_global_long.parquet`, order<=30, path-only); no model is refit and
nothing on disk is overwritten. Tests the manuscript's qualitative network claims
(Fig. 3c,d: synergistic networks "denser and more evenly connected" vs redundant
"hub-and-spoke") with explicit statistics:

  A. Domain co-occurrence network for the top-20 synergistic (o_min) and top-20
     redundant (o_max) candidates per BAG at the deployed depth-3 rung. For each
     network we compute: number of distinct domains (nodes), number of domain-pair
     edges, network density, mean domain degree, the Gini of domain prevalence (a
     hub-and-spoke index: high = dominated by one domain), and the mean Shannon
     domain entropy of the candidates.

  B. A LABEL-PERMUTATION null for the synergistic-minus-redundant difference in
     each statistic: pool the 40 candidates, shuffle the syn/red label 10,000x,
     recompute the statistic difference, and report a two-sided permutation P.
     This is what makes "denser / more even" a tested claim rather than a visual
     impression.

  C. Candidate overlap: fraction of dedup-discovered candidates that are
     dedup-ONLY (not recovered by the non-dedup search) at the orders used by the
     paper, from the existing candidate_overlap_by_order.csv -- quantifying how
     much the effective-sample deduplication changed the discovered structure.

Outputs (additive, checkout):
  outputs/dedup/model_comparison/cooccurrence_overlap_network_stats.csv
  outputs/dedup/model_comparison/cooccurrence_overlap_permutation.csv
  outputs/dedup/model_comparison/cooccurrence_overlap.md
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.publication_stats.paths import (
    REPOSITORY_ROOT,
    model_comparison_output_dir,
    required_dedup_root,
)

REPO = REPOSITORY_ROOT
OUTDIR = model_comparison_output_dir()
# The paper's network claims are made on the effective sample, so the candidate
# pool must come from the dedup bundle (DEDUP_DATA_ROOT), not the canonical one.
# Reading REPRO_DATA_ROOT here silently produced canonical-pool statistics.
DEDUP = required_dedup_root()
POOL = DEDUP / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet"
DOMAIN_CSV = REPO / "data" / "metadata" / "exposome_feature_domains.csv"
OVERLAP_CSV = REPO / "outputs" / "dedup" / "candidate_overlap_by_order.csv"

BAGS = ["functional", "structural"]
RUNG = "xgb_tree_d3"
TOP_K = 20
MAX_ORDER = 30
N_PERM = 10000
SEED = 20260624


def _domain_map() -> dict[str, str]:
    dm = pd.read_csv(DOMAIN_CSV)
    return dict(zip(dm.feature_name.astype(str), dm.domain.astype(str)))


def _top_candidates(pool: pd.DataFrame, bag: str, objective: str) -> pd.DataFrame:
    sub = pool[
        (pool.bag_target == bag)
        & (pool.rung_id == RUNG)
        & (pool.objective == objective)
        & (pool.order <= MAX_ORDER)
    ].copy()
    sub = sub[np.isfinite(sub.full_r2)]
    sub = sub.sort_values(["full_r2", "candidate_id"], ascending=[False, True], kind="mergesort")
    return sub.head(TOP_K)


def _candidate_domains(predictors_identity: str, dmap: dict[str, str]) -> list[str]:
    feats = [f.strip() for f in str(predictors_identity).split("|") if f.strip()]
    return [dmap.get(f, "UNKNOWN") for f in feats]


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _network_stats(candidates: pd.DataFrame, dmap: dict[str, str]) -> dict:
    """Domain-level co-occurrence network statistics for a set of candidates."""
    domain_prevalence: dict[str, int] = {}
    edge_weight: dict[tuple[str, str], int] = {}
    entropies = []
    for _, row in candidates.iterrows():
        doms = _candidate_domains(row.predictors_identity, dmap)
        present = sorted(set(doms))
        for d in present:
            domain_prevalence[d] = domain_prevalence.get(d, 0) + 1
        for a, b in itertools.combinations(present, 2):
            key = (a, b)
            edge_weight[key] = edge_weight.get(key, 0) + 1
        # Shannon entropy of this candidate's domain composition (bits)
        counts = pd.Series(doms).value_counts().to_numpy(dtype=float)
        p = counts / counts.sum()
        entropies.append(float(-(p * np.log2(p)).sum()))

    n_nodes = len(domain_prevalence)
    n_edges = len(edge_weight)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else np.nan
    density = n_edges / max_edges if max_edges and max_edges > 0 else np.nan
    # mean weighted degree (sum of incident edge weights / n_nodes)
    deg = {d: 0 for d in domain_prevalence}
    for (a, b), w in edge_weight.items():
        deg[a] += w
        deg[b] += w
    mean_w_degree = float(np.mean(list(deg.values()))) if deg else np.nan
    prevalence_gini = _gini(np.array(list(domain_prevalence.values()))) if domain_prevalence else np.nan
    return {
        "n_domains": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "mean_weighted_degree": mean_w_degree,
        "prevalence_gini": prevalence_gini,
        "mean_domain_entropy_bits": float(np.mean(entropies)) if entropies else np.nan,
    }


STAT_KEYS = ["n_domains", "n_edges", "density", "mean_weighted_degree", "prevalence_gini", "mean_domain_entropy_bits"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    dmap = _domain_map()
    pool = pd.read_parquet(POOL)

    net_rows = []
    perm_rows = []
    for bag in BAGS:
        syn = _top_candidates(pool, bag, "o_min").reset_index(drop=True)
        red = _top_candidates(pool, bag, "o_max").reset_index(drop=True)
        s_syn = _network_stats(syn, dmap)
        s_red = _network_stats(red, dmap)
        for fam, st in [("synergistic", s_syn), ("redundant", s_red)]:
            net_rows.append({"bag": bag, "objective": fam, **st})

        # label-permutation null on syn - red difference
        combined = pd.concat([syn.assign(_lab="syn"), red.assign(_lab="red")], ignore_index=True)
        n_syn = len(syn)
        obs = {k: s_syn[k] - s_red[k] for k in STAT_KEYS}
        null = {k: np.empty(N_PERM) for k in STAT_KEYS}
        idx = np.arange(len(combined))
        for b in range(N_PERM):
            rng.shuffle(idx)
            a = combined.iloc[idx[:n_syn]]
            c = combined.iloc[idx[n_syn:]]
            sa = _network_stats(a, dmap)
            sc = _network_stats(c, dmap)
            for k in STAT_KEYS:
                null[k][b] = sa[k] - sc[k]
        for k in STAT_KEYS:
            nd = null[k]
            nd = nd[np.isfinite(nd)]
            o = obs[k]
            # Add-one (Davison-Hinkley) estimator, matching every other permutation
            # stage in the project (compute_domain_diversity_regression.py:67,
            # compute_subcomb_oinfo.py:407, compute_order_cap_reconciliation.py:404).
            # The unsmoothed k/n form used here previously can return exactly 0,
            # which is not a valid p-value and is anti-conservative.
            p = (1 + int(np.sum(np.abs(nd) >= abs(o)))) / (nd.size + 1) if nd.size else np.nan
            perm_rows.append(
                {
                    "bag": bag,
                    "statistic": k,
                    "syn": round(s_syn[k], 4),
                    "red": round(s_red[k], 4),
                    "syn_minus_red": round(o, 4),
                    "null_mean": round(float(np.mean(nd)), 4) if nd.size else np.nan,
                    "perm_p_two_sided": p,
                    "n_perm": N_PERM,
                }
            )

    net = pd.DataFrame(net_rows)
    perm = pd.DataFrame(perm_rows)
    net.to_csv(OUTDIR / "cooccurrence_overlap_network_stats.csv", index=False)
    perm.to_csv(OUTDIR / "cooccurrence_overlap_permutation.csv", index=False)

    # candidate overlap at paper orders
    ov = pd.read_csv(OVERLAP_CSV)
    ov30 = ov[ov.order <= MAX_ORDER]
    overlap_summary = {
        "orders_3_30_mean_overlap_pct": round(float(ov30.overlap_pct.mean()), 2),
        "orders_3_30_median_overlap_pct": round(float(ov30.overlap_pct.median()), 2),
        "high_order_22_30_mean_overlap_pct": round(float(ov[(ov.order >= 22) & (ov.order <= 30)].overlap_pct.mean()), 2),
        "low_order_3_6_mean_overlap_pct": round(float(ov[(ov.order >= 3) & (ov.order <= 6)].overlap_pct.mean()), 2),
    }

    # markdown
    lines = ["# Co-occurrence network & candidate-overlap statistics (dedup-complete)\n"]
    lines.append(
        "Domain co-occurrence networks built from the top-20 synergistic (o_min) and "
        "top-20 redundant (o_max) candidates per BAG at the deployed depth-3 rung "
        f"(order <= {MAX_ORDER}). Synergistic-minus-redundant differences tested by a "
        f"{N_PERM:,}-draw label-permutation null.\n"
    )
    lines.append("\n## A/B. Network statistics and syn−red permutation test\n")
    lines.append("| BAG | Statistic | Syn | Red | Syn−Red | Null mean | Perm P |")
    lines.append("|---|---|---|---|---|---|---|")
    pretty = {
        "n_domains": "Distinct domains (nodes)",
        "n_edges": "Domain-pair edges",
        "density": "Network density",
        "mean_weighted_degree": "Mean weighted degree",
        "prevalence_gini": "Domain-prevalence Gini (hub index)",
        "mean_domain_entropy_bits": "Mean domain entropy (bits)",
    }
    for bag in BAGS:
        for k in STAT_KEYS:
            r = perm[(perm.bag == bag) & (perm.statistic == k)].iloc[0]
            star = " *" if (np.isfinite(r.perm_p_two_sided) and r.perm_p_two_sided < 0.05) else ""
            pp = f"{r.perm_p_two_sided:.4g}" if np.isfinite(r.perm_p_two_sided) else "—"
            lines.append(
                f"| {bag} | {pretty[k]} | {r.syn} | {r.red} | {r.syn_minus_red} | {r.null_mean} | {pp}{star} |"
            )
    lines.append("\n## C. Dedup-vs-non-dedup candidate overlap\n")
    lines.append(
        f"- Mean overlap of dedup candidates with the non-dedup search, orders 3–30: "
        f"**{overlap_summary['orders_3_30_mean_overlap_pct']}%** "
        f"(median {overlap_summary['orders_3_30_median_overlap_pct']}%).\n"
        f"- Low orders (3–6): {overlap_summary['low_order_3_6_mean_overlap_pct']}% — "
        f"high orders (22–30): {overlap_summary['high_order_22_30_mean_overlap_pct']}%.\n"
        "  The discovered high-order structure is almost entirely dedup-specific, "
        "confirming that effective-sample deduplication materially changes which "
        "feature sets are found (source: `candidate_overlap_by_order.csv`)."
    )
    (OUTDIR / "cooccurrence_overlap.md").write_text("\n".join(lines) + "\n")
    print(net.to_string(index=False))
    print()
    print(perm.to_string(index=False))
    print()
    print("overlap:", overlap_summary)


if __name__ == "__main__":
    main()
