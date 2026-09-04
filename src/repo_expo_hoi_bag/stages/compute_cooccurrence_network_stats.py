#!/usr/bin/env python3
"""Co-occurrence network connectivity stats across all active rungs.

The existing cooccurrence_network_plotting helpers build one variable-level and
one domain-level network per (bag, family, rung) but were only ever driven one
rung at a time (auto-selecting the single globally-best rung when no rung was
given). This runner explicitly loops over every active rung x family combination,
computes connectivity statistics (weighted degree, modularity, intra/inter-domain
mixing) on each, and runs an edge-permutation null to flag co-occurrence pairs
that are stronger than chance given each model's set size.

Lightweight stats -> outputs/sensitivity/cooccurrence_network/.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from oinfo_bag_ladder.cooccurrence_network import (
    build_domain_network,
    build_panel_network,
    compute_edge_permutation_null,
    compute_network_stats,
    resolve_analysis_root,
)
from scripts.exposome_domains import load_domain_map
from scripts.sensitivity_common import (
    ACTIVE_RUNGS,
    DOMAIN_CSV,
    active_rungs,
    bundle_sensitivity_root,
    bundle_variant_root,
    copy_tree_contents,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
)

FAMILIES = ("synergy", "redundancy")
TOP_K = 20
N_PERMUTATIONS = 1000


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def main() -> None:
    cfg = load_sensitivity_config()
    include_combined = os.environ.get("SENSITIVITY_INCLUDE_COMBINED", "").strip().lower() in {"1", "true", "yes"}
    rungs = active_rungs(cfg)
    n_permutations = _env_int("COOCCURRENCE_NULL_N_PERM", N_PERMUTATIONS)
    top_k = _env_int("COOCCURRENCE_TOP_K", TOP_K)

    analysis_root = resolve_analysis_root(bundle_variant_root())
    domain_map = load_domain_map(DOMAIN_CSV)

    local_root = repo_sensitivity_root(cfg) / "cooccurrence_network"
    local_root.mkdir(parents=True, exist_ok=True)
    # Granular per-network node/edge/null tables are bulk underlying data -> they
    # live in the external runtime. The release keeps the compact
    # {bag}_network_stats.csv summary alongside the figures.
    data_root = bundle_sensitivity_root(cfg) / "cooccurrence_network"
    data_root.mkdir(parents=True, exist_ok=True)

    for bag in selected_bags(cfg, include_combined=include_combined):
        stats_rows = []
        for rung in rungs:
            if rung not in ACTIVE_RUNGS:
                raise ValueError(f"Unsupported rung for cooccurrence-network: {rung!r}.")
            for family in FAMILIES:
                print(f"=== cooccurrence-network | bag={bag} rung={rung} family={family} ===")
                panel = build_panel_network(analysis_root, bag, family, rung, top_k, domain_map)
                domain = build_domain_network(panel, domain_map)

                stats_rows.append(compute_network_stats(panel, domain))

                null_df = compute_edge_permutation_null(panel, n_permutations=n_permutations)
                null_df.to_csv(
                    data_root / f"{bag}_{family}_{rung}_edge_null.csv", index=False
                )
                panel.nodes.to_csv(data_root / f"{bag}_{family}_{rung}_variable_nodes.csv", index=False)
                panel.edges.to_csv(data_root / f"{bag}_{family}_{rung}_variable_edges.csv", index=False)
                domain.nodes.to_csv(data_root / f"{bag}_{family}_{rung}_domain_nodes.csv", index=False)
                domain.edges.to_csv(data_root / f"{bag}_{family}_{rung}_domain_edges.csv", index=False)

        bag_stats = pd.concat(stats_rows, ignore_index=True)
        bag_stats.to_csv(local_root / f"{bag}_network_stats.csv", index=False)

    # Keep the summary in the external runtime so the reproduction is complete
    # (granular written directly above + summary copied here).
    copy_tree_contents(local_root, data_root)


if __name__ == "__main__":
    main()
