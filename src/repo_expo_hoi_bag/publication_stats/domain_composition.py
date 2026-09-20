#!/usr/bin/env python3
"""Domain composition of the top-20 candidates per arm (dedup-complete pool).

Regenerates the numbers behind the manuscript's domain-composition paragraph and
Fig. 3b. Additive aggregation over the EXISTING dedup candidate pool
(`metrics_global_long.parquet`); no model is refit.

Why this file exists: the composition percentages previously written into the
manuscript came from
`$REPRO_DATA_ROOT/paper_exports/paper_figures/fig3_diversity_v2_recipe_top20.csv`,
a stale export of the CANONICAL pool that was additionally uncapped (set sizes to
49), mixed rungs across arms (d1/d2/d3) and included the `combined` BAG, which the
paper does not use. Nothing regenerated those numbers from the dedup pool. This
stage does, under exactly the selection the paper reports.

Selection (identical to `cooccurrence_overlap_stats.py`, so the two paragraphs
describe the same candidates):
  - dedup bundle, path-only synergy definition
  - rung xgb_tree_d3, order <= 30
  - top 20 candidates per (BAG, objective) by full_r2

Two shares are reported per domain:
  - `pct_features`  : pooled share of feature slots across the 20 candidates.
                      This is the quantity quoted in the manuscript.
  - `pct_models`    : fraction of the 20 candidates containing the domain at all
                      (the node prevalence used by the co-occurrence network).

Outputs (additive, checkout):
  outputs/dedup/model_comparison/domain_composition_top20.csv
  outputs/dedup/model_comparison/domain_composition.md
"""
from __future__ import annotations

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
# Effective-sample (dedup) pool: the paper's composition claims are made on it.
DEDUP = required_dedup_root()
POOL = DEDUP / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "metrics_global_long.parquet"
DOMAIN_CSV = REPO / "data" / "metadata" / "exposome_feature_domains.csv"

BAGS = ["structural", "functional"]          # paper order; no `combined` row exists in Fig. 3
ARMS = [("o_max", "redundancy"), ("o_min", "synergy")]
RUNG = "xgb_tree_d3"
TOP_K = 20
MAX_ORDER = 30


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


def main() -> None:
    dmap = _domain_map()
    pool = pd.read_parquet(POOL)

    rows = []
    for bag in BAGS:
        for objective, arm in ARMS:
            top = _top_candidates(pool, bag, objective)
            n_models = len(top)
            feat_counts: dict[str, int] = {}
            model_counts: dict[str, int] = {}
            for _, r in top.iterrows():
                feats = [f.strip() for f in str(r.predictors_identity).split("|") if f.strip()]
                doms = [dmap.get(f, "UNKNOWN") for f in feats]
                for d in doms:
                    feat_counts[d] = feat_counts.get(d, 0) + 1
                for d in set(doms):
                    model_counts[d] = model_counts.get(d, 0) + 1
            total = sum(feat_counts.values())
            for d, c in sorted(feat_counts.items(), key=lambda kv: -kv[1]):
                rows.append(
                    {
                        "bag": bag,
                        "arm": arm,
                        "objective": objective,
                        "rung_id": RUNG,
                        "top_k": n_models,
                        "max_order": MAX_ORDER,
                        "domain": d,
                        "n_features": c,
                        "pct_features": round(100.0 * c / total, 3),
                        "n_models_with_domain": model_counts[d],
                        "pct_models": round(100.0 * model_counts[d] / n_models, 3),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "domain_composition_top20.csv", index=False)

    lines = ["# Domain composition of the top-20 candidates per arm (dedup-complete)\n"]
    lines.append(
        f"Dedup pool, rung `{RUNG}`, set size <= {MAX_ORDER}, top {TOP_K} candidates per "
        "(BAG, arm) by LOCO R^2. `pct_features` is the pooled share of feature slots and is "
        "the quantity quoted in the manuscript; `pct_models` is the fraction of the 20 "
        "candidates containing the domain (the co-occurrence-network node prevalence).\n"
    )
    for bag in BAGS:
        for _, arm in ARMS:
            sub = out[(out.bag == bag) & (out.arm == arm)]
            lines.append(f"\n## {bag} BAG, {arm} arm ({len(sub)} domains)\n")
            lines.append("| Domain | Features | % features | Models | % models |")
            lines.append("|---|---|---|---|---|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r.domain} | {r.n_features} | {r.pct_features:.3f} | "
                    f"{r.n_models_with_domain} | {r.pct_models:.3f} |"
                )
    (OUTDIR / "domain_composition.md").write_text("\n".join(lines) + "\n")

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
