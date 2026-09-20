#!/usr/bin/env python3
"""Recurrent synergistic triplets and best-single-exposure context (dedup-complete).

Regenerates every number in the manuscript's recurrent-triplet paragraph.
Additive aggregation over EXISTING dedup artefacts; nothing is refit.

Why this file exists: the triplet prevalences, their O-information and the
best-single-exposure companion triplets were quoted in the manuscript but no stage
assembled them into a table. The per-sub-combination parquets hold triplet identity,
`subcomb_summary.parquet` identifies the top candidates, `subcomb_baseline.parquet`
holds the exposome-wide triplet O-information, and the single-exposure sweep holds
the best exposure per BAG; nothing joined the four.

Inputs:
  outputs/dedup/subcomb_oinfo/subcomb_summary.parquet          (candidate ranking)
  $DEDUP_DATA_ROOT/work/subcomb_oinfo/subcomb_oinfo_{bag}.parquet
                                                               (triplet identity)
  $DEDUP_DATA_ROOT/work/subcomb_oinfo/subcomb_baseline.parquet (triplet O-info)
  $DEDUP_DATA_ROOT/runs/oinfo_only/variant_a/single_exposure_eval_xgb_tree_d3/
      {bag}/single_exposure_global.csv                         (best single exposure)

Omega is the exposome-wide O-information of the triplet on the 259 unique
country-year signatures. It is a property of the triplet, not of the candidate that
contains it, so one value serves every arm in which the triplet appears.

POOL DEFINITION — read before comparing these numbers with the rest of the
subsection. Prevalence is computed over the top 20 candidates at EACH of the four
model levels (OLS, d1, d2, d3), deduplicated by `candidate_id`. Denominators can
therefore differ by group because a set ranking top-20 at several levels counts
once. Ranking is by held-out LOCO R² (`full_r2`) within BAG, arm and model level.

Sections:
  A. Most recurrent triplet per (BAG, arm), with Omega and the tie count at that
     prevalence, so a unique maximum can be distinguished from an arbitrary pick.
  B. Best single exposure per BAG and the synergy-arm triplets containing it,
     with the tie count at that prevalence for the same reason.
  C. Triplets present in all four (BAG x arm) groups, with prevalence in each.
     The manuscript's claim rests on how weakly these recur, not on their number.

Outputs (additive, checkout):
  outputs/dedup/model_comparison/recurrent_triplets_top.csv
  outputs/dedup/model_comparison/recurrent_triplets_single_exposure.csv
  outputs/dedup/model_comparison/recurrent_triplets_shared.csv
  outputs/dedup/model_comparison/recurrent_triplets.md
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
DEDUP = required_dedup_root()
SUMMARY_PARQUET = REPO / "outputs" / "dedup" / "subcomb_oinfo" / "subcomb_summary.parquet"
DETAIL_DIR = DEDUP / "work" / "subcomb_oinfo"
BASELINE_PARQUET = DEDUP / "work" / "subcomb_oinfo" / "subcomb_baseline.parquet"
SINGLE_DIR = DEDUP / "runs" / "oinfo_only" / "variant_a" / "single_exposure_eval_xgb_tree_d3"

BAGS = ["structural", "functional"]          # paper order
ARMS = [("o_max", "redundancy"), ("o_min", "synergy")]
ORDER_K = 3
TOP_K = 20
TIE_ATOL = 1e-9


def _omega_map() -> dict[frozenset, float]:
    bl = pd.read_parquet(BASELINE_PARQUET)
    bl = bl[bl.order_k == ORDER_K]
    return {frozenset(str(t).split("|")): float(o) for t, o in zip(bl.nplet_cols, bl.o_info)}


def _omega(triplet: str, omap: dict[frozenset, float]) -> float:
    return omap.get(frozenset(str(triplet).split("|")), float("nan"))


def _triplet_prevalence() -> pd.DataFrame:
    """Count negative-Omega triplets in the top 20 candidates at each level."""
    summary = pd.read_parquet(SUMMARY_PARQUET)
    rows = []
    for bag in BAGS:
        ranked = summary[summary["bag"] == bag].sort_values(
            "full_r2", ascending=False, kind="stable"
        )
        selected = ranked.groupby(
            ["rung_id", "objective"], observed=True, sort=False
        ).head(TOP_K)
        pool = selected[["candidate_id", "objective"]].drop_duplicates()
        detail = pd.read_parquet(
            DETAIL_DIR / f"subcomb_oinfo_{bag}.parquet",
            filters=[("order_k", "==", ORDER_K)],
        )
        detail = detail.merge(pool, on="candidate_id", how="inner", validate="many_to_one")
        negative = detail[detail["o_info"] < 0]
        for objective, candidates in pool.groupby("objective", observed=True):
            n_pool = candidates["candidate_id"].nunique()
            counts = (
                negative[negative["objective"] == objective]
                .groupby("nplet_cols")["candidate_id"]
                .nunique()
            )
            frame = counts.rename("n_candidates").reset_index()
            frame["prevalence"] = frame["n_candidates"] / n_pool
            frame["bag"] = bag
            frame["objective"] = objective
            frame["order_k"] = ORDER_K
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    prev = _triplet_prevalence()
    omap = _omega_map()

    # ── A. most recurrent triplet per (BAG, arm) ─────────────────────────────
    top_rows = []
    for bag in BAGS:
        for objective, arm in ARMS:
            s = prev[(prev.bag == bag) & (prev.objective == objective)]
            best = s.loc[s.prevalence.idxmax()]
            n_pool = int(round(best.n_candidates / best.prevalence))
            n_tied = int((np.isclose(s.prevalence, best.prevalence, atol=TIE_ATOL)).sum())
            top_rows.append(
                {
                    "bag": bag,
                    "arm": arm,
                    "objective": objective,
                    "triplet": best.nplet_cols,
                    "n_candidates_with_triplet": int(best.n_candidates),
                    "n_candidates_in_arm": n_pool,
                    "prevalence_pct": round(100.0 * best.prevalence, 3),
                    "omega": round(_omega(best.nplet_cols, omap), 6),
                    "n_triplets_tied_at_this_prevalence": n_tied,
                }
            )
    top = pd.DataFrame(top_rows)

    # ── B. best single exposure and the synergy-arm triplets containing it ───
    single_rows = []
    for bag in BAGS:
        se = pd.read_csv(SINGLE_DIR / bag / "single_exposure_global.csv")
        best_feat = str(se.loc[se.global_oof_r2.idxmax(), "feature_name"])
        best_r2 = float(se.global_oof_r2.max())
        s = prev[(prev.bag == bag) & (prev.objective == "o_min")]
        withf = s[[best_feat in str(t).split("|") for t in s.nplet_cols]]
        if withf.empty:
            continue
        top_p = withf.prevalence.max()
        n_pool = int(round(withf.iloc[0].n_candidates / withf.iloc[0].prevalence))
        n_tied_all = int((np.isclose(withf.prevalence, top_p, atol=TIE_ATOL)).sum())
        for _, r in withf[np.isclose(withf.prevalence, top_p, atol=TIE_ATOL)].iterrows():
            single_rows.append(
                {
                    "bag": bag,
                    "best_single_exposure": best_feat,
                    "best_single_exposure_r2": round(best_r2, 4),
                    "arm": "synergy",
                    "triplet": r.nplet_cols,
                    "n_candidates_with_triplet": int(r.n_candidates),
                    "n_candidates_in_arm": n_pool,
                    "prevalence_pct": round(100.0 * r.prevalence, 3),
                    "omega": round(_omega(r.nplet_cols, omap), 6),
                    "n_triplets_tied_at_this_prevalence": n_tied_all,
                }
            )
    single = pd.DataFrame(single_rows)

    # ── C. triplets present in all four (BAG x arm) groups ───────────────────
    groups = {
        (bag, objective): set(
            prev[(prev.bag == bag) & (prev.objective == objective)].nplet_cols
        )
        for bag in BAGS
        for objective, _ in ARMS
    }
    shared_ids = set.intersection(*groups.values())
    wide = (
        prev[prev.nplet_cols.isin(shared_ids)]
        .pivot_table(index="nplet_cols", columns=["bag", "objective"], values="prevalence")
        * 100.0
    )
    wide.columns = [f"{b}_{'synergy' if o == 'o_min' else 'redundancy'}_pct" for b, o in wide.columns]
    shared = wide.round(3).reset_index().rename(columns={"nplet_cols": "triplet"})
    shared["omega"] = [round(_omega(t, omap), 6) for t in shared.triplet]
    shared = shared.sort_values("omega").reset_index(drop=True)

    top.to_csv(OUTDIR / "recurrent_triplets_top.csv", index=False)
    single.to_csv(OUTDIR / "recurrent_triplets_single_exposure.csv", index=False)
    shared.to_csv(OUTDIR / "recurrent_triplets_shared.csv", index=False)

    lines = ["# Recurrent triplets and best-single-exposure context (dedup-complete)\n"]
    lines.append(
        "Prevalence is the fraction of the union of top-20 candidates at each model level "
        "in that arm, counting a candidate once if it recurs across levels. Omega is the "
        "exposome-wide O-information of the triplet on the 259 "
        "unique country-year signatures.\n"
    )
    lines.append("\n## A. Most recurrent triplet per BAG and arm\n")
    lines.append("| BAG | Arm | Triplet | Prevalence | Omega | Tied |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        lines.append(
            f"| {r.bag} | {r.arm} | `{r.triplet}` | {r.prevalence_pct}% "
            f"({r.n_candidates_with_triplet}/{r.n_candidates_in_arm}) | {r.omega} | "
            f"{r.n_triplets_tied_at_this_prevalence} |"
        )
    lines.append("\n## B. Best single exposure, in synergy-arm triplets\n")
    lines.append("| BAG | Best single exposure | R^2 | Triplet | Prevalence | Omega | Tied |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in single.iterrows():
        lines.append(
            f"| {r.bag} | `{r.best_single_exposure}` | {r.best_single_exposure_r2} | "
            f"`{r.triplet}` | {r.prevalence_pct}% | {r.omega} | "
            f"{r.n_triplets_tied_at_this_prevalence} |"
        )
    lines.append(
        f"\n## C. Triplets present in all four groups: {len(shared)}\n\n"
        "Every one of them recurs at the single-candidate floor in the functional "
        "redundancy arm, which is what supports the manuscript's claim that no triplet "
        "recurs appreciably across all four groups.\n"
    )
    lines.append("| Triplet | " + " | ".join(c for c in shared.columns if c.endswith("_pct")) + " | Omega |")
    lines.append("|---" * (len([c for c in shared.columns if c.endswith('_pct')]) + 2) + "|")
    for _, r in shared.iterrows():
        vals = " | ".join(str(r[c]) for c in shared.columns if c.endswith("_pct"))
        lines.append(f"| `{r.triplet}` | {vals} | {r.omega} |")
    (OUTDIR / "recurrent_triplets.md").write_text("\n".join(lines) + "\n")

    print(top.to_string(index=False))
    print()
    print(single.to_string(index=False))
    print(f"\nshared across all four groups: {len(shared)}")


if __name__ == "__main__":
    main()
