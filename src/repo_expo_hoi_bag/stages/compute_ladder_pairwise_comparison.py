"""
compute_ladder_pairwise_comparison.py

Pairwise statistical comparison of complexity-ladder rungs, separately within
synergistic (o_min) and redundant (o_max) candidate sets.

Three analyses per BAG target × objective:
  A  – top-k per rung independently (different candidate sets per rung)
  B  – union of top-k across all rungs (same fixed candidate set for all rungs)
  C1 – all candidates that beat the covariate baseline at any rung (frontier set)

Statistical test: Wilcoxon signed-rank on per-fold median R² across the
candidate set (~15–22 paired observations = LOCO folds). Holm–Bonferroni
correction within each BAG × objective block (10 pairs from C(5,2)).

Effect size: rank-biserial correlation r_rb ∈ [−1, 1].

Outputs (to STATS_DIR):
  ladder_pairwise_stats.csv   – one row per (analysis, bag, objective, pair)
  best_rung_selection.csv     – best rung per (analysis, bag, objective)

Usage:
  export V3_OUTPUT_ROOT=outputs/variant_a
  python -m scripts.compute_ladder_pairwise_comparison
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ── config ─────────────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
CANONICAL = ROOT / "families" / "pooled_oinfo_ladder" / "canonical"
STATS_DIR = ROOT / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20          # candidates per rung for analyses A and B
C1_THRESHOLD = 0.0  # minimum delta_r2_vs_base to qualify for C1 (0 = beats baseline)

RUNGS = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
# Restrict to the bags actually evaluated via PAPER_FIG_BAGS (same convention as
# the fig2/fig3/fig4/figS generators). Unset -> the canonical three-bag default,
# so existing behaviour is unchanged.
BAGS = [
    b for b in os.environ.get("PAPER_FIG_BAGS", "combined,functional,structural").split(",") if b
] or ["combined", "functional", "structural"]
OBJECTIVES = ["o_min", "o_max"]  # o_min = synergistic, o_max = redundant


# ── helpers ────────────────────────────────────────────────────────────────────

def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation for paired Wilcoxon signed-rank test."""
    diffs = x - y
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank()
    w_plus = ranks[nonzero > 0].sum()
    w_minus = ranks[nonzero < 0].sum()
    w_total = w_plus + w_minus
    return float((w_plus - w_minus) / w_total) if w_total > 0 else 0.0


def _wilcoxon_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank p-value with fallback for degenerate cases."""
    diffs = x - y
    if np.all(diffs == 0):
        return np.nan, 1.0
    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


def _fold_medians(country_df: pd.DataFrame, candidates: set[str]) -> pd.Series:
    """Per-fold median R² across a candidate set. Returns a Series indexed by fold."""
    sub = country_df[country_df["candidate_id"].isin(candidates)]
    return sub.groupby("fold_country")["country_full_r2"].median()


def _pairwise_tests(
    fold_medians: dict[str, pd.Series],
    bag: str,
    objective: str,
    analysis: str,
) -> list[dict]:
    """All C(5,2)=10 pairwise Wilcoxon tests with Holm correction."""
    pairs = list(itertools.combinations(RUNGS, 2))
    rows_raw = []
    for rung_a, rung_b in pairs:
        s_a = fold_medians.get(rung_a, pd.Series(dtype=float))
        s_b = fold_medians.get(rung_b, pd.Series(dtype=float))
        # align on common folds
        common = s_a.index.intersection(s_b.index)
        if len(common) < 3:
            rows_raw.append({
                "rung_a": rung_a, "rung_b": rung_b,
                "n_folds": len(common),
                "median_r2_a": np.nan, "median_r2_b": np.nan,
                "delta_median_r2": np.nan,
                "wilcoxon_stat": np.nan, "p_raw": np.nan,
                "r_rb": np.nan,
            })
            continue
        a_vals = s_a[common].values
        b_vals = s_b[common].values
        stat, p = _wilcoxon_safe(a_vals, b_vals)
        rows_raw.append({
            "rung_a": rung_a, "rung_b": rung_b,
            "n_folds": len(common),
            "median_r2_a": float(np.median(a_vals)),
            "median_r2_b": float(np.median(b_vals)),
            "delta_median_r2": float(np.median(a_vals)) - float(np.median(b_vals)),
            "wilcoxon_stat": stat,
            "p_raw": p,
            "r_rb": _rank_biserial(a_vals, b_vals),
        })

    # Holm–Bonferroni correction on valid p-values
    valid_mask = [not np.isnan(r["p_raw"]) for r in rows_raw]
    valid_ps = [r["p_raw"] for r, m in zip(rows_raw, valid_mask) if m]
    if valid_ps:
        reject, p_corr, _, _ = multipletests(valid_ps, method="holm")
        idx = 0
        for r, m in zip(rows_raw, valid_mask):
            if m:
                r["p_holm"] = float(p_corr[idx])
                r["reject_holm"] = bool(reject[idx])
                idx += 1
            else:
                r["p_holm"] = np.nan
                r["reject_holm"] = False
    else:
        for r in rows_raw:
            r["p_holm"] = np.nan
            r["reject_holm"] = False

    for r in rows_raw:
        r.update({"analysis": analysis, "bag": bag, "objective": objective})
    return rows_raw


def _best_rung(fold_medians: dict[str, pd.Series]) -> dict[str, float]:
    """Return best rung by median R² across folds."""
    scores = {
        rung: float(s.median()) for rung, s in fold_medians.items() if len(s) > 0
    }
    if not scores:
        return {"best_rung": None, "best_median_r2": np.nan}
    best = max(scores, key=scores.__getitem__)
    return {"best_rung": best, "best_median_r2": scores[best], **{f"median_r2_{r}": scores.get(r, np.nan) for r in RUNGS}}


# ── main ───────────────────────────────────────────────────────────────────────

def _per_exp_path(bag: str, table: str) -> Path:
    """Path to per-experiment canonical parquet for one BAG."""
    return CANONICAL / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / f"{table}.parquet"


def main() -> None:
    # Validate all per-experiment parquets exist before starting
    for bag in BAGS:
        for table in ("metrics_global_long", "metrics_country_long"):
            p = _per_exp_path(bag, table)
            if not p.exists():
                raise FileNotFoundError(
                    f"Canonical parquet not found: {p}\n"
                    "Run main_pooled (or ensure cluster data is synced) first."
                )

    # Load all BAGs and concatenate
    mg = pd.concat(
        [pd.read_parquet(_per_exp_path(bag, "metrics_global_long")) for bag in BAGS],
        ignore_index=True,
    )
    mc = pd.concat(
        [pd.read_parquet(_per_exp_path(bag, "metrics_country_long")) for bag in BAGS],
        ignore_index=True,
    )

    # Restrict the candidate pool to interaction order <= PAPER_FIG_ORDER_MAX (same
    # convention as the fig2/fig4/figS generators). Unset -> full pool, behaviour
    # unchanged. This makes best_rung_selection (and everything that reads it: OOF,
    # residuals, fig2/fig4) honour the chosen set-size cap instead of selecting
    # candidates above it.
    _order_cap = os.environ.get("PAPER_FIG_ORDER_MAX", "").strip()
    if _order_cap:
        cap = int(_order_cap)
        mg = mg[pd.to_numeric(mg["order"], errors="coerce") <= cap].copy()
        mc = mc[pd.to_numeric(mc["order"], errors="coerce") <= cap].copy()

    all_pairwise: list[dict] = []
    all_best: list[dict] = []

    for bag in BAGS:
        mg_bag = mg[mg["bag_target"] == bag]
        mc_bag = mc[mc["bag_target"] == bag]

        for obj in OBJECTIVES:
            mg_obj = mg_bag[mg_bag["objective"] == obj]
            mc_obj = mc_bag[mc_bag["objective"] == obj]

            # ── Analysis A: top-k per rung independently ───────────────────
            fold_medians_a: dict[str, pd.Series] = {}
            for rung in RUNGS:
                mg_r = mg_obj[mg_obj["rung_id"] == rung].nlargest(TOP_K, "full_r2")
                top_ids = set(mg_r["candidate_id"])
                mc_r = mc_obj[mc_obj["rung_id"] == rung]
                fold_medians_a[rung] = _fold_medians(mc_r, top_ids)

            all_pairwise.extend(_pairwise_tests(fold_medians_a, bag, obj, "A_top_k_per_rung"))
            best_a = _best_rung(fold_medians_a)
            all_best.append({"analysis": "A_top_k_per_rung", "bag": bag, "objective": obj, **best_a})

            # ── Analysis B: union of top-k across all rungs ─────────────────
            union_ids: set[str] = set()
            for rung in RUNGS:
                top_ids = set(mg_obj[mg_obj["rung_id"] == rung].nlargest(TOP_K, "full_r2")["candidate_id"])
                union_ids |= top_ids

            fold_medians_b: dict[str, pd.Series] = {}
            for rung in RUNGS:
                mc_r = mc_obj[mc_obj["rung_id"] == rung]
                fold_medians_b[rung] = _fold_medians(mc_r, union_ids)

            all_pairwise.extend(_pairwise_tests(fold_medians_b, bag, obj, "B_union_top_k"))
            best_b = _best_rung(fold_medians_b)
            all_best.append({"analysis": "B_union_top_k", "bag": bag, "objective": obj,
                              "n_union_candidates": len(union_ids), **best_b})

            # ── Analysis C1: all candidates beating baseline at any rung ───
            frontier_ids: set[str] = set(
                mg_obj[mg_obj["delta_r2_vs_base"] > C1_THRESHOLD]["candidate_id"]
            )

            fold_medians_c1: dict[str, pd.Series] = {}
            for rung in RUNGS:
                mc_r = mc_obj[mc_obj["rung_id"] == rung]
                fold_medians_c1[rung] = _fold_medians(mc_r, frontier_ids)

            all_pairwise.extend(_pairwise_tests(fold_medians_c1, bag, obj, "C1_baseline_frontier"))
            best_c1 = _best_rung(fold_medians_c1)
            all_best.append({"analysis": "C1_baseline_frontier", "bag": bag, "objective": obj,
                              "n_frontier_candidates": len(frontier_ids), **best_c1})

    # ── save outputs ───────────────────────────────────────────────────────────
    pairwise_df = pd.DataFrame(all_pairwise)
    col_order = [
        "analysis", "bag", "objective", "rung_a", "rung_b",
        "n_folds", "median_r2_a", "median_r2_b", "delta_median_r2",
        "wilcoxon_stat", "p_raw", "p_holm", "reject_holm", "r_rb",
    ]
    pairwise_df = pairwise_df[[c for c in col_order if c in pairwise_df.columns]]
    out_pairwise = STATS_DIR / "ladder_pairwise_stats.csv"
    pairwise_df.to_csv(out_pairwise, index=False)
    print(f"Saved: {out_pairwise}  ({len(pairwise_df)} rows)")

    best_df = pd.DataFrame(all_best)
    out_best = STATS_DIR / "best_rung_selection.csv"
    best_df.to_csv(out_best, index=False)
    print(f"Saved: {out_best}  ({len(best_df)} rows)")

    # ── console summary ────────────────────────────────────────────────────────
    print("\n=== Best rung by analysis / BAG / objective ===")
    summary_cols = ["analysis", "bag", "objective", "best_rung", "best_median_r2"]
    print(best_df[[c for c in summary_cols if c in best_df.columns]].to_string(index=False))

    print("\n=== Significant adjacent-rung differences (Holm p < 0.05) ===")
    adjacent = [("ols", "xgb_tree_d1"),
                ("xgb_tree_d1", "xgb_tree_d2"), ("xgb_tree_d2", "xgb_tree_d3")]
    adj_mask = pairwise_df.apply(
        lambda r: (r["rung_a"], r["rung_b"]) in adjacent or (r["rung_b"], r["rung_a"]) in adjacent,
        axis=1,
    )
    sig = pairwise_df[adj_mask & (pairwise_df["reject_holm"] == True)]
    if len(sig):
        print(sig[["analysis", "bag", "objective", "rung_a", "rung_b",
                   "delta_median_r2", "p_holm", "r_rb"]].to_string(index=False))
    else:
        print("  None found.")


if __name__ == "__main__":
    main()
