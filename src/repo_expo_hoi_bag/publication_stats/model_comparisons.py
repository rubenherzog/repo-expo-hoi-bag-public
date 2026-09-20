#!/usr/bin/env python
"""Paired statistical comparison of LOCO "best models" from per-subject OOF predictions.

Motivation
----------
Every model's reported R² is an *aggregate* over the pooled out-of-fold (OOF)
predictions of the SAME leave-one-country-out (LOCO) procedure. Aggregate R² values
cannot be compared directly (one number per model), but the per-subject predictions
*can*: for each subject i we have predictions from model A and model B produced under
the identical fold/seed/baseline. The natural, maximally powered comparison is the
**paired difference of per-subject loss**:

    d_i = loss(y_i, yhatA_i) - loss(y_i, yhatB_i)

with loss in {absolute error |y-yhat|, squared error (y-yhat)^2}. Squared error ties
directly to R² (R² = 1 - SSE/SST), so the mean of the squared-error difference is the
ΔR² numerator; absolute error is the robust counterpart.

Dependence
----------
Two nested clusters break naive independence: (1) LOCO folds are *countries* (not
independent — they are produced by models sharing training countries), and (2) every
subject from the same country-year shares the identical exposome signature (~70x
pseudo-replication that the dedup step addressed for O-information, but the predictive
sample is still individual-level). We therefore report THREE tests per comparison:

  primary    : country cluster bootstrap on ΔR² (resample whole countries with
               replacement; CI95 + two-sided p from the bootstrap distribution of the
               loss-difference / ΔR²). Respects both the fold non-independence and the
               within-country exposome sharing — the country is the cluster.
  sensitivity: subject-level Wilcoxon signed-rank on d_i (maximal power, but treats
               subjects as independent -> anti-conservative; reported as an upper bound).
  sensitivity: per-country-mean Wilcoxon (average d_i within country, then Wilcoxon
               across the ~16-22 country means -> conservative, fully respects clusters
               but low power).

If all three agree, the conclusion is robust to the dependence assumption.

Inputs
------
A manifest of pairwise comparisons (model A vs model B), each pointing at two OOF
parquets that share a `row_id` column (and matching `y_true`). The default manifest is
auto-built from outputs/dedup/model_comparison/oof/ once regen_oof_models.py has run.

Outputs (additive, nothing overwritten outside this dir)
  outputs/dedup/model_comparison/comparison_results.csv   -- one row per (comparison, loss)
  outputs/dedup/model_comparison/comparison_results.md    -- human-readable digest

Run
  python -m repo_expo_hoi_bag.publication_stats.model_comparisons \
      [--n-boot 10000] [--seed 20260624]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from repo_expo_hoi_bag.publication_stats.paths import model_comparison_output_dir

HERE = model_comparison_output_dir()
OOF_ROOT = HERE / "oof"
DEFAULT_NBOOT = 10000
DEFAULT_SEED = 20260624


# ---------------------------------------------------------------------------
# core statistics
# ---------------------------------------------------------------------------
def _r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if sst <= 0:
        return np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - sse / sst


def _loss(y_true, y_pred, kind: str) -> np.ndarray:
    err = y_true - y_pred
    if kind == "abs":
        return np.abs(err)
    if kind == "sq":
        return err ** 2
    raise ValueError(kind)


def country_cluster_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    country: np.ndarray,
    loss_kind: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Cluster bootstrap by country on (i) the mean paired loss difference and
    (ii) ΔR² = R²(A) - R²(B). Resamples whole countries with replacement.

    The mean-loss-difference statistic is the principled paired quantity; the ΔR²
    statistic is reported because it is the metric the paper headlines. Both are
    bootstrapped on the *same* country resamples so their CIs are mutually consistent.
    """
    countries = np.asarray(country)
    uniq = np.unique(countries)
    idx_by_c = {c: np.where(countries == c)[0] for c in uniq}

    # point estimates
    d = _loss(y_true, pred_a, loss_kind) - _loss(y_true, pred_b, loss_kind)
    point_meandiff = float(np.mean(d))
    point_dr2 = _r2_pooled(y_true, pred_a) - _r2_pooled(y_true, pred_b)

    boot_meandiff = np.empty(n_boot)
    boot_dr2 = np.empty(n_boot)
    n_c = len(uniq)
    for b in range(n_boot):
        draw = rng.choice(uniq, size=n_c, replace=True)
        rows = np.concatenate([idx_by_c[c] for c in draw])
        yt = y_true[rows]
        pa = pred_a[rows]
        pb = pred_b[rows]
        boot_meandiff[b] = np.mean(_loss(yt, pa, loss_kind) - _loss(yt, pb, loss_kind))
        boot_dr2[b] = _r2_pooled(yt, pa) - _r2_pooled(yt, pb)

    def _ci_p(boot, point):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # two-sided bootstrap p: 2 x min(mass below 0, mass above 0), clamped
        p_below = float(np.mean(boot <= 0))
        p_above = float(np.mean(boot >= 0))
        p = 2.0 * min(p_below, p_above)
        p = min(1.0, max(p, 1.0 / n_boot))
        return float(lo), float(hi), p

    md_lo, md_hi, md_p = _ci_p(boot_meandiff, point_meandiff)
    r2_lo, r2_hi, r2_p = _ci_p(boot_dr2, point_dr2)
    return dict(
        meandiff=point_meandiff, meandiff_ci_lo=md_lo, meandiff_ci_hi=md_hi, meandiff_p=md_p,
        delta_r2=point_dr2, delta_r2_ci_lo=r2_lo, delta_r2_ci_hi=r2_hi, delta_r2_p=r2_p,
    )


def subject_wilcoxon(y_true, pred_a, pred_b, loss_kind: str) -> dict:
    d = _loss(y_true, pred_a, loss_kind) - _loss(y_true, pred_b, loss_kind)
    nz = d[d != 0]
    if len(nz) < 10:
        return dict(stat=np.nan, p=np.nan, n=int(len(nz)), median_d=float(np.median(d)))
    stat, p = stats.wilcoxon(nz, alternative="two-sided")
    return dict(stat=float(stat), p=float(p), n=int(len(nz)), median_d=float(np.median(d)))


def country_mean_wilcoxon(y_true, pred_a, pred_b, country, loss_kind: str) -> dict:
    df = pd.DataFrame({
        "c": country,
        "d": _loss(y_true, pred_a, loss_kind) - _loss(y_true, pred_b, loss_kind),
    })
    means = df.groupby("c")["d"].mean().values
    nz = means[means != 0]
    if len(nz) < 6:
        return dict(stat=np.nan, p=np.nan, n_countries=int(len(means)),
                    median_country_d=float(np.median(means)))
    stat, p = stats.wilcoxon(nz, alternative="two-sided")
    return dict(stat=float(stat), p=float(p), n_countries=int(len(means)),
                median_country_d=float(np.median(means)))


# ---------------------------------------------------------------------------
# pair loading / alignment
# ---------------------------------------------------------------------------
def load_pair(path_a: Path, path_b: Path, pred_col_a="y_pred_full", pred_col_b="y_pred_full"):
    """Align two OOF parquets on row_id; return aligned arrays + country.

    Asserts y_true matches across the two files (same subjects, same target).
    For a model-vs-baseline comparison pass pred_col_b='y_pred_base' on the SAME file.
    """
    a = pd.read_parquet(path_a)
    b = pd.read_parquet(path_b)
    m = a[["row_id", "country", "y_true", pred_col_a]].merge(
        b[["row_id", "y_true", pred_col_b]], on="row_id", suffixes=("_a", "_b"),
    )
    ok = (
        np.isfinite(m["y_true_a"]) & np.isfinite(m["y_true_b"])
        & np.isfinite(m[f"{pred_col_a}_a" if pred_col_a == pred_col_b else pred_col_a])
    )
    # column names after merge depend on collision; normalize:
    ycol_a = "y_true_a"
    yt = m[ycol_a].to_numpy(float)
    # predicted columns: if both 'y_pred_full', suffixes apply
    pa_name = f"{pred_col_a}_a" if pred_col_a == pred_col_b else pred_col_a
    pb_name = f"{pred_col_b}_b" if pred_col_a == pred_col_b else pred_col_b
    pa = m[pa_name].to_numpy(float)
    pb = m[pb_name].to_numpy(float)
    ctry = m["country"].to_numpy()
    if not np.allclose(m["y_true_a"], m["y_true_b"], equal_nan=True):
        raise ValueError(f"y_true mismatch between {path_a} and {path_b} — not the same subjects/target")
    mask = np.isfinite(yt) & np.isfinite(pa) & np.isfinite(pb)
    return yt[mask], pa[mask], pb[mask], ctry[mask]


def compare_one(name, path_a, path_b, pcol_a, pcol_b, country_arr_label,
                n_boot, rng) -> list[dict]:
    yt, pa, pb, ctry = load_pair(Path(path_a), Path(path_b), pcol_a, pcol_b)
    out = []
    for loss_kind in ("abs", "sq"):
        boot = country_cluster_bootstrap(yt, pa, pb, ctry, loss_kind, n_boot, rng)
        subj = subject_wilcoxon(yt, pa, pb, loss_kind)
        cmean = country_mean_wilcoxon(yt, pa, pb, ctry, loss_kind)
        out.append(dict(
            comparison=name, loss=loss_kind, n_subjects=int(len(yt)),
            n_countries=int(len(np.unique(ctry))),
            r2_a=_r2_pooled(yt, pa), r2_b=_r2_pooled(yt, pb),
            delta_r2=boot["delta_r2"], delta_r2_ci_lo=boot["delta_r2_ci_lo"],
            delta_r2_ci_hi=boot["delta_r2_ci_hi"],
            boot_p_meandiff=boot["meandiff_p"], boot_p_delta_r2=boot["delta_r2_p"],
            mean_loss_diff=boot["meandiff"],
            wilcoxon_subject_p=subj["p"], wilcoxon_subject_median_d=subj["median_d"],
            wilcoxon_country_p=cmean["p"], wilcoxon_country_median_d=cmean["median_country_d"],
        ))
    return out


# ---------------------------------------------------------------------------
# manifest construction
# ---------------------------------------------------------------------------
def autobuild_manifest() -> list[dict]:
    """Build the standard set of comparisons from whatever OOF parquets exist.

    Each entry: name, path_a, path_b, pcol_a, pcol_b. A model-vs-baseline entry uses
    the same file for A and B with pcol_b='y_pred_base'.
    """
    comps: list[dict] = []
    if not OOF_ROOT.exists():
        return comps
    # discover available (family, bag, rung)
    avail = {}
    for p in OOF_ROOT.glob("*/*/oof_*.parquet"):
        fam = p.parent.parent.name
        bag = p.parent.name
        rung = p.stem.replace("oof_", "")
        avail[(fam, bag, rung)] = p

    bags = sorted({b for (_, b, _) in avail})
    rungs_order = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]

    for bag in bags:
        # 1) syn vs red, each rung (headline "0.479 vs 0.468")
        for rung in rungs_order:
            a = avail.get(("best_syn", bag, rung))
            b = avail.get(("best_red", bag, rung))
            if a and b:
                comps.append(dict(name=f"{bag}|syn_vs_red|{rung}",
                                  path_a=str(a), path_b=str(b),
                                  pcol_a="y_pred_full", pcol_b="y_pred_full"))
        # 2) each model vs its own baseline (exposome contribution)
        for fam in ("best_syn", "best_red", "best_single"):
            for rung in rungs_order:
                a = avail.get((fam, bag, rung))
                if a:
                    comps.append(dict(name=f"{bag}|{fam}_vs_baseline|{rung}",
                                      path_a=str(a), path_b=str(a),
                                      pcol_a="y_pred_full", pcol_b="y_pred_base"))
        # 3) across rungs within syn and within red (e.g. d3 vs ols)
        for fam in ("best_syn", "best_red"):
            present = [r for r in rungs_order if (fam, bag, r) in avail]
            for r_lo, r_hi in itertools.combinations(present, 2):
                comps.append(dict(name=f"{bag}|{fam}|{r_hi}_vs_{r_lo}",
                                  path_a=str(avail[(fam, bag, r_hi)]),
                                  path_b=str(avail[(fam, bag, r_lo)]),
                                  pcol_a="y_pred_full", pcol_b="y_pred_full"))
        # 4) syn vs best_single, syn vs within-domain reps, syn vs whole_pca (at best rung d3)
        for other in ("best_single", "within_domain_single", "within_domain_pc1", "whole_pca"):
            a = avail.get(("best_syn", bag, "xgb_tree_d3"))
            b = avail.get((other, bag, "xgb_tree_d3"))
            if a and b:
                comps.append(dict(name=f"{bag}|syn_vs_{other}|xgb_tree_d3",
                                  path_a=str(a), path_b=str(b),
                                  pcol_a="y_pred_full", pcol_b="y_pred_full"))
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="", help="JSON list of comparisons; else auto-build")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_NBOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.manifest:
        comps = json.loads(Path(args.manifest).read_text())
    else:
        comps = autobuild_manifest()

    if not comps:
        print("No comparisons found. Run regen_oof_models.py first (need outputs/dedup/model_comparison/oof/).")
        return

    rows: list[dict] = []
    for c in comps:
        try:
            rows.extend(compare_one(
                c["name"], c["path_a"], c["path_b"],
                c.get("pcol_a", "y_pred_full"), c.get("pcol_b", "y_pred_full"),
                None, args.n_boot, rng,
            ))
            print(f"[ok] {c['name']}")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {c['name']}: {e}")

    df = pd.DataFrame(rows)
    out_csv = HERE / "comparison_results.csv"
    df.to_csv(out_csv, index=False)

    # markdown digest (squared-error loss = the R²-linked test, primary)
    lines = ["# Paired model comparisons — per-subject OOF, country cluster bootstrap",
             "",
             f"n_boot={args.n_boot}, seed={args.seed}. Primary test: country cluster "
             "bootstrap on ΔR² (squared-error loss). Sensitivity: subject-level and "
             "per-country-mean Wilcoxon. P<0.05 in all three => robust.",
             "",
             "| comparison | loss | R²(A) | R²(B) | ΔR² | ΔR² 95% CI | boot P | Wilcoxon subj P | Wilcoxon ctry P |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['loss']} | {r['r2_a']:.4f} | {r['r2_b']:.4f} | "
            f"{r['delta_r2']:+.4f} | [{r['delta_r2_ci_lo']:+.4f}, {r['delta_r2_ci_hi']:+.4f}] | "
            f"{r['boot_p_delta_r2']:.4g} | {r['wilcoxon_subject_p']:.3g} | {r['wilcoxon_country_p']:.3g} |"
        )
    (HERE / "comparison_results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(df)} rows -> {out_csv}")
    print(df.to_string())


if __name__ == "__main__":
    main()
