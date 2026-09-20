#!/usr/bin/env python3
"""Country-consistency tests: best single exposure vs best high-order model (depth-3).

The aggregate pooled-R² paired comparison (compare_models_stats.py, Table M4) prices
the country-year exposome sharing into a country-cluster bootstrap on ΔR² and -- with
only 16 (functional) / 22 (structural) country clusters -- cannot resolve a +0.01 ΔR².
That is honest but it collapses all per-country information into one diluted number.

This module asks the *consistency* question instead, with four lenses of increasing
sensitivity that ALL still respect the country cluster (the exposome is country-year):

  (1) Mixed model on per-subject loss difference, country random intercept:
          d_i = β0 + u_country + ε_i ,   d_i = loss(single)_i - loss(full)_i
      β0 > 0 == the high-order model improves the *average* subject once each
      country's mean offset is absorbed. NOTE: in every cell the between-country
      variance of d is estimated at the boundary (groupVar -> 0), i.e. the full-vs-
      single loss difference carries no systematic country-level structure for the
      random intercept to absorb; the MLE is on the boundary and the LMM Wald P is
      undefined. When groupVar=0 the LMM degenerates exactly to a one-sample t on the
      per-country mean differences, so we report THAT (cluster-respecting, df =
      n_countries-1) as the valid form of lens (1), and record the groupVar=0
      boundary outcome itself as the (informative) finding.

  (2) Hierarchical (country) bootstrap on the FRACTION of countries where the
      high-order model wins (mean per-subject loss lower). The statistic is a
      consistency proportion, not an aggregate ΔR², so it is stable under resampling
      of few clusters. Reports the win-fraction point estimate + 95% CI + the
      bootstrap mass at/above 0.5 (one-sided evidence of >chance consistency).

  (3) Precision-weighted country test. Each country contributes its mean loss
      difference weighted by sqrt(n_country) (a 2238-subject country is a far more
      precise estimate than an 85-subject one). Reports the weighted mean, a
      weighted one-sample t (df = n_countries-1) and an unweighted sign test for
      reference -- recovering the power a plain sign test discards.

  (4) OLS on per-subject d with a country-CLUSTER-ROBUST (CR) standard error on β0
      (subjects nested in country): keeps every subject for the point estimate but
      prices the within-country correlation into the SE instead of a random intercept.
      The dual of lens (1) -- where the LMM absorbs country structure and collapses,
      this leaves it in the residual and corrects the SE; both should (and do) agree.

loss(single)_i - loss(full)_i  > 0  ==>  full (high-order) model is better.
Two losses: absolute |y-ŷ| and squared (y-ŷ)². No model is refit; OOF is read from
the regenerated dedup parquets that reproduce the canonical pooled R² (Table M5).

Comparisons: best_syn vs best_single  and  best_red vs best_single, both BAGs, d3.

Outputs (additive):
  outputs/dedup/model_comparison/single_vs_best_country_consistency.csv
  outputs/dedup/model_comparison/single_vs_best_country_consistency.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from repo_expo_hoi_bag.publication_stats.paths import model_comparison_output_dir

OUTDIR = model_comparison_output_dir()
OOF = OUTDIR / "oof"
RUNG = "xgb_tree_d3"
N_BOOT = 20000
SEED = 20260304
BAGS = ["functional", "structural"]
# (high-order family, objective label) compared against best_single
HI = {"best_syn": "syn", "best_red": "red"}


def _loss(y, yhat, kind):
    e = np.asarray(y, float) - np.asarray(yhat, float)
    return np.abs(e) if kind == "abs" else e * e


def _load(fam, bag, extra=()):
    df = pd.read_parquet(OOF / fam / bag / f"oof_{RUNG}.parquet")
    return df[["row_id", "country", "y_true", "y_pred_full", *extra]].copy()


def _pair(hi_fam, bag):
    """Aligned per-subject frame: d = loss(single) - loss(full_high_order).
    diagnosis carried from the single OOF (identical subjects/row_id across families)."""
    s = _load("best_single", bag, extra=("diagnosis",))
    h = _load(hi_fam, bag)
    m = s.merge(h, on=["row_id", "country", "y_true"], suffixes=("_single", "_hi"))
    assert len(m) == len(s) == len(h), f"row_id misalignment {hi_fam}/{bag}"
    return m


def _lmm_diff(m, kind):
    """Country-random-intercept LMM on per-subject d; when its between-country
    variance is on the boundary (groupVar->0) the LMM is degenerate and we fall back
    to the equivalent one-sample t on per-country mean d (valid, df=n_countries-1)."""
    d = _loss(m["y_true"], m["y_pred_full_single"], kind) - _loss(
        m["y_true"], m["y_pred_full_hi"], kind
    )
    df = pd.DataFrame({"d": d, "country": m["country"].to_numpy()})
    res = sm.MixedLM.from_formula("d ~ 1", groups="country", data=df).fit(method="lbfgs")
    group_var = float(res.cov_re.iloc[0, 0])
    cm = df.groupby("country")["d"].mean().to_numpy()
    t, p_t = stats.ttest_1samp(cm, 0.0)
    return {
        "lmm_beta0": float(res.params["Intercept"]),
        "lmm_group_var": group_var,
        "lmm_p_raw": float(res.pvalues["Intercept"]),
        "lmm_boundary": bool(not np.isfinite(res.pvalues["Intercept"]) or group_var <= 1e-12),
        # valid cluster-respecting estimate of the same β0 (== LMM mean when groupVar=0)
        "country_mean_d": float(np.mean(cm)),
        "country_t": float(t),
        "country_t_p": float(p_t),
    }


def _ols_cluster(m, kind):
    """OLS on per-subject loss difference d_i = β0 + ε_i with subject-level residual
    variance but country-CLUSTER-ROBUST SE on β0 (subjects nested in country). This is
    the dual of the LMM: instead of absorbing country structure in a random intercept
    (which collapsed to groupVar=0), it keeps every subject and prices the within-
    country correlation into the SE. Gives a β0 P that uses all ~9-11k subjects yet
    still respects the country cluster."""
    d = _loss(m["y_true"], m["y_pred_full_single"], kind) - _loss(
        m["y_true"], m["y_pred_full_hi"], kind
    )
    X = np.ones((len(d), 1))
    res = sm.OLS(d, X).fit(cov_type="cluster",
                           cov_kwds={"groups": m["country"].to_numpy()})
    return {
        "ols_beta0": float(res.params[0]),
        "ols_cluster_se": float(res.bse[0]),
        "ols_cluster_p": float(res.pvalues[0]),
        "ols_n_clusters": int(m["country"].nunique()),
    }


def _ols_by_diagnosis(m, kind):
    """Diagnosis-stratified nested OLS on per-subject loss difference, country-cluster-
    robust SE: tests whether the high-order advantage is CONCENTRATED in the disease
    groups (deviations from the healthy norm) rather than uniform.

        d_i = β_CN + β_AD·1[AD] + β_FTD·1[FTD] + ε_i ,   CR-SE by country

    β_CN  : high-order advantage in healthy controls (the reference cell).
    β_AD  : *additional* advantage in AD relative to CN (the hypothesis term).
    β_FTD : *additional* advantage in FTD relative to CN.
    Also reports the absolute per-group advantage (β_CN + β_dx) and its CR-SE P via a
    refit with that group as the reference, so each group's stand-alone advantage is
    directly testable. Positive d = high-order better."""
    d = _loss(m["y_true"], m["y_pred_full_single"], kind) - _loss(
        m["y_true"], m["y_pred_full_hi"], kind
    )
    dx = m["diagnosis"].astype(str).to_numpy()
    groups = m["country"].to_numpy()
    levels = [g for g in ("CN", "AD", "FTD") if (dx == g).any()]

    out = {}
    # (a) interaction model: CN reference, AD/FTD = extra advantage over CN
    Xi = pd.DataFrame({lvl: (dx == lvl).astype(float) for lvl in levels})
    Xi.insert(0, "const", 1.0)
    Xi = Xi.drop(columns=["CN"])  # CN is the intercept reference
    resi = sm.OLS(d, Xi.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    names = list(Xi.columns)
    for nm, b, se, p in zip(names, resi.params, resi.bse, resi.pvalues):
        tag = "CN" if nm == "const" else nm
        out[f"adv_extra_{tag}"] = float(b)
        out[f"adv_extra_{tag}_p"] = float(p)

    # (b) per-group ABSOLUTE advantage: one CR-SE one-sample OLS within each dx group
    for lvl in levels:
        sel = dx == lvl
        if sel.sum() < 20 or pd.Series(groups[sel]).nunique() < 3:
            out[f"adv_{lvl}"] = float(np.mean(d[sel]))
            out[f"adv_{lvl}_p"] = np.nan
            out[f"adv_{lvl}_n"] = int(sel.sum())
            continue
        r = sm.OLS(d[sel], np.ones((sel.sum(), 1))).fit(
            cov_type="cluster", cov_kwds={"groups": groups[sel]}
        )
        out[f"adv_{lvl}"] = float(r.params[0])
        out[f"adv_{lvl}_p"] = float(r.pvalues[0])
        out[f"adv_{lvl}_n"] = int(sel.sum())
    return out


def _country_meandiff(m, kind):
    d = _loss(m["y_true"], m["y_pred_full_single"], kind) - _loss(
        m["y_true"], m["y_pred_full_hi"], kind
    )
    g = pd.DataFrame({"c": m["country"].to_numpy(), "d": d})
    agg = g.groupby("c")["d"].agg(["mean", "size"])
    return agg  # index country, cols mean (per-country mean diff), size (n)


def _boot_winfrac(agg, rng):
    """Hierarchical bootstrap on countries -> fraction of countries with mean d>0."""
    countries = agg.index.to_numpy()
    means = agg["mean"].to_numpy()
    nC = len(countries)
    point = float(np.mean(means > 0))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, nC, nC)
        boot[b] = np.mean(means[idx] > 0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mass_ge_half = float(np.mean(boot >= 0.5))
    return point, float(lo), float(hi), mass_ge_half


def _weighted_country(agg):
    means = agg["mean"].to_numpy()
    n = agg["size"].to_numpy().astype(float)
    w = np.sqrt(n)
    wmean = float(np.sum(w * means) / np.sum(w))
    # weighted one-sample t against 0
    wvar = np.sum(w * (means - wmean) ** 2) / np.sum(w)
    eff_n = (np.sum(w) ** 2) / np.sum(w ** 2)  # Kish effective sample size
    se = np.sqrt(wvar / eff_n)
    t = wmean / se if se > 0 else np.nan
    p_t = float(2 * stats.t.sf(abs(t), df=max(eff_n - 1, 1))) if np.isfinite(t) else np.nan
    # unweighted sign test for reference
    n_win = int(np.sum(means > 0))
    p_sign = float(stats.binomtest(n_win, len(means), 0.5).pvalue)
    return {
        "wmean_country_d": wmean,
        "weighted_t": float(t) if np.isfinite(t) else np.nan,
        "weighted_t_p": p_t,
        "eff_n_countries": float(eff_n),
        "n_win": n_win,
        "n_countries": int(len(means)),
        "sign_p": p_sign,
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for bag in BAGS:
        for hi_fam, lab in HI.items():
            m = _pair(hi_fam, bag)
            for kind in ("abs", "sq"):
                lmm = _lmm_diff(m, kind)
                ols = _ols_cluster(m, kind)
                dxo = _ols_by_diagnosis(m, kind)
                agg = _country_meandiff(m, kind)
                wf, wf_lo, wf_hi, wf_mass = _boot_winfrac(agg, rng)
                wc = _weighted_country(agg)
                rows.append(
                    {
                        "bag": bag, "objective": lab, "comparison": f"{lab}_vs_single",
                        "loss": kind, "n_subjects": int(len(m)),
                        **lmm,
                        **ols,
                        **dxo,
                        "win_fraction": wf, "win_frac_ci_lo": wf_lo,
                        "win_frac_ci_hi": wf_hi, "boot_mass_ge_half": wf_mass,
                        **wc,
                    }
                )
                print(
                    f"{bag:11s} {lab}_vs_single [{kind}]  "
                    f"country-t β={lmm['country_mean_d']:+.4f} P={lmm['country_t_p']:.3f} "
                    f"(LMM groupVar={lmm['lmm_group_var']:.1e}) | "
                    f"OLS-clrobust β={ols['ols_beta0']:+.4f} P={ols['ols_cluster_p']:.3f} | "
                    f"win {wc['n_win']}/{wc['n_countries']} "
                    f"({wf:.2f} CI[{wf_lo:.2f},{wf_hi:.2f}]) | "
                    f"wmean={wc['wmean_country_d']:+.4f} wt-P={wc['weighted_t_p']:.3f} "
                    f"sign-P={wc['sign_p']:.3f}"
                )
                if kind == "abs":
                    print(
                        f"            └ by-dx adv [abs]: "
                        f"CN β={dxo.get('adv_CN', float('nan')):+.4f} P={dxo.get('adv_CN_p', float('nan')):.3f} | "
                        f"AD β={dxo.get('adv_AD', float('nan')):+.4f} P={dxo.get('adv_AD_p', float('nan')):.3f} "
                        f"(extra vs CN P={dxo.get('adv_extra_AD_p', float('nan')):.3f}) | "
                        f"FTD β={dxo.get('adv_FTD', float('nan')):+.4f} P={dxo.get('adv_FTD_p', float('nan')):.3f} "
                        f"(extra vs CN P={dxo.get('adv_extra_FTD_p', float('nan')):.3f})"
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "single_vs_best_country_consistency.csv", index=False)

    # --- markdown (abs loss is the headline; sq in CSV) ---
    lines = [
        "# Best single exposure vs best high-order model — country-consistency tests (depth-3)\n",
        "Four country-respecting lenses on whether the depth-3 high-order model "
        "(best synergistic / best redundant) beats the best single exposure, where the "
        "aggregate country-cluster bootstrap on pooled ΔR² (Table M4) is under-powered "
        "by the small number of country clusters. Positive = high-order better. "
        "Per-subject **absolute-error** loss shown; squared-error in the CSV.\n",
        "- **Country-mean t (β0)**: the country-random-intercept LMM on per-subject "
        "loss difference collapses to the boundary in every cell (between-country "
        "variance of the difference ≈ 0 — the full-vs-single gap has no country-level "
        "structure for a random intercept to absorb), so we report the equivalent and "
        "valid one-sample t on the per-country mean difference (df = n_countries−1).",
        "- **OLS β0, country-cluster-robust SE**: per-subject loss difference with "
        "subject-level residuals but a country-clustered (CR) standard error on β0 — "
        "the dual of the LMM that keeps all subjects yet prices the within-country "
        "correlation into the SE rather than a (collapsed) random intercept.",
        "- **Win fraction**: share of countries where the high-order model has lower "
        "mean loss; 95% CI from a 20,000-draw hierarchical country bootstrap.",
        "- **Weighted country test**: per-country mean differences weighted by √n "
        "(precision), one-sample weighted t; plain sign-test P for reference.\n",
    ]
    for bag in BAGS:
        lines.append(f"\n## {bag.capitalize()} BAG\n")
        lines.append("| Comparison | Country-mean β0 (t P) | OLS β0 (CR-SE P) | Win frac [95% CI] | Weighted mean (wt-t P) | Sign test |")
        lines.append("|---|---|---|---|---|---|")
        for hi_fam, lab in HI.items():
            r = out[(out.bag == bag) & (out.objective == lab) & (out.loss == "abs")].iloc[0]
            lines.append(
                f"| {lab} vs single | {r.country_mean_d:+.4f} ({r.country_t_p:.3f}) | "
                f"{r.ols_beta0:+.4f} ({r.ols_cluster_p:.3f}) | "
                f"{r.win_fraction:.2f} [{r.win_frac_ci_lo:.2f}, {r.win_frac_ci_hi:.2f}] | "
                f"{r.wmean_country_d:+.4f} ({r.weighted_t_p:.3f}) | "
                f"{int(r.n_win)}/{int(r.n_countries)} (P={r.sign_p:.3f}) |"
            )

    # --- diagnosis-stratified nested OLS (the disease-deviation hypothesis) ---
    lines.append("\n## Diagnosis-stratified high-order advantage (nested OLS, CR-SE by country)\n")
    lines.append(
        "Tests whether any high-order advantage is **concentrated in the disease groups** "
        "(deviations from the healthy norm) rather than uniform. `adv` = high-order minus "
        "single per-subject absolute-error advantage within each diagnosis (positive = "
        "high-order better), with a country-cluster-robust SE; `extra vs CN` is the "
        "interaction term (AD/FTD advantage over the CN reference).\n"
    )
    lines.append("| BAG | Comparison | adv CN (P) | adv AD (P) [extra vs CN P] | adv FTD (P) [extra vs CN P] |")
    lines.append("|---|---|---|---|---|")
    for bag in BAGS:
        for hi_fam, lab in HI.items():
            r = out[(out.bag == bag) & (out.objective == lab) & (out.loss == "abs")].iloc[0]
            def _cell(g):
                adv = r.get(f"adv_{g}", float("nan"))
                p = r.get(f"adv_{g}_p", float("nan"))
                ex = r.get(f"adv_extra_{g}_p", float("nan"))
                base = f"{adv:+.4f} ({p:.3f})"
                return base if g == "CN" else f"{base} [{ex:.3f}]"
            lines.append(
                f"| {bag} | {lab} vs single | {_cell('CN')} | {_cell('AD')} | {_cell('FTD')} |"
            )

    (OUTDIR / "single_vs_best_country_consistency.md").write_text("\n".join(lines) + "\n")
    print("\nWrote single_vs_best_country_consistency.{csv,md}")


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
