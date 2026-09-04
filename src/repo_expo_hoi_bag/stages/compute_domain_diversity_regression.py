"""
Domain diversity regression — Table 2 of the exposome synergy paper.

For each of the 480 models (per BAG), computes:
  - Shannon H over domain composition
  - n_unique_domains
  - frac_max_domain (max domain share)
  - is_synergistic (objective == o_min)

Then fits: lm(R² ~ shannon_H * is_synergistic + order)
           lm(f² ~ shannon_H * is_synergistic + order)

Also runs the air-pollution failure regression (Tier 2.4):
  lm(R² ~ frac_air_pollution + order + score) within o_min only

Outputs:
  - Console: regression tables
  - CSV: {ROOT}/stats/domain_diversity_regression.csv
"""

from __future__ import annotations
import os
import pathlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scripts.exposome_domains import load_domain_map, unweighted_domain_stats

# Greedy candidates overlap heavily (a single triplet recurs across many sets) and
# every R² is evaluated on the same subjects/countries, so classical OLS standard
# errors that assume i.i.d. residuals are anti-conservative. Inference for the
# diversity slopes therefore comes from (a) order-clustered robust SE and (b) a
# permutation null on the diversity×objective interaction term that shuffles the
# synergistic/redundant label WITHIN each interaction order (preserving the
# top-20-per-order-per-objective design balance). Point estimates are unchanged.
_PERM_N = int(os.environ.get("DIVERSITY_PERM_N", "10000"))
_PERM_SEED = int(os.environ.get("DIVERSITY_PERM_SEED", "20260304"))


def _perm_interaction_p(df: pd.DataFrame, response: str, diversity_term: str,
                        n_perm: int = _PERM_N, seed: int = _PERM_SEED) -> dict:
    """Two-sided permutation P for the {diversity_term}:is_synergistic interaction.

    The null shuffles ``is_synergistic`` within ``order`` strata, refits the same
    OLS, and compares the permuted interaction coefficient to the observed one. This
    makes no i.i.d. assumption over the overlapping candidates; it assumes only that
    the synergistic/redundant label is exchangeable within an order under the null.
    """
    formula = f"{response} ~ {diversity_term} * is_synergistic + order"
    inter = f"{diversity_term}:is_synergistic"
    work = df[[response, diversity_term, "is_synergistic", "order"]].dropna().copy()
    obs_beta = smf.ols(formula, data=work).fit().params[inter]
    rng = np.random.default_rng(seed)
    order_groups = [idx.to_numpy() for _, idx in work.groupby("order").groups.items()]
    null_betas = np.empty(n_perm, dtype=float)
    base = work.copy()
    for i in range(n_perm):
        permuted = base["is_synergistic"].to_numpy().copy()
        for pos in order_groups:
            vals = permuted[pos]
            rng.shuffle(vals)
            permuted[pos] = vals
        base["is_synergistic"] = permuted
        null_betas[i] = smf.ols(formula, data=base).fit().params[inter]
    # two-sided empirical P, +1 smoothing
    p = (1 + int(np.sum(np.abs(null_betas) >= abs(obs_beta)))) / (n_perm + 1)
    return {"obs_beta": float(obs_beta), "perm_p": float(p), "n_perm": int(n_perm)}

# ── paths ─────────────────────────────────────────────────────────────
ROOT = pathlib.Path(os.environ.get(
    "V3_OUTPUT_ROOT",
    ""
    "outputs/variant_a",
))
STATS_DIR = ROOT / "stats"
STATS_DIR.mkdir(exist_ok=True)

EXPERIMENTS_ROOT = ROOT / "experiments"
if not EXPERIMENTS_ROOT.exists():
    EXPERIMENTS_ROOT = ROOT / "families" / "pooled_oinfo_ladder" / "experiments"

BAGS = ["combined", "functional", "structural"]

# ── canonical domain map (explicit input shared with domain-greedy) ───────────
DOMAIN_LABELS_CSV = pathlib.Path(os.environ.get(
    "V3_EXPOSOME_DOMAIN_LABELS_CSV",
    pathlib.Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv",
))
DOMAIN_MAP = load_domain_map(DOMAIN_LABELS_CSV)


def domain_stats(predictors_str: str) -> dict:
    """Compute domain diversity stats for a pipe-separated predictor string."""
    vars_list = [v.strip() for v in str(predictors_str).split("|") if v.strip()]
    stats = unweighted_domain_stats(vars_list, domain_map=DOMAIN_MAP)
    counts = stats["domain_counts"]
    total = len(vars_list)
    n_air_pollution = counts.get("Air Pollution", 0)
    frac_air_pollution = n_air_pollution / total if total else 0.0
    n_democracy = counts.get("Democracy", 0)
    frac_democracy = n_democracy / total if total else 0.0
    return {
        "n_domains": stats["n_domains"],
        "shannon_h": stats["shannon_h"],
        "max_domain_frac": stats["max_domain_frac"],
        "dominant_domain": stats["dominant_domain"],
        "frac_air_pollution": frac_air_pollution,
        "frac_democracy": frac_democracy,
    }


# ── main ─────────────────────────────────────────────────────────────────
all_results = []
interaction_results = []

for bag in BAGS:
    print(f"\n{'='*70}")
    print(f"  {bag.upper()} BAG")
    print(f"{'='*70}")

    exp_path = EXPERIMENTS_ROOT / f"pooled_oinfo_ladder_{bag}" / "xgb_tree_d2" / "xgb_summary_with_calibration_complexity.csv"
    if not exp_path.exists():
        print(f"  ⚠ Missing experiments for {bag} — skipping")
        continue
    df = pd.read_csv(exp_path)

    # Compute domain stats per model
    dstats = pd.DataFrame([domain_stats(p) for p in df["predictors_identity"]])
    df = pd.concat([df.reset_index(drop=True), dstats], axis=1)
    df["is_synergistic"] = (df["objective"] == "o_min").astype(int)

    # Quick descriptives
    print(f"\nDescriptives by objective:")
    for obj, grp in df.groupby("objective"):
        label = "Synergistic" if obj == "o_min" else "Redundant"
        print(f"  {label} (n={len(grp)}):")
        print(f"    R²:       {grp['global_oof_r2'].mean():.4f} ± {grp['global_oof_r2'].std():.4f}")
        print(f"    Shannon H:{grp['shannon_h'].mean():.3f} ± {grp['shannon_h'].std():.3f}")
        print(f"    n_domains:{grp['n_domains'].mean():.1f} ± {grp['n_domains'].std():.1f}")
        print(f"    order:    {grp['order'].mean():.1f} ± {grp['order'].std():.1f}")
        print(f"    frac_air: {grp['frac_air_pollution'].mean():.3f}")
        print(f"    frac_dem: {grp['frac_democracy'].mean():.3f}")

    # ── Regression 1: R² ~ shannon_h * is_synergistic + order ────────
    # Classical OLS for point estimates; order-clustered robust SE for inference
    # (candidates of the same order share search sub-structure); permutation P on
    # the interaction term as the assumption-free significance test.
    print(f"\n--- Regression: R² ~ shannon_h * is_synergistic + order ---")
    m1 = smf.ols("global_oof_r2 ~ shannon_h * is_synergistic + order", data=df).fit()
    m1_cl = smf.ols("global_oof_r2 ~ shannon_h * is_synergistic + order",
                    data=df).fit(cov_type="cluster", cov_kwds={"groups": df["order"]})
    print(m1.summary2().tables[1].to_string())
    print(f"Model R² = {m1.rsquared:.4f}, Adj R² = {m1.rsquared_adj:.4f}")
    print("  order-clustered robust SE (interaction term):")
    print(f"    beta={m1_cl.params['shannon_h:is_synergistic']:+.4f} "
          f"cluster-SE P={m1_cl.pvalues['shannon_h:is_synergistic']:.3g}")
    m1_perm = _perm_interaction_p(df, "global_oof_r2", "shannon_h")
    print(f"    permutation P (interaction)={m1_perm['perm_p']:.4g} "
          f"(obs beta={m1_perm['obs_beta']:+.4f}, {m1_perm['n_perm']} perms)")

    # ── Regression 2: f² ~ shannon_h * is_synergistic + order ────────
    print(f"\n--- Regression: f² ~ shannon_h * is_synergistic + order ---")
    m2 = smf.ols("train_f2_mean ~ shannon_h * is_synergistic + order", data=df).fit()
    print(m2.summary2().tables[1].to_string())
    print(f"Model R² = {m2.rsquared:.4f}, Adj R² = {m2.rsquared_adj:.4f}")

    # ── Regression 3: R² ~ n_domains * is_synergistic + order ────────
    print(f"\n--- Regression: R² ~ n_domains * is_synergistic + order ---")
    m3 = smf.ols("global_oof_r2 ~ n_domains * is_synergistic + order", data=df).fit()
    m3_cl = smf.ols("global_oof_r2 ~ n_domains * is_synergistic + order",
                    data=df).fit(cov_type="cluster", cov_kwds={"groups": df["order"]})
    print(m3.summary2().tables[1].to_string())
    print(f"Model R² = {m3.rsquared:.4f}, Adj R² = {m3.rsquared_adj:.4f}")
    print("  order-clustered robust SE (interaction term):")
    print(f"    beta={m3_cl.params['n_domains:is_synergistic']:+.4f} "
          f"cluster-SE P={m3_cl.pvalues['n_domains:is_synergistic']:.3g}")
    m3_perm = _perm_interaction_p(df, "global_oof_r2", "n_domains")
    print(f"    permutation P (interaction)={m3_perm['perm_p']:.4g} "
          f"(obs beta={m3_perm['obs_beta']:+.4f}, {m3_perm['n_perm']} perms)")

    # ── Regression 4 (Tier 2.4): pollution failure within synergistic ─
    syn = df[df["objective"] == "o_min"].copy()
    print(f"\n--- Pollution failure (synergistic only, n={len(syn)}): R² ~ frac_air_pollution + order + score ---")
    m4 = smf.ols("global_oof_r2 ~ frac_air_pollution + order + score", data=syn).fit()
    print(m4.summary2().tables[1].to_string())
    print(f"Model R² = {m4.rsquared:.4f}")

    # ── Regression 5: democracy dominance within synergistic ─────────
    print(f"\n--- Democracy dominance (synergistic only): R² ~ frac_democracy + order + score ---")
    m5 = smf.ols("global_oof_r2 ~ frac_democracy + order + score", data=syn).fit()
    print(m5.summary2().tables[1].to_string())
    print(f"Model R² = {m5.rsquared:.4f}")

    # Save coefficients
    for name, model, scope in [
        ("R2_diversity", m1, "all"),
        ("f2_diversity", m2, "all"),
        ("R2_ndomains", m3, "all"),
        ("R2_pollution", m4, "syn_only"),
        ("R2_democracy", m5, "syn_only"),
    ]:
        tbl = model.summary2().tables[1].copy()
        tbl["bag"] = bag
        tbl["regression"] = name
        tbl["scope"] = scope
        tbl["model_r2"] = model.rsquared
        tbl["model_r2_adj"] = model.rsquared_adj
        tbl["n_obs"] = int(model.nobs)
        all_results.append(tbl)

    # Robust-inference summary for the interaction terms (the headline diversity
    # claim): classical P, order-clustered robust P, and permutation P side by side.
    for name, diversity_term, model_cl, perm in [
        ("R2_diversity", "shannon_h", m1_cl, m1_perm),
        ("R2_ndomains", "n_domains", m3_cl, m3_perm),
    ]:
        inter = f"{diversity_term}:is_synergistic"
        interaction_results.append({
            "bag": bag,
            "regression": name,
            "interaction_term": inter,
            "beta_interaction": float(m1.params[inter]) if name == "R2_diversity" else float(m3.params[inter]),
            "p_classical_ols": float(m1.pvalues[inter]) if name == "R2_diversity" else float(m3.pvalues[inter]),
            "p_cluster_order": float(model_cl.pvalues[inter]),
            "p_permutation": perm["perm_p"],
            "perm_n": perm["n_perm"],
        })

# ── Save combined table ──────────────────────────────────────────────────
combined = pd.concat(all_results, ignore_index=False)
out_path = STATS_DIR / "domain_diversity_regression.csv"
combined.to_csv(out_path)
print(f"\n\nSaved regression table to {out_path}")

inter_df = pd.DataFrame(interaction_results)
inter_path = STATS_DIR / "domain_diversity_interaction_robust.csv"
inter_df.to_csv(inter_path, index=False)
print(f"Saved robust interaction inference to {inter_path}")
print(inter_df.to_string(index=False))
print("Done.")
