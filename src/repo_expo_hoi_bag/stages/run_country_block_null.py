#!/usr/bin/env python3
"""run_country_block_null.py
============================
Country-block permutation null for the exposome×BAG winners.

Scientific purpose
------------------
The greedy candidates are proposed by O-information **without seeing the BAG**, so
each candidate's LOCO R² is clean; the open concern (``HANDOFF_RECONCILIATION.md``
§3.2-3.3) is the *winner's curse* of reporting the best model, and whether a
winner's R² exceeds what is achievable when the exposome is decoupled from the BAG.

Design (§3.3)
-------------
We permute the **country-year → exposome-vector map**: every (country_clean,
exposome_year) block keeps its single shared exposome vector, but that vector is
reassigned to a different block. Subjects, BAG, covariates (age/sex/year/diagnosis)
and the LOCO fold structure (by ``country_clean``) are held fixed, so only the
exposome↔BAG association is broken. The exposome's marginal multiset and its
within-country-year constancy are preserved. The LOCO generalization unit
(country) is untouched; the permutation unit is the country-year (≈205), matching
the §3.3 sensitivity unit.

Two modes:
  * **winner-level (default).** Re-evaluate only the selected winners (best
    synergistic and best redundant per active rung).
    ``p = (1 + #{null R² ≥ observed}) / (N_PERM + 1)``. Cheap.
  * **full max-null (gated, CBN_FULL_MAXNULL=1).** Re-evaluate the whole Fig.2 pool
    capped at ``CBN_ORDER_MAX`` (default 20) under each permutation and record the
    **max** R² per rung → null of the maximum, which also addresses the winner's
    curse directly. Heavy; intended only once cap-20 robustness is established.

Outputs
-------
  bundle  $REPRO_DATA_ROOT/sensitivity/country_block_null/{bag}_null_draws.parquet
  local   outputs/sensitivity/country_block_null/{bag}_winner_pvalues.csv
          outputs/sensitivity/country_block_null/{bag}_maxnull_pvalues.csv (gated)

Usage
-----
    # smoke
    CBN_SMOKE=1 CBN_BAGS=functional CBN_N_PERM=5 CBN_RUNGS=ols \
        PYTHONPATH=. python -m scripts.run_country_block_null
    # winner-level full
    CBN_N_PERM=1000 CBN_N_JOBS=20 PYTHONPATH=. python -m scripts.run_country_block_null
    # gated full max-null at order<=20
    CBN_FULL_MAXNULL=1 CBN_ORDER_MAX=20 CBN_N_PERM=10000 CBN_N_JOBS=40 \
        PYTHONPATH=. python -m scripts.run_country_block_null
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The published OLS winners use the canonical design (year spline df=4 +
# diagnosis×exposome interactions); without this flag the evaluator's OLS is the
# plain version and the null would test the wrong model (HANDOFF_RECONCILIATION
# §2.4). Trees already reproduce canonical exactly, so this only affects OLS.
# Set before any worker is spawned so loky children inherit it.
os.environ.setdefault("SENSITIVITY_OLS_DX_INTERACTIONS", "1")

from repo_expo_hoi_bag.stages.sensitivity_common import (  # noqa: E402
    active_rungs,
    analysis_cfg_from_config,
    baseline_candidate_df,
    build_original_model_df,
    bundle_sensitivity_root,
    canonical_root,
    copy_tree_contents,
    load_best_single_by_rung,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_root,
    selected_bags,
    evaluate_candidates_by_rung,
)

KEY_COUNTRY = "country_clean"
KEY_YEAR = "exposome_year"


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _bags() -> list[str]:
    req = os.environ.get("CBN_BAGS", "").strip()
    if req:
        return [b.strip() for b in req.split(",") if b.strip()]
    cfg = load_sensitivity_config()
    return selected_bags(cfg, include_combined=False)


# --------------------------------------------------------------------------- #
# Winner selection
# --------------------------------------------------------------------------- #
def select_winners(bag: str, rungs: list[str], order_max: int, include_single: bool) -> pd.DataFrame:
    """One row per winner: label, rung_id, objective, order, predictors_identity,
    observed_r2_parquet (the published LOCO R² as a cross-check)."""
    p = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not p.exists():
        p = canonical_root() / "metrics_global_long.parquet"
    df = pd.read_parquet(p)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag]
    df = df[
        df["rung_id"].astype(str).isin(rungs)
        & df["objective"].astype(str).isin(["o_min", "o_max"])
        & (pd.to_numeric(df["order"], errors="coerce") <= order_max)
    ].copy()
    df["full_r2"] = pd.to_numeric(df["full_r2"], errors="coerce")

    rows = []
    for rung in rungs:
        rr = df[df["rung_id"].astype(str) == rung]
        if rr.empty:
            continue
        picks = {
            "best_syn": (rr[rr["objective"].astype(str) == "o_min"]
                         .pipe(lambda d: d.loc[d["full_r2"].idxmax()] if not d.empty else None)),
            "best_red": (rr[rr["objective"].astype(str) == "o_max"]
                         .pipe(lambda d: d.loc[d["full_r2"].idxmax()] if not d.empty else None)),
        }
        for label, row in picks.items():
            if row is None:
                continue
            rows.append({
                "bag": bag, "label": label, "rung_id": rung,
                "objective": str(row["objective"]), "order": int(row["order"]),
                "predictors_identity": str(row["predictors_identity"]),
                "observed_r2_parquet": float(row["full_r2"]),
            })

    if include_single:
        try:
            single = load_best_single_by_rung(bag)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"best-single load failed for {bag}: {exc}")
            single = pd.DataFrame()
        if not single.empty:
            r2col = next((c for c in ["global_oof_r2", "r2", "full_r2"] if c in single.columns), None)
            if r2col is not None:
                single[r2col] = pd.to_numeric(single[r2col], errors="coerce")
                for rung in rungs:
                    sr = single[single["source_rung"].astype(str) == rung]
                    if sr.empty:
                        continue
                    best = sr.loc[sr[r2col].idxmax()]
                    rows.append({
                        "bag": bag, "label": "best_single", "rung_id": rung,
                        "objective": "single", "order": 1,
                        "predictors_identity": str(best["feature_name"]),
                        "observed_r2_parquet": float(best[r2col]),
                    })
    return pd.DataFrame(rows)


def winners_to_candidate_df(winners_for_rung: pd.DataFrame) -> pd.DataFrame:
    """Build an engine-ready candidate_df (one row per unique predictor set)."""
    uniq = winners_for_rung.drop_duplicates("predictors_identity").reset_index(drop=True)
    cand = pd.DataFrame({
        "candidate_id": [f"winner_{i}" for i in range(len(uniq))],
        "objective": uniq["objective"].values,
        "order": uniq["order"].values,
        "score": 0.0,
        "predictors_identity": uniq["predictors_identity"].values,
    })
    cand["feature_id"] = cand["candidate_id"]
    cand["nplet_vars"] = cand["predictors_identity"].apply(lambda s: s.split("|"))
    cand["predictors_identity_n"] = cand["nplet_vars"].apply(len)
    cand["rank"] = range(1, len(cand) + 1)
    cand["candidate_family"] = "country_block_null_winner"
    cand["source_label"] = uniq["label"].values
    return cand


# --------------------------------------------------------------------------- #
# Permutation
# --------------------------------------------------------------------------- #
def _permute_country_exposome(model_df: pd.DataFrame, feature_names: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Reassign each (country, exposome_year) block's exposome vector to another
    block, preserving within-block constancy and the marginal multiset."""
    key = model_df[KEY_COUNTRY].astype(str) + "||" + model_df[KEY_YEAR].astype(str)
    reps = (
        model_df.assign(__key__=key)
        .drop_duplicates("__key__")
        .set_index("__key__")[feature_names]
    )
    keys = reps.index.to_numpy()
    perm = rng.permutation(len(keys))
    mapping = dict(zip(keys, keys[perm]))
    src_keys = key.map(mapping).to_numpy()
    out = model_df.copy()
    out[feature_names] = reps.loc[src_keys, feature_names].to_numpy()
    return out


def country_balanced_r2(country: pd.DataFrame) -> pd.Series:
    """Return the unweighted mean held-out-country R² per predictor set."""
    required = {"predictors_identity", "n_test", "r2"}
    missing = required.difference(country.columns)
    if missing:
        raise ValueError(f"Country metrics missing required columns: {sorted(missing)}")
    scored = country[pd.to_numeric(country["n_test"], errors="coerce").gt(0)].copy()
    scored["r2"] = pd.to_numeric(scored["r2"], errors="coerce")
    return scored.groupby("predictors_identity", observed=True)["r2"].mean()


def _eval_once(model_df: pd.DataFrame, candidate_df: pd.DataFrame, feature_names: list[str],
               bag: str, rung: str, analysis_cfg: dict) -> pd.Series:
    """Return country-balanced LOCO R² per predictor set for one design."""
    with tempfile.TemporaryDirectory(prefix="cbn_") as tmp:
        _summary, country = evaluate_candidates_by_rung(
            model_df=model_df,
            candidate_df=candidate_df,
            exposome_cols=feature_names,
            bag=bag,
            rungs=[rung],
            analysis_cfg=analysis_cfg,
            outdir=Path(tmp),
            n_jobs=1,
        )
    return country_balanced_r2(country)


def _perm_draw(perm_idx: int, model_df: pd.DataFrame, feature_names: list[str], bag: str,
               analysis_cfg: dict, rung_to_cand: dict, seed: int, full_maxnull: bool) -> list[dict]:
    rng = np.random.default_rng(seed + perm_idx)
    perm_df = _permute_country_exposome(model_df, feature_names, rng)
    draws = []
    for rung, cand in rung_to_cand.items():
        r2 = _eval_once(perm_df, cand, feature_names, bag, rung, analysis_cfg)
        if full_maxnull:
            draws.append({"perm_idx": perm_idx, "rung_id": rung,
                          "null_r2": float(r2.max()), "r2_estimand": "country_balanced"})
        else:
            for pid, val in r2.items():
                draws.append({"perm_idx": perm_idx, "rung_id": rung,
                              "predictors_identity": pid, "null_r2": float(val),
                              "r2_estimand": "country_balanced"})
    return draws


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_sensitivity_config()
    analysis_cfg = analysis_cfg_from_config(cfg)
    smoke = _env_bool("CBN_SMOKE") or _env_bool("SMOKE_TEST")
    full_maxnull = _env_bool("CBN_FULL_MAXNULL")
    n_perm = int(os.environ.get("CBN_N_PERM", "5" if smoke else "1000"))
    n_jobs = int(os.environ.get("CBN_N_JOBS", "1" if smoke else "8"))
    seed = int(os.environ.get("CBN_SEED", "20260304"))
    include_single = _env_bool("CBN_INCLUDE_SINGLE", default=True)
    default_order_max = "20" if full_maxnull else "30"
    order_max = int(os.environ.get("CBN_ORDER_MAX", default_order_max))

    rungs = active_rungs(cfg)
    req_rungs = os.environ.get("CBN_RUNGS", "").strip()
    if req_rungs:
        rungs = [r.strip() for r in req_rungs.split(",") if r.strip()]

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, analysis_cfg)
    for col in (KEY_COUNTRY, KEY_YEAR):
        if col not in model_df.columns:
            raise KeyError(f"model_df missing block key column {col!r}")

    local_root = repo_sensitivity_root(cfg) / "country_block_null"
    local_root.mkdir(parents=True, exist_ok=True)
    bundle_root = bundle_sensitivity_root(cfg) / "country_block_null"
    bundle_root.mkdir(parents=True, exist_ok=True)

    for bag in _bags():
        print(f"\n=== Country-block null | bag={bag} | mode={'maxnull' if full_maxnull else 'winner'} "
              f"| n_perm={n_perm} | rungs={rungs} | order_max={order_max} ===")

        if full_maxnull:
            pool = load_fig2_candidate_pool(bag)
            pool = pool[pd.to_numeric(pool["order"], errors="coerce") <= order_max].copy()
            rung_to_cand = {r: pool for r in rungs}
            observed = {}
            for r in rungs:
                obs = _eval_once(model_df, pool, feature_names, bag, r, analysis_cfg)
                observed[r] = float(obs.max())
            obs_rows = [{"bag": bag, "rung_id": r, "observed_max_r2": observed[r]} for r in rungs]
        else:
            winners = select_winners(bag, rungs, order_max, include_single)
            if winners.empty:
                print(f"  no winners for {bag}; skipping")
                continue
            rung_to_cand = {
                r: winners_to_candidate_df(winners[winners["rung_id"] == r])
                for r in rungs if (winners["rung_id"] == r).any()
            }
            observed = {}
            for r, cand in rung_to_cand.items():
                observed[r] = _eval_once(model_df, cand, feature_names, bag, r, analysis_cfg)
            obs_rows = winners.to_dict("records")

        draws = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_perm_draw)(i, model_df, feature_names, bag, analysis_cfg, rung_to_cand, seed, full_maxnull)
            for i in range(n_perm)
        )
        null_df = pd.DataFrame([d for chunk in draws for d in chunk])
        null_path = bundle_root / f"{bag}_{'maxnull' if full_maxnull else 'winner'}_null_draws.parquet"
        null_df.to_parquet(null_path, index=False)
        print(f"  null draws -> {null_path} ({len(null_df)} rows)")

        # p-values
        if full_maxnull:
            rows = []
            for r in rungs:
                nd = null_df[null_df["rung_id"] == r]["null_r2"].to_numpy()
                obs = observed[r]
                p = (1 + int(np.sum(nd >= obs))) / (len(nd) + 1)
                null_std = float(np.std(nd, ddof=1)) if len(nd) > 1 else np.nan
                rows.append({"bag": bag, "rung_id": r, "observed_max_r2": obs,
                             "r2_estimand": "country_balanced",
                             "null_mean": float(np.mean(nd)) if len(nd) else np.nan,
                             "null_p95": float(np.quantile(nd, 0.95)) if len(nd) else np.nan,
                             "null_std": null_std,
                             "cohen_d_vs_null": (obs - float(np.mean(nd))) / null_std if (len(nd) and not np.isnan(null_std) and null_std > 0) else np.nan,
                             "p_value": p, "n_perm": len(nd)})
            out = pd.DataFrame(rows)
            out_path = local_root / f"{bag}_maxnull_pvalues.csv"
        else:
            rows = []
            for w in obs_rows:
                r, pid = w["rung_id"], w["predictors_identity"]
                obs = float(observed[r].get(pid, np.nan))
                nd = null_df[(null_df["rung_id"] == r) & (null_df["predictors_identity"] == pid)]["null_r2"].to_numpy()
                p = (1 + int(np.sum(nd >= obs))) / (len(nd) + 1) if len(nd) else np.nan
                null_std = float(np.std(nd, ddof=1)) if len(nd) > 1 else np.nan
                rows.append({**{k: w[k] for k in ["bag", "label", "rung_id", "objective", "order",
                                                  "predictors_identity", "observed_r2_parquet"]},
                             "observed_r2": obs,
                             "r2_estimand": "country_balanced",
                             "null_mean": float(np.mean(nd)) if len(nd) else np.nan,
                             "null_p95": float(np.quantile(nd, 0.95)) if len(nd) else np.nan,
                             "null_std": null_std,
                             "cohen_d_vs_null": (obs - float(np.mean(nd))) / null_std if (len(nd) and not np.isnan(null_std) and null_std > 0) else np.nan,
                             "p_value": p, "n_perm": len(nd)})
            out = pd.DataFrame(rows)
            out_path = local_root / f"{bag}_winner_pvalues.csv"
        out.to_csv(out_path, index=False)
        print(f"  p-values -> {out_path}")
        print(out.to_string(index=False))

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "country_block_null")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # CBN_FAST=1 selects the additive build-once fast path (scripts/cbn_fast.py).
        # It does not modify main(); the default path remains the correctness oracle
        # for both winner-level and full max-null modes.
        if _env_bool("CBN_FAST") and _env_bool("CBN_FULL_MAXNULL"):
            from scripts.cbn_fast import run_fast_maxnull
            run_fast_maxnull()
        elif _env_bool("CBN_FAST"):
            from scripts.cbn_fast import run_fast
            run_fast()
        else:
            main()
