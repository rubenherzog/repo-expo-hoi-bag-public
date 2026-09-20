#!/usr/bin/env python3
"""cbn_fast.py
============
Additive fast path for the **winner-level** country-block null
(``scripts/run_country_block_null.py``). Opt-in via ``CBN_FAST=1``.

Why this exists
---------------
The default driver re-derives everything on every permutation:
``_perm_draw -> _eval_once -> evaluate_candidates_by_rung`` rebuilds
``prepare_bag_context`` + all per-fold base matrices and writes tempdir CSVs on
*every* call (2 bags x 4 rungs x N_PERM times). But the LOCO fold structure and the
covariate design are **permutation-invariant** — a country-block permutation only
reassigns the exposome vectors. So we can build the context + fold designs **once
per bag** and, per permutation, swap only ``context["X_exp"]`` and call the existing
``_fit_candidate`` directly.

Nothing here overwrites or edits existing functions. The default path in
``run_country_block_null.py`` is untouched and serves as the correctness oracle; the
equivalence gate (``CBN_FAST`` vs default at a fixed seed) must be bit-identical.

Exactness
---------
* ``prepare_analysis_table`` filters rows only on Diagnosis/country/BAG — never on
  exposome values — so the surviving row set ``work`` is identical under any exposome
  permutation. Building ``context`` once from the original ``model_df`` is therefore
  exact.
* ``permute_exposome_fast`` reproduces ``_permute_country_exposome`` exactly: same
  block universe and key order (first-occurrence over ``model_df``), same single
  ``rng.permutation`` per permutation (drawn from ``default_rng(seed + perm_idx)`` and
  reused across rungs, matching ``_perm_draw``), implemented as one vectorized gather.
* Per-fold seeds and ``_fit_candidate`` are the *same* objects as the default path.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Match the default driver: canonical OLS design must be active before any worker
# is spawned so loky children inherit it.
os.environ.setdefault("SENSITIVITY_OLS_DX_INTERACTIONS", "1")

from loco_fusion_matrix_engine import prepare_analysis_table, prepare_bag_context, target_map  # noqa: E402
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs  # noqa: E402
from xgb_loco_engine import _build_base_fold_mats  # noqa: E402

from scripts.sensitivity_common import (  # noqa: E402
    _fit_candidate,
    active_rungs,
    analysis_cfg_from_config,
    build_original_model_df,
    bundle_sensitivity_root,
    copy_tree_contents,
    cv_cfg,
    early_stop_cfg,
    load_raw_and_domains,
    load_sensitivity_config,
    repo_sensitivity_root,
)

# Reuse winner selection / candidate construction from the default driver unchanged.
from scripts.run_country_block_null import (  # noqa: E402
    KEY_COUNTRY,
    KEY_YEAR,
    _bags,
    _env_bool,
    country_balanced_r2,
    select_winners,
    winners_to_candidate_df,
)
from scripts.sensitivity_common import canonical_root, load_fig2_candidate_pool  # noqa: E402

BASE_SEED = 20260304  # identical to evaluate_candidates_by_rung


# --------------------------------------------------------------------------- #
# Build-once context + permutation plan
# --------------------------------------------------------------------------- #
def build_cbn_context_once(model_df: pd.DataFrame, feature_names: list[str], bag: str, analysis_cfg: dict):
    """Reusable ``context`` + ``fold_designs`` for one BAG (rung-independent).

    Mirrors the setup block of ``evaluate_candidates_by_rung`` (same ``cv_cfg``,
    ``early_stop_cfg`` and per-fold seeds ``BASE_SEED + i``) without modifying it.
    """
    cv = cv_cfg()
    y_col = target_map(bag)[bag]
    context = prepare_bag_context(model_df, y_col, bag, analysis_cfg, cv, feature_names)
    fold_designs = {
        c: _build_base_fold_mats(context, c, analysis_cfg, early_stop_cfg(), seed=BASE_SEED + i)
        for i, c in enumerate(context["countries"])
    }
    return context, fold_designs


def build_permutation_plan(model_df: pd.DataFrame, feature_names: list[str], bag: str, analysis_cfg: dict):
    """Precompute the country-year block universe and each surviving row's block.

    Returns ``(reps_matrix, work_block_idx, n_blocks)`` such that, for a permutation
    ``perm = rng.permutation(n_blocks)``, ``reps_matrix[perm[work_block_idx]]`` is the
    permuted exposome aligned to ``context``'s row order — bit-identical to running
    ``_permute_country_exposome`` on ``model_df`` then ``prepare_bag_context``.
    """
    # Block universe + first-occurrence order over model_df (matches _permute_country_exposome).
    key_md = model_df[KEY_COUNTRY].astype(str) + "||" + model_df[KEY_YEAR].astype(str)
    reps = (
        model_df.assign(__key__=key_md)
        .drop_duplicates("__key__")
        .set_index("__key__")[feature_names]
    )
    keys = reps.index.to_numpy()
    reps_matrix = reps.to_numpy(dtype=float)
    key_to_pos = {k: i for i, k in enumerate(keys)}

    # Surviving rows in context order (prepare_analysis_table is deterministic and
    # exposome-independent, so this reproduces prepare_bag_context's `work`).
    cv = cv_cfg()
    y_col = target_map(bag)[bag]
    work = prepare_analysis_table(model_df, y_col, analysis_cfg, cv).reset_index(drop=True)
    work_key = (work[KEY_COUNTRY].astype(str) + "||" + work[KEY_YEAR].astype(str)).to_numpy()
    work_block_idx = np.fromiter((key_to_pos[k] for k in work_key), dtype=int, count=len(work_key))
    return reps_matrix, work_block_idx, len(keys)


def permute_exposome_fast(reps_matrix: np.ndarray, work_block_idx: np.ndarray, n_blocks: int,
                          rng: np.random.Generator) -> np.ndarray:
    """One vectorized country-block permutation of the exposome (see module docstring)."""
    perm = rng.permutation(n_blocks)
    return reps_matrix[perm[work_block_idx]]


# --------------------------------------------------------------------------- #
# Winner evaluation on a (possibly permuted) exposome
# --------------------------------------------------------------------------- #
def eval_winners_fast(context: dict, fold_designs: dict, candidate_df: pd.DataFrame,
                      exposome_cols: list[str], rung: str, xgb_cfg: dict | None,
                      x_exp: np.ndarray | None = None) -> pd.Series:
    """Country-balanced LOCO R² per predictors_identity on one design.

    ``x_exp`` overrides ``context["X_exp"]`` (the permuted exposome); ``None`` uses the
    original exposome (observed). Dedupes candidates on ``predictors_identity`` and
    calls the existing ``_fit_candidate`` — identical results to ``_eval_once``.
    """
    ctx = dict(context)
    if x_exp is not None:
        ctx["X_exp"] = x_exp
    uniq = candidate_df.drop_duplicates("predictors_identity", keep="first")
    out = {}
    for row in uniq.to_dict(orient="records"):
        _summary, country = _fit_candidate(
            row,
            context=ctx,
            fold_designs=fold_designs,
            exposome_cols=exposome_cols,
            rung_id=rung,
            xgb_cfg=xgb_cfg,
            fold_pc=None,
        )
        country = country.copy()
        country["predictors_identity"] = str(row["predictors_identity"])
        out[str(row["predictors_identity"])] = country_balanced_r2(country).iloc[0]
    return pd.Series(out, dtype=float)


def _perm_chunk_fast(perm_indices: list[int], context: dict, fold_designs: dict,
                     reps_matrix: np.ndarray, work_block_idx: np.ndarray, n_blocks: int,
                     rung_to_cand: dict, rung_to_xgb: dict, exposome_cols: list[str],
                     seed: int) -> list[dict]:
    """Evaluate a *chunk* of permutations in one worker (build-once context reused).

    Per perm: draw one block permutation (reused across rungs, matching ``_perm_draw``),
    then evaluate each rung's winners.
    """
    draws = []
    for perm_idx in perm_indices:
        rng = np.random.default_rng(seed + perm_idx)
        x_exp = permute_exposome_fast(reps_matrix, work_block_idx, n_blocks, rng)
        for rung, cand in rung_to_cand.items():
            r2 = eval_winners_fast(context, fold_designs, cand, exposome_cols, rung,
                                   rung_to_xgb[rung], x_exp=x_exp)
            for pid, val in r2.items():
                draws.append({"perm_idx": perm_idx, "rung_id": rung,
                              "predictors_identity": pid, "null_r2": float(val),
                              "r2_estimand": "country_balanced"})
    return draws


# --------------------------------------------------------------------------- #
# Full max-null pool (whole Fig.2 candidate set, not just winners)
# --------------------------------------------------------------------------- #
def select_maxnull_pool(bag: str, rungs: list[str], order_max: int) -> pd.DataFrame:
    """Fig.2 candidate pool capped at ``order_max`` (rung-independent predictor set)."""
    pool = load_fig2_candidate_pool(bag)
    return pool[pd.to_numeric(pool["order"], errors="coerce") <= order_max].copy()


def select_maxnull_observed(bag: str, rungs: list[str], order_max: int) -> dict[str, float]:
    """observed_max_r2 per rung, read directly from the published canonical
    ``metrics_global_long.parquet`` (``full_r2``) -- no LOCO refit. The default
    driver's ``_eval_once`` over the whole pool reproduces this exact value (it is
    the source the pool/metrics were generated from), so taking the max of the
    on-disk column is equivalent and free."""
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
    return {r: float(df.loc[df["rung_id"].astype(str) == r, "full_r2"].max()) for r in rungs}


def _chunk_indices(n_perm: int, n_jobs: int) -> list[list[int]]:
    """Split perm indices into ~4*n_jobs chunks for load balance with few pickles."""
    n_chunks = max(1, min(n_perm, int(n_jobs) * 4))
    bounds = np.array_split(np.arange(n_perm), n_chunks)
    return [b.tolist() for b in bounds if len(b)]


# --------------------------------------------------------------------------- #
# Orchestration (winner-level only; mirrors run_country_block_null.main outputs)
# --------------------------------------------------------------------------- #
def run_fast() -> None:
    cfg = load_sensitivity_config()
    analysis_cfg = analysis_cfg_from_config(cfg)
    n_perm = int(os.environ.get("CBN_N_PERM", "1000"))
    n_jobs = int(os.environ.get("CBN_N_JOBS", "8"))
    seed = int(os.environ.get("CBN_SEED", str(BASE_SEED)))
    include_single = _env_bool("CBN_INCLUDE_SINGLE", default=True)
    order_max = int(os.environ.get("CBN_ORDER_MAX", "30"))

    rungs = active_rungs(cfg)
    req_rungs = os.environ.get("CBN_RUNGS", "").strip()
    if req_rungs:
        rungs = [r.strip() for r in req_rungs.split(",") if r.strip()]

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, analysis_cfg)
    for col in (KEY_COUNTRY, KEY_YEAR):
        if col not in model_df.columns:
            raise KeyError(f"model_df missing block key column {col!r}")

    rung_specs = {spec["rung_id"]: spec for spec in get_rung_specs()}

    local_root = repo_sensitivity_root(cfg) / "country_block_null"
    local_root.mkdir(parents=True, exist_ok=True)
    bundle_root = bundle_sensitivity_root(cfg) / "country_block_null"
    bundle_root.mkdir(parents=True, exist_ok=True)

    for bag in _bags():
        print(f"\n=== Country-block null [FAST] | bag={bag} | mode=winner "
              f"| n_perm={n_perm} | rungs={rungs} | order_max={order_max} | n_jobs={n_jobs} ===")

        winners = select_winners(bag, rungs, order_max, include_single)
        if winners.empty:
            print(f"  no winners for {bag}; skipping")
            continue
        rung_to_cand = {
            r: winners_to_candidate_df(winners[winners["rung_id"] == r])
            for r in rungs if (winners["rung_id"] == r).any()
        }
        rung_to_xgb = {
            r: (None if r == "ols" else build_xgb_cfg_for_rung(rung_specs[r]))
            for r in rung_to_cand
        }

        # Build-once: context/fold designs (rung-independent) + permutation plan.
        context, fold_designs = build_cbn_context_once(model_df, feature_names, bag, analysis_cfg)
        reps_matrix, work_block_idx, n_blocks = build_permutation_plan(
            model_df, feature_names, bag, analysis_cfg
        )

        # Observed: for best_overall/best_syn/best_red (objective o_min/o_max) the
        # equivalence gate (CBN_FAST vs default oracle, fixed seed) proved
        # observed_r2 == observed_r2_parquet bit-for-bit, so read it straight off
        # disk instead of re-fitting -- pure recompute of a known value otherwise.
        # best_single does NOT satisfy that equivalence (its source pipeline,
        # load_best_single_by_rung, differs from evaluate_candidates_by_rung's
        # candidate fit enough to disagree with the published full_r2 by up to
        # ~0.03 R^2 for some rungs) -- it must still be refit here.
        single_mask = winners["objective"].astype(str) == "single"
        single_winners = winners[single_mask]
        observed_single = {
            r: eval_winners_fast(context, fold_designs, winners_to_candidate_df(single_winners[single_winners["rung_id"] == r]),
                                 feature_names, r, rung_to_xgb[r], x_exp=None)
            for r in single_winners["rung_id"].unique()
        } if not single_winners.empty else {}
        obs_rows = winners.to_dict("records")

        # Null draws (parallel over permutation chunks).
        chunks = _chunk_indices(n_perm, n_jobs)
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_perm_chunk_fast)(
                idxs, context, fold_designs, reps_matrix, work_block_idx, n_blocks,
                rung_to_cand, rung_to_xgb, feature_names, seed,
            )
            for idxs in chunks
        )
        null_df = pd.DataFrame([d for chunk in chunk_results for d in chunk])
        null_path = bundle_root / f"{bag}_winner_null_draws.parquet"
        null_df.to_parquet(null_path, index=False)
        print(f"  null draws -> {null_path} ({len(null_df)} rows)")

        # p-values (identical layout to run_country_block_null.main winner branch).
        rows = []
        for w in obs_rows:
            r, pid = w["rung_id"], w["predictors_identity"]
            if str(w["objective"]) == "single":
                obs = float(observed_single[r].get(pid, np.nan))
            else:
                obs = float(w["observed_r2_parquet"])  # proven bit-identical; skip refit
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


# --------------------------------------------------------------------------- #
# Orchestration (full max-null; whole Fig.2 pool, capped order_max)
# --------------------------------------------------------------------------- #
def run_fast_maxnull() -> None:
    """Same build-once optimization as ``run_fast``, applied to the full Fig.2
    candidate pool (every candidate up to ``CBN_ORDER_MAX``, not just winners).
    Observed values are read from disk (``select_maxnull_observed``) instead of
    refitting -- the default driver's ``_eval_once`` over the whole pool only
    reproduces the published ``full_r2`` column, so refitting it is pure recompute.
    """
    cfg = load_sensitivity_config()
    analysis_cfg = analysis_cfg_from_config(cfg)
    n_perm = int(os.environ.get("CBN_N_PERM", "10000"))
    n_jobs = int(os.environ.get("CBN_N_JOBS", "8"))
    seed = int(os.environ.get("CBN_SEED", str(BASE_SEED)))
    order_max = int(os.environ.get("CBN_ORDER_MAX", "30"))

    rungs = active_rungs(cfg)
    req_rungs = os.environ.get("CBN_RUNGS", "").strip()
    if req_rungs:
        rungs = [r.strip() for r in req_rungs.split(",") if r.strip()]

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, analysis_cfg)
    for col in (KEY_COUNTRY, KEY_YEAR):
        if col not in model_df.columns:
            raise KeyError(f"model_df missing block key column {col!r}")

    rung_specs = {spec["rung_id"]: spec for spec in get_rung_specs()}

    local_root = repo_sensitivity_root(cfg) / "country_block_null"
    local_root.mkdir(parents=True, exist_ok=True)
    bundle_root = bundle_sensitivity_root(cfg) / "country_block_null"
    bundle_root.mkdir(parents=True, exist_ok=True)

    for bag in _bags():
        print(f"\n=== Country-block null [FAST] | bag={bag} | mode=maxnull "
              f"| n_perm={n_perm} | rungs={rungs} | order_max={order_max} | n_jobs={n_jobs} ===")

        pool = select_maxnull_pool(bag, rungs, order_max)
        if pool.empty:
            print(f"  empty pool for {bag}; skipping")
            continue
        rung_to_cand = {r: pool for r in rungs}
        rung_to_xgb = {
            r: (None if r == "ols" else build_xgb_cfg_for_rung(rung_specs[r]))
            for r in rungs
        }

        # Build-once: context/fold designs (rung-independent) + permutation plan.
        context, fold_designs = build_cbn_context_once(model_df, feature_names, bag, analysis_cfg)
        reps_matrix, work_block_idx, n_blocks = build_permutation_plan(
            model_df, feature_names, bag, analysis_cfg
        )

        # Observed: instant read of the published max(full_r2) over the pool, no refit.
        observed = select_maxnull_observed(bag, rungs, order_max)

        # Null draws (parallel over permutation chunks); reduce to per-(perm,rung) max
        # to match the default driver's maxnull output schema exactly.
        chunks = _chunk_indices(n_perm, n_jobs)
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_perm_chunk_fast)(
                idxs, context, fold_designs, reps_matrix, work_block_idx, n_blocks,
                rung_to_cand, rung_to_xgb, feature_names, seed,
            )
            for idxs in chunks
        )
        raw_draws = pd.DataFrame([d for chunk in chunk_results for d in chunk])
        null_df = (
            raw_draws.groupby(["perm_idx", "rung_id"], as_index=False)["null_r2"].max()
        )
        null_df["r2_estimand"] = "country_balanced"
        null_path = bundle_root / f"{bag}_maxnull_null_draws.parquet"
        null_df.to_parquet(null_path, index=False)
        print(f"  null draws -> {null_path} ({len(null_df)} rows)")

        rows = []
        for r in rungs:
            nd = null_df[null_df["rung_id"] == r]["null_r2"].to_numpy()
            obs = observed[r]
            p = (1 + int(np.sum(nd >= obs))) / (len(nd) + 1) if len(nd) else np.nan
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
        out.to_csv(out_path, index=False)
        print(f"  p-values -> {out_path}")
        print(out.to_string(index=False))

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "country_block_null")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_fast()
