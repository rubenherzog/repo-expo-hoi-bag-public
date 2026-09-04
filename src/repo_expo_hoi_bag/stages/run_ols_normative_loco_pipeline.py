#!/usr/bin/env python3
"""OLS LOCO normative-transfer pipeline.

This runner mirrors ``run_xgb_normative_loco_pipeline.py`` but evaluates only the
canonical OLS rung. It fits once per normative train group and LOCO fold, then
scores the held-out country by requested diagnosis groups.

Detailed fold/country outputs are written to the external runtime. The local repo
stores only lightweight summaries and provenance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from loco_fusion_matrix_engine import regression_metrics_extended
from scripts.ridge_common import (
    PRIMARY_DX,
    RidgeFamily,
    build_ridge_model_df,
    prepare_ridge_context,
    ridge_analysis_cfg,
)
from scripts.sensitivity_common import (
    _build_canonical_ols_design,
    _predictor_indices,
    baseline_candidate_df,
    bundle_root,
    load_fig2_candidate_pool,
    log_msg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILIES = ["pooled", "cn_norm", "ad_norm", "ftd_norm", "adftd_norm"]
DEFAULT_TEST_DX = ["CN", "AD", "FTD"]
DEFAULT_RUNGS = ["ols"]


@dataclass(frozen=True)
class OLSNormFamily:
    family_id: str
    train_dx: tuple[str, ...] | None
    include_diagnosis: bool
    default_test_dx: tuple[str, ...] | None

    @property
    def train_label(self) -> str:
        return "all" if self.train_dx is None else "+".join(self.train_dx)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y"}


def ols_norm_output_root() -> Path:
    root = Path(os.environ.get("OLS_NORM_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "ols_normative_loco")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def ols_norm_eval_root() -> Path:
    override = os.environ.get("OLS_NORM_EVAL_ROOT", "").strip()
    if override:
        root = Path(override)
    else:
        if not os.environ.get("REPRO_DATA_ROOT", "").strip():
            raise EnvironmentError(
                "REPRO_DATA_ROOT is required. Heavy OLS normative outputs must be "
                "written to the external runtime, not local outputs/."
            )
        subdir = os.environ.get("OLS_NORM_BUNDLE_SUBDIR", "ols_normative_loco").strip() or "ols_normative_loco"
        root = bundle_root() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def ols_norm_bags() -> list[str]:
    requested = os.environ.get("OLS_NORM_BAGS", "").strip()
    bags = [b.strip() for b in requested.split(",") if b.strip()] if requested else ["functional", "structural"]
    bad = [b for b in bags if b not in {"functional", "structural", "combined"}]
    if bad:
        raise ValueError(f"Unsupported OLS_NORM_BAGS: {bad}")
    return [b for b in ["functional", "structural", "combined"] if b in set(bags)]


def ols_norm_requested_families() -> list[str]:
    requested = os.environ.get("OLS_NORM_FAMILIES", "").strip()
    families = [x.strip() for x in requested.split(",") if x.strip()] if requested else list(DEFAULT_FAMILIES)
    bad = [f for f in families if f not in set(DEFAULT_FAMILIES)]
    if bad:
        raise ValueError(f"Unsupported OLS_NORM_FAMILIES: {bad}; allowed={DEFAULT_FAMILIES}")
    return [f for f in DEFAULT_FAMILIES if f in set(families)]


def ols_norm_test_dx() -> list[str]:
    requested = os.environ.get("OLS_NORM_TEST_DX", "").strip()
    tests = [x.strip() for x in requested.split(",") if x.strip()] if requested else list(DEFAULT_TEST_DX)
    bad = [x for x in tests if x not in set(PRIMARY_DX)]
    if bad:
        raise ValueError(f"Unsupported OLS_NORM_TEST_DX: {bad}; allowed={PRIMARY_DX}")
    return [x for x in PRIMARY_DX if x in set(tests)]


def ols_norm_rungs() -> list[str]:
    requested = os.environ.get("OLS_NORM_RUNGS", "").strip()
    rungs = [r.strip() for r in requested.split(",") if r.strip()] if requested else list(DEFAULT_RUNGS)
    bad = [r for r in rungs if r != "ols"]
    if bad:
        raise ValueError(f"OLS normative supports only rung='ols'; got {bad}")
    return ["ols"]


def ols_norm_families() -> list[OLSNormFamily]:
    specs = {
        "pooled": OLSNormFamily("pooled", None, True, None),
        "cn_norm": OLSNormFamily("cn_norm", ("CN",), False, tuple(DEFAULT_TEST_DX)),
        "ad_norm": OLSNormFamily("ad_norm", ("AD",), False, tuple(DEFAULT_TEST_DX)),
        "ftd_norm": OLSNormFamily("ftd_norm", ("FTD",), False, tuple(DEFAULT_TEST_DX)),
        "adftd_norm": OLSNormFamily("adftd_norm", ("AD", "FTD"), True, tuple(DEFAULT_TEST_DX)),
    }
    return [specs[f] for f in ols_norm_requested_families()]


def ols_norm_candidate_pool(bag: str, *, smoke: bool = False) -> pd.DataFrame:
    order_max = int(os.environ.get("OLS_NORM_ORDER_MAX", "30"))
    top_k = int(os.environ.get("OLS_NORM_TOP_K_PER_ORDER", "20"))
    pool = load_fig2_candidate_pool(bag)
    pool["order"] = pd.to_numeric(pool["order"], errors="coerce")
    pool["score"] = pd.to_numeric(pool["score"], errors="coerce")
    pool = pool[pool["objective"].astype(str).isin(["o_min", "o_max"])].copy()
    pool = pool[(pool["order"] >= 3) & (pool["order"] <= order_max)].copy()
    rows = []
    for (objective, order), sub in pool.groupby(["objective", "order"], sort=True, observed=True):
        ascending = str(objective) == "o_min"
        rows.append(sub.sort_values("score", ascending=ascending).head(top_k))
    out = pd.concat(rows, ignore_index=True) if rows else pool.head(0).copy()
    out["ols_norm_order_max"] = order_max
    out["ols_norm_top_k_per_order"] = top_k
    if smoke:
        smoke_orders = sorted(pd.to_numeric(out["order"], errors="coerce").dropna().astype(int).unique().tolist())[:2]
        out = out[out["order"].astype(int).isin(smoke_orders)].copy()
        out = out.groupby(["objective", "order"], group_keys=False, observed=True).head(1).reset_index(drop=True)
    out = pd.concat([out, baseline_candidate_df()], ignore_index=True)
    out["predictors_identity"] = out["predictors_identity"].fillna("").astype(str)
    return out.reset_index(drop=True)


def _write_run_manifest(
    root: Path,
    *,
    eval_root: Path,
    bags: list[str],
    rungs: list[str],
    families: list[str],
    test_dx: list[str],
    n_jobs: int,
    smoke: bool,
) -> None:
    rows = [
        {"key": "local_summary_root", "value": str(root)},
        {"key": "external_eval_root", "value": str(eval_root)},
        {"key": "bags", "value": ",".join(bags)},
        {"key": "rungs", "value": ",".join(rungs)},
        {"key": "families", "value": ",".join(families)},
        {"key": "test_dx", "value": ",".join(test_dx)},
        {"key": "n_jobs", "value": str(n_jobs)},
        {"key": "smoke", "value": str(bool(smoke))},
        {"key": "OLS_NORM_ORDER_MAX", "value": os.environ.get("OLS_NORM_ORDER_MAX", "30")},
        {"key": "OLS_NORM_TOP_K_PER_ORDER", "value": os.environ.get("OLS_NORM_TOP_K_PER_ORDER", "20")},
        {"key": "OLS_NORM_EVAL_CHUNK_SIZE", "value": os.environ.get("OLS_NORM_EVAL_CHUNK_SIZE", "120")},
        {"key": "canonical_ols_design", "value": "intercept+age+sex+year_spline_df4+diagnosis+exposome+diagnosis_x_exposome_when_available"},
    ]
    pd.DataFrame(rows).to_csv(root / "ols_norm_run_manifest.csv", index=False)


def _ridge_family_for_multi_test(family: OLSNormFamily, test_dx: list[str]) -> RidgeFamily:
    test_group = None if family.train_dx is None else tuple(test_dx)
    return RidgeFamily(
        family.family_id,
        family.train_dx,
        test_group,
        family.include_diagnosis,
        bool(family.family_id == "adftd_norm" and "CN" in set(test_dx)),
    )


def _fit_candidate_multi_test(
    row: dict,
    *,
    context: dict,
    exposome_cols: list[str],
    family: OLSNormFamily,
    test_dx: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_id = str(row["candidate_id"])
    pred_all = _predictor_indices(row, exposome_cols)
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)
    diag = np.asarray(context["diag"]).astype(str)
    pred_chunks = []
    country_rows = []

    for country in context["countries"]:
        train_idx = context["train_idx_by_country"][country]
        test_idx = context["test_idx_by_country"][country]
        pred_used = list(pred_all)
        if pred_used:
            var = np.nanvar(x_exp[np.ix_(train_idx, pred_used)], axis=0)
            pred_used = [idx for idx, keep in zip(pred_used, var > 0) if bool(keep)]

        status = "ok"
        try:
            x_train, x_test = _build_canonical_ols_design(context, country, train_idx, test_idx, x_exp, pred_used)
            beta, *_ = np.linalg.lstsq(x_train, y[train_idx], rcond=None)
            y_pred = x_test @ beta
        except Exception:
            y_pred = np.full(len(test_idx), np.nan, dtype=float)
            status = "failed"

        pred_df = pd.DataFrame(
            {
                "candidate_id": model_id,
                "rung_id": "ols",
                "family_id": family.family_id,
                "train_dx": family.train_label,
                "fold_country": country,
                "row_idx": test_idx,
                "diagnosis": diag[test_idx],
                "y_true": y[test_idx],
                "y_pred": y_pred,
                "status": status,
                "predictors_used_n": int(len(pred_used)),
            }
        )
        pred_chunks.append(pred_df)
        for label in (["all"] if family.train_dx is None else test_dx):
            mask = np.ones(len(test_idx), dtype=bool) if label == "all" else (diag[test_idx] == label)
            ok = mask & np.isfinite(y[test_idx]) & np.isfinite(y_pred)
            met = regression_metrics_extended(
                y[test_idx][ok],
                y_pred[ok],
                min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
            )
            country_rows.append(
                {
                    "candidate_id": model_id,
                    "rung_id": "ols",
                    "family_id": family.family_id,
                    "train_dx": family.train_label,
                    "test_dx": label,
                    "fold_country": country,
                    "n_test": int(ok.sum()),
                    "r2": met["r2"],
                    "rmse": met["rmse"],
                    "mae": met["mae"],
                    "corr2": met["corr2"],
                    "unseen_dx_reference_encoded": bool(family.family_id == "adftd_norm" and label == "CN"),
                }
            )

    pred_all_df = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()
    country_df = pd.DataFrame(country_rows)
    summary_rows = []
    for label in (["all"] if family.train_dx is None else test_dx):
        sub = pred_all_df if label == "all" else pred_all_df[pred_all_df["diagnosis"].astype(str) == label]
        ok = np.isfinite(pd.to_numeric(sub.get("y_true", pd.Series(dtype=float)), errors="coerce")) & np.isfinite(
            pd.to_numeric(sub.get("y_pred", pd.Series(dtype=float)), errors="coerce")
        )
        met = regression_metrics_extended(
            sub.loc[ok, "y_true"].to_numpy(dtype=float),
            sub.loc[ok, "y_pred"].to_numpy(dtype=float),
            min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
        )
        summary_rows.append(
            {
                "candidate_id": model_id,
                "rung_id": "ols",
                "family_id": family.family_id,
                "train_dx": family.train_label,
                "test_dx": label,
                "global_oof_r2": met["r2"],
                "global_rmse": met["rmse"],
                "global_mae": met["mae"],
                "global_corr2": met["corr2"],
                "n_scored": int(met["n_scored"]),
                "predictors_used_n": int(len(pred_all)),
                "unseen_dx_reference_encoded": bool(family.family_id == "adftd_norm" and label == "CN"),
            }
        )
    return pd.DataFrame(summary_rows), country_df


def _evaluate_train_family(
    *,
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    bag: str,
    family: OLSNormFamily,
    test_dx: list[str],
    outdir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    analysis_cfg = ridge_analysis_cfg(include_diagnosis=family.include_diagnosis)
    actual_test_dx = ["all"] if family.train_dx is None else list(test_dx)
    context = prepare_ridge_context(
        model_df,
        bag,
        exposome_cols,
        analysis_cfg,
        _ridge_family_for_multi_test(family, test_dx),
    )
    if not context["countries"]:
        raise ValueError(f"No valid folds for bag={bag} family={family.family_id}")

    candidate_df = candidate_df.copy().reset_index(drop=True)
    fit_df = candidate_df.drop_duplicates("predictors_identity", keep="first").reset_index(drop=True)
    fit_map = fit_df[["candidate_id", "predictors_identity"]].rename(columns={"candidate_id": "fit_candidate_id"})
    manifest = (
        candidate_df.groupby("predictors_identity", as_index=False, dropna=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            candidate_ids=("candidate_id", lambda x: "|".join(map(str, x))),
            candidate_families=("candidate_family", lambda x: "|".join(sorted(set(map(str, x))))),
            source_labels=("source_label", lambda x: "|".join(sorted(set(map(str, x))))),
            order=("order", "first"),
        )
        .merge(fit_map, on="predictors_identity", how="left")
    )
    manifest.to_csv(outdir / "fit_manifest.csv", index=False)

    global_path = outdir / f"{bag}_{family.family_id}_{family.train_label}_ols_global.csv"
    country_path = outdir / f"{bag}_{family.family_id}_{family.train_label}_ols_country.csv"
    if global_path.exists() and country_path.exists():
        existing = pd.read_csv(global_path)
        expected_rows = len(candidate_df) * len(actual_test_dx)
        if len(existing) == expected_rows:
            log_msg(f"OLS norm resume hit bag={bag} family={family.family_id} train={family.train_label}")
            return existing, pd.read_csv(country_path)

    records = fit_df.to_dict(orient="records")
    chunk_size = int(os.environ.get("OLS_NORM_EVAL_CHUNK_SIZE", "120"))
    fitted = []
    log_msg(
        f"OLS norm start bag={bag} family={family.family_id} train={family.train_label} "
        f"tests={','.join(actual_test_dx)} candidates={len(candidate_df)} unique={len(records)} "
        f"folds={len(context['countries'])} n_jobs={n_jobs} fit_once_per_train_group=true"
    )
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        if int(n_jobs) <= 1:
            fitted_chunk = [
                _fit_candidate_multi_test(
                    row,
                    context=context,
                    exposome_cols=exposome_cols,
                    family=family,
                    test_dx=test_dx,
                )
                for row in chunk
            ]
        else:
            fitted_chunk = Parallel(n_jobs=int(n_jobs), backend="loky", batch_size=1)(
                delayed(_fit_candidate_multi_test)(
                    row,
                    context=context,
                    exposome_cols=exposome_cols,
                    family=family,
                    test_dx=test_dx,
                )
                for row in chunk
            )
        fitted.extend(fitted_chunk)
        log_msg(
            f"OLS norm progress bag={bag} family={family.family_id} train={family.train_label} "
            f"unique_done={min(start + len(chunk), len(records))}/{len(records)}"
        )

    fit_summary = pd.concat([x[0] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
    fit_country = pd.concat([x[1] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
    fit_summary = fit_summary.rename(columns={"candidate_id": "fit_candidate_id"}).merge(
        fit_map, on="fit_candidate_id", how="left"
    )
    metadata = candidate_df.drop(columns=["nplet_vars"], errors="ignore").copy()
    metric_cols = [
        "predictors_identity",
        "fit_candidate_id",
        "rung_id",
        "family_id",
        "train_dx",
        "test_dx",
        "global_oof_r2",
        "global_rmse",
        "global_mae",
        "global_corr2",
        "n_scored",
        "predictors_used_n",
        "unseen_dx_reference_encoded",
    ]
    summary = metadata.merge(fit_summary[metric_cols], on="predictors_identity", how="left")
    summary.to_csv(global_path, index=False)
    fit_country.to_csv(country_path, index=False)
    summary.to_csv(outdir / f"{bag}_{family.family_id}_{family.train_label}_global_all.csv", index=False)
    fit_country.to_csv(outdir / f"{bag}_{family.family_id}_{family.train_label}_country_all.csv", index=False)
    log_msg(f"OLS norm complete bag={bag} family={family.family_id} train={family.train_label}")
    return summary, fit_country


def main() -> None:
    smoke = _env_bool("OLS_NORM_SMOKE") or _env_bool("SMOKE_TEST")
    bags = ols_norm_bags()
    rungs = ols_norm_rungs()
    families = ols_norm_families()
    family_ids = [family.family_id for family in families]
    test_dx = ols_norm_test_dx()
    if smoke:
        os.environ.setdefault("OLS_NORM_TOP_K_PER_ORDER", "1")
    n_jobs = int(os.environ.get("OLS_NORM_N_JOBS", "40"))
    local_root = ols_norm_output_root()
    eval_root = ols_norm_eval_root()
    _write_run_manifest(
        local_root,
        eval_root=eval_root,
        bags=bags,
        rungs=rungs,
        families=family_ids,
        test_dx=test_dx,
        n_jobs=n_jobs,
        smoke=smoke,
    )
    _write_run_manifest(
        eval_root,
        eval_root=eval_root,
        bags=bags,
        rungs=rungs,
        families=family_ids,
        test_dx=test_dx,
        n_jobs=n_jobs,
        smoke=smoke,
    )

    model_df, feature_names = build_ridge_model_df()
    all_summary = []
    all_country = []
    for bag in bags:
        pool = ols_norm_candidate_pool(bag, smoke=smoke)
        pool.to_csv(local_root / f"{bag}_ols_norm_candidate_pool.csv", index=False)
        pool.to_csv(eval_root / f"{bag}_ols_norm_candidate_pool.csv", index=False)
        log_msg(f"OLS norm candidate pool bag={bag}: rows={len(pool)}")
        for family in families:
            outdir = eval_root / bag / family.family_id / f"train_{family.train_label}"
            summary, country = _evaluate_train_family(
                model_df=model_df,
                candidate_df=pool,
                exposome_cols=feature_names,
                bag=bag,
                family=family,
                test_dx=test_dx,
                outdir=outdir,
                n_jobs=n_jobs,
            )
            for df in (summary, country):
                if not df.empty:
                    df["bag"] = bag
            all_summary.append(summary)
            all_country.append(country)

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        summary_all.to_csv(local_root / "ols_norm_global_all.csv", index=False)
        summary_all.to_csv(eval_root / "ols_norm_global_all.csv", index=False)
    if all_country:
        pd.concat(all_country, ignore_index=True).to_csv(eval_root / "ols_norm_country_all.csv", index=False)
    log_msg(f"OLS normative pipeline complete: local summaries={local_root} detailed eval={eval_root}")


if __name__ == "__main__":
    main()
