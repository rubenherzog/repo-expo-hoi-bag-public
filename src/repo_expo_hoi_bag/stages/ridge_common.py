from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from loco_fusion_matrix_engine import (
    encode_dummies,
    parse_literal_list,
    prepare_bag_context,
    prepare_year_basis_by_fold,
    regression_metrics_extended,
    target_map,
)
from scripts.sensitivity_common import (
    BAG_ORDER,
    analysis_cfg_from_config,
    baseline_candidate_df,
    build_original_model_df,
    bundle_root,
    copy_tree_contents,
    cv_cfg,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    log_msg,
)
from xgb_loco_engine import _split_train_val_by_country


REPO_ROOT = Path(__file__).resolve().parents[1]
RIDGE_RUNGS = [
    "ridge_main",
    "ridge_pairwise",
    "ridge_triple",
    "baseline_main",
    "baseline_pairwise",
    "baseline_triple",
]
DEFAULT_RIDGE_RUNGS = ["ridge_main", "ridge_pairwise"]
VARIANT_A_EXCLUDE_COUNTRIES = ["France", "Italy", "Egypt", "Greece", "Poland"]
VARIANT_A_EXCLUDE_DIAGNOSIS = ["Other", "AFM", "MCI"]
PRIMARY_DX = ["CN", "AD", "FTD"]


@dataclass(frozen=True)
class RidgeFamily:
    family_id: str
    train_dx: tuple[str, ...] | None
    test_dx: tuple[str, ...] | None
    include_diagnosis: bool
    unseen_dx_reference_encoded: bool = False

    @property
    def train_label(self) -> str:
        return "all" if self.train_dx is None else "+".join(self.train_dx)

    @property
    def test_label(self) -> str:
        return "all" if self.test_dx is None else "+".join(self.test_dx)


def ridge_output_root() -> Path:
    root = Path(os.environ.get("RIDGE_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "ridge_loco")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def ridge_eval_root(local_root: Path | None = None) -> Path:
    """Root for detailed/heavy Ridge outputs.

    The repo disk is intentionally reserved for lightweight summary tables and
    figures. Fold-level, alpha, complexity, and per-family eval artifacts go to
    the external runtime by default.
    """
    override = os.environ.get("RIDGE_EVAL_ROOT", "").strip()
    if override:
        root = Path(override)
        root.mkdir(parents=True, exist_ok=True)
        return root
    if not os.environ.get("REPRO_DATA_ROOT", "").strip():
        if os.environ.get("RIDGE_ALLOW_LOCAL_HEAVY", "").strip().lower() in {"1", "true", "yes", "y"}:
            if local_root is None:
                local_root = ridge_output_root()
            root = local_root / "_heavy_eval"
            root.mkdir(parents=True, exist_ok=True)
            return root
        raise EnvironmentError(
            "REPRO_DATA_ROOT is required for Ridge eval outputs. The local repo should "
            "only store lightweight summaries/figures. For temporary development only, "
            "set RIDGE_ALLOW_LOCAL_HEAVY=1."
        )
    subdir = os.environ.get("RIDGE_BUNDLE_SUBDIR", "ridge_loco").strip() or "ridge_loco"
    root = bundle_root() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def ridge_bags(include_combined: bool = False) -> list[str]:
    requested = os.environ.get("RIDGE_BAGS", "").strip()
    if requested:
        bags = [b.strip() for b in requested.split(",") if b.strip()]
    else:
        bags = ["functional", "structural"]
        if include_combined:
            bags.append("combined")
    bad = [b for b in bags if b not in BAG_ORDER]
    if bad:
        raise ValueError(f"Unsupported RIDGE_BAGS values: {bad}")
    return [b for b in BAG_ORDER if b in set(bags)]


def ridge_rungs() -> list[str]:
    requested = os.environ.get("RIDGE_RUNGS", "").strip()
    rungs = [r.strip() for r in requested.split(",") if r.strip()] if requested else list(DEFAULT_RIDGE_RUNGS)
    bad = [r for r in rungs if r not in RIDGE_RUNGS]
    if bad:
        raise ValueError(f"Unsupported RIDGE_RUNGS values: {bad}")
    return rungs


def ridge_alphas() -> np.ndarray:
    raw = os.environ.get("RIDGE_ALPHAS", "").strip()
    if raw:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if not vals:
            raise ValueError("RIDGE_ALPHAS was provided but no numeric values were parsed.")
        return np.asarray(vals, dtype=float)
    return np.logspace(-3, 4, 8)


def ridge_analysis_cfg(include_diagnosis: bool) -> dict:
    cfg = analysis_cfg_from_config(load_sensitivity_config())
    cfg["exclude_countries"] = list(VARIANT_A_EXCLUDE_COUNTRIES)
    cfg["exclude_diagnosis"] = list(VARIANT_A_EXCLUDE_DIAGNOSIS)
    cfg["include_diagnosis"] = bool(include_diagnosis)
    cfg["include_year"] = True
    cfg["year_effect_mode"] = "spline"
    cfg["year_spline_df"] = 4
    return cfg


def ridge_candidate_pool(bag: str, *, smoke: bool = False) -> pd.DataFrame:
    if os.environ.get("RIDGE_BASELINE_ONLY", "").strip().lower() in {"1", "true", "yes", "y"}:
        out = baseline_candidate_df().copy()
        out["ridge_order_max"] = 0
        out["ridge_top_k_per_order"] = 0
        return out.reset_index(drop=True)
    order_max = int(os.environ.get("RIDGE_ORDER_MAX", "30"))
    if "ridge_triple" in set(ridge_rungs()):
        order_max = int(os.environ.get("RIDGE_ORDER_MAX_TRIPLE", str(order_max)))
    top_k = int(os.environ.get("RIDGE_TOP_K_PER_ORDER", "20"))
    pool = load_fig2_candidate_pool(bag)
    pool["order"] = pd.to_numeric(pool["order"], errors="coerce")
    pool["score"] = pd.to_numeric(pool["score"], errors="coerce")
    pool = pool[pool["objective"].astype(str).isin(["o_min", "o_max"])].copy()
    pool = pool[(pool["order"] >= 3) & (pool["order"] <= order_max)].copy()
    rows = []
    for (objective, order), sub in pool.groupby(["objective", "order"], sort=True, observed=True):
        asc = str(objective) == "o_min"
        rows.append(sub.sort_values("score", ascending=asc).head(top_k))
    out = pd.concat(rows, ignore_index=True) if rows else pool.head(0).copy()
    out["ridge_order_max"] = order_max
    out["ridge_top_k_per_order"] = top_k
    if smoke:
        smoke_orders = sorted(pd.to_numeric(out["order"], errors="coerce").dropna().astype(int).unique().tolist())[:2]
        out = out[out["order"].astype(int).isin(smoke_orders)].copy()
        out = out.groupby(["objective", "order"], group_keys=False, observed=True).head(1).reset_index(drop=True)
    out = pd.concat([out, baseline_candidate_df()], ignore_index=True)
    return out.reset_index(drop=True)


def ridge_families() -> list[RidgeFamily]:
    return [
        RidgeFamily("pooled", None, None, True),
        RidgeFamily("cn_norm", ("CN",), ("CN",), False),
        RidgeFamily("cn_norm", ("CN",), ("AD",), False),
        RidgeFamily("cn_norm", ("CN",), ("FTD",), False),
        RidgeFamily("ad_norm", ("AD",), ("AD",), False),
        RidgeFamily("ad_norm", ("AD",), ("CN",), False),
        RidgeFamily("ad_norm", ("AD",), ("FTD",), False),
        RidgeFamily("ftd_norm", ("FTD",), ("FTD",), False),
        RidgeFamily("ftd_norm", ("FTD",), ("AD",), False),
        RidgeFamily("ftd_norm", ("FTD",), ("CN",), False),
        RidgeFamily("adftd_norm", ("AD", "FTD"), ("AD",), True),
        RidgeFamily("adftd_norm", ("AD", "FTD"), ("FTD",), True),
        RidgeFamily("adftd_norm", ("AD", "FTD"), ("CN",), True, True),
    ]


def _as_dx_mask(values: np.ndarray, group: tuple[str, ...] | None) -> np.ndarray:
    if group is None:
        return np.isin(values.astype(str), PRIMARY_DX)
    return np.isin(values.astype(str), list(group))


def prepare_ridge_context(
    model_df: pd.DataFrame,
    bag: str,
    exposome_cols: list[str],
    analysis_cfg: dict,
    family: RidgeFamily,
    *,
    cv_override: dict | None = None,
) -> dict:
    keep_dx = set(PRIMARY_DX)
    if family.train_dx is not None:
        keep_dx.update(family.train_dx)
    if family.test_dx is not None:
        keep_dx.update(family.test_dx)
    subset = model_df[model_df["Diagnosis"].astype(str).isin(sorted(keep_dx))].copy()
    cv = {**cv_cfg(), **(cv_override or {})}
    y_col = target_map(bag)[bag]
    context = prepare_bag_context(subset, y_col, bag, analysis_cfg, cv, exposome_cols)
    diag = np.asarray(context["diag"]).astype(str)
    split = np.asarray(context["country"]).astype(str)
    train_mask = _as_dx_mask(diag, family.train_dx)
    test_mask = _as_dx_mask(diag, family.test_dx)
    countries = []
    train_idx_by_country = {}
    test_idx_by_country = {}
    min_train = int(analysis_cfg.get("min_n_obs", 80))
    for country in sorted(pd.Series(split).dropna().astype(str).unique().tolist()):
        tr = np.where((split != country) & train_mask)[0]
        te = np.where((split == country) & test_mask)[0]
        if len(tr) >= min_train and len(te) > 0:
            countries.append(country)
            train_idx_by_country[country] = tr
            test_idx_by_country[country] = te
    context["countries"] = countries
    context["train_idx_by_country"] = train_idx_by_country
    context["test_idx_by_country"] = test_idx_by_country
    context["year_basis_by_country"] = prepare_year_basis_by_fold(
        context["year"], countries, train_idx_by_country, analysis_cfg
    )
    return context


def _predictor_indices(row: dict, exposome_cols: list[str]) -> list[int]:
    col_to_idx = {c: i for i, c in enumerate(exposome_cols)}
    names = parse_literal_list(row.get("nplet_vars", []))
    return [col_to_idx[n] for n in names if n in col_to_idx]


def _expand_exposome(z: np.ndarray, rung_id: str) -> tuple[np.ndarray, dict]:
    z = np.asarray(z, dtype=np.float32)
    k = int(z.shape[1])
    parts = [z]
    n_pair = 0
    n_triple = 0
    if rung_id in {"ridge_pairwise", "ridge_triple"} and k >= 2:
        pairs = [z[:, i] * z[:, j] for i in range(k) for j in range(i + 1, k)]
        parts.append(np.column_stack(pairs).astype(np.float32))
        n_pair = len(pairs)
    if rung_id == "ridge_triple" and k >= 3:
        triples = [
            z[:, i] * z[:, j] * z[:, l]
            for i in range(k)
            for j in range(i + 1, k)
            for l in range(j + 1, k)
        ]
        parts.append(np.column_stack(triples).astype(np.float32))
        n_triple = len(triples)
    expanded = np.hstack(parts).astype(np.float32) if parts else np.zeros((len(z), 0), dtype=np.float32)
    return expanded, {"n_main": k, "n_pair": n_pair, "n_triple": n_triple, "n_expanded": int(expanded.shape[1])}


def _expand_baseline(x: np.ndarray, rung_id: str) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float32)
    info = {"baseline_main": max(int(x.shape[1]) - 1, 0), "baseline_pair": 0, "baseline_triple": 0}
    if rung_id not in {"baseline_pairwise", "baseline_triple"} or x.shape[1] <= 2:
        info["baseline_expanded"] = int(x.shape[1])
        return x, info
    z = x[:, 1:]
    k = int(z.shape[1])
    parts = [x]
    pairs = [z[:, i] * z[:, j] for i in range(k) for j in range(i + 1, k)]
    if pairs:
        parts.append(np.column_stack(pairs).astype(np.float32))
        info["baseline_pair"] = len(pairs)
    if rung_id == "baseline_triple" and k >= 3:
        triples = [
            z[:, i] * z[:, j] * z[:, l]
            for i in range(k)
            for j in range(i + 1, k)
            for l in range(j + 1, k)
        ]
        if triples:
            parts.append(np.column_stack(triples).astype(np.float32))
            info["baseline_triple"] = len(triples)
    expanded = np.hstack(parts).astype(np.float32)
    info["baseline_expanded"] = int(expanded.shape[1])
    return expanded, info


def _baseline_parts(context: dict, country: str, fit_idx: np.ndarray, apply_idx: np.ndarray) -> tuple[list, list, np.ndarray, np.ndarray, bool]:
    cfg = context["analysis_cfg"]
    age = np.asarray(context["age"], dtype=float)
    sex = np.asarray(context["sex"]).astype(str)
    diag = np.asarray(context["diag"]).astype(str)
    parts_fit = [np.ones((len(fit_idx), 1), dtype=np.float32), age[fit_idx].reshape(-1, 1).astype(np.float32)]
    parts_apply = [np.ones((len(apply_idx), 1), dtype=np.float32), age[apply_idx].reshape(-1, 1).astype(np.float32)]
    if cfg.get("include_sex", True):
        s_fit, s_apply, levels = encode_dummies(sex[fit_idx], sex[apply_idx])
        if len(levels) > 1:
            parts_fit.append(s_fit.astype(np.float32))
            parts_apply.append(s_apply.astype(np.float32))
    if cfg.get("include_year", True):
        yb = context["year_basis_by_country"][country]
        parts_fit.append(yb[fit_idx].astype(np.float32))
        parts_apply.append(yb[apply_idx].astype(np.float32))
    diag_used = False
    d_fit = np.zeros((len(fit_idx), 0), dtype=np.float32)
    d_apply = np.zeros((len(apply_idx), 0), dtype=np.float32)
    if cfg.get("include_diagnosis", True):
        d_fit, d_apply, levels = encode_dummies(diag[fit_idx], diag[apply_idx])
        d_fit = d_fit.astype(np.float32)
        d_apply = d_apply.astype(np.float32)
        if len(levels) > 1 and d_fit.shape[1] > 0:
            diag_used = True
            parts_fit.append(d_fit)
            parts_apply.append(d_apply)
    return parts_fit, parts_apply, d_fit, d_apply, diag_used


def _build_ridge_design(
    context: dict,
    country: str,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    x_exp: np.ndarray,
    pred_used: list[int],
    rung_id: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    parts_fit, parts_apply, d_fit, d_apply, diag_used = _baseline_parts(context, country, fit_idx, apply_idx)
    exp_info = {"n_main": 0, "n_pair": 0, "n_triple": 0, "n_expanded": 0}
    baseline_info = {"baseline_main": np.nan, "baseline_pair": np.nan, "baseline_triple": np.nan, "baseline_expanded": np.nan}
    if rung_id in {"baseline_main", "baseline_pairwise", "baseline_triple"}:
        x_fit = np.hstack(parts_fit).astype(np.float32)
        x_apply = np.hstack(parts_apply).astype(np.float32)
        x_fit, baseline_info = _expand_baseline(x_fit, rung_id)
        x_apply, _ = _expand_baseline(x_apply, rung_id)
        pred_used = []
        parts_fit = [x_fit]
        parts_apply = [x_apply]
    if pred_used:
        raw_fit = x_exp[np.ix_(fit_idx, pred_used)].astype(np.float32)
        raw_apply = x_exp[np.ix_(apply_idx, pred_used)].astype(np.float32)
        scaler = StandardScaler()
        z_fit = scaler.fit_transform(raw_fit).astype(np.float32)
        z_apply = scaler.transform(raw_apply).astype(np.float32)
        e_fit, exp_info = _expand_exposome(z_fit, rung_id)
        e_apply, _ = _expand_exposome(z_apply, rung_id)
        parts_fit.append(e_fit)
        parts_apply.append(e_apply)
        if diag_used and e_fit.shape[1] > 0:
            parts_fit.append((d_fit[:, :, None] * e_fit[:, None, :]).reshape(len(fit_idx), -1))
            parts_apply.append((d_apply[:, :, None] * e_apply[:, None, :]).reshape(len(apply_idx), -1))
    x_fit = np.hstack(parts_fit).astype(np.float32)
    x_apply = np.hstack(parts_apply).astype(np.float32)
    # Penalize the complete design, but standardize non-intercept columns so alpha
    # has comparable meaning across covariates, main effects, and product terms.
    if x_fit.shape[1] > 1:
        mu = x_fit[:, 1:].mean(axis=0)
        sd = x_fit[:, 1:].std(axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32)
        x_fit[:, 1:] = (x_fit[:, 1:] - mu) / sd
        x_apply[:, 1:] = (x_apply[:, 1:] - mu) / sd
    exp_info["p_raw"] = int(x_fit.shape[1])
    exp_info["diag_interaction_cols"] = int(exp_info["n_expanded"] * d_fit.shape[1]) if diag_used else 0
    exp_info["diagnosis_interactions_used"] = bool(diag_used and exp_info["n_expanded"] > 0)
    exp_info.update(baseline_info)
    return x_fit, x_apply, exp_info


def _drop_zero_var(x_train: np.ndarray, x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if x_train.shape[1] == 0:
        return x_train, x_apply, 0
    keep = np.ones(x_train.shape[1], dtype=bool)
    if x_train.shape[1] > 1:
        var = np.nanvar(x_train[:, 1:], axis=0)
        keep[1:] = np.isfinite(var) & (var > 0)
    dropped = int((~keep).sum())
    return x_train[:, keep], x_apply[:, keep], dropped


def _effective_df(x_train: np.ndarray, alpha: float) -> float:
    max_cols = int(os.environ.get("RIDGE_DF_MAX_COLS", "2500"))
    if x_train.shape[1] > max_cols:
        return float("nan")
    try:
        s = np.linalg.svd(x_train, compute_uv=False)
        return float(np.sum((s * s) / ((s * s) + float(alpha))))
    except Exception:
        return float("nan")


def _fit_one_candidate(
    row: dict,
    *,
    context: dict,
    exposome_cols: list[str],
    rung_id: str,
    alphas: np.ndarray,
    family: RidgeFamily,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_id = str(row["candidate_id"])
    pred_all = _predictor_indices(row, exposome_cols)
    y = np.asarray(context["y"], dtype=float)
    x_exp = np.asarray(context["X_exp"], dtype=np.float32)
    y_true_all = []
    y_pred_all = []
    country_rows = []
    alpha_rows = []
    complexity_rows = []

    for fold_i, country in enumerate(context["countries"]):
        train_idx = np.asarray(context["train_idx_by_country"][country], dtype=int)
        test_idx = np.asarray(context["test_idx_by_country"][country], dtype=int)
        tr_inner, val_idx = _split_train_val_by_country(
            context["country"], train_idx, seed=20260304 + fold_i
        )
        pred_used = list(pred_all)
        pred_nonmissing = np.ones(len(y), dtype=bool)
        if pred_used:
            pred_nonmissing = context["m_exp"][:, pred_used].all(axis=1)
        train_ok = context["base_req_train"] & pred_nonmissing
        test_ok = context["base_req_test"] & pred_nonmissing
        train_idx = train_idx[train_ok[train_idx]]
        test_idx = test_idx[test_ok[test_idx]]
        if pred_used:
            var = np.nanvar(x_exp[np.ix_(train_idx, pred_used)], axis=0)
            pred_used = [idx for idx, keep in zip(pred_used, var > 0) if bool(keep)]
            pred_nonmissing = context["m_exp"][:, pred_used].all(axis=1) if pred_used else np.ones(len(y), dtype=bool)
            train_ok = context["base_req_train"] & pred_nonmissing
            test_ok = context["base_req_test"] & pred_nonmissing
            train_idx = np.asarray(context["train_idx_by_country"][country], dtype=int)
            test_idx = np.asarray(context["test_idx_by_country"][country], dtype=int)
            train_idx = train_idx[train_ok[train_idx]]
            test_idx = test_idx[test_ok[test_idx]]
        if len(train_idx) < int(context["analysis_cfg"].get("min_n_obs", 80)):
            tr_inner = np.asarray([], dtype=int)
            val_idx = np.asarray([], dtype=int)
        else:
            tr_inner, val_idx = _split_train_val_by_country(
                context["country"], train_idx, seed=20260304 + fold_i
            )

        y_pred = np.full(len(test_idx), np.nan, dtype=float)
        selected_alpha = float("nan")
        validation_rmse = float("nan")
        validation_r2 = float("nan")
        p_raw = p_used = dropped = 0
        exp_info: dict = {}
        eff_df = float("nan")
        status = "ok"
        error = ""
        try:
            x_inner, x_val, exp_info = _build_ridge_design(
                context, country, tr_inner, val_idx, x_exp, pred_used, rung_id
            )
            x_inner, x_val, dropped_inner = _drop_zero_var(x_inner, x_val)
            best = None
            for alpha in alphas:
                reg = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr", max_iter=1000, tol=1e-3)
                reg.fit(x_inner, y[tr_inner])
                val_pred = reg.predict(x_val)
                ok_val = np.isfinite(y[val_idx]) & np.isfinite(val_pred)
                met_val = regression_metrics_extended(
                    y[val_idx][ok_val],
                    val_pred[ok_val],
                    min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
                )
                score = met_val["rmse"]
                if best is None or (np.isfinite(score) and score < best[0]):
                    best = (score, float(alpha), met_val)
            if best is None:
                raise RuntimeError("alpha_selection_failed")
            selected_alpha = float(best[1])
            validation_rmse = float(best[2]["rmse"])
            validation_r2 = float(best[2]["r2"])
            x_train, x_test, exp_info = _build_ridge_design(
                context, country, train_idx, test_idx, x_exp, pred_used, rung_id
            )
            p_raw = int(x_train.shape[1])
            x_train, x_test, dropped = _drop_zero_var(x_train, x_test)
            p_used = int(x_train.shape[1])
            eff_df = _effective_df(x_train, selected_alpha)
            reg = Ridge(alpha=selected_alpha, fit_intercept=False, solver="lsqr", max_iter=1000, tol=1e-3)
            reg.fit(x_train, y[train_idx])
            y_pred = reg.predict(x_test)
            dropped += dropped_inner
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            y_pred = np.full(len(test_idx), np.nan, dtype=float)

        y_true = y[test_idx]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        met = regression_metrics_extended(
            y_true[ok], y_pred[ok], min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5))
        )
        base = {
            "candidate_id": model_id,
            "rung_id": rung_id,
            "family_id": family.family_id,
            "train_dx": family.train_label,
            "test_dx": family.test_label,
            "fold_country": country,
            "unseen_dx_reference_encoded": bool(family.unseen_dx_reference_encoded),
        }
        country_rows.append(
            {
                **base,
                "n_test": int(ok.sum()),
                "r2": met["r2"],
                "rmse": met["rmse"],
                "mae": met["mae"],
                "corr2": met["corr2"],
                "status": status,
                "error": error,
            }
        )
        alpha_rows.append(
            {
                **base,
                "alpha": selected_alpha,
                "validation_rmse": validation_rmse,
                "validation_r2": validation_r2,
                "n_inner": int(len(tr_inner)),
                "n_val": int(len(val_idx)),
            }
        )
        complexity_rows.append(
            {
                **base,
                "p_raw": int(p_raw),
                "p_used": int(p_used),
                "p_dropped_zero_var": int(dropped),
                "effective_df": eff_df,
                "predictors_used_n": int(len(pred_used)),
                **{k: exp_info.get(k, np.nan) for k in [
                    "n_main",
                    "n_pair",
                    "n_triple",
                    "n_expanded",
                    "diag_interaction_cols",
                    "diagnosis_interactions_used",
                    "baseline_main",
                    "baseline_pair",
                    "baseline_triple",
                    "baseline_expanded",
                ]},
            }
        )

    yt = np.concatenate(y_true_all) if y_true_all else np.asarray([])
    yp = np.concatenate(y_pred_all) if y_pred_all else np.asarray([])
    ok = np.isfinite(yt) & np.isfinite(yp)
    g = regression_metrics_extended(
        yt[ok], yp[ok], min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5))
    )
    summary = {
        "candidate_id": model_id,
        "rung_id": rung_id,
        "family_id": family.family_id,
        "train_dx": family.train_label,
        "test_dx": family.test_label,
        "global_oof_r2": g["r2"],
        "global_rmse": g["rmse"],
        "global_mae": g["mae"],
        "global_corr2": g["corr2"],
        "n_scored": int(g["n_scored"]),
        "predictors_used_n": int(len(pred_all)),
        "unseen_dx_reference_encoded": bool(family.unseen_dx_reference_encoded),
    }
    return summary, pd.DataFrame(country_rows), pd.DataFrame(alpha_rows), pd.DataFrame(complexity_rows)


def _write_delta_pairwise(summary: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    if summary.empty:
        delta = pd.DataFrame()
    else:
        keys = ["candidate_id", "family_id", "train_dx", "test_dx"]
        wide = summary.pivot_table(index=keys, columns="rung_id", values="global_oof_r2", aggfunc="first").reset_index()
        if {"ridge_main", "ridge_pairwise"}.issubset(wide.columns):
            wide["delta_pairwise_minus_main"] = wide["ridge_pairwise"] - wide["ridge_main"]
        delta = wide
    delta.to_csv(outdir / "delta_pairwise_minus_main.csv", index=False)
    return delta


def evaluate_ridge_candidates(
    *,
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    bag: str,
    family: RidgeFamily,
    rungs: list[str],
    analysis_cfg: dict,
    outdir: Path,
    n_jobs: int,
    cv_override: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    context = prepare_ridge_context(model_df, bag, exposome_cols, analysis_cfg, family, cv_override=cv_override)
    if not context["countries"]:
        raise ValueError(f"No valid folds for bag={bag} family={family}")
    alphas = ridge_alphas()
    candidate_df = candidate_df.copy().reset_index(drop=True)
    candidate_df["predictors_identity"] = candidate_df["predictors_identity"].fillna("").astype(str)
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
    records = fit_df.to_dict(orient="records")
    summary_parts = []
    country_parts = []
    alpha_parts = []
    complexity_parts = []
    # Candidate-level tasks have uneven runtimes across orders. Keep the chunk
    # larger than RIDGE_N_JOBS=40 so each loky batch has enough queued work to
    # avoid idle workers and straggler-driven under-utilization.
    chunk_size = int(os.environ.get("RIDGE_EVAL_CHUNK_SIZE", "120"))
    for rung_id in rungs:
        global_path = outdir / f"{bag}_{family.family_id}_{family.test_label}_{rung_id}_global.csv"
        country_path = outdir / f"{bag}_{family.family_id}_{family.test_label}_{rung_id}_country.csv"
        alpha_path = outdir / f"{bag}_{family.family_id}_{family.test_label}_{rung_id}_alpha.csv"
        complexity_path = outdir / f"{bag}_{family.family_id}_{family.test_label}_{rung_id}_complexity.csv"
        if all(p.exists() for p in [global_path, country_path, alpha_path, complexity_path]):
            existing = pd.read_csv(global_path)
            if len(existing) == len(candidate_df):
                summary_parts.append(existing)
                country_parts.append(pd.read_csv(country_path))
                alpha_parts.append(pd.read_csv(alpha_path))
                complexity_parts.append(pd.read_csv(complexity_path))
                log_msg(f"Ridge resume hit bag={bag} family={family.family_id} test={family.test_label} rung={rung_id}")
                continue
        log_msg(
            f"Ridge eval start bag={bag} family={family.family_id} train={family.train_label} "
            f"test={family.test_label} rung={rung_id} candidates={len(candidate_df)} "
            f"unique={len(records)} folds={len(context['countries'])} n_jobs={n_jobs}"
        )
        fitted = []
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            if int(n_jobs) <= 1:
                fitted_chunk = [
                    _fit_one_candidate(
                        row,
                        context=context,
                        exposome_cols=exposome_cols,
                        rung_id=rung_id,
                        alphas=alphas,
                        family=family,
                    )
                    for row in chunk
                ]
            else:
                fitted_chunk = Parallel(n_jobs=int(n_jobs), backend="loky", batch_size=1)(
                    delayed(_fit_one_candidate)(
                        row,
                        context=context,
                        exposome_cols=exposome_cols,
                        rung_id=rung_id,
                        alphas=alphas,
                        family=family,
                    )
                    for row in chunk
                )
            fitted.extend(fitted_chunk)
            log_msg(
                f"Ridge eval progress bag={bag} family={family.family_id} test={family.test_label} "
                f"rung={rung_id} unique_done={min(start + len(chunk), len(records))}/{len(records)}"
            )
        fit_summary = pd.DataFrame([x[0] for x in fitted])
        fit_country = pd.concat([x[1] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
        fit_alpha = pd.concat([x[2] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
        fit_complexity = pd.concat([x[3] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
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
        rung_summary = metadata.merge(fit_summary[metric_cols], on="predictors_identity", how="left")
        rung_summary.to_csv(global_path, index=False)
        fit_country.to_csv(country_path, index=False)
        fit_alpha.to_csv(alpha_path, index=False)
        fit_complexity.to_csv(complexity_path, index=False)
        summary_parts.append(rung_summary)
        country_parts.append(fit_country)
        alpha_parts.append(fit_alpha)
        complexity_parts.append(fit_complexity)
        log_msg(f"Ridge eval complete bag={bag} family={family.family_id} test={family.test_label} rung={rung_id}")

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    country = pd.concat(country_parts, ignore_index=True) if country_parts else pd.DataFrame()
    alpha = pd.concat(alpha_parts, ignore_index=True) if alpha_parts else pd.DataFrame()
    complexity = pd.concat(complexity_parts, ignore_index=True) if complexity_parts else pd.DataFrame()
    summary.to_csv(outdir / f"{bag}_{family.family_id}_{family.test_label}_global_all.csv", index=False)
    country.to_csv(outdir / f"{bag}_{family.family_id}_{family.test_label}_country_all.csv", index=False)
    alpha.to_csv(outdir / f"{bag}_{family.family_id}_{family.test_label}_alpha_all.csv", index=False)
    complexity.to_csv(outdir / f"{bag}_{family.family_id}_{family.test_label}_complexity_all.csv", index=False)
    _write_delta_pairwise(summary, outdir)
    return summary, country, alpha, complexity


def build_ridge_model_df() -> tuple[pd.DataFrame, list[str]]:
    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    model_df = build_original_model_df(raw, feature_names, ridge_analysis_cfg(include_diagnosis=True))
    return model_df, feature_names


def copy_ridge_outputs(local_root: Path) -> None:
    """Deprecated compatibility shim.

    Ridge eval artifacts are now written directly to the external eval root. This
    function intentionally does nothing unless explicitly requested for a legacy
    local-to-bundle copy workflow.
    """
    if os.environ.get("RIDGE_COPY_LOCAL_TO_BUNDLE", "").strip().lower() not in {"1", "true", "yes", "y"}:
        return
    root = ridge_eval_root(local_root)
    copy_tree_contents(local_root, root)
