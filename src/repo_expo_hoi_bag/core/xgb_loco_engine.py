from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from loco_fusion_matrix_engine import (
    atomic_write_csv,
    build_dual_rankings,
    build_model_vs_baseline_tables,
    compute_paired_delta_tables,
    expand_representatives_to_candidates,
    make_candidate_identity_map,
    nanmean_safe,
    prepare_bag_context,
    r2_from_predictions,
    regression_metrics_extended,
)


def require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise ImportError(
            "xgboost is required for this notebook. Install with `pip install xgboost` "
            "in the active kernel environment and rerun."
        ) from e
    return xgb


def _as_clean_str(arr):
    s = pd.Series(arr)
    s = s.astype(str).str.strip()
    s = s.replace('nan', '')
    return s.to_numpy()


def _one_hot_train_apply(train_vals, apply_vals):
    tr = _as_clean_str(train_vals)
    ap = _as_clean_str(apply_vals)
    levels = sorted(pd.Series(tr).unique().tolist())
    if len(levels) == 0:
        return np.zeros((len(tr), 0), dtype=np.float32), np.zeros((len(ap), 0), dtype=np.float32), []
    Xtr = np.column_stack([(tr == lv).astype(np.float32) for lv in levels])
    Xap = np.column_stack([(ap == lv).astype(np.float32) for lv in levels])
    return Xtr, Xap, levels


def _split_train_val_by_country(country_arr, train_idx, val_country_frac=0.20, val_country_min=1, seed=20260304):
    train_idx = np.asarray(train_idx, dtype=int)
    c_tr = np.asarray(country_arr)[train_idx]
    uniq = sorted(pd.Series(c_tr).astype(str).unique().tolist())

    if len(uniq) > 1:
        n_val_c = int(math.ceil(float(val_country_frac) * len(uniq)))
        n_val_c = max(int(val_country_min), n_val_c)
        n_val_c = min(n_val_c, len(uniq) - 1)

        rs = np.random.RandomState(int(seed) % (2**31 - 1))
        perm = rs.permutation(len(uniq))
        val_c = set([uniq[i] for i in perm[:n_val_c]])

        val_mask = np.isin(c_tr, list(val_c))
        val_idx = train_idx[val_mask]
        tr_inner_idx = train_idx[~val_mask]
        if len(tr_inner_idx) > 0 and len(val_idx) > 0:
            return tr_inner_idx, val_idx

    # Fallback when country-level split is impossible.
    if len(train_idx) <= 2:
        cut = 1
    else:
        cut = max(1, int(round(float(val_country_frac) * len(train_idx))))
        cut = min(cut, len(train_idx) - 1)

    rs = np.random.RandomState((int(seed) + 17) % (2**31 - 1))
    perm = rs.permutation(len(train_idx))
    val_pos = perm[:cut]
    tr_pos = perm[cut:]
    return train_idx[tr_pos], train_idx[val_pos]


def _stack_parts(parts):
    if len(parts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    X = np.hstack(parts)
    return np.asarray(X, dtype=np.float32)


def _build_fold_mats_from_indices(
    context,
    fold_country,
    train_idx,
    test_idx,
    analysis_cfg,
    early_stop_cfg,
    seed,
):
    """Build canonical covariate matrices for an explicit train/test split.

    The early-stopping split is drawn only from ``train_idx``.  This is used by
    both outer LOCO evaluation and nested tuning, where the inner scored country
    must be absent from the early-stopping pool.
    """
    age = np.asarray(context['age'], dtype=float)
    year = np.asarray(context['year'], dtype=float)
    sex = context['sex']
    diag = context['diag']
    edu = np.asarray(context.get('edu', np.full(len(age), np.nan)), dtype=float)
    scanner = context.get('scanner', np.asarray([''] * len(age)))

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    tr_inner_idx, val_idx = _split_train_val_by_country(
        context['country'],
        train_idx,
        val_country_frac=float(early_stop_cfg.get('val_country_frac', 0.20)),
        val_country_min=int(early_stop_cfg.get('val_country_min', 1)),
        seed=int(seed),
    )

    def _oh_apply(vals, levels):
        vv = _as_clean_str(vals)
        if len(levels) == 0:
            return np.zeros((len(vv), 0), dtype=np.float32)
        return np.column_stack([(vv == lv).astype(np.float32) for lv in levels]).astype(np.float32)

    # Keep feature-space consistent for this fold: categorical levels are learned on inner-train only.
    sex_levels = []
    diag_levels = []
    scanner_levels = []
    if bool(analysis_cfg.get('include_sex', True)):
        sex_levels = sorted(pd.Series(_as_clean_str(sex[tr_inner_idx])).unique().tolist())
    if bool(analysis_cfg.get('include_diagnosis', True)):
        diag_levels = sorted(pd.Series(_as_clean_str(diag[tr_inner_idx])).unique().tolist())
    if bool(analysis_cfg.get('include_scanner', False)):
        scanner_levels = sorted(pd.Series(_as_clean_str(scanner[tr_inner_idx])).unique().tolist())

    def _build(idx_apply):
        parts_apply = [age[idx_apply].reshape(-1, 1).astype(np.float32)]

        if bool(analysis_cfg.get('include_year', True)):
            parts_apply.append(year[idx_apply].reshape(-1, 1).astype(np.float32))

        if bool(analysis_cfg.get('include_sex', True)) and len(sex_levels) > 0:
            parts_apply.append(_oh_apply(sex[idx_apply], sex_levels))

        if bool(analysis_cfg.get('include_diagnosis', True)) and len(diag_levels) > 0:
            parts_apply.append(_oh_apply(diag[idx_apply], diag_levels))

        if bool(analysis_cfg.get('include_education', False)):
            parts_apply.append(edu[idx_apply].reshape(-1, 1).astype(np.float32))

        if bool(analysis_cfg.get('include_scanner', False)) and len(scanner_levels) > 0:
            parts_apply.append(_oh_apply(scanner[idx_apply], scanner_levels))

        return _stack_parts(parts_apply)

    X_tr_inner = _build(tr_inner_idx)
    X_val = _build(val_idx)
    X_tr_full = _build(train_idx)
    X_test = _build(test_idx)

    return {
        'fold_country': fold_country,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'tr_inner_idx': tr_inner_idx,
        'val_idx': val_idx,
        'Xb_train_inner': X_tr_inner,
        'Xb_val': X_val,
        'Xb_train_full': X_tr_full,
        'Xb_test': X_test,
    }


def _build_base_fold_mats(context, fold_country, analysis_cfg, early_stop_cfg, seed):
    """Backward-compatible outer-LOCO wrapper around explicit split matrices."""
    return _build_fold_mats_from_indices(
        context,
        fold_country,
        context['train_idx_by_country'][fold_country],
        context['test_idx_by_country'][fold_country],
        analysis_cfg,
        early_stop_cfg,
        seed,
    )


def _fit_xgb_fold(
    xgb,
    xgb_params,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    sample_weight=None,
    sample_weight_val=None,
):
    xgb_params = dict(xgb_params)
    if "missing" in xgb_params:
        missing = xgb_params["missing"]
        if missing is None or (isinstance(missing, float) and np.isnan(missing)):
            xgb_params.pop("missing")
        elif isinstance(missing, str):
            if missing.strip().lower() in {"nan", "none", ""}:
                xgb_params.pop("missing")
            else:
                try:
                    xgb_params["missing"] = float(missing)
                except ValueError:
                    raise ValueError(
                        f"Invalid XGBoost missing value: {missing!r}. "
                        "Use a numeric value, None, or omit the parameter."
                    )
    reg = xgb.XGBRegressor(**xgb_params)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if X_val is not None and len(X_val) > 0 and len(y_val) > 0:
        if sample_weight_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        reg.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            **fit_kwargs,
        )
    else:
        reg.fit(X_train, y_train, verbose=False, **fit_kwargs)
    return reg


def _append_predictors(Xbase, Xexp, idx, pred_idx):
    if len(pred_idx) == 0:
        return Xbase
    Xe = np.asarray(Xexp[np.ix_(idx, pred_idx)], dtype=np.float32)
    if Xbase.shape[0] == 0:
        return Xe
    return np.hstack([Xbase, Xe]).astype(np.float32)


def baseline_cache_signature(context, fold_designs, xgb_cfg, fold_xgb_cfgs=None) -> str:
    """Content-addressed key for the covariate-only XGB baseline.

    The baseline OOF depends only on the bag target, the per-fold covariate
    design matrices, the fold partition, the targets, and the XGB config -- it
    never touches the candidate predictors (those enter only in
    `evaluate_xgb_representatives_parallel`). Hashing exactly the inputs the
    baseline fit consumes yields a key that is stable across candidate sets but
    changes whenever anything the fit actually reads changes, so a cache hit can
    never be stale. This is intentionally content-addressed rather than a row
    count, to avoid the stale-but-complete reuse the audit flagged for
    presence/row-count resume checks.
    """
    h = hashlib.sha256()
    h.update(repr(context['bag_name']).encode())
    h.update(json.dumps(xgb_cfg, sort_keys=True, default=str).encode())
    h.update(json.dumps(fold_xgb_cfgs or {}, sort_keys=True, default=str).encode())
    y = np.ascontiguousarray(np.asarray(context['y'], dtype=float))
    h.update(b'y')
    h.update(str(y.shape).encode())
    h.update(y.tobytes())
    for fold_i, c in enumerate(context['countries']):
        fd = fold_designs[c]
        h.update(f'fold:{fold_i}:{c}'.encode())
        for key in ('Xb_train_inner', 'Xb_val', 'Xb_test', 'Xb_train_full'):
            arr = np.ascontiguousarray(np.asarray(fd[key], dtype=float))
            h.update(key.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        for idx_key in ('tr_inner_idx', 'val_idx', 'train_idx', 'test_idx'):
            idx = np.ascontiguousarray(np.asarray(fd[idx_key]))
            h.update(idx_key.encode())
            h.update(str(idx.shape).encode())
            h.update(idx.tobytes())
    return h.hexdigest()


def load_cached_baseline(cache_dir, signature):
    """Return ``(summary_dict, pred_df, country_df, fold_cache)`` or ``None``.

    Any missing/corrupt artifact yields ``None`` so the caller transparently
    refits. Returns the same shapes as :func:`fit_xgb_baseline_across_folds`.
    """
    base = Path(cache_dir) / signature
    summary_path = base / 'base_summary.json'
    pred_path = base / 'base_pred.parquet'
    country_path = base / 'base_country.parquet'
    fold_cache_path = base / 'baseline_fold_cache.parquet'
    if not all(p.exists() for p in (summary_path, pred_path, country_path, fold_cache_path)):
        return None
    try:
        bsum = json.loads(summary_path.read_text())
        bpred = pd.read_parquet(pred_path)
        bcountry = pd.read_parquet(country_path)
        fold_cache_df = pd.read_parquet(fold_cache_path)
        baseline_fold_cache = {
            row['fold_country']: {k: row[k] for k in fold_cache_df.columns}
            for _, row in fold_cache_df.iterrows()
        }
        return bsum, bpred, bcountry, baseline_fold_cache
    except Exception:
        return None


def save_cached_baseline(cache_dir, signature, bsum, bpred, bcountry, baseline_fold_cache) -> None:
    """Persist baseline artifacts under ``cache_dir/signature`` (best effort).

    Written via a temp dir + atomic replace so a crash mid-write cannot leave a
    partial cache entry that would later load as a wrong baseline.
    """
    if bpred is None or len(bpred) == 0:
        return
    base = Path(cache_dir) / signature
    base.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(cache_dir) / f'.{signature}.tmp'
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        (tmp / 'base_summary.json').write_text(json.dumps(bsum, default=str))
        bpred.to_parquet(tmp / 'base_pred.parquet', index=False)
        bcountry.to_parquet(tmp / 'base_country.parquet', index=False)
        fold_cache_df = pd.DataFrame(list(baseline_fold_cache.values()))
        fold_cache_df.to_parquet(tmp / 'baseline_fold_cache.parquet', index=False)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    os.replace(tmp, base)


def fit_xgb_baseline_across_folds(context, fold_designs, xgb_cfg, fold_xgb_cfgs=None):
    xgb = require_xgboost()
    y = np.asarray(context['y'], dtype=float)

    fold_pred_chunks = []
    country_rows = []
    complexity_rows = []
    baseline_fold_cache = {}

    base_seed = int(xgb_cfg.get('random_state', 20260304))

    for fold_i, c in enumerate(context['countries']):
        fd = fold_designs[c]
        tr_inner = fd['tr_inner_idx']
        test_idx = fd['test_idx']
        train_idx = fd['train_idx']
        val_idx = fd['val_idx']

        y_tr_inner = y[tr_inner]
        y_val = y[val_idx]
        y_train_full = y[train_idx]
        y_test = y[test_idx]

        y_pred_test = np.full(len(test_idx), np.nan, dtype=float)
        status = 'ok'
        error = ''

        xgb_params = dict((fold_xgb_cfgs or {}).get(c, xgb_cfg))
        xgb_params['random_state'] = int(xgb_params.get('random_state', base_seed) + fold_i)

        train_r2 = np.nan
        try:
            reg = _fit_xgb_fold(
                xgb,
                xgb_params,
                fd['Xb_train_inner'],
                y_tr_inner,
                fd['Xb_val'],
                y_val,
            )
            y_pred_test = reg.predict(fd['Xb_test'])
            y_pred_train = reg.predict(fd['Xb_train_full'])

            train_r2 = r2_from_predictions(y_train_full, y_pred_train, min_n=2)
        except Exception as e:
            status = 'failed'
            error = f'fit_or_predict_failed::{repr(e)}'

        fold_df = pd.DataFrame({
            'bag_target': context['bag_name'],
            'model_id': '__xgb_baseline_covariates__',
            'fold_country': c,
            'row_id': context['row_id'][test_idx],
            'N_MEGA': context['N_MEGA'][test_idx],
            'country': context['country'][test_idx],
            'y_true': y_test,
            'y_pred': y_pred_test,
            'status': status,
            'error': error,
            'predictors_used_n': int(fd['Xb_train_full'].shape[1]),
            'predictors_unresolved_n': 0,
            'predictors_dropped_zero_var_n': 0,
            'predictors_identity': '',
            'predictors_used': '',
        })
        fold_df['scored'] = np.isfinite(pd.to_numeric(fold_df['y_true'], errors='coerce')) & np.isfinite(pd.to_numeric(fold_df['y_pred'], errors='coerce'))
        fold_pred_chunks.append(fold_df)

        met = regression_metrics_extended(
            fold_df.loc[fold_df['scored'], 'y_true'].values,
            fold_df.loc[fold_df['scored'], 'y_pred'].values,
            min_n=int(context['analysis_cfg'].get('min_n_obs_for_metrics', 5)),
        )
        country_rows.append({
            'bag_target': context['bag_name'],
            'model_id': '__xgb_baseline_covariates__',
            'fold_country': c,
            'n_test_total': int(len(fold_df)),
            'n_test_scored': int(fold_df['scored'].sum()),
            'coverage_pct': float(fold_df['scored'].sum() / max(1, len(fold_df))),
            'r2': met['r2'],
            'rmse': met['rmse'],
            'mae': met['mae'],
            'corr2': met['corr2'],
            'calib_slope': met['calib_slope'],
            'calib_intercept': met['calib_intercept'],
            'bias_mean': met['bias_mean'],
        })

        complexity_rows.append({
            'fold_country': c,
            'train_r2_base': train_r2,
            'k_base': int(fd['Xb_train_full'].shape[1]),
        })

        baseline_fold_cache[c] = complexity_rows[-1]

    pred_df = pd.concat(fold_pred_chunks, ignore_index=True) if fold_pred_chunks else pd.DataFrame()
    country_df = pd.DataFrame(country_rows)

    scored = pred_df[pred_df['scored']]
    g = regression_metrics_extended(
        scored['y_true'].values,
        scored['y_pred'].values,
        min_n=int(context['analysis_cfg'].get('min_n_obs_for_metrics', 5)),
    )

    cdf = pd.DataFrame(complexity_rows)
    train_r2_mean = nanmean_safe(cdf['train_r2_base']) if not cdf.empty else np.nan

    summary = {
        'bag_target': context['bag_name'],
        'model_id': '__xgb_baseline_covariates__',
        'global_oof_r2': g['r2'],
        'global_oof_rmse': g['rmse'],
        'global_oof_mae': g['mae'],
        'global_oof_corr2': g['corr2'],
        'calib_slope': g['calib_slope'],
        'calib_intercept': g['calib_intercept'],
        'bias_mean': g['bias_mean'],
        'n_total_with_y': int(pred_df['y_true'].notna().sum()) if not pred_df.empty else 0,
        'n_scored': int(g['n_scored']),
        'coverage_pct': float(g['n_scored'] / max(1, int(pred_df['y_true'].notna().sum()))) if not pred_df.empty else np.nan,
        'train_r2_adj_mean': np.nan,
        'train_r2_mean': train_r2_mean,
        'train_r2_base_mean': np.nan,
        'train_f2_mean': np.nan,
        'complexity_folds_n': int(len(cdf)) if not cdf.empty else 0,
    }
    return summary, pred_df, country_df, baseline_fold_cache


def fit_xgb_one_rep_across_folds(
    rep_row, context, fold_designs, baseline_fold_cache, xgb_cfg, fold_xgb_cfgs=None
):
    xgb = require_xgboost()
    rep_id = str(rep_row['model_id'])
    pred_idx_all = list(rep_row.get('predictor_idx_list', []))

    y = np.asarray(context['y'], dtype=float)
    X_exp = np.asarray(context['X_exp'], dtype=np.float32)

    fold_pred_chunks = []
    country_rows = []

    train_r2_vals = []
    train_r2_base_vals = []
    train_f2_vals = []

    base_seed = int(xgb_cfg.get('random_state', 20260304))

    for fold_i, c in enumerate(context['countries']):
        fd = fold_designs[c]
        tr_inner = fd['tr_inner_idx']
        test_idx = fd['test_idx']
        train_idx = fd['train_idx']
        val_idx = fd['val_idx']

        y_tr_inner = y[tr_inner]
        y_val = y[val_idx]
        y_train_full = y[train_idx]
        y_test = y[test_idx]

        pred_used_idx = pred_idx_all
        dropped_zero_n = 0

        if len(pred_idx_all) > 0:
            xtr_full = X_exp[np.ix_(train_idx, pred_idx_all)]
            var = np.nanvar(xtr_full, axis=0)
            keep = var > 0
            pred_used_idx = [idx for idx, kk in zip(pred_idx_all, keep) if bool(kk)]
            dropped_zero_n = int(np.sum(~keep))

        y_pred_test = np.full(len(test_idx), np.nan, dtype=float)
        status = 'ok'
        error = ''

        train_r2 = np.nan
        train_r2_base = np.nan
        train_f2 = np.nan

        try:
            X_tr_inner = _append_predictors(fd['Xb_train_inner'], X_exp, tr_inner, pred_used_idx)
            X_val = _append_predictors(fd['Xb_val'], X_exp, val_idx, pred_used_idx)
            X_tr_full = _append_predictors(fd['Xb_train_full'], X_exp, train_idx, pred_used_idx)
            X_test = _append_predictors(fd['Xb_test'], X_exp, test_idx, pred_used_idx)

            xgb_params = dict((fold_xgb_cfgs or {}).get(c, xgb_cfg))
            xgb_params['random_state'] = int(xgb_params.get('random_state', base_seed) + fold_i)

            reg = _fit_xgb_fold(xgb, xgb_params, X_tr_inner, y_tr_inner, X_val, y_val)
            y_pred_test = reg.predict(X_test)
            y_pred_train = reg.predict(X_tr_full)

            train_r2 = r2_from_predictions(y_train_full, y_pred_train, min_n=2)

            b = baseline_fold_cache.get(c, {})
            train_r2_base = b.get('train_r2_base', np.nan)
            if pd.notna(train_r2) and pd.notna(train_r2_base):
                den = float(1.0 - train_r2)
                if den > 1e-12:
                    train_f2 = float((train_r2 - train_r2_base) / den)
        except Exception as e:
            status = 'failed'
            error = f'fit_or_predict_failed::{repr(e)}'

        fold_df = pd.DataFrame({
            'bag_target': context['bag_name'],
            'model_id': rep_id,
            'fold_country': c,
            'row_id': context['row_id'][test_idx],
            'N_MEGA': context['N_MEGA'][test_idx],
            'country': context['country'][test_idx],
            'y_true': y_test,
            'y_pred': y_pred_test,
            'status': status,
            'error': error,
            'predictors_used_n': int(len(pred_used_idx)),
            'predictors_unresolved_n': 0,
            'predictors_dropped_zero_var_n': dropped_zero_n,
            'predictors_identity': str(rep_row.get('predictors_identity', '')),
            'predictors_used': '|'.join(str(int(i)) for i in pred_used_idx),
        })
        fold_df['scored'] = np.isfinite(pd.to_numeric(fold_df['y_true'], errors='coerce')) & np.isfinite(pd.to_numeric(fold_df['y_pred'], errors='coerce'))
        fold_pred_chunks.append(fold_df)

        met = regression_metrics_extended(
            fold_df.loc[fold_df['scored'], 'y_true'].values,
            fold_df.loc[fold_df['scored'], 'y_pred'].values,
            min_n=int(context['analysis_cfg'].get('min_n_obs_for_metrics', 5)),
        )
        country_rows.append({
            'bag_target': context['bag_name'],
            'model_id': rep_id,
            'fold_country': c,
            'n_test_total': int(len(fold_df)),
            'n_test_scored': int(fold_df['scored'].sum()),
            'coverage_pct': float(fold_df['scored'].sum() / max(1, len(fold_df))),
            'r2': met['r2'],
            'rmse': met['rmse'],
            'mae': met['mae'],
            'corr2': met['corr2'],
            'calib_slope': met['calib_slope'],
            'calib_intercept': met['calib_intercept'],
            'bias_mean': met['bias_mean'],
        })

        train_r2_vals.append(train_r2)
        train_r2_base_vals.append(train_r2_base)
        train_f2_vals.append(train_f2)

    pred_df = pd.concat(fold_pred_chunks, ignore_index=True) if fold_pred_chunks else pd.DataFrame()
    country_df = pd.DataFrame(country_rows)

    scored = pred_df[pred_df['scored']]
    g = regression_metrics_extended(
        scored['y_true'].values,
        scored['y_pred'].values,
        min_n=int(context['analysis_cfg'].get('min_n_obs_for_metrics', 5)),
    )

    summary = {
        'bag_target': context['bag_name'],
        'model_id': rep_id,
        'predictors_identity': str(rep_row.get('predictors_identity', '')),
        'predictors_identity_n': int(rep_row.get('predictors_identity_n', 0)),
        'predictors_used_n': pd.to_numeric(pred_df['predictors_used_n'], errors='coerce').max() if not pred_df.empty else np.nan,
        'predictors_dropped_zero_var_n': pd.to_numeric(pred_df['predictors_dropped_zero_var_n'], errors='coerce').max() if not pred_df.empty else np.nan,
        'global_oof_r2': g['r2'],
        'global_oof_rmse': g['rmse'],
        'global_oof_mae': g['mae'],
        'global_oof_corr2': g['corr2'],
        'calib_slope': g['calib_slope'],
        'calib_intercept': g['calib_intercept'],
        'bias_mean': g['bias_mean'],
        'n_total_with_y': int(pred_df['y_true'].notna().sum()) if not pred_df.empty else 0,
        'n_scored': int(g['n_scored']),
        'coverage_pct': float(g['n_scored'] / max(1, int(pred_df['y_true'].notna().sum()))) if not pred_df.empty else np.nan,
        'train_r2_adj_mean': np.nan,
        'train_r2_mean': nanmean_safe(train_r2_vals),
        'train_r2_base_mean': nanmean_safe(train_r2_base_vals),
        'train_f2_mean': nanmean_safe(train_f2_vals),
        'complexity_folds_n': int(len(train_r2_vals)),
    }
    return summary, pred_df, country_df


def evaluate_xgb_representatives_parallel(
    reps_df,
    context,
    fold_designs,
    baseline_fold_cache,
    perf_cfg,
    xgb_cfg,
    fold_xgb_cfgs=None,
    show_progress=True,
    collect_predictions: bool = True,
):
    rep_records = reps_df.to_dict('records')
    n_models = len(rep_records)
    chunk_size = int(max(1, perf_cfg.get('chunk_size_models', 16)))
    chunks = [rep_records[i:i + chunk_size] for i in range(0, n_models, chunk_size)]

    n_jobs = int(perf_cfg.get('n_jobs', 1))
    backend = str(perf_cfg.get('backend', 'loky'))

    summary_chunks = []
    pred_chunks = []
    country_chunks = []

    start = time.time()
    pbar = tqdm(total=n_models, disable=not show_progress, desc=f"XGB-LOCO[{context['bag_name']}] reps")

    for chunk_i, chunk in enumerate(chunks, start=1):
        effective_n_jobs = min(n_jobs, len(chunk))
        if effective_n_jobs <= 1:
            out = [
                fit_xgb_one_rep_across_folds(
                    rr, context, fold_designs, baseline_fold_cache, xgb_cfg, fold_xgb_cfgs
                )
                for rr in chunk
            ]
        else:
            try:
                out = Parallel(n_jobs=effective_n_jobs, backend=backend, verbose=0)(
                    delayed(fit_xgb_one_rep_across_folds)(
                        rr, context, fold_designs, baseline_fold_cache, xgb_cfg, fold_xgb_cfgs
                    ) for rr in chunk
                )
            except Exception:
                out = [
                    fit_xgb_one_rep_across_folds(
                        rr, context, fold_designs, baseline_fold_cache, xgb_cfg, fold_xgb_cfgs
                    )
                    for rr in chunk
                ]

        summary_chunks.append(pd.DataFrame([x[0] for x in out]))
        if collect_predictions:
            pred_chunks.append(pd.concat([x[1] for x in out], ignore_index=True))
        country_chunks.append(pd.concat([x[2] for x in out], ignore_index=True))

        pbar.update(len(chunk))
        elapsed = time.time() - start
        done = pbar.n
        speed = done / elapsed if elapsed > 0 else np.nan
        eta = (n_models - done) / speed if speed and np.isfinite(speed) and speed > 0 else np.nan

        chunk_summary = summary_chunks[-1]
        best_r2 = pd.to_numeric(chunk_summary.get('global_oof_r2', pd.Series(dtype=float)), errors='coerce').max()

        pbar.set_postfix({
            'chunk': f'{chunk_i}/{len(chunks)}',
            'models_s': f'{speed:.2f}' if np.isfinite(speed) else 'nan',
            'eta_s': f'{eta:.0f}' if np.isfinite(eta) else 'nan',
            'chunk_best_r2': f'{best_r2:.4f}' if pd.notna(best_r2) else 'nan',
        })

    pbar.close()

    rep_summary_df = pd.concat(summary_chunks, ignore_index=True) if summary_chunks else pd.DataFrame()
    rep_pred_df = pd.concat(pred_chunks, ignore_index=True) if collect_predictions and pred_chunks else pd.DataFrame()
    rep_country_df = pd.concat(country_chunks, ignore_index=True) if country_chunks else pd.DataFrame()
    return rep_summary_df, rep_pred_df, rep_country_df


def run_xgb_loco_stage(
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: List[str],
    bag_targets: dict,
    outdir: Path,
    analysis_cfg: dict,
    cv_cfg: dict,
    perf_cfg: dict,
    xgb_cfg: dict,
    early_stop_cfg: dict,
    fold_xgb_cfgs: dict | None = None,
    tuning_artifact_path: str | Path | None = None,
    tuning_rung_id: str | None = None,
    tuning_strict: bool = False,
    enable_progress=True,
    storage_cfg: dict | None = None,
    compare_cfg: dict | None = None,
):
    require_xgboost()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    compare_defaults = {
        'compute_baseline_oof': True,
        'compute_paired_delta': True,
        'compute_paired_after_all_bags': True,
        'paired_delta_n_jobs': 0,
        'paired_delta_backend': 'loky',
        'paired_delta_parallel_axis': 'auto',
        'paired_delta_chunk_models': 128,
        'paired_delta_models_backend': 'threading',
        'paired_delta_verbose': True,
        'save_thin_paired_predictions': True,
        'thin_top_n_per_rank': 20,
    }
    compare_cfg = {**compare_defaults, **({} if compare_cfg is None else dict(compare_cfg))}
    compare_cfg['min_n_obs_for_metrics'] = int(analysis_cfg.get('min_n_obs_for_metrics', 5))
    need_baseline = bool(compare_cfg.get('compute_baseline_oof', True))
    need_paired_delta = bool(compare_cfg.get('compute_paired_delta', True))
    compare_defer = bool(compare_cfg.get('compute_paired_after_all_bags', True)) and need_paired_delta

    if int(compare_cfg.get('paired_delta_n_jobs', 0) or 0) <= 0:
        compare_cfg['paired_delta_n_jobs'] = int(perf_cfg.get('n_jobs', 1))

    storage_cfg = {} if storage_cfg is None else dict(storage_cfg)
    write_model_predictions = bool(storage_cfg.get('write_xgb_model_predictions', False))
    write_base_predictions = bool(storage_cfg.get('write_xgb_base_predictions', False))
    keep_model_predictions = bool(storage_cfg.get('keep_xgb_model_predictions_in_memory', False))
    keep_base_predictions = bool(storage_cfg.get('keep_xgb_base_predictions_in_memory', False))
    collect_model_predictions = bool(keep_model_predictions or need_paired_delta or write_model_predictions)
    collect_base_predictions = bool(keep_base_predictions or need_paired_delta or write_base_predictions)

    all_model_pred = []
    all_model_summary = []
    all_model_country = []
    all_rep_summary = []
    all_identity = []
    all_base_summary = []
    all_base_pred = []
    all_base_country = []
    all_delta_summary = []
    all_delta_country = []
    all_paired_thin = []
    timing_all = []

    for bag_name, y_col in bag_targets.items():
        print('\n' + '=' * 80)
        print(f'XGB LOCO -> BAG {bag_name} ({y_col})')

        t0_bag = time.time()
        context = prepare_bag_context(model_df, y_col, bag_name, analysis_cfg, cv_cfg, exposome_cols)

        if tuning_artifact_path:
            if not tuning_rung_id:
                raise ValueError('tuning_rung_id is required when tuning_artifact_path is supplied')
            from xgb_nested_loco_tuning import resolve_tuned_fold_configs
            resolved = resolve_tuned_fold_configs(
                tuning_artifact_path, bag_name, tuning_rung_id, xgb_cfg, context['countries'], strict=tuning_strict,
            )
            if resolved:
                fold_xgb_cfgs = {**(fold_xgb_cfgs or {}), bag_name: resolved}

        fold_designs = {}
        for i, c in enumerate(context['countries']):
            fold_designs[c] = _build_base_fold_mats(
                context,
                c,
                analysis_cfg,
                early_stop_cfg,
                seed=int(xgb_cfg.get('random_state', 20260304)) + i,
            )

        bag_fold_xgb_cfgs = (fold_xgb_cfgs or {}).get(bag_name, fold_xgb_cfgs)
        t_base = time.time()
        if need_baseline:
            # Opt-in baseline reuse: the covariate-only baseline is candidate-
            # independent, so when only the candidate set changes it can be
            # loaded instead of refit.
            baseline_cache_dir = os.environ.get('XGB_BASELINE_CACHE_DIR', '').strip()
            cached_baseline = None
            baseline_sig = None
            if baseline_cache_dir:
                baseline_sig = baseline_cache_signature(context, fold_designs, xgb_cfg, bag_fold_xgb_cfgs)
                cached_baseline = load_cached_baseline(baseline_cache_dir, baseline_sig)
            if cached_baseline is not None:
                bsum, bpred, bcountry, baseline_fold_cache = cached_baseline
                print(f'[{bag_name}] xgb baseline cache hit sig={baseline_sig[:12]}')
            else:
                bsum, bpred, bcountry, baseline_fold_cache = fit_xgb_baseline_across_folds(
                    context, fold_designs, xgb_cfg, bag_fold_xgb_cfgs
                )
                if baseline_cache_dir:
                    save_cached_baseline(baseline_cache_dir, baseline_sig, bsum, bpred, bcountry, baseline_fold_cache)
            base_summary = pd.DataFrame([bsum])
            base_pred = bpred
            base_country = bcountry
        else:
            baseline_fold_cache = {}
            base_summary = pd.DataFrame()
            base_pred = pd.DataFrame()
            base_country = pd.DataFrame()
        t_base_elapsed = time.time() - t_base

        t_models = time.time()
        cmap_df, reps_df = make_candidate_identity_map(candidate_df, exposome_cols, enable_prune=bool(perf_cfg.get('enable_identity_prune', True)))
        rep_summary, rep_pred, rep_country = evaluate_xgb_representatives_parallel(
            reps_df,
            context,
            fold_designs,
            baseline_fold_cache,
            perf_cfg,
            xgb_cfg,
            fold_xgb_cfgs=bag_fold_xgb_cfgs,
            show_progress=enable_progress,
            collect_predictions=collect_model_predictions,
        )
        model_summary, model_pred, model_country = expand_representatives_to_candidates(
            rep_summary,
            rep_pred,
            rep_country,
            cmap_df,
            include_bic=False,
            include_predictions=collect_model_predictions,
        )
        id_map = cmap_df.copy()
        id_map['bag_target'] = bag_name
        t_models_elapsed = time.time() - t_models

        all_model_summary.append(model_summary)
        all_model_country.append(model_country)
        all_rep_summary.append(rep_summary)
        all_identity.append(id_map)

        all_base_summary.append(base_summary)
        all_base_country.append(base_country)

        if keep_model_predictions:
            all_model_pred.append(model_pred)
        if keep_base_predictions:
            all_base_pred.append(base_pred)

        if need_paired_delta and (not model_pred.empty) and (not base_pred.empty):
            if compare_defer:
                pass
            else:
                t_cmp = time.time()
                dsum_b, dcountry_b, thin_b = compute_paired_delta_tables(
                    model_summary,
                    model_pred,
                    base_summary,
                    base_pred,
                    compare_cfg,
                    include_bic=False,
                )
                if not dsum_b.empty:
                    all_delta_summary.append(dsum_b)
                if not dcountry_b.empty:
                    all_delta_country.append(dcountry_b)
                if not thin_b.empty:
                    all_paired_thin.append(thin_b)
                print(f'[{bag_name}] xgb paired delta completed. rows={len(dsum_b)} elapsed_sec={time.time() - t_cmp:.1f}')

        timing = pd.DataFrame([
            {'bag_target': bag_name, 'stage': 'baseline_loco', 'elapsed_sec': t_base_elapsed},
            {'bag_target': bag_name, 'stage': 'models_loco', 'elapsed_sec': t_models_elapsed},
            {'bag_target': bag_name, 'stage': 'total_bag', 'elapsed_sec': time.time() - t0_bag},
        ])
        atomic_write_csv(timing, outdir / f'xgb_timing_profile_{bag_name}.csv')
        timing_all.append(timing)

        baseline_status = f'{t_base_elapsed:.1f}s' if need_baseline else 'skipped'
        print(f'[{bag_name}] baseline={baseline_status} models_sec={t_models_elapsed:.1f} rows={len(model_summary)}')

    model_pred_df = pd.concat(all_model_pred, ignore_index=True) if all_model_pred else pd.DataFrame()
    model_summary_df = pd.concat(all_model_summary, ignore_index=True) if all_model_summary else pd.DataFrame()
    model_country_df = pd.concat(all_model_country, ignore_index=True) if all_model_country else pd.DataFrame()
    rep_summary_df = pd.concat(all_rep_summary, ignore_index=True) if all_rep_summary else pd.DataFrame()
    identity_map_df = pd.concat(all_identity, ignore_index=True) if all_identity else pd.DataFrame()
    base_summary_df = pd.concat(all_base_summary, ignore_index=True) if all_base_summary else pd.DataFrame()
    base_pred_df = pd.concat(all_base_pred, ignore_index=True) if all_base_pred else pd.DataFrame()
    base_country_df = pd.concat(all_base_country, ignore_index=True) if all_base_country else pd.DataFrame()

    if need_paired_delta and compare_defer and (not model_pred_df.empty) and (not base_pred_df.empty):
        tcmp = time.time()
        delta_summary_df, delta_country_df, paired_thin_df = compute_paired_delta_tables(
            model_summary_df,
            model_pred_df,
            base_summary_df,
            base_pred_df,
            compare_cfg,
            include_bic=False,
        )
        print(f'[xgb paired-delta] rows={len(delta_summary_df)} elapsed_sec={time.time()-tcmp:.1f}')
    else:
        delta_summary_df = pd.concat(all_delta_summary, ignore_index=True) if all_delta_summary else pd.DataFrame()
        delta_country_df = pd.concat(all_delta_country, ignore_index=True) if all_delta_country else pd.DataFrame()
        paired_thin_df = pd.concat(all_paired_thin, ignore_index=True) if all_paired_thin else pd.DataFrame()

    rank_full_df, rank_delta_df = build_dual_rankings(model_summary_df, delta_summary_df)
    model_vs_base_global_df, model_vs_base_country_df = build_model_vs_baseline_tables(
        model_summary_df,
        base_summary_df,
        model_country_df,
        base_country_df,
        delta_summary_df=delta_summary_df,
        delta_country_df=delta_country_df,
        include_bic=False,
    )
    summary_with_calib_df = model_vs_base_global_df.copy() if not model_vs_base_global_df.empty else model_summary_df.copy()

    atomic_write_csv(base_summary_df, outdir / 'xgb_base_loco_oof_summary.csv')
    atomic_write_csv(base_country_df, outdir / 'xgb_base_loco_oof_country_metrics.csv')
    if write_base_predictions and not base_pred_df.empty:
        atomic_write_csv(base_pred_df, outdir / 'xgb_base_loco_oof_predictions.csv')

    atomic_write_csv(model_summary_df, outdir / 'xgb_model_loco_summary.csv')
    atomic_write_csv(model_country_df, outdir / 'xgb_model_loco_country_metrics.csv')
    if write_model_predictions and not model_pred_df.empty:
        atomic_write_csv(model_pred_df, outdir / 'xgb_model_loco_oof_predictions.csv')

    atomic_write_csv(model_vs_base_global_df, outdir / 'xgb_model_vs_baseline_global.csv')
    atomic_write_csv(model_vs_base_country_df, outdir / 'xgb_model_vs_baseline_country.csv')
    atomic_write_csv(summary_with_calib_df, outdir / 'xgb_summary_with_calibration_complexity.csv')
    atomic_write_csv(delta_summary_df, outdir / 'xgb_loco_delta_summary.csv')
    atomic_write_csv(delta_country_df, outdir / 'xgb_loco_delta_country_metrics.csv')
    atomic_write_csv(rank_full_df, outdir / 'xgb_loco_rank_full.csv')
    atomic_write_csv(rank_delta_df, outdir / 'xgb_loco_rank_delta.csv')
    atomic_write_csv(rep_summary_df, outdir / 'xgb_model_representative_summary.csv')
    atomic_write_csv(identity_map_df, outdir / 'xgb_candidate_identity_map.csv')
    atomic_write_csv(pd.concat(timing_all, ignore_index=True) if timing_all else pd.DataFrame(), outdir / 'xgb_timing_profile_all_bags.csv')

    if bool(compare_cfg.get('save_thin_paired_predictions', True)):
        atomic_write_csv(paired_thin_df, outdir / 'xgb_loco_paired_predictions_thin.csv')

    return {
        'base_summary_df': base_summary_df,
        'base_pred_df': base_pred_df,
        'base_country_df': base_country_df,
        'model_summary_df': model_summary_df,
        'model_pred_df': model_pred_df,
        'model_country_df': model_country_df,
        'delta_summary_df': delta_summary_df,
        'delta_country_df': delta_country_df,
        'model_vs_base_global_df': model_vs_base_global_df,
        'model_vs_base_country_df': model_vs_base_country_df,
        'summary_with_calib_df': summary_with_calib_df,
        'rank_full_df': rank_full_df,
        'rank_delta_df': rank_delta_df,
        'identity_map_df': identity_map_df,
    }


def build_improvement_first_report_xgb(outdir: Path, cfg: dict | None = None):
    outdir = Path(outdir)
    defaults = {
        'summary_file': 'xgb_summary_with_calibration_complexity.csv',
        'country_file': 'xgb_model_vs_baseline_country.csv',
        'out_markdown': 'xgb_model_selection_report_improvement_first.md',
        'out_csv': 'xgb_model_selection_candidates_improvement_first.csv',
        'min_delta_r2_paired': 0.0,
        'max_delta_mae_paired': 0.0,
        'min_coverage_pct': 0.95,
        'dedupe_by_predictor_identity': True,
        'top_k_per_bag': 8,
    }
    cfg = {**defaults, **({} if cfg is None else dict(cfg))}

    summary_path = outdir / str(cfg['summary_file'])
    country_path = outdir / str(cfg['country_file'])

    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {summary_path}')

    summary = pd.read_csv(summary_path)
    country = pd.read_csv(country_path) if country_path.exists() else pd.DataFrame()

    numeric_cols = [
        'order', 'score', 'predictors_identity_n', 'global_oof_r2', 'global_oof_mae', 'coverage_pct',
        'delta_r2_paired', 'delta_mae_paired', 'delta_r2_vs_base', 'delta_mae_vs_base',
        'global_oof_r2_base', 'global_oof_mae_base', 'global_oof_corr2',
        'calib_slope', 'calib_intercept', 'bias_mean', 'train_f2_mean',
    ]
    for c in numeric_cols:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors='coerce')

    filt = (
        (summary['delta_r2_paired'] > float(cfg['min_delta_r2_paired']))
        & (summary['delta_mae_paired'] < float(cfg['max_delta_mae_paired']))
        & (summary['coverage_pct'] >= float(cfg['min_coverage_pct']))
    )
    passed = summary[filt].copy()

    def country_robustness(bag, model_id):
        if country.empty:
            return {'n_countries': 0, 'n_joint_gain': 0}
        c = country[(country['bag_target'].astype(str) == str(bag)) & (country['model_id'].astype(str) == str(model_id))].copy()
        if c.empty:
            return {'n_countries': 0, 'n_joint_gain': 0}
        c['delta_r2_vs_base'] = pd.to_numeric(c['delta_r2_vs_base'], errors='coerce')
        c['delta_mae_vs_base'] = pd.to_numeric(c['delta_mae_vs_base'], errors='coerce')
        return {
            'n_countries': int(len(c)),
            'n_joint_gain': int(((c['delta_r2_vs_base'] > 0) & (c['delta_mae_vs_base'] < 0)).sum()),
        }

    rows = []
    bag_reports = {}
    for bag in sorted(summary['bag_target'].dropna().astype(str).unique().tolist()):
        sb_all = summary[summary['bag_target'].astype(str) == str(bag)].copy()
        sb = passed[passed['bag_target'].astype(str) == str(bag)].copy()
        if sb.empty:
            bag_reports[bag] = {'n_total': len(sb_all), 'n_pass': 0, 'top': pd.DataFrame()}
            continue

        sb = sb.sort_values(
            ['delta_r2_paired', 'delta_mae_paired', 'global_oof_r2', 'predictors_identity_n'],
            ascending=[False, True, False, True],
        )
        if bool(cfg['dedupe_by_predictor_identity']) and 'predictors_identity' in sb.columns:
            sb['predictors_identity'] = sb['predictors_identity'].astype(str)
            sb = sb.drop_duplicates(subset=['predictors_identity'], keep='first')

        top = sb.head(int(cfg['top_k_per_bag'])).copy()
        rob = [country_robustness(bag, mid) for mid in top['model_id'].astype(str).tolist()]
        if rob:
            top = pd.concat([top, pd.DataFrame(rob, index=top.index)], axis=1)

        bag_reports[bag] = {'n_total': len(sb_all), 'n_pass': len(sb), 'top': top}
        rows.append(top)

    cand_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    atomic_write_csv(cand_df, outdir / str(cfg['out_csv']))

    def fmt(x):
        if pd.isna(x):
            return 'NA'
        if isinstance(x, (float, np.floating)):
            return f'{float(x):.6f}'
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)

    lines = []
    lines.append('# XGBoost Improvement-first Model Selection Report\n')
    lines.append('## Criteria\n')
    lines.append(f"- Filter: delta_r2_paired > {cfg['min_delta_r2_paired']}\n")
    lines.append(f"- Filter: delta_mae_paired < {cfg['max_delta_mae_paired']}\n")
    lines.append(f"- Filter: coverage_pct >= {cfg['min_coverage_pct']}\n")
    lines.append('- Ranking: delta_r2_paired desc, delta_mae_paired asc, global_oof_r2 desc, predictors_identity_n asc\n\n')

    for bag in sorted(bag_reports.keys()):
        info = bag_reports[bag]
        top = info['top']
        lines.append(f'## BAG: {bag}\n')
        lines.append(f"- Total models: {info['n_total']}\n")
        lines.append(f"- Passed filters: {info['n_pass']}\n")
        if top.empty:
            lines.append('- No models passed criteria.\n\n')
            continue
        lines.append('|rank|model_id|objective|order|delta_r2_paired|delta_mae_paired|global_oof_r2|global_oof_mae|coverage|predictors_n|countries_joint_gain|f2|\n')
        lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"|{i}|{fmt(r.get('model_id'))}|{fmt(r.get('objective'))}|{fmt(r.get('order'))}|"
                f"{fmt(r.get('delta_r2_paired'))}|{fmt(r.get('delta_mae_paired'))}|"
                f"{fmt(r.get('global_oof_r2'))}|{fmt(r.get('global_oof_mae'))}|"
                f"{fmt(r.get('coverage_pct'))}|{fmt(r.get('predictors_identity_n'))}|"
                f"{fmt(r.get('n_joint_gain'))}/{fmt(r.get('n_countries'))}|"
                f"{fmt(r.get('train_f2_mean'))}|\n"
            )
        lines.append('\n')

    report_md = ''.join(lines)
    out_md = outdir / str(cfg['out_markdown'])
    out_md.write_text(report_md)

    return {
        'candidate_df': cand_df,
        'bag_reports': bag_reports,
        'report_markdown': report_md,
        'report_path': out_md,
        'candidate_csv_path': outdir / str(cfg['out_csv']),
    }
