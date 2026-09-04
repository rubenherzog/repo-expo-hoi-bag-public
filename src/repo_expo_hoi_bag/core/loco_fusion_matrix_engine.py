from __future__ import annotations

import ast
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

BASELINE_MODEL_ID = '__baseline_covariates__'


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def load_json(path: Path, default: dict) -> dict:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return default


def pin_blas_threads(n_threads: int) -> None:
    n = int(max(1, n_threads))
    for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n)
    except Exception:
        pass


def parse_literal_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        try:
            y = ast.literal_eval(x)
            if isinstance(y, (list, tuple)):
                return list(y)
        except Exception:
            return []
    return []


def load_greedy_feature_names(path: Path) -> List[str]:
    f = pd.read_csv(path)
    if 'feature_name' in f.columns:
        names = f['feature_name'].astype(str).tolist()
    else:
        names = f.iloc[:, 0].astype(str).tolist()
    names = [x.strip() for x in names if str(x).strip()]
    if not names:
        raise ValueError(f'No feature names found in {path}')
    return names


def align_exposome_to_greedy_features(exposome_df: pd.DataFrame, greedy_feature_names: List[str]) -> pd.DataFrame:
    missing = [f for f in greedy_feature_names if f not in exposome_df.columns]
    if missing:
        raise ValueError(f'Prepared exposome matrix is missing {len(missing)} greedy features. Example missing: {missing[:5]}')
    return exposome_df[greedy_feature_names].copy()


def objective_parts(obj: str) -> Tuple[str, str]:
    obj = str(obj).lower()
    if '_' in obj:
        metric, direction = obj.rsplit('_', 1)
    else:
        metric, direction = obj, 'na'
    return metric, direction


def candidate_df_from_greedy_topk(greedy_topk_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = greedy_topk_df.copy()
    if 'objective' not in df.columns or 'nplet_indices' not in df.columns:
        raise ValueError('greedy_topk_by_objective_order.csv must contain objective and nplet_indices columns.')

    df['objective'] = df['objective'].astype(str).str.lower()
    df['nplet_indices'] = df['nplet_indices'].apply(parse_literal_list)

    def vars_from_idx(idx):
        out = []
        for i in idx:
            try:
                out.append(feature_names[int(i)])
            except Exception:
                pass
        return out

    df['nplet_vars'] = df['nplet_indices'].apply(vars_from_idx)
    df['order'] = pd.to_numeric(df.get('order', np.nan), errors='coerce').astype('Int64')
    df['score'] = pd.to_numeric(df.get('score', np.nan), errors='coerce')

    df = df[df['nplet_vars'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

    is_min = df['objective'].astype(str).str.endswith('_min')
    score_sort = np.where(is_min, df['score'], -df['score'])
    df = df.assign(_score_sort=score_sort).sort_values(['objective', 'order', '_score_sort'], ascending=[True, True, True]).copy()
    df['rank'] = df.groupby(['objective', 'order']).cumcount() + 1
    df = df.drop(columns=['_score_sort']).reset_index(drop=True)

    df['feature_id'] = [
        f"m{ix:06d}_{obj}_ord{int(o) if pd.notna(o) else -1}_rk{int(r):05d}"
        for ix, (obj, o, r) in enumerate(zip(df['objective'], df['order'], df['rank']))
    ]
    return df


def select_candidate_space(cand_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fm = cand_df.copy()
    fm['objective'] = fm['objective'].astype(str).str.lower()
    fm['order'] = pd.to_numeric(fm['order'], errors='coerce').astype('Int64')
    fm['rank'] = pd.to_numeric(fm['rank'], errors='coerce')
    fm['score'] = pd.to_numeric(fm['score'], errors='coerce')

    if cfg.get('objective_filter'):
        keep_obj = set(str(x).lower() for x in cfg['objective_filter'])
    else:
        keep_obj = set(str(x).lower() for x in cfg.get('objectives', []))
    if keep_obj:
        fm = fm[fm['objective'].isin(keep_obj)].copy()

    eff_order_min = int(cfg['initial_order'] if cfg['order_min'] is None else cfg['order_min'])
    eff_order_max = int(cfg['max_order'] if cfg['order_max'] is None else cfg['order_max'])
    fm = fm[(fm['order'] >= eff_order_min) & (fm['order'] <= eff_order_max)].copy()

    is_min = fm['objective'].astype(str).str.endswith('_min')
    fm = fm.assign(_score_sort=np.where(is_min, fm['score'], -fm['score']))
    fm = fm.sort_values(['objective', 'order', '_score_sort'], ascending=[True, True, True]).copy()

    top_k_per_order = int(cfg.get('top_k_per_order', 0) or 0)
    if top_k_per_order > 0:
        fm = fm.groupby(['objective', 'order'], as_index=False, group_keys=False).head(top_k_per_order).copy()

    max_per_obj = int(cfg.get('max_candidates_per_objective', 0) or 0)
    if max_per_obj > 0:
        fm = fm.groupby('objective', as_index=False, group_keys=False).head(max_per_obj).copy()

    max_total = int(cfg.get('max_models_total', 0) or 0)
    if max_total > 0:
        fm = fm.head(max_total).copy()

    fm = fm.drop(columns=['_score_sort'], errors='ignore').reset_index(drop=True)
    return fm


def load_artifacts_greedy_only(paths: dict, analysis_cfg: dict):
    input_path = Path(paths['input_csv'])
    greedy_topk_path = Path(paths['greedy_topk_csv'])
    greedy_names_path = Path(paths['greedy_feature_names_csv'])

    need = [input_path, greedy_topk_path, greedy_names_path]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files\n' + '\n'.join(missing))

    greedy_names = load_greedy_feature_names(greedy_names_path)
    raw_df = pd.read_csv(input_path, low_memory=False)
    if 'N_MEGA' not in raw_df.columns:
        raise ValueError(f"Complete-case input CSV missing required column 'N_MEGA': {input_path}")

    raw_df['N_MEGA'] = raw_df['N_MEGA'].astype(str).str.strip()
    raw_df = raw_df.drop_duplicates(subset='N_MEGA', keep='first').reset_index(drop=True)
    exposome_df = align_exposome_to_greedy_features(raw_df, greedy_names)
    expo_numeric = exposome_df.apply(pd.to_numeric, errors='coerce')
    if int(expo_numeric.isna().sum().sum()) != 0:
        raise ValueError(
            'Complete-case input CSV contains missing exposome values after greedy feature alignment. '
            'This workflow requires a fully complete exposome matrix.'
        )
    exposome_df = expo_numeric.reset_index(drop=True)

    greedy_topk_df = pd.read_csv(greedy_topk_path)
    candidate_all_df = candidate_df_from_greedy_topk(greedy_topk_df, exposome_df.columns.tolist())
    candidate_df = select_candidate_space(candidate_all_df, analysis_cfg)

    return raw_df.reset_index(drop=True), exposome_df, candidate_df, candidate_all_df


def coalesce_columns(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        out = out.where(out.notna(), df[c])
    return out


def target_map(mode: str) -> dict:
    m = str(mode).lower()
    if m == 'structural':
        return {'structural': 'bag_struct_resolved'}
    if m == 'functional':
        return {'functional': 'bag_func_resolved'}
    if m == 'combined':
        return {'combined': 'bag_comb_resolved'}
    if m == 'all':
        return {
            'structural': 'bag_struct_resolved',
            'functional': 'bag_func_resolved',
            'combined': 'bag_comb_resolved',
        }
    raise ValueError(f'Unknown BAG_TARGET_MODE: {mode}')


def build_model_df(raw_complete_df: pd.DataFrame, exposome_df: pd.DataFrame, analysis_cfg: dict) -> pd.DataFrame:
    required_cols = [
        'N_MEGA', 'Age', 'Sex', 'Diagnosis', 'Country', 'country_clean',
        'year_acq', 'Year', 'exposome_year',
    ]
    missing_required = [c for c in required_cols if c not in raw_complete_df.columns]
    if missing_required:
        raise ValueError(f'Complete-case input CSV missing required metadata columns: {missing_required}')

    raw = raw_complete_df.copy().reset_index(drop=True)
    raw['N_MEGA'] = raw['N_MEGA'].astype(str).str.strip()

    if len(raw) != len(exposome_df):
        raise ValueError(f'Row mismatch raw_complete_df({len(raw)}) vs exposome_df({len(exposome_df)}).')

    model_base = raw[
        [
            'N_MEGA', 'Age', 'Sex', 'Diagnosis', 'Country', 'country_clean',
            'year_acq', 'Year', 'exposome_year',
            *[c for c in ['BAG_struc', 'BAG_func', 'BAG_OOS_structural', 'BAG_OOS_functional'] if c in raw.columns],
        ]
    ].copy()

    model_base['Age'] = pd.to_numeric(model_base['Age'], errors='coerce')
    model_base['Diagnosis'] = model_base['Diagnosis'].astype(str).str.strip()
    model_base['Sex'] = model_base['Sex'].astype(str).str.strip()
    model_base['Country'] = model_base['Country'].astype(str).str.strip()
    model_base['country_clean'] = model_base['country_clean'].astype(str).str.strip()
    model_base['year_acq'] = pd.to_numeric(model_base['year_acq'], errors='coerce')
    model_base['Year'] = pd.to_numeric(model_base['Year'], errors='coerce')
    model_base['exposome_year'] = pd.to_numeric(model_base['exposome_year'], errors='coerce')

    model_base['bag_struct_resolved'] = pd.to_numeric(
        coalesce_columns(model_base, ['BAG_struc', 'BAG_OOS_structural']),
        errors='coerce',
    )
    model_base['bag_func_resolved'] = pd.to_numeric(
        coalesce_columns(model_base, ['BAG_func', 'BAG_OOS_functional']),
        errors='coerce',
    )
    model_base['bag_comb_resolved'] = np.where(
        model_base['bag_struct_resolved'].notna() & model_base['bag_func_resolved'].notna(),
        (model_base['bag_struct_resolved'] + model_base['bag_func_resolved']) / 2.0,
        np.nan,
    )

    def _blank_or_missing(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.lower()
        return s.isin(['', 'nan', 'none'])

    required_baseline = ['Age', 'Sex', 'Diagnosis', 'country_clean', 'exposome_year']
    baseline_missing_masks = {
        'Age': model_base['Age'].isna(),
        'Sex': _blank_or_missing(model_base['Sex']),
        'Diagnosis': _blank_or_missing(model_base['Diagnosis']),
        'country_clean': _blank_or_missing(model_base['country_clean']),
        'exposome_year': model_base['exposome_year'].isna(),
    }
    drop_mask = np.zeros(len(model_base), dtype=bool)
    for c in required_baseline:
        drop_mask |= baseline_missing_masks[c].to_numpy(dtype=bool)
    if bool(np.any(drop_mask)):
        dropped = int(np.sum(drop_mask))
        reasons = {c: int(baseline_missing_masks[c].sum()) for c in required_baseline}
        print(f'Global baseline cleanup dropped {dropped} rows before CV: {reasons}')
        model_base = model_base.loc[~drop_mask].reset_index(drop=True)
        exposome_df = exposome_df.loc[~drop_mask].reset_index(drop=True)
    else:
        print('Global baseline cleanup dropped 0 rows before CV.')

    model_base['exposome_year_num'] = model_base['exposome_year']
    if model_base['exposome_year_num'].isna().all() and 'year_acq' in model_base.columns:
        model_base['exposome_year_num'] = pd.to_numeric(model_base['year_acq'], errors='coerce')
    if model_base['exposome_year_num'].isna().all() and 'Year' in model_base.columns:
        model_base['exposome_year_num'] = pd.to_numeric(model_base['Year'], errors='coerce')

    model_base['exposome_year_cat'] = model_base['exposome_year_num'].round().astype('Int64').astype(str)

    if int(exposome_df.isna().sum().sum()) != 0:
        raise ValueError('Exposome matrix contains missing values after complete-case loading and baseline cleanup.')

    model_df = pd.concat([model_base.reset_index(drop=True), exposome_df.reset_index(drop=True)], axis=1)
    model_df['row_id'] = np.arange(len(model_df), dtype=int)
    return model_df


def regression_metrics(y_true, y_pred, min_n=2):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < min_n:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n_scored': n}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        'r2': float(r2_score(yt, yp)),
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mae': float(mean_absolute_error(yt, yp)),
        'n_scored': n,
    }


def safe_corr2(y_true, y_pred, min_n=3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < int(min_n):
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    if float(np.nanstd(yt)) <= 0 or float(np.nanstd(yp)) <= 0:
        return np.nan
    c = np.corrcoef(yt, yp)[0, 1]
    if not np.isfinite(c):
        return np.nan
    return float(c * c)


def calibration_stats(y_true, y_pred, min_n=3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < int(min_n):
        return {'calib_slope': np.nan, 'calib_intercept': np.nan, 'bias_mean': np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]
    x_mean = float(np.mean(yp))
    y_mean = float(np.mean(yt))
    var_x = float(np.sum((yp - x_mean) ** 2))
    if var_x <= 0:
        slope = np.nan
        intercept = np.nan
    else:
        cov_xy = float(np.sum((yp - x_mean) * (yt - y_mean)))
        slope = float(cov_xy / var_x)
        intercept = float(y_mean - slope * x_mean)
    bias = float(np.mean(yp - yt))
    return {'calib_slope': slope, 'calib_intercept': intercept, 'bias_mean': bias}


def r2_from_predictions(y_true, y_pred, min_n=2):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < int(min_n):
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    if sst <= 0:
        return np.nan
    sse = float(np.sum((yt - yp) ** 2))
    return float(1.0 - (sse / sst))


def gaussian_sigma2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return np.nan
    resid = y_true[mask] - y_pred[mask]
    return float(np.mean(resid * resid))


def gaussian_bic(y_true, y_pred, k):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n <= 1:
        return np.nan
    sigma2 = gaussian_sigma2(y_true[mask], y_pred[mask])
    if not np.isfinite(sigma2):
        return np.nan
    sigma2 = float(max(1e-12, sigma2))
    k_eff = int(max(0, k))
    return float(n * np.log(sigma2) + k_eff * np.log(max(2, n)))


def gaussian_loglik(y_true, y_pred, sigma2):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0 or (not np.isfinite(sigma2)) or float(sigma2) <= 0:
        return np.nan, 0
    yt = y_true[mask]
    yp = y_pred[mask]
    s2 = float(max(1e-12, sigma2))
    sse = float(np.sum((yt - yp) ** 2))
    ll = float(-0.5 * (n * np.log(2.0 * np.pi * s2) + (sse / s2)))
    return ll, n


def regression_metrics_extended(y_true, y_pred, min_n=2):
    base = regression_metrics(y_true, y_pred, min_n=min_n)
    corr2 = safe_corr2(y_true, y_pred, min_n=max(3, min_n))
    calib = calibration_stats(y_true, y_pred, min_n=max(3, min_n))
    return {
        **base,
        'corr2': corr2,
        'calib_slope': calib['calib_slope'],
        'calib_intercept': calib['calib_intercept'],
        'bias_mean': calib['bias_mean'],
    }


def nanmean_safe(values):
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmean(arr))


def prepare_analysis_table(model_df: pd.DataFrame, y_col: str, analysis_cfg: dict, cv_cfg: dict) -> pd.DataFrame:
    work = model_df.copy()

    if analysis_cfg.get('exclude_diagnosis') and 'Diagnosis' in work.columns:
        work = work[~work['Diagnosis'].isin(analysis_cfg['exclude_diagnosis'])].copy()

    split_col = cv_cfg['split_col']
    if analysis_cfg.get('exclude_countries') and split_col in work.columns:
        work = work[~work[split_col].isin(analysis_cfg['exclude_countries'])].copy()

    work = work[work[split_col].notna()].copy()
    work = work[work[y_col].notna()].copy()
    return work


def make_candidate_identity_map(cand_df: pd.DataFrame, exposome_cols: List[str], enable_prune=True):
    expo_idx = {c: i for i, c in enumerate(exposome_cols)}
    rows = []
    first_rep = {}

    for _, r in cand_df.iterrows():
        model_id = str(r['feature_id'])
        nplet_vars = parse_literal_list(r.get('nplet_vars', []))
        pred_names = [x for x in nplet_vars if x in expo_idx]
        pred_idx = sorted(set(int(expo_idx[x]) for x in pred_names))
        key = ','.join(str(i) for i in pred_idx)
        if not key:
            key = f'EMPTY::{model_id}'

        rep_id = first_rep.get(key)
        if rep_id is None:
            rep_id = model_id
            first_rep[key] = rep_id

        metric, direction = objective_parts(str(r.get('objective', '')))

        rows.append({
            'model_id': model_id,
            'representative_id': rep_id if enable_prune else model_id,
            'objective': str(r.get('objective', '')).lower(),
            'metric': metric,
            'direction': direction,
            'order': pd.to_numeric(r.get('order', np.nan), errors='coerce'),
            'score': pd.to_numeric(r.get('score', np.nan), errors='coerce'),
            'rank': pd.to_numeric(r.get('rank', np.nan), errors='coerce'),
            'predictor_indices': key,
            'predictor_idx_list': pred_idx,
            'predictors_identity': '|'.join(exposome_cols[i] for i in pred_idx),
            'predictors_identity_n': len(pred_idx),
            'nplet_vars': '|'.join(str(v) for v in nplet_vars),
        })

    cmap_df = pd.DataFrame(rows)
    reps_df = (
        cmap_df[['representative_id', 'predictor_indices', 'predictor_idx_list', 'predictors_identity', 'predictors_identity_n']]
        .drop_duplicates('representative_id')
        .rename(columns={'representative_id': 'model_id'})
        .reset_index(drop=True)
    )
    return cmap_df, reps_df


def prepare_year_basis_by_fold(year_values, countries, train_idx_by_country, analysis_cfg):
    if not analysis_cfg.get('include_year', True):
        return {}

    mode = str(analysis_cfg.get('year_effect_mode', 'spline')).lower()
    y = np.asarray(year_values, dtype=float)
    out = {}

    if mode == 'linear':
        yy = y.reshape(-1, 1)
        for c in countries:
            out[c] = yy
        return out

    if mode == 'categorical':
        y_ok = np.isfinite(y)
        med = float(np.nanmedian(y[y_ok])) if bool(y_ok.any()) else 0.0
        y_fill = np.where(np.isfinite(y), np.round(y), np.round(med))
        cats = sorted(pd.Series(y_fill).astype(int).astype(str).unique().tolist())
        mat = []
        for lv in cats[1:]:
            mat.append((pd.Series(y_fill).astype(int).astype(str).values == lv).astype(float))
        YY = np.column_stack(mat) if len(mat) else np.zeros((len(y_fill), 0), dtype=float)
        for c in countries:
            out[c] = YY
        return out

    # spline mode
    df_spline = int(analysis_cfg.get('year_spline_df', 4))
    for c in countries:
        tr_idx = train_idx_by_country[c]
        tr_year = y[tr_idx]
        tr_ok = np.isfinite(tr_year)
        if not bool(tr_ok.any()):
            out[c] = np.zeros((len(y), df_spline), dtype=float)
            continue

        tr_vals = tr_year[tr_ok]
        tr_dm = dmatrix(f"bs(x, df={df_spline}, include_intercept=False)", {'x': tr_vals}, return_type='dataframe')
        di = tr_dm.design_info

        med = float(np.nanmedian(tr_vals))
        y_fill = np.where(np.isfinite(y), y, med)
        tr_min = float(np.nanmin(tr_vals))
        tr_max = float(np.nanmax(tr_vals))
        y_apply = np.clip(y_fill, tr_min, tr_max)

        all_dm = build_design_matrices([di], {'x': y_apply}, return_type='dataframe')[0]
        out[c] = np.asarray(all_dm, dtype=float)

    return out


def prepare_bag_context(model_df, y_col, bag_name, analysis_cfg, cv_cfg, exposome_cols):
    work = prepare_analysis_table(model_df, y_col, analysis_cfg, cv_cfg).reset_index(drop=True)

    split_col = cv_cfg['split_col']
    train_diag_group = cv_cfg.get('train_diagnosis_group')
    test_diag_group = cv_cfg.get('test_diagnosis_group')
    shared_country_only = bool(cv_cfg.get('shared_country_only', False))

    if (train_diag_group is not None or test_diag_group is not None) and 'Diagnosis' not in work.columns:
        raise ValueError('Diagnosis-aware CV requested, but Diagnosis column is not available in the model table.')

    if train_diag_group is not None or test_diag_group is not None:
        keep_diags = set()
        if train_diag_group is not None:
            keep_diags.add(str(train_diag_group))
        if test_diag_group is not None:
            keep_diags.add(str(test_diag_group))
        work = work[work['Diagnosis'].astype(str).isin(sorted(keep_diags))].reset_index(drop=True)

    countries = sorted(work[split_col].astype(str).dropna().unique().tolist())

    if shared_country_only and train_diag_group is not None and test_diag_group is not None:
        diag_country = (
            work[[split_col, 'Diagnosis']]
            .dropna()
            .assign(
                Diagnosis=lambda d: d['Diagnosis'].astype(str),
                fold_country=lambda d: d[split_col].astype(str),
            )
        )
        train_countries = set(diag_country.loc[diag_country['Diagnosis'] == str(train_diag_group), 'fold_country'].tolist())
        test_countries = set(diag_country.loc[diag_country['Diagnosis'] == str(test_diag_group), 'fold_country'].tolist())
        countries = sorted(train_countries & test_countries)

    min_country_size_test = int(cv_cfg.get('min_country_size_test', 0) or 0)
    if min_country_size_test > 0:
        if test_diag_group is not None:
            vc = work.loc[work['Diagnosis'].astype(str) == str(test_diag_group), split_col].astype(str).value_counts()
        else:
            vc = work[split_col].astype(str).value_counts()
        countries = [c for c in countries if int(vc.get(c, 0)) >= min_country_size_test]

    row_id = work['row_id'].to_numpy(dtype=int)
    nmega = work['N_MEGA'].astype(str).to_numpy()
    country = work[split_col].astype(str).to_numpy()
    y = pd.to_numeric(work[y_col], errors='coerce').to_numpy(dtype=float)

    age = pd.to_numeric(work['Age'], errors='coerce').to_numpy(dtype=float)
    sex = np.where(work['Sex'].notna(), work['Sex'].astype(str).to_numpy(), '') if 'Sex' in work.columns else np.asarray([''] * len(work))
    diag = np.where(work['Diagnosis'].notna(), work['Diagnosis'].astype(str).to_numpy(), '') if 'Diagnosis' in work.columns else np.asarray([''] * len(work))
    year = pd.to_numeric(work['exposome_year_num'], errors='coerce').to_numpy(dtype=float)
    edu = pd.to_numeric(work['Edu'], errors='coerce').to_numpy(dtype=float) if 'Edu' in work.columns else np.full(len(work), np.nan)
    scanner = np.where(work['scanner_id'].notna(), work['scanner_id'].astype(str).to_numpy(), '') if 'scanner_id' in work.columns else np.asarray([''] * len(work))

    X_exp = work[exposome_cols].to_numpy(dtype=float)
    m_exp = np.isfinite(X_exp)

    m_y = np.isfinite(y)
    m_age = np.isfinite(age)
    m_year = np.isfinite(year)
    m_sex = np.asarray([bool(str(x).strip()) and str(x).strip().lower() != 'nan' for x in sex], dtype=bool)
    m_diag = np.asarray([bool(str(x).strip()) and str(x).strip().lower() != 'nan' for x in diag], dtype=bool)
    m_edu = np.isfinite(edu)
    m_scanner = np.asarray([bool(str(x).strip()) and str(x).strip().lower() != 'nan' for x in scanner], dtype=bool)

    base_req_train = m_y & m_age
    if analysis_cfg.get('include_year', True):
        base_req_train &= m_year
    if analysis_cfg.get('include_sex', True):
        base_req_train &= m_sex
    if analysis_cfg.get('include_diagnosis', True):
        base_req_train &= m_diag
    if analysis_cfg.get('include_education', False):
        base_req_train &= m_edu
    if analysis_cfg.get('include_scanner', False):
        base_req_train &= m_scanner

    base_req_test = m_age
    if analysis_cfg.get('include_year', True):
        base_req_test &= m_year
    if analysis_cfg.get('include_education', False):
        base_req_test &= m_edu

    train_idx_by_country = {}
    test_idx_by_country = {}
    c_arr = np.asarray(country)
    diag_arr = np.asarray(diag).astype(str)
    train_diag_mask = np.ones(len(c_arr), dtype=bool) if train_diag_group is None else (diag_arr == str(train_diag_group))
    test_diag_mask = np.ones(len(c_arr), dtype=bool) if test_diag_group is None else (diag_arr == str(test_diag_group))
    for c in countries:
        te = np.where((c_arr == c) & test_diag_mask)[0]
        tr = np.where((c_arr != c) & train_diag_mask)[0]
        test_idx_by_country[c] = te
        train_idx_by_country[c] = tr

    year_basis_by_country = prepare_year_basis_by_fold(year, countries, train_idx_by_country, analysis_cfg)

    return {
        'bag_name': bag_name,
        'y_col': y_col,
        'countries': countries,
        'row_id': row_id,
        'N_MEGA': nmega,
        'country': c_arr,
        'y': y,
        'age': age,
        'sex': sex,
        'diag': diag,
        'year': year,
        'edu': edu,
        'scanner': scanner,
        'X_exp': X_exp,
        'm_exp': m_exp,
        'base_req_train': base_req_train,
        'base_req_test': base_req_test,
        'train_idx_by_country': train_idx_by_country,
        'test_idx_by_country': test_idx_by_country,
        'year_basis_by_country': year_basis_by_country,
        'analysis_cfg': analysis_cfg,
        'cv_cfg': cv_cfg,
    }


def adj_r2(y_true, y_pred, p):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n <= (p + 1) or n < 3:
        return np.nan
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= 0:
        return np.nan
    sse = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (sse / sst)
    return float(1.0 - (1.0 - r2) * ((n - 1) / max(1, n - p - 1)))


def encode_dummies(train_vals, test_vals):
    train_vals = np.asarray(train_vals).astype(str)
    test_vals = np.asarray(test_vals).astype(str)
    cats = sorted(pd.Series(train_vals).astype(str).unique().tolist())
    if len(cats) <= 1:
        return np.zeros((len(train_vals), 0), dtype=float), np.zeros((len(test_vals), 0), dtype=float), cats
    use = cats[1:]
    tr = np.column_stack([(train_vals == lv).astype(float) for lv in use]) if len(use) else np.zeros((len(train_vals), 0), dtype=float)
    te = np.column_stack([(test_vals == lv).astype(float) for lv in use]) if len(use) else np.zeros((len(test_vals), 0), dtype=float)
    return tr, te, cats


def fit_predict_one_rep_across_folds(rep_row, context, matrix_cfg):
    rep_id = str(rep_row['model_id'])
    pred_idx_all = list(rep_row.get('predictor_idx_list', []))

    cfg = context['analysis_cfg']
    cv_cfg = context['cv_cfg']

    y = context['y']
    age = context['age']
    sex = context['sex']
    diag = context['diag']
    edu = context['edu']
    scanner = context['scanner']
    X_exp = context['X_exp']
    m_exp = context['m_exp']

    fold_pred_chunks = []
    country_rows = []
    train_adj_vals = []
    fold_complexity_rows = []

    if len(pred_idx_all) == 0:
        # Return empty-scored predictions for all folds.
        for c in context['countries']:
            te_idx = context['test_idx_by_country'][c]
            fold_df = pd.DataFrame({
                'bag_target': context['bag_name'],
                'model_id': rep_id,
                'fold_country': c,
                'row_id': context['row_id'][te_idx],
                'N_MEGA': context['N_MEGA'][te_idx],
                'country': context['country'][te_idx],
                'y_true': y[te_idx],
                'y_pred': np.full(len(te_idx), np.nan),
                'status': 'failed',
                'error': 'no_predictors_found_in_exposome_matrix',
                'predictors_used_n': 0,
                'predictors_unresolved_n': 0,
                'predictors_dropped_zero_var_n': 0,
                'predictors_identity': str(rep_row.get('predictors_identity', '')),
                'predictors_used': '',
            })
            fold_df['scored'] = False
            fold_pred_chunks.append(fold_df)

        pred_df = pd.concat(fold_pred_chunks, ignore_index=True)
        g = regression_metrics_extended(
            pred_df.loc[pred_df['scored'], 'y_true'].values,
            pred_df.loc[pred_df['scored'], 'y_pred'].values,
            min_n=int(cfg.get('min_n_obs_for_metrics', 5)),
        )
        summary = {
            'bag_target': context['bag_name'],
            'model_id': rep_id,
            'predictors_identity': str(rep_row.get('predictors_identity', '')),
            'predictors_identity_n': int(rep_row.get('predictors_identity_n', 0)),
            'predictors_used_n': 0,
            'predictors_dropped_zero_var_n': 0,
            'global_oof_r2': g['r2'],
            'global_oof_rmse': g['rmse'],
            'global_oof_mae': g['mae'],
            'global_oof_corr2': g['corr2'],
            'calib_slope': g['calib_slope'],
            'calib_intercept': g['calib_intercept'],
            'bias_mean': g['bias_mean'],
            'n_total_with_y': int(pred_df['y_true'].notna().sum()),
            'n_scored': int(g['n_scored']),
            'coverage_pct': float(g['n_scored'] / max(1, int(pred_df['y_true'].notna().sum()))),
            'train_r2_adj_mean': np.nan,
            'train_r2_mean': np.nan,
            'train_r2_base_mean': np.nan,
            'train_f2_mean': np.nan,
            'train_bic_mean': np.nan,
            'train_bic_base_mean': np.nan,
            'train_bic_delta_mean': np.nan,
            'pseudo_oos_ll': np.nan,
            'pseudo_oos_bic': np.nan,
            'pseudo_oos_scored_n': 0,
            'complexity_folds_n': 0,
        }
        return summary, pred_df, pd.DataFrame()

    pred_nonmissing = m_exp[:, pred_idx_all].all(axis=1)

    for c in context['countries']:
        tr_idx_full = context['train_idx_by_country'][c]
        te_idx = context['test_idx_by_country'][c]

        train_ok = context['base_req_train'] & pred_nonmissing
        tr_idx = tr_idx_full[train_ok[tr_idx_full]]

        y_pred_fold = np.full(len(te_idx), np.nan, dtype=float)
        status = 'ok'
        error = ''

        pred_used_idx = pred_idx_all
        dropped_zero_n = 0
        train_r2_full = np.nan
        train_r2_base = np.nan
        train_f2 = np.nan
        train_bic_full = np.nan
        train_bic_base = np.nan
        train_sigma2_full = np.nan
        k_full = np.nan
        pseudo_ll_fold = np.nan
        pseudo_bic_fold = np.nan
        n_test_scored_fold = 0

        if len(tr_idx) < int(cfg.get('min_n_obs', 80)):
            status = 'failed'
            error = f'insufficient_train_rows<{int(cfg.get("min_n_obs", 80))}'
        else:
            Xtr_all = X_exp[np.ix_(tr_idx, pred_idx_all)]
            var = np.nanvar(Xtr_all, axis=0)
            keep = var > 0
            pred_used_idx = [idx for idx, kk in zip(pred_idx_all, keep) if bool(kk)]
            dropped_zero_n = int(np.sum(~keep))
            if len(pred_used_idx) == 0:
                status = 'failed'
                error = 'all_predictors_zero_variance'

        if status == 'ok':
            Xtr_pred = X_exp[np.ix_(tr_idx, pred_used_idx)]
            ytr = y[tr_idx]

            Xtr_parts = [np.ones((len(tr_idx), 1), dtype=float), age[tr_idx].reshape(-1, 1)]
            Xte_parts = [np.ones((len(te_idx), 1), dtype=float), age[te_idx].reshape(-1, 1)]

            sex_used = False
            sex_levels = []
            if cfg.get('include_sex', True):
                sx_tr = sex[tr_idx]
                sex_levels = sorted(pd.Series(sx_tr).astype(str).unique().tolist())
                if len(sex_levels) > 1:
                    sex_used = True
                    s_tr, s_te, _ = encode_dummies(sx_tr, sex[te_idx])
                    Xtr_parts.append(s_tr)
                    Xte_parts.append(s_te)

            if cfg.get('include_year', True):
                yb = context['year_basis_by_country'][c]
                Xtr_parts.append(yb[tr_idx])
                Xte_parts.append(yb[te_idx])

            diag_used = False
            diag_levels = []
            Dtr = np.zeros((len(tr_idx), 0), dtype=float)
            Dte = np.zeros((len(te_idx), 0), dtype=float)
            if cfg.get('include_diagnosis', True):
                dg_tr = diag[tr_idx]
                diag_levels = sorted(pd.Series(dg_tr).astype(str).unique().tolist())
                if len(diag_levels) > 1:
                    diag_used = True
                    Dtr, Dte, _ = encode_dummies(dg_tr, diag[te_idx])
                    Xtr_parts.append(Dtr)
                    Xte_parts.append(Dte)

            if cfg.get('include_education', False):
                Xtr_parts.append(edu[tr_idx].reshape(-1, 1))
                Xte_parts.append(edu[te_idx].reshape(-1, 1))

            scanner_used = False
            scanner_levels = []
            if cfg.get('include_scanner', False):
                sc_tr = scanner[tr_idx]
                scanner_levels = sorted(pd.Series(sc_tr).astype(str).unique().tolist())
                if len(scanner_levels) > 1:
                    scanner_used = True
                    sc_tr_enc, sc_te_enc, _ = encode_dummies(sc_tr, scanner[te_idx])
                    Xtr_parts.append(sc_tr_enc)
                    Xte_parts.append(sc_te_enc)

            # Baseline design on the exact same train/test rows as the full fold fit.
            Xtr_base = np.hstack(Xtr_parts)
            Xtr_parts.append(Xtr_pred)
            Xte_pred = X_exp[np.ix_(te_idx, pred_used_idx)]
            Xte_parts.append(Xte_pred)

            if diag_used and Dtr.shape[1] > 0 and Xtr_pred.shape[1] > 0:
                Itr = (Dtr[:, :, None] * Xtr_pred[:, None, :]).reshape(len(tr_idx), -1)
                Ite = (Dte[:, :, None] * Xte_pred[:, None, :]).reshape(len(te_idx), -1)
                Xtr_parts.append(Itr)
                Xte_parts.append(Ite)

            Xtr = np.hstack(Xtr_parts)
            Xte = np.hstack(Xte_parts)

            try:
                beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=matrix_cfg.get('rcond', None))
                ytr_hat = Xtr @ beta
                train_adj_vals.append(adj_r2(ytr, ytr_hat, p=max(0, Xtr.shape[1] - 1)))
                k_full = int(Xtr.shape[1])
                train_r2_full = r2_from_predictions(ytr, ytr_hat, min_n=2)
                train_bic_full = gaussian_bic(ytr, ytr_hat, k=k_full)
                train_sigma2_full = gaussian_sigma2(ytr, ytr_hat)

                try:
                    beta_base, *_ = np.linalg.lstsq(Xtr_base, ytr, rcond=matrix_cfg.get('rcond', None))
                    ytr_hat_base = Xtr_base @ beta_base
                    train_r2_base = r2_from_predictions(ytr, ytr_hat_base, min_n=2)
                    train_bic_base = gaussian_bic(ytr, ytr_hat_base, k=int(Xtr_base.shape[1]))
                    if pd.notna(train_r2_full) and pd.notna(train_r2_base):
                        den = float(1.0 - train_r2_full)
                        if den > 1e-12:
                            train_f2 = float((train_r2_full - train_r2_base) / den)
                except Exception:
                    pass
            except Exception as e:
                status = 'failed'
                error = f'fit_failed::{repr(e)}'
                beta = None

            if status == 'ok':
                test_mask = context['base_req_test'][te_idx] & m_exp[np.ix_(te_idx, pred_idx_all)].all(axis=1)
                if sex_used:
                    test_mask &= np.isin(sex[te_idx], sex_levels)
                if diag_used and str(cv_cfg.get('unseen_diag_policy', 'drop_test_rows')) == 'drop_test_rows':
                    test_mask &= np.isin(diag[te_idx], diag_levels)
                if scanner_used:
                    test_mask &= np.isin(scanner[te_idx], scanner_levels)

                if bool(np.any(test_mask)):
                    try:
                        y_pred_fold[test_mask] = Xte[test_mask] @ beta
                        yt_te = y[te_idx][test_mask]
                        yp_te = y_pred_fold[test_mask]
                        pseudo_ll_fold, n_test_scored_fold = gaussian_loglik(yt_te, yp_te, train_sigma2_full)
                        if n_test_scored_fold > 0 and np.isfinite(pseudo_ll_fold) and np.isfinite(k_full):
                            pseudo_bic_fold = float(-2.0 * pseudo_ll_fold + float(k_full) * np.log(max(2, n_test_scored_fold)))
                    except Exception as e:
                        status = 'failed'
                        error = f'predict_failed::{repr(e)}'

        fold_df = pd.DataFrame({
            'bag_target': context['bag_name'],
            'model_id': rep_id,
            'fold_country': c,
            'row_id': context['row_id'][te_idx],
            'N_MEGA': context['N_MEGA'][te_idx],
            'country': context['country'][te_idx],
            'y_true': y[te_idx],
            'y_pred': y_pred_fold,
            'status': status,
            'error': error,
            'predictors_used_n': len(pred_used_idx),
            'predictors_unresolved_n': 0,
            'predictors_dropped_zero_var_n': dropped_zero_n,
            'predictors_identity': str(rep_row.get('predictors_identity', '')),
            'predictors_used': '|'.join(str(int(i)) for i in pred_used_idx),
        })
        fold_df['scored'] = fold_df['y_true'].notna() & fold_df['y_pred'].notna()
        fold_pred_chunks.append(fold_df)

        met = regression_metrics_extended(
            fold_df.loc[fold_df['scored'], 'y_true'].values,
            fold_df.loc[fold_df['scored'], 'y_pred'].values,
            min_n=int(cfg.get('min_n_obs_for_metrics', 5)),
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

        fold_complexity_rows.append({
            'bag_target': context['bag_name'],
            'model_id': rep_id,
            'fold_country': c,
            'train_r2_full': train_r2_full,
            'train_r2_base': train_r2_base,
            'train_f2': train_f2,
            'train_bic_full': train_bic_full,
            'train_bic_base': train_bic_base,
            'train_bic_delta': (train_bic_full - train_bic_base) if pd.notna(train_bic_full) and pd.notna(train_bic_base) else np.nan,
            'train_sigma2_full': train_sigma2_full,
            'k_full': k_full,
            'pseudo_oos_ll_fold': pseudo_ll_fold,
            'pseudo_oos_bic_fold': pseudo_bic_fold,
            'n_test_scored_fold': n_test_scored_fold,
        })

    pred_df = pd.concat(fold_pred_chunks, ignore_index=True) if fold_pred_chunks else pd.DataFrame()
    country_df = pd.DataFrame(country_rows)
    complexity_df = pd.DataFrame(fold_complexity_rows)

    scored = pred_df[pred_df['scored']]
    g = regression_metrics_extended(
        scored['y_true'].values,
        scored['y_pred'].values,
        min_n=int(cfg.get('min_n_obs_for_metrics', 5)),
    )

    train_r2_mean = np.nan
    train_r2_base_mean = np.nan
    train_f2_mean = np.nan
    train_bic_mean = np.nan
    train_bic_base_mean = np.nan
    train_bic_delta_mean = np.nan
    pseudo_oos_ll = np.nan
    pseudo_oos_bic = np.nan
    pseudo_oos_scored_n = 0
    complexity_folds_n = 0
    if not complexity_df.empty:
        complexity_folds_n = int(len(complexity_df))
        train_r2_mean = nanmean_safe(complexity_df['train_r2_full'])
        train_r2_base_mean = nanmean_safe(complexity_df['train_r2_base'])
        train_f2_mean = nanmean_safe(complexity_df['train_f2'])
        train_bic_mean = nanmean_safe(complexity_df['train_bic_full'])
        train_bic_base_mean = nanmean_safe(complexity_df['train_bic_base'])
        train_bic_delta_mean = nanmean_safe(complexity_df['train_bic_delta'])

        ll_vals = pd.to_numeric(complexity_df['pseudo_oos_ll_fold'], errors='coerce')
        bic_vals = pd.to_numeric(complexity_df['pseudo_oos_bic_fold'], errors='coerce')
        n_vals = pd.to_numeric(complexity_df['n_test_scored_fold'], errors='coerce').fillna(0).astype(int)
        if ll_vals.notna().any():
            pseudo_oos_ll = float(np.nansum(ll_vals.values))
        if bic_vals.notna().any():
            pseudo_oos_bic = float(np.nansum(bic_vals.values))
        pseudo_oos_scored_n = int(n_vals.sum())

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
        'train_r2_adj_mean': nanmean_safe(train_adj_vals),
        'train_r2_mean': train_r2_mean,
        'train_r2_base_mean': train_r2_base_mean,
        'train_f2_mean': train_f2_mean,
        'train_bic_mean': train_bic_mean,
        'train_bic_base_mean': train_bic_base_mean,
        'train_bic_delta_mean': train_bic_delta_mean,
        'pseudo_oos_ll': pseudo_oos_ll,
        'pseudo_oos_bic': pseudo_oos_bic,
        'pseudo_oos_scored_n': pseudo_oos_scored_n,
        'complexity_folds_n': complexity_folds_n,
    }
    return summary, pred_df, country_df


def fit_predict_baseline_across_folds(context, matrix_cfg):
    cfg = context['analysis_cfg']
    cv_cfg = context['cv_cfg']

    y = context['y']
    age = context['age']
    sex = context['sex']
    diag = context['diag']
    edu = context['edu']
    scanner = context['scanner']

    fold_pred_chunks = []
    country_rows = []
    train_adj_vals = []
    fold_complexity_rows = []

    for c in context['countries']:
        tr_idx_full = context['train_idx_by_country'][c]
        te_idx = context['test_idx_by_country'][c]

        train_ok = context['base_req_train']
        tr_idx = tr_idx_full[train_ok[tr_idx_full]]

        y_pred_fold = np.full(len(te_idx), np.nan, dtype=float)
        status = 'ok'
        error = ''
        train_r2 = np.nan
        train_bic = np.nan
        train_sigma2 = np.nan
        k_base = np.nan
        pseudo_ll_fold = np.nan
        pseudo_bic_fold = np.nan
        n_test_scored_fold = 0

        if len(tr_idx) < int(cfg.get('min_n_obs', 80)):
            status = 'failed'
            error = f'insufficient_train_rows<{int(cfg.get("min_n_obs", 80))}'

        if status == 'ok':
            ytr = y[tr_idx]

            Xtr_parts = [np.ones((len(tr_idx), 1), dtype=float), age[tr_idx].reshape(-1, 1)]
            Xte_parts = [np.ones((len(te_idx), 1), dtype=float), age[te_idx].reshape(-1, 1)]

            sex_used = False
            sex_levels = []
            if cfg.get('include_sex', True):
                sx_tr = sex[tr_idx]
                sex_levels = sorted(pd.Series(sx_tr).astype(str).unique().tolist())
                if len(sex_levels) > 1:
                    sex_used = True
                    s_tr, s_te, _ = encode_dummies(sx_tr, sex[te_idx])
                    Xtr_parts.append(s_tr)
                    Xte_parts.append(s_te)

            if cfg.get('include_year', True):
                yb = context['year_basis_by_country'][c]
                Xtr_parts.append(yb[tr_idx])
                Xte_parts.append(yb[te_idx])

            diag_used = False
            diag_levels = []
            if cfg.get('include_diagnosis', True):
                dg_tr = diag[tr_idx]
                diag_levels = sorted(pd.Series(dg_tr).astype(str).unique().tolist())
                if len(diag_levels) > 1:
                    diag_used = True
                    Dtr, Dte, _ = encode_dummies(dg_tr, diag[te_idx])
                    Xtr_parts.append(Dtr)
                    Xte_parts.append(Dte)

            if cfg.get('include_education', False):
                Xtr_parts.append(edu[tr_idx].reshape(-1, 1))
                Xte_parts.append(edu[te_idx].reshape(-1, 1))

            scanner_used = False
            scanner_levels = []
            if cfg.get('include_scanner', False):
                sc_tr = scanner[tr_idx]
                scanner_levels = sorted(pd.Series(sc_tr).astype(str).unique().tolist())
                if len(scanner_levels) > 1:
                    scanner_used = True
                    sc_tr_enc, sc_te_enc, _ = encode_dummies(sc_tr, scanner[te_idx])
                    Xtr_parts.append(sc_tr_enc)
                    Xte_parts.append(sc_te_enc)

            Xtr = np.hstack(Xtr_parts)
            Xte = np.hstack(Xte_parts)

            try:
                beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=matrix_cfg.get('rcond', None))
                ytr_hat = Xtr @ beta
                train_adj_vals.append(adj_r2(ytr, ytr_hat, p=max(0, Xtr.shape[1] - 1)))
                k_base = int(Xtr.shape[1])
                train_r2 = r2_from_predictions(ytr, ytr_hat, min_n=2)
                train_bic = gaussian_bic(ytr, ytr_hat, k=k_base)
                train_sigma2 = gaussian_sigma2(ytr, ytr_hat)
            except Exception as e:
                status = 'failed'
                error = f'fit_failed::{repr(e)}'
                beta = None

            if status == 'ok':
                test_mask = context['base_req_test'][te_idx]
                if sex_used:
                    test_mask &= np.isin(sex[te_idx], sex_levels)
                if diag_used and str(cv_cfg.get('unseen_diag_policy', 'drop_test_rows')) == 'drop_test_rows':
                    test_mask &= np.isin(diag[te_idx], diag_levels)
                if scanner_used:
                    test_mask &= np.isin(scanner[te_idx], scanner_levels)
                if bool(np.any(test_mask)):
                    try:
                        y_pred_fold[test_mask] = Xte[test_mask] @ beta
                        yt_te = y[te_idx][test_mask]
                        yp_te = y_pred_fold[test_mask]
                        pseudo_ll_fold, n_test_scored_fold = gaussian_loglik(yt_te, yp_te, train_sigma2)
                        if n_test_scored_fold > 0 and np.isfinite(pseudo_ll_fold) and np.isfinite(k_base):
                            pseudo_bic_fold = float(-2.0 * pseudo_ll_fold + float(k_base) * np.log(max(2, n_test_scored_fold)))
                    except Exception as e:
                        status = 'failed'
                        error = f'predict_failed::{repr(e)}'

        fold_df = pd.DataFrame({
            'bag_target': context['bag_name'],
            'model_id': BASELINE_MODEL_ID,
            'fold_country': c,
            'row_id': context['row_id'][te_idx],
            'N_MEGA': context['N_MEGA'][te_idx],
            'country': context['country'][te_idx],
            'y_true': y[te_idx],
            'y_pred': y_pred_fold,
            'status': status,
            'error': error,
            'predictors_used_n': 0,
            'predictors_unresolved_n': 0,
            'predictors_dropped_zero_var_n': 0,
            'predictors_identity': '',
            'predictors_used': '',
        })
        fold_df['scored'] = fold_df['y_true'].notna() & fold_df['y_pred'].notna()
        fold_pred_chunks.append(fold_df)

        met = regression_metrics_extended(
            fold_df.loc[fold_df['scored'], 'y_true'].values,
            fold_df.loc[fold_df['scored'], 'y_pred'].values,
            min_n=int(cfg.get('min_n_obs_for_metrics', 5)),
        )
        country_rows.append({
            'bag_target': context['bag_name'],
            'model_id': BASELINE_MODEL_ID,
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
        fold_complexity_rows.append({
            'bag_target': context['bag_name'],
            'model_id': BASELINE_MODEL_ID,
            'fold_country': c,
            'train_r2': train_r2,
            'train_bic': train_bic,
            'train_sigma2': train_sigma2,
            'k_base': k_base,
            'pseudo_oos_ll_fold': pseudo_ll_fold,
            'pseudo_oos_bic_fold': pseudo_bic_fold,
            'n_test_scored_fold': n_test_scored_fold,
        })

    pred_df = pd.concat(fold_pred_chunks, ignore_index=True) if fold_pred_chunks else pd.DataFrame()
    country_df = pd.DataFrame(country_rows)
    complexity_df = pd.DataFrame(fold_complexity_rows)
    scored = pred_df[pred_df['scored']]
    g = regression_metrics_extended(
        scored['y_true'].values,
        scored['y_pred'].values,
        min_n=int(cfg.get('min_n_obs_for_metrics', 5)),
    )

    train_r2_mean = np.nan
    train_bic_mean = np.nan
    pseudo_oos_ll = np.nan
    pseudo_oos_bic = np.nan
    pseudo_oos_scored_n = 0
    complexity_folds_n = 0
    if not complexity_df.empty:
        complexity_folds_n = int(len(complexity_df))
        train_r2_mean = nanmean_safe(complexity_df['train_r2'])
        train_bic_mean = nanmean_safe(complexity_df['train_bic'])
        ll_vals = pd.to_numeric(complexity_df['pseudo_oos_ll_fold'], errors='coerce')
        bic_vals = pd.to_numeric(complexity_df['pseudo_oos_bic_fold'], errors='coerce')
        n_vals = pd.to_numeric(complexity_df['n_test_scored_fold'], errors='coerce').fillna(0).astype(int)
        if ll_vals.notna().any():
            pseudo_oos_ll = float(np.nansum(ll_vals.values))
        if bic_vals.notna().any():
            pseudo_oos_bic = float(np.nansum(bic_vals.values))
        pseudo_oos_scored_n = int(n_vals.sum())

    summary = {
        'bag_target': context['bag_name'],
        'model_id': BASELINE_MODEL_ID,
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
        'train_r2_adj_mean': nanmean_safe(train_adj_vals),
        'train_r2_mean': train_r2_mean,
        'train_r2_base_mean': np.nan,
        'train_f2_mean': np.nan,
        'train_bic_mean': train_bic_mean,
        'train_bic_base_mean': np.nan,
        'train_bic_delta_mean': np.nan,
        'pseudo_oos_ll': pseudo_oos_ll,
        'pseudo_oos_bic': pseudo_oos_bic,
        'pseudo_oos_scored_n': pseudo_oos_scored_n,
        'complexity_folds_n': complexity_folds_n,
    }
    return summary, pred_df, country_df


def compute_paired_delta_tables(base_summary_df, base_pred_df, baseline_summary_df, baseline_pred_df, compare_cfg, include_bic: bool = True):
    if base_summary_df is None or base_summary_df.empty or base_pred_df is None or base_pred_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if baseline_pred_df is None or baseline_pred_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    top_n = int(compare_cfg.get('thin_top_n_per_rank', 20))
    save_thin = bool(compare_cfg.get('save_thin_paired_predictions', True))
    min_n = int(compare_cfg.get('min_n_obs_for_metrics', 5))
    n_jobs = int(compare_cfg.get('paired_delta_n_jobs', 1) or 1)
    if n_jobs <= 0:
        n_jobs = max(1, os.cpu_count() or 1)
    backend = str(compare_cfg.get('paired_delta_backend', 'loky'))
    verbose = bool(compare_cfg.get('paired_delta_verbose', True))
    axis_req = str(compare_cfg.get('paired_delta_parallel_axis', 'auto')).strip().lower()
    chunk_models = int(compare_cfg.get('paired_delta_chunk_models', 128) or 128)
    if chunk_models < 1:
        chunk_models = 1
    models_backend = str(compare_cfg.get('paired_delta_models_backend', 'threading'))

    baseline_core = baseline_pred_df[['bag_target', 'row_id', 'country', 'y_true', 'y_pred']].copy()
    baseline_core = baseline_core.rename(columns={'y_pred': 'y_pred_base'})
    baseline_core = baseline_core.drop_duplicates(subset=['bag_target', 'row_id'], keep='first')

    summary_lookup_cols = [
        'bag_target', 'model_id', 'objective', 'metric', 'direction', 'order', 'score', 'rank',
        'global_oof_r2', 'global_oof_rmse', 'global_oof_mae', 'global_oof_corr2',
        'calib_slope', 'calib_intercept', 'bias_mean',
        'train_f2_mean',
        'n_total_with_y', 'n_scored', 'coverage_pct',
    ]
    if include_bic:
        summary_lookup_cols += ['train_bic_mean', 'pseudo_oos_bic']
    for c in summary_lookup_cols:
        if c not in base_summary_df.columns:
            base_summary_df[c] = np.nan
    summary_lookup = base_summary_df[summary_lookup_cols].copy()

    def _aggregate_metrics(df: pd.DataFrame, key_cols: List[str], pred_col: str, min_n_obs: int) -> pd.DataFrame:
        if df is None or df.empty:
            cols = list(key_cols) + ['n', 'r2', 'rmse', 'mae', 'corr2', 'calib_slope', 'calib_intercept', 'bias_mean']
            return pd.DataFrame(columns=cols)
        z = df[list(key_cols) + ['y_true', pred_col]].copy()
        z['y_true'] = pd.to_numeric(z['y_true'], errors='coerce')
        z[pred_col] = pd.to_numeric(z[pred_col], errors='coerce')
        z = z[np.isfinite(z['y_true'].values) & np.isfinite(z[pred_col].values)].copy()
        if z.empty:
            cols = list(key_cols) + ['n', 'r2', 'rmse', 'mae', 'corr2', 'calib_slope', 'calib_intercept', 'bias_mean']
            return pd.DataFrame(columns=cols)

        y = z['y_true'].to_numpy(dtype=float)
        p = z[pred_col].to_numpy(dtype=float)
        z['_y2'] = y * y
        z['_p2'] = p * p
        z['_yp'] = y * p
        z['_ae'] = np.abs(y - p)

        g = z.groupby(key_cols, sort=False, observed=True).agg(
            n=('y_true', 'size'),
            sum_y=('y_true', 'sum'),
            sum_y2=('_y2', 'sum'),
            sum_p=(pred_col, 'sum'),
            sum_p2=('_p2', 'sum'),
            sum_yp=('_yp', 'sum'),
            sum_abs=('_ae', 'sum'),
        ).reset_index()

        n = pd.to_numeric(g['n'], errors='coerce').to_numpy(dtype=float)
        sum_y = pd.to_numeric(g['sum_y'], errors='coerce').to_numpy(dtype=float)
        sum_y2 = pd.to_numeric(g['sum_y2'], errors='coerce').to_numpy(dtype=float)
        sum_p = pd.to_numeric(g['sum_p'], errors='coerce').to_numpy(dtype=float)
        sum_p2 = pd.to_numeric(g['sum_p2'], errors='coerce').to_numpy(dtype=float)
        sum_yp = pd.to_numeric(g['sum_yp'], errors='coerce').to_numpy(dtype=float)
        sum_abs = pd.to_numeric(g['sum_abs'], errors='coerce').to_numpy(dtype=float)

        sst = sum_y2 - (sum_y * sum_y) / np.maximum(n, 1.0)
        sse = sum_y2 - 2.0 * sum_yp + sum_p2
        sse = np.maximum(sse, 0.0)
        rmse = np.sqrt(sse / np.maximum(n, 1.0))
        mae = sum_abs / np.maximum(n, 1.0)
        r2 = 1.0 - (sse / np.maximum(sst, 1e-12))

        cov = sum_yp - (sum_y * sum_p) / np.maximum(n, 1.0)
        var_p = sum_p2 - (sum_p * sum_p) / np.maximum(n, 1.0)
        corr2 = (cov * cov) / np.maximum(sst * var_p, 1e-12)
        corr2 = np.clip(corr2, 0.0, 1.0)

        slope = cov / np.maximum(var_p, 1e-12)
        ybar = sum_y / np.maximum(n, 1.0)
        pbar = sum_p / np.maximum(n, 1.0)
        intercept = ybar - slope * pbar
        bias = (sum_p - sum_y) / np.maximum(n, 1.0)

        valid = (n >= float(min_n_obs))
        valid_corr = valid & (sst > 0) & (var_p > 0)

        g['r2'] = np.where(valid & (sst > 0), r2, np.nan)
        g['rmse'] = np.where(valid, rmse, np.nan)
        g['mae'] = np.where(valid, mae, np.nan)
        g['corr2'] = np.where(valid_corr, corr2, np.nan)
        g['calib_slope'] = np.where(valid_corr, slope, np.nan)
        g['calib_intercept'] = np.where(valid_corr, intercept, np.nan)
        g['bias_mean'] = np.where(valid, bias, np.nan)

        return g[list(key_cols) + ['n', 'r2', 'rmse', 'mae', 'corr2', 'calib_slope', 'calib_intercept', 'bias_mean']]

    def _compute_model_chunk(chunk_ids, p_chunk: pd.DataFrame, n_total_chunk: pd.DataFrame, bag: str):
        if p_chunk is None or p_chunk.empty:
            return pd.DataFrame(), pd.DataFrame()

        g_full = _aggregate_metrics(p_chunk, ['model_id'], 'y_pred_full', min_n)
        g_base = _aggregate_metrics(p_chunk, ['model_id'], 'y_pred_base', min_n)
        g_full = g_full.rename(columns={
            'n': 'n_full', 'r2': 'r2_full_paired', 'rmse': 'rmse_full_paired', 'mae': 'mae_full_paired',
            'corr2': 'corr2_full_paired', 'calib_slope': 'calib_slope_full_paired',
            'calib_intercept': 'calib_intercept_full_paired', 'bias_mean': 'bias_full_paired',
        })
        g_base = g_base.rename(columns={
            'n': 'n_base', 'r2': 'r2_base_paired', 'rmse': 'rmse_base_paired', 'mae': 'mae_base_paired',
            'corr2': 'corr2_base_paired', 'calib_slope': 'calib_slope_base_paired',
            'calib_intercept': 'calib_intercept_base_paired', 'bias_mean': 'bias_base_paired',
        })
        mg = g_full.merge(g_base, on='model_id', how='inner').merge(n_total_chunk, on='model_id', how='left')
        if not mg.empty:
            mg['bag_target'] = str(bag)
            mg['n_paired_scored'] = pd.to_numeric(mg['n_full'], errors='coerce').fillna(0).astype(int)
            mg['paired_coverage_pct'] = pd.to_numeric(mg['n_paired_scored'], errors='coerce') / np.maximum(1.0, pd.to_numeric(mg['n_total_y'], errors='coerce').fillna(0.0))
            mg['delta_r2_paired'] = pd.to_numeric(mg['r2_full_paired'], errors='coerce') - pd.to_numeric(mg['r2_base_paired'], errors='coerce')
            mg['delta_rmse_paired'] = pd.to_numeric(mg['rmse_full_paired'], errors='coerce') - pd.to_numeric(mg['rmse_base_paired'], errors='coerce')
            mg['delta_mae_paired'] = pd.to_numeric(mg['mae_full_paired'], errors='coerce') - pd.to_numeric(mg['mae_base_paired'], errors='coerce')
            mg['delta_corr2_paired'] = pd.to_numeric(mg['corr2_full_paired'], errors='coerce') - pd.to_numeric(mg['corr2_base_paired'], errors='coerce')

        c_full = _aggregate_metrics(p_chunk, ['model_id', 'country'], 'y_pred_full', min_n).rename(columns={
            'n': 'n_full', 'r2': 'r2_full_paired', 'rmse': 'rmse_full_paired', 'mae': 'mae_full_paired',
            'corr2': 'corr2_full_paired',
        })
        c_base = _aggregate_metrics(p_chunk, ['model_id', 'country'], 'y_pred_base', min_n).rename(columns={
            'n': 'n_base', 'r2': 'r2_base_paired', 'rmse': 'rmse_base_paired', 'mae': 'mae_base_paired',
            'corr2': 'corr2_base_paired',
        })
        mc = c_full.merge(c_base, on=['model_id', 'country'], how='inner')
        if not mc.empty:
            mc['bag_target'] = str(bag)
            mc['n_paired_scored'] = pd.to_numeric(mc['n_full'], errors='coerce').fillna(0).astype(int)
            mc['delta_r2_paired'] = pd.to_numeric(mc['r2_full_paired'], errors='coerce') - pd.to_numeric(mc['r2_base_paired'], errors='coerce')
            mc['delta_rmse_paired'] = pd.to_numeric(mc['rmse_full_paired'], errors='coerce') - pd.to_numeric(mc['rmse_base_paired'], errors='coerce')
            mc['delta_mae_paired'] = pd.to_numeric(mc['mae_full_paired'], errors='coerce') - pd.to_numeric(mc['mae_base_paired'], errors='coerce')
            mc['delta_corr2_paired'] = pd.to_numeric(mc['corr2_full_paired'], errors='coerce') - pd.to_numeric(mc['corr2_base_paired'], errors='coerce')
        else:
            mc = pd.DataFrame(columns=['bag_target', 'model_id', 'country', 'n_paired_scored', 'delta_r2_paired', 'delta_rmse_paired', 'delta_mae_paired', 'delta_corr2_paired'])

        keep_global = [
            'bag_target', 'model_id',
            'r2_full_paired', 'r2_base_paired', 'delta_r2_paired',
            'rmse_full_paired', 'rmse_base_paired', 'delta_rmse_paired',
            'mae_full_paired', 'mae_base_paired', 'delta_mae_paired',
            'corr2_full_paired', 'corr2_base_paired', 'delta_corr2_paired',
            'n_paired_scored', 'paired_coverage_pct',
        ]
        keep_country = [
            'bag_target', 'model_id', 'country',
            'n_paired_scored',
            'r2_full_paired', 'r2_base_paired', 'delta_r2_paired',
            'rmse_full_paired', 'rmse_base_paired', 'delta_rmse_paired',
            'mae_full_paired', 'mae_base_paired', 'delta_mae_paired',
            'corr2_full_paired', 'corr2_base_paired', 'delta_corr2_paired',
        ]
        mg = mg[keep_global] if not mg.empty else pd.DataFrame(columns=keep_global)
        mc = mc[keep_country] if not mc.empty else pd.DataFrame(columns=keep_country)
        return mg, mc

    def _compute_one_bag(bag: str, parallel_in_bag=False):
        bsum = base_summary_df[base_summary_df['bag_target'].astype(str).eq(str(bag))].copy()
        bpred = base_pred_df[base_pred_df['bag_target'].astype(str).eq(str(bag))].copy()
        bb = baseline_core[baseline_core['bag_target'].astype(str).eq(str(bag))].copy()
        if bsum.empty or bpred.empty or bb.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        rank_full = bsum.sort_values(['global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[False, True, True]).copy()
        rank_full_ids = rank_full['model_id'].astype(str).head(top_n).tolist()

        f = bpred[['model_id', 'row_id', 'y_true', 'y_pred']].copy()
        f = f.rename(columns={'y_true': 'y_true_full', 'y_pred': 'y_pred_full'})
        f['model_id'] = f['model_id'].astype(str)
        f = f.drop_duplicates(subset=['model_id', 'row_id'], keep='first')

        pair = f.merge(
            bb[['row_id', 'country', 'y_true', 'y_pred_base']],
            on='row_id',
            how='inner',
        )
        if pair.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        y_true_full = pd.to_numeric(pair['y_true_full'], errors='coerce').to_numpy(dtype=float)
        y_true_base = pd.to_numeric(pair['y_true'], errors='coerce').to_numpy(dtype=float)
        pair['y_true'] = np.where(np.isfinite(y_true_full), y_true_full, y_true_base)

        pair['model_id'] = pair['model_id'].astype(str)
        pair['country'] = pair['country'].astype(str)
        n_total_y = (
            pair[np.isfinite(pd.to_numeric(pair['y_true'], errors='coerce').to_numpy(dtype=float))]
            .groupby('model_id', sort=False, observed=True)
            .size()
            .rename('n_total_y')
            .reset_index()
        )

        scored_mask = (
            np.isfinite(pd.to_numeric(pair['y_true'], errors='coerce').to_numpy(dtype=float))
            & np.isfinite(pd.to_numeric(pair['y_pred_full'], errors='coerce').to_numpy(dtype=float))
            & np.isfinite(pd.to_numeric(pair['y_pred_base'], errors='coerce').to_numpy(dtype=float))
        )
        p = pair.loc[scored_mask, ['model_id', 'row_id', 'country', 'y_true', 'y_pred_full', 'y_pred_base']].copy()
        if p.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        model_ids = sorted(p['model_id'].astype(str).unique().tolist())
        mg_parts = []
        mc_parts = []
        if parallel_in_bag and n_jobs > 1 and len(model_ids) > chunk_models:
            chunks = [model_ids[i:i + chunk_models] for i in range(0, len(model_ids), chunk_models)]
            p2 = p.copy()
            p2['model_id'] = p2['model_id'].astype(str)
            n_total2 = n_total_y.copy()
            n_total2['model_id'] = n_total2['model_id'].astype(str)
            id_set_list = [set(ch) for ch in chunks]

            def _chunk_runner(id_set):
                mask = p2['model_id'].isin(id_set)
                p_chunk = p2.loc[mask, ['model_id', 'row_id', 'country', 'y_true', 'y_pred_full', 'y_pred_base']].copy()
                n_chunk = n_total2[n_total2['model_id'].isin(id_set)].copy()
                return _compute_model_chunk(id_set, p_chunk, n_chunk, bag)

            try:
                out_chunks = Parallel(n_jobs=n_jobs, backend=models_backend, verbose=0)(
                    delayed(_chunk_runner)(id_set) for id_set in id_set_list
                )
            except Exception:
                out_chunks = [_chunk_runner(id_set) for id_set in id_set_list]

            for gch, cch in out_chunks:
                if gch is not None and not gch.empty:
                    mg_parts.append(gch)
                if cch is not None and not cch.empty:
                    mc_parts.append(cch)
        else:
            mg0, mc0 = _compute_model_chunk(model_ids, p, n_total_y, bag)
            if mg0 is not None and not mg0.empty:
                mg_parts.append(mg0)
            if mc0 is not None and not mc0.empty:
                mc_parts.append(mc0)

        mg = pd.concat(mg_parts, ignore_index=True) if mg_parts else pd.DataFrame()
        mc = pd.concat(mc_parts, ignore_index=True) if mc_parts else pd.DataFrame()
        if mg.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if not mc.empty:
            med = mc.groupby('model_id', as_index=False, sort=False)['delta_r2_paired'].median()
            med = med.rename(columns={'delta_r2_paired': 'median_country_delta_r2'})
            mg = mg.merge(med, on='model_id', how='left')
        else:
            mg['median_country_delta_r2'] = np.nan

        thin = pd.DataFrame()
        if save_thin and not mg.empty:
            rank_delta = mg.sort_values(
                ['delta_r2_paired', 'r2_full_paired', 'paired_coverage_pct'],
                ascending=[False, False, False],
            )
            rank_delta_ids = rank_delta['model_id'].astype(str).head(top_n).tolist()
            keep_ids = sorted(set(rank_full_ids).union(set(rank_delta_ids)))
            thin = p[p['model_id'].astype(str).isin(keep_ids)][['model_id', 'row_id', 'country', 'y_true', 'y_pred_base', 'y_pred_full']].copy()
            if not thin.empty:
                thin['bag_target'] = str(bag)
                thin = thin[['bag_target', 'model_id', 'row_id', 'country', 'y_true', 'y_pred_base', 'y_pred_full']]

        keep_global = [
            'bag_target', 'model_id',
            'r2_full_paired', 'r2_base_paired', 'delta_r2_paired',
            'rmse_full_paired', 'rmse_base_paired', 'delta_rmse_paired',
            'mae_full_paired', 'mae_base_paired', 'delta_mae_paired',
            'corr2_full_paired', 'corr2_base_paired', 'delta_corr2_paired',
            'n_paired_scored', 'paired_coverage_pct', 'median_country_delta_r2',
        ]
        mg = mg[keep_global]
        return mg, mc, thin

    delta_rows = []
    delta_country_rows = []
    thin_rows = []

    bag_names = sorted(base_summary_df['bag_target'].dropna().astype(str).unique().tolist())
    axis = axis_req
    if axis not in {'bags', 'models', 'auto'}:
        axis = 'auto'
    if axis == 'auto':
        axis = 'models' if (n_jobs > max(1, len(bag_names))) else 'bags'
    if verbose:
        print(f'[paired-delta] parallel axis={axis} n_jobs={n_jobs} bags={len(bag_names)}')

    if axis == 'bags' and n_jobs > 1 and len(bag_names) > 1:
        if verbose:
            print(f'[paired-delta] parallel over bags: n_jobs={n_jobs} backend={backend} bags={len(bag_names)}')
        try:
            out = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                delayed(_compute_one_bag)(bag, False) for bag in bag_names
            )
        except Exception:
            out = [_compute_one_bag(bag, False) for bag in bag_names]
    else:
        if axis == 'models' and verbose:
            print(f'[paired-delta] parallel within bag over model chunks: n_jobs={n_jobs} backend={models_backend} chunk_models={chunk_models}')
        out = [_compute_one_bag(bag, axis == 'models') for bag in bag_names]

    for d1, d2, d3 in out:
        if d1 is not None and not d1.empty:
            delta_rows.append(d1)
        if d2 is not None and not d2.empty:
            delta_country_rows.append(d2)
        if d3 is not None and not d3.empty:
            thin_rows.append(d3)

    delta_df = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()
    if delta_df.empty:
        delta_country_df = pd.concat(delta_country_rows, ignore_index=True) if delta_country_rows else pd.DataFrame()
        thin_df = pd.concat(thin_rows, ignore_index=True) if thin_rows else pd.DataFrame()
        return pd.DataFrame(), delta_country_df, thin_df

    delta_df = summary_lookup.merge(delta_df, on=['bag_target', 'model_id'], how='left')
    delta_country_df = pd.concat(delta_country_rows, ignore_index=True) if delta_country_rows else pd.DataFrame()
    thin_df = pd.concat(thin_rows, ignore_index=True) if thin_rows else pd.DataFrame(
        columns=['bag_target', 'model_id', 'row_id', 'country', 'y_true', 'y_pred_base', 'y_pred_full']
    )
    return delta_df, delta_country_df, thin_df


def build_dual_rankings(base_summary_df, delta_summary_df):
    rank_full_df = pd.DataFrame()
    rank_delta_df = pd.DataFrame()

    if base_summary_df is not None and not base_summary_df.empty:
        s = base_summary_df.copy()
        s = s.sort_values(['bag_target', 'global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[True, False, True, True])
        s['full_rank'] = s.groupby('bag_target').cumcount() + 1
        rank_full_df = s

    if delta_summary_df is not None and not delta_summary_df.empty:
        d = delta_summary_df.copy()
        d = d.sort_values(['bag_target', 'delta_r2_paired', 'r2_full_paired', 'paired_coverage_pct'], ascending=[True, False, False, False])
        d['delta_rank'] = d.groupby('bag_target').cumcount() + 1
        rank_delta_df = d

    return rank_full_df, rank_delta_df


def build_model_vs_baseline_tables(
    base_summary_df: pd.DataFrame,
    baseline_summary_df: pd.DataFrame,
    base_country_df: pd.DataFrame,
    baseline_country_df: pd.DataFrame,
    delta_summary_df: pd.DataFrame | None = None,
    delta_country_df: pd.DataFrame | None = None,
    include_bic: bool = True,
):
    if base_summary_df is None or base_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bsum = baseline_summary_df.copy() if baseline_summary_df is not None else pd.DataFrame()
    if bsum.empty:
        return pd.DataFrame(), pd.DataFrame()

    base_map_cols = [
        'bag_target',
        'global_oof_r2', 'global_oof_rmse', 'global_oof_mae', 'global_oof_corr2',
        'calib_slope', 'calib_intercept', 'bias_mean',
        'n_scored', 'coverage_pct',
    ]
    if include_bic:
        base_map_cols += ['train_bic_mean', 'pseudo_oos_bic']
    for c in base_map_cols:
        if c not in bsum.columns:
            bsum[c] = np.nan
    bmap = bsum[base_map_cols].drop_duplicates(subset=['bag_target'], keep='first').copy()
    rename_cols = {
        'global_oof_r2': 'global_oof_r2_base',
        'global_oof_rmse': 'global_oof_rmse_base',
        'global_oof_mae': 'global_oof_mae_base',
        'global_oof_corr2': 'global_oof_corr2_base',
        'calib_slope': 'calib_slope_base',
        'calib_intercept': 'calib_intercept_base',
        'bias_mean': 'bias_mean_base',
        'n_scored': 'n_scored_base',
        'coverage_pct': 'coverage_pct_base',
    }
    if include_bic:
        rename_cols.update({
            'train_bic_mean': 'train_bic_mean_base',
            'pseudo_oos_bic': 'pseudo_oos_bic_base',
        })
    bmap = bmap.rename(columns=rename_cols)

    g = base_summary_df.merge(bmap, on='bag_target', how='left')
    g['delta_r2_vs_base'] = pd.to_numeric(g.get('global_oof_r2', np.nan), errors='coerce') - pd.to_numeric(g.get('global_oof_r2_base', np.nan), errors='coerce')
    g['delta_rmse_vs_base'] = pd.to_numeric(g.get('global_oof_rmse', np.nan), errors='coerce') - pd.to_numeric(g.get('global_oof_rmse_base', np.nan), errors='coerce')
    g['delta_mae_vs_base'] = pd.to_numeric(g.get('global_oof_mae', np.nan), errors='coerce') - pd.to_numeric(g.get('global_oof_mae_base', np.nan), errors='coerce')
    g['delta_corr2_vs_base'] = pd.to_numeric(g.get('global_oof_corr2', np.nan), errors='coerce') - pd.to_numeric(g.get('global_oof_corr2_base', np.nan), errors='coerce')
    if include_bic:
        g['delta_train_bic_mean_vs_base'] = pd.to_numeric(g.get('train_bic_mean', np.nan), errors='coerce') - pd.to_numeric(g.get('train_bic_mean_base', np.nan), errors='coerce')
        g['delta_pseudo_oos_bic_vs_base'] = pd.to_numeric(g.get('pseudo_oos_bic', np.nan), errors='coerce') - pd.to_numeric(g.get('pseudo_oos_bic_base', np.nan), errors='coerce')

    if delta_summary_df is not None and not delta_summary_df.empty:
        keep = [
            'bag_target', 'model_id',
            'delta_r2_paired', 'delta_rmse_paired', 'delta_mae_paired',
            'r2_full_paired', 'r2_base_paired', 'paired_coverage_pct',
        ]
        for c in keep:
            if c not in delta_summary_df.columns:
                delta_summary_df[c] = np.nan
        g = g.merge(delta_summary_df[keep], on=['bag_target', 'model_id'], how='left')

    g = g.sort_values(['bag_target', 'global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[True, False, True, True])

    if base_country_df is None or base_country_df.empty or baseline_country_df is None or baseline_country_df.empty:
        return g.reset_index(drop=True), pd.DataFrame()

    bc = base_country_df.copy()
    bbc = baseline_country_df.copy()
    bbc = bbc.rename(columns={
        'fold_country': 'fold_country_base',
        'r2': 'r2_base',
        'rmse': 'rmse_base',
        'mae': 'mae_base',
        'corr2': 'corr2_base',
        'calib_slope': 'calib_slope_base',
        'calib_intercept': 'calib_intercept_base',
        'bias_mean': 'bias_mean_base',
        'n_test_total': 'n_test_total_base',
        'n_test_scored': 'n_test_scored_base',
        'coverage_pct': 'coverage_pct_base',
    })
    c = bc.merge(
        bbc[['bag_target', 'fold_country_base', 'r2_base', 'rmse_base', 'mae_base', 'corr2_base', 'calib_slope_base', 'calib_intercept_base', 'bias_mean_base', 'n_test_total_base', 'n_test_scored_base', 'coverage_pct_base']],
        left_on=['bag_target', 'fold_country'],
        right_on=['bag_target', 'fold_country_base'],
        how='left',
    )
    c = c.drop(columns=['fold_country_base'], errors='ignore')
    c['delta_r2_vs_base'] = pd.to_numeric(c.get('r2', np.nan), errors='coerce') - pd.to_numeric(c.get('r2_base', np.nan), errors='coerce')
    c['delta_rmse_vs_base'] = pd.to_numeric(c.get('rmse', np.nan), errors='coerce') - pd.to_numeric(c.get('rmse_base', np.nan), errors='coerce')
    c['delta_mae_vs_base'] = pd.to_numeric(c.get('mae', np.nan), errors='coerce') - pd.to_numeric(c.get('mae_base', np.nan), errors='coerce')
    c['delta_corr2_vs_base'] = pd.to_numeric(c.get('corr2', np.nan), errors='coerce') - pd.to_numeric(c.get('corr2_base', np.nan), errors='coerce')

    if delta_country_df is not None and not delta_country_df.empty:
        keepc = [
            'bag_target', 'model_id', 'country',
            'delta_r2_paired', 'delta_rmse_paired', 'delta_mae_paired',
            'n_paired_scored',
        ]
        for ccol in keepc:
            if ccol not in delta_country_df.columns:
                delta_country_df[ccol] = np.nan
        dc = delta_country_df.rename(columns={'country': 'fold_country'})
        keepc_renamed = [
            'bag_target', 'model_id', 'fold_country',
            'delta_r2_paired', 'delta_rmse_paired', 'delta_mae_paired',
            'n_paired_scored',
        ]
        c = c.merge(dc[keepc_renamed], on=['bag_target', 'model_id', 'fold_country'], how='left', suffixes=('', '_paired'))

    c = c.sort_values(['bag_target', 'model_id', 'fold_country']).reset_index(drop=True)
    return g.reset_index(drop=True), c


def expand_representatives_to_candidates(
    rep_summary_df,
    rep_pred_df,
    rep_country_df,
    cmap_df,
    include_bic: bool = True,
    include_predictions: bool = True,
):
    if cmap_df is None or cmap_df.empty:
        return rep_summary_df, rep_pred_df, rep_country_df

    rep_summary = rep_summary_df.rename(columns={'model_id': 'representative_id'})
    s = cmap_df.merge(rep_summary, on='representative_id', how='left', suffixes=('', '_rep'))

    bag_col = 'bag_target_rep' if 'bag_target_rep' in s.columns else ('bag_target' if 'bag_target' in s.columns else None)
    pu_col = 'predictors_used_n_rep' if 'predictors_used_n_rep' in s.columns else ('predictors_used_n' if 'predictors_used_n' in s.columns else None)
    pz_col = 'predictors_dropped_zero_var_n_rep' if 'predictors_dropped_zero_var_n_rep' in s.columns else ('predictors_dropped_zero_var_n' if 'predictors_dropped_zero_var_n' in s.columns else None)

    summary_data = {
        'bag_target': s[bag_col] if bag_col is not None else np.nan,
        'model_id': s['model_id'],
        'objective': s['objective'],
        'metric': s['metric'],
        'direction': s['direction'],
        'order': s['order'],
        'score': s['score'],
        'rank': s['rank'],
        'representative_id': s['representative_id'],
        'predictors_identity': s['predictors_identity'],
        'predictors_identity_n': s['predictors_identity_n'],
        'predictors_used_n': s[pu_col] if pu_col is not None else np.nan,
        'predictors_unresolved_n': 0,
        'predictors_dropped_zero_var_n': s[pz_col] if pz_col is not None else np.nan,
        'global_oof_r2': s['global_oof_r2'],
        'global_oof_rmse': s['global_oof_rmse'],
        'global_oof_mae': s['global_oof_mae'],
        'global_oof_corr2': s['global_oof_corr2'] if 'global_oof_corr2' in s.columns else np.nan,
        'calib_slope': s['calib_slope'] if 'calib_slope' in s.columns else np.nan,
        'calib_intercept': s['calib_intercept'] if 'calib_intercept' in s.columns else np.nan,
        'bias_mean': s['bias_mean'] if 'bias_mean' in s.columns else np.nan,
        'n_total_with_y': s['n_total_with_y'],
        'n_scored': s['n_scored'],
        'coverage_pct': s['coverage_pct'],
        'train_r2_adj_mean': s['train_r2_adj_mean'],
        'train_r2_mean': s['train_r2_mean'] if 'train_r2_mean' in s.columns else np.nan,
        'train_r2_base_mean': s['train_r2_base_mean'] if 'train_r2_base_mean' in s.columns else np.nan,
        'train_f2_mean': s['train_f2_mean'] if 'train_f2_mean' in s.columns else np.nan,
        'complexity_folds_n': s['complexity_folds_n'] if 'complexity_folds_n' in s.columns else np.nan,
    }
    if include_bic:
        summary_data.update({
            'train_bic_mean': s['train_bic_mean'] if 'train_bic_mean' in s.columns else np.nan,
            'train_bic_base_mean': s['train_bic_base_mean'] if 'train_bic_base_mean' in s.columns else np.nan,
            'train_bic_delta_mean': s['train_bic_delta_mean'] if 'train_bic_delta_mean' in s.columns else np.nan,
            'pseudo_oos_ll': s['pseudo_oos_ll'] if 'pseudo_oos_ll' in s.columns else np.nan,
            'pseudo_oos_bic': s['pseudo_oos_bic'] if 'pseudo_oos_bic' in s.columns else np.nan,
            'pseudo_oos_scored_n': s['pseudo_oos_scored_n'] if 'pseudo_oos_scored_n' in s.columns else np.nan,
        })
    summary_df = pd.DataFrame(summary_data)

    summary_df = summary_df.sort_values(['global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[False, True, True], na_position='last').reset_index(drop=True)
    summary_df['global_rank'] = np.arange(1, len(summary_df) + 1)

    if (not include_predictions) or rep_pred_df is None or rep_pred_df.empty:
        pred_df = pd.DataFrame(columns=[
            'bag_target', 'model_id', 'objective', 'metric', 'direction', 'order', 'score', 'rank',
            'fold_country', 'row_id', 'N_MEGA', 'country', 'y_true', 'y_pred', 'status', 'error',
            'predictors_identity', 'predictors_used', 'predictors_used_n', 'predictors_unresolved_n',
            'predictors_dropped_zero_var_n', 'scored',
        ])
    else:
        p = cmap_df[['model_id', 'representative_id', 'objective', 'metric', 'direction', 'order', 'score', 'rank', 'predictors_identity']].copy()
        rep_pred = rep_pred_df.rename(columns={'model_id': 'representative_id'})
        rep_pred = rep_pred.drop(columns=['predictors_identity'], errors='ignore')
        p = p.merge(rep_pred, on='representative_id', how='left')
        pred_df = p[[
            'bag_target', 'model_id', 'objective', 'metric', 'direction', 'order', 'score', 'rank',
            'fold_country', 'row_id', 'N_MEGA', 'country', 'y_true', 'y_pred', 'status', 'error',
            'predictors_identity', 'predictors_used', 'predictors_used_n', 'predictors_unresolved_n',
            'predictors_dropped_zero_var_n', 'scored',
        ]].copy()

    if rep_country_df is None or rep_country_df.empty:
        country_df = pd.DataFrame(columns=[
            'bag_target', 'model_id', 'objective', 'metric', 'direction', 'order', 'fold_country',
            'n_test_total', 'n_test_scored', 'coverage_pct', 'r2', 'rmse', 'mae', 'corr2', 'calib_slope', 'calib_intercept', 'bias_mean',
        ])
    else:
        c = cmap_df[['model_id', 'representative_id', 'objective', 'metric', 'direction', 'order']].copy()
        rep_country = rep_country_df.rename(columns={'model_id': 'representative_id'})
        c = c.merge(rep_country, on='representative_id', how='left')
        country_df = c[[
            'bag_target', 'model_id', 'objective', 'metric', 'direction', 'order', 'fold_country',
            'n_test_total', 'n_test_scored', 'coverage_pct', 'r2', 'rmse', 'mae', 'corr2', 'calib_slope', 'calib_intercept', 'bias_mean',
        ]].copy()

    return summary_df, pred_df, country_df


def evaluate_representatives_parallel(reps_df, context, perf_cfg, matrix_cfg, show_progress=True, collect_predictions: bool = True):
    rep_records = reps_df.to_dict('records')
    n_models = len(rep_records)
    chunk_size = int(max(1, perf_cfg.get('chunk_size_models', 32)))
    chunks = [rep_records[i:i + chunk_size] for i in range(0, n_models, chunk_size)]

    n_jobs = int(perf_cfg.get('n_jobs', 1))
    backend = str(perf_cfg.get('backend', 'loky'))

    summary_chunks = []
    pred_chunks = []
    country_chunks = []

    start = time.time()
    pbar = tqdm(total=n_models, disable=not show_progress, desc=f"LOCO[{context['bag_name']}] reps")

    for chunk_i, chunk in enumerate(chunks, start=1):
        t0 = time.time()
        effective_n_jobs = min(n_jobs, len(chunk))
        if effective_n_jobs <= 1:
            out = [fit_predict_one_rep_across_folds(rr, context, matrix_cfg) for rr in chunk]
        else:
            try:
                out = Parallel(n_jobs=effective_n_jobs, backend=backend, verbose=0)(
                    delayed(fit_predict_one_rep_across_folds)(rr, context, matrix_cfg) for rr in chunk
                )
            except Exception:
                out = [fit_predict_one_rep_across_folds(rr, context, matrix_cfg) for rr in chunk]

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


def identity_set(identity_key: str) -> set:
    if not isinstance(identity_key, str) or not identity_key.strip():
        return set()
    return set([x for x in identity_key.split('|') if x])


def jaccard(a: str, b: str) -> float:
    sa = identity_set(a)
    sb = identity_set(b)
    if len(sa) == 0 and len(sb) == 0:
        return 0.0
    den = len(sa.union(sb))
    if den == 0:
        return 0.0
    return len(sa.intersection(sb)) / den


def select_diverse_inputs(summary_df: pd.DataFrame, bag_target: str, fusion_cfg: dict) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    sub = summary_df[summary_df['bag_target'].astype(str).eq(str(bag_target))].copy()
    sub = sub.dropna(subset=['global_oof_r2']).copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(['global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[False, True, True])

    k_target = int(fusion_cfg.get('target_k_inputs', 16))
    thr = float(fusion_cfg.get('jaccard_threshold', 0.6))
    bucket_order = ['tc_max', 'tc_min', 'dtc_max', 'dtc_min', 's_max', 's_min', 'o_max', 'o_min']

    selected = []
    selected_ids = set()
    selected_keys = []

    def try_add(row, stage):
        rid = str(row['model_id'])
        if rid in selected_ids:
            return False
        key = str(row.get('predictors_identity', ''))
        sim = max([jaccard(key, kk) for kk in selected_keys], default=0.0)
        if sim >= thr:
            return False
        rr = row.copy()
        rr['selection_stage'] = stage
        rr['max_similarity_at_selection'] = sim
        selected.append(rr)
        selected_ids.add(rid)
        selected_keys.append(key)
        return True

    for b in bucket_order:
        bsub = sub[sub['objective'].astype(str).eq(b)]
        for _, row in bsub.iterrows():
            if try_add(row, 'coverage'):
                break

    if len(selected) < k_target:
        for _, row in sub.iterrows():
            if len(selected) >= k_target:
                break
            try_add(row, 'fill')

    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected).reset_index(drop=True)
    out['fusion_input_rank'] = np.arange(1, len(out) + 1)
    out['target_k_inputs'] = k_target
    out['jaccard_threshold'] = thr
    return out


def fit_simplex_weights(X, y, fusion_cfg):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return None, {'status': 'failed', 'message': 'invalid_design_matrix'}

    nonnegative = bool(fusion_cfg.get('nonnegative', True))
    sum_to_one = bool(fusion_cfg.get('sum_to_one', True))
    fit_intercept = bool(fusion_cfg.get('fit_intercept', False))
    if fit_intercept:
        return None, {'status': 'failed', 'message': 'fit_intercept=True not implemented by design'}

    n = X.shape[1]
    x0 = np.full(n, 1.0 / n, dtype=float)

    def obj(w):
        res = y - X.dot(w)
        return float(np.mean(res * res))

    bounds = [(0.0, None)] * n if nonnegative else None
    constraints = []
    if sum_to_one:
        constraints.append({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)})

    try:
        res = minimize(obj, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
        if not res.success or not np.all(np.isfinite(res.x)):
            return x0.copy(), {'status': 'fallback_uniform', 'message': f'optimizer_failed::{getattr(res, "message", "unknown")}' }

        w = np.asarray(res.x, dtype=float)
        if nonnegative:
            w = np.clip(w, 0.0, None)
        if sum_to_one:
            s = float(w.sum())
            if s > 0:
                w = w / s
            else:
                w = x0.copy()
        return w, {'status': 'ok', 'message': 'success'}
    except Exception as e:
        return x0.copy(), {'status': 'fallback_uniform', 'message': f'exception::{repr(e)}'}


def build_fusion_matrix(pred_df: pd.DataFrame, model_ids: List[str]):
    if pred_df is None or pred_df.empty or len(model_ids) == 0:
        return pd.DataFrame(), []

    sub = pred_df[pred_df['model_id'].isin(model_ids)].copy()
    if sub.empty:
        return pd.DataFrame(), []

    base = sub[['row_id', 'N_MEGA', 'country', 'y_true']].drop_duplicates(subset=['row_id']).set_index('row_id')
    X = sub.pivot_table(index='row_id', columns='model_id', values='y_pred', aggfunc='first')
    mat = base.join(X, how='left')

    cols = [m for m in model_ids if m in mat.columns]
    if not cols:
        return pd.DataFrame(), []

    mat = mat.dropna(subset=['y_true'] + cols).copy()
    if mat.empty:
        return pd.DataFrame(), cols

    return mat.reset_index(), cols


def summarize_fusion_predictions(pred_df: pd.DataFrame, min_n=5):
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = pred_df.copy()
    d['scored'] = d['y_true'].notna() & d['y_pred_fusion'].notna()

    g = regression_metrics_extended(
        d.loc[d['scored'], 'y_true'].values,
        d.loc[d['scored'], 'y_pred_fusion'].values,
        min_n=min_n,
    )
    summary = pd.DataFrame([{
        'global_oof_r2': g['r2'],
        'global_oof_rmse': g['rmse'],
        'global_oof_mae': g['mae'],
        'global_oof_corr2': g['corr2'],
        'calib_slope': g['calib_slope'],
        'calib_intercept': g['calib_intercept'],
        'bias_mean': g['bias_mean'],
        'n_scored': g['n_scored'],
        'n_total_with_y': int(d['y_true'].notna().sum()),
        'coverage_pct': float(g['n_scored'] / max(1, int(d['y_true'].notna().sum()))),
    }])

    rows = []
    for country, cc in d.groupby('country', dropna=False):
        m = regression_metrics_extended(
            cc.loc[cc['scored'], 'y_true'].values,
            cc.loc[cc['scored'], 'y_pred_fusion'].values,
            min_n=min_n,
        )
        rows.append({
            'country': str(country),
            'r2': m['r2'],
            'rmse': m['rmse'],
            'mae': m['mae'],
            'corr2': m['corr2'],
            'calib_slope': m['calib_slope'],
            'calib_intercept': m['calib_intercept'],
            'bias_mean': m['bias_mean'],
            'n_test_total': int(cc['y_true'].notna().sum()),
            'n_test_scored': int(m['n_scored']),
            'coverage_pct': float(m['n_scored'] / max(1, int(cc['y_true'].notna().sum()))),
        })
    return summary, pd.DataFrame(rows)


def checkpoint_paths(outdir: Path, bag: str):
    return {
        'json': Path(outdir) / f'run_checkpoint_{bag}.json',
        'timing': Path(outdir) / f'timing_profile_{bag}.csv',
        'base_summary': Path(outdir) / f'_base_summary_{bag}.csv',
        'base_pred': Path(outdir) / f'_base_pred_{bag}.csv',
        'base_country': Path(outdir) / f'_base_country_{bag}.csv',
        'base_rep_summary': Path(outdir) / f'_base_rep_summary_{bag}.csv',
        'baseline_summary': Path(outdir) / f'_baseline_summary_{bag}.csv',
        'baseline_pred': Path(outdir) / f'_baseline_pred_{bag}.csv',
        'baseline_country': Path(outdir) / f'_baseline_country_{bag}.csv',
    }


def run_loco_stage(
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: List[str],
    bag_targets: dict,
    outdir: Path,
    analysis_cfg: dict,
    cv_cfg: dict,
    perf_cfg: dict,
    matrix_cfg: dict,
    enable_progress=True,
    storage_cfg: dict | None = None,
    compare_cfg: dict | None = None,
):
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
        'pairing_rule': 'strict',
        'primary_claim_metric': 'delta_r2_pooled',
        'report_median_country_delta_r2': True,
        'save_thin_paired_predictions': True,
        'thin_top_n_per_rank': 20,
        'save_full_paired_predictions': False,
    }
    compare_cfg = {**compare_defaults, **({} if compare_cfg is None else dict(compare_cfg))}
    compare_cfg['min_n_obs_for_metrics'] = int(analysis_cfg.get('min_n_obs_for_metrics', 5))
    need_baseline = bool(compare_cfg.get('compute_baseline_oof', True))
    need_paired_delta = bool(compare_cfg.get('compute_paired_delta', True))
    compare_defer = bool(compare_cfg.get('compute_paired_after_all_bags', True)) and need_paired_delta
    if int(compare_cfg.get('paired_delta_n_jobs', 0) or 0) <= 0:
        compare_cfg['paired_delta_n_jobs'] = int(perf_cfg.get('n_jobs', 1))

    storage_cfg = {} if storage_cfg is None else dict(storage_cfg)
    use_checkpoint_io = bool(storage_cfg.get('write_checkpoint_files', perf_cfg.get('enable_resume', False)))
    write_per_bag_pred_ckpt = bool(storage_cfg.get('write_per_bag_prediction_checkpoints', False))
    write_per_bag_identity = bool(storage_cfg.get('write_per_bag_identity_map', False))
    write_loco_predictions = bool(storage_cfg.get('write_loco_predictions', False))
    write_compat_loco_predictions = bool(storage_cfg.get('write_compat_loco_predictions', False))
    keep_base_predictions_in_memory = bool(
        storage_cfg.get(
            'keep_base_predictions_in_memory',
            bool(write_loco_predictions or write_compat_loco_predictions),
        )
    )
    keep_baseline_predictions_in_memory = bool(storage_cfg.get('keep_baseline_predictions_in_memory', False))
    write_baseline_prediction_checkpoints = bool(storage_cfg.get('write_baseline_prediction_checkpoints', False))
    write_baseline_predictions = bool(storage_cfg.get('write_baseline_predictions', False))
    collect_model_predictions = bool(keep_base_predictions_in_memory or write_per_bag_pred_ckpt or need_paired_delta)
    collect_baseline_predictions = bool(keep_baseline_predictions_in_memory or write_baseline_prediction_checkpoints or need_paired_delta)

    all_base_pred = []
    all_base_summary = []
    all_base_country = []
    all_rep_summary = []
    all_identity = []
    all_baseline_summary = []
    all_baseline_pred = []
    all_baseline_country = []
    all_delta_summary = []
    all_delta_country = []
    all_paired_thin = []
    all_compare_base_pred = []
    all_compare_baseline_pred = []

    for bag_name, y_col in bag_targets.items():
        print('\n' + '=' * 80)
        print(f'LOCO stage -> BAG {bag_name} ({y_col})')

        cp_paths = checkpoint_paths(outdir, bag_name)
        cp = load_json(
            cp_paths['json'],
            {'bag_target': bag_name, 'base_done': False, 'baseline_done': False, 'updated_at': None},
        )

        need_full_pred_for_compare = bool(need_paired_delta)
        need_pred_ckpt_for_resume = bool(write_per_bag_pred_ckpt or need_full_pred_for_compare)

        can_resume = bool(
            use_checkpoint_io
            and perf_cfg.get('enable_resume', True)
            and cp.get('base_done', False)
            and cp_paths['base_summary'].exists()
            and (cp_paths['base_pred'].exists() if need_pred_ckpt_for_resume else True)
            and cp_paths['base_country'].exists()
            and cp_paths['base_rep_summary'].exists()
        )

        context = None
        timing_rows = []

        if can_resume:
            print(f'[{bag_name}] resume base from checkpoint')
            base_summary = pd.read_csv(cp_paths['base_summary'])
            base_pred = pd.read_csv(cp_paths['base_pred']) if cp_paths['base_pred'].exists() else pd.DataFrame()
            base_country = pd.read_csv(cp_paths['base_country'])
            rep_summary = pd.read_csv(cp_paths['base_rep_summary'])
            id_map = pd.read_csv(outdir / f'candidate_identity_map_{bag_name}.csv') if (outdir / f'candidate_identity_map_{bag_name}.csv').exists() else pd.DataFrame()
            elapsed = np.nan
        else:
            t0 = time.time()
            context = prepare_bag_context(model_df, y_col, bag_name, analysis_cfg, cv_cfg, exposome_cols)
            cmap_df, reps_df = make_candidate_identity_map(candidate_df, exposome_cols, enable_prune=bool(perf_cfg.get('enable_identity_prune', True)))

            rep_summary, rep_pred, rep_country = evaluate_representatives_parallel(
                reps_df,
                context,
                perf_cfg,
                matrix_cfg,
                show_progress=enable_progress,
                collect_predictions=collect_model_predictions,
            )
            base_summary, base_pred, base_country = expand_representatives_to_candidates(
                rep_summary,
                rep_pred,
                rep_country,
                cmap_df,
                include_predictions=collect_model_predictions,
            )
            id_map = cmap_df.copy()
            id_map['bag_target'] = bag_name

            elapsed = time.time() - t0

            if use_checkpoint_io:
                atomic_write_csv(base_summary, cp_paths['base_summary'])
                if write_per_bag_pred_ckpt and collect_model_predictions:
                    atomic_write_csv(base_pred, cp_paths['base_pred'])
                atomic_write_csv(base_country, cp_paths['base_country'])
                atomic_write_csv(rep_summary, cp_paths['base_rep_summary'])
                if write_per_bag_identity:
                    atomic_write_csv(id_map, outdir / f'candidate_identity_map_{bag_name}.csv')

            timing_rows.append({'bag_target': bag_name, 'stage': 'loco', 'elapsed_sec': elapsed})

            if use_checkpoint_io:
                cp['base_done'] = True
                cp['updated_at'] = now_iso()
                atomic_write_json(cp, cp_paths['json'])

        baseline_summary = pd.DataFrame()
        baseline_pred = pd.DataFrame()
        baseline_country = pd.DataFrame()
        if need_baseline:
            can_resume_baseline = bool(
                use_checkpoint_io
                and perf_cfg.get('enable_resume', True)
                and cp.get('baseline_done', False)
                and cp_paths['baseline_summary'].exists()
                and (cp_paths['baseline_pred'].exists() if write_baseline_prediction_checkpoints else True)
                and cp_paths['baseline_country'].exists()
            )
            if can_resume_baseline:
                baseline_summary = pd.read_csv(cp_paths['baseline_summary'])
                baseline_pred = pd.read_csv(cp_paths['baseline_pred']) if cp_paths['baseline_pred'].exists() else pd.DataFrame()
                baseline_country = pd.read_csv(cp_paths['baseline_country'])
            else:
                bt0 = time.time()
                if context is None:
                    context = prepare_bag_context(model_df, y_col, bag_name, analysis_cfg, cv_cfg, exposome_cols)
                bsum, bpred, bcountry = fit_predict_baseline_across_folds(context, matrix_cfg)
                baseline_summary = pd.DataFrame([bsum])
                baseline_pred = bpred
                baseline_country = bcountry

                if use_checkpoint_io:
                    atomic_write_csv(baseline_summary, cp_paths['baseline_summary'])
                    if write_baseline_prediction_checkpoints and collect_baseline_predictions:
                        atomic_write_csv(baseline_pred, cp_paths['baseline_pred'])
                    atomic_write_csv(baseline_country, cp_paths['baseline_country'])
                timing_rows.append({'bag_target': bag_name, 'stage': 'baseline_loco', 'elapsed_sec': time.time() - bt0})

                if use_checkpoint_io:
                    cp['baseline_done'] = True
                    cp['updated_at'] = now_iso()
                    atomic_write_json(cp, cp_paths['json'])

        if timing_rows and use_checkpoint_io:
            timing = pd.DataFrame(timing_rows)
            atomic_write_csv(timing, cp_paths['timing'])

        all_base_summary.append(base_summary)
        if keep_base_predictions_in_memory:
            all_base_pred.append(base_pred)
        all_base_country.append(base_country)
        all_rep_summary.append(rep_summary)
        all_identity.append(id_map)
        if not baseline_summary.empty:
            all_baseline_summary.append(baseline_summary)
        if keep_baseline_predictions_in_memory and not baseline_pred.empty:
            all_baseline_pred.append(baseline_pred)
        if not baseline_country.empty:
            all_baseline_country.append(baseline_country)

        if need_paired_delta and (not base_pred.empty) and (not baseline_pred.empty):
            if compare_defer:
                all_compare_base_pred.append(base_pred)
                all_compare_baseline_pred.append(baseline_pred)
            else:
                t_cmp = time.time()
                dsum_b, dcountry_b, thin_b = compute_paired_delta_tables(
                    base_summary,
                    base_pred,
                    baseline_summary,
                    baseline_pred,
                    compare_cfg,
                )
                if not dsum_b.empty:
                    all_delta_summary.append(dsum_b)
                if not dcountry_b.empty:
                    all_delta_country.append(dcountry_b)
                if not thin_b.empty:
                    all_paired_thin.append(thin_b)
                print(f'[{bag_name}] paired delta completed. rows={len(dsum_b)} elapsed_sec={time.time() - t_cmp:.1f}')

        print(f'[{bag_name}] LOCO completed. rows={len(base_summary)} elapsed_sec={elapsed}')

    base_pred_df = pd.concat(all_base_pred, ignore_index=True) if all_base_pred else pd.DataFrame()
    base_summary_df = pd.concat(all_base_summary, ignore_index=True) if all_base_summary else pd.DataFrame()
    base_country_df = pd.concat(all_base_country, ignore_index=True) if all_base_country else pd.DataFrame()
    rep_summary_df = pd.concat(all_rep_summary, ignore_index=True) if all_rep_summary else pd.DataFrame()
    identity_map_df = pd.concat(all_identity, ignore_index=True) if all_identity else pd.DataFrame()
    baseline_summary_df = pd.concat(all_baseline_summary, ignore_index=True) if all_baseline_summary else pd.DataFrame()
    baseline_pred_df = pd.concat(all_baseline_pred, ignore_index=True) if all_baseline_pred else pd.DataFrame()
    baseline_country_df = pd.concat(all_baseline_country, ignore_index=True) if all_baseline_country else pd.DataFrame()
    if need_paired_delta and compare_defer and all_compare_base_pred and all_compare_baseline_pred:
        t_cmp_all = time.time()
        if bool(compare_cfg.get('paired_delta_verbose', True)):
            print(f'[paired-delta] deferred compute start. bags={len(all_compare_base_pred)}')
        cmp_base_pred_df = pd.concat(all_compare_base_pred, ignore_index=True)
        cmp_baseline_pred_df = pd.concat(all_compare_baseline_pred, ignore_index=True)
        delta_summary_df, delta_country_df, paired_thin_df = compute_paired_delta_tables(
            base_summary_df,
            cmp_base_pred_df,
            baseline_summary_df,
            cmp_baseline_pred_df,
            compare_cfg,
        )
        if bool(compare_cfg.get('paired_delta_verbose', True)):
            print(f'[paired-delta] deferred compute done. rows={len(delta_summary_df)} elapsed_sec={time.time() - t_cmp_all:.1f}')
    else:
        delta_summary_df = pd.concat(all_delta_summary, ignore_index=True) if all_delta_summary else pd.DataFrame()
        delta_country_df = pd.concat(all_delta_country, ignore_index=True) if all_delta_country else pd.DataFrame()
        paired_thin_df = pd.concat(all_paired_thin, ignore_index=True) if all_paired_thin else pd.DataFrame()
    rank_full_df, rank_delta_df = build_dual_rankings(base_summary_df, delta_summary_df)
    model_vs_base_global_df, model_vs_base_country_df = build_model_vs_baseline_tables(
        base_summary_df,
        baseline_summary_df,
        base_country_df,
        baseline_country_df,
        delta_summary_df=delta_summary_df,
        delta_country_df=delta_country_df,
    )

    loco_summary_with_calibration_complexity_df = model_vs_base_global_df.copy() if not model_vs_base_global_df.empty else base_summary_df.copy()

    print('LOCO aggregation completed. Writing final stage tables...')

    # Explicit stage outputs
    if write_loco_predictions:
        atomic_write_csv(base_pred_df, outdir / 'loco_oof_predictions.csv')
    atomic_write_csv(base_summary_df, outdir / 'loco_summary.csv')
    atomic_write_csv(base_country_df, outdir / 'loco_country_metrics.csv')
    atomic_write_csv(baseline_summary_df, outdir / 'loco_base_oof_summary.csv')
    atomic_write_csv(baseline_country_df, outdir / 'loco_base_oof_country_metrics.csv')
    if write_baseline_predictions and not baseline_pred_df.empty:
        atomic_write_csv(baseline_pred_df, outdir / 'loco_base_oof_predictions.csv')
    atomic_write_csv(delta_summary_df, outdir / 'loco_delta_summary.csv')
    atomic_write_csv(delta_country_df, outdir / 'loco_delta_country_metrics.csv')
    atomic_write_csv(model_vs_base_global_df, outdir / 'loco_model_vs_baseline_global.csv')
    atomic_write_csv(model_vs_base_country_df, outdir / 'loco_model_vs_baseline_country.csv')
    atomic_write_csv(loco_summary_with_calibration_complexity_df, outdir / 'loco_summary_with_calibration_complexity.csv')
    atomic_write_csv(rank_full_df, outdir / 'loco_rank_full.csv')
    atomic_write_csv(rank_delta_df, outdir / 'loco_rank_delta.csv')
    if bool(compare_cfg.get('save_thin_paired_predictions', True)):
        atomic_write_csv(paired_thin_df, outdir / 'loco_paired_predictions_thin.csv')

    # Compatibility outputs
    if write_compat_loco_predictions:
        atomic_write_csv(base_pred_df, outdir / 'base_model_loco_oof_predictions.csv')
    atomic_write_csv(base_summary_df, outdir / 'base_model_loco_summary.csv')
    atomic_write_csv(base_country_df, outdir / 'base_model_loco_country_metrics.csv')

    # Extra traces
    atomic_write_csv(rep_summary_df, outdir / 'base_model_representative_summary.csv')
    atomic_write_csv(identity_map_df, outdir / 'candidate_identity_map.csv')

    return {
        'base_pred_df': base_pred_df,
        'base_summary_df': base_summary_df,
        'base_country_df': base_country_df,
        'baseline_summary_df': baseline_summary_df,
        'baseline_pred_df': baseline_pred_df,
        'baseline_country_df': baseline_country_df,
        'delta_summary_df': delta_summary_df,
        'delta_country_df': delta_country_df,
        'model_vs_base_global_df': model_vs_base_global_df,
        'model_vs_base_country_df': model_vs_base_country_df,
        'loco_summary_with_calibration_complexity_df': loco_summary_with_calibration_complexity_df,
        'rank_full_df': rank_full_df,
        'rank_delta_df': rank_delta_df,
        'paired_thin_df': paired_thin_df,
        'rep_summary_df': rep_summary_df,
        'identity_map_df': identity_map_df,
    }


def run_fusion_posthoc_stage(
    outdir: Path,
    bag_targets: dict,
    fusion_cfg: dict,
    analysis_cfg: dict,
    skip_single=True,
    require_loco_outputs=True,
    enable_progress=True,
    base_summary_df: pd.DataFrame | None = None,
    base_pred_df: pd.DataFrame | None = None,
    baseline_summary_df: pd.DataFrame | None = None,
    storage_cfg: dict | None = None,
):
    outdir = Path(outdir)
    storage_cfg = {} if storage_cfg is None else dict(storage_cfg)
    write_fusion_predictions = bool(storage_cfg.get('write_fusion_predictions', False))

    has_memory_inputs = (
        base_summary_df is not None and not base_summary_df.empty
        and base_pred_df is not None and not base_pred_df.empty
    )

    if not has_memory_inputs:
        loco_summary_path = outdir / 'loco_summary.csv'
        loco_pred_path = outdir / 'loco_oof_predictions.csv'

        if not loco_summary_path.exists():
            loco_summary_path = outdir / 'base_model_loco_summary.csv'
        if not loco_pred_path.exists():
            loco_pred_path = outdir / 'base_model_loco_oof_predictions.csv'

        if require_loco_outputs and (not loco_summary_path.exists() or not loco_pred_path.exists()):
            raise FileNotFoundError('Fusion post-hoc requested but LOCO outputs are missing.')

        base_summary_df = pd.read_csv(loco_summary_path)
        base_pred_df = pd.read_csv(loco_pred_path)

    if baseline_summary_df is None or baseline_summary_df.empty:
        bsum_path = outdir / 'loco_base_oof_summary.csv'
        baseline_summary_df = pd.read_csv(bsum_path) if bsum_path.exists() else pd.DataFrame()

    selected_chunks = []
    pred_chunks = []
    summary_chunks = []
    country_chunks = []
    weights_chunks = []
    stage_rows = []

    bag_items = list(bag_targets.keys())
    pbar = tqdm(total=len(bag_items), disable=not enable_progress, desc='Fusion post-hoc')

    for bag in bag_items:
        t0 = time.time()
        selected = select_diverse_inputs(base_summary_df, bag, fusion_cfg)

        if selected.empty:
            stage_rows.append({'bag_target': bag, 'status': 'skipped_no_selected_inputs'})
            pbar.update(1)
            continue

        selected_chunks.append(selected)

        if skip_single and len(selected) == 1:
            stage_rows.append({'bag_target': bag, 'status': 'skipped_single_model_selected', 'selected_n': 1})
            pbar.update(1)
            continue

        model_ids = selected['model_id'].astype(str).tolist()
        sub_pred = base_pred_df[base_pred_df['bag_target'].astype(str).eq(str(bag))].copy()

        mat, model_cols = build_fusion_matrix(sub_pred, model_ids)
        if mat.empty or len(model_cols) == 0:
            stage_rows.append({'bag_target': bag, 'status': 'skipped_empty_fusion_matrix', 'selected_n': len(model_ids)})
            pbar.update(1)
            continue

        w, w_info = fit_simplex_weights(mat[model_cols].values, mat['y_true'].values, fusion_cfg)
        if w is None:
            stage_rows.append({'bag_target': bag, 'status': 'failed_weight_fit', 'selected_n': len(model_ids)})
            pbar.update(1)
            continue

        mat['y_pred_fusion'] = mat[model_cols].values.dot(np.asarray(w, dtype=float))
        pred_df = mat[['row_id', 'N_MEGA', 'country', 'y_true', 'y_pred_fusion']].copy()
        pred_df['bag_target'] = bag
        pred_df['scored'] = pred_df['y_true'].notna() & pred_df['y_pred_fusion'].notna()

        summary_df, country_df = summarize_fusion_predictions(pred_df, min_n=int(analysis_cfg.get('min_n_obs_for_metrics', 5)))
        if not summary_df.empty:
            summary_df['bag_target'] = bag
            summary_df['fusion_mode'] = 'posthoc'
            summary_df['selected_n'] = int(len(model_cols))
            summary_df['weight_fit_status'] = w_info.get('status', '')
            summary_df['weight_fit_message'] = w_info.get('message', '')

        if not country_df.empty:
            country_df['bag_target'] = bag

        weights_df = pd.DataFrame({
            'bag_target': bag,
            'model_id': model_cols,
            'weight': [float(x) for x in w],
            'weight_fit_status': w_info.get('status', ''),
            'weight_fit_message': w_info.get('message', ''),
        })

        pred_chunks.append(pred_df)
        summary_chunks.append(summary_df)
        country_chunks.append(country_df)
        weights_chunks.append(weights_df)

        stage_rows.append({
            'bag_target': bag,
            'status': 'ok',
            'selected_n': len(model_cols),
            'elapsed_sec': time.time() - t0,
        })
        pbar.set_postfix({'bag': bag, 'selected': len(model_cols)})
        pbar.update(1)

    pbar.close()

    fusion_selected_df = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
    fusion_pred_df = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()
    fusion_summary_df = pd.concat(summary_chunks, ignore_index=True) if summary_chunks else pd.DataFrame()
    fusion_country_df = pd.concat(country_chunks, ignore_index=True) if country_chunks else pd.DataFrame()
    fusion_weights_df = pd.concat(weights_chunks, ignore_index=True) if weights_chunks else pd.DataFrame()
    fusion_stage_df = pd.DataFrame(stage_rows)

    fusion_vs_rows = []
    for bag in sorted(fusion_summary_df['bag_target'].dropna().astype(str).unique().tolist()) if not fusion_summary_df.empty else []:
        ssub = base_summary_df[base_summary_df['bag_target'].astype(str).eq(str(bag))].copy()
        if ssub.empty:
            continue
        best = ssub.sort_values(['global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[False, True, True]).head(1).iloc[0]
        fsub = fusion_summary_df[fusion_summary_df['bag_target'].astype(str).eq(str(bag))].copy()
        if fsub.empty:
            continue
        frow = fsub.iloc[0]
        bsub = baseline_summary_df[baseline_summary_df['bag_target'].astype(str).eq(str(bag))].copy() if baseline_summary_df is not None else pd.DataFrame()
        brow = bsub.iloc[0] if not bsub.empty else None
        fusion_vs_rows.append({
            'bag_target': bag,
            'best_single_model_id': best['model_id'],
            'baseline_r2': brow['global_oof_r2'] if brow is not None and 'global_oof_r2' in brow else np.nan,
            'best_single_r2': best['global_oof_r2'],
            'fusion_posthoc_r2': frow['global_oof_r2'],
            'delta_best_single_minus_baseline_r2': (best['global_oof_r2'] - brow['global_oof_r2']) if brow is not None and pd.notna(brow['global_oof_r2']) and pd.notna(best['global_oof_r2']) else np.nan,
            'delta_fusion_minus_baseline_r2': (frow['global_oof_r2'] - brow['global_oof_r2']) if brow is not None and pd.notna(brow['global_oof_r2']) and pd.notna(frow['global_oof_r2']) else np.nan,
            'delta_fusion_minus_single_r2': (frow['global_oof_r2'] - best['global_oof_r2']) if pd.notna(frow['global_oof_r2']) and pd.notna(best['global_oof_r2']) else np.nan,
        })
    fusion_vs_df = pd.DataFrame(fusion_vs_rows)

    # Explicit stage outputs
    atomic_write_csv(fusion_selected_df, outdir / 'fusion_posthoc_selected_inputs.csv')
    if write_fusion_predictions:
        atomic_write_csv(fusion_pred_df, outdir / 'fusion_posthoc_predictions.csv')
    atomic_write_csv(fusion_summary_df, outdir / 'fusion_posthoc_summary.csv')
    atomic_write_csv(fusion_country_df, outdir / 'fusion_posthoc_country_metrics.csv')
    atomic_write_csv(fusion_weights_df, outdir / 'fusion_posthoc_weights.csv')
    atomic_write_csv(fusion_stage_df, outdir / 'fusion_posthoc_stage_log.csv')
    atomic_write_csv(fusion_vs_df, outdir / 'fusion_posthoc_vs_baseline_and_bestsingle.csv')

    # Compatibility outputs
    atomic_write_csv(fusion_selected_df, outdir / 'fusion_selected_inputs.csv')
    atomic_write_csv(fusion_summary_df, outdir / 'fusion_single_oof_diagnostic_summary.csv')

    return {
        'fusion_selected_df': fusion_selected_df,
        'fusion_pred_df': fusion_pred_df,
        'fusion_summary_df': fusion_summary_df,
        'fusion_country_df': fusion_country_df,
        'fusion_weights_df': fusion_weights_df,
        'fusion_stage_df': fusion_stage_df,
        'fusion_vs_df': fusion_vs_df,
    }


def render_stage_report(base_summary_df: pd.DataFrame, fusion_summary_df: pd.DataFrame):
    if base_summary_df is None or base_summary_df.empty:
        return pd.DataFrame()

    rows = []
    for bag in sorted(base_summary_df['bag_target'].dropna().astype(str).unique().tolist()):
        sub = base_summary_df[base_summary_df['bag_target'].astype(str).eq(str(bag))].copy()
        if sub.empty:
            continue
        best = sub.sort_values(['global_oof_r2', 'global_oof_rmse', 'global_oof_mae'], ascending=[False, True, True]).head(1).iloc[0]

        fsub = fusion_summary_df[fusion_summary_df['bag_target'].astype(str).eq(str(bag))].copy() if fusion_summary_df is not None and not fusion_summary_df.empty else pd.DataFrame()
        frow = fsub.iloc[0] if not fsub.empty else None

        rows.append({
            'bag_target': bag,
            'best_single_model_id': best['model_id'],
            'best_single_r2': best['global_oof_r2'],
            'best_single_rmse': best['global_oof_rmse'],
            'best_single_coverage': best['coverage_pct'],
            'fusion_posthoc_r2': frow['global_oof_r2'] if frow is not None else np.nan,
            'fusion_posthoc_rmse': frow['global_oof_rmse'] if frow is not None else np.nan,
            'fusion_posthoc_coverage': frow['coverage_pct'] if frow is not None else np.nan,
            'delta_r2_fusion_minus_single': (frow['global_oof_r2'] - best['global_oof_r2']) if frow is not None and pd.notna(frow['global_oof_r2']) else np.nan,
        })

    return pd.DataFrame(rows)
