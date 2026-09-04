#!/usr/bin/env python3
"""
Exposome-only THOI greedy runner (simple version, no imputation).

Outputs:
1) greedy_run_config.json
2) exposome_feature_names.csv
3) greedy_topk_by_objective_order.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scripts.pipeline_utils import env_bool, load_stage_config


# ============================================================================
# USER CONFIG (edit here)
# ============================================================================
BASE_USER_CONFIG = {
    # Data paths
    "input_csv": "data/raw/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
    "feature_list_csv": "data/exposome_feature_names.csv",
    "outdir": os.environ.get("V3_GREEDY_ROOT", "outputs/greedy"),
    # Schema
    "country_col": "country_clean",
    "year_acq_col": "exposome_year",
    "year_fallback_col": "Year",
    # Filtering
    "countries_to_remove": "",
    "outlier_ids": "",
    "year_min": None,
    "year_max": None,
    "min_country_size": 0,
    "initial_order": 3,
    "max_order": 30,
    "greedy_repeat": 40000,
    "batch_size": 1_000_000,
    "repeat_batch_size": 1000,
    "device": "cpu",
    "score_mode": os.environ.get("V3_GREEDY_SCORE_MODE", "oinfo"),
    "objectives": "tc_max,tc_min,dtc_max,dtc_min,s_max,s_min,o_max,o_min",
    "top_k_per_order": 1000,
}

SUPPORTED_METRICS = {"tc", "dtc", "s", "o"}
AUTO_FEATURE_LIST_TOKEN = "__from_input_columns__"


def _split_csv_list(text: str) -> List[str]:
    if text is None:
        return []
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _infer_year(df: pd.DataFrame, year_acq_col: str, year_fallback_col: str) -> pd.Series:
    min_year, max_year = 1900, 2100
    year_acq = pd.to_numeric(df.get(year_acq_col), errors="coerce")
    year_fb = pd.to_numeric(df.get(year_fallback_col), errors="coerce")
    serial_mask = year_acq > 3000
    if serial_mask.any():
        serial_dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(year_acq[serial_mask], unit="D")
        year_acq.loc[serial_mask] = serial_dates.dt.year.astype(float)
    year_clean = year_acq.where((year_acq >= min_year) & (year_acq <= max_year), np.nan)
    year_clean = year_clean.fillna(year_fb.where((year_fb >= min_year) & (year_fb <= max_year), np.nan))
    return year_clean


def _parse_objective(label: str) -> Tuple[str, str, bool]:
    s = str(label).strip().lower()
    parts = s.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid objective: {label!r}. Expected '<metric>_<min|max>'.")
    metric, direction = parts
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Invalid metric: {metric!r}. Supported: {sorted(SUPPORTED_METRICS)}")
    if direction not in {"min", "max"}:
        raise ValueError("Objective direction must be 'min' or 'max'.")
    return s, metric, direction == "max"


def _import_thoi_stack():
    """Import the custom greedy (in-repo) + O-information primitives (pip ``thoi``).

    The custom greedy lives in ``oinfo_bag_ladder.custom_greedy``; everything else
    comes from the upstream ``thoi`` package (see requirements.txt). No vendored
    ``thoi_branch`` is used.
    """
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to run greedy.") from exc
    try:
        from oinfo_bag_ladder.custom_greedy import (
            greedy_no_repeats as greedy_oinfo,
            greedy_no_repeats_os_ratio as greedy_os_ratio,
        )
    except Exception as exc:
        raise RuntimeError("Unable to import the custom greedy (oinfo_bag_ladder.custom_greedy).") from exc
    try:
        from thoi.commons import _normalize_input_data  # type: ignore
        from thoi.heuristics.scoring import _evaluate_nplets  # type: ignore
    except Exception as exc:
        raise RuntimeError("Unable to import THOI primitives — is the 'thoi' package installed?") from exc
    return torch, greedy_oinfo, greedy_os_ratio, _normalize_input_data, _evaluate_nplets


def prepare_exposome_matrix(
    input_csv: Path,
    feature_list_csv: Union[Path, str],
    country_col: str,
    year_acq_col: str,
    year_fallback_col: str,
    countries_to_remove: Sequence[str],
    outlier_ids: Sequence[str],
    year_min: Optional[int],
    year_max: Optional[int],
    min_country_size: int,
) -> Tuple[np.ndarray, pd.Index, Dict[str, Any]]:
    df = pd.read_csv(input_csv, low_memory=False)
    raw_input_columns = list(df.columns)
    row_id_col = "N_MEGA" if "N_MEGA" in df.columns else ("index" if "index" in df.columns else "__row_id__")
    if row_id_col == "__row_id__":
        df[row_id_col] = np.arange(len(df), dtype=np.int64)
    if country_col not in df.columns:
        raise ValueError(f"Missing country column: {country_col!r}.")
    auto_feature_list = str(feature_list_csv).strip().lower() == AUTO_FEATURE_LIST_TOKEN

    rows_input = int(len(df))
    df[row_id_col] = df[row_id_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=row_id_col, keep="first").copy()
    rows_after_dedup = int(len(df))

    df["country_clean"] = df[country_col].astype(str).str.strip()
    df.loc[df["country_clean"].eq(""), "country_clean"] = np.nan
    df["exposome_year"] = _infer_year(df, year_acq_col=year_acq_col, year_fallback_col=year_fallback_col)

    if countries_to_remove:
        df = df[~df["country_clean"].isin(countries_to_remove)].copy()
    if outlier_ids:
        df = df[~df[row_id_col].isin(outlier_ids)].copy()
    if (year_min is not None) or (year_max is not None):
        ymin = year_min if year_min is not None else -np.inf
        ymax = year_max if year_max is not None else np.inf
        df = df[df["exposome_year"].between(ymin, ymax, inclusive="both")].copy()

    df = df[df["country_clean"].notna() & df["exposome_year"].notna()].copy()
    if min_country_size > 0:
        counts = df["country_clean"].value_counts()
        keep = counts[counts >= int(min_country_size)].index
        df = df[df["country_clean"].isin(keep)].copy()
    if len(df) < 50:
        raise ValueError("Too few rows after filters.")

    if auto_feature_list:
        auto_exclude = {"index", "iso3", "Year", row_id_col, country_col, year_acq_col, year_fallback_col}
        feature_cols = [c for c in raw_input_columns if c not in auto_exclude]
    else:
        feature_list_path = Path(feature_list_csv)
        if not feature_list_path.exists():
            raise FileNotFoundError(f"feature_list_csv not found: {feature_list_path}")
        feat_df = pd.read_csv(feature_list_path)
        if "feature_name" not in feat_df.columns:
            raise ValueError("feature_list_csv must contain column 'feature_name'.")
        feature_cols = feat_df["feature_name"].astype(str).tolist()
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"{len(missing_features)} features from feature_list_csv are missing in input_csv. "
                f"First missing: {missing_features[:10]}"
            )

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Strict complete-case only (no imputation).
    X_df = X_df.dropna(axis=0, how="any").copy()
    finite_mask = np.isfinite(X_df.to_numpy(dtype=np.float64)).all(axis=1)
    keep_idx = X_df.index[finite_mask]
    X_df = X_df.loc[keep_idx].reset_index(drop=True)
    X = X_df.to_numpy(dtype=np.float64)

    summary = {
        "rows_input": rows_input,
        "row_id_col": row_id_col,
        "rows_after_dedup_nmega": rows_after_dedup,
        "rows_after_filters_pre_matrix": int(len(df)),
        "rows_model_complete_case": int(X.shape[0]),
        "countries_model": int(df.loc[keep_idx, "country_clean"].nunique()) if len(X_df) else 0,
        "n_features": int(X.shape[1]),
        "feature_list_csv": str(feature_list_csv),
        "missing_data_handling": "drop_rows_with_any_missing",
        "countries_to_remove": list(countries_to_remove),
        "outlier_ids": list(outlier_ids),
        "year_min": year_min,
        "year_max": year_max,
        "min_country_size": int(min_country_size),
    }
    return X, pd.Index(feature_cols), summary


def extract_ranked_nplets(
    best_nplets: Any,
    best_scores: Any,
    initial_order: int,
    largest: bool,
) -> pd.DataFrame:
    npl = _to_numpy(best_nplets)
    sc = _to_numpy(best_scores)
    if sc.ndim == 1:
        sc = sc[:, None]
    repeats, n_orders = sc.shape
    order_values = np.arange(int(initial_order), int(initial_order) + int(n_orders))
    records: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for r in range(repeats):
        row = npl[r]
        row = row[row >= 0] if np.any(row < 0) else row
        for oi, order_k in enumerate(order_values):
            key = tuple(sorted(int(i) for i in row[: int(order_k)]))
            score = float(sc[r, oi])
            if key not in records:
                records[key] = {"score": score, "order": int(order_k), "nplet_indices": list(key), "count": 1}
            else:
                records[key]["count"] += 1
                old_score = float(records[key]["score"])
                better = (score > old_score) if largest else (score < old_score)
                if better:
                    records[key]["score"] = score
    rows = [
        {
            "score": float(rec["score"]),
            "order": int(rec["order"]),
            "count": int(rec["count"]),
            "nplet_indices": rec["nplet_indices"],
        }
        for rec in records.values()
    ]
    df = pd.DataFrame(rows).sort_values("score", ascending=not largest).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _parse_nplet_indices(value: object) -> List[int]:
    if isinstance(value, list):
        return [int(i) for i in value]
    if isinstance(value, tuple):
        return [int(i) for i in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
        if isinstance(parsed, (list, tuple)):
            return [int(i) for i in parsed]
    return []


def _ensure_oinfo_sinfo_columns(
    df: pd.DataFrame,
    *,
    covmats: Any,
    T: Any,
    evaluate_nplets: Any,
    torch_mod: Any,
    device: Any,
    batch_size: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "thoi_o" not in out.columns:
        out["thoi_o"] = np.nan
    if "thoi_s" not in out.columns:
        out["thoi_s"] = np.nan

    needs_fill = out["thoi_o"].isna().any() or out["thoi_s"].isna().any()
    if not needs_fill:
        return out

    for order_val, group_idx in out.groupby("order", sort=False).groups.items():
        rows = list(group_idx)
        parsed = [_parse_nplet_indices(out.at[i, "nplet_indices"]) for i in rows]
        valid_pairs = [(i, idxs) for i, idxs in zip(rows, parsed) if len(idxs) == int(order_val)]
        if not valid_pairs:
            continue
        row_ids = [i for i, _ in valid_pairs]
        batch = torch_mod.as_tensor([idxs for _, idxs in valid_pairs], dtype=torch_mod.long, device=device)
        o_vals = evaluate_nplets(
            covmats,
            T,
            batch,
            metric="o",
            batch_size=int(batch_size),
            device=device,
        ).detach().cpu().numpy()
        s_vals = evaluate_nplets(
            covmats,
            T,
            batch,
            metric="s",
            batch_size=int(batch_size),
            device=device,
        ).detach().cpu().numpy()
        out.loc[row_ids, "thoi_o"] = o_vals
        out.loc[row_ids, "thoi_s"] = s_vals

    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run exposome-only THOI greedy (simple, no imputation).")
    # Use USER_CONFIG as base, and only override keys explicitly passed in CLI.
    ap.set_defaults(**{})
    ap.add_argument("--input-csv", type=str, dest="input_csv", default=argparse.SUPPRESS)
    ap.add_argument("--feature-list-csv", type=str, dest="feature_list_csv", default=argparse.SUPPRESS)
    ap.add_argument("--outdir", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--country-col", type=str, dest="country_col", default=argparse.SUPPRESS)
    ap.add_argument("--year-acq-col", type=str, dest="year_acq_col", default=argparse.SUPPRESS)
    ap.add_argument("--year-fallback-col", type=str, dest="year_fallback_col", default=argparse.SUPPRESS)
    ap.add_argument("--countries-to-remove", type=str, dest="countries_to_remove", default=argparse.SUPPRESS)
    ap.add_argument("--outlier-ids", type=str, dest="outlier_ids", default=argparse.SUPPRESS)
    ap.add_argument("--year-min", type=int, dest="year_min", default=argparse.SUPPRESS)
    ap.add_argument("--year-max", type=int, dest="year_max", default=argparse.SUPPRESS)
    ap.add_argument("--min-country-size", type=int, dest="min_country_size", default=argparse.SUPPRESS)
    ap.add_argument("--initial-order", type=int, dest="initial_order", default=argparse.SUPPRESS)
    ap.add_argument("--max-order", type=int, dest="max_order", default=argparse.SUPPRESS)
    ap.add_argument("--greedy-repeat", type=int, dest="greedy_repeat", default=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    ap.add_argument("--repeat-batch-size", type=int, dest="repeat_batch_size", default=argparse.SUPPRESS)
    ap.add_argument("--device", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--score-mode", type=str, dest="score_mode", default=argparse.SUPPRESS)
    ap.add_argument("--objectives", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--top-k-per-order", type=int, dest="top_k_per_order", default=argparse.SUPPRESS)
    ap.add_argument("--config-path", type=str, dest="config_path", default=argparse.SUPPRESS,
                   help="Optional path to a pipeline YAML config file.")
    return ap


def main() -> None:
    cli_overrides = vars(_build_parser().parse_args())
    config_path = cli_overrides.pop("config_path", None)
    smoke = env_bool("SMOKE_TEST")
    cfg = dict(BASE_USER_CONFIG)
    cfg.update(load_stage_config("greedy", smoke=smoke, config_path=config_path))
    cfg.update(cli_overrides)

    outdir = Path(cfg["outdir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    objectives = _split_csv_list(cfg["objectives"])
    if not objectives:
        raise ValueError("No objectives provided.")
    if int(cfg["top_k_per_order"]) <= 0:
        raise ValueError("--top-k-per-order must be > 0.")
    score_mode = str(cfg.get("score_mode", "oinfo")).strip().lower()
    if score_mode not in {"oinfo", "os_ratio"}:
        raise ValueError("Unsupported score_mode. Expected 'oinfo' or 'os_ratio'.")

    input_csv = Path(str(cfg["input_csv"])).resolve()
    feature_list_csv_cfg = str(cfg["feature_list_csv"]).strip()
    feature_list_csv = (
        Path(feature_list_csv_cfg).resolve()
        if feature_list_csv_cfg.lower() != AUTO_FEATURE_LIST_TOKEN
        else feature_list_csv_cfg
    )
    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    if isinstance(feature_list_csv, Path) and not feature_list_csv.exists():
        raise FileNotFoundError(f"feature_list_csv not found: {feature_list_csv}")

    hdr = pd.read_csv(input_csv, nrows=0)
    country_col = str(cfg["country_col"])
    year_acq_col = str(cfg["year_acq_col"])
    year_fallback_col = str(cfg["year_fallback_col"])
    for col_name, col_kind in [
        (country_col, "country_col"),
        (year_acq_col, "year_acq_col"),
        (year_fallback_col, "year_fallback_col"),
    ]:
        if col_name not in hdr.columns:
            raise ValueError(f"{col_kind}='{col_name}' not found in input CSV columns.")

    X, feature_names, prep_summary = prepare_exposome_matrix(
        input_csv=input_csv,
        feature_list_csv=feature_list_csv,
        country_col=country_col,
        year_acq_col=year_acq_col,
        year_fallback_col=year_fallback_col,
        countries_to_remove=_split_csv_list(cfg["countries_to_remove"]),
        outlier_ids=_split_csv_list(cfg["outlier_ids"]),
        year_min=cfg["year_min"],
        year_max=cfg["year_max"],
        min_country_size=int(cfg["min_country_size"]),
    )

    print(
        f"[INFO] prepared rows={X.shape[0]} features={X.shape[1]} "
        f"orders={cfg['initial_order']}-{cfg['max_order']} repeat={cfg['greedy_repeat']}",
        flush=True,
    )

    run_cfg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv.resolve()),
        "feature_list_csv": (
            str(feature_list_csv.resolve()) if isinstance(feature_list_csv, Path) else str(feature_list_csv)
        ),
        "outdir": str(outdir),
        "country_col": country_col,
        "year_acq_col": year_acq_col,
        "year_fallback_col": year_fallback_col,
        "countries_to_remove": _split_csv_list(cfg["countries_to_remove"]),
        "outlier_ids": _split_csv_list(cfg["outlier_ids"]),
        "year_min": cfg["year_min"],
        "year_max": cfg["year_max"],
        "min_country_size": int(cfg["min_country_size"]),
        "missing_data_handling": "drop_rows_with_any_missing",
        "initial_order": int(cfg["initial_order"]),
        "max_order": int(cfg["max_order"]),
        "greedy_repeat": int(cfg["greedy_repeat"]),
        "batch_size": int(cfg["batch_size"]),
        "repeat_batch_size": int(cfg["repeat_batch_size"]),
        "device": str(cfg["device"]),
        "score_mode": score_mode,
        "objectives": objectives,
        "top_k_per_order": int(cfg["top_k_per_order"]),
        "greedy_impl": "greedy_no_repeats_os_ratio" if score_mode == "os_ratio" else "greedy_no_repeats",
        "cli_overrides": cli_overrides,
        "prep_summary": prep_summary,
    }

    run_tag = (
        f"io{int(cfg['initial_order'])}"
        f"_mo{int(cfg['max_order'])}"
        f"_r{int(cfg['greedy_repeat'])}"
        f"_k{int(cfg['top_k_per_order'])}"
        f"_sm_{score_mode}"
        f"_impl_gnr"
    )
    checkpoint_dir = outdir / f"checkpoints_{run_tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_cfg["checkpoint_dir"] = str(checkpoint_dir)
    run_cfg["checkpoint_tag"] = run_tag

    with open(outdir / "greedy_run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    pd.DataFrame({"feature_name": feature_names.tolist()}).to_csv(
        outdir / "exposome_feature_names.csv",
        index=False,
    )

    torch, greedy_oinfo, greedy_os_ratio, normalize_input_data, evaluate_nplets = _import_thoi_stack()
    greedy = greedy_os_ratio if score_mode == "os_ratio" else greedy_oinfo
    covmats, _, _, T = normalize_input_data(X, False, None, torch.device(str(cfg["device"])))
    greedy_kwargs = {
        "initial_order": int(cfg["initial_order"]),
        "order": int(cfg["max_order"]),
        "repeat": int(cfg["greedy_repeat"]),
        "batch_size": int(cfg["batch_size"]),
        "repeat_batch_size": int(cfg["repeat_batch_size"]),
        "covmat_precomputed": False,
        "device": torch.device(str(cfg["device"])),
    }

    topk_chunks: List[pd.DataFrame] = []
    for label in objectives:
        obj_label, metric, largest = _parse_objective(label)
        objective_out = checkpoint_dir / f"greedy_topk_{obj_label}.csv"

        if objective_out.exists():
            chosen = pd.read_csv(objective_out)
            if "score_mode" not in chosen.columns:
                chosen["score_mode"] = score_mode
            chosen = _ensure_oinfo_sinfo_columns(
                chosen,
                covmats=covmats,
                T=T,
                evaluate_nplets=evaluate_nplets,
                torch_mod=torch,
                device=torch.device(str(cfg["device"])),
                batch_size=int(cfg["batch_size"]),
            )
            chosen.to_csv(objective_out, index=False)
            topk_chunks.append(
                chosen[[
                    "objective",
                    "metric",
                    "direction",
                    "order",
                    "rank",
                    "score",
                    "thoi_o",
                    "thoi_s",
                    "score_mode",
                    "count",
                    "nplet_indices",
                ]]
            )
            print(f"[INFO] resume {obj_label}: loaded existing ({len(chosen)} rows)", flush=True)
            continue

        print(f"[INFO] greedy {obj_label}...", flush=True)
        t0 = time.perf_counter()
        best_nplets, best_scores = greedy(
            X,
            metric=metric,
            largest=largest,
            **greedy_kwargs,
        )
        ranked_df = extract_ranked_nplets(
            best_nplets=best_nplets,
            best_scores=best_scores,
            initial_order=int(cfg["initial_order"]),
            largest=largest,
        )
        chosen = ranked_df.groupby("order", as_index=False, group_keys=False).head(int(cfg["top_k_per_order"])).copy()
        chosen = chosen.sort_values(["order", "score"], ascending=[True, not largest]).copy()
        chosen["rank"] = chosen.groupby("order").cumcount() + 1
        chosen.insert(0, "objective", obj_label)
        chosen.insert(1, "metric", metric)
        chosen.insert(2, "direction", "max" if largest else "min")
        chosen = _ensure_oinfo_sinfo_columns(
            chosen,
            covmats=covmats,
            T=T,
            evaluate_nplets=evaluate_nplets,
            torch_mod=torch,
            device=torch.device(str(cfg["device"])),
            batch_size=int(cfg["batch_size"]),
        )
        chosen["score_mode"] = score_mode
        chosen = chosen[[
            "objective",
            "metric",
            "direction",
            "order",
            "rank",
            "score",
            "thoi_o",
            "thoi_s",
            "score_mode",
            "count",
            "nplet_indices",
        ]]
        chosen.to_csv(objective_out, index=False)
        topk_chunks.append(chosen)
        elapsed_sec = time.perf_counter() - t0
        print(
            f"[INFO] done {obj_label}: selected={len(chosen)} | saved={objective_out.name} | elapsed={elapsed_sec:.1f}s",
            flush=True,
        )

    topk_df = pd.concat(topk_chunks, ignore_index=True).sort_values(["objective", "order", "rank"]).reset_index(drop=True)
    topk_df.to_csv(outdir / "greedy_topk_by_objective_order.csv", index=False)
    print(f"[DONE] saved {len(topk_df)} rows to greedy_topk_by_objective_order.csv", flush=True)


if __name__ == "__main__":
    main()
