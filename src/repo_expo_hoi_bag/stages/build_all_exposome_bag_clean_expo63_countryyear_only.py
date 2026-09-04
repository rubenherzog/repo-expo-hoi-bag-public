#!/usr/bin/env python3
"""
Create analysis-ready datasets with country-year-only imputation (no country/global fallback).

Inputs:
- data/all_exposome_bag_clean_expo63_from_nmega.csv  (raw integrated)
- data/exposome_feature_names.csv

Outputs:
1) keep-all rows after country-year-only imputation
2) complete-case rows only (drop rows with any remaining NA in the 63 features)
3) summary JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULTS = {
    "input_raw_csv": "data/all_exposome_bag_clean_expo63_from_nmega.csv",
    "feature_list_csv": "data/exposome_feature_names.csv",
    "out_keep_all_csv": "data/all_exposome_bag_clean_expo63_countryyear_only.csv",
    "out_complete_cases_csv": "data/all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv",
    "out_summary_json": "data/all_exposome_bag_clean_expo63_countryyear_only_summary.json",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Country-year-only imputation for expo63 integrated dataset.")
    p.add_argument("--input-raw-csv", type=str, default=DEFAULTS["input_raw_csv"])
    p.add_argument("--feature-list-csv", type=str, default=DEFAULTS["feature_list_csv"])
    p.add_argument("--out-keep-all-csv", type=str, default=DEFAULTS["out_keep_all_csv"])
    p.add_argument("--out-complete-cases-csv", type=str, default=DEFAULTS["out_complete_cases_csv"])
    p.add_argument("--out-summary-json", type=str, default=DEFAULTS["out_summary_json"])
    return p


def main() -> None:
    args = _build_parser().parse_args()

    input_raw_csv = Path(args.input_raw_csv)
    feature_list_csv = Path(args.feature_list_csv)
    out_keep_all_csv = Path(args.out_keep_all_csv)
    out_complete_cases_csv = Path(args.out_complete_cases_csv)
    out_summary_json = Path(args.out_summary_json)

    for p in [input_raw_csv, feature_list_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    for p in [out_keep_all_csv, out_complete_cases_csv, out_summary_json]:
        p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_raw_csv, low_memory=False)
    feats = pd.read_csv(feature_list_csv)
    if "feature_name" not in feats.columns:
        raise ValueError("feature_list_csv must contain 'feature_name'.")
    features = feats["feature_name"].astype(str).str.strip().tolist()
    features = [f for f in features if f]

    for c in ["Country", "Year"]:
        if c not in df.columns:
            raise ValueError(f"input_raw_csv missing required column '{c}'.")
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing target features in input_raw_csv: {missing_features[:10]}")

    out = df.copy()
    country = out["Country"].astype(str).str.strip()
    year = pd.to_numeric(out["Year"], errors="coerce").round().astype("Int64")

    cells_imputed_country_year = 0
    per_feature_counts = {}
    for f in features:
        s = pd.to_numeric(out[f], errors="coerce")
        grp = s.groupby([country, year]).transform("median")
        mask = s.isna() & grp.notna()
        n = int(mask.sum())
        cells_imputed_country_year += n
        per_feature_counts[f] = n
        out[f] = s.where(~mask, grp)

    miss_after = out[features].isna()
    keep_all = out
    complete_cases = out.loc[~miss_after.any(axis=1)].copy()

    keep_all.to_csv(out_keep_all_csv, index=False)
    complete_cases.to_csv(out_complete_cases_csv, index=False)

    summary = {
        "inputs": {
            "input_raw_csv": str(input_raw_csv.resolve()),
            "feature_list_csv": str(feature_list_csv.resolve()),
        },
        "outputs": {
            "out_keep_all_csv": str(out_keep_all_csv.resolve()),
            "out_complete_cases_csv": str(out_complete_cases_csv.resolve()),
            "out_summary_json": str(out_summary_json.resolve()),
        },
        "counts": {
            "rows_input": int(len(df)),
            "rows_keep_all": int(len(keep_all)),
            "rows_complete_cases": int(len(complete_cases)),
            "rows_complete_cases_rate": float(len(complete_cases) / len(df)),
            "cells_imputed_country_year": int(cells_imputed_country_year),
            "remaining_missing_cells": int(miss_after.sum().sum()),
            "remaining_rows_with_any_na": int(miss_after.any(axis=1).sum()),
            "n_features": int(len(features)),
        },
        "top_country_year_imputed_features": (
            pd.Series(per_feature_counts).sort_values(ascending=False).head(20).astype(int).to_dict()
        ),
    }

    with open(out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] keep-all country-year-only file: {out_keep_all_csv}")
    print(f"[DONE] complete-case file: {out_complete_cases_csv}")
    print(f"[DONE] summary: {out_summary_json}")
    print(
        "[SUMMARY] "
        f"rows_input={summary['counts']['rows_input']}, "
        f"rows_complete_cases={summary['counts']['rows_complete_cases']}, "
        f"cells_imputed_country_year={summary['counts']['cells_imputed_country_year']}, "
        f"remaining_rows_any_na={summary['counts']['remaining_rows_with_any_na']}"
    )


if __name__ == "__main__":
    main()
