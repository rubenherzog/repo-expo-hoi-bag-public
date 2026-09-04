#!/usr/bin/env python3
"""
Build a reproducible reduced exposome table from a subject-level file.

The goal is to collapse repeated rows to the effective unit for O-information
ranking. In this dataset the exposure columns are mostly country-year level,
but a small number of country-year blocks contain more than one distinct
exposure vector. Because of that, the script supports three grouping modes:

  - auto: choose the finest unit actually present in the data
  - country_year: collapse on country + year only
  - exposure_vector: collapse on the unique 63-feature exposure signature

The default is auto. The script writes a reduced CSV plus a JSON summary that
records the chosen unit and the candidate block counts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"
DEFAULT_FEATURES = REPO_ROOT / "data" / "metadata" / "exposome_feature_names.csv"
DEFAULT_OUTPUT = None

META_COLS = {
    "N_MEGA",
    "ID",
    "ID_original",
    "imag_ID",
    "otro_ID",
    "Acceso",
    "Dataset1",
    "Dataset",
    "Site",
    "Continent",
    "subregion_name",
    "Country",
    "Diagnosis",
    "Age",
    "Sex",
    "Edu",
    "Cognitive_score",
    "MCIfilter",
    "CDR",
    "year_acq",
    "Year",
    "INCOME",
    "T1",
    "rsfMRI",
    "rsEEG",
    "rsMEG",
    "FUENTES",
    "funcional",
    "OOSV_func",
    "SOCIOECONOM",
    "PHYSICAL",
    "DEMOCRACY",
    "OTROS",
    "BAG_struc",
    "BAG_func",
    "BAG_comb",
    "BAG_comb2",
    "MMSE_total",
    "OOSV_structural",
    "long_data",
    "BAG_OOS_functional",
    "BAG_OOS_structural",
    "country_clean",
    "exposome_year",
    "country_year",
    "expo63_match_found",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reduce the exposome table to its effective sample.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURES)
    p.add_argument(
        "--grouping",
        type=str,
        default="auto",
        choices=("auto", "country_year", "exposure_vector"),
        help="Reduction unit to use.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path for the reduced table. Defaults to the bundle public_input directory when available.",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional JSON summary path. Defaults beside the reduced CSV.",
    )
    p.add_argument(
        "--round-decimals",
        type=int,
        default=12,
        help="Decimal precision used to fingerprint exposure vectors.",
    )
    return p


def _read_feature_list(path: Path) -> list[str]:
    feat_df = pd.read_csv(path)
    if "feature_name" not in feat_df.columns:
        raise ValueError(f"{path} must contain a 'feature_name' column.")
    features = feat_df["feature_name"].astype(str).str.strip().tolist()
    return [f for f in features if f]


def _canonicalize_features(df: pd.DataFrame, features: Iterable[str], round_decimals: int) -> pd.DataFrame:
    num = df.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    return num.round(round_decimals)


def _fingerprint_frame(frame: pd.DataFrame) -> pd.Series:
    filled = frame.astype(object).where(frame.notna(), "__NA__")
    return filled.astype(str).agg("||".join, axis=1)


def _candidate_country_year_blocks(df: pd.DataFrame, exposure_fp: pd.Series) -> pd.DataFrame:
    tmp = df.loc[:, ["country_year", "country_clean", "exposome_year"]].copy()
    tmp["_expo_fp"] = exposure_fp.values
    return (
        tmp.groupby("country_year", dropna=False)
        .agg(
            n_rows=("country_year", "size"),
            n_exposure_vectors=("_expo_fp", "nunique"),
            n_countries=("country_clean", "nunique"),
        )
        .reset_index()
    )


def _choose_grouping(grouping: str, country_year_blocks: pd.DataFrame) -> str:
    if grouping != "auto":
        return grouping
    multi_vector_blocks = int((country_year_blocks["n_exposure_vectors"] > 1).sum())
    return "exposure_vector" if multi_vector_blocks > 0 else "country_year"


def main() -> None:
    args = _build_parser().parse_args()

    input_csv = Path(args.input_csv)
    feature_list_csv = Path(args.feature_list_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")
    if not feature_list_csv.exists():
        raise FileNotFoundError(f"Missing feature list CSV: {feature_list_csv}")

    if args.output_csv is None:
        repro_root = os.environ.get("REPRO_DATA_ROOT")
        if repro_root:
            default_out = Path(repro_root) / "public_input" / "all_exposome_bag_clean_expo63_effective_sample.csv"
        else:
            raise EnvironmentError(
                "REPRO_DATA_ROOT is required for generated reduced-data outputs unless "
                "--output-csv is provided explicitly."
            )
        output_csv = default_out
    else:
        output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    features = _read_feature_list(feature_list_csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing {len(missing)} feature columns. First missing: {missing[:10]}")

    if "country_year" not in df.columns:
        if "country_clean" not in df.columns or "exposome_year" not in df.columns:
            raise ValueError("Input must contain either country_year or both country_clean and exposome_year.")
        df["country_year"] = df["country_clean"].astype(str).str.strip() + "__" + df["exposome_year"].astype(str).str.strip()

    if "country_clean" not in df.columns:
        df["country_clean"] = df["country_year"].astype(str).str.split("__", n=1).str[0]
    if "exposome_year" not in df.columns:
        df["exposome_year"] = pd.to_numeric(df["country_year"].astype(str).str.split("__", n=1).str[1], errors="coerce")

    feature_frame = _canonicalize_features(df, features, int(args.round_decimals))
    exposure_fp = _fingerprint_frame(feature_frame)
    country_year_blocks = _candidate_country_year_blocks(df, exposure_fp)
    chosen = _choose_grouping(args.grouping, country_year_blocks)

    if chosen == "country_year":
        group_cols = ["country_year"]
    elif chosen == "exposure_vector":
        group_cols = ["_exposure_fp"]
    else:
        raise ValueError(f"Unsupported grouping: {chosen}")

    work = df.copy()
    work["_exposure_fp"] = exposure_fp
    work["_row_order"] = range(len(work))
    work = work.sort_values(group_cols + ["_row_order"], kind="mergesort").copy()

    summary_rows = []
    reduced_parts = []
    for key, sub in work.groupby(group_cols, dropna=False, sort=False):
        rep = sub.iloc[0].copy()
        summary_rows.append(
            {
                "group_id": key if isinstance(key, str) else "||".join(map(str, key)),
                "n_rows": int(len(sub)),
                "n_country_years": int(sub["country_year"].nunique(dropna=False)),
                "n_countries": int(sub["country_clean"].nunique(dropna=False)),
                "country_years": " ; ".join(sorted(sub["country_year"].astype(str).unique().tolist())),
                "countries": " ; ".join(sorted(sub["country_clean"].astype(str).unique().tolist())),
            }
        )
        rep["effective_grouping"] = chosen
        rep["effective_group_id"] = summary_rows[-1]["group_id"]
        rep["effective_group_rows"] = int(len(sub))
        rep["effective_group_country_years"] = int(sub["country_year"].nunique(dropna=False))
        rep["effective_group_countries"] = int(sub["country_clean"].nunique(dropna=False))
        reduced_parts.append(rep)

    reduced = pd.DataFrame(reduced_parts).reset_index(drop=True)
    reduced.to_csv(output_csv, index=False)

    summary = {
        "inputs": {
            "input_csv": str(input_csv.resolve()),
            "feature_list_csv": str(feature_list_csv.resolve()),
            "round_decimals": int(args.round_decimals),
        },
        "reduction": {
            "requested_grouping": args.grouping,
            "chosen_grouping": chosen,
            "output_csv": str(output_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
        },
        "counts": {
            "rows_input": int(len(df)),
            "rows_reduced": int(len(reduced)),
            "country_year_blocks": int(len(country_year_blocks)),
            "country_year_blocks_with_multiple_exposure_vectors": int((country_year_blocks["n_exposure_vectors"] > 1).sum()),
            "country_year_blocks_with_single_exposure_vector": int((country_year_blocks["n_exposure_vectors"] == 1).sum()),
            "unique_exposure_vectors": int(exposure_fp.nunique()),
        },
        "candidate_country_year_blocks": {
            "rows_per_block": country_year_blocks["n_rows"].describe().to_dict(),
            "exposure_vectors_per_block": country_year_blocks["n_exposure_vectors"].describe().to_dict(),
        },
        "top_multivector_country_year_blocks": (
            country_year_blocks.sort_values(["n_exposure_vectors", "n_rows"], ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
        "reduced_preview": reduced.head(5).loc[:, ["country_clean", "exposome_year", "country_year", "effective_grouping", "effective_group_rows"]].to_dict(orient="records"),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] reduced table: {output_csv}")
    print(f"[DONE] summary: {summary_json}")
    print(
        "[SUMMARY] "
        f"rows_input={summary['counts']['rows_input']}, "
        f"rows_reduced={summary['counts']['rows_reduced']}, "
        f"country_year_blocks={summary['counts']['country_year_blocks']}, "
        f"multi_vector_blocks={summary['counts']['country_year_blocks_with_multiple_exposure_vectors']}, "
        f"unique_exposure_vectors={summary['counts']['unique_exposure_vectors']}, "
        f"chosen_grouping={chosen}"
    )


if __name__ == "__main__":
    main()
