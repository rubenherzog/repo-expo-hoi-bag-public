#!/usr/bin/env python3
"""Normative transfer grid: pooled + CN/AD/FTD norms, per BAG.

Each condition column follows the three-panel BAG rows in
``plot_fig2_grid_v3.py``: held-out LOCO R2, set size and the percentage of
synergistic triplets. All three rows show the top candidates as jittered points
with boxplot summaries. The stage reads existing canonical/normative summaries
and the exposome-wide triplet O-information table; no model is refitted.
"""

from __future__ import annotations

import os
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

try:
    for font_path in [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/local/share/fonts/Arial.ttf",
        "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",       # Arial-metric clone
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
except Exception:
    pass
matplotlib.use("Agg")
matplotlib.set_loglevel("error")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
# family="sans-serif" (not "Arial"): when Arial.ttf is absent, family="Arial"
# silently falls back to DejaVu; the list below resolves to Arimo instead.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"]

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT = "variant_a"
_repro_data_root = os.environ.get("REPRO_DATA_ROOT", "").strip()
REPRO_DATA_ROOT = Path(_repro_data_root) if _repro_data_root else REPO_ROOT
_default_output_root = Path(_repro_data_root) / "results" / VARIANT if _repro_data_root else Path("outputs") / VARIANT
V3_OUTPUT_ROOT = Path(os.environ.get("V3_OUTPUT_ROOT", str(_default_output_root)))
BUNDLE_ROOT = V3_OUTPUT_ROOT if V3_OUTPUT_ROOT.name == VARIANT else V3_OUTPUT_ROOT / VARIANT
POOLED_CANONICAL_ROOT = Path(
    os.environ.get(
        "NORM_POOLED_CANONICAL_ROOT",
        str(BUNDLE_ROOT / "families" / "pooled_oinfo_ladder" / "canonical"),
    )
)

PAPER_FIGURES_DIR = Path(os.environ.get("REPRO_FIGURES_ROOT", str(REPO_ROOT / "paper_figures")))
LOCAL_STATS_DIR = Path(
    os.environ.get(
        "NORM_TRANSFER_STATS_DIR",
        str(BUNDLE_ROOT / "stats" / "normative_transfer_grid"),
    )
)
FIG_SUFFIX = os.environ.get("NORM_FIG_SUFFIX", "").strip()
FIG_SUFFIX_PART = f"_{FIG_SUFFIX}" if FIG_SUFFIX else ""
FIG_STEM_TEMPLATE = os.environ.get(
    "NORM_FIG_STEM_TEMPLATE",
    "fig_normative_transfer_grid_{bag}{suffix}",
)

SINGLE_COLOR = "#E07B00"
SYN_COLOR = "#1B6B2E"
RED_COLOR = "#4B0082"

RUNG_ORDER = [
    rung.strip()
    for rung in os.environ.get(
        "NORM_TRANSFER_RUNGS",
        "ols,xgb_tree_d1,xgb_tree_d2,xgb_tree_d3",
    ).split(",")
    if rung.strip()
]
RUNG_COLORS = {
    "ols": "#666666",
    "xgb_tree_d1": "#E6AB02",
    "xgb_tree_d2": "#66A61E",
    "xgb_tree_d3": "#CC00CC",
}
RUNG_DISPLAY = {
    "ols": "OLS",
    "xgb_tree_d1": "d1",
    "xgb_tree_d2": "d2",
    "xgb_tree_d3": "d3",
}
RUNG_LABELS = [RUNG_DISPLAY.get(rung, rung) for rung in RUNG_ORDER]

BAGS = ["functional", "structural"]
CONDITION_ORDER = [
    "Pooled",
    "CN->CN",
    "AD->AD",
    "AD->FTD",
    "FTD->FTD",
    "FTD->AD",
    "AD+FTD->AD",
    "AD+FTD->FTD",
]
FAMILY_TRAIN = {
    "cn_norm": "CN",
    "ad_norm": "AD",
    "ftd_norm": "FTD",
    "adftd_norm": "AD+FTD",
}
TOP_N = int(os.environ.get("NORM_TRANSFER_TOP_N", "20"))
ORDER_MAX = int(os.environ.get("NORM_TRANSFER_ORDER_MAX", "30"))
SUBCOMB_BASELINE_PARQUET = Path(
    os.environ.get(
        "SUBCOMB_BASELINE_PARQUET",
        str(REPRO_DATA_ROOT / "work" / "subcomb_oinfo" / "subcomb_baseline.parquet"),
    )
)

# Keep the figure style tied to Fig. 2, but make tick labels viable in a
# 13-column supplement grid.
FS = 22
FS_TK = 20
GRID_FS = 15
GRID_FS_TK = 13

SINGLE_EXP_DIRS = {
    "ols": BUNDLE_ROOT / "single_exposure_eval_ols",
    "xgb_tree_d1": BUNDLE_ROOT / "single_exposure_eval_xgb_tree_d1",
    "xgb_tree_d2": BUNDLE_ROOT / "single_exposure_eval_xgb_tree_d2",
    "xgb_tree_d3": BUNDLE_ROOT / "single_exposure_eval_xgb_tree_d3",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def _condition_from_family(row: pd.Series) -> str:
    train = FAMILY_TRAIN.get(str(row["family_id"]), str(row["train_dx"]))
    return f"{train}->{row['test_dx']}"


def _cohen_f2(r2: pd.Series, baseline_r2: pd.Series) -> pd.Series:
    denom = 1.0 - pd.to_numeric(r2, errors="coerce")
    out = (pd.to_numeric(r2, errors="coerce") - pd.to_numeric(baseline_r2, errors="coerce")) / denom
    return out.replace([np.inf, -np.inf], np.nan)


def _country_balanced_r2(path: Path) -> pd.Series:
    """Return the unweighted country mean used by the main Fig. 2."""
    frame = pd.read_csv(_require(path))
    frame["r2"] = pd.to_numeric(frame["r2"], errors="coerce")
    if "n_test" in frame.columns:
        frame = frame[pd.to_numeric(frame["n_test"], errors="coerce").gt(0)]
    frame = frame[np.isfinite(frame["r2"])].copy()
    return frame.groupby("candidate_id", observed=True)["r2"].mean()


def _apply_main_pooled_scores(df: pd.DataFrame, bag: str) -> pd.DataFrame:
    """Replace adapter scores with the exact country-balanced Fig. 2 scores."""
    configured_root = os.environ.get("NORM_POOLED_MAIN_RUN_ROOT", "").strip()
    if not configured_root:
        return df
    root = Path(configured_root)
    parts = []
    for rung in RUNG_ORDER:
        family = "ols" if rung == "ols" else "xgb"
        leaf = "ols" if rung == "ols" else "k10"
        rung_root = root / family / bag / rung
        scores = _country_balanced_r2(rung_root / leaf / "metrics_country.csv")
        baseline = float(
            _country_balanced_r2(rung_root / "baseline" / "metrics_country.csv").iloc[0]
        )
        part = df[df["rung_id"].astype(str).eq(rung)].copy()
        part["r2"] = part["candidate_id"].map(scores)
        if part["r2"].isna().any():
            missing = sorted(part.loc[part["r2"].isna(), "candidate_id"].astype(str).unique())
            raise ValueError(f"Main pooled scores are missing {bag}/{rung}: {missing[:5]}")
        part["baseline_r2"] = baseline
        part["delta_r2"] = part["r2"] - baseline
        part["source"] = "main_fig2_country_balanced"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _load_pooled(bag: str) -> pd.DataFrame:
    path = (
        POOLED_CANONICAL_ROOT
        / "per_experiment"
        / f"pooled_oinfo_ladder_{bag}"
        / "metrics_global_long.parquet"
    )
    df = pd.read_parquet(_require(path))
    pooled_ols_root = os.environ.get("NORM_POOLED_OLS_CANONICAL_ROOT", "").strip()
    if pooled_ols_root:
        ols_path = (
            Path(pooled_ols_root)
            / "per_experiment"
            / f"pooled_oinfo_ladder_{bag}"
            / "metrics_global_long.parquet"
        )
        historical_ols = pd.read_parquet(_require(ols_path))
        historical_ols = historical_ols[historical_ols["rung_id"].astype(str).eq("ols")]
        df = pd.concat([df, historical_ols], ignore_index=True)
    df = df[
        df["rung_id"].isin(RUNG_ORDER)
        & df["objective"].isin(["o_min", "o_max"])
        & (pd.to_numeric(df["order"], errors="coerce") <= ORDER_MAX)
    ].copy()
    full_r2 = pd.to_numeric(df["full_r2"], errors="coerce")
    base_r2 = pd.to_numeric(df["base_r2"], errors="coerce")
    delta_r2 = (
        pd.to_numeric(df["delta_r2_vs_base"], errors="coerce")
        if "delta_r2_vs_base" in df.columns
        else full_r2 - base_r2
    )
    out = pd.DataFrame(
        {
            "bag": bag,
            "condition": "Pooled",
            "family_id": "pooled",
            "train_dx": "all",
            "test_dx": "all",
            "rung_id": df["rung_id"],
            "objective": df["objective"],
            "order": pd.to_numeric(df["order"], errors="coerce"),
            "candidate_id": df["candidate_id"],
            "predictors_identity": df.get("predictors_identity", ""),
            "r2": full_r2,
            "baseline_r2": base_r2,
            "delta_r2": delta_r2,
            # Evaluated O-information, carried through for the optional negative-
            # O-info synergy criterion (SYN_OINFO_NEGATIVE) in downstream stages.
            "oinfo": pd.to_numeric(df.get("thoi_o", np.nan), errors="coerce"),
            "source": "pooled_canonical",
        }
    )
    out = _apply_main_pooled_scores(out, bag)
    out["cohen_f2"] = _cohen_f2(out["r2"], out["baseline_r2"])
    return out


def _load_normative_model(bag: str, model_family: str) -> pd.DataFrame:
    configured_root = os.environ.get(f"NORM_{model_family.upper()}_ROOT", "").strip()
    normative_root = (
        Path(configured_root)
        if configured_root
        else REPRO_DATA_ROOT / f"{model_family}_normative_loco"
    )
    path = normative_root / bag / f"{model_family}_norm_global_all.csv"
    country_path = normative_root / bag / f"{model_family}_norm_country_all.csv"
    # When NORM_ALLOW_MISSING is set, treat an absent normative aggregate as "no
    # data yet" and return an empty frame (the consumer concatenates only the
    # non-empty parts), instead of raising. Used for partial/interim order-cap
    # reconciliations where only the Pooled arm is available so far. The
    # canonical figure render leaves this unset, so missing inputs stay fatal.
    if os.environ.get("NORM_ALLOW_MISSING", "").strip() in ("1", "true", "yes") and not path.exists():
        warnings.warn(f"Normative model not found for {bag}/{model_family}: {path} (left blank)")
        return pd.DataFrame()
    df = pd.read_csv(_require(path))
    country = pd.read_csv(_require(country_path))
    country["r2"] = pd.to_numeric(country["r2"], errors="coerce")
    if "n_test" in country.columns:
        country = country[pd.to_numeric(country["n_test"], errors="coerce").gt(0)]
    country = country[np.isfinite(country["r2"])].copy()
    country_balanced = (
        country.groupby(
            ["candidate_id", "rung_id", "family_id", "train_dx", "test_dx"],
            observed=True,
        )["r2"]
        .mean()
        .rename("country_balanced_r2")
        .reset_index()
    )
    df = df.merge(
        country_balanced,
        on=["candidate_id", "rung_id", "family_id", "train_dx", "test_dx"],
        how="left",
        validate="one_to_one",
    )
    if df["country_balanced_r2"].isna().any():
        raise ValueError(f"Missing country-balanced normative scores in {country_path}")
    df["condition"] = df.apply(_condition_from_family, axis=1)
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()

    baseline = (
        df[df["objective"] == "baseline"][
            ["family_id", "train_dx", "test_dx", "condition", "rung_id", "country_balanced_r2"]
        ]
        .drop_duplicates()
        .rename(columns={"country_balanced_r2": "baseline_r2"})
    )
    models = df[
        df["rung_id"].isin(RUNG_ORDER)
        & df["objective"].isin(["o_min", "o_max"])
        & (pd.to_numeric(df["order"], errors="coerce") <= ORDER_MAX)
    ].copy()
    models = models.merge(
        baseline,
        on=["family_id", "train_dx", "test_dx", "condition", "rung_id"],
        how="left",
        validate="many_to_one",
    )
    out = pd.DataFrame(
        {
            "bag": bag,
            "condition": models["condition"],
            "family_id": models["family_id"],
            "train_dx": models["train_dx"],
            "test_dx": models["test_dx"],
            "rung_id": models["rung_id"],
            "objective": models["objective"],
            "order": pd.to_numeric(models["order"], errors="coerce"),
            "candidate_id": models["candidate_id"],
            "predictors_identity": models.get("predictors_identity", ""),
            "r2": pd.to_numeric(models["country_balanced_r2"], errors="coerce"),
            "baseline_r2": pd.to_numeric(models["baseline_r2"], errors="coerce"),
            # Evaluated O-information (normative CSVs carry it as `score`).
            "oinfo": pd.to_numeric(models.get("score", np.nan), errors="coerce"),
            "source": f"{model_family}_normative_country_balanced",
        }
    )
    out["delta_r2"] = out["r2"] - out["baseline_r2"]
    out["cohen_f2"] = _cohen_f2(out["r2"], out["baseline_r2"])
    return out


def load_plot_data(bag: str) -> pd.DataFrame:
    pooled = _load_pooled(bag)
    ols = _load_normative_model(bag, "ols")
    xgb = _load_normative_model(bag, "xgb")
    df = pd.concat([pooled, ols, xgb], ignore_index=True)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    df["rung_id"] = pd.Categorical(df["rung_id"], categories=RUNG_ORDER, ordered=True)
    return df.sort_values(["condition", "rung_id", "objective", "r2"], ascending=[True, True, True, False])


def _load_single_pooled(bag: str) -> pd.DataFrame:
    rows = []
    for rung, root in SINGLE_EXP_DIRS.items():
        configured_root = os.environ.get("NORM_POOLED_MAIN_RUN_ROOT", "").strip()
        if configured_root:
            family = "ols" if rung == "ols" else "xgb"
            path = Path(configured_root) / family / bag / rung / "single" / "metrics_country.csv"
        else:
            legacy_root = os.environ.get("NORM_POOLED_SINGLE_RUN_ROOT", "").strip()
            path = (
                Path(legacy_root) / "xgb" / bag / rung / "single" / "metrics_global.csv"
                if legacy_root
                else root / bag / "single_exposure_global.csv"
            )
        if rung not in RUNG_ORDER:
            continue
        if not path.exists():
            print(f"WARNING: missing pooled single exposure file for {bag}/{rung}: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if configured_root:
            scores = df.assign(r2=pd.to_numeric(df["r2"], errors="coerce")).groupby(
                "candidate_id", observed=True
            )["r2"].mean()
            best_id = str(scores.idxmax())
            best_r2 = float(scores.loc[best_id])
            best = df[df["candidate_id"].astype(str).eq(best_id)].iloc[0]
        else:
            if "global_oof_r2" not in df.columns:
                continue
            best = df.loc[pd.to_numeric(df["global_oof_r2"], errors="coerce").idxmax()]
            best_r2 = float(best["global_oof_r2"])
        feature = best.get("feature_name", "")
        if not feature:
            feature = str(best.get("candidate_id", "")).removeprefix("__single__")
        rows.append(
            {
                "bag": bag,
                "condition": "Pooled",
                "rung_id": rung,
                "single_r2": best_r2,
                "single_feature": feature,
                "source": "pooled_single",
            }
        )
    return pd.DataFrame(rows)


def _load_single_normative(bag: str) -> pd.DataFrame:
    configured_root = os.environ.get("NORM_SINGLE_NORMATIVE_ROOT", "").strip()
    if configured_root:
        root = Path(configured_root)
        per_bag = root / bag / "single_exposure_normative_global_all.csv"
        nested = root / bag / "single_exposure" / "single_exposure_normative_global_all.csv"
        path = per_bag if per_bag.exists() else nested if nested.exists() else root / "single_exposure_normative_global_all.csv"
        per_bag_country = root / bag / "single_exposure_normative_country_all.csv"
        nested_country = root / bag / "single_exposure" / "single_exposure_normative_country_all.csv"
        country_path = (
            per_bag_country
            if per_bag_country.exists()
            else nested_country
            if nested_country.exists()
            else root / "single_exposure_normative_country_all.csv"
        )
    else:
        path = REPRO_DATA_ROOT / "single_exposure_normative" / "single_exposure_normative_global_all.csv"
        country_path = REPRO_DATA_ROOT / "single_exposure_normative" / "single_exposure_normative_country_all.csv"
    df = pd.read_csv(_require(path))
    country = pd.read_csv(_require(country_path))
    if configured_root and "ols" in RUNG_ORDER and not df["rung_id"].astype(str).eq("ols").any():
        legacy_root = REPRO_DATA_ROOT / "single_exposure_normative"
        legacy_global = pd.read_csv(_require(legacy_root / "single_exposure_normative_global_all.csv"))
        legacy_country = pd.read_csv(_require(legacy_root / "single_exposure_normative_country_all.csv"))
        df = pd.concat([df, legacy_global[legacy_global["rung_id"].astype(str).eq("ols")]], ignore_index=True)
        country = pd.concat(
            [country, legacy_country[legacy_country["rung_id"].astype(str).eq("ols")]],
            ignore_index=True,
        )
    country["r2"] = pd.to_numeric(country["r2"], errors="coerce")
    if "n_test" in country.columns:
        country = country[pd.to_numeric(country["n_test"], errors="coerce").gt(0)]
    country = country[np.isfinite(country["r2"])].copy()
    balanced = (
        country.groupby(
            ["bag", "candidate_id", "rung_id", "family_id", "train_dx", "test_dx"],
            observed=True,
        )["r2"]
        .mean()
        .rename("country_balanced_r2")
        .reset_index()
    )
    df = df.merge(
        balanced,
        on=["bag", "candidate_id", "rung_id", "family_id", "train_dx", "test_dx"],
        how="left",
        validate="one_to_one",
    )
    if df["country_balanced_r2"].isna().any():
        raise ValueError(f"Missing country-balanced single-exposure scores in {country_path}")
    reference_path = os.environ.get("NORM_OLS_SINGLE_REFERENCE_CSV", "").strip()
    if reference_path and "ols" in RUNG_ORDER:
        reference = pd.read_csv(_require(Path(reference_path)))
        reference = reference[
            reference["bag"].astype(str).eq(bag)
            & ~reference["condition"].astype(str).eq("Pooled")
            & reference["rung_id"].astype(str).eq("ols")
        ].copy()
    else:
        reference = pd.DataFrame()
    df = df[df["bag"].eq(bag)].copy()
    df["condition"] = df.apply(_condition_from_family, axis=1)
    df = df[df["condition"].isin(CONDITION_ORDER) & df["rung_id"].isin(RUNG_ORDER)].copy()
    idx = (
        df.assign(_r2=pd.to_numeric(df["country_balanced_r2"], errors="coerce"))
        .groupby(["condition", "rung_id"], observed=True)["_r2"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best = df.loc[idx].copy()
    result = pd.DataFrame(
        {
            "bag": bag,
            "condition": best["condition"],
            "rung_id": best["rung_id"],
            "single_r2": pd.to_numeric(best["country_balanced_r2"], errors="coerce"),
            "single_feature": best.get("feature_name", ""),
            "source": "normative_single_country_balanced",
        }
    )
    if not reference.empty:
        result = result[~result["rung_id"].astype(str).eq("ols")]
        result = pd.concat(
            [
                result,
                reference.rename(columns={"single_feature": "single_feature"})[
                    ["bag", "condition", "rung_id", "single_r2", "single_feature", "source"]
                ],
            ],
            ignore_index=True,
        )
    return result


def load_single_data(bag: str, df: pd.DataFrame) -> pd.DataFrame:
    single = pd.concat([_load_single_pooled(bag), _load_single_normative(bag)], ignore_index=True)
    baselines = (
        df[["condition", "rung_id", "baseline_r2"]]
        .drop_duplicates()
        .dropna(subset=["baseline_r2"])
    )
    single = single.merge(baselines, on=["condition", "rung_id"], how="left", validate="one_to_one")
    single["single_f2"] = _cohen_f2(single["single_r2"], single["baseline_r2"])
    single["condition"] = pd.Categorical(single["condition"], categories=CONDITION_ORDER, ordered=True)
    single["rung_id"] = pd.Categorical(single["rung_id"], categories=RUNG_ORDER, ordered=True)
    return single.sort_values(["condition", "rung_id"])


def select_top20(df: pd.DataFrame) -> pd.DataFrame:
    top = (
        df.groupby(["condition", "rung_id", "objective"], observed=True, group_keys=False)
        .apply(lambda g: g.nlargest(TOP_N, "r2"))
        .reset_index(drop=True)
    )
    return top


def load_triplet_omega() -> tuple[dict[object, float], float]:
    """Load all exposome triplets and derive their negative-Omega baseline."""
    available = pd.read_parquet(_require(SUBCOMB_BASELINE_PARQUET)).columns
    if {"candidate_id", "frac_neg_k3"}.issubset(available):
        measured = pd.read_parquet(
            SUBCOMB_BASELINE_PARQUET,
            columns=["candidate_id", "frac_neg_k3"],
        ).drop_duplicates("candidate_id")
        lookup = dict(
            zip(
                measured["candidate_id"].astype(str),
                pd.to_numeric(measured["frac_neg_k3"], errors="raise"),
            )
        )
        reference_root = os.environ.get("NORM_TRIPLET_REFERENCE_ROOT", "").strip()
        if reference_root:
            references = sorted(Path(reference_root).glob("*triplets.csv"))
            if not references:
                raise FileNotFoundError(f"No triplet reference tables under {reference_root}")
            reference = pd.concat([pd.read_csv(path) for path in references], ignore_index=True)
            reference = reference.drop_duplicates("candidate_id")
            lookup.update(
                zip(
                    reference["candidate_id"].astype(str),
                    pd.to_numeric(reference["pct_synergistic_triplets"], errors="raise") / 100.0,
                )
            )
            metadata_parts = []
            for family in ("ols", "xgb"):
                configured = os.environ.get(f"NORM_{family.upper()}_ROOT", "").strip()
                if not configured:
                    continue
                for bag in ("structural", "functional"):
                    metadata_path = Path(configured) / bag / f"{family}_norm_global_all.csv"
                    if metadata_path.exists():
                        metadata_parts.append(
                            pd.read_csv(
                                metadata_path,
                                usecols=["candidate_id", "predictors_identity"],
                            )
                        )
            canonical_root = os.environ.get("NORM_POOLED_CANONICAL_ROOT", "").strip()
            if canonical_root:
                for bag in ("structural", "functional"):
                    canonical_path = (
                        Path(canonical_root) / "per_experiment"
                        / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
                    )
                    if canonical_path.exists():
                        metadata_parts.append(
                            pd.read_parquet(
                                canonical_path,
                                columns=["candidate_id", "predictors_identity"],
                            )
                        )
            if metadata_parts:
                metadata = pd.concat(metadata_parts, ignore_index=True).drop_duplicates(
                    "candidate_id"
                )
                reference_identity = reference.merge(
                    metadata, on="candidate_id", how="left", validate="one_to_one"
                ).dropna(subset=["predictors_identity"])
                lookup.update(
                    zip(
                        "identity:" + reference_identity["predictors_identity"].astype(str),
                        pd.to_numeric(
                            reference_identity["pct_synergistic_triplets"], errors="raise"
                        )
                        / 100.0,
                    )
                )
        return lookup, float(
            os.environ.get("EXPOSOME_TRIPLET_NEGATIVE_FRACTION", "0.332301")
        )
    baseline = pd.read_parquet(
        SUBCOMB_BASELINE_PARQUET,
        filters=[("order_k", "==", 3)],
        columns=["nplet_cols", "o_info"],
    )
    omega = {
        frozenset(str(names).split("|")): float(value)
        for names, value in zip(baseline["nplet_cols"], baseline["o_info"])
    }
    baseline_fraction = float((pd.to_numeric(baseline["o_info"], errors="raise") < 0).mean())
    return omega, baseline_fraction


def add_synergistic_triplet_fraction(
    top: pd.DataFrame,
    omega: dict[object, float] | None = None,
    baseline_fraction: float | None = None,
) -> pd.DataFrame:
    """Add the Fig. 2 triplet measure to every displayed normative candidate.

    The baseline parquet contains O-information for every exposome triplet.
    Because triplet O-information is a property of the exposure combination,
    the same lookup applies to a candidate evaluated in any diagnostic transfer
    condition. A negative value denotes a synergistic triplet.
    """
    if omega is None or baseline_fraction is None:
        omega, baseline_fraction = load_triplet_omega()

    cache: dict[str, tuple[int, int, float]] = {}

    def summarise(identity: object) -> tuple[int, int, float]:
        key = str(identity)
        if key in cache:
            return cache[key]
        exposures = [name.strip() for name in key.split("|") if name.strip()]
        triplets = list(combinations(exposures, 3))
        values = [omega.get(frozenset(triplet), np.nan) for triplet in triplets]
        n_total = len(values)
        n_measured = int(np.isfinite(values).sum())
        if n_total == 0 or n_measured != n_total:
            missing = n_total - n_measured
            raise ValueError(
                f"Triplet O-information coverage failed for candidate {key!r}: "
                f"{missing} of {n_total} triplets are absent from {SUBCOMB_BASELINE_PARQUET}"
            )
        n_synergistic = int((np.asarray(values, dtype=float) < 0).sum())
        result = (n_total, n_synergistic, 100.0 * n_synergistic / n_total)
        cache[key] = result
        return result

    def candidate_or_identity(row: pd.Series) -> tuple[int, int, float]:
        candidate_id = str(row.get("candidate_id", ""))
        identity_key = "identity:" + str(row["predictors_identity"])
        lookup_key = candidate_id if candidate_id in omega else identity_key
        if lookup_key in omega:
            exposures = [
                name.strip()
                for name in str(row["predictors_identity"]).split("|")
                if name.strip()
            ]
            n_total = len(list(combinations(exposures, 3)))
            fraction = float(omega[lookup_key])
            n_synergistic = int(round(fraction * n_total))
            return n_total, n_synergistic, 100.0 * fraction
        return summarise(row["predictors_identity"])

    summaries = top.apply(candidate_or_identity, axis=1)
    out = top.copy()
    out["n_triplets"] = summaries.map(lambda value: value[0])
    out["n_synergistic_triplets"] = summaries.map(lambda value: value[1])
    out["pct_synergistic_triplets"] = summaries.map(lambda value: value[2])
    out["exposome_baseline_pct"] = 100.0 * baseline_fraction
    return out


def _split_box_values(top: pd.DataFrame, condition: str, rung: str, objective: str, y_col: str) -> np.ndarray:
    vals = top[
        top["condition"].astype(str).eq(condition)
        & top["rung_id"].astype(str).eq(rung)
        & top["objective"].eq(objective)
    ][y_col]
    return pd.to_numeric(vals, errors="coerce").dropna().to_numpy()


def draw_split_boxes(
    ax,
    top: pd.DataFrame,
    single: pd.DataFrame,
    condition: str,
    y_col: str,
    ylim: tuple[float, float],
    show_baseline: bool,
    show_single: bool,
) -> None:
    """Draw the same jittered split boxplots used for BAG rows in Fig. 2."""
    half = 0.32
    box_half_width = half * 0.5
    rng = np.random.default_rng(0)
    for i, rung in enumerate(RUNG_ORDER):
        for objective, color, x_center in [
            ("o_min", SYN_COLOR, i - half * 0.5),
            ("o_max", RED_COLOR, i + half * 0.5),
        ]:
            vals = _split_box_values(top, condition, rung, objective, y_col)
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-box_half_width * 0.9, box_half_width * 0.9, size=len(vals))
            ax.scatter(
                x_center + jitter,
                vals,
                c=color,
                s=30,
                alpha=0.55,
                edgecolors="none",
                zorder=3,
            )
            q25, median, q75 = np.percentile(vals, [25, 50, 75])
            iqr = q75 - q25
            whisker_low = max(vals.min(), q25 - 1.5 * iqr)
            whisker_high = min(vals.max(), q75 + 1.5 * iqr)
            ax.add_patch(
                mpatches.Rectangle(
                    (x_center - box_half_width, q25),
                    2 * box_half_width,
                    iqr,
                    facecolor=color,
                    alpha=0.40,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=4,
                )
            )
            ax.plot(
                [x_center - box_half_width, x_center + box_half_width],
                [median, median],
                color=color,
                lw=1.6,
                zorder=5,
            )
            ax.plot([x_center, x_center], [whisker_low, q25], color=color, lw=0.8, zorder=4)
            ax.plot([x_center, x_center], [q75, whisker_high], color=color, lw=0.8, zorder=4)
        if show_baseline:
            base = top[
                top["condition"].astype(str).eq(condition)
                & top["rung_id"].astype(str).eq(rung)
            ]["baseline_r2"].dropna()
            if len(base):
                val = float(base.iloc[0])
                ax.plot([i - half, i + half], [val, val], color="black", lw=2.0, ls="-", zorder=4)
        if show_single:
            col = "single_r2"
            s = single[
                single["condition"].astype(str).eq(condition)
                & single["rung_id"].astype(str).eq(rung)
            ][col].dropna()
            if len(s):
                val = float(s.iloc[0])
                ax.plot([i - half, i + half], [val, val], color=SINGLE_COLOR, lw=2.0, ls="-", zorder=5)

    ax.set_xlim(-0.5, len(RUNG_ORDER) - 0.5)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(RUNG_ORDER)))
    ax.set_xticklabels(RUNG_LABELS, fontsize=GRID_FS_TK)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=GRID_FS_TK)
    ax.spines[["top", "right"]].set_visible(False)


def _shared_limits(vals: pd.Series | list[float], pad_frac: float = 0.06) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(arr) == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if np.isclose(lo, hi):
        pad = max(abs(hi) * pad_frac, 0.01)
    else:
        pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _audit_counts(df: pd.DataFrame, top: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        for rung in RUNG_ORDER:
            sub = df[df["condition"].astype(str).eq(condition) & df["rung_id"].astype(str).eq(rung)]
            top_sub = top[top["condition"].astype(str).eq(condition) & top["rung_id"].astype(str).eq(rung)]
            single_sub = single[single["condition"].astype(str).eq(condition) & single["rung_id"].astype(str).eq(rung)]
            rows.append(
                {
                    "condition": condition,
                    "rung_id": rung,
                    "n_models": int(len(sub)),
                    "n_top_syn": int((top_sub["objective"] == "o_min").sum()),
                    "n_top_red": int((top_sub["objective"] == "o_max").sum()),
                    "has_baseline": bool(sub["baseline_r2"].notna().any()),
                    "has_single": bool(single_sub["single_r2"].notna().any()),
                }
            )
    return pd.DataFrame(rows)


def save_plot_tables(bag: str, df: pd.DataFrame, top: pd.DataFrame, single: pd.DataFrame) -> None:
    stats_dir = LOCAL_STATS_DIR / FIG_SUFFIX if FIG_SUFFIX else LOCAL_STATS_DIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    compact_cols = [
        "bag",
        "condition",
        "family_id",
        "train_dx",
        "test_dx",
        "rung_id",
        "objective",
        "order",
        "candidate_id",
        "r2",
        "baseline_r2",
        "delta_r2",
        "cohen_f2",
        "source",
    ]
    df[compact_cols].to_csv(stats_dir / f"{bag}_plot_data.csv", index=False)
    top.to_csv(stats_dir / f"{bag}_top{TOP_N}_plot_data.csv", index=False)
    single.to_csv(stats_dir / f"{bag}_single_lines.csv", index=False)
    _audit_counts(df, top, single).to_csv(stats_dir / f"{bag}_audit_counts.csv", index=False)


_OBJ_GLOSS = ("o_min = synergistic (min O-information) arm; "
              "o_max = redundant (max O-information) arm")


def _build_transfer_panels(bag: str, top: pd.DataFrame, single: pd.DataFrame) -> list[Panel]:
    """Three panels matching the three quantities in each BAG row of Fig. 2."""
    cols = [c for c in ["condition", "rung_id", "objective", "candidate_id", "order",
                        "r2", "baseline_r2", "n_triplets", "n_synergistic_triplets",
                        "pct_synergistic_triplets", "exposome_baseline_pct"] if c in top.columns]
    drawn = top[cols].copy()
    drawn["rung_id"] = drawn["rung_id"].astype(str)
    drawn = drawn.rename(columns={"rung_id": "model_level"})
    drawn["model_level"] = drawn["model_level"].map(RUNG_DISPLAY).fillna(drawn["model_level"])

    single_cols = [c for c in ["condition", "rung_id", "single_r2", "single_feature"]
                   if c in single.columns]
    single_tidy = single[single_cols].copy()
    if "rung_id" in single_tidy.columns:
        single_tidy["rung_id"] = single_tidy["rung_id"].astype(str)
        single_tidy = single_tidy.rename(columns={"rung_id": "model_level"})
        single_tidy["model_level"] = single_tidy["model_level"].map(RUNG_DISPLAY).fillna(single_tidy["model_level"])

    shared_cols = {
        "condition": "transfer condition: the normative model's training group -> the evaluated group",
        "model_level": "model level (OLS, d1, d2, d3)",
        "objective": _OBJ_GLOSS,
        "candidate_id": "identifier of the exposure set",
        "order": "set size (number of exposures)",
    }
    top_note = (f"One row per plotted candidate: the top {TOP_N} by held-out LOCO R² within each "
                "condition, model level and arm. Candidates are capped at set size "
                f"{ORDER_MAX}.")

    panels = [
        Panel(
            panel_id=f"a_{bag}_loco_r2",
            frame=drawn[[c for c in ["condition", "model_level", "objective", "candidate_id", "r2", "baseline_r2"] if c in drawn.columns]].reset_index(drop=True),
            description=f"{bag.capitalize()} BAG: held-out LOCO R² of the transferred models, by condition and model level.",
            columns={**shared_cols, "r2": "held-out LOCO R² of that candidate",
                     "baseline_r2": "covariate-only baseline R² for that cell -- the black line"},
            notes=top_note,
        ),
        Panel(
            panel_id=f"b_{bag}_set_size",
            frame=drawn[[c for c in ["condition", "model_level", "objective", "candidate_id", "order"] if c in drawn.columns]].reset_index(drop=True),
            description=f"{bag.capitalize()} BAG: set size of the same candidates, by condition and model level.",
            columns=shared_cols,
            notes=top_note,
        ),
        Panel(
            panel_id=f"c_{bag}_triplets",
            frame=drawn[[c for c in ["condition", "model_level", "objective", "candidate_id",
                                      "n_triplets", "n_synergistic_triplets",
                                      "pct_synergistic_triplets", "exposome_baseline_pct"] if c in drawn.columns]].reset_index(drop=True),
            description=(f"{bag.capitalize()} BAG: synergistic-triplet content of the same candidates, "
                         "by condition and model level."),
            columns={**shared_cols,
                     "n_triplets": "number of triplet sub-combinations in the candidate",
                     "n_synergistic_triplets": "number of those triplets with negative O-information",
                     "pct_synergistic_triplets": "percentage of the candidate's triplets with negative O-information",
                     "exposome_baseline_pct": "percentage of all exposome triplets with negative O-information -- the dashed line"},
            notes=(top_note + f" The dashed line marks the exposome-wide rate over all triplets, "
                   f"{float(drawn['exposome_baseline_pct'].iloc[0]):.1f}%."),
        ),
    ]
    if not single_tidy.empty:
        panels.append(Panel(
            panel_id=f"best_single_{bag}",
            frame=single_tidy.reset_index(drop=True),
            description=(f"{bag.capitalize()} BAG: the best single-exposure reference drawn as the "
                         "orange line in the R² row."),
            columns={"condition": shared_cols["condition"], "model_level": shared_cols["model_level"],
                     "single_r2": "held-out LOCO R² of the best single exposure",
                     "single_feature": "the exposure itself"},
            notes="One row per condition and model level; each value is a single reference line, not a distribution.",
        ))
    return panels


def plot_bag(bag: str) -> None:
    print(f"Loading {bag} from REPRO_DATA_ROOT={REPRO_DATA_ROOT}")
    df = load_plot_data(bag)
    single = load_single_data(bag, df)
    omega, baseline_fraction = load_triplet_omega()
    top = add_synergistic_triplet_fraction(select_top20(df), omega, baseline_fraction)
    save_plot_tables(bag, df, top, single)

    r2_vals = list(top["r2"]) + list(top["baseline_r2"]) + list(single["single_r2"])
    r2_ylim = _shared_limits(r2_vals, pad_frac=0.05)

    fig, axes = plt.subplots(3, len(CONDITION_ORDER), figsize=(18, 12), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.02, wspace=0.005, hspace=0.02)

    for col, condition in enumerate(CONDITION_ORDER):
        axes[0, col].set_title(condition, fontsize=GRID_FS, pad=8)
        draw_split_boxes(axes[0, col], top, single, condition, "r2", r2_ylim, True, True)
        draw_split_boxes(axes[1, col], top, single, condition, "order", (0, ORDER_MAX + 1), False, False)
        axes[1, col].axhline(21, color="black", lw=1.0, ls="--", zorder=2)
        axes[1, col].set_yticks(np.arange(0, ORDER_MAX + 1, 10))
        draw_split_boxes(
            axes[2, col], top, single, condition, "pct_synergistic_triplets", (-2, 102), False, False
        )
        axes[2, col].axhline(
            float(top["exposome_baseline_pct"].iloc[0]), color="black", lw=1.0, ls="--", zorder=2
        )
        axes[2, col].set_yticks(range(0, 101, 20))

        if col == 0:
            axes[0, col].set_ylabel("R² LOCO", fontsize=GRID_FS)
            axes[1, col].set_ylabel("Set size", fontsize=GRID_FS)
            axes[2, col].set_ylabel("% synergistic triplets", fontsize=GRID_FS)
        else:
            for row in range(3):
                axes[row, col].set_yticklabels([])

        if col == 0:
            handles = [
                mpatches.Patch(color=SYN_COLOR, alpha=0.55, label="Min O-info"),
                mpatches.Patch(color=RED_COLOR, alpha=0.55, label="Max O-info"),
                mlines.Line2D([0], [0], color="black", lw=2.0, ls="-", label="Baseline"),
                mlines.Line2D([0], [0], color=SINGLE_COLOR, lw=2.0, ls="-", label="Best single"),
            ]
            axes[0, col].legend(handles=handles, fontsize=GRID_FS_TK, loc="lower right", framealpha=0.7)

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_stem = FIG_STEM_TEMPLATE.format(bag=bag, suffix=FIG_SUFFIX_PART)
    for ext in ("pdf", "svg", "png"):
        out = PAPER_FIGURES_DIR / f"{figure_stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    for path in write_source_data(
        figure_stem,
        _build_transfer_panels(bag, top, single),
        PAPER_FIGURES_DIR,
    ):
        print(f"Saved: {path}")

    stats_dir = LOCAL_STATS_DIR / FIG_SUFFIX if FIG_SUFFIX else LOCAL_STATS_DIR
    audit = pd.read_csv(stats_dir / f"{bag}_audit_counts.csv")
    missing = audit[(~audit["has_baseline"]) | (~audit["has_single"]) | (audit["n_top_syn"] == 0) | (audit["n_top_red"] == 0)]
    if not missing.empty:
        print("WARNING: incomplete plotted cells:")
        print(missing.to_string(index=False))
    print(f"Shared {bag} R2 ylim: {r2_ylim}")


def main() -> None:
    print(f"Using V3_OUTPUT_ROOT={V3_OUTPUT_ROOT}")
    print(f"Using REPRO_DATA_ROOT={REPRO_DATA_ROOT}")
    bags = [b.strip() for b in os.environ.get("NORM_TRANSFER_BAGS", ",".join(BAGS)).split(",") if b.strip()]
    for bag in bags:
        if bag not in BAGS:
            raise ValueError(f"Unknown bag {bag!r}; expected one of {BAGS}")
        plot_bag(bag)


if __name__ == "__main__":
    main()
