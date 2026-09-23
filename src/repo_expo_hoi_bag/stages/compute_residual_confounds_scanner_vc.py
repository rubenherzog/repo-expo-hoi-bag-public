"""Scanner variance-component addendum for residual-confounds.

Writes only explicitly suffixed artifacts and leaves the country-only analysis
and its delivered figure/table untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.compute_residual_confounds import (
    BAGS,
    BAG_SHORT,
    OBJECTIVES,
    _build_design,
    _plot_residual_confounds,
    fit_country_scanner_variance_components,
)
from scripts.sensitivity_common import load_raw_and_domains


def _required_dir(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is required")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scanner_table() -> pd.DataFrame:
    value = os.environ.get("SCANNER_XLSX_PATH", "").strip()
    if not value:
        raise EnvironmentError("SCANNER_XLSX_PATH is required for scanner variance components")
    frame = pd.read_excel(value, sheet_name="Sheet 1", usecols=["N_MEGA", "resonador"])
    frame["N_MEGA"] = frame["N_MEGA"].astype(str).str.strip()
    frame = frame.dropna(subset=["N_MEGA", "resonador"]).drop_duplicates("N_MEGA", keep="first")
    return frame.rename(columns={"resonador": "scanner_id"})


def main() -> None:
    source_root = _required_dir("RESIDUAL_CONFOUNDS_SOURCE_ROOT")
    figure_root = _required_dir("RESIDUAL_CONFOUNDS_SCANNER_FIGURE_ROOT")
    table_root = _required_dir("RESIDUAL_CONFOUNDS_SCANNER_TABLE_ROOT")
    scanner = _scanner_table()
    raw, _domains, _features, _domain_map = load_raw_and_domains()
    raw["N_MEGA"] = raw["N_MEGA"].astype(str).str.strip()
    designs = {}
    estimates = []
    fixed_effects = None
    result_rows = []
    for bag in BAGS:
        residual = pd.read_csv(next((source_root / "source_data").glob(f"*_{BAG_SHORT[bag]}_residuals.csv")))
        confounds = pd.read_csv(next((source_root / "source_data").glob(f"*_{BAG_SHORT[bag]}_confounds.csv")))
        confounds["bag"] = bag
        estimates.append(confounds)
        terms = confounds["term"].drop_duplicates().astype(str).tolist()
        if fixed_effects is None:
            fixed_effects = terms
        elif fixed_effects != terms:
            raise ValueError("Fixed-effect terms differ across BAGs")
        for objective in OBJECTIVES:
            designs[(bag, objective)] = residual[residual["objective"].eq(objective)].copy()
            design, model_terms, _levels = _build_design(bag, objective, raw)
            if model_terms != fixed_effects:
                raise ValueError("Scanner VC fixed-effect design differs from original figure")
            result_rows.append(fit_country_scanner_variance_components(
                design, model_terms, scanner, bag=bag, objective=objective
            ))
    results = pd.concat(result_rows, ignore_index=True)
    results.to_csv(table_root / "residual_confounds_scanner_vc_variance.csv", index=False)
    for bag in BAGS:
        results[results["bag"].eq(bag)].to_csv(
            table_root / f"{bag}_scanner_vc_variance.csv", index=False
        )
    _plot_residual_confounds(
        designs, pd.concat(estimates, ignore_index=True), fixed_effects, figure_root,
        scanner_vc=results, stem="residual_confounds_scanner_vc",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
