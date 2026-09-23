#!/usr/bin/env python3
"""Write consolidated source-data tables for paired hierarchical OOF contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.stages.compute_hierarchical_oof_improvement import ROOT

QUESTIONS = (
    "top20_vs_baseline",
    "top20_vs_best_single",
    "synergy_vs_redundancy",
    "model_complexity",
    "topk_frontier_sensitivity",
)


def build(results_path: Path, output_dir: Path, workbook: Path) -> tuple[Path, ...]:
    results = pd.read_csv(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with pd.ExcelWriter(workbook) as writer:
        for question in QUESTIONS:
            frame = results[results["question"].eq(question)].copy()
            if frame.empty:
                continue
            path = output_dir / f"paired_hierarchical_oof_{question}.csv"
            frame.to_csv(path, index=False)
            frame.to_excel(writer, sheet_name=question[:31], index=False)
            written.append(path)
    return tuple(written)


def main() -> None:
    root = ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/paired_bootstrap"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "paired_hierarchical_oof_comparisons.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/main/paper/complete/tables/source_data/hierarchical_oof")
    parser.add_argument("--workbook", type=Path, default=ROOT / "outputs/main/paper/complete/tables/Supplementary_Table_Paired_Hierarchical_OOF.xlsx")
    args = parser.parse_args()
    for path in build(args.results, args.output_dir, args.workbook):
        print(f"Saved: {path}")
    print(f"Saved: {args.workbook}")


if __name__ == "__main__":
    main()
