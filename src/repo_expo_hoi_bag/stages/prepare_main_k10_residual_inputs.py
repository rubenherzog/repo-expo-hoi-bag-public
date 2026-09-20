#!/usr/bin/env python3
"""Adapt completed main k10 winner OOF files to the parent residual contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument("--adapter-run-id", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.resolve()
    source = runtime / "results" / "analysis_runs" / args.source_run_id / "main_statistics" / "model_comparison" / "oof"
    destination = runtime / "results" / "analysis_runs" / args.adapter_run_id / "main_statistics"
    paths = [source / f"level_best_{suffix}" / bag / "oof_xgb_tree_d3.parquet" for bag in ("structural", "functional") for suffix in ("syn", "red")]
    if args.smoke_test:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        print("Smoke test passed: main k10 residual OOF inputs are complete.")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing residual adapter: {destination}")
    residuals = destination / "residuals"
    residuals.mkdir(parents=True, exist_ok=False)
    outputs: list[str] = []
    selections: list[dict[str, str]] = []
    for bag in ("structural", "functional"):
        for suffix in ("syn", "red"):
            source_path = source / f"level_best_{suffix}" / bag / "oof_xgb_tree_d3.parquet"
            frame = pd.read_parquet(source_path)
            required = {"N_MEGA", "country", "diagnosis", "age", "sex", "y_true", "y_pred_full"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"OOF file lacks required residual columns: {sorted(missing)}")
            frame = frame.copy()
            frame["bag_target"] = bag
            frame["residual"] = frame["y_pred_full"] - frame["y_true"]
            frame["abs_residual"] = frame["residual"].abs()
            output = residuals / f"residuals_subject_{bag}_{suffix}.csv"
            frame.to_csv(output, index=False)
            outputs.append(str(output))
            selections.append(
                {
                    "analysis": "A_top_k_per_rung",
                    "bag": bag,
                    "objective": "o_min" if suffix == "syn" else "o_max",
                    "best_rung": "xgb_tree_d3",
                }
            )
    pd.DataFrame(selections).to_csv(destination / "best_rung_selection.csv", index=False)
    (destination / "main_k10_residual_adapter_manifest.json").write_text(
        json.dumps({"analysis_label": "main", "source_run_id": args.source_run_id, "rung": "xgb_tree_d3", "outputs": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared main-k10 residual inputs: {destination}")


if __name__ == "__main__":
    main()
