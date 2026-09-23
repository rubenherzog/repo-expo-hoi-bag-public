#!/usr/bin/env python3
"""Set-size sensitivity (Supplementary Fig. S10 and ST18) under global OOF R2.

Estimand counterpart of
``finalize_main_k10_delivery.refresh_country_balanced_order_cap_tests``.  The
analysis is unchanged -- the same maximum set sizes 5..30, the same parent
reconciliation and plotting functions, the same pooled d3 selection -- and only
the R2 estimand that the winners are selected and reported on differs.

It writes the ST18 selection table and the figure together and asserts that
they carry identical winners, so the figure can never disagree with its table.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BAGS = ("structural", "functional")  # structural precedes functional
CAPS = list(range(5, 31))
POOLED_RUNG = "xgb_tree_d3"
EXPECTED_SOURCE = "main_fig2_global_oof"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.resolve()
    run_root = ROOT / "outputs" / "main" / args.run_id / "sensitivity"
    figures = run_root / "figures"
    tables = run_root / "selection_sensitivities"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    existing = figures / "fig_s10_set_size_sensitivity.pdf"
    if existing.exists():
        raise FileExistsError(f"Refusing to overwrite existing global figure: {existing}")

    os.environ["R2_MODE"] = "global_oof"
    os.environ["NORM_TRANSFER_RUNGS"] = POOLED_RUNG
    os.environ.setdefault(
        "NORM_POOLED_CANONICAL_ROOT",
        str(runtime / "results/analysis_runs" / args.run_id / "input_adapter"),
    )
    os.environ.setdefault(
        "NORM_POOLED_MAIN_RUN_ROOT",
        str(runtime / "results/analysis_runs/paper_reanalysis_k10"),
    )

    from repo_expo_hoi_bag.stages import compute_order_cap_reconciliation as stage
    from repo_expo_hoi_bag.stages.plot_normative_transfer_grid import load_plot_data

    stage.ORDER_CAPS = CAPS
    stage.OUT_DIR = run_root / "order_cap"
    stage.FIGURES_DIR = figures
    stage.OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for bag in BAGS:
        data = load_plot_data(bag)
        pooled = data[
            data["condition"].astype(str).eq("Pooled")
            & data["rung_id"].astype(str).eq(POOLED_RUNG)
        ].copy()
        sources = set(pooled["source"].dropna().astype(str))
        if sources != {EXPECTED_SOURCE}:
            raise ValueError(
                f"Set-size analysis must use the exact main Fig. 2 global-OOF scores "
                f"for {bag}; found sources={sorted(sources)}"
            )
        frames[bag] = pooled

    recon = stage.reconciliation_long(frames, {})
    selection = pd.concat(
        [
            recon[["bag", "order_cap", "best_syn_order", "best_syn_r2"]]
            .rename(columns={"order_cap": "maximum_set_size", "best_syn_order": "selected_order", "best_syn_r2": "global_oof_r2"})
            .assign(objective="o_min"),
            recon[["bag", "order_cap", "best_red_order", "best_red_r2"]]
            .rename(columns={"order_cap": "maximum_set_size", "best_red_order": "selected_order", "best_red_r2": "global_oof_r2"})
            .assign(objective="o_max"),
        ],
        ignore_index=True,
    )
    # Candidate identity for each selected winner, using exactly the tie-break
    # rule of reconciliation_long: r2 desc, order asc, candidate_id asc.
    picks = []
    for bag, frame in frames.items():
        for cap in CAPS:
            capped = frame[pd.to_numeric(frame["order"], errors="coerce") <= cap]
            for objective in ("o_min", "o_max"):
                scoped = capped[capped["objective"].astype(str).eq(objective)]
                if scoped.empty:
                    continue
                winner = scoped.sort_values(
                    by=["r2", "order", "candidate_id"],
                    ascending=[False, True, True],
                    na_position="last",
                ).iloc[0]
                picks.append(
                    {
                        "bag": bag,
                        "objective": objective,
                        "maximum_set_size": cap,
                        "candidate_id": str(winner["candidate_id"]),
                    }
                )
    selection = selection.merge(
        pd.DataFrame(picks), on=["bag", "objective", "maximum_set_size"], validate="one_to_one"
    )
    selection = selection.sort_values(["bag", "objective", "maximum_set_size"]).reset_index(drop=True)
    selection["r2_mode"] = "global_oof"
    selection = selection[
        ["bag", "objective", "maximum_set_size", "candidate_id", "selected_order", "global_oof_r2", "r2_mode"]
    ]
    st18 = tables / "ST18_set_size_cap_selection.csv"
    selection.to_csv(st18, index=False)

    betas = stage.diversity_betas_by_cap(frames)
    by_order = stage.r2_by_order(frames)
    recon.to_csv(stage.OUT_DIR / stage.RECON_CSV_NAME, index=False)
    by_order.to_csv(stage.OUT_DIR / "r2_by_order.csv", index=False)
    betas.to_csv(stage.OUT_DIR / "diversity_r2_betas_by_cap.csv", index=False)
    stage.plot_s10_set_size_sensitivity(recon, betas)

    # The figure and ST18 must describe exactly the same winners.
    plotted_wide = pd.read_csv(
        figures / "source_data" / "fig_s10_set_size_sensitivity_source_data_b_selected_set_size.csv"
    )
    plotted = pd.concat(
        [
            plotted_wide[["bag", "order_cap", "best_syn_order"]]
            .rename(columns={"order_cap": "maximum_set_size", "best_syn_order": "figure_order"})
            .assign(objective="o_min"),
            plotted_wide[["bag", "order_cap", "best_red_order"]]
            .rename(columns={"order_cap": "maximum_set_size", "best_red_order": "figure_order"})
            .assign(objective="o_max"),
        ],
        ignore_index=True,
    )
    merged = selection.merge(
        plotted, on=["bag", "objective", "maximum_set_size"], validate="one_to_one"
    )
    if len(merged) != len(selection):
        raise ValueError("Set-size figure and ST18 do not cover the same rows")
    if not merged["selected_order"].astype(int).eq(merged["figure_order"].astype(int)).all():
        raise ValueError("Set-size figure winners do not match the global ST18 source")

    manifest = {
        "r2_mode": "global_oof",
        "run_id": args.run_id,
        "maximum_set_sizes": [CAPS[0], CAPS[-1]],
        "pooled_rung": POOLED_RUNG,
        "pooled_source": EXPECTED_SOURCE,
        "st18_rows": int(len(selection)),
        "figure_table_agreement": "verified",
    }
    (run_root / "global_oof_set_size_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Rendered global-OOF set-size sensitivity: {figures} and {st18}")


if __name__ == "__main__":
    main()
