#!/usr/bin/env python3
"""Local selection-only negative-Ω and set-size-cap sensitivities for main k10."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


BAGS = ("structural", "functional")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-data-root", type=Path, required=True); p.add_argument("--source-run-id", default="paper_reanalysis_k10"); p.add_argument("--main-run-id", default="main"); p.add_argument("--candidate-registry", type=Path, required=True); p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    a = _args(); runtime = a.repro_data_root.resolve(); source = runtime / "results/analysis_runs" / a.source_run_id
    registry = pd.read_parquet(a.candidate_registry).copy(); registry["bag"] = registry.experiment_id.astype(str).str.removeprefix("pooled_oinfo_ladder_")
    # ``country_balanced_r2`` is the carrier column for the active estimand;
    # under R2_MODE=global_oof it holds the pooled global OOF R2 instead.
    global_oof = os.environ.get("R2_MODE", "").strip() == "global_oof"
    parts = []
    for bag in BAGS:
        leaf = source / "xgb" / bag / "xgb_tree_d3" / "k10"
        if global_oof:
            x = pd.read_csv(leaf / "metrics_global.csv")
            x = x.groupby("candidate_id", observed=True).global_oof_r2.max().rename("country_balanced_r2").reset_index()
        else:
            x = pd.read_csv(leaf / "metrics_country.csv")
            x = x[x.n_test > 0].groupby("candidate_id", observed=True).r2.mean().rename("country_balanced_r2").reset_index()
        x["bag"] = bag; parts.append(x)
    score = pd.concat(parts, ignore_index=True).merge(registry[["candidate_id", "bag", "objective", "order", "thoi_o", "predictors_identity"]], on=["candidate_id", "bag"], validate="one_to_one")
    if a.smoke_test:
        print(f"Smoke test passed: {len(score)} new k10 d3 scores")
        return
    out = runtime / "results/analysis_runs" / a.main_run_id / "selection_sensitivities"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    negative_rows = []
    for bag in BAGS:
        for objective in ("o_min", "o_max"):
            base = score[(score.bag.eq(bag)) & (score.objective.eq(objective))]
            restricted = base[base.thoi_o.lt(0)] if objective == "o_min" else base
            winner = restricted.nlargest(1, "country_balanced_r2").iloc[0]
            negative_rows.append({"bag": bag, "objective": objective, "criterion": "evaluated_omega_negative" if objective == "o_min" else "path_only_redundancy", "n_candidates": len(restricted), "candidate_id": winner.candidate_id, "order": winner.order, "thoi_o": winner.thoi_o, "country_balanced_r2": winner.country_balanced_r2})
    pd.DataFrame(negative_rows).to_csv(out / "ST14_negative_omega_selection.csv", index=False)
    cap_rows = []
    for cap in range(5, 31):
        for bag in BAGS:
            for objective in ("o_min", "o_max"):
                winner = score[(score.bag.eq(bag)) & (score.objective.eq(objective)) & (score.order.le(cap))].nlargest(1, "country_balanced_r2").iloc[0]
                cap_rows.append({"bag": bag, "objective": objective, "maximum_set_size": cap, "candidate_id": winner.candidate_id, "selected_order": winner.order, "country_balanced_r2": winner.country_balanced_r2})
    pd.DataFrame(cap_rows).to_csv(out / "ST18_set_size_cap_selection.csv", index=False)
    print(f"Saved local selection sensitivities: {out}")


if __name__ == "__main__": main()
