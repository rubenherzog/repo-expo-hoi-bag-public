#!/usr/bin/env python3
"""Build the main workbook from locally completed k10-derived analyses only."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--main-run-id", default="main")
    parser.add_argument("--local-run-id", default="main_local_20260915_v4")
    parser.add_argument("--output-name", default="Supplementary_Tables_main_k10.xlsx")
    parser.add_argument("--adapter-subdir", default="table_adapters")
    parser.add_argument("--xgb-comparison-run-id", default="main")
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=CHECKOUT_ROOT / "outputs/main/paper/complete/tables",
        help="Checkout delivery path for the lightweight supplementary workbook.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args(); root = args.repro_data_root.resolve() / "results" / "analysis_runs"
    main, local = root / args.main_run_id, root / args.local_run_id / "local_analyses"
    adapters = main / args.adapter_subdir
    xgb_comparison = root / args.xgb_comparison_run_id / "model_comparison"
    sources = {
        "ST01_HigherOrder": adapters / "ST01_greedy_oinfo_by_order.csv",
        "ST02_ModelComplexity": xgb_comparison / "complexity_and_arm_comparisons.csv",
        "ST02_EffectSize": xgb_comparison / "level_winner_cohen_f2.csv",
        "ST03_BestVsSingle": adapters / "ST03_best_multivariate_vs_single.csv",
        "ST05_Top50": adapters / "ST05_top50_composition.csv",
        "ST06_Diversity": local / "diversity_permutation_d3.csv",
        "ST07_DomainComposition": local / "domain_composition_top20_d3.csv",
        "ST08_Networks": local / "network_permutation_d3.csv",
        "ST08_NetworkStats": local / "network_stats_d3.csv",
        "ST09_TripletsD3": local / "recurrent_triplets_top20_d3.csv",
        "ST09_TripletsAllLevels": adapters / "ST09_recurrent_triplets_all_levels.csv",
        "ST10_ResidualSummary": main / "derived" / "main_residual_subject_summary.csv",
        "ST10_ResidualTests": main / "derived" / "main_residual_tests.csv",
        "ST14_NegativeOmegaSelection": main / "selection_sensitivities" / "ST14_negative_omega_selection.csv",
        "ST18_SetSizeSelection": main / "selection_sensitivities" / "ST18_set_size_cap_selection.csv",
        "ST12_MixedLMFixed": main / "secondary_oof_stats" / "ST12_residual_mixedlm_fixed_effects.csv",
        "ST12_MixedLMVariance": main / "secondary_oof_stats" / "ST12_residual_mixedlm_variance.csv",
        "ST16_CountryMeta": main / "secondary_oof_stats" / "ST16_country_meta_regression.csv",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing completed local main inputs: {missing}")
    if args.smoke_test:
        print(f"Smoke test passed: {len(sources)} local table inputs")
        return
    output = args.delivery_root.resolve() / args.output_name
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    status = pd.DataFrame([
        {"table": "ST01", "status": "complete_local", "reason": "greedy/O-information structure; identical by definition"},
        {"table": "ST02", "status": "complete_local", "reason": "k10 OOF country-bootstrap comparisons"},
        {"table": "ST03", "status": "complete_local", "reason": "k10 OOF best-multivariate versus best-single"},
        {"table": "ST04", "status": "pending_cluster", "reason": "country-block permutation refits"},
        {"table": "ST05", "status": "complete_local", "reason": "top-50 country-balanced set-size composition permutation"},
        {"table": "ST06", "status": "complete_local", "reason": "country-balanced diversity permutation"},
        {"table": "ST07", "status": "complete_local", "reason": "k10 top-20 domain composition"},
        {"table": "ST08", "status": "complete_local", "reason": "k10 top-20 network permutation"},
        {"table": "ST09", "status": "complete_local", "reason": "permitted Ω triplet index, d3 and all-level aggregation"},
        {"table": "ST10", "status": "complete_local", "reason": "k10 OOF residual summaries and country bootstrap"},
        {"table": "ST11", "status": "pending_cluster", "reason": "requires new normative transfer refits"},
        {"table": "ST12", "status": "partial_local", "reason": "MixedLM fitted from k10 OOF; parametric-bootstrap LRT remains pending"},
        {"table": "ST13", "status": "pending_cluster", "reason": "requires alternative-representation refits"},
        {"table": "ST14", "status": "partial_local", "reason": "negative-Ω k10 selection complete; matched winner OOF refit pending"},
        {"table": "ST15", "status": "pending_cluster", "reason": "requires diagnostic context and weighting refits"},
        {"table": "ST16", "status": "partial_local", "reason": "country meta-regression complete; LORO refit sensitivity pending"},
        {"table": "ST17", "status": "pending_cluster", "reason": "requires covariate and target refits"},
        {"table": "ST18", "status": "partial_local", "reason": "k10 selection by caps complete; matched cap-winner OOF refits pending"},
    ])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        status.to_excel(writer, sheet_name="README", index=False)
        for sheet, source in sources.items(): pd.read_csv(source).to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"Saved: {output}")


if __name__ == "__main__": main()
