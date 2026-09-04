#!/usr/bin/env python3
"""
run_full_analysis_v3.py
=======================
End-to-end v3 analysis driver.  Runs the full pipeline for one variant by
setting V3_* env vars and calling each script as a subprocess.

Variants
--------
A  (main)     CN + AD + FTD, exclude France / Italy / Egypt / Greece / Poland + MCI
B  (AD+FTD)   AD + FTD only, same country exclusions + CN excluded

Steps (run in order, can be filtered with --steps)
-------------------------------------------------------
main_pooled       Main LOCO pipeline, pooled_oinfo_ladder family (per BAG)
main_single_dx    single_dx_oinfo_ladder (per BAG)
main_portability  single_dx_portability  (per BAG, uses pooled canonical as ref)
main_cn_transfer  cn_normative_transfer  (per BAG, uses single_dx canonical as ref)
dx_stratified     Dx-stratified LOCO eval (per BAG)
transfer          CN→disease transfer eval (per BAG; skipped for variant B)
null_model        10 000-draw null model  (per BAG)
single_exposure   63 single-feature baselines (all BAGs in one call)
oof_predictions   Subject-level OOF predictions, best synergistic model (all BAGs)
shap_oof          Generate SHAP OOF outputs and publication-ready SHAP plots for the best models.
post_stats        Compute phase-2 summary stats, domain diversity, country meta-regression, and bootstrap stability.
post_figures      Generate diagnostic figures from processed v3 outputs.

Usage
-----
    # Full variant A run:
    python run_full_analysis_v3.py --variant a

    # Full variant B run:
    python run_full_analysis_v3.py --variant b

    # Specific BAGs only:
    python run_full_analysis_v3.py --variant a --bags structural functional

    # Specific steps only:
    python run_full_analysis_v3.py --variant a --steps main_pooled dx_stratified

    # SHAP-only smoke test:
    SMOKE_TEST=1 python run_full_analysis_v3.py --variant a --steps shap_oof

    # Smoke test (fast, 3 folds):
    SMOKE_TEST=1 python run_full_analysis_v3.py --variant a --steps main_pooled

    # Post-processing only (pipeline already run):
    python run_full_analysis_v3.py --variant a --steps post_stats post_figures

Notes
-----
- Set SMOKE_TEST=1 in the environment to propagate smoke mode to all subprocesses.
- N_JOBS env var controls parallelism for downstream scripts (default 32).
- The script uses sys.executable so it must be run from the correct conda env.
- The pipeline now generates the greedy candidate artifacts in `outputs/greedy` when needed.
- top_k_per_order=10 is hardcoded in config.py → 480 candidates per objective,
  960 total.  This is a v3 change from v1's 5 per order (240 per objective).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from project_paths import RAW_DATA_CSV, FEATURE_NAMES_CSV
from scripts.pipeline_utils import load_pipeline_driver_config

# ── project root (directory containing the repo) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
SMOKE_CONFIG_PATH = PROJECT_ROOT / "config" / "smoke.yaml"
PIPELINE_CONFIG, SMOKE_CONFIG, CONFIG_DEFAULTS, SMOKE_DEFAULTS = load_pipeline_driver_config(
    CONFIG_PATH,
    SMOKE_CONFIG_PATH,
)

BASE_OUTPUT_ROOT = Path(os.environ.get(
    "V3_BASE_OUTPUT_ROOT",
    str(PROJECT_ROOT / CONFIG_DEFAULTS.get("base_output_root", "outputs")),
))
GREEDY_ROOT = Path(os.environ.get(
    "V3_GREEDY_ROOT",
    str(BASE_OUTPUT_ROOT / CONFIG_DEFAULTS.get("greedy_subdir", "greedy")),
))

# ── shared constants ───────────────────────────────────────────────────────────
GREEDY_TOPK_CSV = GREEDY_ROOT / "greedy_topk_by_objective_order.csv"
GREEDY_FEATURE_NAMES_CSV = GREEDY_ROOT / "exposome_feature_names.csv"

_EXCL_COUNTRIES = "France,Italy,Egypt,Greece,Poland"

# ── variant definitions ───────────────────────────────────────────────────────
VARIANTS: Dict[str, Dict] = {
    key: {**variant, "output_root": Path(variant["output_root"]) if "output_root" in variant else BASE_OUTPUT_ROOT / variant["label"], "skip_steps": set(variant.get("skip_steps", []))}
    for key, variant in PIPELINE_CONFIG.get("variants", {}).items()
}

ALL_BAGS   = ["structural", "functional"]
ALL_STEPS  = PIPELINE_CONFIG.get("stages", [
    "greedy",
    "main_pooled",
    "main_single_dx",
    "main_portability",
    "main_cn_transfer",
    "dx_stratified",
    "transfer",
    "null_model",
    "single_exposure",
    "oof_predictions",
    "shap_oof",
    "post_stats",
    "post_figures",
  ])
def _run(cmd: List[str], env: Dict[str, str], label: str) -> None:
    """Run a subprocess, streaming stdout/stderr, raising on failure."""
    full_env = {**os.environ, **env}
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"  CMD:     {' '.join(cmd)}")
    relevant = {k: v for k, v in env.items() if k.startswith("V3_") or k in
                ("BAG_TARGET_MODE", "SMOKE_TEST", "N_JOBS", "N_NULL")}
    for k, v in sorted(relevant.items()):
        print(f"  ENV:     {k}={v}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, env=full_env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed (exit {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} — {elapsed:.0f}s")


def _base_env(variant: Dict, top_k_per_order: Optional[int] = None) -> Dict[str, str]:
    """Env vars shared by every subprocess for this variant."""
    env = {
        "V3_EXCLUDE_COUNTRIES": variant["exclude_countries"],
        "V3_EXCLUDE_DIAGNOSIS": variant["exclude_diagnosis"],
        "V3_OUTPUT_ROOT":       str(variant["output_root"]),
        "V3_GREEDY_ROOT":       str(GREEDY_ROOT),
    }
    greedy_score_mode = os.environ.get("V3_GREEDY_SCORE_MODE", "").strip().lower()
    if greedy_score_mode:
        env["V3_GREEDY_SCORE_MODE"] = greedy_score_mode
    if top_k_per_order is not None:
        env["V3_TOP_K_PER_ORDER"] = str(top_k_per_order)
    return env


def _experiments_root(variant: dict) -> str:
    """Path where runner.py writes per-experiment XGB results (used by downstream)."""
    return str(variant["output_root"] / "families" / "pooled_oinfo_ladder" / "experiments")


def _canonical_per_exp_root(variant: dict) -> str:
    """Path to per-experiment canonical metrics (used by oof_predictions)."""
    return str(
        variant["output_root"] / "families" / "pooled_oinfo_ladder"
        / "canonical" / "per_experiment"
    )


def _pooled_canonical_root(variant: dict) -> str:
    return str(variant["output_root"] / "families" / "pooled_oinfo_ladder" / "canonical")


def _single_dx_canonical_root(variant: dict) -> str:
    return str(variant["output_root"] / "families" / "single_dx_oinfo_ladder" / "canonical")


# ── step runners ──────────────────────────────────────────────────────────────

def step_greedy(variant: dict, smoke: bool) -> None:
    env = _base_env(variant)
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_exposome_greedy_only",
        "--input-csv", str(RAW_DATA_CSV),
        "--feature-list-csv", str(FEATURE_NAMES_CSV),
        "--outdir", str(GREEDY_ROOT),
    ]
    greedy_score_mode = env.get("V3_GREEDY_SCORE_MODE", "").strip().lower()
    if greedy_score_mode:
        cmd += ["--score-mode", greedy_score_mode]
    if smoke:
        cmd += [
            "--objectives", "o_max,o_min",
            "--max-order", "5",
            "--top-k-per-order", "5",
            "--greedy-repeat", "10",
        ]
    _run(cmd, env, "greedy")


def step_main_runner(
    variant: Dict,
    bag: str,
    analysis_family: str,
    extra_args: Optional[List[str]] = None,
    top_k_per_order: Optional[int] = None,
) -> None:
    env = _base_env(variant, top_k_per_order=top_k_per_order)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "oinfo_bag_ladder" / "runner.py"),
        "--bag-target-mode", bag,
        "--analysis-family", analysis_family,
        *(extra_args or []),
    ]
    _run(cmd, env, f"runner/{analysis_family}/{bag}")


def step_dx_stratified(variant: dict, bag: str, n_jobs: int, smoke: bool) -> None:
    env = {
        **_base_env(variant),
        "BAG_TARGET_MODE":    bag,
        "V3_OUTDIR_BASE":     str(variant["output_root"] / "dx_stratified_eval"),
        "V3_EXPERIMENTS_ROOT": _experiments_root(variant),
        "N_JOBS":             str(n_jobs),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.run_dx_stratified_eval"], env, f"dx_stratified/{bag}")


def step_transfer(variant: dict, bag: str, n_jobs: int, smoke: bool) -> None:
    # Transfer trains on CN; pass only base exclusions (not CN)
    # so that CN subjects remain available for training.
    base_excl_dx = ",".join(
        d for d in variant["exclude_diagnosis"].split(",")
        if d.strip() not in ("CN", "")
    )
    env = {
        **_base_env(variant),
        "V3_EXCLUDE_DIAGNOSIS": base_excl_dx,
        "BAG_TARGET_MODE":      bag,
        "V3_OUTDIR_BASE":       str(variant["output_root"] / "transfer_dx_eval"),
        "V3_EXPERIMENTS_ROOT":  _experiments_root(variant),
        "N_JOBS":               str(n_jobs),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.run_transfer_dx_eval"], env, f"transfer/{bag}")


def step_null_model(variant: dict, bag: str, n_jobs: int, smoke: bool, n_null: int) -> None:
    env = {
        **_base_env(variant),
        "BAG_TARGET_MODE":     bag,
        "V3_OUTDIR_BASE":      str(variant["output_root"] / "null_model"),
        "V3_EXPERIMENTS_ROOT": _experiments_root(variant),
        "N_JOBS":              str(n_jobs),
        "N_NULL":              "50" if smoke else str(n_null),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.run_null_model"], env, f"null_model/{bag}")


def step_single_exposure(variant: dict, bag: str, n_jobs: int, smoke: bool) -> None:
    env = {
        **_base_env(variant),
        "BAG_TARGET_MODE": bag,
        "V3_OUTDIR_BASE":  str(variant["output_root"] / "single_exposure_eval"),
        "N_JOBS":          str(n_jobs),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.run_single_exposure_eval"], env, f"single_exposure/{bag}")


def step_oof_predictions(variant: dict, bag: str, smoke: bool) -> None:
    env = {
        **_base_env(variant),
        "BAG_TARGET_MODE":   bag,
        "V3_OUTDIR_BASE":    str(variant["output_root"] / "subject_level_oof"),
        "V3_CANONICAL_ROOT": _canonical_per_exp_root(variant),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.run_best_syn_oof_predictions"], env, f"oof_predictions/{bag}")

def step_shap_oof(variant: Dict, bags: List[str], smoke: bool) -> None:
    variant_root = variant["output_root"]
    canonical_root = Path(variant_root) / "families" / "pooled_oinfo_ladder" / "canonical"
    if not canonical_root.exists():
        raise FileNotFoundError(
            f"SHAP step requires existing canonical metrics output at {canonical_root}. "
            "Run the main_pooled step first."
        )
    env = {
        **_base_env(variant),
        "V3_OUTDIR_BASE": str(variant_root / "shap_oof"),
        "V3_CANONICAL_ROOT": str(variant_root),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    print(f"  SHAP step confirmed. Using variant root: {variant_root}")
    bag_args = ["--bags", *bags]
    _run([sys.executable, "-m", "scripts.run_shap_oof", *bag_args], env, "shap_oof/run_shap_oof")
    for bag in bags:
        _run(
            [
                sys.executable,
                "-m",
                "scripts.plot_shap_summary",
                "--bag",
                bag,
                "--shap-root",
                str(variant["output_root"] / "shap_oof"),
                "--fig-dir",
                str(variant["output_root"] / "figures" / "shap"),
                "--stats-dir",
                str(variant["output_root"] / "stats" / "shap"),
            ],
            env,
            f"shap_oof/plot_shap_summary/{bag}",
        )


def step_post_stats(variant: dict, smoke: bool) -> None:
    env = {
        **_base_env(variant),
        "V3_RAW_PATH": str(RAW_DATA_CSV),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.compute_phase2_stats"], env, "post_stats/compute_phase2_stats")
    _run([sys.executable, "-m", "scripts.compute_domain_diversity_regression"], env, "post_stats/compute_domain_diversity_regression")
    _run([sys.executable, "-m", "scripts.compute_country_meta_regression"], env, "post_stats/compute_country_meta_regression")
    _run([sys.executable, "-m", "scripts.compute_bootstrap_stability"], env, "post_stats/compute_bootstrap_stability")


def step_post_figures(variant: dict, smoke: bool) -> None:
    env = {
        **_base_env(variant),
        "V3_RAW_PATH": str(RAW_DATA_CSV),
        **({"SMOKE_TEST": "1"} if smoke else {}),
    }
    _run([sys.executable, "-m", "scripts.plot_enrichment_ladder"], env, "post_figures/plot_enrichment_ladder")
    _run([sys.executable, "-m", "scripts.plot_null_model_figure"], env, "post_figures/plot_null_model_figure")
    _run([sys.executable, "-m", "scripts.plot_dx_stratified_figure"], env, "post_figures/plot_dx_stratified_figure")
    _run([sys.executable, "-m", "scripts.plot_fig2_main_results"], env, "post_figures/plot_fig2_main_results")
    _run([sys.executable, "-m", "scripts.make_exposome_domain_legend"], env, "post_figures/make_exposome_domain_legend")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 full analysis driver")
    p.add_argument("--variant", required=True, choices=["a", "b"],
                   help="Analysis variant: a = CN+AD+FTD, b = AD+FTD only")
    p.add_argument("--bags", nargs="+", default=ALL_BAGS,
                   choices=ALL_BAGS, metavar="BAG",
                   help="BAG modalities to run (default: all three)")
    p.add_argument("--steps", nargs="+", default=ALL_STEPS,
                   choices=ALL_STEPS, metavar="STEP",
                   help="Pipeline steps to run (default: all)")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", "32")),
                   help="Parallel jobs for downstream scripts (default 32)")
    p.add_argument("--n-null", type=int, default=10000,
                   help="Null model draws (default 10000)")
    p.add_argument("--greedy-score-mode", type=str, default=os.environ.get("V3_GREEDY_SCORE_MODE", "oinfo"),
                   choices=["oinfo", "os_ratio"],
                   help="Score mode for the greedy stage (default: oinfo).")
    p.add_argument("--order-min", type=int, default=None,
                   help="Override minimum candidate order for main pipeline steps")
    p.add_argument("--order-max", type=int, default=None,
                   help="Override maximum candidate order for main pipeline steps")
    p.add_argument("--candidate-limit", type=int, default=None,
                   help="Limit top-k candidates per objective/order for main pipeline steps")
    p.add_argument("--top-k-per-order", type=int, default=None,
                   help="Override top_k_per_order in config (candidates loaded from greedy CSV per order). "
                        "Default: 10 (v3). Use e.g. 20 to double the candidate space.")
    return p.parse_args()


def _manifest_artifact(path: Path, description: str) -> Dict[str, str]:
    return {
        "path": str(path.resolve()),
        "description": description,
        "exists": path.exists(),
    }


def _write_pipeline_manifest(
    variant: Dict,
    steps: List[str],
    smoke: bool,
    start_time: float,
    end_time: float,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "run_info": {
            "variant": variant["label"],
            "description": variant["description"],
            "steps": steps,
            "smoke": smoke,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "output_root": str(variant["output_root"].resolve()),
            "config_path": str(CONFIG_PATH.resolve()),
            "smoke_config_path": str(SMOKE_CONFIG_PATH.resolve()),
            "n_jobs": args.n_jobs,
            "n_null": args.n_null,
            "greedy_score_mode": args.greedy_score_mode,
            "order_min": args.order_min,
            "order_max": args.order_max,
            "candidate_limit": args.candidate_limit,
            "top_k_per_order": args.top_k_per_order,
        },
        "artifacts": [
            _manifest_artifact(GREEDY_TOPK_CSV, "Greedy top-k candidate table"),
            _manifest_artifact(GREEDY_FEATURE_NAMES_CSV, "Greedy feature list"),
            _manifest_artifact(variant["output_root"], "Variant output root"),
        ],
    }
    manifest_path = variant["output_root"] / CONFIG_DEFAULTS.get("pipeline_manifest", "pipeline_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        import json
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Wrote {manifest_path}")


def main() -> None:
    args   = parse_args()
    variant = VARIANTS[args.variant]
    smoke  = os.environ.get("SMOKE_TEST", "0").strip() in ("1", "true", "yes")
    os.environ["V3_GREEDY_SCORE_MODE"] = str(args.greedy_score_mode).strip().lower()

    steps_to_run = [s for s in args.steps if s not in variant["skip_steps"]]

    smoke_order_min = SMOKE_DEFAULTS.get("order_min", 3)
    smoke_order_max = SMOKE_DEFAULTS.get("order_max", 5)
    smoke_cand_limit = SMOKE_DEFAULTS.get("candidate_limit", 5)
    eff_order_min = args.order_min if args.order_min is not None else (smoke_order_min if smoke else None)
    eff_order_max = args.order_max if args.order_max is not None else (smoke_order_max if smoke else None)
    eff_cand_limit = args.candidate_limit if args.candidate_limit is not None else (smoke_cand_limit if smoke else None)

    print(f"\n{'#'*70}")
    print(f"  v3 Analysis — {variant['description']}")
    print(f"  Output root : {variant['output_root']}")
    print(f"  Excl. countries : {variant['exclude_countries']}")
    print(f"  Excl. diagnoses : {variant['exclude_diagnosis']}")
    print(f"  BAGs        : {args.bags}")
    print(f"  Steps       : {steps_to_run}")
    print(f"  Smoke test  : {smoke}")
    print(f"  N jobs      : {args.n_jobs}")
    if smoke or eff_order_min is not None or eff_order_max is not None or eff_cand_limit is not None:
        print(f"  Order range : {eff_order_min} – {eff_order_max}  |  Cand. limit: {eff_cand_limit}")
    if args.top_k_per_order is not None:
        print(f"  top-k/order : {args.top_k_per_order}  (→ {args.top_k_per_order * 48 * 2} candidates total)")
    print(f"{'#'*70}\n")

    variant["output_root"].mkdir(parents=True, exist_ok=True)

    # Always forward n_jobs → runner parallelism.  Also apply effective
    # order/candidate limits (set explicitly or auto-filled for smoke tests).
    main_runner_extra_args: List[str] = [
        "--xgb-n-jobs", str(args.n_jobs),
        "--ols-n-jobs", str(args.n_jobs),
    ]
    if eff_order_min is not None:
        main_runner_extra_args += ["--order-min", str(eff_order_min)]
    if eff_order_max is not None:
        main_runner_extra_args += ["--order-max", str(eff_order_max)]
    if eff_cand_limit is not None:
        main_runner_extra_args += ["--candidate-limit", str(eff_cand_limit)]

    t_total = time.time()

    for step in steps_to_run:

        # ── main LOCO pipeline steps ──────────────────────────────────────
        if step == "greedy":
            step_greedy(variant, smoke)

        elif step == "main_pooled":
            for bag in args.bags:
                step_main_runner(variant, bag, "pooled_oinfo_ladder",
                                 extra_args=main_runner_extra_args,
                                 top_k_per_order=args.top_k_per_order)

        elif step == "main_single_dx":
            for bag in args.bags:
                step_main_runner(variant, bag, "single_dx_oinfo_ladder",
                                 extra_args=main_runner_extra_args,
                                 top_k_per_order=args.top_k_per_order)

        elif step == "main_portability":
            pooled_can = _pooled_canonical_root(variant)
            for bag in args.bags:
                step_main_runner(
                    variant, bag, "single_dx_portability",
                    extra_args=["--pooled-reference-canonical-root", pooled_can, *main_runner_extra_args],
                    top_k_per_order=args.top_k_per_order,
                )

        elif step == "main_cn_transfer":
            single_dx_can = _single_dx_canonical_root(variant)
            for bag in args.bags:
                step_main_runner(
                    variant, bag, "cn_normative_transfer",
                    extra_args=["--cn-reference-canonical-root", single_dx_can, *main_runner_extra_args],
                    top_k_per_order=args.top_k_per_order,
                )

        # ── downstream steps ──────────────────────────────────────────────
        elif step == "dx_stratified":
            for bag in args.bags:
                step_dx_stratified(variant, bag, args.n_jobs, smoke)

        elif step == "transfer":
            for bag in args.bags:
                step_transfer(variant, bag, args.n_jobs, smoke)

        elif step == "null_model":
            for bag in args.bags:
                step_null_model(variant, bag, args.n_jobs, smoke, args.n_null)

        elif step == "single_exposure":
            for bag in args.bags:
                step_single_exposure(variant, bag, args.n_jobs, smoke)

        elif step == "oof_predictions":
            for bag in args.bags:
                step_oof_predictions(variant, bag, smoke)

        elif step == "shap_oof":
            step_shap_oof(variant, args.bags, smoke)

        elif step == "post_stats":
            step_post_stats(variant, smoke)

        elif step == "post_figures":
            step_post_figures(variant, smoke)

    end_time = time.time()
    _write_pipeline_manifest(variant, steps_to_run, smoke, t_total, end_time, args)

    print(f"\n{'#'*70}")
    print(f"  ALL STEPS COMPLETE — total {(end_time-t_total)/3600:.1f}h")
    print(f"  Output: {variant['output_root']}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
