from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_cluster_runner_is_xgb_k10_only() -> None:
    text = (ROOT / "scripts" / "jobs" / "run_main_k10_sensitivity_bag.sh").read_text()
    assert "MAIN_K10_MODE=1" in text
    assert "XGB_TUNING_STRICT=1" in text
    assert "CBN_INCLUDE_SINGLE=0" in text
    assert "SENSITIVITY_RUNGS=\"xgb_tree_d1,xgb_tree_d2,xgb_tree_d3\"" in text
    assert 'MAIN_K10_SENSITIVITY_RUNGS' in text
    assert "normative-transfer-ols" not in text
    assert "--preflight" in text
    assert "prepare_legacy_runtime" in text
    assert 'DEDUP_EXPOSOME_CSV="$EFFECTIVE_EXPOSOME"' in text
    assert 'effective_exposome_country_year.csv' in text
    assert 'DOMAIN_PC1_OINFO_REFERENCE="$DOMAIN_PC1_OINFO"' in text


def test_submit_launcher_preflight_never_contacts_scheduler() -> None:
    text = (ROOT / "scripts" / "submit_main_k10_cluster.sh").read_text()
    dry_run = text.split('if [[ "$MODE" == "--dry-run" ]]; then', 1)[1].split("fi", 1)[0]
    assert '"$JOB" "$stage" "$bag" "$cores" --preflight' in dry_run
    assert " run -t " not in dry_run


def test_print_only_handoff_has_no_submission_side_effect() -> None:
    text = (ROOT / "scripts" / "prepare_main_k10_cluster_commands.sh").read_text()
    assert "Print-only" in text
    assert "main_k10_cbn_structural" in text
    assert "country-block-null structural" in text
    assert "Fig. 2 and Fig. 3 are intentionally absent" in text


def test_main_adapter_is_new_k10_only() -> None:
    text = (ROOT / "src" / "repo_expo_hoi_bag" / "stages" / "prepare_main_k10_sensitivity_adapter.py").read_text()
    assert 'source-run-id", default="paper_reanalysis_k10"' in text
    assert '"model_family": "xgboost_k10_only"' in text
    assert '"ols_policy": "reused_reference_not_materialized"' in text
    assert 'global_frame["base_r2"] = baseline_r2' in text
    assert 'EFFECTIVE_EXPOSOME_NAME = "effective_exposome_country_year.csv"' in text
    assert 'DOMAIN_PC1_OINFO_NAME = "domain_pc1_oinfo_scores.csv"' in text
    assert '"unit": "unique_exposome_signature"' in text


def test_residual_adapter_supplies_parent_bag_target_contract() -> None:
    text = (ROOT / "src" / "repo_expo_hoi_bag" / "stages" / "prepare_main_k10_residual_inputs.py").read_text()
    assert 'frame["bag_target"] = bag' in text


def test_main_k10_evaluation_cache_is_run_scoped() -> None:
    text = (ROOT / "src" / "repo_expo_hoi_bag" / "stages" / "sensitivity_common.py").read_text()
    assert '"work" / "analysis_runs" / run_id / "sensitivity_eval" / name' in text


def test_normative_single_completion_reuses_assigned_run_and_output_tree() -> None:
    runner = (ROOT / "scripts" / "jobs" / "run_main_k10_sensitivity_bag.sh").read_text()
    launcher = (ROOT / "scripts" / "submit_main_k10_normative_single_completion.sh").read_text()

    assert "normative-transfer-single-xgb" in runner
    assert 'SINGLE_NORM_MODELS=xgb' in runner
    assert 'sensitivity/normative_transfer/$BAG"' in runner
    assert 'normative_transfer/$BAG/single_exposure' in runner
    assert 'RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"' in launcher
    assert "results/analysis_runs/$RUN_ID/input_adapter" in launcher
    assert "main_k10_single_norm_structural" in launcher
    assert "main_k10_single_norm_functional" in launcher


def test_normative_single_completion_dry_run_never_contacts_scheduler() -> None:
    text = (ROOT / "scripts" / "submit_main_k10_normative_single_completion.sh").read_text()
    dry_run = text.split('if [[ "$MODE" == "--dry-run" ]]; then', 1)[1].split("fi", 1)[0]
    assert '"$JOB" normative-transfer-single-xgb "$bag" "$cores" --preflight' in dry_run
    assert " run -t " not in dry_run


def test_normative_transfer_renderer_accepts_main_run_inputs() -> None:
    text = (ROOT / "src" / "repo_expo_hoi_bag" / "stages" / "plot_normative_transfer_grid.py").read_text()

    assert "NORM_TRANSFER_RUNGS" in text
    assert "NORM_POOLED_SINGLE_RUN_ROOT" in text
    assert "NORM_SINGLE_NORMATIVE_ROOT" in text
    assert "NORM_FIG_STEM_TEMPLATE" in text


def test_existing_correction_launcher_supports_ols_only_completion() -> None:
    text = (ROOT / "scripts" / "submit_main_k10_hpo_scope_corrections.sh").read_text()
    assert "--ols-dry-run" in text
    assert "--ols-submit" in text
    assert "MAIN_K10_SENSITIVITY_RUNGS=ols" in text
    assert "main_k10_ols_edu_structural" in text
    assert "main_k10_ols_resid_functional" in text
