from __future__ import annotations

from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "config",
    REPO_ROOT / "docs",
    REPO_ROOT / "README.md",
)
FORBIDDEN_PATTERNS = (
    re.compile(r"/" + "home/rherzog"),
    re.compile("Documents/" + "Brainlat"),
    re.compile("miniconda3/" + "envs"),
    re.compile("/data/" + "workspaces"),
    re.compile(r"exposome_hoi/output_v4"),
    re.compile(r"expo_thoi"),
)
REQUIRED_GITIGNORE_PATTERNS = {
    ".code-review-graph/",
    ".mcp.json",
    ".pytest_cache/",
    "*.egg-info/",
    "REPRO_DATA_ROOT/",
    # outputs/ is excluded one level at a time so the versioned delivered figure
    # subtree can be re-included; git cannot re-include through an excluded dir.
    "outputs/*",
    "work/",
    "*.parquet",
    "*.pkl",
    "*.joblib",
    "*.npy",
    "*.npz",
}
PAPER_ANALYSIS_STAGES = (
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/run_diagnosis_balance_sensitivity.py",
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/compute_regression_residuals.py",
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/compute_residual_confounds.py",
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/compute_country_meta_regression.py",
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/compute_normative_transfer_stats.py",
    REPO_ROOT / "src/repo_expo_hoi_bag/stages/compute_negative_o_arm_comparison.py",
)


def _text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCANNED_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".txt"}:
                files.append(path)
    return files


def test_active_sources_do_not_contain_personal_absolute_paths() -> None:
    offenders: list[str] = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append(str(path.relative_to(REPO_ROOT)))
                break
    assert offenders == []


def test_gitignore_excludes_local_audit_and_runtime_artifacts() -> None:
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert REQUIRED_GITIGNORE_PATTERNS.issubset(patterns)
    # Delivered figures are the versioned part of outputs/.
    # Both negations are needed: git cannot re-include a path whose parent
    # directory is excluded, so each level is unwound in turn.
    assert {"!outputs/figures/", "!outputs/figures/dedup/"}.issubset(patterns)


def test_paper_bias_stages_read_shared_model_and_cohort_configuration() -> None:
    config_path = (
        REPO_ROOT
        / "src/repo_expo_hoi_bag/stages/resources/sensitivity.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paper = config["paper_analysis"]

    assert paper["primary_diagnoses"]
    assert paper["control_diagnosis"] in paper["primary_diagnoses"]
    assert set(paper["diagnosis_labels"]) == set(paper["primary_diagnoses"])
    assert set(paper["objectives"]) == {"o_min", "o_max"}
    assert set(paper["objective_metadata"]) == set(paper["objectives"])
    assert paper["deployed_rung"] in config["defaults"]["active_rungs"]
    assert int(paper["order_max"]) > 0
    assert "{order_max}" in paper["reference_oof_dir_template"]

    forbidden_literals = (
        re.compile(r'RUNG_ID\s*=\s*["\']xgb_tree_d[123]["\']'),
        re.compile(r'ORDER_MAX\s*=\s*\d+'),
        re.compile(r'max_30'),
        re.compile(r'["\']max_depth["\']\s*:\s*3'),
        re.compile(r'\b(?:PRIMARY_)?DIAGNOSES\s*=\s*[\[\{\(]'),
    )
    offenders: list[str] = []
    for path in PAPER_ANALYSIS_STAGES:
        source = path.read_text(encoding="utf-8")
        if any(pattern.search(source) for pattern in forbidden_literals):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
