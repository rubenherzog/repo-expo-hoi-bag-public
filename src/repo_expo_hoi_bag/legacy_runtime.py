"""Compatibility bridge used while the numerical stages are migrated module by module."""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import os
import shutil
import subprocess
import sys
from typing import Iterable

from repo_expo_hoi_bag.config.models import RuntimePaths
from repo_expo_hoi_bag.data.contracts import sha256_file
from repo_expo_hoi_bag.runtime.cache import CacheManifest, cache_is_valid, write_cache_manifest
from repo_expo_hoi_bag.runtime.context import RunContext


class LegacyStageError(RuntimeError):
    """Raised when a compatibility stage cannot be launched."""


def _link(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise LegacyStageError(
            f"Compatibility path {link} already exists and is not the expected symlink. "
            "Remove it manually before running the stage."
        )
    link.symlink_to(target, target_is_directory=True)


def prepare_legacy_runtime(repository_root: Path, runtime: RuntimePaths) -> Path:
    """Build an external execution layout from active package source."""
    package_root = Path(repository_root) / "src" / "repo_expo_hoi_bag"
    stages_source = package_root / "stages"
    core_source = package_root / "core"
    if not stages_source.is_dir() or not core_source.is_dir():
        raise LegacyStageError("Missing active stages or core source package")
    runtime.ensure_output_directories()
    root = runtime.work_root / "compatibility_runtime"
    shutil.copytree(stages_source, root / "scripts", dirs_exist_ok=True)
    shutil.copytree(core_source / "oinfo_bag_ladder", root / "oinfo_bag_ladder", dirs_exist_ok=True)
    for source in core_source.glob("*.py"):
        shutil.copy2(source, root / source.name)
    resources = stages_source / "resources"
    if resources.is_dir():
        shutil.copytree(resources, root / "config", dirs_exist_ok=True)
    _link(root / "data", Path(repository_root) / "data")
    _link(root / "outputs", runtime.results_root)
    _link(root / "paper_figures", runtime.figures_root)
    return root


def _environment(runtime: RuntimePaths, legacy_root: Path, bags: Iterable[str]) -> dict[str, str]:
    context = RunContext.create(
        repository_root=Path(__file__).resolve().parents[2],
        runtime=runtime,
        config=_shim_config_for_compatibility(Path(__file__).resolve().parents[2]),
        bags=bags,
    )
    return _environment_from_context(context, legacy_root)


def _environment_from_context(context: RunContext, legacy_root: Path) -> dict[str, str]:
    env = context.base_environment()
    package_root = context.repository_root / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(legacy_root), str(package_root), os.environ.get("PYTHONPATH", "")]
    )
    return env


def run_stage(
    repository_root: Path,
    runtime: RuntimePaths,
    stage: str,
    bags: Iterable[str],
    *,
    sensitivity: str | None = None,
) -> None:
    """Launch a preserved numerical stage with all generated paths externalized."""
    context = RunContext.create(
        repository_root=repository_root,
        runtime=runtime,
        config=_shim_config_for_compatibility(repository_root),
        bags=bags,
    )
    legacy_root = prepare_legacy_runtime(repository_root, runtime)
    env = _environment_from_context(context, legacy_root)
    stage_steps = {
        "greedy": ["greedy"],
        "evaluate": ["main_pooled"],
        "analyses": ["post_stats"],
    }
    sensitivity_modules = {
        "domain-imbalance": "scripts.run_domain_imbalance_sensitivity",
        "country-region": "scripts.run_country_region_sensitivity",
        "whole-exposome-pca": "scripts.run_whole_exposome_pca_sensitivity",
        "order-cap": "scripts.compute_order_cap_reconciliation",
        "cooccurrence-network": "scripts.compute_cooccurrence_network_stats",
        "education-scanner-baseline": "scripts.run_education_scanner_baseline_sensitivity",
        "residual-confounds": "scripts.compute_residual_confounds",
        "residualized-bag-target": "scripts.compute_residualized_bag_target",
        "residualized-bag": "scripts.run_residualized_bag_sensitivity",
        "diagnosis-balance": "scripts.run_diagnosis_balance_sensitivity",
        "negative-o-arm-comparison": "scripts.compute_negative_o_arm_comparison",
        "normative-transfer-summary": "scripts.compute_normative_transfer_stats",
        "normative-transfer-ols": "scripts.run_ols_normative_loco_pipeline",
        "normative-transfer-xgb": "scripts.run_xgb_normative_loco_pipeline",
    }
    if sensitivity:
        try:
            command = [sys.executable, "-m", sensitivity_modules[sensitivity]]
        except KeyError as exc:
            raise LegacyStageError(f"Unknown sensitivity target: {sensitivity}") from exc
    else:
        try:
            command = [
                sys.executable,
                "-m",
                "scripts.run_full_analysis_v3",
                "--variant",
                "a",
                "--bags",
                *bags,
                "--steps",
                *stage_steps[stage],
            ]
        except KeyError as exc:
            raise LegacyStageError(f"Unknown pipeline stage: {stage}") from exc
    cache_path, cache_manifest = _stage_cache(context, stage=stage, sensitivity=sensitivity)
    if cache_is_valid(cache_path, cache_manifest):
        print(f"Stage cache hit; skipping {stage}{f'/{sensitivity}' if sensitivity else ''}: {cache_path}")
        return
    result = subprocess.run(command, cwd=legacy_root, env=env, check=False)
    if result.returncode:
        raise LegacyStageError(f"Stage {' '.join(command)} failed with exit code {result.returncode}")
    write_cache_manifest(cache_path, cache_manifest)


def render_target(
    repository_root: Path,
    runtime: RuntimePaths,
    target: str,
    bags: Iterable[str],
) -> None:
    """Render an approved figure family using external canonical results."""
    context = RunContext.create(
        repository_root=repository_root,
        runtime=runtime,
        config=_shim_config_for_compatibility(repository_root),
        bags=bags,
    )
    legacy_root = prepare_legacy_runtime(repository_root, runtime)
    env = _environment_from_context(context, legacy_root)
    modules = {
        "fig2_grid_v3": "scripts.plot_fig2_grid_v3",
        "fig3_diversity_v2": "scripts.plot_fig3_diversity_v2",
        "fig4_residual_bias": "scripts.plot_fig4_residual_bias",
        "domain_imbalance": "scripts.run_domain_imbalance_sensitivity",
        "country_region": "scripts.run_country_region_sensitivity",
        "whole_exposome_pca": "scripts.run_whole_exposome_pca_sensitivity",
        "order_cap": "scripts.compute_order_cap_reconciliation",
        "education_scanner_baseline": "scripts.run_education_scanner_baseline_sensitivity",
        "residual_confounds": "scripts.compute_residual_confounds",
        "residualized_bag": "scripts.run_residualized_bag_sensitivity",
        "normative_diversity": "scripts.plot_normative_diversity_r2",
        "normative_transfer": "scripts.plot_normative_transfer_grid",
    }
    try:
        module = modules[target]
    except KeyError as exc:
        raise LegacyStageError(f"Unknown render target: {target}") from exc
    caps = (None,)
    for cap in caps:
        call_env = env.copy()
        if cap is not None:
            call_env["NORM_FIG_SUFFIX"] = f"max_{cap}"
            call_env["NORM_TRANSFER_ORDER_MAX"] = str(cap)
        result = subprocess.run([sys.executable, "-m", module], cwd=legacy_root, env=call_env, check=False)
        if result.returncode:
            label = f"{target} cap={cap}" if cap is not None else target
            raise LegacyStageError(f"Render target {label} failed with exit code {result.returncode}")


def _shim_config_for_compatibility(repository_root: Path):
    from repo_expo_hoi_bag.config.models import load_paper_config

    return load_paper_config(Path(repository_root) / "config" / "paper.yaml")


def _stage_cache(
    context: RunContext,
    *,
    stage: str,
    sensitivity: str | None,
) -> tuple[Path, CacheManifest]:
    label = f"{stage}_{sensitivity}" if sensitivity else stage
    inputs = {
        "cohort": sha256_file(context.public_input_csv),
        "feature_domains": sha256_file(context.repository_root / context.config.feature_domains_csv),
        "feature_names": sha256_file(context.repository_root / context.config.feature_names_csv),
        "source_tree": _source_tree_hash(context.repository_root / "src" / "repo_expo_hoi_bag"),
    }
    parameters = {
        "bags": ",".join(context.bags),
        "order_min": context.config.order_min,
        "order_max": context.config.order_max,
        "top_k": context.config.top_k,
        "seed": context.config.random_seed,
    }
    return (
        context.runtime.results_root / "manifests" / "stages" / f"{label}.json",
        CacheManifest(stage=label, inputs=inputs, parameters=parameters),
    )


def _source_tree_hash(package_root: Path) -> str:
    digest = sha256()
    for path in sorted(Path(package_root).rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(package_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
