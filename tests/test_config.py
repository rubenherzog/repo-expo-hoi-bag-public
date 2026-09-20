from __future__ import annotations

import os
from pathlib import Path

import pytest

from repo_expo_hoi_bag.config.models import (
    ConfigurationError,
    PRIMARY_BAGS,
    RuntimePaths,
    load_historical_paper_reference,
    load_paper_config,
)
from repo_expo_hoi_bag.runtime.context import RunContext, env_bool


def test_paper_config_defaults_to_primary_bags() -> None:
    config = load_paper_config("config/paper.yaml")
    assert config.selected_bags() == PRIMARY_BAGS


def test_runtime_paths_requires_external_root(monkeypatch) -> None:
    monkeypatch.delenv("REPRO_DATA_ROOT", raising=False)
    with pytest.raises(ConfigurationError, match="REPRO_DATA_ROOT"):
        RuntimePaths.from_environment()


def test_runtime_paths_use_explicit_external_root(tmp_path) -> None:
    paths = RuntimePaths.from_environment(os.fspath(tmp_path / "runtime"))
    paths.ensure_output_directories()
    assert paths.figures_root.is_dir()


def test_run_context_centralizes_external_paths(tmp_path) -> None:
    root = os.path.abspath(os.curdir)
    config = load_paper_config("config/paper.yaml")
    runtime = RuntimePaths.from_environment(os.fspath(tmp_path / "runtime"))
    context = RunContext.create(repository_root=root, runtime=runtime, config=config)
    env = context.base_environment()
    assert env["REPRO_DATA_ROOT"] == os.fspath(tmp_path / "runtime")
    assert env["V3_OUTPUT_ROOT"] == os.fspath(runtime.results_root / "variant_a")
    assert env["V3_GREEDY_ROOT"] == os.fspath(runtime.work_root / "greedy")
    assert context.bags == PRIMARY_BAGS


def test_main_analysis_run_is_isolated_from_the_immutable_paper_reference(tmp_path) -> None:
    root = os.path.abspath(os.curdir)
    config = load_paper_config("config/paper.yaml")
    runtime = RuntimePaths.from_environment(os.fspath(tmp_path / "runtime"))
    context = RunContext.create(
        repository_root=root,
        runtime=runtime,
        config=config,
        analysis_run_id="hpo_gp_current3_20260912",
    )
    assert context.variant_root == runtime.results_root / "analysis_runs" / "hpo_gp_current3_20260912" / "variant_a"
    assert context.greedy_root == runtime.work_root / "analysis_runs" / "hpo_gp_current3_20260912" / "greedy"
    assert context.paper_reference_root == (
        runtime.data_root.parent.parent
        / "expo_hoi_bag_repro_data_dedup"
        / "results/variant_a/families/pooled_oinfo_ladder/canonical"
    )
    assert context.paper_reference_backup_root == runtime.results_root / "variant_a" / "paper_dedup_max30" / "canonical"
    assert context.paper_reference_root != context.variant_root
    assert context.base_environment()["PAPER_REFERENCE_CANONICAL_ROOT"] == os.fspath(context.paper_reference_root)


def test_analysis_run_id_rejects_paths_and_other_unsafe_identifiers(tmp_path) -> None:
    config = load_paper_config("config/paper.yaml")
    runtime = RuntimePaths.from_environment(os.fspath(tmp_path / "runtime"))
    with pytest.raises(ConfigurationError, match="analysis_run_id"):
        RunContext.create(
            repository_root=os.path.abspath(os.curdir),
            runtime=runtime,
            config=config,
            analysis_run_id="../paper_reference",
        )


def test_historical_paper_reference_is_external_read_only_and_catalogs_sensitivities() -> None:
    reference = load_historical_paper_reference(
        Path("config/paper_reference.yaml"), repro_data_root=Path("/tmp/current_runtime")
    )
    assert reference.root.is_absolute()
    assert reference.canonical_root == reference.root / "results/variant_a/families/pooled_oinfo_ladder/canonical"
    assert reference.sensitivity_relative_path == Path("sensitivity")
    assert reference.oof_relative_path == Path("results/variant_a/subject_level_oof_max_30")


def test_env_bool_has_one_strict_parser() -> None:
    assert env_bool("FLAG", environ={"FLAG": "yes"})
    assert not env_bool("FLAG", default=True, environ={"FLAG": "0"})
    with pytest.raises(ValueError, match="FLAG"):
        env_bool("FLAG", environ={"FLAG": "maybe"})
