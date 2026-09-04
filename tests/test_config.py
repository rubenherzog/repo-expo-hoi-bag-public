from __future__ import annotations

import os

import pytest

from repo_expo_hoi_bag.config.models import ConfigurationError, PRIMARY_BAGS, RuntimePaths, load_paper_config
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


def test_env_bool_has_one_strict_parser() -> None:
    assert env_bool("FLAG", environ={"FLAG": "yes"})
    assert not env_bool("FLAG", default=True, environ={"FLAG": "0"})
    with pytest.raises(ValueError, match="FLAG"):
        env_bool("FLAG", environ={"FLAG": "maybe"})
