from __future__ import annotations

from repo_expo_hoi_bag.runtime.cache import CacheManifest, cache_is_valid, write_cache_manifest


def test_cache_manifest_requires_exact_stage_inputs_and_parameters(tmp_path) -> None:
    manifest = CacheManifest(
        stage="greedy",
        inputs={"cohort": "abc", "metadata": "def"},
        parameters={"top_k": 5, "seed": 2026},
    )
    path = tmp_path / "cache" / "manifest.json"
    assert not cache_is_valid(path, manifest)

    write_cache_manifest(path, manifest)
    assert cache_is_valid(path, manifest)
    assert not cache_is_valid(
        path,
        CacheManifest(
            stage="greedy",
            inputs={"cohort": "abc", "metadata": "changed"},
            parameters={"top_k": 5, "seed": 2026},
        ),
    )
