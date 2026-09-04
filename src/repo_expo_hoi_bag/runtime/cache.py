"""Small hash-based cache manifests for expensive stages."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CacheManifest:
    """A reproducibility cache key stored next to generated artifacts."""

    stage: str
    inputs: Mapping[str, str]
    parameters: Mapping[str, str | int | float | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "inputs": dict(sorted(self.inputs.items())),
            "parameters": dict(sorted(self.parameters.items())),
        }


def cache_is_valid(path: Path, manifest: CacheManifest) -> bool:
    """Return True when an existing cache manifest exactly matches."""
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        current = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return current == manifest.as_dict()


def write_cache_manifest(path: Path, manifest: CacheManifest) -> None:
    """Write a stable manifest after an expensive stage completes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
