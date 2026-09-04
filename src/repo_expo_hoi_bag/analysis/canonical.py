"""Canonical result loaders shared by analyses, figures, and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from repo_expo_hoi_bag.config.models import ConfigurationError
from repo_expo_hoi_bag.data.contracts import sha256_file


@dataclass(frozen=True)
class CanonicalTable:
    """A named canonical table with a repo-relative path and checksum."""

    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class CanonicalTableStore:
    """Read-only access to canonical tables under one root."""

    root: Path

    def resolve(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigurationError(f"Canonical table path must stay under {self.root}: {relative_path}")
        resolved = self.root / path
        if not resolved.is_file():
            raise ConfigurationError(f"Missing canonical table: {resolved}")
        return resolved

    def load_csv(self, relative_path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a canonical CSV with consistent path validation."""
        return pd.read_csv(self.resolve(relative_path), **kwargs)

    def hashes(self, relative_paths: Iterable[str | Path]) -> tuple[CanonicalTable, ...]:
        """Return SHA-256 hashes for canonical files in manifest order."""
        rows: list[CanonicalTable] = []
        for relative_path in relative_paths:
            path = self.resolve(relative_path)
            rows.append(
                CanonicalTable(
                    name=Path(relative_path).as_posix(),
                    path=path,
                    sha256=sha256_file(path),
                )
            )
        return tuple(rows)
