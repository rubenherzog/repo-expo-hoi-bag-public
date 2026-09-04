"""Declarative registry for all approved paper and sensitivity figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FigureTarget:
    name: str
    group: str
    generator: str
    formats: tuple[str, ...]
    bags: tuple[str, ...]
    reference: str


@dataclass(frozen=True)
class FigureSet:
    """A flat, versioned collection of approved figure assets."""

    name: str
    root: str
    assets: tuple[str, ...]


def load_manifest(path: Path) -> tuple[FigureTarget, ...]:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    targets = raw.get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError("Figure manifest requires a non-empty targets list")
    return tuple(
        FigureTarget(
            name=item["name"], group=item["group"], generator=item["generator"],
            formats=tuple(item["formats"]), bags=tuple(item["bags"]), reference=item["reference"],
        )
        for item in targets
    )


def active_targets(targets: tuple[FigureTarget, ...]) -> tuple[FigureTarget, ...]:
    return tuple(target for target in targets if "combined" not in target.bags)


def load_figure_sets(path: Path) -> tuple[FigureSet, ...]:
    """Load optional exact asset contracts for versioned paper figure sets."""
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    sets = raw.get("sets", [])
    if not isinstance(sets, list):
        raise ValueError("Figure manifest sets must be a list")
    return tuple(
        FigureSet(name=item["name"], root=item["root"], assets=tuple(item["assets"]))
        for item in sets
    )
