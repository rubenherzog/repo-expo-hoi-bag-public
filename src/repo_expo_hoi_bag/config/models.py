"""Configuration contracts shared by all analysis and rendering stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Literal

import yaml

BagName = Literal["functional", "structural", "combined"]
PRIMARY_BAGS: tuple[BagName, ...] = ("structural", "functional")
ALL_BAGS: tuple[BagName, ...] = (*PRIMARY_BAGS, "combined")


class ConfigurationError(ValueError):
    """Raised when a paper configuration cannot be used safely."""


@dataclass(frozen=True)
class RuntimePaths:
    """Absolute external paths; runtime outputs are never written to the checkout."""

    data_root: Path
    results_root: Path
    work_root: Path
    figures_root: Path

    @classmethod
    def from_environment(cls, repro_data_root: str | None = None) -> "RuntimePaths":
        root_value = repro_data_root or os.environ.get("REPRO_DATA_ROOT")
        if not root_value:
            raise ConfigurationError(
                "REPRO_DATA_ROOT is required. Set it to a writable external directory; "
                "repo-expo-hoi-bag never writes generated artifacts locally."
            )
        root = Path(root_value).expanduser().resolve()
        return cls(
            data_root=root / "data",
            results_root=root / "results",
            work_root=root / "work",
            figures_root=root / "figures",
        )

    def ensure_output_directories(self) -> None:
        for path in (self.results_root, self.work_root, self.figures_root):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PaperConfig:
    """Scientific invariants for the published analysis."""

    input_csv: Path
    feature_domains_csv: Path
    feature_names_csv: Path
    bags: tuple[BagName, ...]
    rung_order: tuple[str, ...]
    top_k: int
    order_min: int
    order_max: int
    random_seed: int
    include_combined: bool = False

    def selected_bags(self) -> tuple[BagName, ...]:
        return ALL_BAGS if self.include_combined else PRIMARY_BAGS


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a YAML mapping")
    return value


def load_paper_config(path: Path, *, include_combined: bool = False) -> PaperConfig:
    """Load a portable config and reject missing scientific invariants."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle), "paper config")
    inputs = _require_mapping(raw.get("inputs"), "inputs")
    analysis = _require_mapping(raw.get("analysis"), "analysis")
    try:
        bags = tuple(analysis["bags"])
        if bags not in {PRIMARY_BAGS, ALL_BAGS}:
            raise ConfigurationError(
                "analysis.bags must declare structural, functional, with optional combined last"
            )
        if include_combined and bags != ALL_BAGS:
            raise ConfigurationError(
                "include_combined=True requires combined to be declared in analysis.bags"
            )
        return PaperConfig(
            input_csv=Path(inputs["cohort_csv"]),
            feature_domains_csv=Path(inputs["feature_domains_csv"]),
            feature_names_csv=Path(inputs["feature_names_csv"]),
            bags=bags,  # type: ignore[arg-type]
            rung_order=tuple(analysis["rung_order"]),
            top_k=int(analysis["top_k"]),
            order_min=int(analysis["order_min"]),
            order_max=int(analysis["order_max"]),
            random_seed=int(analysis["random_seed"]),
            include_combined=include_combined,
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing required configuration key: {exc.args[0]}") from exc
