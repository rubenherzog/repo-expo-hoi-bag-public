"""External-runtime context used by the CLI and numerical stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable, Mapping

from repo_expo_hoi_bag.config.models import PaperConfig, RuntimePaths


TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})


def env_bool(name: str, default: bool = False, *, environ: Mapping[str, str] | None = None) -> bool:
    """Parse a boolean environment flag consistently across stages."""
    value = (environ or os.environ).get(name, "").strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}; got {value!r}")


@dataclass(frozen=True)
class RunContext:
    """All paths and scientific selection state for one reproducible run.

    Generated files are rooted at ``runtime``. The checkout is used only for
    source, config, public inputs, delivered references, and documentation.
    """

    repository_root: Path
    runtime: RuntimePaths
    config: PaperConfig
    bags: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        repository_root: Path,
        runtime: RuntimePaths,
        config: PaperConfig,
        bags: Iterable[str] | None = None,
    ) -> "RunContext":
        selected = tuple(bags if bags is not None else config.selected_bags())
        return cls(
            repository_root=Path(repository_root).resolve(),
            runtime=runtime,
            config=config,
            bags=selected,
        )

    @property
    def compatibility_root(self) -> Path:
        return self.runtime.work_root / "compatibility_runtime"

    @property
    def greedy_root(self) -> Path:
        return self.runtime.work_root / "greedy"

    @property
    def variant_root(self) -> Path:
        return self.runtime.results_root / "variant_a"

    @property
    def canonical_per_experiment_root(self) -> Path:
        return self.variant_root / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"

    @property
    def public_input_csv(self) -> Path:
        return self.repository_root / self.config.input_csv

    def base_environment(self) -> dict[str, str]:
        """Compatibility environment for migrated legacy stages.

        The ``V3_*`` names are kept only at this boundary because the preserved
        historical stage code still consumes them. New package code should use
        ``RunContext`` directly.
        """
        return {
            **os.environ,
            "REPO_CHECKOUT_ROOT": str(self.repository_root),
            "REPRO_DATA_ROOT": str(self.runtime.data_root.parent),
            "REPRO_FIGURES_ROOT": str(self.runtime.figures_root),
            "V3_BASE_OUTPUT_ROOT": str(self.runtime.results_root),
            "V3_OUTPUT_ROOT": str(self.variant_root),
            "V3_INPUT_ROOT": str(self.variant_root),
            "V3_GREEDY_ROOT": str(self.greedy_root),
            "V3_CANONICAL_ROOT": str(self.canonical_per_experiment_root),
            "V3_RAW_PATH": str(self.public_input_csv),
            "V3_EXPOSOME_DOMAIN_LABELS_CSV": str(self.repository_root / self.config.feature_domains_csv),
            "BAG_TARGET_MODE": ",".join(self.bags),
            "PAPER_FIG_BAGS": ",".join(self.bags),
        }
