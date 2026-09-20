"""External-runtime context used by the CLI and numerical stages."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import re
from typing import Iterable, Mapping

from repo_expo_hoi_bag.config.models import (
    ConfigurationError,
    HistoricalPaperReference,
    PaperConfig,
    RuntimePaths,
    load_historical_paper_reference,
)


TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})
_ANALYSIS_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


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
    analysis_run_id: str | None = None
    historical_paper_reference: HistoricalPaperReference | None = None

    @classmethod
    def create(
        cls,
        *,
        repository_root: Path,
        runtime: RuntimePaths,
        config: PaperConfig,
        bags: Iterable[str] | None = None,
        analysis_run_id: str | None = None,
    ) -> "RunContext":
        normalized_run_id = (analysis_run_id or "").strip() or None
        if normalized_run_id and not _ANALYSIS_RUN_ID_PATTERN.fullmatch(normalized_run_id):
            raise ConfigurationError(
                "analysis_run_id must contain 1-80 letters, digits, '-' or '_', "
                "and must begin with a letter or digit"
            )
        selected = tuple(bags if bags is not None else config.selected_bags())
        reference = load_historical_paper_reference(
            Path(repository_root).resolve() / "config" / "paper_reference.yaml",
            repro_data_root=runtime.data_root.parent,
        )
        return cls(
            repository_root=Path(repository_root).resolve(),
            runtime=runtime,
            config=config,
            bags=selected,
            analysis_run_id=normalized_run_id,
            historical_paper_reference=reference,
        )

    @property
    def compatibility_root(self) -> Path:
        checkout_id = sha256(str(self.repository_root).encode("utf-8")).hexdigest()[:16]
        return self.runtime.work_root / "compatibility_runtime" / f"checkout-{checkout_id}"

    @property
    def greedy_root(self) -> Path:
        if self.analysis_run_id:
            return self.runtime.work_root / "analysis_runs" / self.analysis_run_id / "greedy"
        return self.runtime.work_root / "greedy"

    @property
    def variant_root(self) -> Path:
        if self.analysis_run_id:
            return self.analysis_run_root / "variant_a"
        return self.runtime.results_root / "variant_a"

    @property
    def analysis_run_root(self) -> Path:
        """Dedicated namespace for one newly computed main analysis."""
        if not self.analysis_run_id:
            raise ConfigurationError("analysis_run_root requires an explicit analysis_run_id")
        return self.runtime.results_root / "analysis_runs" / self.analysis_run_id

    @property
    def paper_reference_root(self) -> Path:
        """Immutable deduplicated max-30 candidate universe behind the paper."""
        if self.historical_paper_reference is None:
            raise ConfigurationError("historical paper reference was not configured")
        return self.historical_paper_reference.canonical_root

    @property
    def paper_reference_backup_root(self) -> Path:
        """Optional copied backup, intentionally not the default reference source."""
        return self.runtime.results_root / "variant_a" / "paper_dedup_max30" / "canonical"

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
            "PAPER_REFERENCE_CANONICAL_ROOT": str(self.paper_reference_root),
            "REPO_ANALYSIS_RUN_ID": self.analysis_run_id or "",
        }
