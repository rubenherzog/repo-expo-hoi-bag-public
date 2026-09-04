"""Input and output contracts for reproducible tabular artifacts."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd

from repo_expo_hoi_bag.config.models import ConfigurationError, PaperConfig

REQUIRED_COHORT_COLUMNS = frozenset({"country_clean", "Diagnosis"})
REQUIRED_DOMAIN_COLUMNS = frozenset({"feature_name", "domain"})


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(config: PaperConfig, repository_root: Path) -> dict[str, str]:
    """Validate public inputs and return stable source hashes."""
    root = Path(repository_root)
    files = {
        "cohort": root / config.input_csv,
        "feature_domains": root / config.feature_domains_csv,
        "feature_names": root / config.feature_names_csv,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ConfigurationError(f"Missing public input files: {', '.join(missing)}")
    cohort = pd.read_csv(files["cohort"], nrows=1)
    domains = pd.read_csv(files["feature_domains"], nrows=1)
    if not REQUIRED_COHORT_COLUMNS.issubset(cohort.columns):
        raise ConfigurationError("Cohort is missing required country_clean/Diagnosis columns")
    if not REQUIRED_DOMAIN_COLUMNS.issubset(domains.columns):
        raise ConfigurationError("Feature-domain metadata is missing feature_name/domain columns")
    return {name: sha256_file(path) for name, path in files.items()}
