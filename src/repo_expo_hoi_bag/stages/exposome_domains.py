from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_DOMAIN_LABELS_CSV = Path(__file__).resolve().parents[1] / "data" / "exposome_feature_domains.csv"


def _read_domain_csv(path: Path, feature_col: str, domain_col: str) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Domain label CSV not found: {path}")
    df = pd.read_csv(path)
    if feature_col not in df.columns or domain_col not in df.columns:
        raise ValueError(
            f"Domain label CSV must contain columns {feature_col!r} and {domain_col!r}: {path}"
        )
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        feature = str(row[feature_col]).strip()
        domain = str(row[domain_col]).strip()
        if feature and domain and domain.lower() != "nan":
            out[feature] = domain
    return out


def load_domain_map(
    domain_labels_csv: str | Path = DEFAULT_DOMAIN_LABELS_CSV,
    *,
    feature_col: str = "feature_name",
    domain_col: str = "domain",
) -> dict[str, str]:
    """Load the canonical feature -> exposome domain mapping."""
    return _read_domain_csv(Path(domain_labels_csv), feature_col, domain_col)


def load_domain_labels(
    feature_names: Sequence[str],
    *,
    domain_labels_csv: str | Path,
    feature_col: str = "feature_name",
    domain_col: str = "domain",
    missing_policy: str = "error",
) -> list[str]:
    """Return one domain label per feature from an explicit feature-domain CSV."""
    domain_map = _read_domain_csv(Path(domain_labels_csv), feature_col, domain_col)

    policy = str(missing_policy).strip().lower()
    if policy not in {"other", "error"}:
        raise ValueError("missing_policy must be 'other' or 'error'.")

    labels: list[str] = []
    missing: list[str] = []
    for feature in feature_names:
        key = str(feature)
        domain = domain_map.get(key)
        if domain is None:
            missing.append(key)
            domain = "Other"
        labels.append(domain)

    if missing and policy == "error":
        raise ValueError(f"Missing domain labels for {len(missing)} features. First missing: {missing[:10]}")
    return labels


def domain_sizes(domain_labels: Sequence[str]) -> dict[str, int]:
    return dict(Counter(str(x) for x in domain_labels))


def domain_weights(domain_labels: Sequence[str], alpha: float) -> dict[str, float]:
    sizes = domain_sizes(domain_labels)
    raw_weights = {domain: float(1.0 / (size ** float(alpha))) for domain, size in sizes.items()}
    total = float(sum(raw_weights.values()))
    if total <= 0:
        return {domain: 0.0 for domain in raw_weights}
    return {domain: float(value / total) for domain, value in raw_weights.items()}


def weighted_domain_entropy(
    selected_domains: Sequence[str],
    *,
    all_domain_labels: Sequence[str],
    alpha: float,
    base: float = 2.0,
) -> tuple[float, dict[str, float]]:
    weights = domain_weights(all_domain_labels, alpha)
    mass: Counter[str] = Counter()
    for domain in selected_domains:
        mass[str(domain)] += float(weights[str(domain)])
    total = float(sum(mass.values()))
    if total <= 0:
        return 0.0, {}
    proportions = {domain: float(value / total) for domain, value in sorted(mass.items())}
    probs = np.asarray(list(proportions.values()), dtype=float)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log(probs) / np.log(float(base))).sum())
    return entropy, proportions


def domain_summary_for_indices(
    indices: Iterable[int],
    *,
    feature_names: Sequence[str],
    domain_labels: Sequence[str],
    alpha: float,
) -> dict[str, object]:
    idxs = [int(i) for i in indices]
    domains = [str(domain_labels[i]) for i in idxs]
    counts = Counter(domains)
    total = len(domains)
    entropy, weighted_proportions = weighted_domain_entropy(
        domains,
        all_domain_labels=domain_labels,
        alpha=alpha,
    )
    max_domain_frac = float(max(counts.values()) / total) if total else 0.0
    dominant_domain = max(counts, key=counts.get) if counts else ""
    return {
        "n_domains": int(len(counts)),
        "domain_counts": dict(sorted(counts.items())),
        "domain_sequence": domains,
        "weighted_domain_proportions": weighted_proportions,
        "weighted_entropy": entropy,
        "max_domain_frac": max_domain_frac,
        "dominant_domain": dominant_domain,
        "nplet_vars": [str(feature_names[i]) for i in idxs],
    }


def unweighted_domain_stats(
    feature_names: Sequence[str],
    *,
    domain_map: dict[str, str],
) -> dict[str, object]:
    """Compute ordinary domain-count Shannon stats for a set of feature names."""
    domains = [domain_map.get(str(feature).strip(), "Other") for feature in feature_names]
    counts = Counter(domains)
    total = len(domains)
    if total == 0:
        return {
            "n_domains": 0,
            "shannon_h": 0.0,
            "max_domain_frac": 0.0,
            "dominant_domain": "",
            "domain_counts": {},
        }
    fracs = np.asarray(list(counts.values()), dtype=float) / float(total)
    shannon_h = float(-(fracs * np.log2(fracs)).sum())
    dominant_domain = max(counts, key=counts.get)
    return {
        "n_domains": int(len(counts)),
        "shannon_h": shannon_h,
        "max_domain_frac": float(max(fracs)),
        "dominant_domain": dominant_domain,
        "domain_counts": dict(sorted(counts.items())),
    }
