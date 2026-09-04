from __future__ import annotations

import logging
import math
import textwrap
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from exposome_labels import display_label
from oinfo_bag_ladder.config import FIGURE_DPI
from oinfo_bag_ladder.make_boss_figure import DOMAIN_COLORS, DOMAIN_ORDER
from oinfo_bag_ladder.rungs import RUNG_LABELS


LOGGER = logging.getLogger(__name__)

BAG_LABELS = {
    "structural": "Structural BAG",
    "functional": "Functional BAG",
    "combined": "Combined BAG",
}

FAMILY_OBJECTIVE = {
    "synergy": "o_min",
    "redundancy": "o_max",
}

DEFAULT_FAMILY_ORDER = ["synergy", "redundancy"]
DEFAULT_BAG_ORDER = ["structural", "functional", "combined"]
EDGE_WIDTH_LEVELS = (1.0, 3.0, 4.0, 6.0)
EDGE_COLOR_CAP = 0.1


@dataclass(frozen=True)
class PanelNetwork:
    bag: str
    family: str
    rung: str
    top_k: int
    models: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    top1_model_id: str
    top1_vars: tuple[str, ...]


@dataclass(frozen=True)
class DomainNetwork:
    bag: str
    family: str
    rung: str
    top_k: int
    models: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    top1_model_id: str
    top1_domains: tuple[str, ...]


def _split_predictors_identity(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _safe_display_label(name: str, width: int = 14) -> str:
    label = display_label(name)
    return textwrap.fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _domain_for_variable(variable: str, domain_map: dict[str, str]) -> str:
    return domain_map.get(str(variable).strip(), "Other")


def resolve_analysis_root(
    data_root: str | Path | None = None,
) -> Path:
    """Resolve the root that contains the canonical pooled ladder outputs."""
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))

    from os import environ

    env_output = environ.get("V3_OUTPUT_ROOT", "").strip()
    if env_output:
        candidates.append(Path(env_output))

    candidates.extend(
        [
            Path("outputs/variant_a"),
            Path("outputs/variant_a_diversity"),
            Path("outputs/variant_b"),
            Path("outputs"),
        ]
    )

    for candidate in candidates:
        canonical = candidate / "families" / "pooled_oinfo_ladder" / "canonical" / "per_experiment"
        if canonical.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a pooled_oinfo_ladder canonical output root. "
        "Set --data-root or V3_OUTPUT_ROOT to a valid analysis directory."
    )


def load_pooled_metrics(
    analysis_root: str | Path,
    bag: str,
    rung: str,
) -> pd.DataFrame:
    """Load the canonical pooled metrics table for one BAG and rung."""
    root = Path(analysis_root)
    canonical = (
        root
        / "families"
        / "pooled_oinfo_ladder"
        / "canonical"
        / "per_experiment"
        / f"pooled_oinfo_ladder_{bag}"
        / "metrics_global_long.parquet"
    )
    if canonical.exists():
        df = pd.read_parquet(canonical)
    else:
        experiment_dir = (
            root
            / "families"
            / "pooled_oinfo_ladder"
            / "experiments"
            / f"pooled_oinfo_ladder_{bag}"
            / rung
        )
        csv_candidates = [
            experiment_dir / "xgb_summary_with_calibration_complexity.csv",
            experiment_dir / "loco_summary_with_calibration_complexity.csv",
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find canonical metrics or stage summary for {bag!r} / {rung!r} under {root}"
            )
        df = pd.read_csv(csv_path)
        if "global_oof_r2" in df.columns and "full_r2" not in df.columns:
            df = df.rename(columns={"global_oof_r2": "full_r2"})
        if "model_id" in df.columns and "candidate_id" not in df.columns:
            df["candidate_id"] = df["model_id"].astype(str)
    required = {"bag_target", "rung_id", "objective", "full_r2", "candidate_id", "predictors_identity"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {missing}")
    return df.copy()


def select_top_models(df: pd.DataFrame, family: str, top_k: int, rung: str | None = None) -> pd.DataFrame:
    """Select the top-K models for one family by full-model R^2."""
    objective = FAMILY_OBJECTIVE[family]
    sub = df.copy()
    if rung is not None and "rung_id" in sub.columns:
        sub = sub[sub["rung_id"].astype(str) == str(rung)].copy()
    sub = sub[sub["objective"].astype(str) == objective].copy()
    sub["full_r2"] = pd.to_numeric(sub["full_r2"], errors="coerce")
    sub = sub.dropna(subset=["full_r2"])
    sub = sub.sort_values(by=["full_r2", "candidate_id"], ascending=[False, True], kind="mergesort")
    return sub.head(int(top_k)).reset_index(drop=True)


def _build_node_edge_tables(
    models: pd.DataFrame,
    *,
    bag: str,
    family: str,
    rung: str,
    top_k: int,
    domain_map: dict[str, str],
) -> PanelNetwork:
    """Create node and edge summaries from a top-K model table."""
    if models.empty:
        empty_nodes = pd.DataFrame(
            columns=[
                "bag",
                "family",
                "rung",
                "variable",
                "display_label",
                "domain",
                "frequency_topk",
                "frequency_fraction",
                "in_top1_model",
                "rank1_model_id",
                "node_size_used",
            ]
        )
        empty_edges = pd.DataFrame(
            columns=[
                "bag",
                "family",
                "rung",
                "var_a",
                "var_b",
                "cooccurrence_count",
                "normalized_cooccurrence",
                "in_top1_pair",
                "edge_width_used",
            ]
        )
        return PanelNetwork(
            bag=bag,
            family=family,
            rung=rung,
            top_k=top_k,
            models=models.copy(),
            nodes=empty_nodes,
            edges=empty_edges,
            top1_model_id="",
            top1_vars=(),
        )

    rows: list[dict[str, object]] = []
    edge_counter: Counter[tuple[str, str]] = Counter()
    node_counter: Counter[str] = Counter()

    models = models.copy().reset_index(drop=True)
    models["var_list"] = models["predictors_identity"].apply(_split_predictors_identity)
    top1_row = models.iloc[0]
    top1_model_id = str(top1_row.get("candidate_id", ""))
    top1_vars = tuple(dict.fromkeys(top1_row["var_list"]))
    top1_set = set(top1_vars)

    for _, row in models.iterrows():
        vars_here = list(dict.fromkeys(_split_predictors_identity(row.get("predictors_identity"))))
        for var in vars_here:
            node_counter[var] += 1
        for a, b in combinations(sorted(vars_here), 2):
            edge_counter[(a, b)] += 1

    node_rows: list[dict[str, object]] = []
    for var, freq in node_counter.items():
        domain = _domain_for_variable(var, domain_map)
        node_rows.append(
            {
                "bag": bag,
                "family": family,
                "rung": rung,
                "variable": var,
                "display_label": _safe_display_label(var),
                "domain": domain,
                "frequency_topk": int(freq),
                "frequency_fraction": float(freq / float(top_k)) if top_k > 0 else np.nan,
                "in_top1_model": bool(var in top1_set),
                "rank1_model_id": top1_model_id if var in top1_set else "",
                "node_size_used": np.nan,
            }
        )

    edge_rows: list[dict[str, object]] = []
    for (a, b), count in edge_counter.items():
        edge_rows.append(
            {
                "bag": bag,
                "family": family,
                "rung": rung,
                "var_a": a,
                "var_b": b,
                "cooccurrence_count": int(count),
                "normalized_cooccurrence": float(count / float(top_k)) if top_k > 0 else np.nan,
                "in_top1_pair": bool(a in top1_set and b in top1_set),
                "edge_width_used": np.nan,
            }
        )

    node_df = pd.DataFrame(node_rows)
    if not node_df.empty:
        node_df = node_df.sort_values(
            by=["frequency_topk", "domain", "display_label", "variable"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        node_df["frequency_rank"] = np.arange(1, len(node_df) + 1)

    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty:
        edge_df = edge_df.sort_values(
            by=["cooccurrence_count", "var_a", "var_b"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        edge_df["edge_rank"] = np.arange(1, len(edge_df) + 1)

    return PanelNetwork(
        bag=bag,
        family=family,
        rung=rung,
        top_k=top_k,
        models=models.copy(),
        nodes=node_df,
        edges=edge_df,
        top1_model_id=top1_model_id,
        top1_vars=top1_vars,
    )


def build_panel_network(
    analysis_root: str | Path,
    bag: str,
    family: str,
    rung: str,
    top_k: int,
    domain_map: dict[str, str],
) -> PanelNetwork:
    df = load_pooled_metrics(analysis_root, bag, rung)
    sub = df[(df["bag_target"].astype(str) == str(bag)) & (df["rung_id"].astype(str) == str(rung))].copy()
    return _build_node_edge_tables(
        select_top_models(sub, family=family, top_k=top_k, rung=rung),
        bag=bag,
        family=family,
        rung=rung,
        top_k=top_k,
        domain_map=domain_map,
    )


def build_domain_network(
    panel: PanelNetwork,
    domain_map: dict[str, str],
) -> DomainNetwork:
    """Collapse a variable-level panel into a domain-level network."""
    models = panel.models.copy().reset_index(drop=True)
    models["var_list"] = models["predictors_identity"].apply(_split_predictors_identity)
    models["domain_list"] = models["var_list"].apply(lambda vars_: [domain_map.get(v, "Other") for v in vars_])
    models["pair_opportunities"] = models["var_list"].apply(lambda vars_: math.comb(len(dict.fromkeys(vars_)), 2) if len(dict.fromkeys(vars_)) >= 2 else 0)
    total_pair_opportunities = float(models["pair_opportunities"].sum()) if not models.empty else 1.0

    domain_counts: Counter[str] = Counter()
    within_counts: Counter[str] = Counter()
    cross_counts: Counter[tuple[str, str]] = Counter()
    top1_domains = tuple(dict.fromkeys([domain_map.get(v, "Other") for v in panel.top1_vars]))
    top1_set = set(top1_domains)

    for _, row in models.iterrows():
        domains = list(dict.fromkeys([d for d in row["domain_list"] if d]))
        # model presence count per domain
        for domain in domains:
            domain_counts[domain] += 1

        # within-domain co-occurrence = all variable pairs whose endpoints share a domain
        vars_here = list(dict.fromkeys(_split_predictors_identity(row.get("predictors_identity"))))
        by_domain: dict[str, list[str]] = {}
        for var in vars_here:
            by_domain.setdefault(domain_map.get(var, "Other"), []).append(var)
        for domain, members in by_domain.items():
            if len(members) >= 2:
                within_counts[domain] += math.comb(len(members), 2)

        # cross-domain co-occurrence = all unordered cross-domain variable pairs
        domain_keys = sorted(by_domain.keys())
        for i, dom_a in enumerate(domain_keys):
            for dom_b in domain_keys[i + 1 :]:
                cross_counts[tuple(sorted((dom_a, dom_b)))] += len(by_domain[dom_a]) * len(by_domain[dom_b])

    node_rows: list[dict[str, object]] = []
    for domain in DOMAIN_ORDER:
        if domain not in domain_counts and domain not in within_counts and domain not in top1_set:
            continue
        node_rows.append(
            {
                "bag": panel.bag,
                "family": panel.family,
                "rung": panel.rung,
                "domain": domain,
                "display_label": domain,
                "domain_color": DOMAIN_COLORS.get(domain, DOMAIN_COLORS.get("Other", "#cccccc")),
                "model_presence_topk": int(domain_counts.get(domain, 0)),
                "model_presence_fraction": float(domain_counts.get(domain, 0) / float(panel.top_k)) if panel.top_k > 0 else np.nan,
                "within_domain_pair_count": int(within_counts.get(domain, 0)),
                "in_top1_model": bool(domain in top1_set),
                "rank1_model_id": panel.top1_model_id if domain in top1_set else "",
                "node_size_used": np.nan,
            }
        )

    edge_rows: list[dict[str, object]] = []
    for (dom_a, dom_b), count in cross_counts.items():
        edge_rows.append(
            {
                "bag": panel.bag,
                "family": panel.family,
                "rung": panel.rung,
                "domain_a": dom_a,
                "domain_b": dom_b,
                "cooccurrence_count": int(count),
                "normalized_cooccurrence": float(count / total_pair_opportunities) if total_pair_opportunities > 0 else np.nan,
                "in_top1_pair": bool(dom_a in top1_set and dom_b in top1_set),
                "edge_width_used": np.nan,
            }
        )

    node_df = pd.DataFrame(node_rows)
    if node_df.empty:
        node_df = pd.DataFrame(
            columns=[
                "bag",
                "family",
                "rung",
                "domain",
                "display_label",
                "domain_color",
                "model_presence_topk",
                "model_presence_fraction",
                "within_domain_pair_count",
                "in_top1_model",
                "rank1_model_id",
                "node_size_used",
            ]
        )
    if not node_df.empty:
        node_df = node_df.sort_values(
            by=["within_domain_pair_count", "model_presence_topk", "domain"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        node_df["frequency_rank"] = np.arange(1, len(node_df) + 1)

    edge_df = pd.DataFrame(edge_rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(
            columns=[
                "bag",
                "family",
                "rung",
                "domain_a",
                "domain_b",
                "cooccurrence_count",
                "normalized_cooccurrence",
                "in_top1_pair",
                "edge_width_used",
            ]
        )
    if not edge_df.empty:
        edge_df = edge_df.sort_values(
            by=["cooccurrence_count", "domain_a", "domain_b"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        edge_df["edge_rank"] = np.arange(1, len(edge_df) + 1)

    return DomainNetwork(
        bag=panel.bag,
        family=panel.family,
        rung=panel.rung,
        top_k=panel.top_k,
        models=models.copy(),
        nodes=node_df,
        edges=edge_df,
        top1_model_id=panel.top1_model_id,
        top1_domains=top1_domains,
    )


def _weighted_graph_from_edges(
    edges: pd.DataFrame, *, source_col: str, target_col: str, weight_col: str = "cooccurrence_count"
) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row[source_col], row[target_col], weight=float(row[weight_col]))
    return graph


def compute_network_stats(panel: PanelNetwork, domain: DomainNetwork) -> pd.DataFrame:
    """Connectivity summary for one (bag, family, rung): degree, modularity, domain mixing.

    `panel` carries the variable-level co-occurrence graph used for degree and
    modularity; `domain` carries the domain-level collapse used for the
    intra/inter-domain ratio (within_domain_pair_count vs cross-domain edge count).
    """
    graph = _weighted_graph_from_edges(panel.edges, source_col="var_a", target_col="var_b")
    for node in panel.nodes["variable"]:
        graph.add_node(node)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    weighted_degree = dict(graph.degree(weight="weight"))
    mean_weighted_degree = float(np.mean(list(weighted_degree.values()))) if weighted_degree else np.nan

    modularity = np.nan
    n_communities = 0
    if n_edges > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        n_communities = len(communities)
        modularity = float(nx.algorithms.community.quality.modularity(graph, communities, weight="weight"))

    within_total = float(domain.nodes["within_domain_pair_count"].sum()) if not domain.nodes.empty else 0.0
    cross_total = float(domain.edges["cooccurrence_count"].sum()) if not domain.edges.empty else 0.0
    intra_inter_ratio = (
        within_total / cross_total if cross_total > 0 else (np.inf if within_total > 0 else np.nan)
    )

    return pd.DataFrame(
        [
            {
                "bag": panel.bag,
                "family": panel.family,
                "rung": panel.rung,
                "top_k": panel.top_k,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "mean_weighted_degree": mean_weighted_degree,
                "modularity": modularity,
                "n_communities": n_communities,
                "within_domain_pair_total": within_total,
                "cross_domain_pair_total": cross_total,
                "intra_inter_domain_ratio": intra_inter_ratio,
            }
        ]
    )


def compute_edge_permutation_null(
    panel: PanelNetwork,
    *,
    n_permutations: int = 1000,
    seed: int = 20260304,
) -> pd.DataFrame:
    """Empirical null for edge co-occurrence counts.

    Permutes variable labels within each model independently (preserving each
    model's set size, i.e. its order) and recomputes pairwise co-occurrence
    counts, `n_permutations` times. Returns one row per observed edge with the
    empirical p-value `p = (1 + #{null_count >= observed_count}) / (n_permutations + 1)`.
    """
    rng = np.random.default_rng(seed)
    models = panel.models
    if models.empty or panel.edges.empty:
        return pd.DataFrame(
            columns=[
                "bag",
                "family",
                "rung",
                "var_a",
                "var_b",
                "observed_count",
                "null_mean",
                "null_p95",
                "p_value",
            ]
        )

    all_vars = sorted(panel.nodes["variable"].tolist())
    var_lists = [list(dict.fromkeys(v)) for v in models["var_list"].tolist()]
    sizes = [len(v) for v in var_lists]

    observed_counter: Counter[tuple[str, str]] = Counter()
    for var_list in var_lists:
        for a, b in combinations(sorted(var_list), 2):
            observed_counter[(a, b)] += 1

    null_counts: dict[tuple[str, str], list[int]] = {pair: [] for pair in observed_counter}
    for _ in range(int(n_permutations)):
        draw_counter: Counter[tuple[str, str]] = Counter()
        for size in sizes:
            draw = rng.choice(all_vars, size=size, replace=False)
            for a, b in combinations(sorted(draw), 2):
                draw_counter[(a, b)] += 1
        for pair in null_counts:
            null_counts[pair].append(draw_counter.get(pair, 0))

    rows = []
    for (a, b), observed in observed_counter.items():
        draws = np.asarray(null_counts[(a, b)], dtype=float)
        p_value = float((1 + np.sum(draws >= observed)) / (len(draws) + 1))
        rows.append(
            {
                "bag": panel.bag,
                "family": panel.family,
                "rung": panel.rung,
                "var_a": a,
                "var_b": b,
                "observed_count": int(observed),
                "null_mean": float(np.mean(draws)) if len(draws) else np.nan,
                "null_p95": float(np.quantile(draws, 0.95)) if len(draws) else np.nan,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["p_value", "observed_count"], ascending=[True, False], kind="mergesort").reset_index(
            drop=True
        )
    return out
