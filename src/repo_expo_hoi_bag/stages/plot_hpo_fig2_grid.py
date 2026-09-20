#!/usr/bin/env python3
"""Render Figure 2 from new HPO LOCO results only.

Candidate membership and O-information envelopes come from the immutable greedy
catalogue.  Every displayed model comparison uses newly calculated,
country-balanced LOCO R².  Triplet synergy is recomputed from the public
country-year exposome matrix, never from a previous figure's enrichment table.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("warning")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from repo_expo_hoi_bag.config.models import load_historical_paper_reference
from repo_expo_hoi_bag.figures.source_data import write_source_data

ROOT = Path(__file__).resolve().parents[3]
RUNG_ORDER = ("ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
BAGS = ("structural", "functional")
TOP_K = 50
_FIG2: Any | None = None


def _figure_module() -> Any:
    global _FIG2
    if _FIG2 is None:
        from repo_expo_hoi_bag.stages import plot_fig2_grid_v3
        _FIG2 = plot_fig2_grid_v3
    return _FIG2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--hpo-set", choices=("k10", "k63", "all"), default="all")
    parser.add_argument(
        "--analysis-run-id",
        help="Analysis-run directory supplying the evaluated metrics; defaults to paper_reanalysis_<hpo-set>.",
    )
    parser.add_argument(
        "--candidate-scope",
        help="Metrics subdirectory for candidate sets; defaults to --hpo-set.",
    )
    parser.add_argument(
        "--output-label",
        help="Label used only in delivered HPO figure paths and filenames; defaults to --hpo-set.",
    )
    parser.add_argument("--r2-estimator", choices=("country-balanced", "global-oof"), default="country-balanced")
    parser.add_argument("--output-directory", type=Path, help="Exact delivery directory; defaults to <output-root>/<label>/paper/complete.")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "figures" / "HPO")
    parser.add_argument("--triplet-batch-size", type=int, default=10_000)
    return parser.parse_args()


def _reference_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return load_historical_paper_reference(
        ROOT / "config" / "paper_reference.yaml"
    ).root.resolve()


def _metrics_paths(
    root: Path,
    analysis_run_id: str,
    candidate_scope: str,
    bag: str,
    rung: str | None = None,
) -> tuple[Path, Path, Path]:
    """Candidate, baseline, and single metrics_country.csv paths for one cell."""
    if rung is None:
        hpo_set = analysis_run_id
        rung = bag
        bag = candidate_scope
        analysis_run_id = f"paper_reanalysis_{hpo_set}"
        candidate_scope = hpo_set
    analysis_root = root / "results" / "analysis_runs" / analysis_run_id
    fallback_root = root / "results" / "analysis_runs" / "paper_reanalysis_k10"

    def existing_or_fallback(path: Path, fallback: Path) -> Path:
        return path if path.is_file() else fallback

    if rung == "ols":
        arm = analysis_root / "ols" / bag / "ols"
        fallback_arm = fallback_root / "ols" / bag / "ols"
        return tuple(
            existing_or_fallback(arm / name / "metrics_country.csv", fallback_arm / name / "metrics_country.csv")
            for name in ("ols", "baseline", "single")
        )  # type: ignore[return-value]
    arm = analysis_root / "xgb" / bag / rung
    fallback_arm = fallback_root / "xgb" / bag / rung
    return (
        arm / candidate_scope / "metrics_country.csv",
        existing_or_fallback(arm / "baseline" / "metrics_country.csv", fallback_arm / "baseline" / "metrics_country.csv"),
        existing_or_fallback(arm / "single" / "metrics_country.csv", fallback_arm / "single" / "metrics_country.csv"),
    )


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required new analysis artifact is missing: {path}")
    return path


def _r2_metrics(path: Path, estimator: str) -> pd.DataFrame:
    if estimator == "global-oof":
        global_path = path.with_name("metrics_global.csv")
        frame = pd.read_csv(_require(global_path))
        if "candidate_id" not in frame or "global_oof_r2" not in frame:
            raise ValueError(f"{global_path} lacks candidate_id/global_oof_r2")
        return frame[["candidate_id", "global_oof_r2"]].rename(columns={"global_oof_r2": "country_balanced_r2"})
    frame = pd.read_csv(_require(path))
    required = {"candidate_id", "r2", "n_test"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    valid = frame[pd.to_numeric(frame["n_test"], errors="coerce") > 0].copy()
    valid["r2"] = pd.to_numeric(valid["r2"], errors="raise")
    return valid.groupby("candidate_id", as_index=False, observed=True)["r2"].mean().rename(
        columns={"r2": "country_balanced_r2"}
    )


def _registry_for_bag(registry: pd.DataFrame, bag: str) -> pd.DataFrame:
    selected = registry[registry["experiment_id"].eq(f"pooled_oinfo_ladder_{bag}")].copy()
    selected = selected[["candidate_id", "objective", "order", "thoi_o", "predictors_identity"]]
    if selected.empty or selected["candidate_id"].duplicated().any():
        raise ValueError(f"Official candidate registry is invalid for bag={bag!r}")
    return selected


def _load_bag(
    repro_root: Path,
    analysis_run_id: str,
    candidate_scope: str,
    estimator: str,
    bag: str,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    frames: list[pd.DataFrame] = []
    singles: dict[str, float] = {}
    sources: list[str] = []
    for rung in RUNG_ORDER:
        candidate_path, baseline_path, single_path = _metrics_paths(
            repro_root, analysis_run_id, candidate_scope, bag, rung
        )
        candidate = _r2_metrics(candidate_path, estimator)
        baseline = _r2_metrics(baseline_path, estimator)
        single = _r2_metrics(single_path, estimator)
        if len(baseline) != 1:
            raise ValueError(f"Expected one country-balanced baseline in {baseline_path}")
        merged = candidate.merge(registry, on="candidate_id", how="inner", validate="one_to_one")
        if len(merged) != 1120:
            raise ValueError(f"Candidate registry mismatch for {candidate_path}")
        base = float(baseline.loc[0, "country_balanced_r2"])
        merged["rung_id"] = rung
        merged["full_r2"] = merged["country_balanced_r2"]
        merged["base_r2"] = base
        merged["delta_r2_vs_base"] = merged["full_r2"] - base
        frames.append(merged)
        singles[rung] = float(single["country_balanced_r2"].max())
        metric_paths = (candidate_path, baseline_path, single_path)
        if estimator == "global-oof":
            metric_paths = tuple(path.with_name("metrics_global.csv") for path in metric_paths)
        sources.extend(map(str, metric_paths))
    return pd.concat(frames, ignore_index=True), singles, sources


def _load_greedy(reference_root: Path) -> pd.DataFrame:
    figure = _figure_module()
    path = reference_root / "work" / "greedy" / "greedy_topk_by_objective_order.csv"
    frame = pd.read_csv(_require(path))
    frame = frame[pd.to_numeric(frame["order"], errors="raise") <= 30].copy()
    frame["omega"] = figure.extract_omega(frame)
    return frame


def _domain_diversity(greedy: pd.DataFrame) -> pd.DataFrame:
    names = pd.read_csv(ROOT / "data" / "metadata" / "exposome_feature_names.csv")["feature_name"].tolist()
    domains = pd.read_csv(ROOT / "data" / "metadata" / "exposome_feature_domains.csv").set_index("feature_name")["domain"]
    index_to_domain = {index: domains.get(name, "Unknown") for index, name in enumerate(names)}
    rows = []
    for row in greedy[greedy["objective"].isin(("o_min", "o_max"))].itertuples(index=False):
        indices = [int(value) for value in str(row.nplet_indices).strip("[]").split(",")]
        counts = pd.Series([index_to_domain[index] for index in indices]).value_counts()
        fractions = counts / len(indices)
        rows.append({"objective": row.objective, "order": int(row.order), "n_domains": len(counts), "shannon_h": float(-(fractions * np.log2(fractions)).sum())})
    return pd.DataFrame(rows)


def _load_all(
    repro_root: Path,
    analysis_run_id: str,
    candidate_scope: str,
    estimator: str,
    reference_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    registry_path = reference_root / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical" / "candidate_registry.parquet"
    registry = pd.read_parquet(_require(registry_path))
    greedy = _load_greedy(reference_root)
    result: dict[str, Any] = {}
    sources = [str(registry_path), str(reference_root / "work" / "greedy" / "greedy_topk_by_objective_order.csv")]
    for bag in BAGS:
        global_frame, single, cell_sources = _load_bag(
            repro_root, analysis_run_id, candidate_scope, estimator, bag, _registry_for_bag(registry, bag)
        )
        result[bag] = {"global": global_frame, "greedy": greedy, "single_exp_r2": single, "single_exp_f2": {}}
        sources.extend(cell_sources)
    result["_domain_diversity"] = _domain_diversity(greedy)
    return result, sources


def _selected_rows(data: dict[str, Any], hpo_set: str) -> pd.DataFrame:
    rows = []
    for bag in BAGS:
        for rung in RUNG_ORDER:
            frame = data[bag]["global"]
            for objective in ("o_min", "o_max"):
                rows.append(frame[(frame["rung_id"] == rung) & (frame["objective"] == objective)].nlargest(TOP_K, "delta_r2_vs_base").assign(hpo_set=hpo_set, bag=bag))
    return pd.concat(rows, ignore_index=True)


def _countryyear_exposome() -> tuple[np.ndarray, list[str]]:
    raw = pd.read_csv(ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv", low_memory=False)
    features = pd.read_csv(ROOT / "data" / "metadata" / "exposome_feature_names.csv")["feature_name"].tolist()
    grouped = raw.groupby(["country_clean", "exposome_year"], as_index=False, dropna=True)[features].mean()
    matrix = grouped[features].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any").to_numpy(dtype=float)
    if matrix.shape[1] != len(features) or len(matrix) < 50:
        raise ValueError("Invalid current country-year exposome matrix for triplet calculation")
    return matrix, features


def _reusable_triplet_cache(root: Path) -> tuple[pd.DataFrame, float] | None:
    """Load prior O-information summaries only when they share this immutable input."""
    cache_dir = root / "results" / "analysis_runs" / "paper_reanalysis_fig2_inputs"
    table_path = cache_dir / "triplet_measurements_country_balanced.parquet"
    manifest_path = cache_dir / "triplet_measurements_manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        return None
    manifest = pd.read_json(manifest_path)
    expected_source = str(ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv")
    if len(manifest) != 1 or str(manifest.loc[0, "source_cohort"]) != expected_source:
        return None
    required = {"candidate_id", "bag", "rung_id", "objective", "n_triplets", "n_synergistic_triplets", "frac_neg_k3"}
    table = pd.read_parquet(table_path)
    if required.difference(table.columns):
        return None
    return table, float(manifest.loc[0, "negative_triplet_fraction"])


def _compute_triplets(
    selected: pd.DataFrame,
    batch_size: int,
    reusable: tuple[pd.DataFrame, float] | None,
) -> tuple[pd.DataFrame, float]:
    """Reuse compatible candidate summaries and calculate O-information only for missing sets."""
    from thoi.measures.gaussian_copula import nplets_measures

    matrix, features = _countryyear_exposome()
    keys = ["candidate_id", "bag", "rung_id", "objective"]
    summary_columns = ["n_triplets", "n_synergistic_triplets", "frac_neg_k3"]
    if reusable is None:
        all_indices = np.asarray(list(combinations(range(len(features)), 3)), dtype=int)
        measured = nplets_measures(matrix, nplets=all_indices, batch_size=batch_size, verbose=0).detach().cpu().numpy()
        omega = measured[:, 0, 2]
        baseline = float((omega < 0).mean())
        by_triplet = {tuple(indices): float(value) for indices, value in zip(all_indices, omega)}
        missing = selected.copy()
        joined = selected.copy()
    else:
        cached, baseline = reusable
        cached = cached[keys + summary_columns].drop_duplicates(keys)
        by_triplet: dict[tuple[int, int, int], float] = {}
        joined = selected.merge(cached, on=keys, how="left", validate="one_to_one")
        missing = joined[joined["frac_neg_k3"].isna()].copy()
    if missing.empty:
        return joined, baseline
    feature_index = {feature: index for index, feature in enumerate(features)}
    if reusable is not None:
        required_indices: set[tuple[int, int, int]] = set()
        for row in missing.itertuples(index=False):
            predictor_indices = [feature_index[feature] for feature in str(row.predictors_identity).split("|")]
            required_indices.update(combinations(predictor_indices, 3))
        requested = np.asarray(sorted(required_indices), dtype=int)
        measured = nplets_measures(matrix, nplets=requested, batch_size=batch_size, verbose=0).detach().cpu().numpy()
        by_triplet.update({tuple(indices): float(value) for indices, value in zip(requested, measured[:, 0, 2])})
    summaries = []
    for row in missing.itertuples(index=False):
        predictors = tuple(str(row.predictors_identity).split("|"))
        values = [by_triplet[tuple(sorted((feature_index[feature] for feature in triplet)))] for triplet in combinations(predictors, 3)]
        summaries.append({"n_triplets": len(values), "n_synergistic_triplets": int(np.sum(np.asarray(values) < 0)), "frac_neg_k3": float(np.mean(np.asarray(values) < 0))})
    joined.loc[missing.index, summary_columns] = pd.DataFrame(summaries, index=missing.index)
    out = joined
    return out, baseline


def _write_triplets(root: Path, output_label: str, table: pd.DataFrame, baseline: float) -> Path:
    out_dir = root / "results" / "analysis_runs" / f"paper_reanalysis_fig2_inputs_{output_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "triplet_measurements_country_balanced.parquet"
    table.to_parquet(path, index=False)
    pd.DataFrame([{"countryyear_rows": 248, "universe_triplets": 39711, "negative_triplet_fraction": baseline,
                   "source_cohort": str(ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv")}]).to_json(out_dir / "triplet_measurements_manifest.json", orient="records", indent=2)
    return path


def _render(data: dict[str, Any], hpo_set: str, output_dir: Path, triplets: pd.DataFrame, baseline: float, sources: list[str], estimator: str) -> list[Path]:
    figure = _figure_module()
    figure.BAGS, figure.ORDER_MAX, figure.ORDER_XLIM, figure.SUBCOMB_BASELINE_K3 = list(BAGS), 30, (3, 30), baseline
    data["_enrichment_per_rung"] = triplets[["bag", "rung_id", "objective", "frac_neg_k3", "full_r2"]].rename(columns={"rung_id": "rung"}).assign(mode="frac_neg_k3")
    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    figure.draw_panel_a(fig.add_subplot(grid[0, 0]), data["structural"]["greedy"], figure.ROW_LABELS[0])
    figure.draw_panel_n_domains(fig.add_subplot(grid[0, 1]), data["_domain_diversity"], "", show_legend=False)
    figure.draw_panel_shannon(fig.add_subplot(grid[0, 2]), data["_domain_diversity"], "", show_legend=False)
    for index, bag in enumerate(BAGS, start=1):
        figure.draw_panel_b(fig.add_subplot(grid[index, 0]), data[bag]["global"], figure.ROW_LABELS[index], show_legend=index == 1, single_exp_r2=data[bag]["single_exp_r2"])
        figure.draw_panel_c(fig.add_subplot(grid[index, 1]), data[bag]["global"], "")
        figure.draw_panel_d(fig.add_subplot(grid[index, 2]), data["_enrichment_per_rung"], bag, "", show_legend=index == 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig2_grid_v3_max_30_hpo_{hpo_set}"
    written = []
    for ext in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    panels = figure._build_fig2_panels(data, list(BAGS))
    for panel in panels:
        label = "global pooled held-out LOCO R²" if estimator == "global-oof" else "country-balanced held-out LOCO R² (simple mean over countries)"
        panel.columns["full_r2"] = label
        panel.columns["baseline_r2"] = label.replace("held-out", "covariate-only")
        panel.columns["best_single_r2"] = f"best single-exposure {label}"
    written.extend(write_source_data(stem, panels, output_dir, source_paths=sources))
    return written


def main() -> None:
    args = _args()
    repro_root, reference_root = args.repro_data_root.resolve(), _reference_root(args.reference_root)
    if args.hpo_set == "all" and any(value is not None for value in (args.analysis_run_id, args.candidate_scope, args.output_label)):
        raise ValueError("--analysis-run-id, --candidate-scope and --output-label require one --hpo-set")
    sets = ("k10", "k63") if args.hpo_set == "all" else (args.hpo_set,)
    all_data, sources, selected = {}, {}, []
    for hpo_set in sets:
        analysis_run_id = args.analysis_run_id or f"paper_reanalysis_{hpo_set}"
        candidate_scope = args.candidate_scope or hpo_set
        output_label = args.output_label or hpo_set
        data, source_paths = _load_all(repro_root, analysis_run_id, candidate_scope, args.r2_estimator, reference_root)
        all_data[output_label], sources[output_label] = data, source_paths
        selected.append(_selected_rows(data, output_label))
    triplets, baseline = _compute_triplets(
        pd.concat(selected, ignore_index=True), args.triplet_batch_size, _reusable_triplet_cache(repro_root)
    )
    for hpo_set in sets:
        output_label = args.output_label or hpo_set
        triplet_path = _write_triplets(repro_root, output_label, triplets[triplets["hpo_set"] == output_label], baseline)
        current = triplets[triplets["hpo_set"] == output_label].copy()
        source_cohort = ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"
        output_dir = args.output_directory.resolve() if args.output_directory else args.output_root.resolve() / output_label / "paper" / "complete"
        for path in _render(all_data[output_label], output_label, output_dir, current, baseline, sources[output_label] + [str(source_cohort), str(triplet_path)], args.r2_estimator):
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
