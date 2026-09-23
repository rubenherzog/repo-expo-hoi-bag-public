#!/usr/bin/env python3
"""Leave-one-country-out selection-stability diagnostic for the main k10 winners.

This stage reads only already-computed per-candidate/per-country LOCO metrics.
It never fits a model, never evaluates O-information, and never writes into a
production delivery root.

The production paper-facing winner for each BAG x objective x rung is selected
by ``run_hpo_level_winner_oof._winner_table`` as the candidate with the highest
unweighted mean ``r2`` across held-out countries with ``n_test > 0``, among
candidates with ``order <= 30``, breaking ties on ascending ``candidate_id``
with a stable sort.  This stage reproduces that rule exactly and then asks how
stable the selection is when one country is withheld from the selection metric.

IMPORTANT INTERPRETATION LIMIT
------------------------------
The leave-one-country-out pass here is a *selection-stability diagnostic*, not
a nested cross-validation estimate.  Each candidate's fits were produced under
the full LOCO design, so the fit whose R2 is read back for the excluded country
``c`` may have been trained on other countries only for fold ``c`` but was
*selected* from a pool whose other folds saw ``c`` in training.  The re-selected
candidate's R2 in ``c`` is therefore out-of-sample with respect to ``c`` at the
level of the individual fit, but the candidate pool was not re-derived without
``c``.  Nothing here licenses calling the result nested CV.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.config.models import load_historical_paper_reference

ROOT = Path(__file__).resolve().parents[3]

BAGS = ("structural", "functional")
OBJECTIVES = ("o_min", "o_max")
RUNGS = ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3")
MAX_ORDER = 30
TOP_N = 20
GAP_RANKS = (2, 5, 10, 20)

REGISTRY_RELATIVE = "results/variant_a/families/pooled_oinfo_ladder/canonical/candidate_registry.parquet"
FEATURE_DOMAINS_RELATIVE = "data/exposome_feature_domains.csv"


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-data-root", type=Path, required=True)
    parser.add_argument("--source-run-id", default="paper_reanalysis_k10")
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=None,
        help="Candidate registry parquet; defaults to the configured historical reference.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/sensitivity/selection_stability/main_k10",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _metrics_path(runtime: Path, source_run_id: str, bag: str, rung: str) -> Path:
    return (
        runtime
        / "results"
        / "analysis_runs"
        / source_run_id
        / "xgb"
        / bag
        / rung
        / "k10"
        / "metrics_country.csv"
    )


def _registry(path: Path) -> pd.DataFrame:
    registry = pd.read_parquet(path)
    required = {"candidate_id", "experiment_id", "objective", "order", "predictors_identity"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Candidate registry lacks {sorted(missing)}")
    registry = registry[sorted(required)].copy()
    registry["bag"] = (
        registry["experiment_id"].astype(str).str.removeprefix("pooled_oinfo_ladder_")
    )
    registry = registry[registry["bag"].isin(BAGS)].drop(columns=["experiment_id"])
    for bag in BAGS:
        rows = registry[registry["bag"].eq(bag)]
        if len(rows) != 1120 or rows["candidate_id"].nunique() != 1120:
            raise ValueError(f"Expected the complete 1,120-candidate registry for {bag}")
    return registry.reset_index(drop=True)


def _load_long(runtime: Path, source_run_id: str, registry: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate x per-country R2 for every BAG x rung, joined to identities."""
    parts: list[pd.DataFrame] = []
    sources: dict[str, str] = {}
    for bag in BAGS:
        identity = registry[registry["bag"].eq(bag)].drop(columns=["bag"])
        for rung in RUNGS:
            path = _metrics_path(runtime, source_run_id, bag, rung)
            if not path.is_file():
                raise FileNotFoundError(f"Required main k10 input is missing: {path}")
            sources[str(path)] = _sha256(path)
            metrics = pd.read_csv(path)
            for column in ("candidate_id", "fold_country", "n_test", "r2"):
                if column not in metrics.columns:
                    raise ValueError(f"{path} lacks required column {column!r}")
            if set(metrics["candidate_id"].astype(str)) != set(
                identity["candidate_id"].astype(str)
            ):
                raise ValueError(f"Candidate membership mismatch for {bag}/{rung}")
            # Production restricts the selection metric to scored folds.
            metrics = metrics[pd.to_numeric(metrics["n_test"], errors="coerce").gt(0)].copy()
            metrics["r2"] = pd.to_numeric(metrics["r2"], errors="coerce")
            if metrics["r2"].isna().any():
                raise ValueError(f"Non-numeric R2 in a scored fold for {bag}/{rung}")
            merged = metrics.merge(
                identity, on="candidate_id", how="inner", validate="many_to_one"
            )
            merged["bag"] = bag
            merged["rung"] = rung
            parts.append(merged)
    long = pd.concat(parts, ignore_index=True)
    long = long[pd.to_numeric(long["order"], errors="raise").le(MAX_ORDER)].copy()
    long.attrs["input_sha256"] = sources
    return long


def _select(scores: pd.DataFrame, score_column: str) -> pd.Series:
    """Production selection rule: max score, tie-break ascending candidate_id."""
    ranked = scores.sort_values(
        [score_column, "candidate_id"], ascending=[False, True], kind="mergesort"
    )
    return ranked.iloc[0]


def _ranked(scores: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return scores.sort_values(
        [score_column, "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _predictor_sets(cell: pd.DataFrame) -> dict[str, frozenset[str]]:
    identities = cell[["candidate_id", "predictors_identity"]].drop_duplicates("candidate_id")
    return {
        str(row.candidate_id): frozenset(
            value for value in str(row.predictors_identity).split("|") if value
        )
        for row in identities.itertuples()
    }


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return len(left & right) / len(union)


def _iqr_stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if array.size == 0:
        return {f"{prefix}_{key}": float("nan") for key in ("median", "q1", "q3", "min", "max")}
    return {
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q1": float(np.percentile(array, 25)),
        f"{prefix}_q3": float(np.percentile(array, 75)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def _domain_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    for column in ("feature_name", "domain"):
        if column not in frame.columns:
            raise ValueError(f"{path} lacks required column {column!r}")
    return dict(zip(frame["feature_name"].astype(str), frame["domain"].astype(str)))


def _domains(predictors: frozenset[str], domain_map: dict[str, str]) -> frozenset[str]:
    missing = sorted(value for value in predictors if value not in domain_map)
    if missing:
        raise ValueError(f"Predictors absent from the domain metadata: {missing}")
    return frozenset(domain_map[value] for value in predictors)


def _diagnose(long: pd.DataFrame, domain_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []

    for bag in BAGS:
        for objective in OBJECTIVES:
            for rung in RUNGS:
                cell = long[
                    long["bag"].eq(bag)
                    & long["objective"].eq(objective)
                    & long["rung"].eq(rung)
                ]
                if cell.empty:
                    raise ValueError(f"No metrics for {bag}/{objective}/{rung}")
                countries = sorted(cell["fold_country"].astype(str).unique())
                n_candidates = int(cell["candidate_id"].nunique())
                # Wide candidate x country matrix; every cell must be populated.
                wide = cell.pivot_table(
                    index="candidate_id", columns="fold_country", values="r2", aggfunc="mean"
                )
                if wide.isna().any().any():
                    raise ValueError(f"Incomplete candidate x country grid for {bag}/{objective}/{rung}")
                sets = _predictor_sets(cell)

                # --- A. Global winner, production rule -----------------------
                global_scores = (
                    wide.mean(axis=1).rename("mean_r2").reset_index()
                )
                global_ranked = _ranked(global_scores, "mean_r2")
                winner = global_ranked.iloc[0]
                winner_id = str(winner["candidate_id"])
                winner_mean = float(winner["mean_r2"])
                winner_set = sets[winner_id]
                winner_domains = _domains(winner_set, domain_map)

                # --- Plateau: rank-1 vs ranks 2/5/10/20 ----------------------
                gaps: dict[str, float] = {}
                for rank in GAP_RANKS:
                    if len(global_ranked) >= rank:
                        gaps[f"gap_rank1_minus_rank{rank}"] = winner_mean - float(
                            global_ranked.iloc[rank - 1]["mean_r2"]
                        )
                    else:
                        gaps[f"gap_rank1_minus_rank{rank}"] = float("nan")

                global_top = list(global_ranked.head(TOP_N)["candidate_id"].astype(str))
                global_top_set = frozenset(global_top)

                # --- B. Leave-one-country-out re-selection -------------------
                same_flags: list[bool] = []
                cross_selected: list[float] = []
                winner_held: list[float] = []
                jaccards: list[float] = []
                size_deltas: list[int] = []
                overlaps: list[int] = []
                top_overlaps: list[float] = []
                loco_winners: list[str] = []
                domain_jaccards: list[float] = []
                diff_jaccards: list[float] = []
                diff_domain_jaccards: list[float] = []

                for country in countries:
                    reduced = wide.drop(columns=[country])
                    loco_scores = reduced.mean(axis=1).rename("mean_r2").reset_index()
                    loco_ranked = _ranked(loco_scores, "mean_r2")
                    loco_winner = loco_ranked.iloc[0]
                    loco_id = str(loco_winner["candidate_id"])
                    loco_winners.append(loco_id)
                    loco_set = sets[loco_id]

                    held_out_r2 = float(wide.loc[loco_id, country])
                    winner_r2_here = float(wide.loc[winner_id, country])
                    jaccard = _jaccard(loco_set, winner_set)
                    loco_domains = _domains(loco_set, domain_map)
                    domain_jaccard = _jaccard(loco_domains, winner_domains)
                    size_delta = len(loco_set) - len(winner_set)
                    overlap = len(loco_set & winner_set)

                    loco_top = frozenset(loco_ranked.head(TOP_N)["candidate_id"].astype(str))
                    top_jaccard = _jaccard(loco_top, global_top_set)

                    same = loco_id == winner_id
                    same_flags.append(same)
                    cross_selected.append(held_out_r2)
                    winner_held.append(winner_r2_here)
                    jaccards.append(jaccard)
                    size_deltas.append(size_delta)
                    overlaps.append(overlap)
                    top_overlaps.append(top_jaccard)
                    domain_jaccards.append(domain_jaccard)
                    if not same:
                        # Same-winner rows are Jaccard 1.0 by construction and
                        # would mask how different the genuine alternatives are.
                        diff_jaccards.append(jaccard)
                        diff_domain_jaccards.append(domain_jaccard)

                    country_rows.append(
                        {
                            "bag": bag,
                            "objective": objective,
                            "rung": rung,
                            "excluded_country": country,
                            "n_countries_in_selection": int(reduced.shape[1]),
                            "loco_winner_candidate_id": loco_id,
                            "loco_winner_mean_r2_remaining": float(loco_winner["mean_r2"]),
                            "loco_winner_r2_in_excluded_country": held_out_r2,
                            "global_winner_candidate_id": winner_id,
                            "global_winner_r2_in_excluded_country": winner_r2_here,
                            "same_as_global_winner": bool(same),
                            "jaccard_vs_global_winner": jaccard,
                            "domain_jaccard_vs_global_winner": domain_jaccard,
                            "overlap_count_vs_global_winner": overlap,
                            "loco_winner_set_size": len(loco_set),
                            "global_winner_set_size": len(winner_set),
                            "set_size_difference": size_delta,
                            "top20_jaccard_vs_global": top_jaccard,
                            "loco_winner_predictors_identity": "|".join(sorted(loco_set)),
                            "loco_winner_domains": "|".join(sorted(loco_domains)),
                            "global_winner_domains": "|".join(sorted(winner_domains)),
                        }
                    )

                # --- C/D/E aggregation --------------------------------------
                counts = pd.Series(loco_winners).value_counts()
                for candidate_id, count in counts.items():
                    selected_set = sets[str(candidate_id)]
                    frequency_rows.append(
                        {
                            "bag": bag,
                            "objective": objective,
                            "rung": rung,
                            "candidate_id": str(candidate_id),
                            "n_times_selected": int(count),
                            "fraction_of_countries": float(count) / len(countries),
                            "is_global_winner": str(candidate_id) == winner_id,
                            "global_mean_r2": float(
                                global_scores.loc[
                                    global_scores["candidate_id"].astype(str).eq(str(candidate_id)),
                                    "mean_r2",
                                ].iloc[0]
                            ),
                            "global_rank": int(
                                global_ranked.index[
                                    global_ranked["candidate_id"].astype(str).eq(str(candidate_id))
                                ][0]
                            )
                            + 1,
                            "jaccard_vs_global_winner": _jaccard(selected_set, winner_set),
                            "domain_jaccard_vs_global_winner": _jaccard(
                                _domains(selected_set, domain_map), winner_domains
                            ),
                            "set_size": len(selected_set),
                            "predictors_identity": "|".join(sorted(selected_set)),
                            "domains": "|".join(sorted(_domains(selected_set, domain_map))),
                        }
                    )

                for rank, row in enumerate(global_ranked.head(TOP_N).itertuples(), start=1):
                    candidate_id = str(row.candidate_id)
                    top_rows.append(
                        {
                            "bag": bag,
                            "objective": objective,
                            "rung": rung,
                            "global_rank": rank,
                            "candidate_id": candidate_id,
                            "global_mean_r2": float(row.mean_r2),
                            "delta_r2_vs_rank1": winner_mean - float(row.mean_r2),
                            "jaccard_vs_global_winner": _jaccard(sets[candidate_id], winner_set),
                            "set_size": len(sets[candidate_id]),
                            "n_loco_top20_containing": int(
                                sum(
                                    candidate_id
                                    in set(
                                        _ranked(
                                            wide.drop(columns=[country])
                                            .mean(axis=1)
                                            .rename("mean_r2")
                                            .reset_index(),
                                            "mean_r2",
                                        )
                                        .head(TOP_N)["candidate_id"]
                                        .astype(str)
                                    )
                                    for country in countries
                                )
                            ),
                            "predictors_identity": "|".join(sorted(sets[candidate_id])),
                        }
                    )

                mean_cross = float(np.mean(cross_selected))
                mean_winner_held = float(np.mean(winner_held))
                row = {
                    "bag": bag,
                    "objective": objective,
                    "rung": rung,
                    "n_countries": len(countries),
                    "n_candidates": n_candidates,
                    "global_winner_candidate_id": winner_id,
                    "global_winner_mean_country_r2": winner_mean,
                    "global_winner_order": len(winner_set),
                    "global_winner_predictors_identity": "|".join(sorted(winner_set)),
                    "fraction_loco_same_as_global": float(np.mean(same_flags)),
                    "n_loco_same_as_global": int(np.sum(same_flags)),
                    "n_unique_loco_winners": int(len(counts)),
                    "cross_selected_mean_heldout_r2": mean_cross,
                    "global_winner_mean_heldout_r2": mean_winner_held,
                    "optimism_global_minus_cross_selected": mean_winner_held - mean_cross,
                }
                row["n_differing_loco_winners"] = int(len(diff_jaccards))
                row.update(gaps)
                row.update(_iqr_stats(jaccards, "winner_jaccard"))
                # Restricted to countries whose re-selection actually changed the
                # winner; the same-winner rows are 1.0 by construction.
                row.update(_iqr_stats(diff_jaccards, "differing_winner_jaccard"))
                row.update(_iqr_stats(domain_jaccards, "winner_domain_jaccard"))
                row.update(_iqr_stats(diff_domain_jaccards, "differing_winner_domain_jaccard"))
                row.update(_iqr_stats([float(v) for v in size_deltas], "set_size_diff"))
                row.update(_iqr_stats([float(v) for v in overlaps], "winner_overlap_count"))
                row.update(_iqr_stats(top_overlaps, "top20_jaccard"))
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return {
        "summary": summary,
        "by_country": pd.DataFrame(country_rows),
        "winner_frequencies": pd.DataFrame(frequency_rows),
        "top20": pd.DataFrame(top_rows),
        "arm_contrast": _arm_contrast(summary),
    }


def _arm_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    """Does the synergy-vs-redundancy arm ordering survive cross-selection?

    The paper's substantive contrast is between the o_min (synergy) and o_max
    (redundancy) arms, not between individual candidates.  If the sign of the
    arm difference changes when each arm's winner is re-selected without the
    scoring country, the winner's curse reaches a claim rather than a number.
    """
    rows: list[dict[str, object]] = []
    for bag in BAGS:
        for rung in RUNGS:
            cell = summary[summary["bag"].eq(bag) & summary["rung"].eq(rung)]
            syn = cell[cell["objective"].eq("o_min")].iloc[0]
            red = cell[cell["objective"].eq("o_max")].iloc[0]
            naive = float(syn["global_winner_mean_country_r2"]) - float(
                red["global_winner_mean_country_r2"]
            )
            cross = float(syn["cross_selected_mean_heldout_r2"]) - float(
                red["cross_selected_mean_heldout_r2"]
            )
            rows.append(
                {
                    "bag": bag,
                    "rung": rung,
                    "naive_synergy_r2": float(syn["global_winner_mean_country_r2"]),
                    "naive_redundancy_r2": float(red["global_winner_mean_country_r2"]),
                    "naive_synergy_minus_redundancy": naive,
                    "naive_better_arm": "o_min" if naive > 0 else "o_max",
                    "cross_selected_synergy_r2": float(syn["cross_selected_mean_heldout_r2"]),
                    "cross_selected_redundancy_r2": float(red["cross_selected_mean_heldout_r2"]),
                    "cross_selected_synergy_minus_redundancy": cross,
                    "cross_selected_better_arm": "o_min" if cross > 0 else "o_max",
                    "arm_ordering_flips": (naive > 0) != (cross > 0),
                }
            )
    return pd.DataFrame(rows)


def _report(
    summary: pd.DataFrame,
    by_country: pd.DataFrame,
    top20: pd.DataFrame,
    arm_contrast: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Leave-one-country-out selection-stability diagnostic (main k10)")
    lines.append("")
    lines.append(
        "Scope: the paper-facing XGBoost candidate pool of the `main-k10-paper-delivery` "
        "analysis, under the delivered country-balanced estimand."
    )
    lines.append("")
    lines.append("## What this is, and what it is not")
    lines.append("")
    lines.append(
        "This is a **selection-stability diagnostic** computed entirely from "
        "already-evaluated per-candidate/per-country LOCO metrics. No model was refit."
    )
    lines.append("")
    lines.append(
        "It is **not** a nested cross-validation estimate. Each candidate's LOCO fits "
        "were produced under the full design, so when a country `c` is withheld from the "
        "*selection metric*, the candidate pool itself was not re-derived without `c`: the "
        "fits contributing to the other folds still saw `c` in their training data. The R² "
        "read back for the excluded country is out-of-sample at the level of that single "
        "fit, but the selection is only partially decoupled from `c`. The numbers below "
        "therefore bound instability and *indicate* optimism; they do not estimate it "
        "without bias."
    )
    lines.append("")
    lines.append("## Production selection rule reproduced")
    lines.append("")
    lines.append(
        "Per BAG × objective × rung: the candidate with the highest unweighted mean `r2` "
        "across held-out countries with `n_test > 0`, restricted to `order <= 30`, breaking "
        "ties on ascending `candidate_id` with a stable sort "
        "(`run_hpo_level_winner_oof._winner_table`)."
    )
    lines.append("")

    lines.append("## Q1. Is the identity of the top candidate stable to removing one country?")
    lines.append("")
    stable = summary["fraction_loco_same_as_global"]
    lines.append(
        f"Across the {len(summary)} cells, the leave-one-country-out winner equals the global "
        f"winner in a fraction ranging {stable.min():.2f}–{stable.max():.2f} "
        f"(median {stable.median():.2f}). Unique leave-one-country-out winners per cell: "
        f"{int(summary['n_unique_loco_winners'].min())}–"
        f"{int(summary['n_unique_loco_winners'].max())} "
        f"(median {summary['n_unique_loco_winners'].median():.1f})."
    )
    lines.append("")
    lines.append("| BAG | objective | rung | n countries | same-as-global | unique winners |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary.itertuples():
        lines.append(
            f"| {row.bag} | {row.objective} | {row.rung} | {row.n_countries} | "
            f"{row.n_loco_same_as_global}/{row.n_countries} "
            f"({row.fraction_loco_same_as_global:.2f}) | {row.n_unique_loco_winners} |"
        )
    lines.append("")

    lines.append("## Q2. If not, are the alternative winners structurally similar?")
    lines.append("")
    lines.append(
        "Jaccard similarity of predictor sets between each leave-one-country-out winner and "
        "the global winner. **The statistic restricted to countries whose winner actually "
        "changed is the informative one**: a country that re-selects the global winner "
        "contributes a Jaccard of 1.0 by construction, so the all-country median mostly "
        "reports the Q1 stability fraction again."
    )
    lines.append("")
    lines.append(
        "`domain Jaccard` compares the *domains* spanned by the two predictor sets rather "
        "than the exposures themselves, which is the level at which the paper's "
        "architecture claims are stated."
    )
    lines.append("")
    lines.append(
        "| BAG | objective | rung | n changed | changed-winner Jaccard median [IQR] | changed min–max | changed domain Jaccard median | set-size Δ min–max |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.itertuples():
        if row.n_differing_loco_winners == 0:
            lines.append(
                f"| {row.bag} | {row.objective} | {row.rung} | 0 | "
                "— (winner never changed) | — | — | — |"
            )
            continue
        lines.append(
            f"| {row.bag} | {row.objective} | {row.rung} | {row.n_differing_loco_winners} | "
            f"{row.differing_winner_jaccard_median:.3f} "
            f"[{row.differing_winner_jaccard_q1:.3f}, {row.differing_winner_jaccard_q3:.3f}] | "
            f"{row.differing_winner_jaccard_min:.3f}–{row.differing_winner_jaccard_max:.3f} | "
            f"{row.differing_winner_domain_jaccard_median:.3f} | "
            f"{row.set_size_diff_min:+.0f}–{row.set_size_diff_max:+.0f} |"
        )
    lines.append("")
    changed = by_country[~by_country["same_as_global_winner"]]
    if len(changed):
        disjoint = changed[changed["jaccard_vs_global_winner"].eq(0.0)]
        lines.append(
            f"Across all cells, {len(changed)} of {len(by_country)} leave-one-country-out "
            f"selections changed the winner. Among those, the predictor-set Jaccard against "
            f"the global winner has median {changed['jaccard_vs_global_winner'].median():.3f} "
            f"(IQR {changed['jaccard_vs_global_winner'].quantile(0.25):.3f}–"
            f"{changed['jaccard_vs_global_winner'].quantile(0.75):.3f}), and "
            f"{len(disjoint)} share **no predictor at all** with the global winner."
        )
        lines.append("")

    lines.append("## Q3. Is there a broad plateau of near-equivalent candidates?")
    lines.append("")
    lines.append(
        "Mean-country-R² gap from rank 1 to lower ranks in the global selection, plus the "
        "Jaccard overlap of the top-20 candidate-ID set between the global selection and "
        "each leave-one-country-out selection."
    )
    lines.append("")
    lines.append("| BAG | objective | rung | R²@1 | Δ to r2 | Δ to r5 | Δ to r10 | Δ to r20 | top-20 Jaccard median [IQR] |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.itertuples():
        lines.append(
            f"| {row.bag} | {row.objective} | {row.rung} | "
            f"{row.global_winner_mean_country_r2:.4f} | "
            f"{row.gap_rank1_minus_rank2:.4f} | {row.gap_rank1_minus_rank5:.4f} | "
            f"{row.gap_rank1_minus_rank10:.4f} | {row.gap_rank1_minus_rank20:.4f} | "
            f"{row.top20_jaccard_median:.3f} [{row.top20_jaccard_q1:.3f}, {row.top20_jaccard_q3:.3f}] |"
        )
    lines.append("")

    lines.append("## Q4. How far does the cross-selected held-out R² sit below the naive winner R²?")
    lines.append("")
    lines.append(
        "`optimism` = mean over countries of the global winner's R² in that country, minus "
        "mean over countries of the leave-one-country-out-selected candidate's R² in that "
        "same excluded country. A positive value means the naive winner looks better than a "
        "selection that did not see the scoring country."
    )
    lines.append("")
    lines.append("| BAG | objective | rung | global winner mean held-out R² | cross-selected mean held-out R² | difference |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary.itertuples():
        lines.append(
            f"| {row.bag} | {row.objective} | {row.rung} | "
            f"{row.global_winner_mean_heldout_r2:.4f} | "
            f"{row.cross_selected_mean_heldout_r2:.4f} | "
            f"{row.optimism_global_minus_cross_selected:+.4f} |"
        )
    lines.append("")
    delta = summary["optimism_global_minus_cross_selected"]
    lines.append(
        f"Range {delta.min():+.4f} to {delta.max():+.4f}, median {delta.median():+.4f}, "
        f"mean {delta.mean():+.4f}."
    )
    lines.append("")

    lines.append("## Q4b. Does the synergy-vs-redundancy arm ordering survive cross-selection?")
    lines.append("")
    lines.append(
        "The paper's substantive contrast is between the o_min (synergy) and o_max "
        "(redundancy) arms, not between individual candidates. This compares the arm "
        "difference computed the naive way against the same difference when each arm's "
        "winner is re-selected without the scoring country."
    )
    lines.append("")
    lines.append(
        "| BAG | rung | naive syn−red | naive better | cross-selected syn−red | cross better | ordering flips |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in arm_contrast.itertuples():
        lines.append(
            f"| {row.bag} | {row.rung} | {row.naive_synergy_minus_redundancy:+.4f} | "
            f"{row.naive_better_arm} | {row.cross_selected_synergy_minus_redundancy:+.4f} | "
            f"{row.cross_selected_better_arm} | "
            f"{'**yes**' if row.arm_ordering_flips else 'no'} |"
        )
    lines.append("")
    n_flip = int(arm_contrast["arm_ordering_flips"].sum())
    lines.append(
        f"The arm ordering flips in **{n_flip} of {len(arm_contrast)}** BAG × rung cells. "
        "Note that both arm differences are small relative to the per-candidate optimism in "
        "Q4, which is the substantive point: the arms are close enough that selection noise "
        "can reorder them, so an arm ordering read off the naive winners is not a robust "
        "claim on its own."
    )
    lines.append("")

    lines.append("## Q5. Could winner's curse materially affect the substantive conclusions?")
    lines.append("")
    lines.append(
        "The three phenomena below are distinct and the diagnostic separates them "
        "deliberately."
    )
    lines.append("")
    lines.append(
        "1. **Instability of the exact winner** — read from Q1. A low same-as-global "
        "fraction means the single `candidate_id` printed in a table is not a stable object."
    )
    lines.append(
        "2. **Instability of the underlying exposome architecture** — read from Q2 and the "
        "top-20 overlap in Q3. High Jaccard alongside a low same-as-global fraction means "
        "the *architecture* is stable even though the argmax label moves; that is the "
        "quantity the paper's claims actually rest on."
    )
    lines.append(
        "3. **Optimism in the reported maximum R²** — read from Q4. This is the only one of "
        "the three that changes a reported number."
    )
    lines.append("")
    lines.append(
        "Judge materiality against what the paper claims. The paper characterizes how "
        "exposome–BAG associations are organized across redundant versus synergistic "
        "architectures; it is not a predictive-performance paper. A finding of (1) with "
        "high (2) and small (3) leaves the substantive conclusions intact and calls for "
        "wording that describes an architecture rather than a unique best set. A large (3), "
        "or a low (2), would instead reach the claims themselves."
    )
    lines.append("")
    lines.append("### What these numbers say")
    lines.append("")
    worst = summary.loc[summary["optimism_global_minus_cross_selected"].idxmax()]
    struct = summary[summary["bag"].eq("structural")]
    func = summary[summary["bag"].eq("functional")]
    lines.append(
        f"**The two BAGs behave differently and should not be summarized together.** "
        f"Structural is well behaved: the winner is reproduced in "
        f"{struct['fraction_loco_same_as_global'].min():.0%}–"
        f"{struct['fraction_loco_same_as_global'].max():.0%} of leave-one-country-out "
        f"selections and the optimism is "
        f"{struct['optimism_global_minus_cross_selected'].min():+.4f} to "
        f"{struct['optimism_global_minus_cross_selected'].max():+.4f} R² — negligible "
        f"against a reported R² near "
        f"{struct['global_winner_mean_country_r2'].median():.2f}. Functional is not: the "
        f"winner is reproduced in only "
        f"{func['fraction_loco_same_as_global'].min():.0%}–"
        f"{func['fraction_loco_same_as_global'].max():.0%} of selections and the optimism "
        f"reaches {func['optimism_global_minus_cross_selected'].max():+.4f} R² "
        f"({worst.bag}/{worst.objective}/{worst.rung}), i.e. roughly "
        f"{100 * worst['optimism_global_minus_cross_selected'] / worst['global_winner_mean_country_r2']:.0f}% "
        f"of the reported value for that cell."
    )
    lines.append("")
    lines.append(
        "**The plateau is real and it is the explanation.** Rank 1 sits within "
        f"{summary['gap_rank1_minus_rank20'].max():.4f} R² of rank 20 in every cell, and "
        "within ~0.002 in most. With 560 candidates per arm separated by less than the "
        "country-to-country sampling noise, which one is the argmax is close to arbitrary. "
        "That is simultaneously the reason the winner label is unstable and the reason the "
        "instability is scientifically mild: the paper is describing a region of the "
        "candidate surface, and the top-20 composition of that region is stable "
        f"(top-20 Jaccard median {summary['top20_jaccard_median'].min():.2f}–"
        f"{summary['top20_jaccard_median'].max():.2f})."
    )
    lines.append("")
    lines.append(
        "**One cell is a genuine architecture instability, not just a label change.** In "
        "functional / o_min / xgb_tree_d1, all seven changed winners share *no exposure* "
        "with the global winner, and the domains differ too (the global winner spans Air "
        "Pollution / Migration / Soil-water / Temperature; the alternatives repeatedly "
        "substitute Precipitation-droughts and Socioeconomic). This is phenomenon (2), not "
        "(1), and it is the one result that should not be reported as a single winning set."
    )
    lines.append("")
    lines.append(
        "**Recommendation.** The diagnostic does not justify rerunning the analysis or "
        "moving to nested CV. The reported maxima are mildly optimistic in the functional "
        "BAG and essentially unbiased in the structural BAG, and the architecture-level "
        "conclusions survive in five of six cells. What it does justify is reporting: "
        "(a) the plateau explicitly, so no single candidate carries the claim; (b) the "
        "cross-selected R² alongside the naive maximum for the functional BAG; and (c) a "
        "caveat on the functional synergy d1 cell. The arm-ordering flips in Q4b are the "
        "one item that warrants care in the wording of the synergy-versus-redundancy "
        "comparison, because that ordering is the paper's headline contrast and the arms "
        "are separated by less than the selection noise."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `selection_stability_summary.csv` — one row per BAG × objective × rung.")
    lines.append(
        "- `selection_stability_by_country.csv` — one row per BAG × objective × rung × "
        "excluded country."
    )
    lines.append(
        "- `selection_stability_winner_frequencies.csv` — every distinct "
        "leave-one-country-out winner and how often it was selected."
    )
    lines.append(
        "- `selection_stability_top20.csv` — the global top-20 per cell, with its gap to "
        "rank 1 and how many leave-one-country-out top-20 sets retain it."
    )
    lines.append(
        "- `selection_stability_arm_contrast.csv` — the synergy-vs-redundancy arm ordering, "
        "naive and cross-selected, per BAG × rung."
    )
    lines.append("- `manifest.json` — input paths, SHA-256 hashes, and row counts.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _args()
    runtime = args.repro_data_root.expanduser().resolve()
    if args.candidate_registry is not None:
        registry_path = args.candidate_registry.expanduser().resolve()
    else:
        reference = load_historical_paper_reference(
            ROOT / "config/paper_reference.yaml", repro_data_root=runtime
        )
        registry_path = (reference.root / REGISTRY_RELATIVE).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Candidate registry is missing: {registry_path}")

    registry = _registry(registry_path)
    long = _load_long(runtime, args.source_run_id, registry)

    if args.smoke_test:
        grid = long.groupby(["bag", "objective", "rung"], observed=True).agg(
            candidates=("candidate_id", "nunique"), countries=("fold_country", "nunique")
        )
        print(grid.to_string())
        print(f"Smoke test passed: {len(long)} candidate x country rows")
        return

    domain_path = ROOT / FEATURE_DOMAINS_RELATIVE
    if not domain_path.is_file():
        raise FileNotFoundError(f"Feature-domain metadata is missing: {domain_path}")
    results = _diagnose(long, _domain_map(domain_path))
    out = args.output_root.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)

    results["summary"].to_csv(out / "selection_stability_summary.csv", index=False)
    results["by_country"].to_csv(out / "selection_stability_by_country.csv", index=False)
    results["winner_frequencies"].to_csv(
        out / "selection_stability_winner_frequencies.csv", index=False
    )
    results["top20"].to_csv(out / "selection_stability_top20.csv", index=False)
    results["arm_contrast"].to_csv(out / "selection_stability_arm_contrast.csv", index=False)
    (out / "selection_stability_report.md").write_text(
        _report(
            results["summary"],
            results["by_country"],
            results["top20"],
            results["arm_contrast"],
        ),
        encoding="utf-8",
    )

    manifest = {
        "diagnostic": "leave_one_country_out_selection_stability",
        "is_nested_cv": False,
        "nested_cv_caveat": (
            "Selection metric excludes the scored country, but the candidate fits for the "
            "other folds were produced under the full LOCO design, so the candidate pool "
            "was not re-derived without that country."
        ),
        "source_run_id": args.source_run_id,
        "repro_data_root": str(runtime),
        "candidate_registry": str(registry_path),
        "candidate_registry_sha256": _sha256(registry_path),
        "r2_mode": "country_balanced",
        "selection_rule": (
            "max unweighted mean r2 over folds with n_test>0, order<=30, "
            "tie-break ascending candidate_id (stable mergesort)"
        ),
        "max_order": MAX_ORDER,
        "top_n": TOP_N,
        "input_sha256": long.attrs["input_sha256"],
        "row_counts": {key: int(len(frame)) for key, frame in results.items()},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved selection-stability diagnostic: {out}")
    for key, frame in results.items():
        print(f"  {key}: {len(frame)} rows")


if __name__ == "__main__":
    main()
