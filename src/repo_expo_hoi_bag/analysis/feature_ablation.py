"""Pure data contracts for leave-one-exposome-out ablation analyses."""
from __future__ import annotations

from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.analysis.contracts import ResultContractError
from repo_expo_hoi_bag.analysis.selection import select_top_per_rung


ARM_ORDER = ("o_min", "o_max")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ResultContractError(f"{label} missing required columns: {sorted(missing)}")


def select_ranked_top_models(
    metrics: pd.DataFrame,
    *,
    bags: Iterable[str],
    rungs: Iterable[str],
    top_k: int,
    order_max: int,
) -> pd.DataFrame:
    """Select and rank top models independently within BAG, rung, and arm."""
    if top_k < 1:
        raise ResultContractError("top_k must be >= 1")
    _require_columns(
        metrics,
        {"bag_target", "rung_id", "objective", "candidate_id", "predictors_identity", "full_r2", "order"},
        label="Canonical metrics",
    )
    wanted_bags = tuple(str(bag) for bag in bags)
    wanted_rungs = tuple(str(rung) for rung in rungs)
    work = metrics.copy()
    work["full_r2"] = pd.to_numeric(work["full_r2"], errors="coerce")
    work["order"] = pd.to_numeric(work["order"], errors="coerce")
    work = work.loc[
        work["bag_target"].astype(str).isin(wanted_bags)
        & work["rung_id"].astype(str).isin(wanted_rungs)
        & work["objective"].astype(str).isin(ARM_ORDER)
        & work["order"].le(int(order_max))
    ].copy()
    if work["full_r2"].isna().any():
        raise ResultContractError("Canonical metrics contain non-finite full_r2 values in the selected pool")

    selected: list[pd.DataFrame] = []
    for bag in wanted_bags:
        for rung in wanted_rungs:
            for arm in ARM_ORDER:
                subset = work.loc[
                    (work["bag_target"].astype(str) == bag)
                    & (work["rung_id"].astype(str) == rung)
                    & (work["objective"].astype(str) == arm)
                ].copy()
                # Pre-sort candidate ID so the shared stable top-k helper also
                # defines deterministic ordering for exact R² ties.
                subset = subset.sort_values("candidate_id", kind="stable")
                subset = select_top_per_rung(
                    subset, objective=arm, top_k=top_k, score_column="full_r2"
                )
                if len(subset) != top_k:
                    raise ResultContractError(
                        f"Expected {top_k} models for bag={bag}, rung={rung}, arm={arm}; found {len(subset)}"
                    )
                if subset["predictors_identity"].astype(str).duplicated().any():
                    raise ResultContractError(
                        f"Top models contain duplicate predictor identities for bag={bag}, rung={rung}, arm={arm}"
                    )
                subset["rank_within_arm"] = np.arange(1, top_k + 1, dtype=int)
                selected.append(subset)
    return pd.concat(selected, ignore_index=True)


def build_ablation_requests(parents: pd.DataFrame) -> pd.DataFrame:
    """Return one request per parent candidate and removed exposure."""
    _require_columns(
        parents,
        {"bag_target", "rung_id", "objective", "candidate_id", "predictors_identity", "full_r2", "rank_within_arm"},
        label="Parent selection",
    )
    rows: list[dict[str, object]] = []
    for parent in parents.itertuples(index=False):
        features = [feature for feature in str(parent.predictors_identity).split("|") if feature]
        if len(features) < 2:
            raise ResultContractError(
                f"Parent candidate {parent.candidate_id!r} has fewer than two predictors and cannot be ablated"
            )
        if len(features) != len(set(features)):
            raise ResultContractError(f"Parent candidate {parent.candidate_id!r} has duplicate predictors")
        for feature in features:
            identity = "|".join(name for name in features if name != feature)
            rows.append(
                {
                    "bag": str(parent.bag_target),
                    "rung_id": str(parent.rung_id),
                    "arm": str(parent.objective),
                    "rank_within_arm": int(parent.rank_within_arm),
                    "parent_candidate_id": str(parent.candidate_id),
                    "parent_predictors_identity": str(parent.predictors_identity),
                    "parent_full_r2": float(parent.full_r2),
                    "parent_order": len(features),
                    "removed_feature": feature,
                    "ablation_predictors_identity": identity,
                    "ablation_order": len(features) - 1,
                }
            )
    return pd.DataFrame(rows)


def add_relative_r2_loss_percentage(results: pd.DataFrame) -> pd.DataFrame:
    """Add signed LOFO loss as a percentage of the complete parent-model R²."""
    _require_columns(results, {"parent_full_r2", "r2_loss"}, label="Ablation results")
    out = results.copy()
    out["parent_full_r2"] = pd.to_numeric(out["parent_full_r2"], errors="coerce")
    out["r2_loss"] = pd.to_numeric(out["r2_loss"], errors="coerce")
    invalid = (~np.isfinite(out["parent_full_r2"])) | (~np.isfinite(out["r2_loss"])) | (out["parent_full_r2"] <= 0)
    if invalid.any():
        raise ResultContractError(
            "Relative ablation loss requires finite r2_loss and strictly positive parent_full_r2 values"
        )
    out["r2_loss_percentage"] = 100.0 * out["r2_loss"] / out["parent_full_r2"]
    return out


def ablation_candidate_table(requests: pd.DataFrame) -> pd.DataFrame:
    """Build one deterministic evaluator candidate per unique ablated identity."""
    _require_columns(requests, {"ablation_predictors_identity", "ablation_order"}, label="Ablation requests")
    unique = requests.drop_duplicates("ablation_predictors_identity", keep="first").copy()
    unique["candidate_id"] = unique["ablation_predictors_identity"].map(
        lambda value: f"feature_ablation_{sha256(str(value).encode('utf-8')).hexdigest()[:20]}"
    )
    unique["feature_id"] = unique["candidate_id"]
    unique["objective"] = "feature_ablation"
    unique["order"] = unique["ablation_order"].astype(int)
    unique["rank"] = np.arange(1, len(unique) + 1, dtype=int)
    unique["score"] = np.nan
    unique["nplet_vars"] = unique["ablation_predictors_identity"].str.split("|")
    unique["predictors_identity"] = unique["ablation_predictors_identity"]
    unique["predictors_identity_n"] = unique["ablation_order"].astype(int)
    unique["candidate_family"] = "feature_ablation"
    unique["source_label"] = "leave_one_exposure_out"
    return unique[
        [
            "candidate_id",
            "feature_id",
            "objective",
            "order",
            "rank",
            "score",
            "nplet_vars",
            "predictors_identity",
            "predictors_identity_n",
            "candidate_family",
            "source_label",
        ]
    ].reset_index(drop=True)


def attach_ablation_metrics(requests: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach ablated R² and provenance, rejecting incomplete evaluations."""
    _require_columns(
        requests, {"ablation_predictors_identity", "parent_full_r2"}, label="Ablation requests"
    )
    _require_columns(metrics, {"ablation_predictors_identity", "ablated_full_r2", "provenance"}, label="Ablation metrics")
    join_columns = ["ablation_predictors_identity"]
    for column in ("bag", "rung_id"):
        if column in requests.columns and column in metrics.columns:
            join_columns.insert(0, column)
    metric_rows = metrics.drop_duplicates(join_columns, keep="first").copy()
    metric_rows["ablated_full_r2"] = pd.to_numeric(metric_rows["ablated_full_r2"], errors="coerce")
    out = requests.merge(metric_rows, on=join_columns, how="left", validate="many_to_one")
    if out["ablated_full_r2"].isna().any():
        missing = out.loc[out["ablated_full_r2"].isna(), "ablation_predictors_identity"].nunique()
        raise ResultContractError(f"Ablation evaluation is incomplete: {missing} predictor identities have no finite R²")
    out["r2_loss"] = pd.to_numeric(out["parent_full_r2"], errors="coerce") - out["ablated_full_r2"]
    if out["r2_loss"].isna().any():
        raise ResultContractError("Ablation results contain non-finite R² losses")
    return out


def parent_ablation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize leave-one-out loss once per selected parent model."""
    keys = [
        "bag",
        "rung_id",
        "arm",
        "rank_within_arm",
        "parent_candidate_id",
        "parent_predictors_identity",
        "parent_full_r2",
        "parent_order",
    ]
    _require_columns(results, {*keys, "r2_loss"}, label="Ablation results")
    return (
        results.groupby(keys, as_index=False, dropna=False)["r2_loss"]
        .agg(n_ablations="size", mean_r2_loss="mean", median_r2_loss="median", min_r2_loss="min", max_r2_loss="max")
        .sort_values(["bag", "rung_id", "arm", "rank_within_arm"], kind="stable")
        .reset_index(drop=True)
    )


def cloud_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compute the IQR and median marks drawn for each rank-specific cloud."""
    keys = ["bag", "rung_id", "arm", "rank_within_arm", "parent_candidate_id"]
    _require_columns(results, {*keys, "r2_loss"}, label="Ablation results")
    grouped = results.groupby(keys, as_index=False, dropna=False)["r2_loss"]
    return grouped.agg(
        n_ablations="size",
        q25_r2_loss=lambda values: float(np.quantile(values, 0.25)),
        median_r2_loss="median",
        q75_r2_loss=lambda values: float(np.quantile(values, 0.75)),
    )


def arm_contrasts(
    summaries: pd.DataFrame,
    *,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Compute equal-parent arm contrasts with bootstrap CIs and permutation p-values."""
    if bootstrap_draws < 1 or permutation_draws < 1:
        raise ResultContractError("bootstrap_draws and permutation_draws must be >= 1")
    _require_columns(summaries, {"bag", "rung_id", "arm", "mean_r2_loss"}, label="Parent summaries")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (bag, rung), group in summaries.groupby(["bag", "rung_id"], sort=True):
        syn = pd.to_numeric(group.loc[group["arm"] == "o_min", "mean_r2_loss"], errors="coerce").dropna().to_numpy()
        red = pd.to_numeric(group.loc[group["arm"] == "o_max", "mean_r2_loss"], errors="coerce").dropna().to_numpy()
        if len(syn) == 0 or len(red) == 0:
            raise ResultContractError(f"Missing parent summaries for bag={bag}, rung={rung}")
        difference = float(syn.mean() - red.mean())
        boot = (
            rng.choice(syn, size=(bootstrap_draws, len(syn)), replace=True).mean(axis=1)
            - rng.choice(red, size=(bootstrap_draws, len(red)), replace=True).mean(axis=1)
        )
        combined = np.concatenate([syn, red])
        exceedances = 0
        for _ in range(permutation_draws):
            permuted = rng.permutation(combined)
            candidate = permuted[: len(syn)].mean() - permuted[len(syn) :].mean()
            exceedances += int(abs(candidate) >= abs(difference))
        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "n_synergy_parents": len(syn),
                "n_redundancy_parents": len(red),
                "mean_r2_loss_synergy": float(syn.mean()),
                "mean_r2_loss_redundancy": float(red.mean()),
                "mean_difference_synergy_minus_redundancy": difference,
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "permutation_p_raw": float((exceedances + 1) / (permutation_draws + 1)),
                "bootstrap_draws": bootstrap_draws,
                "permutation_draws": permutation_draws,
                "seed": seed,
            }
        )
    out = pd.DataFrame(rows)
    out["permutation_p_holm"] = holm_adjust(out["permutation_p_raw"].to_numpy())
    return out


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ResultContractError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted
