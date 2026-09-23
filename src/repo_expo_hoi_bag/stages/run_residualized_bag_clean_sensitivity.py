#!/usr/bin/env python3
"""Clean residualized-BAG sensitivity (`residualized_bag_clean` namespace).

WHY THIS EXISTS
---------------
The delivered `residualized_bag` sensitivity answers the coauthor question
("does the exposome-BAG association survive removing covariate-related
variation?") with an estimand that cannot answer it, for two independent
reasons established by audit against the cached predictions:

1.  Methodological inconsistency. It removes age + sex + diagnosis from the
    TARGET, then rebuilds every candidate design on age + year + sex +
    diagnosis + exposome -- reintroducing as predictors exactly the variables
    just removed. Its `__baseline__` arm is a covariates-only model fitted
    against a target those covariates no longer explain, so the reference the
    figure draws is not an interpretable quantity. Year compounded this: it
    was in the downstream design but never in the residualizer.

2.  Metric pathology. The delivered figure aggregates with the unweighted mean
    of per-country R2. Because the residualizer is fit on training countries
    only, each held-out country inherits a non-zero residual mean by
    construction; per-country R2 charges every model for that constant. On the
    cached d3 predictions, removing only the squared fold mean offset moves
    structural from -0.0127 to +0.0350 and functional from -0.0986 to +0.0069,
    and the count of countries with R2 > 0 from 10/22 to 17/22 and 4/16 to
    10/16. The negativity was the offset, not a loss of association.

WHAT THIS STAGE DOES INSTEAD
----------------------------
Residualizer (per LOCO fold, training countries only, leakage-free):

    BAG ~ 1 + Age + Year + Sex + Diagnosis

Year enters as the fold's own year basis, whose spline knots are already
placed from training rows. This matches the covariate set the main analysis
controls for, so the target and the design agree.

Downstream:

    residual_BAG ~ exposome          (NO covariates reintroduced)

Comparator: an explicit `__null__` arm predicting the fold's TRAINING residual
mean. This is not the old covariate baseline, and not a literal zero -- a
zero-column OLS design would predict exactly 0.0, which is not the training
mean on a held-out fold.

Statistics: the primary inferential test is a PAIRED COUNTRY-LEVEL comparison
in prediction error (delta MSE vs the null arm; Wilcoxon signed-rank, matched
rank-biserial effect size, bootstrap CI). Prediction error is the right scale
here because the fold offset enters both arms identically and cancels in the
paired difference, which is exactly what R2 fails to do. Pooled subject-level
OOF R2 is reported as a DESCRIPTIVE landscape summary; the unweighted
mean-held-out-country R2 is retained as an AUDIT metric only, so the delivered
figure's numbers stay reproducible and auditable rather than silently dropped.

Landscape ordering is REPORTED PER RUNG and never asserted: it is not stable
across model levels. On the delivered cache the functional ordering reverses
with depth (d1 o_max > single > o_min, P(o_min>o_max)=0.28, p=5e-38; d3
o_min > o_max > single, P=0.65, p=3e-17), while structural is stable from d2.
Encoding a single expected ordering would hard-code a rung-specific result.

Candidate identities, O-information architecture, LOCO folds and frozen HPO
parameters are all unchanged and BAG-blind. Only the target and the design
matrix change, so no new HPO campaign is required.

Lightweight stats/figures -> outputs/sensitivity/residualized_bag_clean/.
Heavy per-fold eval artifacts -> the external runtime, under a namespace that
can never collide with the delivered `residualized_bag` cache.
"""
from __future__ import annotations

import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_ROW_ORDER,
    LEVEL_LABELS,
    OBJECTIVE_ARM_LABELS,
    RED_COLOR,
    ROW_LETTERS,
    SINGLE_COLOR,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from scripts.sensitivity_common import (
    active_rungs,
    analysis_cfg_from_config,
    build_original_model_df,
    bundle_sensitivity_root,
    combo_candidate_table,
    copy_tree_contents,
    evaluate_candidates_by_rung,
    filter_syn_pool,
    load_fig2_candidate_pool,
    load_raw_and_domains,
    load_sensitivity_config,
    log_msg,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    selected_bags,
    sensitivity_eval_work_root,
)

NAMESPACE = "residualized_bag_clean"
NULL_ARM_ID = "__null__"
BOOTSTRAP_DRAWS = int(os.environ.get("RESID_CLEAN_BOOTSTRAP_DRAWS", "10000"))
BOOTSTRAP_SEED = int(os.environ.get("RESID_CLEAN_BOOTSTRAP_SEED", "20260731"))

# The panel summarises each arm's whole candidate distribution, so these are
# arm names, not "Best ..." as in the delivered argmax figure.
ARM_LABELS = {
    "single_exposure": "Single exposure",
    "o_min": OBJECTIVE_ARM_LABELS["o_min"],
    "o_max": OBJECTIVE_ARM_LABELS["o_max"],
}
ARM_COLORS = {"single_exposure": SINGLE_COLOR, "o_min": SYN_COLOR, "o_max": RED_COLOR}


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "y"} if val else default


ORDER_MAX = int(os.environ["PAPER_FIG_ORDER_MAX"]) if os.environ.get("PAPER_FIG_ORDER_MAX") else 30


def null_candidate_df() -> pd.DataFrame:
    """The intercept-only comparator for a covariate-free residualized target.

    Deliberately NOT `baseline_candidate_df()`: that candidate is named
    `__baseline__` and means "covariates, no exposome". Under this namespace no
    candidate carries covariates, so the only meaningful floor is the training
    residual mean. Giving it a distinct id keeps the two un-confusable in any
    downstream table.
    """
    return pd.DataFrame(
        [
            {
                "candidate_id": NULL_ARM_ID,
                "feature_id": NULL_ARM_ID,
                "objective": "null",
                "order": 0,
                "rank": 0,
                "score": np.nan,
                "nplet_vars": [],
                "predictors_identity": "",
                "predictors_identity_n": 0,
                "candidate_family": "null",
                "source_label": "null",
            }
        ]
    )


def _cap_by_order(df: pd.DataFrame) -> pd.DataFrame:
    return df if ORDER_MAX is None else df[df["order"] <= ORDER_MAX]


def _arm_of(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame["candidate_id"].astype(str).eq(NULL_ARM_ID),
        "null",
        np.where(
            frame["candidate_family"].astype(str).eq("single_exposure"),
            "single_exposure",
            frame["objective"].astype(str),
        ),
    )


def _rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.

    r = (sum of positive signed ranks - sum of negative signed ranks) / total.
    Bounded in [-1, 1]; sign follows `differences`. Zero differences are
    dropped, matching `scipy.stats.wilcoxon`'s default handling.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return float("nan")
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    total = float(ranks.sum())
    if total <= 0:
        return float("nan")
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def _bootstrap_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a paired country-level statistic.

    Resamples COUNTRIES (the unit the paired test is taken over), not subjects.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(draws, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_error_tests(
    country: pd.DataFrame, chosen: dict[str, str], bag: str, rung: str
) -> pd.DataFrame:
    """Primary inference: paired country-level delta MSE of each arm vs null.

    One row per arm. Negative delta MSE favours the arm. The test is two-sided
    because the sensitivity asks whether the association survives, not whether
    it improves.
    """
    from scipy.stats import wilcoxon

    null_id = chosen["null"]
    null_rows = (
        country[country["candidate_id"].astype(str).eq(null_id)]
        .set_index("fold_country")
    )
    rows = []
    for arm, candidate_id in chosen.items():
        if arm == "null":
            continue
        arm_rows = (
            country[country["candidate_id"].astype(str).eq(candidate_id)]
            .set_index("fold_country")
        )
        paired = pd.concat(
            [
                (arm_rows["rmse"] ** 2).rename("arm_mse"),
                (null_rows["rmse"] ** 2).rename("null_mse"),
                arm_rows["mae"].rename("arm_mae"),
                null_rows["mae"].rename("null_mae"),
                arm_rows["r2"].rename("arm_r2"),
                null_rows["r2"].rename("null_r2"),
                arm_rows["n_test"].rename("n_test"),
            ],
            axis=1,
        ).dropna(subset=["arm_mse", "null_mse"])
        delta_mse = (paired["arm_mse"] - paired["null_mse"]).to_numpy()
        delta_mae = (paired["arm_mae"] - paired["null_mae"]).to_numpy()
        statistic, p_value = (
            wilcoxon(paired["arm_mse"], paired["null_mse"])
            if len(paired) >= 3 and np.any(delta_mse != 0)
            else (np.nan, np.nan)
        )
        low, high = _bootstrap_ci(delta_mse, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "arm": arm,
                "arm_label": ARM_LABELS.get(arm, arm),
                "candidate_id": candidate_id,
                "null_candidate_id": null_id,
                "n_countries": int(len(paired)),
                "mean_delta_mse": float(np.mean(delta_mse)) if delta_mse.size else np.nan,
                "median_delta_mse": float(np.median(delta_mse)) if delta_mse.size else np.nan,
                "delta_mse_ci_low": low,
                "delta_mse_ci_high": high,
                "countries_favouring_arm": int(np.sum(delta_mse < 0)),
                "wilcoxon_statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "wilcoxon_p": float(p_value) if np.isfinite(p_value) else np.nan,
                "rank_biserial": _rank_biserial(delta_mse),
                "mean_delta_mae": float(np.mean(delta_mae)) if delta_mae.size else np.nan,
                # Audit-only, never the inferential statistic.
                "audit_mean_country_r2_arm": float(paired["arm_r2"].mean()),
                "audit_mean_country_r2_null": float(paired["null_r2"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _holm(frame: pd.DataFrame, column: str, within: list[str]) -> pd.Series:
    """Holm-Bonferroni within each (bag, rung) family of arm-vs-null tests."""
    adjusted = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(within, dropna=False):
        values = group[column].dropna().sort_values()
        m = len(values)
        running = 0.0
        for rank, (idx, p) in enumerate(values.items()):
            running = max(running, min(1.0, float(p) * (m - rank)))
            adjusted.loc[idx] = running
    return adjusted


def landscape_summary(
    global_frame: pd.DataFrame, country: pd.DataFrame, bag: str, rung: str
) -> pd.DataFrame:
    """Descriptive landscape summary over ALL candidates, not the argmax.

    Reports pooled OOF R2 (primary descriptive), the audit mean-country R2, and
    the share of candidates above the null arm, per arm. This is what makes the
    comparison landscape-level rather than a best-candidate claim.
    """
    frame = global_frame.copy()
    frame["arm"] = _arm_of(frame)
    audit = (
        country[country["n_test"].gt(0)]
        .groupby("candidate_id", observed=True)["r2"]
        .mean()
        .rename("audit_mean_country_r2")
    )
    frame = frame.merge(audit, left_on="candidate_id", right_index=True, how="left")
    null_pooled = float(
        frame.loc[frame["arm"].eq("null"), "global_oof_r2"].iloc[0]
    ) if (frame["arm"] == "null").any() else np.nan

    rows = []
    for arm, group in frame.groupby("arm"):
        if arm == "null":
            continue
        pooled = pd.to_numeric(group["global_oof_r2"], errors="coerce").dropna()
        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "arm": arm,
                "arm_label": ARM_LABELS.get(arm, arm),
                "n_candidates": int(len(group)),
                # `_best` is what the figure plots, matching the delivered
                # figure's "Best single exposure / Best Min O-info / Best Max
                # O-info" series. The quantiles stay for the landscape-level
                # reading, which is the summary that does not depend on argmax.
                "pooled_oof_r2_best": float(pooled.max()),
                "pooled_oof_r2_median": float(pooled.median()),
                "pooled_oof_r2_q25": float(pooled.quantile(0.25)),
                "pooled_oof_r2_q75": float(pooled.quantile(0.75)),
                "frac_pooled_above_null": float(np.mean(pooled > null_pooled)) if np.isfinite(null_pooled) else np.nan,
                "frac_pooled_above_zero": float(np.mean(pooled > 0.0)),
                "audit_mean_country_r2_median": float(
                    pd.to_numeric(group["audit_mean_country_r2"], errors="coerce").median()
                ),
                "null_pooled_oof_r2": null_pooled,
            }
        )
    return pd.DataFrame(rows)


def landscape_ordering(summary: pd.DataFrame) -> pd.DataFrame:
    """Observed ordering per (bag, rung), with pairwise separation.

    Deliberately descriptive: no expected ordering is asserted anywhere, because
    the ordering is not stable across rungs (see module docstring).
    """
    rows = []
    for (bag, rung), group in summary.groupby(["bag", "rung_id"], sort=False):
        ordered = group.sort_values(
            ["pooled_oof_r2_median", "arm"], ascending=[False, True], kind="stable"
        )
        labels = (
            ordered["arm_label"]
            if "arm_label" in ordered.columns
            else ordered["arm"].map(ARM_LABELS).fillna(ordered["arm"])
        )
        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "observed_ordering": " > ".join(ordered["arm"].tolist()),
                "observed_ordering_labels": " > ".join(labels.tolist()),
                **{
                    f"median_{arm}": float(
                        group.loc[group["arm"].eq(arm), "pooled_oof_r2_median"].iloc[0]
                    )
                    for arm in group["arm"].tolist()
                },
            }
        )
    return pd.DataFrame(rows)


def pairwise_landscape_separation(
    global_frame: pd.DataFrame, bag: str, rung: str
) -> pd.DataFrame:
    """Mann-Whitney separation between arm landscapes, with a CLES effect size.

    Reported so a reader can see whether an ordering is a real separation or a
    coin flip -- the delivered-cache functional ordering at d2 was not separated
    (P(o_min>o_max)=0.48, p=0.17) even though it had a nominal ordering.
    """
    from scipy.stats import mannwhitneyu

    frame = global_frame.copy()
    frame["arm"] = _arm_of(frame)
    pools = {
        arm: pd.to_numeric(group["global_oof_r2"], errors="coerce").dropna().to_numpy()
        for arm, group in frame.groupby("arm")
        if arm != "null"
    }
    rows = []
    for left, right in (("o_min", "single_exposure"), ("o_min", "o_max"), ("o_max", "single_exposure")):
        if left not in pools or right not in pools or not pools[left].size or not pools[right].size:
            continue
        statistic, p_value = mannwhitneyu(pools[left], pools[right], alternative="two-sided")
        rows.append(
            {
                "bag": bag,
                "rung_id": rung,
                "comparison": f"{left}_vs_{right}",
                "cles_left_gt_right": float(statistic / (pools[left].size * pools[right].size)),
                "mannwhitney_p": float(p_value),
                "n_left": int(pools[left].size),
                "n_right": int(pools[right].size),
            }
        )
    return pd.DataFrame(rows)


def _select_arms(global_frame: pd.DataFrame, country: pd.DataFrame) -> dict[str, str]:
    """Best candidate per arm, selected on POOLED OOF R2.

    Selection uses the descriptive estimand rather than the audit mean-country
    R2 on purpose: selecting on a statistic known to be distorted by the fold
    offset would propagate that distortion into the paired test.
    """
    frame = global_frame.copy()
    frame["arm"] = _arm_of(frame)
    chosen: dict[str, str] = {}
    for arm, group in frame.groupby("arm"):
        scores = pd.to_numeric(group["global_oof_r2"], errors="coerce")
        if scores.notna().any():
            chosen[arm] = str(group.loc[scores.idxmax(), "candidate_id"])
    return chosen


def _draw_panel(
    ax,
    bag: str,
    row_letter: str,
    tests: pd.DataFrame,
    landscape: pd.DataFrame,
    rungs: list[str],
) -> Panel:
    """One BAG panel: the R2 LANDSCAPE per arm (median + IQR), by model level.

    Plotting convention follows the delivered `residualized_bag` figure: same
    `R2 LOCO` axis, same arm colours, `lw=1.9`, `marker="o"`, `ms=5`, the black
    `lw=2.2`, `marker="s"` reference series, `mpatches.Patch` legend and
    "<letter>. <BAG>" row label. Under this namespace the black reference is
    the null (intercept-only) comparator, because no candidate carries
    covariates any more.

    WHY MEDIAN + IQR RATHER THAN THE BEST CANDIDATE. The best-per-arm R2 is not
    a safe summary here. The exposome is assigned at country-year level and
    replicated to subjects, so this population has only ~205 unique country-year
    signatures behind 12,868 subject rows. Unregularised OLS with 20-30
    predictors therefore fits on ~205 effective points and then extrapolates to
    a country it never saw (LOCO), which makes its landscape extremely wide:
    functional o_max spans max +0.0066 to min -12.41 at OLS, against -0.0288 at
    d3. Taking the maximum over 1183 candidates from a distribution that wide
    selects noise, not signal, and made OLS appear to beat every tree rung. The
    degradation is monotone in candidate order (corr(order, R2) = -0.412 for
    OLS; XGB is flat), which is the signature of unregularised extrapolation
    rather than of a real OLS advantage.

    The median and IQR describe the whole arm and are not distorted by that
    selection, which is also what makes this panel a landscape-level statement
    rather than an argmax claim. The paired ΔMSE test remains the primary
    inferential statistic; it is carried in the Source Data columns and the
    statistics tables, not drawn.
    """
    x = list(range(len(rungs)))

    def _series(column, arm):
        out = []
        for rung in rungs:
            sel = landscape[
                landscape["bag"].eq(bag)
                & landscape["rung_id"].eq(rung)
                & landscape["arm"].eq(arm)
            ]
            out.append(float(sel[column].iloc[0]) if len(sel) else np.nan)
        return out

    null_series = _series("null_pooled_oof_r2", "o_min")
    if not np.isfinite(null_series).any():
        null_series = _series("null_pooled_oof_r2", "single_exposure")
    ax.plot(
        x, null_series, color="black", lw=2.2, marker="s", ms=5,
        label="Null (intercept-only)",
    )

    present: list[str] = []
    for arm in ("single_exposure", "o_min", "o_max"):
        median = _series("pooled_oof_r2_median", arm)
        if not np.isfinite(median).any():
            continue
        present.append(arm)
        low = _series("pooled_oof_r2_q25", arm)
        high = _series("pooled_oof_r2_q75", arm)
        ax.fill_between(x, low, high, color=ARM_COLORS[arm], alpha=0.16, linewidth=0)
        ax.plot(x, median, color=ARM_COLORS[arm], lw=1.9, marker="o", ms=5, label=ARM_LABELS[arm])

    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS.get(r, r) for r in rungs])
    ax.set_xlabel("Model level")
    ax.set_ylabel("R² LOCO")
    ax.set_title(f"{row_letter}. {BAG_LABELS[bag]}", loc="left", fontsize=11)
    # Symmetric log: OLS's collapse (down to -0.39 IQR on functional) would
    # otherwise flatten every tree rung into the zero line. `linthresh` keeps
    # the |R2| < 0.01 region -- where the tree rungs live -- linear and legible,
    # while the OLS excursion stays visible rather than clipped away.
    ax.set_yscale("symlog", linthresh=0.01, linscale=1.1)
    ax.grid(axis="y", alpha=0.2)
    style_axis(ax)

    handles = [mpatches.Patch(color="black", label="Null (intercept-only)")]
    handles.extend(mpatches.Patch(color=ARM_COLORS[a], label=ARM_LABELS[a]) for a in present)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")

    frame = pd.DataFrame({"model_level": [LEVEL_LABELS.get(r, r) for r in rungs]})
    frame["null_r2"] = null_series
    for arm, suffix in (
        ("single_exposure", "single"),
        ("o_min", "synergy"),
        ("o_max", "redundancy"),
    ):
        if arm not in present:
            continue
        frame[f"{suffix}_r2_median"] = _series("pooled_oof_r2_median", arm)
        frame[f"{suffix}_r2_q25"] = _series("pooled_oof_r2_q25", arm)
        frame[f"{suffix}_r2_q75"] = _series("pooled_oof_r2_q75", arm)
        frame[f"{suffix}_frac_above_null"] = _series("frac_pooled_above_null", arm)
        frame[f"{suffix}_n_candidates"] = _series("n_candidates", arm)

    stats = tests[tests["bag"].eq(bag)].copy()
    stats["model_level"] = stats["rung_id"].map(LEVEL_LABELS).fillna(stats["rung_id"])
    for arm, suffix in (("single_exposure", "single"), ("o_min", "synergy"), ("o_max", "redundancy")):
        sub = stats[stats["arm"].eq(arm)].set_index("model_level")
        if sub.empty:
            continue
        frame[f"{suffix}_delta_mse_vs_null"] = frame["model_level"].map(sub["mean_delta_mse"])
        frame[f"{suffix}_holm_p_vs_null"] = frame["model_level"].map(sub["holm_p"])

    return Panel(
        panel_id=f"{row_letter}_{bag}_r2_landscape",
        frame=frame,
        description=(
            f"Held-out LOCO R² landscape by model level under a {BAG_LABELS[bag]} target with "
            "age, year, sex and diagnosis already regressed out within each training fold, "
            "using exposome predictors only. Lines are the median across all candidates in "
            "each arm and bands are the interquartile range; the reference is the null "
            "(intercept-only) comparator."
        ),
        columns={
            "model_level": "model level (OLS, d1, d2, d3)",
            "null_r2": "null comparator: fold training residual mean, zero exposome predictors",
            "single_r2_median": "median pooled OOF R² across the 63 single-exposure candidates",
            "single_r2_q25": "25th percentile of that arm's candidate R² distribution",
            "single_r2_q75": "75th percentile of that arm's candidate R² distribution",
            "single_frac_above_null": "share of the arm's candidates above the null comparator",
            "single_n_candidates": "candidates contributing to the arm",
            "synergy_r2_median": "median pooled OOF R² across the synergistic (o_min) candidates",
            "synergy_r2_q25": "25th percentile of that arm's candidate R² distribution",
            "synergy_r2_q75": "75th percentile of that arm's candidate R² distribution",
            "synergy_frac_above_null": "share of the arm's candidates above the null comparator",
            "synergy_n_candidates": "candidates contributing to the arm",
            "redundancy_r2_median": "median pooled OOF R² across the redundant (o_max) candidates",
            "redundancy_r2_q25": "25th percentile of that arm's candidate R² distribution",
            "redundancy_r2_q75": "75th percentile of that arm's candidate R² distribution",
            "redundancy_frac_above_null": "share of the arm's candidates above the null comparator",
            "redundancy_n_candidates": "candidates contributing to the arm",
            "single_delta_mse_vs_null": "primary statistic: mean held-out-country ΔMSE, best single vs null",
            "single_holm_p_vs_null": "Holm-adjusted paired Wilcoxon p for that ΔMSE",
            "synergy_delta_mse_vs_null": "primary statistic: mean held-out-country ΔMSE, best synergy vs null",
            "synergy_holm_p_vs_null": "Holm-adjusted paired Wilcoxon p for that ΔMSE",
            "redundancy_delta_mse_vs_null": "primary statistic: mean held-out-country ΔMSE, best redundancy vs null",
            "redundancy_holm_p_vs_null": "Holm-adjusted paired Wilcoxon p for that ΔMSE",
        },
        test=(
            "Two-sided paired Wilcoxon signed-rank test on held-out-country MSE, each arm's "
            "best candidate versus the null comparator, Holm-adjusted within each BAG and "
            "model level. The plotted R² landscape is descriptive; the ΔMSE columns carry the "
            "inference."
        ),
        notes=(
            "R² estimand: pooled subject-level out-of-fold LOCO R². Residualisation is per-fold "
            "and leakage-free: BAG ~ age + year + sex + diagnosis is refit on each LOCO fold's "
            "training countries only, so the held-out country never informs its own "
            "residualisation. Downstream models use the exposome ONLY; no covariate is "
            "reintroduced, so the reference is the null comparator rather than a covariate "
            "baseline. The arm summary is the median and IQR over all candidates, not the best "
            "candidate: the exposome is assigned at country-year level (~205 unique signatures "
            "here), so unregularised OLS on high-order candidate sets extrapolates to unseen "
            "countries and produces an extremely wide landscape (functional o_max reaches "
            "-12.41 at OLS versus -0.0288 at d3); a maximum taken over 1183 such candidates "
            "selects noise. These R² values are NOT comparable with the delivered "
            "residualized_bag figure, which left year in the target and carried the covariates "
            "as predictors. Unweighted mean held-out-country R² is retained as an audit metric "
            "in the statistics tables only."
        ),
    )


def build_figure(tests, landscape, bags: list[str], rungs: list[str], outdir) -> None:
    present = [b for b in bags if not tests[tests["bag"].eq(b)].empty]
    if not present:
        return
    fig, axes = plt.subplots(
        1, len(present), figsize=(7.2 * len(present), 4.6), squeeze=False,
        gridspec_kw={"wspace": 0.22},
    )
    panels = [
        _draw_panel(axes[0][i], bag, ROW_LETTERS[i], tests, landscape, rungs)
        for i, bag in enumerate(present)
    ]
    save_figure(fig, NAMESPACE, outdir)
    plt.close(fig)
    write_source_data(NAMESPACE, panels, outdir)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = _env_bool("SMOKE_TEST") or _env_bool("SENSITIVITY_SMOKE")
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = int(os.environ.get("SENSITIVITY_MAX_CANDIDATES", "24")) if smoke else None

    base_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / NAMESPACE
    local_root.mkdir(parents=True, exist_ok=True)
    eval_root = sensitivity_eval_work_root(cfg, NAMESPACE)

    raw, _domains, feature_names, _domain_map = load_raw_and_domains()
    full_model = build_original_model_df(raw, feature_names, base_cfg)

    for bag in selected_bags(cfg, include_combined=False):
        log_msg(f"=== {NAMESPACE} | bag={bag} | smoke={smoke} | order_max={ORDER_MAX} ===")
        bag_dir = local_root / bag
        bag_dir.mkdir(parents=True, exist_ok=True)
        eval_dir = eval_root / bag
        eval_dir.mkdir(parents=True, exist_ok=True)

        multi_pool = _cap_by_order(load_fig2_candidate_pool(bag))
        if max_candidates is not None:
            per_objective = max(1, max_candidates // max(1, multi_pool["objective"].nunique()))
            multi_pool = (
                multi_pool.groupby("objective", group_keys=False)
                .head(per_objective)
                .reset_index(drop=True)
            )
        single_pool = combo_candidate_table(
            feature_names,
            family="single_exposure",
            source_label="single_exposure",
            order_min=1,
            order_max=1,
            prefix=f"single_{bag}",
            extra={"objective": "single_exposure"},
        )
        single_pool["feature_id"] = single_pool["nplet_vars"].apply(lambda v: v[0])
        pool = pd.concat([multi_pool, single_pool, null_candidate_df()], ignore_index=True)
        log_msg(
            f"  candidate pool: {len(pool)} "
            f"(multi={len(multi_pool)}, single={len(single_pool)}, null=1)"
        )

        for rung in rungs:
            log_msg(f"  rung={rung}: {len(pool)} candidates, covariate-free residualised target")
            summary, country = evaluate_candidates_by_rung(
                model_df=full_model,
                candidate_df=pool,
                exposome_cols=feature_names,
                bag=bag,
                rungs=[rung],
                analysis_cfg=base_cfg,
                outdir=eval_dir / f"clean_eval_{rung}",
                n_jobs=n_jobs,
                max_candidates=None,
                fold_residualize_with_year=True,
                drop_covariates=True,
                residualization_diagnostics_path=(
                    bag_dir / f"{bag}_{rung}_residualization_diagnostics.csv"
                ),
            )

    # Assemble deliveries from whatever completed, so a per-BAG scheduled job
    # that finishes second still writes the single combined figure.
    tests_all, landscape_all, ordering_all, separation_all = [], [], [], []
    for bag in BAG_ROW_ORDER:
        for rung in rungs:
            folder = eval_root / bag / f"clean_eval_{rung}"
            global_path = folder / f"{bag}_{rung}_global.csv"
            country_path = folder / f"{bag}_{rung}_country.csv"
            if not (global_path.is_file() and country_path.is_file()):
                continue
            global_frame = pd.read_csv(global_path)
            country = pd.read_csv(country_path)
            chosen = _select_arms(global_frame, country)
            if "null" not in chosen:
                log_msg(f"  WARNING: no null arm for bag={bag} rung={rung}; skipping tests")
                continue
            tests_all.append(paired_error_tests(country, chosen, bag, rung))
            landscape_all.append(landscape_summary(global_frame, country, bag, rung))
            separation_all.append(pairwise_landscape_separation(global_frame, bag, rung))

    if not tests_all:
        log_msg("No completed evaluations found; nothing to deliver yet.")
        return

    tests = pd.concat(tests_all, ignore_index=True)
    tests["holm_p"] = _holm(tests, "wilcoxon_p", ["bag", "rung_id"])
    landscape = pd.concat(landscape_all, ignore_index=True)
    separation = pd.concat(separation_all, ignore_index=True)
    ordering = landscape_ordering(landscape)

    tests.to_csv(local_root / f"{NAMESPACE}_paired_error_tests.csv", index=False)
    landscape.to_csv(local_root / f"{NAMESPACE}_landscape_summary.csv", index=False)
    ordering.to_csv(local_root / f"{NAMESPACE}_landscape_ordering.csv", index=False)
    separation.to_csv(local_root / f"{NAMESPACE}_landscape_separation.csv", index=False)

    manifest = {
        "namespace": NAMESPACE,
        "residualizer_formula": "BAG ~ 1 + Age + Year + Sex + Diagnosis",
        "residualizer_fit_scope": "LOCO training countries only (leakage-free)",
        "downstream_design": "exposome only; no covariates reintroduced",
        "comparator": "null arm (__null__): fold training residual mean",
        "primary_statistic": "paired held-out-country delta MSE vs null",
        "primary_test": "two-sided Wilcoxon signed-rank, Holm within (bag, rung)",
        "effect_size": "matched-pairs rank-biserial correlation",
        "interval": f"percentile bootstrap over countries, {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}",
        "descriptive_statistic": "pooled subject-level OOF R2",
        "audit_statistic": "unweighted mean held-out-country R2 (audit columns only)",
        "landscape_ordering_policy": "reported per rung; never asserted",
        "order_max": ORDER_MAX,
        "rungs": list(rungs),
        "bags": [b for b in BAG_ROW_ORDER if not tests[tests["bag"].eq(b)].empty],
        "exclude_countries": list(base_cfg.get("exclude_countries", [])),
        "exclude_diagnosis": list(base_cfg.get("exclude_diagnosis", [])),
        "hpo_reused_unchanged": True,
        "preserves_namespace": "outputs/sensitivity/residualized_bag (untouched)",
    }
    (local_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig_root = repo_sensitivity_figures_root(cfg, NAMESPACE)
    build_figure(tests, landscape, list(BAG_ROW_ORDER), rungs, fig_root)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / NAMESPACE)
    copy_tree_contents(fig_root, bundle_sensitivity_root(cfg) / NAMESPACE)

    log_msg("\n=== observed landscape ordering (reported, not asserted) ===")
    for _, row in ordering.iterrows():
        log_msg(f"  {row['bag']:11s} {row['rung_id']:12s} {row['observed_ordering']}")


if __name__ == "__main__":
    main()
