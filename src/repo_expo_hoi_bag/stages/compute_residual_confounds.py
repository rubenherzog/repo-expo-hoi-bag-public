#!/usr/bin/env python3
"""Test whether country structures signed BAG bias after measured covariates.

For each paper BAG and model arm, this stage fits two maximum-likelihood models
to subject-level out-of-fold bias (predicted BAG minus observed BAG):

1. a fixed-only model containing individual diagnosis, age and sex plus
   country sample size, within-country BAG dispersion and dataset heterogeneity;
2. the same fixed effects plus a country random intercept.

The adjusted country variance and ICC describe remaining country clustering.
Because the null hypothesis places the random-intercept variance on the boundary
at zero, country is tested with a parametric-bootstrap likelihood-ratio test,
not an ordinary chi-square reference distribution. The bootstrap uses a fast,
profiled one-way random-intercept likelihood that is checked against Statsmodels
for the observed data.

Lightweight outputs -> outputs/sensitivity[/dedup]/residual_confounds/
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse, stats
from scipy.optimize import minimize_scalar

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data
from repo_expo_hoi_bag.figures.style import (
    BAG_LABELS,
    BAG_SHORT,
    NULL_COLOR,
    RED_COLOR,
    ROW_LETTERS,
    SYN_COLOR,
    save_figure,
    style_axis,
)
from scripts.sensitivity_common import (
    REPO_ROOT,
    bundle_sensitivity_root,
    copy_tree_contents,
    load_raw_and_domains,
    load_sensitivity_config,
    paper_analysis_config,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
)


RESIDUALS_DIR = Path(
    os.environ.get(
        "V3_RESIDUALS_DIR",
        str(REPO_ROOT / "outputs" / "variant_a" / "stats" / "residuals"),
    )
)
_SENSITIVITY_CFG = load_sensitivity_config()
_PAPER_CFG = paper_analysis_config(_SENSITIVITY_CFG)
_CONFOUND_CFG = _PAPER_CFG.residual_confounds
OBJ_SUFFIX = {
    objective: values["file_suffix"]
    for objective, values in _PAPER_CFG.objective_metadata.items()
}
_REQUESTED_BAGS = [item.strip() for item in os.environ.get("SENSITIVITY_BAGS", "").split(",") if item.strip()]
BAGS = [bag for bag in _PAPER_CFG.primary_bags if not _REQUESTED_BAGS or bag in _REQUESTED_BAGS]
OBJECTIVES = list(_PAPER_CFG.objectives)
PRIMARY_DIAGNOSES = tuple(_PAPER_CFG.primary_diagnoses)
CONTROL_DIAGNOSIS = _PAPER_CFG.control_diagnosis
INDIVIDUAL_COVARIATES = tuple(
    str(value) for value in _CONFOUND_CFG["individual_covariates"]
)
COUNTRY_COVARIATES = tuple(
    str(value) for value in _CONFOUND_CFG["country_covariates"]
)
SEX_REFERENCE = str(_CONFOUND_CFG["sex_reference"])
OPTIMIZERS = tuple(str(value) for value in _CONFOUND_CFG["optimizers"])
OPTIMIZER_MAX_ITERATIONS = int(_CONFOUND_CFG["optimizer_max_iterations"])
BOOTSTRAP_DRAWS = int(_CONFOUND_CFG["random_effect_bootstrap_draws"])
BOOTSTRAP_SEED = int(_CONFOUND_CFG["random_effect_bootstrap_seed"])
BOOTSTRAP_BATCH_SIZE = int(
    _CONFOUND_CFG["random_effect_bootstrap_batch_size"]
)
TAU_MAX = float(_CONFOUND_CFG["random_effect_tau_max"])

_SUPPORTED_INDIVIDUAL_COVARIATES = {"age", "sex", "diagnosis"}
_SUPPORTED_COUNTRY_COVARIATES = {
    "log_n_country",
    "bag_sd",
    "dataset_heterogeneity",
}

if set(INDIVIDUAL_COVARIATES).difference(_SUPPORTED_INDIVIDUAL_COVARIATES):
    raise ValueError("Unsupported individual residual-confound covariate")
if set(COUNTRY_COVARIATES).difference(_SUPPORTED_COUNTRY_COVARIATES):
    raise ValueError("Unsupported country residual-confound covariate")
if not OPTIMIZERS or OPTIMIZER_MAX_ITERATIONS < 1:
    raise ValueError("Invalid residual-confounds optimizer configuration")
if BOOTSTRAP_DRAWS < 1 or BOOTSTRAP_BATCH_SIZE < 1 or TAU_MAX <= 0:
    raise ValueError("Invalid random-intercept bootstrap configuration")


def _safe_term(value: object) -> str:
    return "".join(char if str(char).isalnum() else "_" for char in str(value))


def _shannon_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    standard_deviation = numeric.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return numeric * 0.0
    return (numeric - numeric.mean()) / standard_deviation


def _build_design(
    bag: str,
    objective: str,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    suffix = OBJ_SUFFIX[objective]
    residual_path = RESIDUALS_DIR / f"residuals_subject_{bag}_{suffix}.csv"
    if not residual_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction residuals at {residual_path}. "
            "Run compute_regression_residuals first."
        )

    design = pd.read_csv(residual_path)
    design["N_MEGA"] = design["N_MEGA"].astype(str).str.strip()
    design = design[
        design["diagnosis"].astype(str).isin(PRIMARY_DIAGNOSES)
    ].copy()
    design["residual"] = pd.to_numeric(design["residual"], errors="coerce")
    design["y_true"] = pd.to_numeric(design["y_true"], errors="coerce")

    raw_subset = raw[
        raw["N_MEGA"].astype(str).str.strip().isin(set(design["N_MEGA"]))
    ].copy()
    dataset_heterogeneity = (
        raw_subset.groupby("country_clean")["Dataset"]
        .apply(_shannon_entropy)
        .rename("dataset_heterogeneity")
    )
    country_level = (
        design.groupby("country", observed=True)
        .agg(n_country=("residual", "size"), bag_sd=("y_true", "std"))
        .join(dataset_heterogeneity, how="left")
        .reset_index()
    )
    country_level["log_n_country"] = np.log(country_level["n_country"])
    for covariate in COUNTRY_COVARIATES:
        country_level[f"{covariate}_z"] = _zscore(country_level[covariate])
    design = design.merge(country_level, on="country", how="left", validate="many_to_one")

    fixed_effects: list[str] = []
    effect_levels: dict[str, str] = {}
    if "age" in INDIVIDUAL_COVARIATES:
        design["age_z"] = _zscore(design["age"])
        fixed_effects.append("age_z")
        effect_levels["age_z"] = "individual"

    if "sex" in INDIVIDUAL_COVARIATES:
        sex_levels = sorted(design["sex"].dropna().astype(str).unique())
        if SEX_REFERENCE not in sex_levels:
            raise ValueError(
                f"Configured sex reference {SEX_REFERENCE!r} is absent: {sex_levels}"
            )
        for level in sex_levels:
            if level == SEX_REFERENCE:
                continue
            term = f"sex__{_safe_term(level)}"
            design[term] = (design["sex"].astype(str) == level).astype(float)
            fixed_effects.append(term)
            effect_levels[term] = "individual"

    if "diagnosis" in INDIVIDUAL_COVARIATES:
        for diagnosis in PRIMARY_DIAGNOSES:
            if diagnosis == CONTROL_DIAGNOSIS:
                continue
            term = f"diagnosis__{_safe_term(diagnosis)}"
            design[term] = (
                design["diagnosis"].astype(str) == diagnosis
            ).astype(float)
            fixed_effects.append(term)
            effect_levels[term] = "individual"

    for covariate in COUNTRY_COVARIATES:
        term = f"{covariate}_z"
        fixed_effects.append(term)
        effect_levels[term] = "country"

    if not fixed_effects:
        raise ValueError("The adjusted country model has no fixed effects")
    missing = [term for term in fixed_effects if term not in design.columns]
    if missing:
        raise KeyError(f"Residual-confound design is missing terms: {missing}")
    return design, fixed_effects, effect_levels


def _fit_mixed_model(formula: str, fit_df: pd.DataFrame):
    attempts = []
    for method in OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = sm.MixedLM.from_formula(
                    formula,
                    groups="country",
                    data=fit_df,
                ).fit(
                    reml=False,
                    method=method,
                    maxiter=OPTIMIZER_MAX_ITERATIONS,
                    disp=False,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                attempts.append((method, None, repr(exc)))
                continue
        warning_text = " | ".join(str(item.message) for item in caught)
        attempts.append((method, result, warning_text))
        singular_covariance = "covariance is singular" in warning_text.lower()
        valid = (
            result.converged
            and np.isfinite(float(result.llf))
            and np.isfinite(float(result.cov_re.iloc[0, 0]))
            and float(result.cov_re.iloc[0, 0]) >= 0
            and not singular_covariance
        )
        if valid:
            return result, method, warning_text

    fitted = [attempt for attempt in attempts if attempt[1] is not None]
    if not fitted:
        details = "; ".join(
            f"{method}: {message}" for method, _, message in attempts
        )
        raise RuntimeError(f"Every mixed-model optimizer failed: {details}")
    method, result, warning_text = max(
        fitted, key=lambda item: float(item[1].llf)
    )
    return result, method, warning_text


@dataclass(frozen=True)
class _RandomInterceptProfile:
    """Fixed-design sufficient statistics for a one-way random intercept."""

    x: np.ndarray
    group_codes: np.ndarray
    group_sizes: np.ndarray
    xx: np.ndarray
    group_x_sums: np.ndarray

    @classmethod
    def from_design(
        cls,
        x: np.ndarray,
        groups: pd.Series,
    ) -> "_RandomInterceptProfile":
        group_codes = pd.Categorical(groups.astype(str)).codes.astype(int)
        if np.any(group_codes < 0):
            raise ValueError("Country group contains missing values")
        n_groups = int(group_codes.max()) + 1
        group_sizes = np.bincount(group_codes, minlength=n_groups).astype(float)
        group_x_sums = np.zeros((n_groups, x.shape[1]), dtype=float)
        np.add.at(group_x_sums, group_codes, x)
        return cls(
            x=x,
            group_codes=group_codes,
            group_sizes=group_sizes,
            xx=x.T @ x,
            group_x_sums=group_x_sums,
        )

    def y_statistics(
        self,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        group_y_sums = np.bincount(
            self.group_codes,
            weights=y,
            minlength=len(self.group_sizes),
        ).astype(float)
        return self.x.T @ y, group_y_sums, float(y @ y)

    def log_likelihood(
        self,
        tau: float,
        xy: np.ndarray,
        group_y_sums: np.ndarray,
        y_squared: float,
    ) -> tuple[float, float, np.ndarray]:
        """Profile ML log-likelihood for variance ratio tau = var(country)/var(error)."""
        alpha = tau / (1.0 + tau * self.group_sizes)
        weighted_group_x = self.group_x_sums * alpha[:, None]
        information = self.xx - self.group_x_sums.T @ weighted_group_x
        score = xy - self.group_x_sums.T @ (alpha * group_y_sums)
        transformed_y_squared = y_squared - float(
            np.sum(alpha * np.square(group_y_sums))
        )
        try:
            beta = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(information, score, rcond=None)[0]
        residual_sum = transformed_y_squared - float(beta @ score)
        residual_sum = max(residual_sum, np.finfo(float).tiny)
        n_observations = self.x.shape[0]
        residual_variance = residual_sum / n_observations
        log_determinant = float(np.log1p(tau * self.group_sizes).sum())
        log_likelihood = -0.5 * (
            n_observations
            * (
                np.log(2.0 * np.pi)
                + 1.0
                + np.log(residual_variance)
            )
            + log_determinant
        )
        return float(log_likelihood), float(residual_variance), beta

    def maximize(
        self,
        xy: np.ndarray,
        group_y_sums: np.ndarray,
        y_squared: float,
    ) -> tuple[float, float, float, np.ndarray]:
        null_ll, null_variance, null_beta = self.log_likelihood(
            0.0, xy, group_y_sums, y_squared
        )

        def objective(log_tau: float) -> float:
            tau = float(np.exp(log_tau))
            return -self.log_likelihood(
                tau, xy, group_y_sums, y_squared
            )[0]

        optimized = minimize_scalar(
            objective,
            bounds=(np.log(1e-10), np.log(TAU_MAX)),
            method="bounded",
            options={"xatol": 1e-7},
        )
        tau = float(np.exp(optimized.x))
        alternative_ll, residual_variance, beta = self.log_likelihood(
            tau, xy, group_y_sums, y_squared
        )
        if not optimized.success or alternative_ll <= null_ll:
            return null_ll, 0.0, null_variance, null_beta
        if tau >= 0.999 * TAU_MAX:
            raise RuntimeError(
                "Profiled country variance reached random_effect_tau_max; "
                "increase the configured bound"
            )
        return alternative_ll, tau, residual_variance, beta


def _bootstrap_country_random_intercept(
    fit_df: pd.DataFrame,
    fixed_only,
    adjusted_mixed,
    *,
    bag: str,
    objective: str,
) -> dict[str, object]:
    x = np.asarray(fixed_only.model.exog, dtype=float)
    y = np.asarray(fixed_only.model.endog, dtype=float)
    profile = _RandomInterceptProfile.from_design(x, fit_df["country"])
    xy, group_y_sums, y_squared = profile.y_statistics(y)
    null_profile_ll, _, _ = profile.log_likelihood(
        0.0, xy, group_y_sums, y_squared
    )
    alternative_profile_ll, tau, residual_variance, _ = profile.maximize(
        xy, group_y_sums, y_squared
    )
    observed_lrt = max(
        0.0, 2.0 * (alternative_profile_ll - null_profile_ll)
    )

    null_agreement = abs(null_profile_ll - float(fixed_only.llf))
    mixed_agreement = abs(alternative_profile_ll - float(adjusted_mixed.llf))
    if null_agreement > 1e-6:
        raise RuntimeError(
            f"Profile/OLS null log-likelihood mismatch: {null_agreement}"
        )
    if mixed_agreement > 0.05:
        raise RuntimeError(
            "Profile/Statsmodels mixed log-likelihood mismatch: "
            f"{mixed_agreement}"
        )

    fitted_mean = np.asarray(fixed_only.fittedvalues, dtype=float)
    null_sigma = float(np.sqrt(np.mean(np.square(fixed_only.resid))))
    n_observations = len(fitted_mean)
    n_groups = len(profile.group_sizes)
    group_indicator = sparse.csr_matrix(
        (
            np.ones(n_observations, dtype=float),
            (profile.group_codes, np.arange(n_observations)),
        ),
        shape=(n_groups, n_observations),
    )
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + sum(ord(char) for char in f"{bag}|{objective}|country_random_intercept")
    )
    exceedances = 0
    completed = 0
    null_lrt_values = np.empty(BOOTSTRAP_DRAWS, dtype=float)

    while completed < BOOTSTRAP_DRAWS:
        batch_size = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_DRAWS - completed)
        simulated = fitted_mean[:, None] + null_sigma * rng.standard_normal(
            size=(n_observations, batch_size)
        )
        xy_batch = x.T @ simulated
        group_y_batch = np.asarray(group_indicator @ simulated)
        y_squared_batch = np.square(simulated).sum(axis=0)

        for index in range(batch_size):
            simulated_null_ll, _, _ = profile.log_likelihood(
                0.0,
                xy_batch[:, index],
                group_y_batch[:, index],
                float(y_squared_batch[index]),
            )
            simulated_alt_ll, _, _, _ = profile.maximize(
                xy_batch[:, index],
                group_y_batch[:, index],
                float(y_squared_batch[index]),
            )
            statistic = max(
                0.0, 2.0 * (simulated_alt_ll - simulated_null_ll)
            )
            null_lrt_values[completed + index] = statistic
            if statistic >= observed_lrt:
                exceedances += 1
        completed += batch_size

    return {
        "country_random_intercept_lrt": observed_lrt,
        "country_random_intercept_bootstrap_p": (exceedances + 1)
        / (BOOTSTRAP_DRAWS + 1),
        "country_random_intercept_exceedances": exceedances,
        "country_random_intercept_bootstrap_draws": BOOTSTRAP_DRAWS,
        "country_random_intercept_bootstrap_seed": BOOTSTRAP_SEED,
        "country_random_intercept_null_boundary_mass": float(
            np.mean(null_lrt_values <= 1e-10)
        ),
        "country_random_intercept_null_lrt_q95": float(
            np.quantile(null_lrt_values, 0.95)
        ),
        "country_random_intercept_null_lrt_q99": float(
            np.quantile(null_lrt_values, 0.99)
        ),
        "profile_tau": tau,
        "profile_country_variance": tau * residual_variance,
        "profile_residual_variance": residual_variance,
        "profile_adjusted_icc": tau / (1.0 + tau),
        "profile_null_llf": null_profile_ll,
        "profile_mixed_llf": alternative_profile_ll,
        "statsmodels_fixed_llf": float(fixed_only.llf),
        "statsmodels_mixed_llf": float(adjusted_mixed.llf),
        "profile_null_llf_abs_difference": null_agreement,
        "profile_mixed_llf_abs_difference": mixed_agreement,
        "random_effect_test": "parametric_bootstrap_lrt_boundary_null",
        "random_effect_null": "country_variance_equals_zero",
    }


def _fit_lme(
    design: pd.DataFrame,
    fixed_effects: list[str],
    effect_levels: dict[str, str],
    bag: str,
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["residual", "country", *fixed_effects]
    fit_df = design[columns].dropna().copy()
    formula = "residual ~ " + " + ".join(fixed_effects)
    intercept_only, intercept_method, intercept_warnings = _fit_mixed_model(
        "residual ~ 1", fit_df
    )
    adjusted, adjusted_method, adjusted_warnings = _fit_mixed_model(
        formula, fit_df
    )
    fixed_only = sm.OLS.from_formula(formula, data=fit_df).fit()
    if list(fixed_only.model.exog_names) != list(adjusted.model.exog_names):
        raise RuntimeError("Fixed-only and mixed-model design matrices differ")

    confidence_interval = adjusted.conf_int()
    estimate_rows = []
    for term in ["Intercept", *fixed_effects]:
        if term not in adjusted.params.index:
            continue
        estimate_rows.append(
            {
                "bag": bag,
                "objective": objective,
                "term": term,
                "term_level": effect_levels.get(term, "intercept"),
                "coef": float(adjusted.params[term]),
                "se": float(adjusted.bse[term]),
                "ci_lo": float(confidence_interval.loc[term, 0]),
                "ci_hi": float(confidence_interval.loc[term, 1]),
                "p_value": float(adjusted.pvalues[term]),
                "n_obs": int(fit_df.shape[0]),
                "n_countries": int(fit_df["country"].nunique()),
                "converged": bool(adjusted.converged),
                "optimizer": adjusted_method,
                "optimizer_warnings": adjusted_warnings,
            }
        )

    unadjusted_country_variance = float(intercept_only.cov_re.iloc[0, 0])
    unadjusted_residual_variance = float(intercept_only.scale)
    adjusted_country_variance = float(adjusted.cov_re.iloc[0, 0])
    adjusted_residual_variance = float(adjusted.scale)
    unadjusted_icc = unadjusted_country_variance / (
        unadjusted_country_variance + unadjusted_residual_variance
    )
    adjusted_icc = adjusted_country_variance / (
        adjusted_country_variance + adjusted_residual_variance
    )
    fixed_effect_lrt = max(
        0.0, 2.0 * (float(adjusted.llf) - float(intercept_only.llf))
    )
    random_effect_test = _bootstrap_country_random_intercept(
        fit_df,
        fixed_only,
        adjusted,
        bag=bag,
        objective=objective,
    )
    variance_row = {
        "bag": bag,
        "objective": objective,
        "n_obs": int(fit_df.shape[0]),
        "n_countries": int(fit_df["country"].nunique()),
        "fixed_effects": "|".join(fixed_effects),
        "unadjusted_country_variance": unadjusted_country_variance,
        "unadjusted_residual_variance": unadjusted_residual_variance,
        "unadjusted_icc": unadjusted_icc,
        "adjusted_country_variance": adjusted_country_variance,
        "adjusted_residual_variance": adjusted_residual_variance,
        "adjusted_icc": adjusted_icc,
        "country_variance_reduction": (
            1.0 - adjusted_country_variance / unadjusted_country_variance
            if unadjusted_country_variance > 0
            else np.nan
        ),
        "fixed_effect_lrt_statistic": fixed_effect_lrt,
        "fixed_effect_lrt_df": len(fixed_effects),
        "fixed_effect_lrt_p": float(
            stats.chi2.sf(fixed_effect_lrt, len(fixed_effects))
        ),
        "unadjusted_converged": bool(intercept_only.converged),
        "adjusted_converged": bool(adjusted.converged),
        "unadjusted_optimizer": intercept_method,
        "adjusted_optimizer": adjusted_method,
        "unadjusted_optimizer_warnings": intercept_warnings,
        "adjusted_optimizer_warnings": adjusted_warnings,
        "residual_source_dir": str(RESIDUALS_DIR.resolve()),
        **random_effect_test,
    }
    return pd.DataFrame(estimate_rows), pd.DataFrame([variance_row])


def _draw_residual_distribution(
    designs: dict[tuple[str, str], pd.DataFrame],
    bag: str,
    axis,
    row_letter: str,
    show_legend: bool,
) -> None:
    for objective, color in zip(OBJECTIVES, (SYN_COLOR, RED_COLOR)):
        residual = pd.to_numeric(
            designs[(bag, objective)]["residual"], errors="coerce"
        ).dropna()
        axis.hist(
            residual,
            bins=40,
            histtype="step",
            color=color,
            density=True,
            label=_PAPER_CFG.objective_metadata[objective]["arm_label"],
        )
    axis.axvline(0, color=NULL_COLOR, ls=":", lw=0.8)
    # Row label on the row's first column (the plot_fig2_grid_v3 convention).
    axis.set_title(
        f"{row_letter}. {BAG_LABELS[bag]} — residual distribution",
        fontsize=10, loc="left",
    )
    axis.set_xlabel("Signed BAG bias (predicted - observed)")
    axis.set_ylabel("Density")
    if show_legend:
        # Upper-left: the right edge abuts the forest panel's term labels.
        axis.legend(fontsize=7, loc="upper left")
    style_axis(axis)


def _draw_forest(
    estimates: pd.DataFrame,
    fixed_effects: list[str],
    bag: str,
    axis,
) -> None:
    offsets = np.linspace(0.15, -0.15, num=len(OBJECTIVES))
    for objective, color, offset in zip(
        OBJECTIVES,
        (SYN_COLOR, RED_COLOR),
        offsets,
    ):
        subset = estimates[
            (estimates["bag"] == bag)
            & (estimates["objective"] == objective)
        ]
        for index, term in enumerate(fixed_effects):
            row = subset[subset["term"] == term]
            if row.empty:
                continue
            y_position = index + offset
            coefficient = float(row["coef"].iloc[0])
            lower = float(row["ci_lo"].iloc[0])
            upper = float(row["ci_hi"].iloc[0])
            axis.plot(
                [lower, upper],
                [y_position, y_position],
                color=color,
                lw=1.5,
            )
            axis.scatter(
                [coefficient],
                [y_position],
                color=color,
                s=18,
                zorder=3,
            )
    axis.axvline(0, color=NULL_COLOR, ls=":", lw=0.8)
    axis.set_yticks(list(range(len(fixed_effects))))
    axis.set_yticklabels(fixed_effects, fontsize=8)
    axis.set_title("Confound coefficients", fontsize=10, loc="left")
    axis.set_xlabel("LME coefficient (95% CI)")
    style_axis(axis)


def _plot_residual_confounds(
    designs: dict[tuple[str, str], pd.DataFrame],
    estimates: pd.DataFrame,
    fixed_effects: list[str],
    output_directory: Path,
) -> None:
    """One figure with a row per BAG, structural first. No suptitle.

    Each row holds that BAG's residual distribution and its confound forest, so
    a row is a single modality end to end.
    """
    figure, axes = plt.subplots(
        len(BAGS), 2,
        figsize=(12.0, 4.2 * len(BAGS)),
        squeeze=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.26},
    )
    panels: list[Panel] = []
    for i, bag in enumerate(BAGS):
        _draw_residual_distribution(designs, bag, axes[i][0], ROW_LETTERS[i], show_legend=(i == 0))
        _draw_forest(estimates, fixed_effects, bag, axes[i][1])

        residual_frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "objective": objective,
                        "residual": pd.to_numeric(
                            designs[(bag, objective)]["residual"], errors="coerce"
                        ).dropna(),
                    }
                )
                for objective in OBJECTIVES
            ],
            ignore_index=True,
        )
        panels.append(
            Panel(
                panel_id=f"{ROW_LETTERS[i]}1_{BAG_SHORT[bag]}_residuals",
                frame=residual_frame,
                description=(
                    f"Subject-level signed {BAG_LABELS[bag]} bias (predicted - observed) "
                    f"for the best synergy-arm and redundancy-arm models."
                ),
                columns={
                    "objective": "o_min = synergy arm, o_max = redundancy arm",
                    "residual": "signed BAG bias in years (predicted - observed)",
                },
                notes="Histogram is plotted as a density over 40 bins.",
            )
        )

        forest_frame = estimates[
            estimates["bag"].eq(bag) & estimates["term"].isin(fixed_effects)
        ][
            [c for c in ["objective", "term", "coef", "se", "ci_lo", "ci_hi", "p_value",
                         "n_obs", "n_countries"] if c in estimates.columns]
        ].reset_index(drop=True)
        panels.append(
            Panel(
                panel_id=f"{ROW_LETTERS[i]}2_{BAG_SHORT[bag]}_confounds",
                frame=forest_frame,
                description=(
                    f"Country-random-intercept mixed-model fixed effects on subject-level "
                    f"signed {BAG_LABELS[bag]} bias."
                ),
                columns={
                    "objective": "o_min = synergy arm, o_max = redundancy arm",
                    "term": "fixed effect (continuous terms standardized)",
                    "coef": "coefficient (plotted point)",
                    "se": "standard error",
                    "ci_lo": "lower bound of the 95% CI (plotted whisker)",
                    "ci_hi": "upper bound of the 95% CI (plotted whisker)",
                    "p_value": "exact p value as fitted; not rounded or capped",
                    "n_obs": "subjects entering the model",
                    "n_countries": "countries entering the random intercept",
                },
                test="Two-sided Wald test on the mixed-model fixed effect.",
                notes="Sex uses female and diagnosis uses HC as the reference level.",
            )
        )

    save_figure(figure, "residual_confounds", output_directory)
    plt.close(figure)
    write_source_data("residual_confounds", panels, output_directory)


def render_completed_parts(parts_root: Path, output_directory: Path) -> None:
    """Render both BAG rows from already completed per-BAG source-data parts."""
    source_root = parts_root / "source_data"
    designs: dict[tuple[str, str], pd.DataFrame] = {}
    estimates = []
    fixed_effects: list[str] | None = None
    for bag in BAGS:
        short = BAG_SHORT[bag]
        residual_paths = sorted(source_root.glob(f"*_{short}_residuals.csv"))
        confound_paths = sorted(source_root.glob(f"*_{short}_confounds.csv"))
        if len(residual_paths) != 1 or len(confound_paths) != 1:
            raise FileNotFoundError(
                f"Expected one completed residual/confound source pair for {bag} in {source_root}"
            )
        residual = pd.read_csv(residual_paths[0])
        confounds = pd.read_csv(confound_paths[0])
        confounds["bag"] = bag
        estimates.append(confounds)
        terms = confounds["term"].drop_duplicates().astype(str).tolist()
        if fixed_effects is None:
            fixed_effects = terms
        elif terms != fixed_effects:
            raise ValueError(f"Completed residual-confound terms differ for {bag}")
        for objective in OBJECTIVES:
            designs[(bag, objective)] = residual[residual["objective"].eq(objective)].copy()
    if fixed_effects is None:
        raise RuntimeError("No completed residual-confound source data found")
    _plot_residual_confounds(
        designs,
        pd.concat(estimates, ignore_index=True),
        fixed_effects,
        output_directory,
    )


def main() -> None:
    completed_parts = os.environ.get("RESIDUAL_CONFOUNDS_COMPLETED_PARTS", "").strip()
    completed_output = os.environ.get("RESIDUAL_CONFOUNDS_COMPLETED_OUTPUT", "").strip()
    if completed_parts or completed_output:
        if not completed_parts or not completed_output:
            raise ValueError(
                "RESIDUAL_CONFOUNDS_COMPLETED_PARTS and "
                "RESIDUAL_CONFOUNDS_COMPLETED_OUTPUT must be set together"
            )
        render_completed_parts(Path(completed_parts), Path(completed_output))
        return
    cfg = _SENSITIVITY_CFG
    local_root = repo_sensitivity_root(cfg) / "residual_confounds"
    local_root.mkdir(parents=True, exist_ok=True)
    figures_root = repo_sensitivity_figures_root(cfg, "residual_confounds")

    raw, _domains, _feature_names, _domain_map = load_raw_and_domains()
    raw["N_MEGA"] = raw["N_MEGA"].astype(str).str.strip()

    designs: dict[tuple[str, str], pd.DataFrame] = {}
    estimates_all = []
    variance_all = []
    shared_fixed_effects: list[str] | None = None
    for bag in BAGS:
        for objective in OBJECTIVES:
            design, fixed_effects, effect_levels = _build_design(
                bag,
                objective,
                raw,
            )
            if shared_fixed_effects is None:
                shared_fixed_effects = fixed_effects
            elif fixed_effects != shared_fixed_effects:
                raise RuntimeError("Residual-confound fixed effects differ across fits")
            designs[(bag, objective)] = design
            estimates, variance = _fit_lme(
                design,
                fixed_effects,
                effect_levels,
                bag,
                objective,
            )
            estimates_all.append(estimates)
            variance_all.append(variance)
            suffix = OBJ_SUFFIX[objective]
            estimate_path = local_root / f"{bag}_{suffix}_lme_estimates.csv"
            estimates.to_csv(estimate_path, index=False)
            variance.to_csv(
                local_root / f"{bag}_{suffix}_lme_variance.csv",
                index=False,
            )
            print(
                f"=== residual-confounds | bag={bag} objective={objective} "
                f"-> {estimate_path} ===",
                flush=True,
            )
            print(estimates.to_string(index=False), flush=True)
            print(variance.to_string(index=False), flush=True)

    if shared_fixed_effects is None:
        raise RuntimeError("No residual-confound models were fitted")
    combined_estimates = pd.concat(estimates_all, ignore_index=True)
    combined_variance = pd.concat(variance_all, ignore_index=True)
    combined_estimates.to_csv(local_root / "lme_estimates_all.csv", index=False)
    combined_variance.to_csv(local_root / "lme_variance_all.csv", index=False)
    random_effect_columns = [
        "bag",
        "objective",
        "n_obs",
        "n_countries",
        "fixed_effects",
        "unadjusted_icc",
        "adjusted_icc",
        "country_variance_reduction",
        "country_random_intercept_lrt",
        "country_random_intercept_bootstrap_p",
        "country_random_intercept_exceedances",
        "country_random_intercept_bootstrap_draws",
        "country_random_intercept_bootstrap_seed",
        "country_random_intercept_null_boundary_mass",
        "random_effect_test",
        "random_effect_null",
    ]
    combined_variance[random_effect_columns].to_csv(
        local_root / "country_random_intercept_test.csv",
        index=False,
    )

    _plot_residual_confounds(designs, combined_estimates, shared_fixed_effects, figures_root)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "residual_confounds")
    copy_tree_contents(
        figures_root,
        bundle_sensitivity_root(cfg) / "residual_confounds",
    )


if __name__ == "__main__":
    main()
