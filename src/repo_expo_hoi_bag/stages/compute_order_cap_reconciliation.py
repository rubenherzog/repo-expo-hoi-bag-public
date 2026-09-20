#!/usr/bin/env python3
"""compute_order_cap_reconciliation.py
=======================================
Master reconciliation of the main pooled analysis against the full-Fig.2-pool
sensitivity analyses under a **cap on the maximum interaction order**.

Scientific purpose
------------------
High-order O-information sets are fit from only ~205 country-years
The strongest open question is whether
the tree>linear and synergy>redundancy signals survive when high-order candidate
sets are removed. Every candidate is already evaluated at every order/rung on
disk; this script is pure re-aggregation — it filters the existing evaluations to
``order <= cap`` for ``cap in {5, 6, ..., 30}`` and re-derives best-per-rung
for each analysis, the best and top-20 median interaction orders, plus the
diversity×synergy LME betas at each cap.

Analyses included (only those drawing from the full Fig.2 candidate pool, where the
cap actually bites):
  - MAIN pooled (Pooled condition of the normative grid; ``metrics_global_long``).
  - Normative transfer conditions (CN->CN subsumes the old HC-only analysis,
    AD->AD, AD->FTD, FTD->FTD, FTD->AD, AD+FTD->AD, AD+FTD->FTD).
  - LORO leave-one-region-out.

``domain_imbalance`` (sizes 3-10) and ``whole_exposome_pca`` (PC1..PC10) are
excluded: they are low-order by construction and not subject to high-order overfit.

Outputs (lightweight, local only)
---------------------------------
  outputs/sensitivity/order_cap/
    order_cap_summary_by_cap_bag_rung.csv   tidy cap x bag x analysis x rung summary
    r2_by_order.csv                 running-best R2 vs order (overfit check, §3.1)
    diversity_r2_betas_by_cap.csv   r2 ~ shannon_h * is_synergistic + order betas
  outputs/figures/sensitivity/order_cap/
    fig_s10_set_size_sensitivity.{pdf,svg,png}
    source_data/fig_s10_set_size_sensitivity_source_data.xlsx

Usage
-----
    PYTHONPATH=. python -m scripts.compute_order_cap_reconciliation
    ORDER_CAPS=30,20,10 python -m scripts.compute_order_cap_reconciliation
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKOUT_ROOT = Path(
    os.environ.get("REPRO_DATA_ROOT", "").strip()
    or os.environ.get("REPO_CHECKOUT_ROOT", str(REPO_ROOT))
).resolve()

# Shared publication style (Arial, fonttype 42, spines, colours) — single source of
# truth, must be imported before pyplot so rcParams apply.
from scripts.fig_style import (  # noqa: E402
    BAG_LABELS,
    GRID_FS,
    GRID_FS_TK,
    NULL_COLOR,
    RED_COLOR,
    RUNG_COLOR,
    SYN_COLOR,
    save_figure as save_fig,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the canonical normative loaders + diversity beta machinery so the cap
# reconciliation stays numerically identical to the published figures.
from scripts.plot_normative_transfer_grid import (  # noqa: E402
    BAGS,
    CONDITION_ORDER,
    RUNG_ORDER,
    V3_OUTPUT_ROOT,
    VARIANT,
    load_plot_data,
)
from scripts.plot_normative_diversity_r2 import (  # noqa: E402
    add_domain_diversity,
)

# Route to the parallel dedup/ subtree when the dedup re-analysis drives this stage
# (SENSITIVITY_DEDUP=1), mirroring sensitivity_common.repo_sensitivity_{,figures_}root.
# Without this the dedup run would overwrite the canonical order_cap outputs.
_DEDUP = os.environ.get("SENSITIVITY_DEDUP", "").strip().lower() in {"1", "true", "yes", "y", "on"}
_DEDUP_NS = os.environ.get("DEDUP_NS", "").strip() or "dedup"
# Optional negative-O-info synergy criterion: an o_min candidate counts as
# synergistic only if its evaluated O-info (score) < 0. o_max untouched.
_SYN_OINFO_NEGATIVE = os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in {
    "1", "true", "yes", "y", "on",
}
if _DEDUP:
    OUT_DIR = CHECKOUT_ROOT / "outputs" / "sensitivity" / _DEDUP_NS / "order_cap"
    FIGURES_DIR = CHECKOUT_ROOT / "outputs" / "figures" / _DEDUP_NS / "sensitivity" / "order_cap"
else:
    OUT_DIR = CHECKOUT_ROOT / "outputs" / "sensitivity" / "order_cap"
    FIGURES_DIR = CHECKOUT_ROOT / "outputs" / "figures" / "sensitivity" / "order_cap"
SENS_ROOT = REPO_ROOT / "outputs" / "sensitivity"

# Fine grid around the synergy boundary (negative O-info in o_min ends at order 22)
# so the order-cap figures resolve where syn/red and the diversity betas change.
ORDER_CAPS = [int(c) for c in os.environ.get("ORDER_CAPS", ",".join(str(c) for c in range(5, 31))).split(",") if c.strip()]

# Variant A sample definition (for target-SD context, §3.5).
RAW_PATH = Path(
    os.environ.get(
        "V3_RAW_PATH",
        str(REPO_ROOT / "data" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"),
    )
)
EXCLUDE_COUNTRIES = set(
    c.strip() for c in os.environ.get("V3_EXCLUDE_COUNTRIES", "France,Italy,Egypt,Greece,Poland").split(",") if c.strip()
)
RETAINED_DX = ["CN", "AD", "FTD"]
BAG_COLUMN = {"functional": "BAG_func", "structural": "BAG_struc"}

# Map a normative condition to the test-diagnosis subset whose BAG variance is the
# relevant denominator for that condition's R² (§3.5).
CONDITION_TEST_DX = {
    "Pooled": RETAINED_DX,
    "CN->CN": ["CN"],
    "AD->AD": ["AD"],
    "AD->FTD": ["FTD"],
    "FTD->FTD": ["FTD"],
    "FTD->AD": ["AD"],
    "AD+FTD->AD": ["AD"],
    "AD+FTD->FTD": ["FTD"],
    "LORO-region": RETAINED_DX,
}

ANALYSIS_ORDER = CONDITION_ORDER + ["LORO-region"]


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_loro(bag: str) -> pd.DataFrame:
    """Leave-one-region-out evaluations, shaped like ``load_plot_data`` output."""
    path = SENS_ROOT / "country_region" / bag / "eval_loro" / f"{bag}_global_all_rungs.csv"
    if not path.exists():
        warnings.warn(f"LORO eval not found for {bag}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    baseline = (
        df[df["candidate_id"] == "__baseline__"][["rung_id", "global_oof_r2"]]
        .drop_duplicates()
        .rename(columns={"global_oof_r2": "baseline_r2"})
    )
    models = df[
        df["rung_id"].isin(RUNG_ORDER)
        & df["objective"].isin(["o_min", "o_max"])
        & (df["candidate_id"] != "__baseline__")
    ].copy()
    models = models.merge(baseline, on="rung_id", how="left")
    out = pd.DataFrame(
        {
            "bag": bag,
            "condition": "LORO-region",
            "rung_id": models["rung_id"],
            "objective": models["objective"],
            "order": pd.to_numeric(models["order"], errors="coerce"),
            "candidate_id": models["candidate_id"],
            "predictors_identity": models.get("predictors_identity", ""),
            "r2": pd.to_numeric(models["global_oof_r2"], errors="coerce"),
            "baseline_r2": pd.to_numeric(models["baseline_r2"], errors="coerce"),
            "oinfo": pd.to_numeric(models.get("score", np.nan), errors="coerce"),
            "source": "loro_region",
        }
    )
    return out


def assemble(bag: str) -> pd.DataFrame:
    """Unified candidate-level table: MAIN(=Pooled) + normative conditions + LORO.

    ``load_plot_data`` applies the default ``NORM_TRANSFER_ORDER_MAX`` (30) filter,
    which is exactly our largest cap, so all caps are obtained by sub-filtering.
    """
    norm = load_plot_data(bag)
    keep = ["bag", "condition", "rung_id", "objective", "order", "candidate_id",
            "predictors_identity", "r2", "baseline_r2", "oinfo"]
    norm = norm[[c for c in keep if c in norm.columns]].copy()
    norm["condition"] = norm["condition"].astype(str)
    parts = [norm, _load_loro(bag)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    # Optional negative-O-info synergy criterion: drop o_min candidates whose
    # evaluated O-info (oinfo) is not < 0. Single chokepoint feeding every
    # downstream selection (best-per-rung, top20, diversity betas). o_max kept.
    if _SYN_OINFO_NEGATIVE:
        if "oinfo" not in df.columns:
            raise KeyError("SYN_OINFO_NEGATIVE set but 'oinfo' absent from assembled pool.")
        o = pd.to_numeric(df["oinfo"], errors="coerce")
        df = df[(df["objective"] != "o_min") | (o < 0)].copy()
    return df.dropna(subset=["order", "r2"])


def target_sd_table() -> dict[tuple[str, str], float]:
    """SD of the BAG target per (bag, condition) test sample, variant-A filtered."""
    out: dict[tuple[str, str], float] = {}
    try:
        raw = pd.read_csv(RAW_PATH, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not read cohort for target SD context: {exc}")
        return out
    raw = raw[~raw["country_clean"].isin(EXCLUDE_COUNTRIES)]
    raw = raw[raw["Diagnosis"].isin(RETAINED_DX)]
    for bag, col in BAG_COLUMN.items():
        if col not in raw.columns:
            continue
        for cond, dx in CONDITION_TEST_DX.items():
            vals = pd.to_numeric(raw.loc[raw["Diagnosis"].isin(dx), col], errors="coerce").dropna()
            if len(vals):
                out[(bag, cond)] = float(vals.std(ddof=1))
    return out


# Tidy cap x bag x analysis x rung summary (renamed from reconciliation_long.csv,
# which read as opaque; the new name states the table's index).
RECON_CSV_NAME = "order_cap_summary_by_cap_bag_rung.csv"


def load_existing_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recon_path = OUT_DIR / RECON_CSV_NAME
    order_path = OUT_DIR / "r2_by_order.csv"
    beta_path = OUT_DIR / "diversity_r2_betas_by_cap.csv"
    missing = [path for path in (recon_path, order_path, beta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local sensitivity outputs; run the order-cap stage once before plotting-only refreshes: "
            + ", ".join(str(path) for path in missing)
        )
    return pd.read_csv(recon_path), pd.read_csv(order_path), pd.read_csv(beta_path)


def load_or_build_recon(
    sd_map: dict[tuple[str, str], float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (and persist) all three reconciliation tables from the candidate frames.

    Previously main() built only the cap-summary table and then expected
    r2_by_order.csv / diversity_r2_betas_by_cap.csv to already exist, so the very
    first run always failed. Build and write all three here from the same frames.
    """
    frames = {bag: assemble(bag) for bag in BAGS}
    recon = reconciliation_long(frames, sd_map)
    by_order = r2_by_order(frames)
    betas = diversity_betas_by_cap(frames)
    recon.to_csv(OUT_DIR / RECON_CSV_NAME, index=False)
    by_order.to_csv(OUT_DIR / "r2_by_order.csv", index=False)
    betas.to_csv(OUT_DIR / "diversity_r2_betas_by_cap.csv", index=False)
    return recon, by_order, betas


# --------------------------------------------------------------------------- #
# Reconciliation tables
# --------------------------------------------------------------------------- #
def reconciliation_long(frames: dict[str, pd.DataFrame], sd_map: dict) -> pd.DataFrame:
    rows = []
    for bag, df in frames.items():
        for cap in ORDER_CAPS:
            capped = df[df["order"] <= cap]
            for cond, cdf in capped.groupby("condition", observed=True):
                for rung, rdf in cdf.groupby("rung_id", observed=True):
                    if rdf.empty:
                        continue
                    ordered = rdf.sort_values(
                        by=["r2", "order", "candidate_id"],
                        ascending=[False, True, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    top20 = ordered.head(20).copy()
                    syn_rdf = rdf[rdf["objective"] == "o_min"].copy()
                    red_rdf = rdf[rdf["objective"] == "o_max"].copy()
                    syn_ordered = syn_rdf.sort_values(
                        by=["r2", "order", "candidate_id"],
                        ascending=[False, True, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    red_ordered = red_rdf.sort_values(
                        by=["r2", "order", "candidate_id"],
                        ascending=[False, True, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    best_idx = rdf["r2"].idxmax()
                    best_syn_idx = syn_rdf["r2"].idxmax() if len(syn_rdf) else None
                    best_red_idx = red_rdf["r2"].idxmax() if len(red_rdf) else None
                    top20_syn = syn_ordered.head(20)
                    top20_red = red_ordered.head(20)
                    syn = rdf[rdf["objective"] == "o_min"]["r2"]
                    red = rdf[rdf["objective"] == "o_max"]["r2"]
                    baseline = pd.to_numeric(rdf.get("baseline_r2"), errors="coerce").dropna()
                    rows.append(
                        {
                            "bag": bag,
                            "analysis": cond,
                            "order_cap": cap,
                            "rung_id": str(rung),
                            "best_r2": float(rdf.loc[best_idx, "r2"]),
                            "best_order": int(rdf.loc[best_idx, "order"]),
                            "best_objective": str(rdf.loc[best_idx, "objective"]),
                            "best_predictors_identity": str(rdf.loc[best_idx, "predictors_identity"]),
                            "best_syn_order": int(syn_rdf.loc[best_syn_idx, "order"]) if best_syn_idx is not None else np.nan,
                            "best_red_order": int(red_rdf.loc[best_red_idx, "order"]) if best_red_idx is not None else np.nan,
                            "best_syn_r2": float(syn.max()) if len(syn) else np.nan,
                            "best_red_r2": float(red.max()) if len(red) else np.nan,
                            "baseline_r2": float(baseline.median()) if len(baseline) else np.nan,
                            "n_candidates": int(len(rdf)),
                            "top20_n_candidates": int(len(top20)),
                            "top20_median_r2": float(np.nanmedian(top20["r2"])) if len(top20) else np.nan,
                            "top20_median_order": float(np.nanmedian(top20["order"])) if len(top20) else np.nan,
                            "top20_median_syn_r2": float(np.nanmedian(top20_syn["r2"])) if len(top20_syn) else np.nan,
                            "top20_median_red_r2": float(np.nanmedian(top20_red["r2"])) if len(top20_red) else np.nan,
                            "top20_median_syn_order": float(np.nanmedian(top20_syn["order"])) if len(top20_syn) else np.nan,
                            "top20_median_red_order": float(np.nanmedian(top20_red["order"])) if len(top20_red) else np.nan,
                            "target_sd": sd_map.get((bag, str(cond)), np.nan),
                        }
                    )
    out = pd.DataFrame(rows)
    out["syn_minus_red"] = out["best_syn_r2"] - out["best_red_r2"]
    out["best_minus_baseline"] = out["best_r2"] - out["baseline_r2"]
    cat = pd.Categorical(out["analysis"], categories=ANALYSIS_ORDER, ordered=True)
    out = out.assign(_a=cat).sort_values(["bag", "_a", "order_cap", "rung_id"]).drop(columns="_a")
    return out.reset_index(drop=True)


def r2_by_order(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Running-best R2 as a function of the order cap (overfit diagnostic, §3.1)."""
    rows = []
    for bag, df in frames.items():
        for cond, cdf in df.groupby("condition", observed=True):
            for rung, rdf in cdf.groupby("rung_id", observed=True):
                for obj, odf in rdf.groupby("objective", observed=True):
                    prev = np.nan
                    for k in range(5, int(odf["order"].max()) + 1):
                        sub = odf[odf["order"] <= k]["r2"]
                        if sub.empty:
                            continue
                        running = float(sub.max())
                        rows.append(
                            {
                                "bag": bag,
                                "analysis": str(cond),
                                "rung_id": str(rung),
                                "objective": str(obj),
                                "order_cap": k,
                                "running_best_r2": running,
                                "marginal_gain": (running - prev) if np.isfinite(prev) else 0.0,
                            }
                        )
                        prev = running
    return pd.DataFrame(rows)


# Inference for the per-cap diversity interaction matches the main diversity
# regression: the candidates within a cap overlap and share their evaluation sample,
# so the classical OLS p-value is anti-conservative. We therefore additionally report
# (a) an order-clustered robust p-value and (b) a permutation p-value that shuffles the
# synergistic/redundant label within order. The point estimates are unchanged.
_ORDER_CAP_PERM_N = int(os.environ.get("ORDER_CAP_DIVERSITY_PERM_N", "2000"))
_ORDER_CAP_PERM_SEED = int(os.environ.get("ORDER_CAP_DIVERSITY_PERM_SEED", "20260304"))


def _perm_diversity_interaction_p(fit: pd.DataFrame, n_perm: int, seed: int):
    import statsmodels.formula.api as smf
    inter = "shannon_h:is_synergistic"
    formula = "r2 ~ shannon_h * is_synergistic + order"
    obs = smf.ols(formula, data=fit).fit().params.get(inter, np.nan)
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    groups = [np.where(fit["order"].to_numpy() == o)[0] for o in fit["order"].unique()]
    base = fit.reset_index(drop=True).copy()
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        lab = base["is_synergistic"].to_numpy().copy()
        for pos in groups:
            v = lab[pos]
            rng.shuffle(v)
            lab[pos] = v
        base["is_synergistic"] = lab
        try:
            null[i] = smf.ols(formula, data=base).fit().params.get(inter, np.nan)
        except Exception:
            null[i] = np.nan
    valid = null[np.isfinite(null)]
    if valid.size == 0:
        return np.nan
    return (1 + int(np.sum(np.abs(valid) >= abs(obs)))) / (valid.size + 1)


def _fit_diversity_beta(d: pd.DataFrame) -> dict[str, object]:
    """Fit r2 ~ shannon_h * is_synergistic + order on one (bag,condition,rung,cap)
    slice and return the synergy/redundancy diversity slopes.

    The interaction p-value is reported three ways: classical OLS (anti-conservative,
    kept for reference), order-clustered robust SE, and a within-order label
    permutation. Significance should be read off the robust/permutation columns."""
    import statsmodels.formula.api as smf
    fit = d[["r2", "shannon_h", "is_synergistic", "order"]].apply(pd.to_numeric, errors="coerce").astype(float)
    fit = fit.replace([np.inf, -np.inf], np.nan).dropna()
    out = {"n_obs": int(len(fit)), "beta_h_red": np.nan, "beta_h_syn": np.nan,
           "beta_interaction": np.nan, "p_h_syn_interaction": np.nan,
           "p_h_syn_interaction_cluster_order": np.nan,
           "p_h_syn_interaction_perm": np.nan, "model_r2": np.nan}
    if len(fit) < 12 or fit["shannon_h"].nunique() < 2 or fit["is_synergistic"].nunique() < 2:
        return out
    try:
        m = smf.ols("r2 ~ shannon_h * is_synergistic + order", data=fit).fit()
    except Exception:
        return out
    bH = float(m.params.get("shannon_h", np.nan))
    bint = float(m.params.get("shannon_h:is_synergistic", np.nan))
    # order-clustered robust SE (needs >1 order cluster)
    p_cluster = np.nan
    if fit["order"].nunique() > 1:
        try:
            mcl = smf.ols("r2 ~ shannon_h * is_synergistic + order", data=fit).fit(
                cov_type="cluster", cov_kwds={"groups": fit["order"]})
            p_cluster = float(mcl.pvalues.get("shannon_h:is_synergistic", np.nan))
        except Exception:
            p_cluster = np.nan
    p_perm = _perm_diversity_interaction_p(fit, _ORDER_CAP_PERM_N, _ORDER_CAP_PERM_SEED)
    out.update({"beta_h_red": bH, "beta_h_syn": bH + bint, "beta_interaction": bint,
                "p_h_syn_interaction": float(m.pvalues.get("shannon_h:is_synergistic", np.nan)),
                "p_h_syn_interaction_cluster_order": p_cluster,
                "p_h_syn_interaction_perm": p_perm,
                "model_r2": float(m.rsquared)})
    return out


def diversity_betas_by_cap(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-cap r2 ~ shannon_h * is_synergistic + order betas, fit **separately per
    rung** (no best-rung selection). Selecting the best rung per cap swaps the whole
    candidate cloud at near-ties and produces spurious sign flips in the slopes; a
    fixed rung makes the cap comparison apples-to-apples (see sec16)."""
    # Normative grid conditions (incl. Pooled) carry predictors_identity; LORO excluded.
    base = pd.concat(
        [df[df["condition"].isin(CONDITION_ORDER)].assign(bag=bag) for bag, df in frames.items()],
        ignore_index=True,
    )
    enriched = add_domain_diversity(base)
    r2 = pd.to_numeric(enriched["r2"], errors="coerce")
    enriched = enriched[np.isfinite(r2)].copy()

    # Enumerate (cap, bag, condition, rung) slices, then fit in parallel: each slice
    # runs an independent within-order label permutation (the heavy part), so the
    # work is embarrassingly parallel. n_jobs from ORDER_CAP_N_JOBS (default: all
    # cores). BLAS stays single-threaded per worker (set by the caller's env).
    slices = []
    for cap in ORDER_CAPS:
        capped = enriched[enriched["order"] <= cap]
        for (bag, cond, rung), d in capped.groupby(["bag", "condition", "rung_id"], observed=True):
            slices.append((cap, str(bag), str(cond), str(rung), d.copy()))

    n_jobs = int(os.environ.get("ORDER_CAP_N_JOBS", str(os.cpu_count() or 1)))

    def _one(cap, bag, cond, rung, d):
        return {"order_cap": cap, "bag": bag, "condition": cond,
                "rung_id": rung, **_fit_diversity_beta(d)}

    if n_jobs == 1 or len(slices) <= 1:
        rows = [_one(*s) for s in slices]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)(*s) for s in slices
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures (style + colours from scripts.fig_style)
# --------------------------------------------------------------------------- #
def _set_native_x_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))


def _row_limits(
    frames: list[pd.DataFrame],
    value_cols: str | Sequence[str],
    pad: float = 0.03,
    min_extra_frac: float = 0.01,
) -> tuple[float, float]:
    if isinstance(value_cols, str):
        cols = (value_cols,)
    else:
        cols = tuple(value_cols)
    vals: list[float] = []
    for frame in frames:
        for value_col in cols:
            if value_col not in frame.columns:
                continue
            series = pd.to_numeric(frame[value_col], errors="coerce").dropna()
            vals.extend(series.astype(float).tolist())
    if not vals:
        return (0.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    if span <= 0:
        span = max(1.0, abs(hi) * 0.1)
    extra = max(span * min_extra_frac, 1e-6)
    return lo - span * pad, hi + span * pad + extra


def _rung_legend(fig: plt.Figure, rung_handles: list[mlines.Line2D]) -> None:
    fig.legend(
        handles=rung_handles,
        fontsize=GRID_FS_TK - 2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(rung_handles),
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _combined_legend(
    fig: plt.Figure,
    rung_handles: list[mlines.Line2D],
    obj_handles: list[mlines.Line2D],
) -> None:
    handles = rung_handles + obj_handles
    fig.legend(
        handles=handles,
        fontsize=GRID_FS_TK - 2,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.505),
        ncol=len(handles),
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def plot_dual_metric_by_rung(
    recon: pd.DataFrame,
    syn_col: str,
    red_col: str,
    ylabel: str,
    outstem: str,
    add_identity_line: bool = False,
) -> None:
    bag_order = [bag for bag in ["structural", "functional"] if bag in recon["bag"].unique()]
    if not bag_order:
        return
    analyses = [a for a in ANALYSIS_ORDER if a in recon["analysis"].unique()]
    ncol = len(analyses)
    nrow = len(bag_order)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.7 * nrow), squeeze=False)
    row_limits: list[tuple[float, float]] = []
    for bag in bag_order:
        sub = recon[recon["bag"] == bag]
        row_limits.append(_row_limits([sub], (syn_col, red_col), pad=0.03))
        for i, analysis in enumerate(analyses):
            ax = axes[bag_order.index(bag)][i]
            adf = sub[sub["analysis"] == analysis]
            for rung in RUNG_ORDER:
                r = adf[adf["rung_id"] == rung].sort_values("order_cap")
                if r.empty:
                    continue
                syn = r[["order_cap", syn_col]].dropna()
                red = r[["order_cap", red_col]].dropna()
                if not syn.empty:
                    ax.plot(
                        syn["order_cap"],
                        syn[syn_col],
                        ls="-",
                        lw=1.5,
                        ms=4,
                        marker="o",
                        color=RUNG_COLOR.get(rung),
                        markerfacecolor=RUNG_COLOR.get(rung),
                    )
                if not red.empty:
                    ax.plot(
                        red["order_cap"],
                        red[red_col],
                        ls="--",
                        lw=1.3,
                        ms=4,
                        marker="s",
                        color=RUNG_COLOR.get(rung),
                        markerfacecolor="none",
                    )
            ax.axvline(22, ls="--", lw=0.8, color="0.6", zorder=0)  # synergy (neg O-info) boundary
            if add_identity_line:
                ax.plot([5, 30], [5, 30], color="0.35", lw=1.0, ls=":", zorder=0)
            ax.set_title(analysis, fontsize=GRID_FS)
            ax.tick_params(labelsize=GRID_FS_TK)
            _set_native_x_ticks(ax)
            style_axis(ax)
            if i == 0:
                ax.set_ylabel(f"{bag}\n{ylabel}", fontsize=GRID_FS)
            else:
                ax.tick_params(left=False, labelleft=False)
            if bag == bag_order[-1]:
                ax.set_xlabel("max set size", fontsize=GRID_FS)
        for j in range(len(analyses), ncol):
            axes[bag_order.index(bag)][j].axis("off")
    for row_i, (lo, hi) in enumerate(row_limits):
        for ax in axes[row_i, :len(analyses)]:
            ax.set_ylim(lo, hi)
    rung_handles = [mlines.Line2D([], [], color=RUNG_COLOR[r], lw=1.8,
                                  label=lab) for r, lab in zip(RUNG_ORDER, ["OLS", "d1", "d2", "d3"])]
    obj_handles = [
        mlines.Line2D([], [], color="black", marker="o", ls="-", lw=1.6, label="synergistic"),
        mlines.Line2D([], [], color="black", marker="s", markerfacecolor="none", ls="--", lw=1.3, label="redundant"),
    ]
    _combined_legend(fig, rung_handles, obj_handles)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.08, wspace=0.05, hspace=0.4)
    save_fig(fig, outstem, FIGURES_DIR)
    plt.close(fig)


def plot_diversity_betas(betas: pd.DataFrame) -> None:
    """One β(Shannon-H) curve per rung vs order cap, separately for synergistic
    (solid, filled circle) and redundant (dashed, open square) sets. No best-rung
    selection — each rung is shown so cross-cap reading is artefact-free."""
    if betas.empty:
        return
    conds = [c for c in CONDITION_ORDER if c in betas["condition"].unique()]
    bag_order = [bag for bag in ["structural", "functional"] if bag in betas["bag"].unique()]
    fig, axes = plt.subplots(len(bag_order), len(conds), figsize=(3.2 * len(conds), 3.2 * len(bag_order)),
                             squeeze=False, sharex=True)
    for r, bag in enumerate(bag_order):
        for c, cond in enumerate(conds):
            ax = axes[r][c]
            panel_vals: list[float] = []
            for rung in RUNG_ORDER:
                sub = betas[(betas["bag"] == bag) & (betas["condition"] == cond)
                            & (betas["rung_id"] == rung)].sort_values("order_cap")
                if sub.empty:
                    continue
                col = RUNG_COLOR.get(rung)
                syn = pd.to_numeric(sub["beta_h_syn"], errors="coerce")
                red = pd.to_numeric(sub["beta_h_red"], errors="coerce")
                panel_vals.extend(syn.dropna().astype(float).tolist())
                panel_vals.extend(red.dropna().astype(float).tolist())
                ax.plot(sub["order_cap"], sub["beta_h_syn"], ls="-", lw=1.6, marker="o", ms=4,
                        color=col, markerfacecolor=col)
                ax.plot(sub["order_cap"], sub["beta_h_red"], ls="--", lw=1.3, marker="s", ms=4,
                        color=col, markerfacecolor="none")
            if panel_vals:
                lo = float(np.min(np.asarray(panel_vals, dtype=float)))
                hi = float(np.max(np.asarray(panel_vals, dtype=float)))
                span = hi - lo
                if span <= 0:
                    span = max(0.01, abs(hi) * 0.1)
                pad = max(span * 0.05, 0.002)
                lo = min(lo - pad, -pad)
                hi = max(hi + pad, pad)
            else:
                lo, hi = (-0.01, 0.01)
            ax.set_ylim(lo, hi)
            ticks = sorted({lo, 0.0, hi})
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.3f}" for t in ticks])
            ax.axhline(0, ls="-", lw=1.8, color="black", zorder=0)
            ax.axvline(22, ls="--", lw=0.8, color="0.6", zorder=0)
            ax.tick_params(labelsize=GRID_FS_TK)
            _set_native_x_ticks(ax)
            style_axis(ax)
            if r == 0:
                ax.set_title(cond, fontsize=GRID_FS)
            if c == 0:
                ax.set_ylabel(f"{bag}\nβ Shannon-H", fontsize=GRID_FS)
            if r == len(bag_order) - 1:
                ax.set_xlabel("max set size", fontsize=GRID_FS)
    rung_handles = [mlines.Line2D([], [], color=RUNG_COLOR[r], lw=1.8, label=lab)
                    for r, lab in zip(RUNG_ORDER, ["OLS", "d1", "d2", "d3"])]
    obj_handles = [mlines.Line2D([], [], color="black", ls="-", marker="o", lw=1.6, label="synergistic"),
                   mlines.Line2D([], [], color="black", ls="--", marker="s", markerfacecolor="none",
                                 lw=1.3, label="redundant")]
    _combined_legend(fig, rung_handles, obj_handles)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.08, wspace=0.26, hspace=0.4)
    save_fig(fig, "fig_order_cap_diversity_betas", FIGURES_DIR)
    plt.close(fig)


def plot_s10_set_size_sensitivity(recon: pd.DataFrame, betas: pd.DataFrame) -> None:
    """Render the single pooled, depth-3 multipanel used as Supplementary Fig. S10."""
    bags = [bag for bag in ("structural", "functional") if bag in set(recon["bag"])]
    pooled = recon[
        recon["analysis"].eq("Pooled")
        & recon["rung_id"].eq("xgb_tree_d3")
        & recon["bag"].isin(bags)
    ].copy()
    pooled_betas = betas[
        betas["condition"].eq("Pooled")
        & betas["rung_id"].eq("xgb_tree_d3")
        & betas["bag"].isin(bags)
    ].copy()
    if pooled.empty or pooled_betas.empty:
        raise ValueError("Supplementary Fig. S10 requires pooled depth-3 cap summaries and betas")

    figure, axes = plt.subplots(
        len(bags), 3, figsize=(12.6, 6.8), squeeze=False, sharex="col",
        gridspec_kw={"wspace": 0.32, "hspace": 0.30},
    )
    column_titles = (
        "a. Best held-out performance",
        "b. Selected set size",
        "c. Domain-diversity slope",
    )
    panels: list[Panel] = []

    for row_index, bag in enumerate(bags):
        summary = pooled[pooled["bag"].eq(bag)].sort_values("order_cap")
        diversity = pooled_betas[pooled_betas["bag"].eq(bag)].sort_values("order_cap")
        if summary.empty or diversity.empty:
            raise ValueError(f"Supplementary Fig. S10 has no pooled depth-3 data for {bag}")

        performance_axis, size_axis, diversity_axis = axes[row_index]
        performance_axis.plot(
            summary["order_cap"], summary["best_syn_r2"], color=SYN_COLOR,
            marker="o", ms=4.5, lw=2.0, label="Min O-info",
        )
        performance_axis.plot(
            summary["order_cap"], summary["best_red_r2"], color=RED_COLOR,
            marker="o", ms=4.5, lw=2.0, label="Max O-info",
        )
        performance_axis.plot(
            summary["order_cap"], summary["baseline_r2"], color="black",
            lw=1.5, label="Baseline",
        )

        size_axis.plot(
            summary["order_cap"], summary["best_syn_order"], color=SYN_COLOR,
            marker="o", ms=4.5, lw=2.0,
        )
        size_axis.plot(
            summary["order_cap"], summary["best_red_order"], color=RED_COLOR,
            marker="o", ms=4.5, lw=2.0,
        )
        size_axis.plot([5, 30], [5, 30], color=NULL_COLOR, lw=1.0, ls=":", zorder=0)

        diversity_axis.plot(
            diversity["order_cap"], diversity["beta_h_syn"], color=SYN_COLOR,
            marker="o", ms=4.5, lw=2.0,
        )
        diversity_axis.plot(
            diversity["order_cap"], diversity["beta_h_red"], color=RED_COLOR,
            marker="o", ms=4.5, lw=2.0,
        )
        diversity_axis.axhline(0, color="black", lw=1.0, zorder=0)

        for column_index, axis in enumerate(axes[row_index]):
            axis.axvline(21, color=NULL_COLOR, lw=0.9, ls="--", zorder=0)
            axis.set_xlim(5, 30)
            axis.set_xticks([5, 10, 15, 20, 25, 30])
            axis.tick_params(labelsize=GRID_FS_TK - 2)
            style_axis(axis)
            if row_index == 0:
                axis.set_title(column_titles[column_index], loc="left", fontsize=GRID_FS)
            if row_index == len(bags) - 1:
                axis.set_xlabel("Maximum set size", fontsize=GRID_FS)

        performance_axis.set_ylabel(f"{BAG_LABELS[bag]}\nLOCO R²", fontsize=GRID_FS)
        size_axis.set_ylabel("Selected set size", fontsize=GRID_FS)
        diversity_axis.set_ylabel("Diversity slope (R² per bit)", fontsize=GRID_FS)
        size_axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    legend_handles = [
        mlines.Line2D([], [], color=SYN_COLOR, marker="o", lw=2.0, label="Min O-info"),
        mlines.Line2D([], [], color=RED_COLOR, marker="o", lw=2.0, label="Max O-info"),
        mlines.Line2D([], [], color="black", lw=1.5, label="Baseline"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False,
        fontsize=GRID_FS_TK - 1, bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.15)

    stem = "fig_s10_set_size_sensitivity"
    save_fig(figure, stem, FIGURES_DIR)
    plt.close(figure)

    performance_frame = pooled[
        ["bag", "order_cap", "rung_id", "best_syn_r2", "best_red_r2", "baseline_r2"]
    ].sort_values(["bag", "order_cap"]).reset_index(drop=True)
    size_frame = pooled[
        ["bag", "order_cap", "rung_id", "best_syn_order", "best_red_order"]
    ].sort_values(["bag", "order_cap"]).reset_index(drop=True)
    diversity_columns = [
        "bag", "order_cap", "rung_id", "beta_h_syn", "beta_h_red", "beta_interaction",
        "p_h_syn_interaction_cluster_order", "p_h_syn_interaction_perm",
    ]
    diversity_frame = pooled_betas[
        [column for column in diversity_columns if column in pooled_betas.columns]
    ].sort_values(["bag", "order_cap"]).reset_index(drop=True)
    panels.extend(
        [
            Panel(
                panel_id="a_best_r2",
                frame=performance_frame,
                description=("Pooled depth-3 held-out LOCO R² across maximum set sizes, "
                             "structural BAG followed by functional BAG."),
                columns={
                    "bag": "brain-age-gap measure and figure row",
                    "order_cap": "maximum candidate set size",
                    "rung_id": "model level (xgb_tree_d3 = d3)",
                    "best_syn_r2": "best unweighted mean held-out-country R² in the minimum-O-information (synergy) arm",
                    "best_red_r2": "best unweighted mean held-out-country R² in the maximum-O-information (redundancy) arm",
                    "baseline_r2": "covariate-only unweighted mean held-out-country R²",
                },
                notes=("All performance values use the country-balanced estimand from main Fig. 2. "
                       "The dashed vertical line marks cap 21, the last cap before evaluated "
                       "O-information changes sign at set size 22."),
            ),
            Panel(
                panel_id="b_selected_set_size",
                frame=size_frame,
                description="Set size of each pooled depth-3 arm winner across maximum set sizes.",
                columns={
                    "bag": "brain-age-gap measure and figure row",
                    "order_cap": "maximum candidate set size",
                    "rung_id": "model level (xgb_tree_d3 = d3)",
                    "best_syn_order": "set size of the minimum-O-information arm winner",
                    "best_red_order": "set size of the maximum-O-information arm winner",
                },
                notes="The grey identity line is selected set size = maximum allowed set size.",
            ),
            Panel(
                panel_id="c_diversity_slopes",
                frame=diversity_frame,
                description=("Set-size-adjusted pooled depth-3 domain-diversity slopes across "
                             "maximum set sizes."),
                columns={
                    "bag": "brain-age-gap measure and figure row",
                    "order_cap": "maximum candidate set size",
                    "rung_id": "model level (xgb_tree_d3 = d3)",
                    "beta_h_syn": "diversity slope in the minimum-O-information arm (country-balanced R² per bit)",
                    "beta_h_red": "diversity slope in the maximum-O-information arm (country-balanced R² per bit)",
                    "beta_interaction": "minimum-minus-maximum arm slope interaction",
                    "p_h_syn_interaction_cluster_order": "set-size-clustered two-sided interaction p value",
                    "p_h_syn_interaction_perm": "within-order label-permutation p value",
                },
                notes="The horizontal black line marks a zero diversity slope.",
            ),
        ]
    )
    write_source_data(
        stem,
        panels,
        FIGURES_DIR,
        source_paths=[
            str(OUT_DIR / RECON_CSV_NAME),
            str(OUT_DIR / "diversity_r2_betas_by_cap.csv"),
        ],
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sd_map = target_sd_table()
    recon, by_order, betas = load_or_build_recon(sd_map)

    plot_s10_set_size_sensitivity(recon, betas)

    # Console sanity: MAIN pooled best-per-rung at cap 30.
    main30 = recon[(recon["analysis"] == "Pooled") & (recon["order_cap"] == 30)]
    print("\nMAIN pooled best_r2 per rung at cap=30:")
    for bag in BAGS:
        d = main30[main30["bag"] == bag].set_index("rung_id")["best_r2"].reindex(RUNG_ORDER)
        print(f"  {bag}: {d.round(4).to_dict()}")
        d_order = main30[main30["bag"] == bag].set_index("rung_id")["best_order"].reindex(RUNG_ORDER)
        print(f"    best_order: {d_order.to_dict()}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
