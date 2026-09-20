#!/usr/bin/env python3
"""Figure 2 v3 grid: 4 rows × 3 cols.

Row 1: set-size synergy-redundancy balance.
Rows 2-4: BAG-exposome association panels for structural, functional,
and combined BAGs.

The first panel of each row carries the row label.
Rectangles and scatter dots have no edges (Illustrator-friendly).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from scipy.stats import gaussian_kde

import matplotlib
import matplotlib.font_manager as fm
from pathlib import Path as _Path
for _fp in [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/local/share/fonts/Arial.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]:
    if _Path(_fp).exists():
        fm.fontManager.addfont(_fp)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Arimo', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data

GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", "outputs/greedy"))
OUTPUT_ROOT = Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
VARIANT     = "variant_a"

LOCAL_STATS_DIR = OUTPUT_ROOT / VARIANT / "stats"
LOCAL_VARIANT_ROOT = OUTPUT_ROOT / VARIANT
PAPER_FIGURES_DIR  = Path(__file__).resolve().parents[1] / "paper_figures"

# Order cap for the paper figure variants. PAPER_FIG_SUFFIX is appended to
# output stems, PAPER_FIG_OUTDIR overrides the output directory, and
# PAPER_FIG_BAGS restricts the BAG rows.
ORDER_MAX = int(os.environ.get("PAPER_FIG_ORDER_MAX", "30"))
SUFFIX = os.environ.get("PAPER_FIG_SUFFIX", "")
OUT_DIR = Path(os.environ["PAPER_FIG_OUTDIR"]) if os.environ.get("PAPER_FIG_OUTDIR") else PAPER_FIGURES_DIR


def _cap(df):
    return df if ORDER_MAX is None else df[df["order"] <= ORDER_MAX]


# Optional negative-O-info synergy criterion (SYN_OINFO_NEGATIVE=1): an o_min
# candidate counts as synergistic only if its evaluated O-info (thoi_o) < 0.
# Applied to the R²-bearing global metrics that drive best-rung / top-K *model
# selection*; the descriptive O-info-vs-order distributions (panel A) are NOT
# filtered (they exist to show the sign by order). Off by default.
SYN_OINFO_NEGATIVE = os.environ.get("SYN_OINFO_NEGATIVE", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)


def _apply_syn_oinfo_filter(df_global: pd.DataFrame) -> pd.DataFrame:
    """Drop o_min rows whose evaluated O-info (thoi_o) is not < 0, iff the flag
    is on. o_max rows are untouched."""
    if not SYN_OINFO_NEGATIVE:
        return df_global
    if "thoi_o" not in df_global.columns:
        raise KeyError("SYN_OINFO_NEGATIVE set but 'thoi_o' absent from global metrics.")
    o = pd.to_numeric(df_global["thoi_o"], errors="coerce")
    keep = (df_global["objective"] != "o_min") | (o < 0)
    return df_global[keep].copy()

# Mapping rung_id -> single_exposure_eval directory. Each entry is a list of
# candidate paths tried in order; the first that exists is used.
# REPRO_DATA_ROOT covers the dedup bundle layout where evals land under
# $BUNDLE/results/variant_a/single_exposure_eval_<rung_id>.
_REPRO_ROOT = Path(os.environ.get("REPRO_DATA_ROOT", ""))

def _resolve_single_exp_dir(candidates):
    for p in candidates:
        if p and p.exists():
            return p
    return None

SINGLE_EXP_DIRS = {
    rung: _resolve_single_exp_dir(paths)
    for rung, paths in {
        "ols":         [LOCAL_VARIANT_ROOT / "single_exposure_eval_ols",
                        _REPRO_ROOT / "results" / "variant_a" / "single_exposure_eval_ols"],
        "xgb_tree_d1": [LOCAL_VARIANT_ROOT / "single_exposure_eval_d1",
                        _REPRO_ROOT / "results" / "variant_a" / "single_exposure_eval_xgb_tree_d1"],
        "xgb_tree_d2": [LOCAL_VARIANT_ROOT / "single_exposure_eval",
                        _REPRO_ROOT / "results" / "variant_a" / "single_exposure_eval_xgb_tree_d2"],
        "xgb_tree_d3": [LOCAL_VARIANT_ROOT / "single_exposure_eval_d3",
                        _REPRO_ROOT / "results" / "variant_a" / "single_exposure_eval_xgb_tree_d3"],
    }.items()
}

SINGLE_COLOR = "#E07B00"  # orange — best single exposure line

SYN_COLOR  = "#1B6B2E"
RED_COLOR  = "#4B0082"
NULL_COLOR = "#AAAAAA"

RUNG_ORDER  = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
RUNG_LABELS = ["OLS", "d1", "d2", "d3"]

BAGS        = [b for b in os.environ.get("PAPER_FIG_BAGS", "functional,structural,combined").split(",") if b]
BAG_LABELS  = {"functional": "Functional BAG", "structural": "Structural BAG", "combined": "Combined BAG"}

FS    = 24
FS_TK = 22

ROW_LABELS = [
    "a. Exposome synergy-redundancy balance across set sizes",
    "b. Strucural BAG-exposome association",
    "c. Functional BAG-exposome association",
    "d. Combined BAG-exposome association",
]


# ── data helpers ──────────────────────────────────────────────────────────────

def extract_omega(df: pd.DataFrame) -> pd.Series:
    candidates = ["omega", "raw_thoi_o", "thoi_o", "thoi_o_normalized", "o"]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    excluded = {
        "objective", "metric", "direction", "order", "rank", "score", "count",
        "nplet_indices", "domain_indices", "source_objective", "score_mode",
        "weighted_entropy", "n_domains", "max_domain_frac", "dominant_domain_index",
        "thoi_s",
    }
    o_cols = [c for c in df.columns if "o" in c.lower() and c.lower() not in excluded]
    if o_cols:
        return pd.to_numeric(df[o_cols[0]], errors="coerce")
    return pd.to_numeric(df["score"], errors="coerce")


def _best_rung(best_rung_df: pd.DataFrame, bag: str, objective: str) -> str:
    row = best_rung_df[
        (best_rung_df["analysis"] == "A_top_k_per_rung") &
        (best_rung_df["bag"] == bag) &
        (best_rung_df["objective"] == objective)
    ]
    return row["best_rung"].iloc[0]


async def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load parquet {path}: {exc}") from exc


async def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV {path}: {exc}") from exc


async def _noop_df() -> pd.DataFrame:
    """Awaitable empty frame, for inputs that the capped variant recomputes."""
    return pd.DataFrame()


def _null_rung(df_global: pd.DataFrame) -> str:
    """Replicate the null script's _best_primary_rung logic: rung with highest max syn R²."""
    best_rung, best_r2 = "xgb_tree_d2", -np.inf
    for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
        sub = df_global[(df_global["rung_id"] == rung) & (df_global["objective"] == "o_min")]
        r2 = sub["full_r2"].max() if len(sub) else -np.inf
        if r2 > best_r2:
            best_r2, best_rung = r2, rung
    return best_rung


def _load_single_exp_r2(bag: str) -> dict[str, float | None]:
    """Return {rung_id: best_global_oof_r2} from local single_exposure_global.csv files."""
    result = {}
    for rung, d in SINGLE_EXP_DIRS.items():
        if d is None:
            result[rung] = None
            continue
        csv_path = d / bag / "single_exposure_global.csv"
        if not csv_path.exists():
            result[rung] = None
            continue
        df = pd.read_csv(csv_path)
        result[rung] = float(df["global_oof_r2"].max())
    return result


async def load_bag_data(bag: str, df_best_rung: pd.DataFrame, df_greedy: pd.DataFrame) -> dict:
    canonical = (
        OUTPUT_ROOT / VARIANT / "families" / "pooled_oinfo_ladder"
        / "canonical" / "per_experiment" / f"pooled_oinfo_ladder_{bag}"
    )
    df_global = _apply_syn_oinfo_filter(_cap(await load_parquet(canonical / "metrics_global_long.parquet")))
    df_global["cohen_f2"] = df_global["delta_r2_vs_base"] / (1.0 - df_global["full_r2"])
    single_exp_r2 = _load_single_exp_r2(bag)
    single_exp_f2 = {}
    for rung, r2 in single_exp_r2.items():
        if r2 is None:
            single_exp_f2[rung] = None
            continue
        rung_rows = df_global[df_global["rung_id"] == rung]
        if len(rung_rows) == 0:
            single_exp_f2[rung] = None
            continue
        base_r2 = float(rung_rows["base_r2"].iloc[0])
        denom = 1.0 - r2
        single_exp_f2[rung] = (r2 - base_r2) / denom if denom != 0 else None
    return {
        "global": df_global, "best_rung": df_best_rung, "greedy": df_greedy,
        "single_exp_r2": single_exp_r2, "single_exp_f2": single_exp_f2,
    }


def _build_domain_diversity(df_greedy_raw: pd.DataFrame) -> pd.DataFrame:
    """For each candidate compute n_domains and Shannon H (log2) over domain fractions.

    Returns DataFrame with columns: objective, order, n_domains, shannon_h.
    Uses all candidates (all ranks) per order × objective.
    """
    # This stage lives under ``src/repo_expo_hoi_bag/stages``; public inputs
    # remain at the checkout root rather than beside the package.
    repo_root = Path(__file__).resolve().parents[3]
    feat_names = pd.read_csv(repo_root / "data" / "metadata" / "exposome_feature_names.csv")["feature_name"].tolist()
    feat_domains = pd.read_csv(repo_root / "data" / "metadata" / "exposome_feature_domains.csv").set_index("feature_name")["domain"]
    idx_to_domain = {i: feat_domains.get(name, "Unknown") for i, name in enumerate(feat_names)}

    rows = []
    for _, row in df_greedy_raw[df_greedy_raw["objective"].isin(["o_min", "o_max"])].iterrows():
        indices = [int(x) for x in str(row["nplet_indices"]).strip("[]").split(",")]
        domain_list = [idx_to_domain.get(i, "Unknown") for i in indices]
        n = len(domain_list)
        domain_counts = {}
        for d in domain_list:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        fracs = np.array([c / n for c in domain_counts.values()])
        shannon_h = float(-(fracs * np.log2(fracs)).sum())
        rows.append({"objective": row["objective"], "order": int(row["order"]),
                     "n_domains": len(domain_counts), "shannon_h": shannon_h})
    return pd.DataFrame(rows)


# Baseline synergy rates from the full exposome (computed in compute_subcomb_oinfo.py)
SUBCOMB_BASELINE_K3 = 0.3323  # fraction of all C(63,3) triplets with O-info < 0
SUBCOMB_BASELINE_K4 = 0.1901

# Path to the per-candidate enrichment table (frac_neg by order_k, rung, bag).
# Set SUBCOMB_ENRICHMENT_CSV to override. REPO_CHECKOUT_ROOT is required by the
# compatibility runtime, whose working directory is outside the checkout.
CHECKOUT_ROOT = Path(os.environ.get(
    "REPO_CHECKOUT_ROOT",
    Path(__file__).resolve().parents[3],
))
SUBCOMB_ENRICHMENT_CSV = Path(os.environ.get(
    "SUBCOMB_ENRICHMENT_CSV",
    CHECKOUT_ROOT / "outputs/dedup/subcomb_oinfo/subcomb_enrichment.csv",
))


def _load_subcomb_enrichment() -> pd.DataFrame:
    """Load the required per-candidate k=3 synergistic-triplet measurements."""
    if not SUBCOMB_ENRICHMENT_CSV.exists():
        raise FileNotFoundError(
            "Figure 2 requires measured synergistic-triplet content; missing "
            f"{SUBCOMB_ENRICHMENT_CSV}. Refusing to render the obsolete "
            "'Fraction synergistic' objective-proxy panel."
        )
    df = pd.read_csv(SUBCOMB_ENRICHMENT_CSV)
    required = {"candidate_id", "bag", "rung_id", "order_k", "frac_neg"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{SUBCOMB_ENRICHMENT_CSV} lacks required columns: {sorted(missing)}"
        )
    # Keep only k=3 rows; drop duplicates arising from candidates shared across bags
    triplets = df[df["order_k"] == 3][
        ["candidate_id", "bag", "rung_id", "frac_neg"]
    ].copy()
    if triplets.empty:
        raise ValueError(f"{SUBCOMB_ENRICHMENT_CSV} contains no order_k=3 rows")
    return triplets


def _enrichment_per_rung_capped(
    bag_globals: dict,
    subcomb_k3: pd.DataFrame,
) -> pd.DataFrame:
    """Per-candidate frac_neg_k3 data for panel d (strip + boxplot).

    Returns one row per evaluated
    candidate with columns: bag, rung, objective, frac_neg_k3, full_r2, mode.
    Only top-20 syn and top-20 red candidates have sub-combination O-info
    computed; the function restricts to exactly those rows so every returned
    candidate has a real measurement.
    """
    rows = []
    for bag, dfg in bag_globals.items():
        for rung in RUNG_ORDER:
            sub = dfg[dfg["rung_id"] == rung]
            if sub.empty:
                continue

            k3_bag = subcomb_k3[
                (subcomb_k3["bag"] == bag) & (subcomb_k3["rung_id"] == rung)
            ][["candidate_id", "frac_neg"]].drop_duplicates("candidate_id")
            # Inner join: only candidates with actual measurements
            merged = sub.merge(k3_bag, on="candidate_id", how="inner")
            if merged.empty:
                raise ValueError(
                    "Figure 2 has no synergistic-triplet measurements for "
                    f"bag={bag!r}, rung={rung!r}"
                )
            for _, row in merged.iterrows():
                rows.append({
                    "bag": bag, "rung": rung,
                    "objective": row["objective"],
                    "frac_neg_k3": float(row["frac_neg"]),
                    "full_r2": float(row["full_r2"]),
                    "mode": "frac_neg_k3",
                })
    return pd.DataFrame(rows)


async def load_all() -> dict:
    # In capped mode (ORDER_MAX set) enrichment_per_rung is recomputed from the
    # capped per-bag globals below, so the on-disk enrichment_per_rung.csv is
    # never used -- skip the read instead of requiring a file that the capped
    # variant does not produce.
    enr_path = LOCAL_STATS_DIR / "enrichment_per_rung.csv"
    enr_coro = load_csv(enr_path) if (ORDER_MAX is None or enr_path.exists()) else _noop_df()
    df_best_rung, df_greedy_raw, df_enr_rung = await asyncio.gather(
        load_csv(LOCAL_STATS_DIR / "best_rung_selection.csv"),
        load_csv(GREEDY_ROOT / "greedy_topk_by_objective_order.csv"),
        enr_coro,
    )
    df_greedy_raw = _cap(df_greedy_raw)
    df_greedy_raw["omega"] = extract_omega(df_greedy_raw)
    df_domain_div = _build_domain_diversity(df_greedy_raw)
    bag_data = await asyncio.gather(*[load_bag_data(bag, df_best_rung, df_greedy_raw) for bag in BAGS])
    result = {bag: data for bag, data in zip(BAGS, bag_data)}

    # Load the measured sub-combination O-info enrichment. Figure 2 must fail
    # rather than silently substituting the obsolete objective-proxy panel.
    subcomb_k3 = _load_subcomb_enrichment()
    print(f"  Loaded sub-combination enrichment ({len(subcomb_k3)} rows, k=3) — panel d uses frac_neg_k3")

    df_enr_rung = _enrichment_per_rung_capped(
        {bag: result[bag]["global"] for bag in BAGS},
        subcomb_k3=subcomb_k3,
    )
    result["_enrichment_per_rung"] = df_enr_rung
    result["_domain_diversity"] = df_domain_div
    return result




# ── drawing helpers ────────────────────────────────────────────────────────────

def _rung_top50(df_global: pd.DataFrame, rung: str):
    r = df_global[df_global["rung_id"] == rung]
    syn = r[r["objective"] == "o_min"].nlargest(50, "delta_r2_vs_base")
    red = r[r["objective"] == "o_max"].nlargest(50, "delta_r2_vs_base")
    base = r["base_r2"].iloc[0] if len(r) else np.nan
    return syn, red, base


def _draw_split_boxes(ax, df_global: pd.DataFrame, y_col: str,
                      baseline_col: str | None = None,
                      single_exp_r2: dict | None = None) -> list[float]:
    """Strip + boxplot per rung, syn left / red right.

    baseline_col  : solid black line at that column's value per rung.
    single_exp_r2 : solid orange line at best single-exposure value per rung.
    """
    HALF = 0.32        # half-width of each objective strip
    BW   = HALF * 0.5  # box half-width
    rng_jitter = np.random.default_rng(0)
    all_vals = []

    for i, rung in enumerate(RUNG_ORDER):
        syn, red, base = _rung_top50(df_global, rung)

        for vals_df, color, x_center in [
            (syn, SYN_COLOR, i - HALF * 0.5),
            (red, RED_COLOR, i + HALF * 0.5),
        ]:
            vals = vals_df[y_col].replace([np.inf, -np.inf], np.nan).dropna().values
            if not len(vals):
                continue
            all_vals.extend(vals.tolist())

            # jitter strip
            jitter = rng_jitter.uniform(-BW * 0.9, BW * 0.9, size=len(vals))
            ax.scatter(x_center + jitter, vals, c=color, s=54, alpha=0.55,
                       edgecolors="none", zorder=3)

            # boxplot overlay
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            iqr = q75 - q25
            w_lo = max(vals.min(), q25 - 1.5 * iqr)
            w_hi = min(vals.max(), q75 + 1.5 * iqr)
            ax.add_patch(mpatches.Rectangle(
                (x_center - BW, q25), 2 * BW, iqr,
                facecolor=color, alpha=0.40, edgecolor=color, lw=1.2, zorder=4,
            ))
            ax.plot([x_center - BW, x_center + BW], [med, med],
                    color=color, lw=2.0, zorder=5)
            ax.plot([x_center, x_center], [w_lo, q25], color=color, lw=1.0, zorder=4)
            ax.plot([x_center, x_center], [q75, w_hi], color=color, lw=1.0, zorder=4)

        if baseline_col is not None:
            rows = df_global[df_global["rung_id"] == rung]
            if len(rows):
                base_val = rows[baseline_col].iloc[0]
                if np.isfinite(base_val):
                    ax.plot([i - HALF, i + HALF], [base_val, base_val],
                            color="black", lw=2.5, ls="-", zorder=6)
                    all_vals.append(base_val)

        if single_exp_r2 is not None:
            val = single_exp_r2.get(rung)
            if val is not None and np.isfinite(val):
                ax.plot([i - HALF, i + HALF], [val, val],
                        color=SINGLE_COLOR, lw=2.5, ls="-", zorder=7)
                all_vals.append(val)

    ax.set_xticks(range(len(RUNG_LABELS)))
    ax.set_xticklabels(RUNG_LABELS, fontsize=FS_TK)
    ax.set_xlim(-0.5, len(RUNG_ORDER) - 0.5)
    return all_vals


# ── individual panel drawers ───────────────────────────────────────────────────

def draw_panel_a(ax, df_greedy: pd.DataFrame, label: str) -> None:
    df = df_greedy[df_greedy["objective"].isin(["o_min", "o_max"])].copy()
    for obj, color, leg in [
        ("o_min", SYN_COLOR, "Min O-info"),
        ("o_max", RED_COLOR,  "Max O-info"),
    ]:
        grp = df[df["objective"] == obj].groupby("order")["omega"]
        mn, mx = grp.min(), grp.max()
        orders = mn.index.values
        ax.fill_between(orders, mn.values, mx.values, color=color, alpha=0.30, label=leg, zorder=2)
        ax.plot(orders, mn.values, color=color, lw=0.8, alpha=0.6, zorder=3)
        ax.plot(orders, mx.values, color=color, lw=0.8, alpha=0.6, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    all_omega = df["omega"].dropna()
    pad = (all_omega.max() - all_omega.min()) * 0.04
    ax.set_ylim(all_omega.min() - pad, all_omega.max() + pad)
    ax.set_xlim(ORDER_XLIM)
    ax.set_xlabel("Set size", fontsize=FS)
    ax.set_ylabel("O-information", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.legend(fontsize=FS_TK, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_title(label, fontsize=FS, loc="left")


def draw_panel_b(ax, df_global: pd.DataFrame, label: str, show_legend: bool = True,
                 single_exp_r2: dict | None = None) -> None:
    all_vals = _draw_split_boxes(ax, df_global, "full_r2", baseline_col="base_r2",
                                 single_exp_r2=single_exp_r2)
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, hi + pad)
    if show_legend:
        patches = [
            mpatches.Patch(color=SYN_COLOR, alpha=0.55, label="Min O-info"),
            mpatches.Patch(color=RED_COLOR, alpha=0.55, label="Max O-info"),
            mlines.Line2D([0], [0], color="black", lw=2.5, ls="-", label="Baseline"),
            mlines.Line2D([0], [0], color=SINGLE_COLOR, lw=2.5, ls="-", label="Best single"),
        ]
        ax.legend(handles=patches, fontsize=FS_TK, loc="lower right", framealpha=0.7)
    ax.set_ylabel("R² LOCO", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_title(label, fontsize=FS, loc="left")


def draw_panel_c(ax, df_global: pd.DataFrame, label: str,
                 ylim: tuple[float, float] | None = None,
                 single_exp_f2: dict | None = None) -> tuple[float, float]:
    all_vals = _draw_split_boxes(ax, df_global, "order", baseline_col=None,
                                 single_exp_r2=None)
    # Synergy boundary: O-info flips sign at order 22 (last negative order is 21)
    ax.axhline(21, color="black", lw=1.2, ls="--", zorder=4)
    cap = ORDER_MAX
    ax.set_ylim(0, cap + 1)
    ax.set_yticks(np.arange(0, cap + 1, 10))
    ax.set_ylabel("Set size", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_title(label, fontsize=FS, loc="left")
    return ylim


RUNG_COLORS = {
    "ols":          "#666666",
    "xgb_tree_d1":  "#E6AB02",
    "xgb_tree_d2":  "#66A61E",
    "xgb_tree_d3":  "#CC00CC",
}
RUNG_DISPLAY = {
    "ols":          "OLS",
    "xgb_tree_d1":  "d1",
    "xgb_tree_d2":  "d2",
    "xgb_tree_d3":  "d3",
}


def draw_panel_d(ax, df_enr: pd.DataFrame, bag: str, label: str,
                 show_legend: bool = True) -> None:
    """Synergistic triplet content per candidate: strip + boxplot per rung.

    frac_neg_k3 mode (preferred): each dot is one evaluated candidate
    (top-20 syn + top-20 red); x = rung grouped by objective, y = fraction
    of triplet sub-combinations with negative O-information. Baseline line
    at the exposome-wide rate (33.2%).

    Rendering without measured triplet content is intentionally unsupported.
    """
    sub = df_enr[df_enr["bag"] == bag].copy()
    if sub.empty or set(sub["mode"]) != {"frac_neg_k3"}:
        raise ValueError(
            f"Figure 2 requires frac_neg_k3 measurements for bag={bag!r}"
        )

    # ── Strip + boxplot: one dot per evaluated candidate ─────────────
    # Geometry matches _draw_split_boxes exactly (HALF=0.32, BW=0.16)
    HALF = 0.32
    BW   = HALF * 0.5
    OBJ_OFFSET = {"o_min": -HALF * 0.5, "o_max": +HALF * 0.5}
    OBJ_COLOR  = {"o_min": SYN_COLOR, "o_max": RED_COLOR}
    rng_jitter = np.random.default_rng(0)

    all_y = []
    for xi, rung in enumerate(RUNG_ORDER):
        for obj in ("o_min", "o_max"):
            grp = sub[(sub["rung"] == rung) & (sub["objective"] == obj)]["frac_neg_k3"].dropna()
            if grp.empty:
                continue
            x_center = xi + OBJ_OFFSET[obj]
            color = OBJ_COLOR[obj]
            y = grp.values * 100

            jitter = rng_jitter.uniform(-BW * 0.9, BW * 0.9, size=len(y))
            ax.scatter(x_center + jitter, y, c=color, s=54, alpha=0.65,
                       edgecolors="none", zorder=4)

            # boxplot elements: median, IQR box, whiskers (no fliers)
            q25, med, q75 = np.percentile(y, [25, 50, 75])
            iqr = q75 - q25
            w_lo = max(y.min(), q25 - 1.5 * iqr)
            w_hi = min(y.max(), q75 + 1.5 * iqr)
            ax.add_patch(mpatches.Rectangle(
                (x_center - BW, q25), 2 * BW, iqr,
                facecolor=color, alpha=0.35, edgecolor=color, lw=1.2, zorder=5,
            ))
            ax.plot([x_center - BW, x_center + BW], [med, med],
                    color=color, lw=2.0, zorder=6)
            ax.plot([x_center, x_center], [w_lo, q25], color=color, lw=1.0, zorder=5)
            ax.plot([x_center, x_center], [q75, w_hi], color=color, lw=1.0, zorder=5)

            all_y.extend(y.tolist())

    # Baseline reference line (no text label)
    ax.axhline(SUBCOMB_BASELINE_K3 * 100, color="black", lw=1.2, ls="--", zorder=3)

    ax.set_xticks(range(len(RUNG_ORDER)))
    ax.set_xticklabels(RUNG_LABELS, fontsize=FS_TK)
    ax.set_xlim(-0.5, len(RUNG_ORDER) - 0.5)
    ax.set_ylim(-2, 102)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% synergistic triplets", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    # No legend on panel d

    if label:
        ax.set_title(label, fontsize=FS, loc="left")


def draw_panel_e(ax, df_global: pd.DataFrame, df_null: pd.DataFrame,
                 null_rung: str, label: str) -> None:
    """Null KDE and best syn/red lines, all in Cohen's f² space.

    null_rung is the rung at which the null model was evaluated (determined by
    _null_rung, matching the null script's _best_primary_rung logic).  We use
    its base_r2 to convert null R² → f², and pick the best-f2 candidate from
    that same rung for the vertical lines so everything is on the same footing.
    """
    rung_data = df_global[df_global["rung_id"] == null_rung]
    base_r2   = rung_data["base_r2"].iloc[0]

    syn_sub = rung_data[rung_data["objective"] == "o_min"]
    red_sub = rung_data[rung_data["objective"] == "o_max"]
    best_syn_f2 = syn_sub["cohen_f2"].replace([np.inf, -np.inf], np.nan).max()
    best_red_f2 = red_sub["cohen_f2"].replace([np.inf, -np.inf], np.nan).max()

    null_r2 = df_null["global_oof_r2"].dropna().values
    null_f2 = (null_r2 - base_r2) / (1.0 - null_r2)

    x_min = null_f2.min() - 0.002
    x_max = max(null_f2.max(), best_syn_f2, best_red_f2) + 0.002
    xs  = np.linspace(x_min, x_max, 400)
    kde = gaussian_kde(null_f2)
    ax.fill_between(xs, kde(xs), alpha=0.50, color=NULL_COLOR, label="Null (10K random)", zorder=2)

    # Add-one (Davison-Hinkley) permutation estimator, (k+1)/(n+1): the
    # unsmoothed k/n can return a literal 0.0, which is not a valid p-value and
    # is anti-conservative. At 10,000 draws with zero exceedances this yields an
    # exact 9.999e-05 -- a computed value, not a floor bound, so it is printed.
    n_null = null_f2.size
    p_syn = (int((null_f2 >= best_syn_f2).sum()) + 1) / (n_null + 1)
    p_red = (int((null_f2 >= best_red_f2).sum()) + 1) / (n_null + 1)
    fmt = lambda p: f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
    ax.axvline(best_syn_f2, color=SYN_COLOR, lw=2.0, ls="-", zorder=4,
               label=f"Best syn ({fmt(p_syn)})")
    ax.axvline(best_red_f2, color=RED_COLOR,  lw=2.0, ls="-", zorder=4,
               label=f"Best red ({fmt(p_red)})")
    ax.set_xlabel("Cohen's f²", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    ax.legend(fontsize=FS_TK - 2, framealpha=0.7)
    ax.yaxis.set_visible(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(label, fontsize=FS, loc="left")


ORDER_XLIM = (3, ORDER_MAX)  # shared x-axis for panels A, F, K

def _div_stats(df_div: pd.DataFrame, objective: str, col: str):
    """Return (orders, means, lo, hi) — ribbon is mean ± 1 SD."""
    sub = df_div[df_div["objective"] == objective]
    orders = np.array(sorted(sub["order"].unique()))
    means, lo, hi = [], [], []
    for k in orders:
        vals = sub[sub["order"] == k][col].values
        m, s = np.mean(vals), np.std(vals)
        means.append(m)
        lo.append(m - s)
        hi.append(m + s)
    return orders, np.array(means), np.array(lo), np.array(hi)


def draw_panel_n_domains(ax, df_div: pd.DataFrame, label: str, show_legend: bool = True) -> None:
    """Mean ± 1 SD ribbon of unique domains vs order k — syn and red."""
    for obj, color, lbl in [("o_min", SYN_COLOR, "Synergistic"),
                             ("o_max", RED_COLOR,  "Redundant")]:
        orders, means, lo, hi = _div_stats(df_div, obj, "n_domains")
        ax.fill_between(orders, np.clip(lo, 0, 10), np.clip(hi, 0, 10),
                        color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(orders, means, color=color, lw=2.0, zorder=4, label=lbl)

    ax.axhline(10, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(ORDER_XLIM)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("Set size", fontsize=FS)
    ax.set_ylabel("Unique domains per candidate", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    if show_legend:
        ax.legend(fontsize=FS_TK - 2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_title(label, fontsize=FS, loc="left")


def draw_panel_shannon(ax, df_div: pd.DataFrame, label: str, show_legend: bool = True) -> None:
    """Mean ± 1 SD ribbon of Shannon H (log2) vs order k — syn and red."""
    for obj, color, lbl in [("o_min", SYN_COLOR, "Synergistic"),
                             ("o_max", RED_COLOR,  "Redundant")]:
        orders, means, lo, hi = _div_stats(df_div, obj, "shannon_h")
        ax.fill_between(orders, np.clip(lo, 0, None), hi,
                        color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(orders, means, color=color, lw=2.0, zorder=4, label=lbl)

    ax.set_xlim(ORDER_XLIM)
    ax.set_ylim(0, None)
    ax.set_xlabel("Set size", fontsize=FS)
    ax.set_ylabel("Shannon H (bits)", fontsize=FS)
    ax.tick_params(labelsize=FS_TK)
    if show_legend:
        ax.legend(fontsize=FS_TK - 2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_title(label, fontsize=FS, loc="left")


# ── Source Data ───────────────────────────────────────────────────────────────
#
# One frame per drawn panel, built from the SAME selections the draw_* helpers
# use (_rung_top50 for the split boxes, _div_stats for the ribbons) so a reader
# reproduces the marks rather than the upstream candidate pool, which holds far
# more rows than any panel shows.

def _sd_panel_a(df_greedy: pd.DataFrame) -> pd.DataFrame:
    """Row 1 col 1: min/max O-info envelope by set size, per objective."""
    df = df_greedy[df_greedy["objective"].isin(["o_min", "o_max"])]
    rows = []
    for obj in ("o_min", "o_max"):
        grp = df[df["objective"] == obj].groupby("order")["omega"]
        for order, lo, hi in zip(grp.min().index, grp.min().values, grp.max().values):
            rows.append({
                "objective": obj, "set_size": int(order),
                "omega_min": float(lo), "omega_max": float(hi),
            })
    return pd.DataFrame(rows).sort_values(["objective", "set_size"]).reset_index(drop=True)


def _sd_split_boxes(df_global: pd.DataFrame, y_col: str,
                    single_exp: dict | None = None) -> pd.DataFrame:
    """The per-candidate values behind a strip+box panel, one row per dot.

    Mirrors _draw_split_boxes: top-50 by delta_r2_vs_base within each
    (model level, objective), which is the pool the box statistics summarise.
    """
    rows = []
    for rung in RUNG_ORDER:
        syn, red, base = _rung_top50(df_global, rung)
        for vals_df, obj in ((syn, "o_min"), (red, "o_max")):
            vals = vals_df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
            for candidate_id, value in zip(vals_df.loc[vals.index, "candidate_id"], vals):
                rows.append({
                    "model_level": RUNG_DISPLAY[rung], "objective": obj,
                    "candidate_id": str(candidate_id), y_col: float(value),
                    "baseline_r2": float(base) if np.isfinite(base) else np.nan,
                    "best_single_r2": (
                        float(single_exp[rung])
                        if single_exp is not None and single_exp.get(rung) is not None
                        else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def _sd_ribbon(df_div: pd.DataFrame, col: str) -> pd.DataFrame:
    """Row 1 cols 2-3: mean +/- 1 SD ribbon by set size, per objective."""
    rows = []
    for obj in ("o_min", "o_max"):
        orders, means, lo, hi = _div_stats(df_div, obj, col)
        for order, mean, low, high in zip(orders, means, lo, hi):
            rows.append({
                "objective": obj, "set_size": int(order), f"{col}_mean": float(mean),
                f"{col}_minus_1sd": float(low), f"{col}_plus_1sd": float(high),
            })
    return pd.DataFrame(rows)


def _sd_panel_d(df_enr: pd.DataFrame, bag: str) -> pd.DataFrame:
    """Row 2+ col 3: % synergistic triplets per evaluated candidate."""
    sub = df_enr[df_enr["bag"] == bag]
    if sub.empty or sub.get("mode", pd.Series(dtype=str)).iloc[0] != "frac_neg_k3":
        return pd.DataFrame()
    out = sub[["rung", "objective", "frac_neg_k3", "full_r2"]].copy()
    out["pct_synergistic_triplets"] = out.pop("frac_neg_k3") * 100.0
    out["model_level"] = out.pop("rung").map(RUNG_DISPLAY)
    return out[["model_level", "objective", "pct_synergistic_triplets", "full_r2"]].reset_index(drop=True)


_OBJ_GLOSS = {
    "objective": "o_min = synergistic (min O-information) arm; o_max = redundant (max O-information) arm",
    "model_level": "model level (OLS, d1, d2, d3)",
    "set_size": "number of exposures in the candidate set",
}
_TOP50_NOTE = (
    "One row per plotted candidate: the top 50 by delta_r2_vs_base within each "
    "model level and objective, which is the pool the box and whiskers summarise. "
    "baseline_r2 and best_single_r2 repeat per row -- each is a single horizontal "
    "line drawn once per model level."
)


def _build_fig2_panels(all_data: dict, bag_row_order: list[str]) -> list[Panel]:
    df_div = all_data["_domain_diversity"]
    df_enr = all_data["_enrichment_per_rung"]
    head_bag = "functional" if "functional" in BAGS else BAGS[0]

    panels = [
        Panel(
            panel_id="a1_oinfo_by_set_size",
            frame=_sd_panel_a(all_data[head_bag]["greedy"]),
            description="Exposome synergy-redundancy balance across set sizes: the O-information envelope (min to max) at each set size.",
            columns={**_OBJ_GLOSS, "omega_min": "lowest O-information (nats) at that set size",
                     "omega_max": "highest O-information (nats) at that set size"},
            notes="Shaded band spans min to max across all candidates of that set size; it is a range, not a confidence interval.",
        ),
        Panel(
            panel_id="a2_unique_domains",
            frame=_sd_ribbon(df_div, "n_domains"),
            description="Unique exposome domains per candidate against set size.",
            columns={**_OBJ_GLOSS, "n_domains_mean": "mean unique domains per candidate",
                     "n_domains_minus_1sd": "mean - 1 SD", "n_domains_plus_1sd": "mean + 1 SD"},
            notes="Ribbon is mean +/- 1 SD across candidates, not a confidence interval. Drawn values are clipped to [0, 10]; the tabulated bounds are unclipped.",
        ),
        Panel(
            panel_id="a3_shannon_entropy",
            frame=_sd_ribbon(df_div, "shannon_h"),
            description="Shannon domain entropy H per candidate against set size.",
            columns={**_OBJ_GLOSS, "shannon_h_mean": "mean Shannon domain entropy H (bits)",
                     "shannon_h_minus_1sd": "mean - 1 SD", "shannon_h_plus_1sd": "mean + 1 SD"},
            notes="Ribbon is mean +/- 1 SD across candidates, not a confidence interval.",
        ),
    ]

    for letter, bag in zip(ROW_LETTERS[1:], bag_row_order):
        d = all_data[bag]
        short = BAG_SHORT[bag]
        panels.append(Panel(
            panel_id=f"{letter}1_{short}_r2",
            frame=_sd_split_boxes(d["global"], "full_r2", single_exp=d["single_exp_r2"]),
            description=f"{BAG_LABELS_SD[bag]}-exposome association: held-out LOCO R² by model level.",
            columns={**_OBJ_GLOSS, "candidate_id": "identifier of the exposure set",
                     "full_r2": "held-out LOCO R² of the full model",
                     "baseline_r2": "covariate-only baseline R² at that model level",
                     "best_single_r2": "best single-exposure R² at that model level"},
            notes=_TOP50_NOTE,
        ))
        panels.append(Panel(
            panel_id=f"{letter}2_{short}_set_size",
            frame=_sd_split_boxes(d["global"], "order"),
            description=f"{BAG_LABELS_SD[bag]}: set size of the same top-50 candidates by model level.",
            columns={**_OBJ_GLOSS, "candidate_id": "identifier of the exposure set",
                     "order": "set size (number of exposures) of that candidate"},
            notes=_TOP50_NOTE + " The dashed line marks set size 21, the last set size with negative O-information.",
        ))
        panels.append(Panel(
            panel_id=f"{letter}3_{short}_syn_triplets",
            frame=_sd_panel_d(df_enr, bag),
            description=f"{BAG_LABELS_SD[bag]}: synergistic triplet content of each evaluated candidate.",
            columns={**_OBJ_GLOSS,
                     "pct_synergistic_triplets": "percentage of the candidate's triplet sub-combinations with negative O-information",
                     "full_r2": "held-out LOCO R² of that candidate"},
            notes=(f"One row per evaluated candidate (top-20 synergistic + top-20 redundant, the only "
                   f"candidates with sub-combination O-information computed). The dashed line is the "
                   f"exposome-wide rate over all C(63,3) triplets, {SUBCOMB_BASELINE_K3 * 100:.1f}%."),
        ))
    return panels


BAG_LABELS_SD = {"structural": "Structural BAG", "functional": "Functional BAG", "combined": "Combined BAG"}
BAG_SHORT = {"structural": "struct", "functional": "func", "combined": "comb"}
ROW_LETTERS = ("a", "b", "c", "d", "e", "f")


# ── main ──────────────────────────────────────────────────────────────────────

async def plot_fig2_grid() -> None:
    print("Loading all data …")
    all_data = await load_all()

    # BAG rows preserve the canonical order (structural, functional, combined),
    # restricted to whichever BAGs are requested (PAPER_FIG_BAGS).
    bag_row_order = [b for b in ("structural", "functional", "combined") if b in BAGS]
    row_label_for = {"structural": ROW_LABELS[1], "functional": ROW_LABELS[2], "combined": ROW_LABELS[3]}
    nrows = 1 + len(bag_row_order)

    fig = plt.figure(figsize=(24, 6 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 3)

    df_enr = all_data["_enrichment_per_rung"]
    df_div = all_data["_domain_diversity"]

    head_bag = "functional" if "functional" in BAGS else BAGS[0]
    ax = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax, all_data[head_bag]["greedy"], ROW_LABELS[0])
    draw_panel_n_domains(fig.add_subplot(gs[0, 1]), df_div, "", show_legend=False)
    draw_panel_shannon(fig.add_subplot(gs[0, 2]), df_div, "", show_legend=False)

    for row_idx, bag in enumerate(bag_row_order, start=1):
        d = all_data[bag]
        show_legend = row_idx == 1
        draw_panel_b(fig.add_subplot(gs[row_idx, 0]), d["global"], row_label_for[bag],
                     show_legend=show_legend, single_exp_r2=d["single_exp_r2"])
        draw_panel_c(fig.add_subplot(gs[row_idx, 1]), d["global"], "")
        draw_panel_d(fig.add_subplot(gs[row_idx, 2]), df_enr, bag, "",
                     show_legend=show_legend)

    fig_dir = OUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "svg", "png"):
        out = fig_dir / f"fig2_grid_v3{SUFFIX}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")

    for path in write_source_data(
        f"fig2_grid_v3{SUFFIX}",
        _build_fig2_panels(all_data, bag_row_order),
        fig_dir,
        source_paths=[
            str(OUTPUT_ROOT / VARIANT / "families" / "pooled_oinfo_ladder" / "canonical"
                / "per_experiment" / "pooled_oinfo_ladder_<bag>" / "metrics_global_long.parquet"),
            str(GREEDY_ROOT / "greedy_topk_by_objective_order.csv"),
            str(SUBCOMB_ENRICHMENT_CSV),
        ],
    ):
        print(f"Saved: {path}")

    if ORDER_MAX is not None:
        plt.close(fig)
        return

    plt.close(fig)

    # ── Best single exposure Cohen's f² CSV ───────────────────────────────────
    rows = []
    for bag in BAGS:
        for rung, d in SINGLE_EXP_DIRS.items():
            if d is None:
                continue
            csv_path = d / bag / "single_exposure_global.csv"
            if not csv_path.exists():
                continue
            df_se = pd.read_csv(csv_path)
            best = df_se.loc[df_se["global_oof_r2"].idxmax()]
            full_r2   = float(best["global_oof_r2"])
            delta_r2  = float(best["delta_r2_vs_base"])
            cohen_f2  = delta_r2 / (1.0 - full_r2) if (1.0 - full_r2) != 0 else np.nan
            rows.append({
                "bag":          bag,
                "rung":         rung,
                "feature_name": best["feature_name"],
                "domain":       best["domain"],
                "full_r2":      full_r2,
                "delta_r2":     delta_r2,
                "cohen_f2":     cohen_f2,
            })
    df_out = pd.DataFrame(rows)
    out_csv = fig_dir / "best_single_exposure_cohen_f2.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    asyncio.run(plot_fig2_grid())
