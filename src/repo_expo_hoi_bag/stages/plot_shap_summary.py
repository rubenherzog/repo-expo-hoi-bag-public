#!/usr/bin/env python3
"""Plot signed SHAP summaries and interaction matrices from v3 SHAP outputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_shap_dir(root: Path, bag_target: str, run_id: str | None = None) -> Path:
    path = root / bag_target
    if not path.exists():
        raise FileNotFoundError(f"SHAP bag directory not found: {path}")
    if (path / "shap_summary.parquet").exists():
        return path
    if run_id:
        candidate = path / run_id
        if not candidate.exists():
            raise FileNotFoundError(f"Requested SHAP run not found: {candidate}")
        return candidate
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    raise FileNotFoundError(
        f"SHAP output directory for '{bag_target}' is ambiguous."
        f" Found subdirectories: {[p.name for p in subdirs]}"
        " Specify --run-id if needed."
    )


BASELINE_FEATURE_PREFIXES = ("age", "year", "sex_", "diag_")


def is_baseline_feature(feature: str) -> bool:
    return (
        feature == "__bias__"
        or any(feature.startswith(prefix) for prefix in BASELINE_FEATURE_PREFIXES)
    )


def format_exposome_label(feature: str) -> str:
    """Use compact semantic labels on the figure axis."""
    replacements = {
        "Carbon monoxide (CO) emissions": "CO emissions",
        "Sulphur dioxide (SO₂) emissions": "SO₂ emissions",
        "Average share of green area in city/ urban area": "Green area share",
        "Pop_basic_drinking-water(%)": "Drinking water %",
        "deaths_trans": "Transport deaths",
        "unemp": "Unemployment",
        "mean_temp_areaw_o": "Temp (area)",
        "mean_prec2_areaw_o": "Precip. (area)",
        "effect_parl_est": "Parliament",
        "elect_part_est": "Electoral",
        "PM2.5": "PM2.5",
        "GINI": "GINI",
    }
    return replacements.get(feature, feature)


def load_shap_outputs(shap_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    summary_path = shap_dir / "shap_summary.parquet"
    values_path = shap_dir / "shap_values_oof.parquet"
    interaction_path = shap_dir / "shap_interaction_oof.npz"

    if not summary_path.exists() or not values_path.exists() or not interaction_path.exists():
        raise FileNotFoundError(
            f"Missing SHAP outputs in {shap_dir}."
            " Expected shap_summary.parquet, shap_values_oof.parquet, shap_interaction_oof.npz"
        )

    shap_summary = pd.read_parquet(summary_path)
    shap_values = pd.read_parquet(values_path)
    interactions = np.load(interaction_path, allow_pickle=True)
    feature_names = [str(x) for x in np.asarray(interactions["feature_names_full"])]
    interaction_matrix = np.asarray(interactions["interactions_full"])

    return shap_summary, shap_values, feature_names, interaction_matrix


def build_median_signed_summary(
    shap_summary: pd.DataFrame,
    shap_values: pd.DataFrame,
    exposomic_only: bool = False,
) -> pd.DataFrame:
    full_prefix = "shap_full__"
    full_cols = [c for c in shap_values.columns if c.startswith(full_prefix)]
    if not full_cols:
        return pd.DataFrame(
            columns=[
                "bag_target", "candidate_id", "model", "scope", "feature",
                "median_signed_shap", "mean_shap", "mean_abs_shap", "n_subjects",
            ]
        )

    selected_cols = []
    for col in full_cols:
        feature = col[len(full_prefix):]
        if feature == "__bias__":
            continue
        if exposomic_only and is_baseline_feature(feature):
            continue
        selected_cols.append(col)

    if not selected_cols:
        return pd.DataFrame(
            columns=[
                "bag_target", "candidate_id", "model", "scope", "feature",
                "median_signed_shap", "mean_shap", "mean_abs_shap", "n_subjects",
            ]
        )

    values = shap_values[selected_cols].median(axis=0, skipna=True)
    data = []
    for col, median_value in values.items():
        feature = col[len(full_prefix):]
        data.append({"feature": feature, "median_signed_shap": float(median_value)})
    median_df = pd.DataFrame(sorted(data, key=lambda row: abs(row["median_signed_shap"]), reverse=True))

    reference = (
        shap_summary[(shap_summary["model"] == "full") & (shap_summary["scope"] == "global")]
        .loc[:, ["bag_target", "candidate_id", "model", "scope", "feature", "mean_abs_shap", "mean_shap", "n_subjects"]]
    )
    merged = median_df.merge(reference, on="feature", how="left")
    merged = merged.loc[:, ["bag_target", "candidate_id", "model", "scope", "feature", "median_signed_shap", "mean_shap", "mean_abs_shap", "n_subjects"]]
    return merged


def build_country_exposome_heatmap(
    shap_summary: pd.DataFrame,
    shap_values: pd.DataFrame,
    top_n: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a country x exposome median-SHAP matrix for the full model."""
    full_prefix = "shap_full__"
    full_cols = [c for c in shap_values.columns if c.startswith(full_prefix)]
    if not full_cols:
        return pd.DataFrame(), []

    reference = shap_summary[
        (shap_summary["model"] == "full")
        & (shap_summary["scope"] == "global")
        & (~shap_summary["feature"].isin(["__bias__"]))
    ].copy()
    reference = reference[~reference["feature"].str.startswith(BASELINE_FEATURE_PREFIXES)]
    if reference.empty:
        return pd.DataFrame(), []

    top_features = (
        reference.sort_values("mean_abs_shap", ascending=False)
        .head(int(top_n))["feature"]
        .astype(str)
        .tolist()
    )

    wanted_cols = [f"{full_prefix}{feat}" for feat in top_features if f"{full_prefix}{feat}" in shap_values.columns]
    if not wanted_cols:
        return pd.DataFrame(), []

    meta_cols = ["country"]
    needed = meta_cols + wanted_cols
    missing = [c for c in needed if c not in shap_values.columns]
    if missing:
        raise FileNotFoundError(f"Missing expected SHAP columns for country heatmap: {missing[:5]}")

    long_df = shap_values[needed].copy()
    countries = [c for c in sorted(long_df["country"].dropna().astype(str).unique().tolist()) if c]
    rows = []
    for country in countries:
        sub = long_df[long_df["country"].astype(str) == country]
        row = {"country": country}
        for feat in top_features:
            col = f"{full_prefix}{feat}"
            row[feat] = float(sub[col].median(skipna=True))
        rows.append(row)
    heatmap_df = pd.DataFrame(rows).set_index("country")
    heatmap_df = heatmap_df.loc[countries, top_features]
    return heatmap_df, top_features


def plot_median_signed_shap(df: pd.DataFrame, out_png: Path, out_pdf: Path, title_suffix: str | None = None) -> None:
    plot_df = df.dropna(subset=["median_signed_shap"]).copy()
    plot_df = plot_df.sort_values("median_signed_shap")
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.25)))
    colors = ["#2166ac" if x < 0 else "#b2182b" for x in plot_df["median_signed_shap"]]
    ax.barh(plot_df["feature"], plot_df["median_signed_shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Median signed SHAP")
    ax.set_ylabel("")
    title = "Median signed SHAP per feature"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    if plot_df.empty:
        print(f"WARNING: No data to plot median signed SHAP to {out_png}")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_matrix(
    matrix: np.ndarray,
    features: list[str],
    out_png: Path,
    out_pdf: Path,
    title_suffix: str | None = None,
    exclude_baseline: bool = False,
) -> None:
    keep = [
        i for i, f in enumerate(features)
        if f != "__bias__" and not (exclude_baseline and is_baseline_feature(f))
    ]
    matrix = matrix[:, keep, :][:, :, keep]
    labels = [features[i] for i in keep]
    if len(labels) == 0 or matrix.size == 0:
        print(f"WARNING: No SHAP interaction features to plot for {out_png}")
        return
    median_matrix = np.nanmedian(matrix, axis=0)
    order = np.argsort(-np.nanmedian(np.abs(median_matrix), axis=0))
    median_matrix = median_matrix[order][:, order]
    labels = [labels[i] for i in order]

    mask = np.eye(len(median_matrix), dtype=bool)
    masked_matrix = np.ma.masked_where(mask, median_matrix)
    vlim = np.nanmax(np.abs(median_matrix[~mask]))
    if np.isnan(vlim) or vlim == 0:
        vlim = np.nanmax(np.abs(median_matrix))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(masked_matrix, cmap="RdBu_r", aspect="equal", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    title = "Median signed SHAP interaction matrix"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Median signed interaction")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_country_exposome_heatmap(
    heatmap_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title_suffix: str | None = None,
) -> None:
    if heatmap_df.empty:
        print(f"WARNING: No data to plot country exposome heatmap to {out_png}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    matrix = heatmap_df.to_numpy(dtype=float)  # countries on y-axis, features on x-axis
    countries = list(heatmap_df.index)
    features = list(heatmap_df.columns)
    row_strength = heatmap_df.abs().mean(axis=1).to_numpy(dtype=float)

    # Keep the color scale symmetric and robust to a few outliers.
    vlim = np.nanquantile(np.abs(matrix), 0.75)
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0
    vlim = max(vlim, 0.05)

    fig_w = max(13.5, 1.00 * len(features) + 4.0)
    fig_h = max(6.5, 0.34 * len(countries) + 1.8)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 2.2], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")

    ax.set_xticks(np.arange(len(features)))
    display_features = [format_exposome_label(f) for f in features]
    ax.set_xticklabels(display_features, rotation=0, ha="center", fontsize=8)
    ax.set_yticks(np.arange(len(countries)))
    ax.set_yticklabels(countries, fontsize=8)

    ax.set_xlabel("Exposome variable")
    ax.set_ylabel("Country")
    title = "Country-level median SHAP for exposomic variables"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(countries), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax_bar.barh(np.arange(len(countries)), row_strength, color="#666666")
    ax_bar.set_xlim(0, max(float(np.nanmax(row_strength)) * 1.05, 0.01))
    ax_bar.set_xticks([])
    ax_bar.set_xlabel("Mean |SHAP|", fontsize=8)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    for spine in ("top", "right", "bottom"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_visible(True)

    cbar = fig.colorbar(im, ax=ax, label="Median SHAP", shrink=0.92, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_interaction_csv(
    matrix: np.ndarray,
    features: list[str],
    out_path: Path,
    exclude_baseline: bool = False,
) -> None:
    keep = [
        i for i, f in enumerate(features)
        if f != "__bias__" and not (exclude_baseline and is_baseline_feature(f))
    ]
    labels = [features[i] for i in keep]
    median_matrix = np.nanmedian(matrix[:, keep, :][:, :, keep], axis=0)
    if len(labels) == 0 or matrix.size == 0:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    matrix_df = pd.DataFrame(median_matrix, index=labels, columns=labels)
    matrix_df.to_csv(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHAP summary figures and stats for v3 models.")
    v3_root = os.environ.get("V3_OUTPUT_ROOT", "")
    default_shap_root = str(Path(v3_root) / "shap_oof") if v3_root else "outputs/variant_a/shap_oof"
    default_fig_dir = str(Path(v3_root) / "figures" / "shap") if v3_root else "outputs/variant_a/figures/shap"
    default_stats_dir = str(Path(v3_root) / "stats" / "shap") if v3_root else "outputs/variant_a/stats/shap"

    parser.add_argument("--bag", default="functional", help="BAG target name to use (e.g. functional, structural, combined)")
    parser.add_argument("--shap-root", default=default_shap_root, help="Root directory containing SHAP output folders")
    parser.add_argument("--fig-dir", default=default_fig_dir, help="Output directory for SHAP figures")
    parser.add_argument("--stats-dir", default=default_stats_dir, help="Output directory for SHAP stats")
    parser.add_argument("--run-id", default=None, help="Specific SHAP run subdirectory if there are multiple runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shap_root = Path(args.shap_root)
    out_fig_dir = Path(args.fig_dir)
    out_stats_dir = Path(args.stats_dir)
    shap_dir = resolve_shap_dir(shap_root, args.bag, args.run_id)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    shap_summary, shap_values, feature_names, interaction_matrix = load_shap_outputs(shap_dir)
    bag_suffix = f"_{args.bag}" if args.bag else ""

    full_df = build_median_signed_summary(shap_summary, shap_values, exposomic_only=False)
    full_df.to_csv(out_stats_dir / f"shap_median_signed{bag_suffix}.csv", index=False)
    write_interaction_csv(
        interaction_matrix,
        feature_names,
        out_stats_dir / f"shap_interaction_matrix{bag_suffix}.csv",
        exclude_baseline=False,
    )
    plot_median_signed_shap(
        full_df,
        out_fig_dir / f"fig_shap_median_signed{bag_suffix}.png",
        out_fig_dir / f"fig_shap_median_signed{bag_suffix}.pdf",
        title_suffix="full model",
    )
    plot_interaction_matrix(
        interaction_matrix,
        feature_names,
        out_fig_dir / f"fig_shap_interaction_matrix{bag_suffix}.png",
        out_fig_dir / f"fig_shap_interaction_matrix{bag_suffix}.pdf",
        title_suffix="full model",
        exclude_baseline=False,
    )

    exposomic_df = build_median_signed_summary(shap_summary, shap_values, exposomic_only=True)
    exposomic_df.to_csv(out_stats_dir / f"shap_median_signed_exposomic{bag_suffix}.csv", index=False)
    write_interaction_csv(
        interaction_matrix,
        feature_names,
        out_stats_dir / f"shap_interaction_matrix_exposomic{bag_suffix}.csv",
        exclude_baseline=True,
    )
    plot_median_signed_shap(
        exposomic_df,
        out_fig_dir / f"fig_shap_median_signed_exposomic{bag_suffix}.png",
        out_fig_dir / f"fig_shap_median_signed_exposomic{bag_suffix}.pdf",
        title_suffix="exposomic only",
    )
    plot_interaction_matrix(
        interaction_matrix,
        feature_names,
        out_fig_dir / f"fig_shap_interaction_matrix_exposomic{bag_suffix}.png",
        out_fig_dir / f"fig_shap_interaction_matrix_exposomic{bag_suffix}.pdf",
        title_suffix="exposomic only",
        exclude_baseline=True,
    )

    heatmap_df, top_features = build_country_exposome_heatmap(shap_summary, shap_values, top_n=10)
    if not heatmap_df.empty:
        heatmap_csv = out_stats_dir / f"shap_country_median_exposomic{bag_suffix}.csv"
        heatmap_df.to_csv(heatmap_csv)
        plot_country_exposome_heatmap(
            heatmap_df,
            out_fig_dir / f"fig_shap_country_median_exposomic{bag_suffix}.png",
            out_fig_dir / f"fig_shap_country_median_exposomic{bag_suffix}.pdf",
            title_suffix=f"top {len(top_features)} features",
        )
    else:
        print(f"WARNING: No country exposome heatmap produced for {args.bag}")
    print(f"Wrote figures to {out_fig_dir}")
    print(f"Wrote stats to {out_stats_dir}")


if __name__ == "__main__":
    main()
