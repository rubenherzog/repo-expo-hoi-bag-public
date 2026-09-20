#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
import hashlib
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
# Publication style (Arial, fonttype 42, editable SVG text) must be imported
# before pyplot so its rcParams apply to every figure this stage draws.
from repo_expo_hoi_bag.figures.style import (  # noqa: E402
    BAG_LABELS,
    BAG_ROW_ORDER,
    BAG_SHORT,
    LEVEL_LABELS,
    ROW_LETTERS,
    save_figure,
    style_axis,
)
from repo_expo_hoi_bag.figures.source_data import Panel, write_source_data  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.sensitivity_common import (
    ACTIVE_RUNGS,
    RUNG_COLORS,
    RUNG_LABELS,
    active_rungs,
    analysis_cfg_from_config,
    apply_order_cap,
    build_domain_pc_model_from_reference,
    build_original_model_df,
    bundle_sensitivity_root,
    canonical_root,
    cap_split_subdir,
    combo_candidate_table,
    copy_tree_contents,
    load_best_single_by_rung,
    load_existing_baselines,
    load_greedy_reference_exposome,
    load_raw_and_domains,
    load_sensitivity_config,
    log_msg,
    repo_sensitivity_figures_root,
    repo_sensitivity_root,
    score_oinfo_with_cache,
    selected_bags,
    evaluate_candidates_by_rung,
    env_bool,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


ROW_LABELS = {
    "best_single_per_domain": "Best single per domain",
    "within_domain_pc1": "Domain PC1",
}
# Compact family tags for Source Data sheet names (xlsx caps names at 31 chars).
FAMILY_SHORT = {
    "best_single_per_domain": "bestsingle",
    "within_domain_pc1": "pc1",
}
C_SYN = "#2166ac"
C_RED = "#b2182b"
C_BEST = "#238b45"
C_SINGLE = "#111111"
C_ORIGINAL = "#6a3d9a"


def _max_candidates() -> int | None:
    val = os.environ.get("SENSITIVITY_MAX_CANDIDATES", "").strip()
    return int(val) if val else None


def _reuse_domain_pc1_oinfo(candidate_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Attach the parent PC1 O-information scores by exact predictor identity."""
    reference = pd.read_csv(path)
    required = {"predictors_identity", "score", "thoi_o"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Domain-PC1 O-information reference lacks {sorted(missing)}")
    if reference["predictors_identity"].astype(str).duplicated().any():
        raise ValueError("Domain-PC1 O-information reference has duplicate identities")
    out = candidate_df.copy()
    score_map = reference.set_index("predictors_identity")["score"]
    thoi_map = reference.set_index("predictors_identity")["thoi_o"]
    out["score"] = out["predictors_identity"].astype(str).map(score_map)
    out["thoi_o"] = out["predictors_identity"].astype(str).map(thoi_map)
    if out[["score", "thoi_o"]].isna().any().any():
        absent = out.loc[out["score"].isna(), "predictors_identity"].head(5).tolist()
        raise ValueError(f"Domain-PC1 O-information identities are missing: {absent}")
    out["rank_o_min"] = out.groupby("order")["score"].rank(method="first", ascending=True)
    out["rank_o_max"] = out.groupby("order")["score"].rank(method="first", ascending=False)
    out["oinfo_reference_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _best_single_representatives(
    single_df: pd.DataFrame,
    source_rung: str,
    domain_order: list[str],
) -> list[str]:
    sub = single_df[single_df["source_rung"].astype(str) == source_rung].copy()
    reps = []
    for domain in domain_order:
        dd = sub[sub["domain"].astype(str) == domain].copy()
        if dd.empty:
            continue
        dd["global_oof_r2"] = pd.to_numeric(dd["global_oof_r2"], errors="coerce")
        reps.append(str(dd.sort_values("global_oof_r2", ascending=False).iloc[0]["feature_name"]))
    if len(reps) != len(domain_order):
        raise ValueError(f"Best-single set for {source_rung} resolved {len(reps)} domains, expected {len(domain_order)}.")
    return reps


def _family_performance(summary: pd.DataFrame, family: str, rungs: list[str], top_k: int) -> pd.DataFrame:
    rows = []
    sub = summary[summary["candidate_family"].astype(str) == family].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "candidate_family",
                "rung_id",
                "series",
                "median_r2",
                "min_r2",
                "max_r2",
                "n",
                "top20_scope",
            ]
        )
    sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
    sub["global_oof_r2"] = pd.to_numeric(sub["global_oof_r2"], errors="coerce")
    for rung in rungs:
        rr = sub[sub["rung_id"].astype(str) == rung].copy()
        syn_ids = set(
            rr[rr["score"] < 0]
            .nlargest(int(top_k), "global_oof_r2")["candidate_id"]
            .astype(str)
        )
        red_ids = set(
            rr[rr["score"] > 0]
            .nlargest(int(top_k), "global_oof_r2")["candidate_id"]
            .astype(str)
        )
        for label, ids in [("top20_syn", syn_ids), ("top20_red", red_ids)]:
            vals = pd.to_numeric(rr.loc[rr["candidate_id"].astype(str).isin(ids), "global_oof_r2"], errors="coerce")
            vals = vals[np.isfinite(vals)]
            rows.append(
                {
                    "candidate_family": family,
                    "rung_id": rung,
                    "series": label,
                    "median_r2": float(vals.median()) if len(vals) else np.nan,
                    "min_r2": float(vals.min()) if len(vals) else np.nan,
                    "max_r2": float(vals.max()) if len(vals) else np.nan,
                    "n": int(vals.notna().sum()),
                    "top20_scope": "intra_rung",
                }
            )
        vals_all = pd.to_numeric(rr["global_oof_r2"], errors="coerce")
        rows.append(
            {
                "candidate_family": family,
                "rung_id": rung,
                "series": "best_exhaustive",
                "median_r2": float(vals_all.max()) if len(vals_all) else np.nan,
                "min_r2": np.nan,
                "max_r2": np.nan,
                "n": int(vals_all.notna().sum()),
                "top20_scope": "intra_rung",
            }
        )
    return pd.DataFrame(rows)


def _best_by_order(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in summary.groupby(["bag", "candidate_family", "source_label", "rung_id", "order"], dropna=False):
        vals = pd.to_numeric(sub["global_oof_r2"], errors="coerce")
        if vals.notna().any():
            best = sub.loc[vals.idxmax()].copy()
            rows.append(
                {
                    "bag": keys[0],
                    "candidate_family": keys[1],
                    "source_label": keys[2],
                    "rung_id": keys[3],
                    "order": keys[4],
                    "candidate_id": best["candidate_id"],
                    "global_oof_r2": best["global_oof_r2"],
                    "score": best["score"],
                    "predictors_identity": best["predictors_identity"],
                }
            )
    return pd.DataFrame(rows)


def _load_original_complete_best_by_rung(bag: str, rungs: list[str]) -> pd.DataFrame:
    path = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not path.exists():
        path = canonical_root() / "metrics_global_long.parquet"
    df = pd.read_parquet(path)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag].copy()
    df = df[df["rung_id"].astype(str).isin(rungs)].copy()
    # Cap the candidate orders (cap21 vs complete) before picking the per-rung
    # best so the "original best model" reference matches the capped paper set.
    df = apply_order_cap(df)
    df["full_r2"] = pd.to_numeric(df["full_r2"], errors="coerce")
    rows = []
    for rung in rungs:
        rr = df[df["rung_id"].astype(str) == rung].copy()
        rr = rr[np.isfinite(rr["full_r2"])]
        if rr.empty:
            continue
        best = rr.loc[rr["full_r2"].idxmax()]
        rows.append(
            {
                "rung_id": rung,
                "original_complete_best_r2": float(best["full_r2"]),
                "original_complete_best_candidate_id": best.get("candidate_id", ""),
                "original_complete_best_objective": best.get("objective", ""),
                "original_complete_best_order": best.get("order", np.nan),
                "original_complete_best_predictors": best.get("predictors_identity", ""),
            }
        )
    return pd.DataFrame(rows)


def _families_present(candidate_scores: pd.DataFrame) -> list[str]:
    preferred = ["best_single_per_domain", "within_domain_pc1"]
    observed = set(candidate_scores["candidate_family"].astype(str))
    families = [family for family in preferred if family in observed]
    if not families:
        raise ValueError("No candidate families available for domain-imbalance figure.")
    return families


def _draw_domain_imbalance_row(
    bag: str,
    summary: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    best_single: pd.DataFrame,
    baselines: pd.DataFrame,
    original_best: pd.DataFrame,
    rungs: list[str],
    top_k: int,
    row_axes,
    row_letter: str,
    show_legend: bool,
) -> list[Panel]:
    """Draw one BAG as a single row of 4 columns.

    Columns are family-major: (family 1 O-info, family 1 performance, family 2
    O-info, family 2 performance). The row label sits on the first column, in
    the plot_fig2_grid_v3 style. Returns that BAG's Source Data panels.
    """
    families = _families_present(candidate_scores)
    x = np.arange(len(rungs))
    base_map = dict(zip(baselines["rung_id"], baselines["baseline_r2"]))
    original_map = (
        dict(zip(original_best["rung_id"], original_best["original_complete_best_r2"]))
        if original_best is not None and not original_best.empty
        else {}
    )

    best_single_by_rung = (
        best_single.groupby("source_rung", as_index=False)["global_oof_r2"].max()
        if not best_single.empty
        else pd.DataFrame(columns=["source_rung", "global_oof_r2"])
    )
    single_map = dict(zip(best_single_by_rung["source_rung"], best_single_by_rung["global_oof_r2"]))

    panels: list[Panel] = []
    for row_i, family in enumerate(families):
        ax_o = row_axes[2 * row_i]
        ax_p = row_axes[2 * row_i + 1]
        sub_scores = candidate_scores[candidate_scores["candidate_family"].astype(str) == family].copy()
        curve_groups = {}
        for source_label, ss in sub_scores.groupby("source_label", sort=True):
            by_order = ss.groupby("order")["score"].agg(["min", "max"]).reset_index()
            signature = tuple(
                np.round(
                    np.column_stack([by_order["order"].to_numpy(), by_order["min"].to_numpy(), by_order["max"].to_numpy()]).ravel(),
                    10,
                )
            )
            if signature not in curve_groups:
                curve_groups[signature] = {"labels": [], "by_order": by_order}
            curve_groups[signature]["labels"].append(str(source_label))
        line_styles = ["-", "--", ":", "-."]
        for group_i, payload in enumerate(curve_groups.values()):
            by_order = payload["by_order"]
            source_text = "/".join(payload["labels"])
            alpha = 0.80 if family == "best_single_per_domain" else 0.90
            ls = line_styles[group_i % len(line_styles)]
            label_suffix = f" ({source_text})" if family == "best_single_per_domain" else ""
            ax_o.plot(by_order["order"], by_order["min"], color=C_SYN, alpha=alpha, ls=ls, lw=1.6, label=f"min O{label_suffix}")
            ax_o.plot(by_order["order"], by_order["max"], color=C_RED, alpha=alpha, ls=ls, lw=1.6, label=f"max O{label_suffix}")
        ax_o.axhline(0, color="#808080", lw=0.8, ls=":")
        style_axis(ax_o)
        panels.append(
            Panel(
                # Shortened so the xlsx 31-char sheet-name cap cannot truncate
                # the candidate family (best_single_per_domain / within_domain_pc1).
                panel_id=f"{row_letter}{2 * row_i + 1}_{BAG_SHORT[bag]}_{FAMILY_SHORT[family]}_oinfo",
                frame=(
                    sub_scores.groupby(["source_label", "order"])["score"]
                    .agg(min_o_info="min", max_o_info="max")
                    .reset_index()
                    .sort_values(["source_label", "order"])
                    .reset_index(drop=True)
                ),
                description=(
                    f"Minimum and maximum O-information (nats) by set size for the "
                    f"{ROW_LABELS[family]} candidate construction, {bag} BAG."
                ),
                columns={
                    "source_label": "candidate source the curve was drawn for",
                    "order": "set size (number of exposures in the set)",
                    "min_o_info": "minimum O-information in nats at that set size (synergistic extreme)",
                    "max_o_info": "maximum O-information in nats at that set size (redundant extreme)",
                },
                notes=(
                    "O-information is computed on the exposome matrix only and is "
                    "independent of which BAG target is regressed."
                ),
            )
        )
        if row_i == 0:
            # Row label on the first column of the row (fig2 convention).
            ax_o.set_title(
                f"{row_letter}. {BAG_LABELS[bag]} — {ROW_LABELS[family]}: min/max O-information",
                loc="left", fontsize=10,
            )
        else:
            ax_o.set_title(f"{ROW_LABELS[family]}: min/max O-information", loc="left", fontsize=10)
        ax_o.set_xlabel("Set size")
        ax_o.set_ylabel("O-information (nats)")
        if show_legend and row_i == 0:
            ax_o.legend(fontsize=6, ncol=2)

        perf = _family_performance(summary, family, rungs, top_k)
        for rung_i, rung in enumerate(rungs):
            if rung in base_map:
                ax_p.scatter(rung_i, base_map[rung], color=RUNG_COLORS[rung], marker="_", s=180, linewidths=2.0)
            if rung in single_map:
                ax_p.scatter(rung_i, single_map[rung], color=C_SINGLE, marker="*", s=58)

        for series, color, marker, label in [
            ("top20_syn", C_SYN, "o", "Top-20 synergistic"),
            ("top20_red", C_RED, "s", "Top-20 redundant"),
            ("best_exhaustive", C_BEST, "^", "Best exhaustive"),
        ]:
            pp = perf[perf["series"] == series].set_index("rung_id").reindex(rungs)
            y = pp["median_r2"].to_numpy(dtype=float)
            ax_p.plot(x, y, color=color, marker=marker, lw=1.8, label=label)
            if series in {"top20_syn", "top20_red"}:
                lo = pp["min_r2"].to_numpy(dtype=float)
                hi = pp["max_r2"].to_numpy(dtype=float)
                ax_p.fill_between(x, lo, hi, color=color, alpha=0.12)
        if original_map:
            original_y = np.array([original_map.get(rung, np.nan) for rung in rungs], dtype=float)
            ax_p.plot(
                x,
                original_y,
                color=C_ORIGINAL,
                marker="X",
                lw=1.7,
                ms=6,
                label="Original complete best",
            )
        ax_p.set_xticks(x)
        ax_p.set_xticklabels([LEVEL_LABELS[r] for r in rungs], rotation=0)
        ax_p.set_ylabel("R² LOCO")
        ax_p.set_title(f"{ROW_LABELS[family]}: performance by model level", loc="left", fontsize=10)
        if show_legend and row_i == 0:
            handles, labels = ax_p.get_legend_handles_labels()
            handles.extend(
                [
                    mpl.lines.Line2D([0], [0], color="#666666", marker="_", linestyle="None", markersize=12, markeredgewidth=2),
                    mpl.lines.Line2D([0], [0], color=C_SINGLE, marker="*", linestyle="None", markersize=8),
                ]
            )
            labels.extend(["Baseline", "Best single exposure"])
            ax_p.legend(handles, labels, fontsize=7)
        style_axis(ax_p)

        perf_frame = perf[perf["rung_id"].isin(rungs)][
            [c for c in ["rung_id", "series", "median_r2", "min_r2", "max_r2"] if c in perf.columns]
        ].copy()
        perf_frame["baseline_r2"] = perf_frame["rung_id"].map(base_map)
        perf_frame["best_single_exposure_r2"] = perf_frame["rung_id"].map(single_map)
        perf_frame["original_complete_best_r2"] = perf_frame["rung_id"].map(original_map)
        perf_frame = perf_frame.sort_values(["series", "rung_id"]).reset_index(drop=True)
        perf_frame = perf_frame.rename(columns={"rung_id": "model_level"})
        perf_frame["model_level"] = perf_frame["model_level"].map(LEVEL_LABELS)
        panels.append(
            Panel(
                panel_id=f"{row_letter}{2 * row_i + 2}_{BAG_SHORT[bag]}_{FAMILY_SHORT[family]}_perf",
                frame=perf_frame,
                description=(
                    f"Held-out LOCO R² by model level for the {ROW_LABELS[family]} candidate "
                    f"construction, {bag} BAG."
                ),
                columns={
                    "model_level": "model level (OLS, d1, d2, d3)",
                    "series": "top20_syn, top20_red, or best_exhaustive",
                    "median_r2": "median held-out LOCO R² for that series (plotted marker)",
                    "min_r2": "minimum held-out LOCO R² (lower bound of shaded band)",
                    "max_r2": "maximum held-out LOCO R² (upper bound of shaded band)",
                    "baseline_r2": "covariate-only baseline for that level",
                    "best_single_exposure_r2": "best single-exposure model for that level",
                    "original_complete_best_r2": "original complete-pool best model for that level",
                },
                notes=f"Top-K per series: {top_k}.",
            )
        )

    return panels


def _plot_domain_imbalance(bag_inputs: dict[str, dict], outdir: Path, rungs: list[str], top_k: int) -> None:
    """One figure for both BAGs: structural row first, then functional.

    Each BAG is a single row of 4 columns (2 candidate families x O-info +
    performance), so the merged figure is 2 rows. No suptitle.
    """
    bags = [b for b in BAG_ROW_ORDER if b in bag_inputs]
    if not bags:
        return
    ncols = 2 * len(_families_present(bag_inputs[bags[0]]["candidate_scores"]))
    fig, axes = plt.subplots(
        len(bags), ncols,
        figsize=(6.75 * ncols, 4.0 * len(bags)),
        gridspec_kw={"wspace": 0.30, "hspace": 0.42},
        squeeze=False,
    )
    panels: list[Panel] = []
    for i, bag in enumerate(bags):
        d = bag_inputs[bag]
        panels.extend(
            _draw_domain_imbalance_row(
                bag, d["summary"], d["candidate_scores"], d["best_single"],
                d["baselines"], d["original_best"], rungs, top_k,
                axes[i], ROW_LETTERS[i], show_legend=(i == 0),
            )
        )
    stem = "domain_imbalance_sensitivity"
    save_figure(fig, stem, outdir)
    plt.close(fig)
    write_source_data(stem, panels, outdir)


def main() -> None:
    cfg = load_sensitivity_config()
    smoke = env_bool("SMOKE_TEST") or env_bool("SENSITIVITY_SMOKE")
    include_combined = env_bool("SENSITIVITY_INCLUDE_COMBINED", default=False)
    rungs = active_rungs(cfg)
    if smoke:
        rungs = rungs[:2]
    analysis_cfg = analysis_cfg_from_config(cfg)
    local_root = repo_sensitivity_root(cfg) / "domain_imbalance"
    local_root.mkdir(parents=True, exist_ok=True)

    raw, domains, feature_names, _domain_map = load_raw_and_domains()
    domain_order = domains.groupby("domain", sort=True).size().index.astype(str).tolist()
    original_model = build_original_model_df(raw, feature_names, analysis_cfg)
    greedy_reference_df, greedy_reference_summary = load_greedy_reference_exposome()
    missing_ref = [c for c in feature_names if c not in greedy_reference_df.columns]
    if missing_ref:
        raise ValueError(f"Greedy reference matrix is missing expected exposome features: {missing_ref[:10]}")
    top_k = int(cfg["domain_imbalance"].get("top_k_plot", 20))
    order_min = int(cfg["domain_imbalance"].get("order_min", 3))
    order_max = int(cfg["domain_imbalance"].get("order_max", 10))
    batch_size = int(cfg["domain_imbalance"].get("thoi_batch_size", 100000))
    n_jobs = int(os.environ.get("SENSITIVITY_N_JOBS", "1"))
    max_candidates = _max_candidates()
    if smoke and max_candidates is None:
        max_candidates = 12
    bags = selected_bags(cfg, include_combined=include_combined)

    log_msg("PHASE 1 candidate definition + THOI scoring starts")
    pc_model, reference_pc_df, pc_variance = build_domain_pc_model_from_reference(
        original_model,
        greedy_reference_df,
        domains,
        feature_names,
    )
    pc_names = pc_variance["feature_name"].astype(str).tolist()
    # Raw feature groups per domain, in the same domain order as pc_names, so the
    # evaluator can refit each domain's PC1 per LOCO fold (training rows only).
    domain_groups = [
        [c for c in sub["feature_name"].astype(str).tolist() if c in feature_names]
        for _, sub in domains.groupby("domain", sort=True)
    ]
    # Carry raw features alongside the (descriptive) reference-fit PC columns.
    pc_eval_model = pc_model.join(original_model[feature_names])
    pc_candidates_template = combo_candidate_table(
        pc_names,
        family="within_domain_pc1",
        source_label="within_domain_pc1",
        order_min=order_min,
        order_max=order_max,
        prefix="domainpc",
    )
    pc_score_cache = local_root / "global_hoi_cache" / "within_domain_pc1_thoi_scores.csv"
    pc1_reference = os.environ.get("DOMAIN_PC1_OINFO_REFERENCE", "").strip()
    if env_bool("MAIN_K10_MODE") and not pc1_reference:
        raise EnvironmentError(
            "MAIN_K10_MODE requires DOMAIN_PC1_OINFO_REFERENCE; PC1 O-information is reused"
        )
    if pc1_reference:
        pc_candidates_template = _reuse_domain_pc1_oinfo(
            pc_candidates_template, Path(pc1_reference).resolve()
        )
    else:
        pc_candidates_template = score_oinfo_with_cache(
            reference_pc_df,
            pc_candidates_template,
            pc_names,
            cache_path=pc_score_cache,
            batch_size=batch_size,
        )

    original_score_cache = local_root / "global_hoi_cache" / "original_feature_thoi_scores.csv"
    log_msg("PHASE 1 random one-per-domain candidates skipped by request")

    candidate_defs: dict[str, dict[str, pd.DataFrame]] = {}
    best_single_sources: dict[str, pd.DataFrame] = {}
    for bag in bags:
        log_msg(f"PHASE 1 defining candidates bag={bag}")
        bag_dir = local_root / bag
        cand_dir = bag_dir / "candidates"
        cand_def_dir = local_root / "candidate_definitions" / bag
        for d in [cand_dir, cand_def_dir]:
            d.mkdir(parents=True, exist_ok=True)

        best_single = load_best_single_by_rung(bag)
        if best_single.empty:
            raise FileNotFoundError(f"No single-exposure rung outputs found for bag={bag}.")
        best_single_sources[bag] = best_single
        best_single.to_csv(cand_dir / "best_single_by_rung_source.csv", index=False)
        best_single.to_csv(cand_def_dir / "best_single_by_rung_source.csv", index=False)
        pd.DataFrame([greedy_reference_summary]).to_csv(cand_dir / "hoi_greedy_reference_summary.csv", index=False)
        pd.DataFrame([greedy_reference_summary]).to_csv(cand_def_dir / "hoi_greedy_reference_summary.csv", index=False)

        best_tables = []
        for source_rung in rungs:
            reps = _best_single_representatives(best_single, source_rung, domain_order)
            best_tables.append(
                combo_candidate_table(
                    reps,
                    family="best_single_per_domain",
                    source_label=source_rung,
                    order_min=order_min,
                    order_max=order_max,
                    prefix=f"bestdom_{bag}_{source_rung}",
                    extra={"source_rung": source_rung},
                )
            )
        best_candidates = pd.concat(best_tables, ignore_index=True)
        log_msg(
            f"PHASE 1 best-single candidates bag={bag}: source_rungs={len(rungs)} "
            f"candidate_rows={len(best_candidates)} "
            f"unique_predictor_sets={best_candidates['predictors_identity'].astype(str).nunique()}"
        )
        best_candidates = score_oinfo_with_cache(
            greedy_reference_df,
            best_candidates,
            feature_names,
            cache_path=original_score_cache,
            batch_size=batch_size,
        )
        best_candidates.to_csv(cand_dir / "best_single_per_domain_candidates.csv", index=False)
        best_candidates.to_csv(cand_def_dir / "best_single_per_domain_candidates.csv", index=False)

        pc_candidates = pc_candidates_template.copy()
        pc_candidates["feature_id"] = pc_candidates["feature_id"].astype(str).radd(f"{bag}_")
        pc_candidates["candidate_id"] = pc_candidates["feature_id"]
        pc_variance.to_csv(cand_dir / "within_domain_pc1_variance.csv", index=False)
        pc_variance.to_csv(cand_def_dir / "within_domain_pc1_variance.csv", index=False)
        pc_candidates.to_csv(cand_dir / "within_domain_pc1_candidates.csv", index=False)
        pc_candidates.to_csv(cand_def_dir / "within_domain_pc1_candidates.csv", index=False)

        candidate_defs[bag] = {
            "best_single_per_domain": best_candidates,
            "within_domain_pc1": pc_candidates,
        }
        log_msg(f"PHASE 1 candidate definitions complete bag={bag}")

    log_msg("PHASE 1 candidate definition + THOI scoring complete")
    log_msg("PHASE 2 BAG LOCO association starts")

    # Both BAGs share one figure folder and one merged figure; cap split stays
    # (applied inside repo_sensitivity_figures_root).
    fig_dir = repo_sensitivity_figures_root(cfg, "domain_imbalance")
    fig_dir.mkdir(parents=True, exist_ok=True)
    bag_inputs: dict[str, dict] = {}

    for bag in bags:
        print(f"\n=== Domain-imbalance sensitivity | bag={bag} | smoke={smoke} ===")
        bag_dir = local_root / bag
        cand_dir = bag_dir / "candidates"
        eval_dir = bag_dir / "eval"
        for d in [cand_dir, eval_dir, fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        best_single = best_single_sources[bag]
        best_candidates = candidate_defs[bag]["best_single_per_domain"]
        pc_candidates = candidate_defs[bag]["within_domain_pc1"]

        total_original = len(best_candidates)
        unique_original = best_candidates["predictors_identity"].astype(str).nunique()
        log_msg(
            f"PHASE 2 best-single dedupe bag={bag}: "
            f"candidate_rows={total_original} unique_models={unique_original} "
            f"duplicates_skipped={total_original - unique_original}"
        )

        best_summary, best_country = evaluate_candidates_by_rung(
            model_df=original_model,
            candidate_df=best_candidates,
            exposome_cols=feature_names,
            bag=bag,
            rungs=rungs,
            analysis_cfg=analysis_cfg,
            outdir=eval_dir / "best_single_per_domain",
            n_jobs=n_jobs,
            max_candidates=max_candidates,
        )
        best_summary["bag"] = bag
        best_country["bag"] = bag

        pc_summary, pc_country = evaluate_candidates_by_rung(
            model_df=pc_eval_model,
            candidate_df=pc_candidates,
            exposome_cols=pc_names,
            bag=bag,
            rungs=rungs,
            analysis_cfg=analysis_cfg,
            outdir=eval_dir / "within_domain_pc1",
            n_jobs=n_jobs,
            max_candidates=max_candidates,
            fold_pca={"mode": "domain", "raw_cols": list(feature_names), "domain_groups": domain_groups},
        )
        pc_summary["bag"] = bag
        pc_country["bag"] = bag

        all_summaries = [best_summary, pc_summary]
        all_countries = [best_country, pc_country]

        summary_all = pd.concat(all_summaries, ignore_index=True)
        country_all = pd.concat(all_countries, ignore_index=True)
        candidate_scores = pd.concat([best_candidates, pc_candidates], ignore_index=True)
        summary_all.to_csv(bag_dir / "domain_imbalance_global_all.csv", index=False)
        country_all.to_csv(bag_dir / "domain_imbalance_country_all.csv", index=False)
        candidate_scores.to_csv(bag_dir / "domain_imbalance_candidate_scores.csv", index=False)
        best_by_order = _best_by_order(summary_all)
        best_by_order.to_csv(bag_dir / "domain_imbalance_best_by_order.csv", index=False)

        baselines = load_existing_baselines(bag)
        baselines.to_csv(bag_dir / "existing_baselines_by_rung.csv", index=False)
        # The "original best model" reference is cap-dependent (cap21 vs complete);
        # write it (and the cap-split figures) under the per-cap leaf so the two
        # variants never overwrite each other. The heavy eval + cap-invariant
        # summaries stay in bag_dir (shared, reused across caps).
        original_best = _load_original_complete_best_by_rung(bag, rungs)
        ref_dir = bag_dir / cap_split_subdir() if cap_split_subdir() else bag_dir
        ref_dir.mkdir(parents=True, exist_ok=True)
        original_best.to_csv(ref_dir / "original_complete_best_by_rung.csv", index=False)
        bag_inputs[bag] = {
            "summary": summary_all,
            "candidate_scores": candidate_scores,
            "best_single": best_single,
            "baselines": baselines,
            "original_best": original_best,
        }

    # One figure for both BAGs (structural row first), drawn after the loop.
    _plot_domain_imbalance(bag_inputs, fig_dir, rungs, top_k)

    copy_tree_contents(local_root, bundle_sensitivity_root(cfg) / "domain_imbalance")
    copy_tree_contents(repo_sensitivity_figures_root(cfg, "domain_imbalance"), bundle_sensitivity_root(cfg) / "domain_imbalance")


if __name__ == "__main__":
    main()
