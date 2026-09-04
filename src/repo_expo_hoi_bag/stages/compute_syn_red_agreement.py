#!/usr/bin/env python3
"""Synergy-vs-redundancy subject-level agreement.

Reviewer question: do the best *synergistic* (o_min) and best *redundant* (o_max)
models fail / succeed on the **same subjects**, or on different ones?

Reads the subject-level out-of-fold predictions for the best synergistic and best
redundant model per BAG (subject_level_oof/{bag}/oof_predictions_best_{syn,red}.parquet),
joins them per subject, and reports:
  - per-subject signed/absolute errors for each model and their difference,
  - correlation of |error| between syn and red (do they struggle on the same subjects?),
  - win/tie/loss rates (which model is closer per subject), overall and by diagnosis / country,
  - paired Wilcoxon on |error| (syn vs red).

Inputs come from V3_OUTPUT_ROOT (set it to a bundle variant directory). Outputs to
``$V3_OUTPUT_ROOT/stats/syn_red_agreement/`` (lightweight CSVs).

Usage:
    V3_OUTPUT_ROOT=$REPRO_DATA_ROOT/runs/oinfo_only/variant_a \
        python -m scripts.compute_syn_red_agreement
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(os.environ.get("V3_OUTPUT_ROOT", "outputs/variant_a"))
OOF_DIR = ROOT / "subject_level_oof"
OUT_DIR = ROOT / "stats" / "syn_red_agreement"
BAGS = ["combined", "functional", "structural"]
WIN_EPS = 1e-9  # |error| difference below this counts as a tie


def _load(bag: str, tag: str) -> pd.DataFrame | None:
    f = OOF_DIR / bag / f"oof_predictions_best_{tag}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["err"] = df["y_true"] - df["y_pred_full"]
    df["abs_err"] = df["err"].abs()
    return df


def _merge_bag(bag: str) -> pd.DataFrame | None:
    syn, red = _load(bag, "syn"), _load(bag, "red")
    if syn is None or red is None:
        print(f"  ⚠ {bag}: missing syn or red OOF — skipping "
              f"(red needs BEST_OBJECTIVE_MODE=o_max run_best_syn_oof_predictions)")
        return None
    key = "row_id" if "row_id" in syn.columns else "N_MEGA"
    meta = [c for c in ("N_MEGA", "country", "diagnosis", "age", "sex", "y_true") if c in syn.columns]
    m = syn[[key, *meta, "err", "abs_err"]].merge(
        red[[key, "err", "abs_err"]], on=key, suffixes=("_syn", "_red"), how="inner"
    )
    m["bag"] = bag
    m["abs_err_diff"] = m["abs_err_syn"] - m["abs_err_red"]   # <0 ⇒ syn closer
    m["winner"] = np.where(m["abs_err_diff"] < -WIN_EPS, "syn",
                   np.where(m["abs_err_diff"] > WIN_EPS, "red", "tie"))
    return m


def _agreement_row(df: pd.DataFrame, bag: str, group: str, level: str) -> dict:
    n = len(df)
    vc = df["winner"].value_counts()
    # spearman/pearson of |error| between syn and red
    if n >= 3 and df["abs_err_syn"].std() > 0 and df["abs_err_red"].std() > 0:
        r_abs = float(np.corrcoef(df["abs_err_syn"], df["abs_err_red"])[0, 1])
        rho_abs = float(scipy_stats.spearmanr(df["abs_err_syn"], df["abs_err_red"]).correlation)
    else:
        r_abs = rho_abs = np.nan
    # paired Wilcoxon on |error| (syn vs red)
    try:
        w_p = float(scipy_stats.wilcoxon(df["abs_err_syn"], df["abs_err_red"]).pvalue) if n >= 10 else np.nan
    except ValueError:
        w_p = np.nan
    return {
        "bag": bag, "group_level": level, "group": group, "n": n,
        "syn_win_frac": float(vc.get("syn", 0) / n) if n else np.nan,
        "red_win_frac": float(vc.get("red", 0) / n) if n else np.nan,
        "tie_frac": float(vc.get("tie", 0) / n) if n else np.nan,
        "median_abs_err_syn": float(df["abs_err_syn"].median()),
        "median_abs_err_red": float(df["abs_err_red"].median()),
        "corr_abs_err_pearson": r_abs,
        "corr_abs_err_spearman": rho_abs,
        "wilcoxon_p_abs_err": w_p,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Synergy-vs-redundancy subject-level agreement")
    print(f"  ROOT={ROOT}")
    print("=" * 70)

    per_subject, summary = [], []
    for bag in BAGS:
        m = _merge_bag(bag)
        if m is None:
            continue
        per_subject.append(m)
        summary.append(_agreement_row(m, bag, "all", "overall"))
        if "diagnosis" in m.columns:
            for dx, sub in m.groupby("diagnosis"):
                summary.append(_agreement_row(sub, bag, str(dx), "diagnosis"))
        if "country" in m.columns:
            for ctry, sub in m.groupby("country"):
                if len(sub) >= 20:
                    summary.append(_agreement_row(sub, bag, str(ctry), "country"))

    if not per_subject:
        print("No BAGs had both syn and red OOF. Nothing written.")
        return

    ps = pd.concat(per_subject, ignore_index=True)
    ps.to_csv(OUT_DIR / "syn_red_subject_level.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT_DIR / "syn_red_agreement_summary.csv", index=False)

    print(f"\n  ✓ Saved {OUT_DIR/'syn_red_subject_level.csv'}  ({len(ps)} rows)")
    print(f"  ✓ Saved {OUT_DIR/'syn_red_agreement_summary.csv'}  ({len(sm)} rows)")
    print("\n  Overall (|error| correlation syn↔red, win fractions):")
    for _, r in sm[sm.group_level == "overall"].iterrows():
        print(f"    {r['bag']:<11} corr|err|={r['corr_abs_err_pearson']:.3f}  "
              f"syn_win={r['syn_win_frac']:.2f} red_win={r['red_win_frac']:.2f}  "
              f"medAE syn={r['median_abs_err_syn']:.3f} red={r['median_abs_err_red']:.3f}")


if __name__ == "__main__":
    main()
