#!/usr/bin/env python3
"""Run non-fitting country meta-regressions and residual mixed models on main OOF."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro-data-root", type=Path, required=True); p.add_argument("--source-run-id", default="paper_reanalysis_k10"); p.add_argument("--main-run-id", default="main"); p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def _r2(group: pd.DataFrame) -> float:
    y = group.y_true.to_numpy(float); return float(1 - np.square(y - group.y_pred_full.to_numpy(float)).sum() / np.square(y - y.mean()).sum())


def main() -> None:
    a = _args(); runtime = a.repro_data_root.resolve(); base = runtime / "results/analysis_runs" / a.source_run_id / "main_statistics/model_comparison/oof"
    frames = []
    for bag in ("structural", "functional"):
        for suffix, objective in (("syn", "o_min"), ("red", "o_max")):
            x = pd.read_parquet(base / f"level_best_{suffix}" / bag / "oof_xgb_tree_d3.parquet")
            x = x[np.isfinite(x.y_true) & np.isfinite(x.y_pred_full)].copy(); x["bag"], x["objective"], x["residual"] = bag, objective, x.y_pred_full - x.y_true; frames.append(x)
    if a.smoke_test:
        print(f"Smoke test passed: {sum(len(x) for x in frames)} main k10 OOF rows")
        return
    out = runtime / "results/analysis_runs" / a.main_run_id / "secondary_oof_stats"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True); meta_rows = []; fixed_rows = []; variance_rows = []
    for x in frames:
        bag, objective = x.bag.iloc[0], x.objective.iloc[0]
        country = x.groupby("country", observed=True).apply(lambda g: pd.Series({"country_r2": _r2(g), "n_test": len(g), "bag_sd": g.y_true.std()})).reset_index(drop=True)
        country["log_n_test"] = np.log(country.n_test)
        model = smf.ols("country_r2 ~ log_n_test + bag_sd", data=country).fit()
        for term, value in model.params.items(): meta_rows.append({"bag": bag, "objective": objective, "term": term, "estimate": value, "p_value": model.pvalues[term], "n_countries": len(country), "r_squared": model.rsquared})
        x = x[x.diagnosis.isin(["CN", "AD", "FTD"])].copy()
        fit = smf.mixedlm("residual ~ age + C(sex) + C(diagnosis)", x, groups=x["country"]).fit(reml=False, method="lbfgs", disp=False)
        for term, value in fit.params.items(): fixed_rows.append({"bag": bag, "objective": objective, "term": term, "estimate": value, "p_value_wald": fit.pvalues.get(term, np.nan), "n_subjects": len(x), "n_countries": x.country.nunique()})
        variance_rows.append({"bag": bag, "objective": objective, "country_random_intercept_variance": float(fit.cov_re.iloc[0, 0]), "residual_variance": float(fit.scale), "fit_converged": bool(fit.converged)})
    pd.DataFrame(meta_rows).to_csv(out / "ST16_country_meta_regression.csv", index=False); pd.DataFrame(fixed_rows).to_csv(out / "ST12_residual_mixedlm_fixed_effects.csv", index=False); pd.DataFrame(variance_rows).to_csv(out / "ST12_residual_mixedlm_variance.csv", index=False)
    print(f"Saved secondary OOF statistics: {out}")


if __name__ == "__main__": main()
