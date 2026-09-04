"""
Standalone runner for single-exposure LOCO at an arbitrary rung.

Called by run_single_exposure_per_rung.sh via a fresh subprocess per rung.
Reads SINGLE_EXPOSURE_MAX_DEPTH from env (0 = OLS/gblinear, else gbtree).
Inherits all other V3_* env vars from the shell.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

# ── patch default_xgb_cfg before any other scripts import it ─────────────────
import scripts.pipeline_utils as _pu

_max_depth = int(os.environ.get("SINGLE_EXPOSURE_MAX_DEPTH", "2"))
_orig_xgb_cfg = _pu.default_xgb_cfg.__wrapped__ if hasattr(_pu.default_xgb_cfg, "__wrapped__") else _pu.default_xgb_cfg

if _max_depth == 0:
    # OLS mode: signal run_single_exposure_eval to use np.linalg.lstsq
    os.environ["SINGLE_EXPOSURE_OLS_MODE"] = "1"
else:
    def _patched_xgb_cfg():
        cfg = _orig_xgb_cfg()
        cfg["max_depth"] = _max_depth
        cfg["booster"]   = "gbtree"
        return cfg

    _pu.default_xgb_cfg = _patched_xgb_cfg

# Override n_jobs if provided
_n_jobs = os.environ.get("SINGLE_EXPOSURE_N_JOBS")
if _n_jobs:
    _orig_perf = _pu.default_perf_cfg

    def _patched_perf_cfg(stage_cfg):
        cfg = _orig_perf(stage_cfg)
        cfg["n_jobs"] = int(_n_jobs)
        return cfg

    _pu.default_perf_cfg = _patched_perf_cfg

# ── now import and run the actual script ─────────────────────────────────────
from scripts.run_single_exposure_eval import main
main()
