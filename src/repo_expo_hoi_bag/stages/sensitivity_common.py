from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from loco_fusion_matrix_engine import (
    build_model_df,
    encode_dummies,
    parse_literal_list,
    pin_blas_threads,
    prepare_analysis_table,
    prepare_bag_context,
    regression_metrics_extended,
    target_map,
)
from oinfo_bag_ladder.rungs import build_xgb_cfg_for_rung, get_rung_specs
try:
    # Compatibility workers import these as ``scripts.*`` from the copied
    # runtime. Spawned joblib workers in the active checkout instead receive
    # the stages directory directly on PYTHONPATH; support both layouts.
    from scripts.pipeline_utils import load_stage_config
    from scripts.run_exposome_greedy_only import prepare_exposome_matrix
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.pipeline_utils", "scripts.run_exposome_greedy_only"}:
        raise
    from pipeline_utils import load_stage_config
    from run_exposome_greedy_only import prepare_exposome_matrix
from xgb_loco_engine import _append_predictors, _build_base_fold_mats, _fit_xgb_fold, require_xgboost
from xgb_nested_loco_tuning import resolve_tuned_fold_configs


REPO_ROOT = Path(__file__).resolve().parents[1]
# When a stage runs through the CLI, its code is COPIED into
# $REPRO_DATA_ROOT/work/compatibility_runtime/ before execution, so REPO_ROOT
# (used for INPUT files -- config, raw data, domain metadata -- that only exist
# in that runtime layout) resolves in the external runtime. That is correct for inputs.
#
# Parent delivery contract: lightweight figures and tables are delivered in the
# checkout, while fold-level evaluation, OOF, checkpoints and logs stay in the
# external runtime.  The CLI injects REPO_CHECKOUT_ROOT for copied stages.
CHECKOUT_ROOT = Path(
    os.environ.get("REPO_CHECKOUT_ROOT", "").strip() or REPO_ROOT.parents[1]
)
_RUNTIME_CONFIG_PATH = REPO_ROOT / "config" / "sensitivity.yaml"
_SOURCE_CONFIG_PATH = Path(__file__).resolve().parent / "resources" / "sensitivity.yaml"
CONFIG_PATH = (
    _RUNTIME_CONFIG_PATH if _RUNTIME_CONFIG_PATH.is_file() else _SOURCE_CONFIG_PATH
)
DOMAIN_CSV = CHECKOUT_ROOT / "data" / "metadata" / "exposome_feature_domains.csv"
RAW_CSV = CHECKOUT_ROOT / "data" / "raw" / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"

ACTIVE_RUNGS = ["ols", "xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
RUNG_LABELS = {
    "ols": "OLS",
    "xgb_tree_d1": "XGB d1",
    "xgb_tree_d2": "XGB d2",
    "xgb_tree_d3": "XGB d3",
}
RUNG_COLORS = {
    "ols": "#4d4d4d",
    "xgb_tree_d1": "#1b9e77",
    "xgb_tree_d2": "#2166ac",
    "xgb_tree_d3": "#b2182b",
}
BAG_ORDER = ["functional", "structural", "combined"]

_MAIN_K10_HPO_ENV = {
    "baseline": "XGB_TUNING_BASELINE_ARTIFACT",
    "single_exposure": "XGB_TUNING_SINGLE_ARTIFACT",
    "k10": "XGB_TUNING_ARTIFACT",
}
_MAIN_K10_HPO_FEATURE_SCOPE = {
    "baseline": "baseline",
    "single_exposure": "single_exposure",
    "k10": "domain_balanced_k10",
}


@dataclass(frozen=True)
class PaperAnalysisConfig:
    """Validated shared settings for all paper-facing bias analyses."""

    primary_diagnoses: tuple[str, ...]
    control_diagnosis: str
    diagnosis_labels: dict[str, str]
    objectives: tuple[str, ...]
    objective_metadata: dict[str, dict[str, str]]
    primary_bags: tuple[str, ...]
    residual_bias_bags: tuple[str, ...]
    deployed_rung: str
    order_max: int
    reference_oof_dir_template: str
    country_meta_regression_objective: str
    residual_bias: dict
    diagnosis_balance: dict
    normative_transfer: dict
    residual_confounds: dict


def log_msg(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value; got {value!r}")


def dedup_namespace() -> str:
    """Namespace label for the dedup re-analysis output subtree.

    Defaults to ``dedup``. The driver may set ``DEDUP_NS`` (e.g. ``dedup_neg_o``)
    to route a parallel variant -- such as the negative-O-info synergy criterion --
    into its own ``outputs/figures/<ns>/``, ``outputs/sensitivity/<ns>/`` and
    ``outputs/<ns>/`` trees so it never overwrites the existing dedup outputs.
    """
    return os.environ.get("DEDUP_NS", "").strip() or "dedup"


def syn_oinfo_negative() -> bool:
    """Whether "synergistic" requires a NEGATIVE evaluated O-information.

    Off (default): a candidate is synergistic iff it was discovered under the
    greedy ``o_min`` objective, regardless of the sign of its evaluated O-info
    (the path-only criterion). On (``SYN_OINFO_NEGATIVE=1``): an ``o_min``
    candidate additionally must have evaluated O-info < 0 to count as
    synergistic. Redundancy (``o_max``) is never filtered by this flag.
    """
    return env_bool("SYN_OINFO_NEGATIVE")


def filter_syn_pool(df: "pd.DataFrame", objective: str, oinfo_col: str) -> "pd.DataFrame":
    """Apply the negative-O-info synergy filter to an ``o_min`` candidate pool.

    No-op unless ``syn_oinfo_negative()`` is on AND ``objective == "o_min"``.
    When active, drops rows whose evaluated O-info (``oinfo_col``) is not < 0.
    ``oinfo_col`` is ``"thoi_o"`` for canonical-metrics parquets and ``"score"``
    for the normative / LORO / greedy CSVs. The column MUST exist when the flag
    is on (fail loud rather than silently skipping the filter)."""
    if not syn_oinfo_negative() or str(objective) != "o_min":
        return df
    if df.empty:
        # Nothing to filter (e.g. a bag with no greedy o_min pool, like combined).
        return df
    if oinfo_col not in df.columns:
        raise KeyError(
            f"SYN_OINFO_NEGATIVE is set but O-info column {oinfo_col!r} is absent "
            f"(columns: {list(df.columns)}); cannot apply the synergy sign filter."
        )
    return df[pd.to_numeric(df[oinfo_col], errors="coerce") < 0].copy()


def order_cap() -> int | None:
    """The active interaction-order cap, or None for the full pool.

    Read from PAPER_FIG_ORDER_MAX (the same env var the paper-figure generators
    and run_best_syn_oof_predictions honour). When set, best-model-per-rung
    selection in the cap-dependent sensitivity analyses (domain_imbalance,
    country_region, whole_exposome_pca) restricts candidates to order <= cap
    BEFORE picking the best, so the cap21 / complete (cap30) variants compare
    against the correct best model. Per-candidate evaluation is unaffected (the
    heavy eval CSVs already span the full pool and are reused)."""
    raw = os.environ.get("PAPER_FIG_ORDER_MAX", "").strip()
    return int(raw) if raw else None


def order_cap_suffix() -> str:
    """`max_<cap>` when a cap is active, else `complete`. Used as the per-cap
    subdirectory name for dedup sensitivity outputs (cap21 / complete split)."""
    cap = order_cap()
    return f"max_{cap}" if cap is not None else "complete"


def apply_order_cap(df: pd.DataFrame, order_col: str = "order") -> pd.DataFrame:
    """Filter a candidate/metrics frame to order <= PAPER_FIG_ORDER_MAX.

    No-op when the cap is unset or the frame lacks the order column. Returns a
    copy so callers can mutate freely."""
    cap = order_cap()
    if cap is None or order_col not in df.columns:
        return df
    keep = pd.to_numeric(df[order_col], errors="coerce") <= cap
    return df[keep].copy()


def _ols_dx_interactions() -> bool:
    """Whether the OLS rung should include diagnosis x exposome interactions
    (to match the main pipeline's OLS). Env-gated; default off for reproducibility."""
    return env_bool("SENSITIVITY_OLS_DX_INTERACTIONS", default=False)


def load_sensitivity_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Main k10 sensitivity jobs preserve the parent-stage implementation while
    # switching only the explicitly requested BAG population and model rungs.
    # OLS is deliberately absent: it is an immutable reused reference, never a
    # sensitivity refit.
    if env_bool("MAIN_K10_MODE"):
        run_id = os.environ.get("MAIN_K10_CLUSTER_RUN_ID", "").strip()
        if not run_id:
            raise ValueError("MAIN_K10_CLUSTER_RUN_ID is required when MAIN_K10_MODE is enabled")
        defaults = config["defaults"]
        defaults["exclude_countries"] = ["France", "Italy", "Egypt", "Greece", "Poland"]
        defaults["exclude_diagnosis"] = ["Other", "AFM", "MCI"]
        defaults["active_rungs"] = ["xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"]
        defaults["primary_bags"] = ["structural", "functional"]
        defaults["optional_bags"] = []
        defaults["output_root"] = f"outputs/main/{run_id}/sensitivity"
        defaults["bundle_sensitivity_subdir"] = f"results/analysis_runs/{run_id}/sensitivity_bundle"
    return config


def paper_analysis_config(cfg: dict) -> PaperAnalysisConfig:
    """Return the single source of truth for paper-facing analysis choices."""
    try:
        section = cfg["paper_analysis"]
        defaults = cfg["defaults"]
        diagnoses = tuple(str(value) for value in section["primary_diagnoses"])
        control = str(section["control_diagnosis"])
        diagnosis_labels = {
            str(diagnosis): str(label)
            for diagnosis, label in section["diagnosis_labels"].items()
        }
        objectives = tuple(str(value) for value in section["objectives"])
        objective_metadata = {
            str(objective): {
                "file_suffix": str(values["file_suffix"]),
                "arm_label": str(values["arm_label"]),
            }
            for objective, values in section["objective_metadata"].items()
        }
        primary_bags = tuple(str(value) for value in defaults["primary_bags"])
        residual_bias_bags = tuple(
            str(value) for value in section["residual_bias_bags"]
        )
        deployed_rung = str(section["deployed_rung"])
        order_max_value = int(section["order_max"])
        reference_template = str(section["reference_oof_dir_template"])
        meta_objective = str(section["country_meta_regression_objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or incomplete paper_analysis configuration") from exc

    if not diagnoses or len(set(diagnoses)) != len(diagnoses):
        raise ValueError("paper_analysis.primary_diagnoses must be unique and non-empty")
    if control not in diagnoses:
        raise ValueError("paper_analysis.control_diagnosis must be a primary diagnosis")
    if set(diagnosis_labels) != set(diagnoses):
        raise ValueError("paper_analysis.diagnosis_labels must cover every diagnosis")
    if set(objectives) != {"o_min", "o_max"}:
        raise ValueError("paper_analysis.objectives must contain o_min and o_max exactly")
    if set(objective_metadata) != set(objectives):
        raise ValueError("paper_analysis.objective_metadata must cover every objective")
    suffixes = {
        values["file_suffix"] for values in objective_metadata.values()
    }
    if len(suffixes) != len(objectives):
        raise ValueError("paper objective file suffixes must be unique")
    known_bags = set(BAG_ORDER)
    if not primary_bags or not set(primary_bags).issubset(known_bags):
        raise ValueError("defaults.primary_bags contains an unsupported BAG")
    if not residual_bias_bags or not set(residual_bias_bags).issubset(known_bags):
        raise ValueError("paper_analysis.residual_bias_bags contains an unsupported BAG")
    configured_rungs = {str(value) for value in defaults["active_rungs"]}
    known_rungs = {str(spec["rung_id"]) for spec in get_rung_specs()}
    if deployed_rung not in configured_rungs or deployed_rung not in known_rungs:
        raise ValueError(
            "paper_analysis.deployed_rung must be an active, defined model rung"
        )
    if deployed_rung == "ols":
        raise ValueError("paper_analysis.deployed_rung must be an XGBoost rung")
    if order_max_value < 1:
        raise ValueError("paper_analysis.order_max must be positive")
    if "{order_max}" not in reference_template:
        raise ValueError(
            "paper_analysis.reference_oof_dir_template must contain {order_max}"
        )
    if meta_objective not in objectives:
        raise ValueError(
            "paper_analysis.country_meta_regression_objective is not configured"
        )

    return PaperAnalysisConfig(
        primary_diagnoses=diagnoses,
        control_diagnosis=control,
        diagnosis_labels=diagnosis_labels,
        objectives=objectives,
        objective_metadata=objective_metadata,
        primary_bags=primary_bags,
        residual_bias_bags=residual_bias_bags,
        deployed_rung=deployed_rung,
        order_max=order_max_value,
        reference_oof_dir_template=reference_template,
        country_meta_regression_objective=meta_objective,
        residual_bias=dict(section["residual_bias"]),
        diagnosis_balance=dict(section["diagnosis_balance"]),
        normative_transfer=dict(section["normative_transfer"]),
        residual_confounds=dict(section["residual_confounds"]),
    )


def bundle_root() -> Path:
    root = os.environ.get("REPRO_DATA_ROOT", "").strip()
    if not root:
        raise EnvironmentError(
            "REPRO_DATA_ROOT must point to the external data bundle for sensitivity runs "
            "because these scripts reuse existing canonical baselines and single-exposure outputs."
        )
    return Path(root)


def bundle_variant_root() -> Path:
    return bundle_root() / "runs" / "oinfo_only" / "variant_a"


def canonical_root() -> Path:
    main_k10_root = os.environ.get("MAIN_K10_CANONICAL_ROOT", "").strip()
    if main_k10_root:
        root = Path(main_k10_root).resolve()
        marker = root / "main_k10_adapter_manifest.json"
        if not marker.is_file():
            raise FileNotFoundError(
                "MAIN_K10_CANONICAL_ROOT must point to a prepared main-k10 adapter "
                f"with its manifest: {marker}"
            )
        return root
    return bundle_variant_root() / "families" / "pooled_oinfo_ladder" / "canonical"


def repo_sensitivity_root(cfg: dict) -> Path:
    """Checkout destination for lightweight sensitivity tables."""
    root = CHECKOUT_ROOT / str(cfg["defaults"].get("output_root", "outputs/sensitivity"))
    # Dedup re-analysis writes its lightweight CSVs to a parallel dedup/ subtree
    # (README "outputs/ layout"), never mixing with canonical sensitivity outputs.
    if env_bool("SENSITIVITY_DEDUP"):
        root = root / dedup_namespace()
    root.mkdir(parents=True, exist_ok=True)
    return root


def cap_split_subdir() -> str:
    """Per-cap leaf (`cap21` / `complete`) appended AFTER the analysis name for
    cap-dependent dedup analyses, or "" when not splitting. The best-model-per-rung
    baselines in domain_imbalance / country_region / whole_exposome_pca change with
    the order cap, so their dedup outputs land under <name>/{cap21,complete}/.
    Gated by SENSITIVITY_CAP_SPLIT so the cap-agnostic order_cap sweep stays flat.

    The leaf label is decoupled from the selection cap: CAP_SPLIT_LABEL lets the
    driver name the cap-30 pass `complete` (the full computed extent) while still
    selecting with PAPER_FIG_ORDER_MAX=30. Falls back to `cap<cap>` / `complete`."""
    if not (env_bool("SENSITIVITY_DEDUP") and env_bool("SENSITIVITY_CAP_SPLIT")):
        return ""
    label = os.environ.get("CAP_SPLIT_LABEL", "").strip()
    if label:
        return label
    cap = order_cap()
    return f"cap{cap}" if cap is not None else "complete"


def repo_sensitivity_figures_root(cfg: dict, name: str) -> Path:
    """Checkout destination for sensitivity figures, kept under
    outputs/figures/sensitivity/<name>/ -- never inside repo_sensitivity_root()
    (which is also used for CSVs/lightweight tables). Mirrors the split already
    used by compute_order_cap_reconciliation.py's OUT_DIR/FIGURES_DIR pair.
    """
    # Dedup figures land under outputs/figures/dedup/sensitivity/<name>;
    # canonical figures stay under outputs/figures/sensitivity.
    main_run_id = os.environ.get("MAIN_K10_CLUSTER_RUN_ID", "").strip()
    if env_bool("MAIN_K10_MODE"):
        root = CHECKOUT_ROOT / "outputs" / "main" / main_run_id / "sensitivity" / "figures" / name
    elif env_bool("SENSITIVITY_DEDUP"):
        root = CHECKOUT_ROOT / "outputs" / "figures" / dedup_namespace() / "sensitivity" / name
        leaf = cap_split_subdir()
        if leaf:
            root = root / leaf
    else:
        root = CHECKOUT_ROOT / "outputs" / "figures" / "sensitivity" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def sensitivity_eval_work_root(cfg: dict, name: str) -> Path:
    """External scratch root for heavy per-fold LOCO evaluation outputs.

    Analyses that re-evaluate the candidate pool route these artifacts to
    ${REPRO_DATA_ROOT}/work/sensitivity_eval/<name>/.
    """
    # The heavy per-fold LOCO eval is SELECTION-INDEPENDENT (it scores the full
    # candidate pool regardless of the synergy criterion), so it is NOT namespaced:
    # a dedup_neg_o re-derivation resumes the same evaluation and only redoes the
    # selection and summaries with the negative-O filter.
    # The main k10 re-analysis must never resume or overwrite an earlier
    # sensitivity cache.  Its candidate universe and five-country population
    # differ from the generic sensitivity runs, so an immutable run namespace
    # is required even when the analysis name is the same.
    if env_bool("MAIN_K10_MODE"):
        run_id = os.environ.get("MAIN_K10_CLUSTER_RUN_ID", "").strip()
        if not run_id:
            raise ValueError("MAIN_K10_CLUSTER_RUN_ID is required when MAIN_K10_MODE is enabled")
        root = bundle_root() / "work" / "analysis_runs" / run_id / "sensitivity_eval" / name
    else:
        root = bundle_root() / "work" / "sensitivity_eval" / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def bundle_sensitivity_root(cfg: dict) -> Path:
    # SENSITIVITY_BUNDLE_SUBDIR lets execution wrappers route heavy outputs to a
    # dedicated folder (e.g. "sensitivity_repro") to keep the reproduction runs clean.
    subdir = os.environ.get("SENSITIVITY_BUNDLE_SUBDIR", "").strip() or str(
        cfg["defaults"].get("bundle_sensitivity_subdir", "sensitivity")
    )
    root = bundle_root() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def baseline_candidate_df() -> pd.DataFrame:
    """A single empty-predictor candidate whose LOCO R2 is the covariate-only
    baseline. Evaluated inside each analysis it recomputes the baseline on that
    analysis's own sample (e.g. CN-only) and fold scheme (e.g. region LORO)."""
    return pd.DataFrame(
        [
            {
                "candidate_id": "__baseline__",
                "feature_id": "__baseline__",
                "objective": "baseline",
                "order": 0,
                "rank": 0,
                "score": np.nan,
                "nplet_vars": [],
                "predictors_identity": "",
                "predictors_identity_n": 0,
                "candidate_family": "baseline",
                "source_label": "baseline",
            }
        ]
    )


def candidate_hpo_scope(row: dict | pd.Series) -> str:
    """Return the frozen-HPO family required by one main-k10 candidate."""
    values = {
        str(row.get(name, "")).strip().lower()
        for name in ("candidate_id", "candidate_family", "source_label", "objective")
    }
    if "__baseline__" in values or "baseline" in values:
        return "baseline"
    if (
        "single_exposure" in values
        or "single" in values
        or any(value.startswith(("__single__", "single_")) for value in values)
    ):
        return "single_exposure"
    return "k10"


def _validated_hpo_identity(path: Path, expected_feature_scope: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen HPO artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_scope = str(payload.get("provenance", {}).get("feature_scope", ""))
    if actual_scope != expected_feature_scope:
        raise ValueError(
            f"HPO artifact {path} declares feature_scope={actual_scope!r}; "
            f"expected {expected_feature_scope!r}"
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_main_k10_hpo(
    candidate_df: pd.DataFrame,
    default_artifact_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Annotate candidates and validate scope-specific main-k10 HPO artifacts."""
    routed = candidate_df.copy()
    routed["hpo_scope"] = routed.apply(candidate_hpo_scope, axis=1)
    artifacts: dict[str, Path] = {}
    identities: dict[str, str] = {}
    for scope in sorted(set(routed["hpo_scope"].astype(str))):
        env_name = _MAIN_K10_HPO_ENV[scope]
        raw_path = (
            str(default_artifact_path)
            if scope == "k10"
            else os.environ.get(env_name, "").strip()
        )
        if not raw_path:
            raise ValueError(
                f"MAIN_K10_MODE requires {env_name} for {scope} candidates"
            )
        path = Path(raw_path).resolve()
        identities[scope] = _validated_hpo_identity(
            path, _MAIN_K10_HPO_FEATURE_SCOPE[scope]
        )
        artifacts[scope] = path
    routed["hpo_artifact_sha256"] = routed["hpo_scope"].map(identities)
    return routed, artifacts


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, target)


def analysis_cfg_from_config(cfg: dict) -> dict:
    d = cfg["defaults"]
    return {
        "objectives": ["o_max", "o_min"],
        "initial_order": 3,
        "max_order": 30,
        "top_k_per_order": 100,
        "max_candidates_per_objective": 0,
        "max_models_total": 0,
        "objective_filter": [],
        "order_min": None,
        "order_max": 30,
        "include_sex": True,
        "include_diagnosis": True,
        "include_year": True,
        "min_n_obs": 80,
        "min_n_obs_for_metrics": 5,
        "exclude_countries": list(d.get("exclude_countries", [])),
        "exclude_diagnosis": list(d.get("exclude_diagnosis", [])),
    }


def cv_cfg() -> dict:
    return {
        "split_col": "country_clean",
        "min_country_size_test": 0,
        "unseen_diag_policy": "drop_test_rows",
    }


def early_stop_cfg() -> dict:
    return {"val_country_frac": 0.20, "val_country_min": 1}


def load_raw_and_domains() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    domains = pd.read_csv(DOMAIN_CSV)
    feature_names = domains["feature_name"].astype(str).tolist()
    domain_map = dict(zip(domains["feature_name"].astype(str), domains["domain"].astype(str)))
    missing = [c for c in feature_names if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw cohort table is missing domain features: {missing[:10]}")
    return raw, domains, feature_names, domain_map


def load_greedy_reference_exposome() -> tuple[pd.DataFrame, dict]:
    """Return the exact exposome matrix used by the greedy candidate generator.

    This intentionally does not filter by BAG availability or by downstream LOCO
    analysis exclusions. It mirrors scripts.run_exposome_greedy_only so the
    O-information values in sensitivity candidates are generated from the same
    subject-feature matrix as the original greedy analysis.
    """
    greedy_cfg = load_stage_config("greedy")
    # Effective-sample correction: the exposome is assigned at country-year level and
    # replicated across subjects, so O-information computed on the ~18k subject rows
    # uses a ~70x inflated effective n. When DEDUP_EXPOSOME_CSV is set, read the
    # pre-built deduplicated matrix (the exact effective-sample file the dedup greedy
    # consumed -- 259 unique signatures), instead of re-deriving it on the fly. Only
    # structure-on-X (O-info, domain PCA loadings) uses this; LOCO evaluation stays
    # subject-level.
    _dedup_csv = os.environ.get("DEDUP_EXPOSOME_CSV", "").strip()
    if env_bool("MAIN_K10_MODE") and not _dedup_csv:
        raise EnvironmentError(
            "MAIN_K10_MODE requires DEDUP_EXPOSOME_CSV for exposome-structure analyses"
        )
    input_csv = Path(_dedup_csv) if _dedup_csv else REPO_ROOT / str(greedy_cfg["input_csv"])
    X, feature_idx, summary = prepare_exposome_matrix(
        input_csv=input_csv,
        feature_list_csv=REPO_ROOT / str(greedy_cfg["feature_list_csv"]),
        country_col=str(greedy_cfg["country_col"]),
        year_acq_col=str(greedy_cfg["year_acq_col"]),
        year_fallback_col=str(greedy_cfg["year_fallback_col"]),
        countries_to_remove=[
            c.strip() for c in str(greedy_cfg.get("countries_to_remove", "")).split(",") if c.strip()
        ],
        outlier_ids=[x.strip() for x in str(greedy_cfg.get("outlier_ids", "")).split(",") if x.strip()],
        year_min=greedy_cfg.get("year_min"),
        year_max=greedy_cfg.get("year_max"),
        min_country_size=int(greedy_cfg.get("min_country_size", 0)),
    )
    ref = pd.DataFrame(X, columns=feature_idx.astype(str).tolist())
    if _dedup_csv:
        summary = {**summary, "dedup_structure_csv": str(input_csv), "dedup_structure_rows": int(len(ref))}
    return ref, summary


def build_original_model_df(raw: pd.DataFrame, feature_names: list[str], analysis_cfg: dict) -> pd.DataFrame:
    return build_model_df(raw, raw[feature_names].copy(), analysis_cfg)


def filtered_rows_for_bag(model_df: pd.DataFrame, bag: str, analysis_cfg: dict) -> pd.DataFrame:
    y_col = target_map(bag)[bag]
    return prepare_analysis_table(model_df, y_col, analysis_cfg, cv_cfg())


def selected_bags(cfg: dict, include_combined: bool) -> list[str]:
    bags = list(cfg["defaults"].get("primary_bags", ["functional", "structural"]))
    if include_combined:
        bags += list(cfg["defaults"].get("optional_bags", ["combined"]))
    requested = os.environ.get("SENSITIVITY_BAGS", "").strip()
    if requested:
        bags = [b.strip() for b in requested.split(",") if b.strip()]
    return [b for b in BAG_ORDER if b in set(bags)]


def active_rungs(cfg: dict) -> list[str]:
    rungs = [r for r in cfg["defaults"].get("active_rungs", ACTIVE_RUNGS) if r in ACTIVE_RUNGS]
    requested = os.environ.get("SENSITIVITY_RUNGS", "").strip()
    if requested:
        rungs = [r.strip() for r in requested.split(",") if r.strip()]
    bad = [r for r in rungs if r not in ACTIVE_RUNGS]
    if bad:
        raise ValueError(f"Unsupported new-analysis rung(s): {bad}.")
    return rungs


def standardize_fit_transform(
    model_df: pd.DataFrame,
    fit_index: pd.Index,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_values = model_df.loc[fit_index, feature_names].to_numpy(dtype=float)
    mean = np.nanmean(fit_values, axis=0)
    std = np.nanstd(fit_values, axis=0, ddof=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    all_values = model_df[feature_names].to_numpy(dtype=float)
    z_all = (all_values - mean) / std
    z_fit = (fit_values - mean) / std
    return z_all, z_fit, std


def build_domain_pc_model_df(
    model_df: pd.DataFrame,
    fit_index: pd.Index,
    domains: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pc_cols: dict[str, np.ndarray] = {}
    variance_rows: list[dict] = []
    for domain, sub in domains.groupby("domain", sort=True):
        cols = [c for c in sub["feature_name"].astype(str).tolist() if c in feature_names]
        z_all, z_fit, _ = standardize_fit_transform(model_df, fit_index, cols)
        pc = PCA(n_components=1, random_state=0)
        pc.fit(z_fit)
        safe_domain = (
            str(domain)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        name = f"domain_pc1__{safe_domain}"
        pc_cols[name] = pc.transform(z_all)[:, 0]
        variance_rows.append(
            {
                "domain": domain,
                "feature_name": name,
                "n_features": len(cols),
                "explained_variance_ratio": float(pc.explained_variance_ratio_[0]),
            }
        )
    base = model_df.drop(columns=[c for c in feature_names if c in model_df.columns]).copy()
    pc_df = pd.DataFrame(pc_cols, index=model_df.index)
    return pd.concat([base, pc_df], axis=1), pd.DataFrame(variance_rows)


def build_domain_pc_model_from_reference(
    model_df: pd.DataFrame,
    reference_exposome_df: pd.DataFrame,
    domains: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit domain PC1 transforms on the greedy-reference matrix, apply to model_df."""
    pc_cols: dict[str, np.ndarray] = {}
    ref_pc_cols: dict[str, np.ndarray] = {}
    variance_rows: list[dict] = []
    for domain, sub in domains.groupby("domain", sort=True):
        cols = [c for c in sub["feature_name"].astype(str).tolist() if c in feature_names]
        scaler = StandardScaler()
        ref_z = scaler.fit_transform(reference_exposome_df[cols].to_numpy(dtype=float))
        all_z = scaler.transform(model_df[cols].to_numpy(dtype=float))
        pc = PCA(n_components=1, random_state=0)
        ref_score = pc.fit_transform(ref_z)[:, 0]
        all_score = pc.transform(all_z)[:, 0]
        safe_domain = (
            str(domain)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        name = f"domain_pc1__{safe_domain}"
        pc_cols[name] = all_score
        ref_pc_cols[name] = ref_score
        variance_rows.append(
            {
                "domain": domain,
                "feature_name": name,
                "n_features": len(cols),
                "explained_variance_ratio": float(pc.explained_variance_ratio_[0]),
                "pc_fit_dataset": "original_greedy_reference_matrix",
            }
        )
    base = model_df.drop(columns=[c for c in feature_names if c in model_df.columns]).copy()
    pc_df = pd.DataFrame(pc_cols, index=model_df.index)
    ref_pc_df = pd.DataFrame(ref_pc_cols)
    return pd.concat([base, pc_df], axis=1), ref_pc_df, pd.DataFrame(variance_rows)


def build_whole_pca_model_df(
    model_df: pd.DataFrame,
    fit_index: pd.Index,
    feature_names: list[str],
    max_pcs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Effective-sample correction (see load_greedy_reference_exposome): the PCA
    # loadings + explained variance describe exposome structure, so when
    # DEDUP_EXPOSOME_CSV is set they are estimated on the pre-built deduplicated
    # matrix (the effective-sample file the dedup greedy consumed), not the
    # pseudo-replicated subject rows. Scores are still projected for every subject
    # row (model_df), so the downstream LOCO evaluation stays subject-level.
    _dedup_csv = os.environ.get("DEDUP_EXPOSOME_CSV", "").strip()
    if env_bool("MAIN_K10_MODE") and not _dedup_csv:
        raise EnvironmentError(
            "MAIN_K10_MODE requires DEDUP_EXPOSOME_CSV for whole-exposome PCA"
        )
    if _dedup_csv:
        dd = pd.read_csv(_dedup_csv, low_memory=False)
        fit_values = dd[feature_names].to_numpy(dtype=float)
        mean = np.nanmean(fit_values, axis=0)
        std = np.nanstd(fit_values, axis=0, ddof=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        z_fit = (fit_values - mean) / std
        z_all = (model_df[feature_names].to_numpy(dtype=float) - mean) / std
    else:
        z_all, z_fit, _ = standardize_fit_transform(model_df, fit_index, feature_names)
    pca = PCA(n_components=int(max_pcs), random_state=0)
    pca.fit(z_fit)
    scores = pca.transform(z_all)
    pc_names = [f"whole_pc{i:02d}" for i in range(1, int(max_pcs) + 1)]
    pc_df = pd.DataFrame(scores, columns=pc_names, index=model_df.index)
    base = model_df.drop(columns=[c for c in feature_names if c in model_df.columns]).copy()
    var = pd.DataFrame(
        {
            "pc_n": np.arange(1, int(max_pcs) + 1),
            "feature_name": pc_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return pd.concat([base, pc_df], axis=1), var, pc_names


def _pca_scores_train_fit(x_cols: np.ndarray, train_idx: np.ndarray, n_comp: int) -> np.ndarray:
    """Standardize + PCA with parameters fit on training rows only, then project all rows.

    This is the leakage-free counterpart to fitting PCA on the full cohort: the
    scaler mean/std and the PCA loadings are estimated from `train_idx` (the LOCO
    training countries) and applied to every row, so a held-out test country never
    contributes to the basis it is later predicted with.
    """
    x_tr = x_cols[train_idx]
    return _pca_scores_fit_on(x_tr, x_cols, n_comp)


def _pca_scores_fit_on(fit_rows: np.ndarray, x_cols: np.ndarray, n_comp: int) -> np.ndarray:
    """Standardize + PCA fit on `fit_rows`, then project every row of `x_cols`.

    `fit_rows` carries the basis-defining sample (subject-level training rows, or
    -- under DEDUP_EXPOSOME_CSV -- the deduplicated effective-sample rows for the
    fold's training countries). Projection is always applied to all subject rows so
    the LOCO scoring stays subject-level.
    """
    mean = np.nanmean(fit_rows, axis=0)
    std = np.nanstd(fit_rows, axis=0, ddof=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    z_fit = (fit_rows - mean) / std
    z_all = (x_cols - mean) / std
    k = min(int(n_comp), z_fit.shape[1])
    pca = PCA(n_components=k, random_state=0)
    pca.fit(z_fit)
    return pca.transform(z_all)


def build_fold_pc_matrices(
    x_raw: np.ndarray,
    countries: list[str],
    train_idx_by_country: dict,
    spec: dict,
) -> dict:
    """Per-fold PCA score matrices (fit on each fold's training rows only).

    Returns {country: (n_rows, n_pc) float32}. For mode="whole" the columns are the
    first `max_pcs` whole-exposome PCs; for mode="domain" each column is a domain's
    PC1, ordered to match the precomputed PC column order in the model table.
    """
    out: dict = {}
    mode = spec["mode"]
    # Effective-sample correction (DEDUP_EXPOSOME_CSV): when present, the per-fold
    # PCA basis is fit on the deduplicated effective-sample rows of the fold's
    # training countries (read from disk), not the pseudo-replicated subject rows.
    # `dedup_x` is the dedup feature matrix aligned to x_raw's columns, `dedup_split`
    # its per-row split label, `row_split` the subject rows' split label. Held-out
    # country is excluded from the fit (leakage-free); scoring stays subject-level.
    dedup_x = spec.get("dedup_x")
    dedup_split = spec.get("dedup_split")
    row_split = spec.get("row_split")

    def _dedup_fit_rows(c: str, tr_idx) -> np.ndarray | None:
        if dedup_x is None:
            return None
        train_vals = set(np.asarray(row_split)[tr_idx].tolist())
        mask = np.isin(dedup_split, list(train_vals))
        return dedup_x[mask]

    if mode == "whole":
        max_pcs = int(spec["max_pcs"])
        for c in countries:
            tr = train_idx_by_country[c]
            fit_rows = _dedup_fit_rows(c, tr)
            if fit_rows is not None:
                out[c] = _pca_scores_fit_on(fit_rows, x_raw, max_pcs).astype(np.float32)
            else:
                out[c] = _pca_scores_train_fit(x_raw, tr, max_pcs).astype(np.float32)
    elif mode == "domain":
        groups = spec["domain_group_idx"]
        for c in countries:
            tr = train_idx_by_country[c]
            fit_rows = _dedup_fit_rows(c, tr)
            if fit_rows is not None:
                cols = [_pca_scores_fit_on(fit_rows[:, g], x_raw[:, g], 1)[:, 0] for g in groups]
            else:
                cols = [_pca_scores_train_fit(x_raw[:, g], tr, 1)[:, 0] for g in groups]
            out[c] = np.column_stack(cols).astype(np.float32)
    else:
        raise ValueError(f"Unknown fold_pca mode: {mode!r}")
    return out


def _residualize_train_fit(
    y: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    diag: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    """OLS-residualize y ~ 1 + age + sex + diag, fit on training rows only.

    Country is deliberately NOT in this formula: country is the LOCO grouping
    variable for the main analysis and the baseline candidate, not a confound
    to strip out of BAG -- residualizing it out of the target would itself
    change what the baseline (and every other candidate) is being scored
    against, and conflates "control for country" with the LOCO fold structure
    that already handles country. This isolates whatever the exposome-BAG
    association is "on top of" age + sex + diagnosis alone, leaving country's
    role exactly as in the original/main analysis.

    Leakage-free counterpart to fitting the residualization on the full
    cohort: the sex/diagnosis dummy vocabulary and the OLS coefficients are
    estimated from `train_idx` (the LOCO training countries) only, then
    applied to every row -- so a held-out test country never contributes to
    the model that residualizes it. Mirrors `_pca_scores_train_fit`.
    """
    sex_tr, sex_all, _ = encode_dummies(sex[train_idx], sex)
    diag_tr, diag_all, _ = encode_dummies(diag[train_idx], diag)
    X_all = np.hstack([np.ones((len(y), 1)), age.reshape(-1, 1), sex_all, diag_all])
    X_tr = np.hstack([np.ones((len(train_idx), 1)), age[train_idx].reshape(-1, 1), sex_tr, diag_tr])
    keep_tr = np.isfinite(y[train_idx]) & np.isfinite(age[train_idx])
    beta, *_ = np.linalg.lstsq(X_tr[keep_tr], y[train_idx][keep_tr], rcond=None)
    residual = np.full(len(y), np.nan)
    finite_all = np.isfinite(y) & np.isfinite(age)
    residual[finite_all] = y[finite_all] - X_all[finite_all] @ beta
    return residual


def build_fold_residualized_y(
    y: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    diag: np.ndarray,
    countries: list[str],
    train_idx_by_country: dict,
) -> dict:
    """Per-fold residualized target (fit on each fold's training rows only).

    Returns {country: residualized_y_array}, mirroring build_fold_pc_matrices.
    """
    return {
        c: _residualize_train_fit(y, age, sex, diag, train_idx_by_country[c])
        for c in countries
    }


def combo_candidate_table(
    representative_names: list[str],
    *,
    family: str,
    source_label: str,
    order_min: int,
    order_max: int,
    prefix: str,
    extra: dict | None = None,
) -> pd.DataFrame:
    rows = []
    rank_by_order = {order: 0 for order in range(int(order_min), int(order_max) + 1)}
    for order in range(int(order_min), int(order_max) + 1):
        for combo in itertools.combinations(representative_names, order):
            rank_by_order[order] += 1
            ident = "|".join(combo)
            row = {
                "candidate_id": f"{prefix}_ord{order:02d}_rk{rank_by_order[order]:06d}",
                "feature_id": f"{prefix}_ord{order:02d}_rk{rank_by_order[order]:06d}",
                "objective": "o_domain",
                "order": order,
                "rank": rank_by_order[order],
                "score": np.nan,
                "nplet_vars": list(combo),
                "predictors_identity": ident,
                "predictors_identity_n": order,
                "candidate_family": family,
                "source_label": source_label,
            }
            if extra:
                row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def incremental_pc_candidate_table(pc_names: list[str]) -> pd.DataFrame:
    rows = []
    for k in range(1, len(pc_names) + 1):
        names = pc_names[:k]
        rows.append(
            {
                "candidate_id": f"whole_pca_pc{k:02d}",
                "feature_id": f"whole_pca_pc{k:02d}",
                "objective": "whole_pca",
                "order": k,
                "rank": k,
                "score": np.nan,
                "nplet_vars": list(names),
                "predictors_identity": "|".join(names),
                "predictors_identity_n": k,
                "candidate_family": "whole_exposome_pca",
                "source_label": "whole_exposome_pca",
                "pc_n": k,
            }
        )
    return pd.DataFrame(rows)


def score_oinfo_for_candidates_on_matrix(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    *,
    batch_size: int,
) -> pd.DataFrame:
    from thoi.measures.gaussian_copula import nplets_measures

    out = candidate_df.copy()
    col_to_idx = {c: i for i, c in enumerate(exposome_cols)}
    X = reference_df[exposome_cols].to_numpy(dtype=float)
    scores = pd.Series(np.nan, index=out.index, dtype=float)

    for order, sub in out.groupby("order", sort=True):
        log_msg(f"THOI scoring order={order} n_candidates={len(sub)} n_rows={X.shape[0]} n_features={X.shape[1]}")
        nplets = []
        idxs = []
        for idx, row in sub.iterrows():
            names = parse_literal_list(row["nplet_vars"])
            if len(names) != int(order):
                continue
            try:
                nplets.append([col_to_idx[n] for n in names])
            except KeyError:
                continue
            idxs.append(idx)
        if not nplets:
            continue
        arr = np.asarray(nplets, dtype=int)
        measured = nplets_measures(X, nplets=arr, batch_size=int(batch_size), verbose=0)
        vals = measured.detach().cpu().numpy()
        if vals.ndim == 3:
            vals = vals[:, 0, :]
        scores.loc[idxs] = vals[:, 2]
        log_msg(f"THOI scoring order={order} complete")

    out["score"] = scores.to_numpy(dtype=float)
    out["thoi_o"] = out["score"]
    out["rank_o_min"] = out.groupby("order")["score"].rank(method="first", ascending=True)
    out["rank_o_max"] = out.groupby("order")["score"].rank(method="first", ascending=False)
    return out


def score_oinfo_for_candidates(
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    *,
    batch_size: int,
) -> pd.DataFrame:
    return score_oinfo_for_candidates_on_matrix(
        model_df[exposome_cols],
        candidate_df,
        exposome_cols,
        batch_size=batch_size,
    )


def score_oinfo_with_cache(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    *,
    cache_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = candidate_df.copy()
    out["predictors_identity"] = out["nplet_vars"].apply(lambda v: "|".join(str(x) for x in parse_literal_list(v)))
    reference = reference_df[exposome_cols]
    reference_hash = hashlib.sha256()
    reference_hash.update("\0".join(exposome_cols).encode("utf-8"))
    reference_hash.update(pd.util.hash_pandas_object(reference, index=False).values.tobytes())
    reference_sha256 = reference_hash.hexdigest()
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(
            columns=["predictors_identity", "order", "score", "thoi_o", "reference_sha256"]
        )
    if "reference_sha256" not in cache.columns:
        cache = cache.iloc[0:0].copy()
        cache["reference_sha256"] = pd.Series(dtype=str)
    else:
        cache = cache[cache["reference_sha256"].astype(str).eq(reference_sha256)].copy()
    cached_ids = set(cache["predictors_identity"].astype(str).tolist()) if not cache.empty else set()
    missing = out[~out["predictors_identity"].astype(str).isin(cached_ids)].copy()
    log_msg(
        f"THOI cache {cache_path.name}: total={len(out)} cached={len(out) - len(missing)} missing={len(missing)}"
    )
    if not missing.empty:
        scored_missing = score_oinfo_for_candidates_on_matrix(
            reference_df,
            missing,
            exposome_cols,
            batch_size=batch_size,
        )
        new_cache = scored_missing[["predictors_identity", "order", "score", "thoi_o"]].drop_duplicates(
            "predictors_identity"
        )
        new_cache["reference_sha256"] = reference_sha256
        if cache.empty:
            cache = new_cache.copy()
        else:
            cache = pd.concat([cache, new_cache], ignore_index=True).drop_duplicates(
                "predictors_identity", keep="first"
            )
        cache.to_csv(cache_path, index=False)
        log_msg(f"THOI cache {cache_path.name}: wrote {len(cache)} cached scores")
    score_map = cache.set_index("predictors_identity")["score"]
    out["score"] = out["predictors_identity"].astype(str).map(score_map)
    out["thoi_o"] = out["score"]
    out["rank_o_min"] = out.groupby("order")["score"].rank(method="first", ascending=True)
    out["rank_o_max"] = out.groupby("order")["score"].rank(method="first", ascending=False)
    out["oinfo_reference_sha256"] = reference_sha256
    return out


def _predictor_indices(row: dict, exposome_cols: list[str]) -> list[int]:
    col_to_idx = {c: i for i, c in enumerate(exposome_cols)}
    names = parse_literal_list(row.get("nplet_vars", []))
    return [col_to_idx[n] for n in names if n in col_to_idx]


def _build_canonical_ols_design(context, country, train_idx, test_idx, x_exp, pred_used):
    """Rebuild the main pipeline's OLS design matrices for one fold (engine L835-883)."""
    cfg = context["analysis_cfg"]
    age = np.asarray(context["age"], dtype=float)
    sex = np.asarray(context["sex"])
    diag = np.asarray(context["diag"]).astype(str)
    edu = np.asarray(context["edu"], dtype=float)
    scanner = np.asarray(context["scanner"]).astype(str)
    parts_tr = [np.ones((len(train_idx), 1)), age[train_idx].reshape(-1, 1)]
    parts_te = [np.ones((len(test_idx), 1)), age[test_idx].reshape(-1, 1)]
    if cfg.get("include_sex", True):
        s_tr, s_te, lv = encode_dummies(sex[train_idx], sex[test_idx])
        if len(lv) > 1:
            parts_tr.append(s_tr)
            parts_te.append(s_te)
    if cfg.get("include_year", True):
        yb = context["year_basis_by_country"][country]
        parts_tr.append(yb[train_idx])
        parts_te.append(yb[test_idx])
    diag_used = False
    Dtr = Dte = None
    if cfg.get("include_diagnosis", True):
        Dtr, Dte, dlv = encode_dummies(diag[train_idx], diag[test_idx])
        if len(dlv) > 1 and Dtr.shape[1] > 0:
            diag_used = True
            parts_tr.append(Dtr)
            parts_te.append(Dte)
    if cfg.get("include_education", False):
        parts_tr.append(edu[train_idx].reshape(-1, 1))
        parts_te.append(edu[test_idx].reshape(-1, 1))
    if cfg.get("include_scanner", False):
        sc_tr, sc_te, sclv = encode_dummies(scanner[train_idx], scanner[test_idx])
        if len(sclv) > 1:
            parts_tr.append(sc_tr)
            parts_te.append(sc_te)
    Etr = x_exp[np.ix_(train_idx, pred_used)].astype(float) if pred_used else np.zeros((len(train_idx), 0))
    Ete = x_exp[np.ix_(test_idx, pred_used)].astype(float) if pred_used else np.zeros((len(test_idx), 0))
    parts_tr.append(Etr)
    parts_te.append(Ete)
    if diag_used and Dtr.shape[1] > 0 and Etr.shape[1] > 0:
        parts_tr.append((Dtr[:, :, None] * Etr[:, None, :]).reshape(len(train_idx), -1))
        parts_te.append((Dte[:, :, None] * Ete[:, None, :]).reshape(len(test_idx), -1))
    return np.hstack(parts_tr).astype(float), np.hstack(parts_te).astype(float)


def _fit_candidate(
    row: dict,
    *,
    context: dict,
    fold_designs: dict,
    exposome_cols: list[str],
    rung_id: str,
    xgb_cfg: dict | None,
    fold_xgb_cfgs: dict | None = None,
    fold_xgb_ensembles: dict[str, list[dict]] | None = None,
    fold_pc: dict | None = None,
    fold_y: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    model_id = str(row["candidate_id"])
    pred_all = _predictor_indices(row, exposome_cols)
    y = np.asarray(context["y"], dtype=float)
    X_exp = np.asarray(context["X_exp"], dtype=np.float32)
    base_seed = 20260304 if xgb_cfg is None else int(xgb_cfg.get("random_state", 20260304))
    y_true_all = []
    y_pred_all = []
    country_rows = []
    xgb_mod = None if rung_id == "ols" else require_xgboost()

    for fold_i, country in enumerate(context["countries"]):
        fd = fold_designs[country]
        train_idx = fd["train_idx"]
        test_idx = fd["test_idx"]
        # Leakage-free predictors: when fold_pc is supplied the exposome columns are
        # PCA scores refit on this fold's training rows only (test country excluded).
        X_exp_use = X_exp if fold_pc is None else np.asarray(fold_pc[country], dtype=np.float32)
        # Leakage-free target: when fold_y is supplied the regression target is
        # the residualized BAG refit on this fold's training rows only (test
        # country excluded), mirroring the fold_pc swap above.
        y_use = y if fold_y is None else np.asarray(fold_y[country], dtype=float)
        pred_used = list(pred_all)
        if pred_used:
            var = np.nanvar(X_exp_use[np.ix_(train_idx, pred_used)], axis=0)
            pred_used = [idx for idx, keep in zip(pred_used, var > 0) if bool(keep)]

        y_pred = np.full(len(test_idx), np.nan, dtype=float)
        try:
            if rung_id == "ols" and _ols_dx_interactions():
                # Faithful reproduction of the main pipeline's OLS design
                # (loco_fusion_matrix_engine.py L835-883): intercept + age +
                # encode_dummies(sex) + year spline basis + encode_dummies(diag) +
                # exposome + diagnosis x exposome interactions. The evaluator's default
                # OLS (Xb-based, linear year, no interactions) underrepresents this.
                X_train, X_test = _build_canonical_ols_design(
                    context, country, train_idx, test_idx, X_exp_use, pred_used
                )
                beta, *_ = np.linalg.lstsq(X_train, y_use[train_idx], rcond=None)
                y_pred = X_test @ beta
            elif rung_id == "ols":
                X_train = _append_predictors(fd["Xb_train_full"], X_exp_use, train_idx, pred_used).astype(float)
                X_test = _append_predictors(fd["Xb_test"], X_exp_use, test_idx, pred_used).astype(float)
                X_train = np.column_stack([np.ones(len(train_idx)), X_train])
                X_test = np.column_stack([np.ones(len(test_idx)), X_test])
                beta, *_ = np.linalg.lstsq(X_train, y_use[train_idx], rcond=None)
                y_pred = X_test @ beta
            else:
                tr_inner = fd["tr_inner_idx"]
                val_idx = fd["val_idx"]
                X_tr_inner = _append_predictors(fd["Xb_train_inner"], X_exp_use, tr_inner, pred_used)
                X_val = _append_predictors(fd["Xb_val"], X_exp_use, val_idx, pred_used)
                X_test = _append_predictors(fd["Xb_test"], X_exp_use, test_idx, pred_used)
                ensemble = (fold_xgb_ensembles or {}).get(country)
                members = ensemble if ensemble is not None else [dict((fold_xgb_cfgs or {}).get(country, xgb_cfg or {}))]
                if not members:
                    raise ValueError(f"XGBoost ensemble has no members for outer country {country}")
                member_predictions = []
                for member in members:
                    params = dict(member)
                    params["random_state"] = int(params.get("random_state", base_seed) + fold_i)
                    reg = _fit_xgb_fold(
                        xgb_mod, params, X_tr_inner, y_use[tr_inner], X_val, y_use[val_idx]
                    )
                    member_predictions.append(reg.predict(X_test))
                y_pred = np.mean(np.vstack(member_predictions), axis=0)
        except Exception:
            y_pred = np.full(len(test_idx), np.nan, dtype=float)

        y_true = y_use[test_idx]
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        met = regression_metrics_extended(
            y_true[ok],
            y_pred[ok],
            min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
        )
        country_rows.append(
            {
                "candidate_id": model_id,
                "rung_id": rung_id,
                "fold_country": country,
                "n_test": int(ok.sum()),
                "r2": met["r2"],
                "rmse": met["rmse"],
                "mae": met["mae"],
                "corr2": met["corr2"],
            }
        )

    yt = np.concatenate(y_true_all) if y_true_all else np.asarray([])
    yp = np.concatenate(y_pred_all) if y_pred_all else np.asarray([])
    ok = np.isfinite(yt) & np.isfinite(yp)
    g = regression_metrics_extended(
        yt[ok],
        yp[ok],
        min_n=int(context["analysis_cfg"].get("min_n_obs_for_metrics", 5)),
    )
    summary = {
        "candidate_id": model_id,
        "rung_id": rung_id,
        "global_oof_r2": g["r2"],
        "global_rmse": g["rmse"],
        "global_mae": g["mae"],
        "global_corr2": g["corr2"],
        "n_scored": int(g["n_scored"]),
        "predictors_used_n": int(len(pred_all)),
    }
    return summary, pd.DataFrame(country_rows)


def evaluate_candidates_by_rung(
    *,
    model_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    exposome_cols: list[str],
    bag: str,
    rungs: list[str],
    analysis_cfg: dict,
    outdir: Path,
    n_jobs: int,
    max_candidates: int | None = None,
    fold_pca: dict | None = None,
    fold_residualize: bool = False,
    cv_override: dict | None = None,
    tuning_artifact_path: str | Path | None = None,
    tuning_strict: bool = False,
    fold_xgb_cfgs: dict[str, dict] | None = None,
    fold_xgb_ensembles: dict[str, list[dict]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pin BLAS/OpenMP threads to 1 in this process before fanning out joblib
    # workers below -- every other stage (run_null_model, run_single_exposure_eval,
    # stages_xgb/stages_ols) already does this; without it, each of the n_jobs
    # worker processes can spawn its own BLAS threads internally, oversubscribing
    # the CPUs n_jobs-fold and silently degrading wall-clock time under load.
    pin_blas_threads(1)
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_df = candidate_df.copy().reset_index(drop=True)
    if max_candidates is not None and int(max_candidates) > 0:
        candidate_df = candidate_df.head(int(max_candidates)).copy()
    if "predictors_identity" not in candidate_df.columns:
        candidate_df["predictors_identity"] = candidate_df["nplet_vars"].apply(lambda x: "|".join(parse_literal_list(x)))
    duplicate_model_count = int(candidate_df.duplicated("predictors_identity").sum())

    # cv_override lets an analysis change the fold column (e.g. split_col=
    # "subregion_name" for leave-one-region-out). Country/diagnosis exclusions are
    # expected to be pre-applied to model_df by the caller in that case.
    cv = {**cv_cfg(), **(cv_override or {})}
    y_col = target_map(bag)[bag]
    context = prepare_bag_context(model_df, y_col, bag, analysis_cfg, cv, exposome_cols)
    base_seed = 20260304
    fold_designs = {
        c: _build_base_fold_mats(context, c, analysis_cfg, early_stop_cfg(), seed=base_seed + i)
        for i, c in enumerate(context["countries"])
    }

    # Leakage-free per-fold PCA: refit the standardization+PCA on each fold's
    # training countries only and project all rows, so the held-out country never
    # contributes to the basis. Re-derive the analysis table (deterministic, same
    # row order as prepare_bag_context) to align the raw feature matrix to folds.
    fold_pc = None
    fold_pca_sha256 = ""
    if fold_pca is not None:
        spec = dict(fold_pca)
        raw_cols = list(spec["raw_cols"])
        work = prepare_analysis_table(model_df, y_col, analysis_cfg, cv).reset_index(drop=True)
        x_raw = work[raw_cols].to_numpy(dtype=float)
        # Effective-sample correction: fit the per-fold PCA basis on the pre-built
        # deduplicated matrix (DEDUP_EXPOSOME_CSV), subset per fold to the training
        # countries via the split column, instead of the pseudo-replicated subject
        # rows. Read from disk; no on-the-fly re-deduplication. Scoring stays
        # subject-level (projection onto x_raw / subject rows).
        _dedup_csv = os.environ.get("DEDUP_EXPOSOME_CSV", "").strip()
        if env_bool("MAIN_K10_MODE") and not _dedup_csv:
            raise EnvironmentError(
                "MAIN_K10_MODE requires DEDUP_EXPOSOME_CSV for fold-wise PCA"
            )
        if _dedup_csv:
            dedup_path = Path(_dedup_csv).resolve()
            fold_pca_sha256 = hashlib.sha256(dedup_path.read_bytes()).hexdigest()
            split_col = str(cv["split_col"])
            _dd = pd.read_csv(dedup_path, low_memory=False)
            missing = [c for c in raw_cols + [split_col] if c not in _dd.columns]
            if missing:
                raise ValueError(f"DEDUP_EXPOSOME_CSV missing columns for fold PCA: {missing[:8]}")
            spec["dedup_x"] = _dd[raw_cols].to_numpy(dtype=float)
            spec["dedup_split"] = _dd[split_col].astype(str).to_numpy()
            spec["row_split"] = work[split_col].astype(str).to_numpy()
        if spec["mode"] == "domain":
            col_to_idx = {c: i for i, c in enumerate(raw_cols)}
            spec["domain_group_idx"] = [
                np.asarray([col_to_idx[n] for n in group], dtype=int) for group in spec["domain_groups"]
            ]
        fold_pc = build_fold_pc_matrices(
            x_raw, context["countries"], context["train_idx_by_country"], spec
        )

    # Leakage-free per-fold residualization: refit BAG ~ age + sex + diag on
    # each fold's training countries only and apply it to every row, so the
    # held-out country's target is never informed by its own data. Country is
    # not in the formula -- it stays the LOCO grouping variable, unchanged
    # from the original/main analysis.
    fold_y = None
    if fold_residualize:
        fold_y = build_fold_residualized_y(
            context["y"],
            context["age"],
            context["sex"],
            context["diag"],
            context["countries"],
            context["train_idx_by_country"],
        )

    summary_parts = []
    country_parts = []
    rung_specs = {spec["rung_id"]: spec for spec in get_rung_specs()}
    chunk_size = int(os.environ.get("SENSITIVITY_EVAL_CHUNK_SIZE", "128"))

    for rung_id in rungs:
        xgb_cfg = None if rung_id == "ols" else build_xgb_cfg_for_rung(rung_specs[rung_id])
        # Preserve the existing environment override for legacy callers, while
        # allowing a paired comparison to explicitly disable it for its fixed arm.
        artifact_path = (
            os.environ.get("XGB_TUNING_ARTIFACT", "").strip()
            if tuning_artifact_path is None
            else str(tuning_artifact_path)
        )
        main_k10_routing = xgb_cfg is not None and env_bool("MAIN_K10_MODE")
        if main_k10_routing and fold_xgb_cfgs is not None:
            raise ValueError("Direct fold_xgb_cfgs cannot bypass main-k10 HPO scope routing")
        if main_k10_routing:
            rung_candidate_df, hpo_artifacts = route_main_k10_hpo(candidate_df, artifact_path)
            fold_cfgs_by_scope = {
                scope: resolve_tuned_fold_configs(
                    path,
                    bag,
                    rung_id,
                    xgb_cfg,
                    context["countries"],
                    strict=True,
                )
                for scope, path in hpo_artifacts.items()
            }
            artifact_fold_xgb_cfgs = None
        else:
            rung_candidate_df = candidate_df.copy()
            rung_candidate_df["hpo_scope"] = "ols" if xgb_cfg is None else "configured"
            rung_candidate_df["hpo_artifact_sha256"] = ""
            fold_cfgs_by_scope = {}
            artifact_fold_xgb_cfgs = None if xgb_cfg is None else resolve_tuned_fold_configs(
                artifact_path, bag, rung_id, xgb_cfg, context["countries"],
                strict=tuning_strict if tuning_artifact_path is not None else env_bool("XGB_TUNING_STRICT", default=False),
            )
        if fold_pca is not None:
            rung_candidate_df["fold_pca_sha256"] = fold_pca_sha256
        if fold_xgb_cfgs is not None and artifact_fold_xgb_cfgs is not None:
            raise ValueError("Provide either direct fold_xgb_cfgs or a tuning artifact, not both")
        resolved_fold_xgb_cfgs = fold_xgb_cfgs or artifact_fold_xgb_cfgs
        expected_ids = set(rung_candidate_df["candidate_id"].astype(str).tolist())
        identity_columns = ["hpo_scope", "hpo_artifact_sha256"]
        if fold_pca is not None:
            identity_columns.append("fold_pca_sha256")
        expected_hpo = (
            rung_candidate_df.set_index("candidate_id")[identity_columns]
            .astype(str)
            .sort_index()
        )
        global_path = outdir / f"{bag}_{rung_id}_global.csv"
        country_path = outdir / f"{bag}_{rung_id}_country.csv"
        if global_path.exists() and country_path.exists():
            try:
                existing_global = pd.read_csv(global_path)
                existing_ids = set(existing_global.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist())
                has_hpo_identity = set(identity_columns).issubset(existing_global.columns)
                existing_hpo = (
                    existing_global.set_index("candidate_id")[identity_columns]
                    .astype(str)
                    .sort_index()
                    if has_hpo_identity
                    else pd.DataFrame()
                )
                complete = (
                    expected_ids == existing_ids
                    and len(existing_global) == len(expected_ids)
                    and has_hpo_identity
                    and existing_hpo.equals(expected_hpo)
                )
                if complete:
                    existing_country = pd.read_csv(country_path)
                    current_metadata = rung_candidate_df.set_index("candidate_id")
                    for column in current_metadata.columns:
                        if column == "candidate_id":
                            continue
                        values = current_metadata[column]
                        existing_global[column] = existing_global["candidate_id"].map(values)
                        if "candidate_id" in existing_country.columns:
                            existing_country[column] = existing_country["candidate_id"].map(values)
                    log_msg(
                        f"LOCO eval resume hit bag={bag} rung={rung_id} "
                        f"candidates={len(existing_global)}; loading existing outputs"
                    )
                    summary_parts.append(existing_global)
                    country_parts.append(existing_country)
                    continue
                log_msg(
                    f"LOCO eval resume miss bag={bag} rung={rung_id}: "
                    f"expected={len(expected_ids)} existing={len(existing_ids)}; recomputing"
                )
            except Exception as exc:
                log_msg(f"LOCO eval resume read failed bag={bag} rung={rung_id}: {exc!r}; recomputing")
        fit_candidate_df = rung_candidate_df.drop_duplicates("predictors_identity", keep="first").reset_index(drop=True)
        records = fit_candidate_df.to_dict(orient="records")
        fit_map = fit_candidate_df[["candidate_id", "predictors_identity"]].rename(
            columns={"candidate_id": "fit_candidate_id"}
        )
        fit_manifest_path = outdir / f"{bag}_{rung_id}_fit_manifest.csv"
        fit_manifest = (
            rung_candidate_df.groupby("predictors_identity", as_index=False, dropna=False)
            .agg(
                candidate_count=("candidate_id", "size"),
                candidate_ids=("candidate_id", lambda x: "|".join(map(str, x))),
                candidate_families=("candidate_family", lambda x: "|".join(sorted(set(map(str, x))))),
                source_labels=("source_label", lambda x: "|".join(sorted(set(map(str, x))))),
                hpo_scopes=("hpo_scope", lambda x: "|".join(sorted(set(map(str, x))))),
                hpo_artifact_sha256=("hpo_artifact_sha256", lambda x: "|".join(sorted(set(map(str, x))))),
                order=("order", "first"),
            )
            .merge(fit_map, on="predictors_identity", how="left")
        )
        fit_manifest.to_csv(fit_manifest_path, index=False)
        log_msg(
            f"LOCO eval start bag={bag} rung={rung_id} candidates={len(candidate_df)} "
            f"unique_models={len(records)} duplicates_skipped={duplicate_model_count} "
            f"n_jobs={int(n_jobs)} chunk_size={chunk_size}"
        )
        fitted = []
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            if int(n_jobs) <= 1:
                fitted_chunk = [
                    _fit_candidate(
                        row,
                        context=context,
                        fold_designs=fold_designs,
                        exposome_cols=exposome_cols,
                        rung_id=rung_id,
                        xgb_cfg=xgb_cfg,
                        fold_xgb_cfgs=(
                            fold_cfgs_by_scope[str(row["hpo_scope"])]
                            if main_k10_routing
                            else resolved_fold_xgb_cfgs
                        ),
                        fold_xgb_ensembles=fold_xgb_ensembles,
                        fold_pc=fold_pc,
                        fold_y=fold_y,
                    )
                    for row in chunk
                ]
            else:
                fitted_chunk = Parallel(n_jobs=int(n_jobs), backend="loky")(
                    delayed(_fit_candidate)(
                        row,
                        context=context,
                        fold_designs=fold_designs,
                        exposome_cols=exposome_cols,
                        rung_id=rung_id,
                        xgb_cfg=xgb_cfg,
                        fold_xgb_cfgs=(
                            fold_cfgs_by_scope[str(row["hpo_scope"])]
                            if main_k10_routing
                            else resolved_fold_xgb_cfgs
                        ),
                        fold_xgb_ensembles=fold_xgb_ensembles,
                        fold_pc=fold_pc,
                        fold_y=fold_y,
                    )
                    for row in chunk
                )
            fitted.extend(fitted_chunk)
            log_msg(
                f"LOCO eval progress bag={bag} rung={rung_id} "
                f"unique_done={min(start + len(chunk), len(records))}/{len(records)}"
            )
        fit_summary = pd.DataFrame([x[0] for x in fitted])
        fit_country = pd.concat([x[1] for x in fitted], ignore_index=True) if fitted else pd.DataFrame()
        fit_summary = fit_summary.rename(columns={"candidate_id": "fit_candidate_id"}).merge(
            fit_map,
            on="fit_candidate_id",
            how="left",
        )
        metadata = rung_candidate_df.drop(columns=["nplet_vars"], errors="ignore").copy()
        metric_cols = [
            "predictors_identity",
            "fit_candidate_id",
            "rung_id",
            "global_oof_r2",
            "global_rmse",
            "global_mae",
            "global_corr2",
            "n_scored",
            "predictors_used_n",
        ]
        rung_summary = metadata.merge(fit_summary[metric_cols], on="predictors_identity", how="left")
        if fit_country.empty:
            rung_country = pd.DataFrame()
        else:
            fit_country = fit_country.rename(columns={"candidate_id": "fit_candidate_id"}).merge(
                fit_map,
                on="fit_candidate_id",
                how="left",
            )
            country_metric_cols = [
                "predictors_identity",
                "fit_candidate_id",
                "rung_id",
                "fold_country",
                "n_test",
                "r2",
                "rmse",
                "mae",
                "corr2",
            ]
            country_meta = rung_candidate_df[
                ["candidate_id", "candidate_family", "source_label", "order", "score", "predictors_identity",
                 "hpo_scope", "hpo_artifact_sha256"]
            ].copy()
            rung_country = country_meta.merge(fit_country[country_metric_cols], on="predictors_identity", how="left")
        rung_summary.to_csv(global_path, index=False)
        rung_country.to_csv(country_path, index=False)
        summary_parts.append(rung_summary)
        country_parts.append(rung_country)
        log_msg(f"LOCO eval complete bag={bag} rung={rung_id}")

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    country = pd.concat(country_parts, ignore_index=True) if country_parts else pd.DataFrame()
    summary.to_csv(outdir / f"{bag}_global_all_rungs.csv", index=False)
    country.to_csv(outdir / f"{bag}_country_all_rungs.csv", index=False)
    return summary, country


def load_existing_baselines(bag: str) -> pd.DataFrame:
    p = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not p.exists():
        p = canonical_root() / "metrics_global_long.parquet"
    df = pd.read_parquet(p)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag].copy()
    out = (
        df[df["rung_id"].isin(ACTIVE_RUNGS)]
        .groupby("rung_id", as_index=False, observed=True)["base_r2"]
        .max()
        .rename(columns={"base_r2": "baseline_r2"})
    )
    return out


def load_original_complete_best(bag: str) -> pd.DataFrame:
    """Best (max LOCO full_r2) original Fig.2 model per active rung.

    Matches the best-model extraction in scripts/plot_fig2_grid_v3.py (panel B
    plots full_r2 as "R² LOCO"); here we take the per-rung maximum across all
    original greedy candidates regardless of syn/red sign.
    """
    p = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not p.exists():
        p = canonical_root() / "metrics_global_long.parquet"
    df = pd.read_parquet(p)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag].copy()
    df = df[df["rung_id"].isin(ACTIVE_RUNGS)].copy()
    # Respect the active order cap (cap21 vs complete): the best model per rung
    # must be chosen among candidates of order <= cap so the reference line matches
    # the capped paper figures. No-op when PAPER_FIG_ORDER_MAX is unset.
    df = apply_order_cap(df)
    df["full_r2"] = pd.to_numeric(df["full_r2"], errors="coerce")
    out = (
        df.groupby("rung_id", as_index=False, observed=True)["full_r2"]
        .max()
        .rename(columns={"full_r2": "best_model_r2"})
    )
    return out


def _fig2_candidate_pool_from_greedy(bag: str) -> pd.DataFrame:
    """Rebuild the Fig.2 candidate pool directly from the greedy CSV.

    Last-resort fallback used only when neither the canonical nor the results/
    pooled parquet exists. Reproduces the pooled stage's pool construction exactly:
    greedy top-k -> select_candidate_space -> identity prune by feature-set. The
    cap (order_min=3, order_max=30, top_k_per_order=20) is the dedup paper contract
    and is pinned HERE explicitly so the pool does not silently shrink to the
    ANALYSIS_CFG debug defaults (top_k=10) when a downstream job has not exported
    V3_TOP_K_PER_ORDER/V3_ORDER_MAX -- which is exactly how this produced a 560-row
    (instead of 1120) pool before. The result is identical to the candidates the
    pooled stage evaluates; only the per-model evaluation metrics are absent.
    """
    from loco_fusion_matrix_engine import (
        load_artifacts_greedy_only,
        make_candidate_identity_map,
    )
    from oinfo_bag_ladder.candidate_sources import build_greedy_o_only_candidates
    from oinfo_bag_ladder.config import ANALYSIS_CFG, DATA_PATHS

    # Pin the dedup paper cap explicitly; do NOT inherit ANALYSIS_CFG debug defaults.
    cap_cfg = dict(ANALYSIS_CFG)
    cap_cfg["order_min"] = int(os.environ.get("V3_ORDER_MIN", "3") or "3")
    cap_cfg["order_max"] = int(os.environ.get("V3_ORDER_MAX", "30") or "30")
    cap_cfg["top_k_per_order"] = int(os.environ.get("V3_TOP_K_PER_ORDER", "20") or "20")

    _raw, exposome_df, candidate_df, _all = load_artifacts_greedy_only(DATA_PATHS, cap_cfg)
    exposome_cols = exposome_df.columns.tolist()

    # Identity prune by feature-set (keep one representative per unique predictor set).
    _cmap, reps_df = make_candidate_identity_map(
        candidate_df, exposome_cols, enable_prune=True
    )
    rep_ids = set(reps_df["model_id"].astype(str))
    pruned = candidate_df[candidate_df["feature_id"].astype(str).isin(rep_ids)].copy()

    registry, _reg2 = build_greedy_o_only_candidates(pruned, exposome_cols)
    pool = registry[
        ["candidate_id", "objective", "order", "score", "predictors_identity"]
    ].copy()
    pool = pool.dropna(subset=["predictors_identity"])
    pool = pool.sort_values("candidate_id").drop_duplicates("candidate_id").reset_index(drop=True)
    pool["feature_id"] = pool["candidate_id"].astype(str)
    pool["nplet_vars"] = pool["predictors_identity"].astype(str).apply(lambda s: s.split("|"))
    pool["predictors_identity_n"] = pool["nplet_vars"].apply(len)
    pool["rank"] = pool.groupby("order").cumcount() + 1
    pool["candidate_family"] = "fig2_greedy"
    pool["source_label"] = pool["objective"].astype(str)
    return pool


def load_fig2_candidate_pool(bag: str) -> pd.DataFrame:
    """Rebuild the original Fig.2 greedy candidate pool (o_min/o_max) for a BAG.

    Returns one row per unique candidate with `nplet_vars` (feature-name list) and
    the candidate-intrinsic O-information `score`, ready for re-evaluation under a
    new CV scheme (LORO) or population (CN-only) via evaluate_candidates_by_rung.
    The pool is read from the canonical metrics_global_long.parquet so it matches
    exactly the models behind the published Fig.2. When that parquet does not exist
    yet (the pooled/evaluate stage has not run), the pool is rebuilt directly from
    the greedy CSV (identical candidate set) so the sensitivity jobs can run in
    parallel with — and independently of — the pooled stage.
    """
    # Prefer the per-bag parquet, then the global, in BOTH the canonical (runs/)
    # and the results/ trees (the pooled stage writes results/; promotion to
    # canonical via symlink is a separate step). Same pool either way.
    candidates: list[Path] = []
    for base in (
        canonical_root(),
        bundle_root() / "results" / "variant_a" / "families" / "pooled_oinfo_ladder" / "canonical",
    ):
        candidates.append(base / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet")
        candidates.append(base / "metrics_global_long.parquet")
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return _fig2_candidate_pool_from_greedy(bag)
    df = pd.read_parquet(p)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag].copy()
    df = df[df["objective"].astype(str).isin(["o_min", "o_max"])].copy()
    df = df.dropna(subset=["predictors_identity"])
    pool = (
        df.sort_values("candidate_id")
        .drop_duplicates("candidate_id")[["candidate_id", "objective", "order", "score", "predictors_identity"]]
        .reset_index(drop=True)
    )
    pool["feature_id"] = pool["candidate_id"].astype(str)
    pool["nplet_vars"] = pool["predictors_identity"].astype(str).apply(lambda s: s.split("|"))
    pool["predictors_identity_n"] = pool["nplet_vars"].apply(len)
    pool["rank"] = pool.groupby("order").cumcount() + 1
    pool["candidate_family"] = "fig2_greedy"
    pool["source_label"] = pool["objective"].astype(str)
    return pool


def load_best_single_by_rung(bag: str) -> pd.DataFrame:
    if env_bool("MAIN_K10_MODE"):
        runtime = bundle_root()
        source_run = os.environ.get("MAIN_K10_SOURCE_RUN_ID", "paper_reanalysis_k10").strip()
        parts = []
        for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
            root = runtime / "results" / "analysis_runs" / source_run / "xgb" / bag / rung / "single"
            global_path = root / "metrics_global.csv"
            country_path = root / "metrics_country.csv"
            if not global_path.is_file() or not country_path.is_file():
                raise FileNotFoundError(f"Missing main k10 single-exposure metrics for {bag}/{rung}")
            global_metrics = pd.read_csv(global_path)
            country_metrics = pd.read_csv(country_path)
            balanced = (
                country_metrics[pd.to_numeric(country_metrics["n_test"], errors="coerce").gt(0)]
                .groupby("candidate_id", as_index=False, observed=True)["r2"].mean()
                .rename(columns={"r2": "global_oof_r2"})
            )
            frame = global_metrics[["candidate_id"]].merge(balanced, on="candidate_id", how="inner", validate="one_to_one")
            frame["feature_name"] = frame["candidate_id"].astype(str).str.removeprefix("__single__")
            domain_map = pd.read_csv(DOMAIN_CSV).set_index("feature_name")["domain"]
            frame["domain"] = frame["feature_name"].map(domain_map)
            if frame["domain"].isna().any():
                missing = frame.loc[frame["domain"].isna(), "feature_name"].astype(str).tolist()
                raise ValueError(f"Missing canonical domain labels for main k10 single exposures: {missing[:8]}")
            frame["source_rung"] = rung
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)
    root = bundle_variant_root()
    dirs = {
        "ols": root / "single_exposure_eval_ols" / bag / "single_exposure_global.csv",
        "xgb_tree_d1": root / "single_exposure_eval_xgb_tree_d1" / bag / "single_exposure_global.csv",
        "xgb_tree_d2": root / "single_exposure_eval_xgb_tree_d2" / bag / "single_exposure_global.csv",
        "xgb_tree_d3": root / "single_exposure_eval_xgb_tree_d3" / bag / "single_exposure_global.csv",
    }
    parts = []
    for rung, path in dirs.items():
        if not path.exists() and rung == "xgb_tree_d2":
            path = root / "single_exposure_eval" / bag / "single_exposure_global.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["feature_name"].astype(str) != "__baseline__"].copy()
        df["source_rung"] = rung
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_main_top20_reference(bag: str) -> pd.DataFrame:
    p = canonical_root() / "per_experiment" / f"pooled_oinfo_ladder_{bag}" / "metrics_global_long.parquet"
    if not p.exists():
        p = canonical_root() / "metrics_global_long.parquet"
    df = pd.read_parquet(p)
    if "bag_target" in df.columns:
        df = df[df["bag_target"].astype(str) == bag].copy()
    rows = []
    for rung in ACTIVE_RUNGS:
        rr = df[df["rung_id"].astype(str) == rung]
        for objective, label, asc in [("o_min", "main_top20_syn", False), ("o_max", "main_top20_red", False)]:
            obj_pool = filter_syn_pool(rr[rr["objective"].astype(str) == objective], objective, "thoi_o")
            sub = obj_pool.nlargest(20, "full_r2")
            if sub.empty:
                continue
            rows.append(
                {
                    "rung_id": rung,
                    "reference": label,
                    "median_r2": float(sub["full_r2"].median()),
                    "q25_r2": float(sub["full_r2"].quantile(0.25)),
                    "q75_r2": float(sub["full_r2"].quantile(0.75)),
                    "max_r2": float(sub["full_r2"].max()),
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)
