from __future__ import annotations

from typing import Iterable

import pandas as pd

from oinfo_bag_ladder.config import BAG_ORDER, BAG_TARGETS, DEFAULT_ANALYSIS_FAMILY, DIAGNOSIS_ORDER


def _normalize_bags(bag_target_mode: str = "all") -> list[str]:
    mode = str(bag_target_mode).strip().lower()
    if mode == "all":
        return list(BAG_ORDER)
    if mode not in BAG_TARGETS:
        raise ValueError(f"Unsupported bag_target_mode: {bag_target_mode!r}")
    return [mode]


def _normalize_diagnoses(diagnosis_mode: str | Iterable[str] = "all") -> list[str]:
    if isinstance(diagnosis_mode, str):
        mode = diagnosis_mode.strip()
        if mode.lower() == "all":
            return list(DIAGNOSIS_ORDER)
        vals = [x.strip() for x in mode.split(",") if x.strip()]
    else:
        vals = [str(x).strip() for x in diagnosis_mode if str(x).strip()]
    if not vals:
        return list(DIAGNOSIS_ORDER)
    bad = [x for x in vals if x not in DIAGNOSIS_ORDER]
    if bad:
        raise ValueError(f"Unsupported diagnosis values: {bad}")
    return vals


def build_experiment_registry(
    bag_target_mode: str = "all",
    analysis_family: str = DEFAULT_ANALYSIS_FAMILY,
    diagnosis_mode: str | Iterable[str] = "all",
) -> pd.DataFrame:
    bags = _normalize_bags(bag_target_mode)
    diagnoses = _normalize_diagnoses(diagnosis_mode)
    family = str(analysis_family).strip()

    rows = []
    if family == "pooled_oinfo_ladder":
        for bag in bags:
            rows.append(
                {
                    "experiment_id": f"pooled_oinfo_ladder_{bag}",
                    "analysis_family": family,
                    "bag_target": bag,
                    "bag_target_column": BAG_TARGETS[bag],
                    "candidate_source": "greedy_o_only",
                    "population_mode": "pooled_4dx",
                    "objective_scope": "o_only",
                    "cv_scheme": "loco_country",
                    "train_diagnosis_group": "all",
                    "test_diagnosis_group": "all",
                    "selection_family": "greedy_o_only",
                    "selection_diagnosis_group": "all",
                    "evaluation_mode": "pooled_refit",
                }
            )
    elif family == "single_dx_oinfo_ladder":
        for bag in bags:
            for diagnosis in diagnoses:
                rows.append(
                    {
                        "experiment_id": f"single_dx_oinfo_ladder_{bag}_{diagnosis}",
                        "analysis_family": family,
                        "bag_target": bag,
                        "bag_target_column": BAG_TARGETS[bag],
                        "candidate_source": "greedy_o_only",
                        "population_mode": "single_dx",
                        "objective_scope": "o_only",
                        "cv_scheme": "loco_country",
                        "train_diagnosis_group": diagnosis,
                        "test_diagnosis_group": diagnosis,
                        "selection_family": "greedy_o_only",
                        "selection_diagnosis_group": diagnosis,
                        "evaluation_mode": "single_dx_refit",
                    }
                )
    elif family == "single_dx_portability":
        for bag in bags:
            for diagnosis in diagnoses:
                rows.append(
                    {
                        "experiment_id": f"single_dx_portability_{bag}_{diagnosis}",
                        "analysis_family": family,
                        "bag_target": bag,
                        "bag_target_column": BAG_TARGETS[bag],
                        "candidate_source": "pooled_top_tail_per_rung",
                        "population_mode": "single_dx",
                        "objective_scope": "o_only",
                        "cv_scheme": "loco_country",
                        "train_diagnosis_group": diagnosis,
                        "test_diagnosis_group": diagnosis,
                        "selection_family": "pooled_oinfo_ladder",
                        "selection_diagnosis_group": "all",
                        "evaluation_mode": "single_dx_refit",
                    }
                )
    elif family == "cn_normative_transfer":
        for bag in bags:
            for diagnosis in diagnoses:
                rows.append(
                    {
                        "experiment_id": f"cn_normative_transfer_{bag}_{diagnosis}",
                        "analysis_family": family,
                        "bag_target": bag,
                        "bag_target_column": BAG_TARGETS[bag],
                        "candidate_source": "cn_top_tail_per_rung",
                        "population_mode": "cn_transfer",
                        "objective_scope": "o_only",
                        "cv_scheme": "loco_country",
                        "train_diagnosis_group": "CN",
                        "test_diagnosis_group": diagnosis,
                        "selection_family": "single_dx_oinfo_ladder",
                        "selection_diagnosis_group": "CN",
                        "evaluation_mode": "cn_transfer",
                    }
                )
    else:
        raise ValueError(f"Unsupported analysis_family: {analysis_family!r}")

    return pd.DataFrame(rows)
