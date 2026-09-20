#!/bin/bash
# Submit one CPU job for nested-LOCO XGBoost tuning: d1/d2/d3, both primary
# BAGs, and 200 sequential Bayesian-optimization trials per BAG/country/rung.
#
# Usage:
#   bash scripts/submit_xgb_nested_loco_tuning.sh
#   bash scripts/submit_xgb_nested_loco_tuning.sh historical_a
#   N_JOBS=42 N_TRIALS=200 bash scripts/submit_xgb_nested_loco_tuning.sh
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [country_variant]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/..}"
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/env.sh"
fi

REPRO_DATA_ROOT="${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to the external analysis runtime}"

SCHEDULER_LOG_DIR="${SCHEDULER_LOG_DIR:-${REPRO_DATA_ROOT}/work/xgb_nested_loco_tuning/scheduler_logs}"
mkdir -p "${SCHEDULER_LOG_DIR}"

N_JOBS="${N_JOBS:-42}"
N_TRIALS="${N_TRIALS:-200}"
SELECTION_OBJECTIVE="${XGB_TUNING_SELECTION_OBJECTIVE:-country_r2_p25}"
FITTING_MODE="${XGB_TUNING_FITTING_MODE:-nested_per_outer_country}"
OPTIMIZER="${XGB_TUNING_OPTIMIZER:-custom_gp}"
PARENT_RUN_SIGNATURE="${XGB_TUNING_PARENT_RUN_SIGNATURE:-}"
# The cluster's ``run`` wrapper drops an empty positional argument while
# constructing its command line.  Keep the later positional arguments stable
# for a fresh run by sending an explicit, runner-recognised sentinel instead.
PARENT_RUN_ARGUMENT="${PARENT_RUN_SIGNATURE:--}"
FEATURE_SCOPE="${XGB_TUNING_FEATURE_SCOPE:-full_exposome}"
INCLUDE_MANUAL_INCUMBENT="${XGB_TUNING_INCLUDE_MANUAL_INCUMBENT:-1}"
TIME_LIMIT="${XGB_TUNING_TIME_LIMIT:-12:00}"
COUNTRY_VARIANT="${1:-a}"
SUBMITTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
case "${SELECTION_OBJECTIVE}" in
    country_r2_p25)
        OBJECTIVE_LABEL=""
        ;;
    country_balanced_r2)
        OBJECTIVE_LABEL="_balr2"
        ;;
    mean_country_mse)
        OBJECTIVE_LABEL="_globalmse"
        ;;
    *)
        echo "ERROR: unsupported XGB_TUNING_SELECTION_OBJECTIVE ${SELECTION_OBJECTIVE}" >&2
        exit 1
        ;;
esac
case "${FITTING_MODE}" in
    nested_per_outer_country|global_loco_country_mse) ;;
    *)
        echo "ERROR: unsupported XGB_TUNING_FITTING_MODE ${FITTING_MODE}" >&2
        exit 1
        ;;
esac
case "${OPTIMIZER}" in
    custom_gp|optuna_tpe|optuna_gp) ;;
    *)
        echo "ERROR: unsupported XGB_TUNING_OPTIMIZER ${OPTIMIZER}" >&2
        exit 1
        ;;
esac
case "${FEATURE_SCOPE}" in
    baseline|single_exposure|full_exposome|domain_balanced_k10) ;;
    *) echo "ERROR: unsupported XGB_TUNING_FEATURE_SCOPE ${FEATURE_SCOPE}" >&2; exit 1 ;;
esac
case "${INCLUDE_MANUAL_INCUMBENT}" in
    0|1|false|true|no|yes) ;;
    *) echo "ERROR: XGB_TUNING_INCLUDE_MANUAL_INCUMBENT must be boolean" >&2; exit 1 ;;
esac
if [ "${FITTING_MODE}" = "global_loco_country_mse" ]; then
    if [ "${SELECTION_OBJECTIVE}" != "mean_country_mse" ]; then
        echo "ERROR: global_loco_country_mse requires mean_country_mse" >&2
        exit 1
    fi
    OBJECTIVE_LABEL="_globalmse_${OPTIMIZER}"
fi
if [ "${FEATURE_SCOPE}" = "baseline" ]; then
    OBJECTIVE_LABEL="_baseline_gp${N_TRIALS}"
elif [ "${FEATURE_SCOPE}" = "single_exposure" ]; then
    OBJECTIVE_LABEL="_singles_gp${N_TRIALS}"
elif [ "${FEATURE_SCOPE}" = "domain_balanced_k10" ]; then
    OBJECTIVE_LABEL="_domaink10_gp${N_TRIALS}"
fi
if [ -n "${PARENT_RUN_SIGNATURE}" ]; then
    if ! [[ "${PARENT_RUN_SIGNATURE}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "ERROR: XGB_TUNING_PARENT_RUN_SIGNATURE may contain only letters, digits, '-' and '_'" >&2
        exit 1
    fi
    if [ "${FITTING_MODE}" != "global_loco_country_mse" ]; then
        echo "ERROR: a continuation requires global_loco_country_mse" >&2
        exit 1
    fi
    OBJECTIVE_LABEL="_gp_extend${N_TRIALS}"
fi
case "${COUNTRY_VARIANT}" in
    a)
        if [ -n "${PARENT_RUN_SIGNATURE}" ]; then
            JOB_NAME="xgb_global_gp_extend${N_TRIALS}_current3"
            RUN_ID="current3_gp_extend${N_TRIALS}_${SUBMITTED_AT}"
        else
            JOB_NAME="xgb_nested_loco_all${OBJECTIVE_LABEL}"
            RUN_ID="current3${OBJECTIVE_LABEL}_${SUBMITTED_AT}"
        fi
        ;;
    historical_a)
        if [ -n "${PARENT_RUN_SIGNATURE}" ]; then
            JOB_NAME="xgb_global_gp_extend${N_TRIALS}_historical5"
            RUN_ID="historical5_gp_extend${N_TRIALS}_${SUBMITTED_AT}"
        else
            JOB_NAME="xgb_nested_loco_all_historical5${OBJECTIVE_LABEL}"
            RUN_ID="historical5${OBJECTIVE_LABEL}_${SUBMITTED_AT}"
        fi
        ;;
    *)
        echo "ERROR: unsupported country variant ${COUNTRY_VARIANT}; expected a or historical_a" >&2
        exit 1
        ;;
esac

echo "Launching ${JOB_NAME} | REPRO_DATA_ROOT=${REPRO_DATA_ROOT} | COUNTRY_VARIANT=${COUNTRY_VARIANT} | FITTING_MODE=${FITTING_MODE} | OPTIMIZER=${OPTIMIZER} | FEATURE_SCOPE=${FEATURE_SCOPE} | MANUAL_INCUMBENT=${INCLUDE_MANUAL_INCUMBENT} | SELECTION_OBJECTIVE=${SELECTION_OBJECTIVE} | N_JOBS=${N_JOBS} | N_TRIALS=${N_TRIALS} | TIME_LIMIT=${TIME_LIMIT} | PARENT_RUN_SIGNATURE=${PARENT_RUN_SIGNATURE:-<none>} | RUN_ID=${RUN_ID}"
run -t "${TIME_LIMIT}" -j "${JOB_NAME}" -c "${N_JOBS}" -m 40 \
    -o "${SCHEDULER_LOG_DIR}/${JOB_NAME}_%j.out" \
    -e "${SCHEDULER_LOG_DIR}/${JOB_NAME}_%j.err" \
    bash "${PROJECT_ROOT}/scripts/jobs/run_xgb_nested_loco_tuning.sh" \
    "${REPRO_DATA_ROOT}" "${N_JOBS}" "${N_TRIALS}" "${COUNTRY_VARIANT}" "${RUN_ID}" "${SELECTION_OBJECTIVE}" "${FITTING_MODE}" "${OPTIMIZER}" "${PARENT_RUN_ARGUMENT}" "${FEATURE_SCOPE}" "${INCLUDE_MANUAL_INCUMBENT}"
