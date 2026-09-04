#!/usr/bin/env bash
set -euo pipefail

: "${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to a writable external directory}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run() {
  "${PYTHON_BIN}" -m repo_expo_hoi_bag.cli --config "${REPO_ROOT}/config/paper.yaml" \
    --repro-data-root "${REPRO_DATA_ROOT}" "$@"
}

run validate-data
run run greedy
run run evaluate
run run analyses
run render
run verify
