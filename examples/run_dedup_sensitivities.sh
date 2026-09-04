#!/usr/bin/env bash
set -euo pipefail

: "${REPRO_DATA_ROOT:?Set REPRO_DATA_ROOT to a writable external directory}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run() {
  "${PYTHON_BIN}" -m repo_expo_hoi_bag.cli --config "${REPO_ROOT}/config/paper.yaml" \
    --repro-data-root "${REPRO_DATA_ROOT}" run sensitivity "$1"
}

for sensitivity in \
  domain-imbalance \
  country-region \
  whole-exposome-pca \
  order-cap \
  cooccurrence-network \
  education-scanner-baseline \
  residual-confounds \
  residualized-bag-target \
  residualized-bag \
  diagnosis-balance \
  normative-transfer-summary \
  normative-transfer-ols \
  normative-transfer-xgb
do
  run "${sensitivity}"
done

DEDUP_NS=dedup_neg_o SYN_OINFO_NEGATIVE=1 \
  "${PYTHON_BIN}" -m repo_expo_hoi_bag.cli --config "${REPO_ROOT}/config/paper.yaml" \
  --repro-data-root "${REPRO_DATA_ROOT}" run sensitivity negative-o-arm-comparison
