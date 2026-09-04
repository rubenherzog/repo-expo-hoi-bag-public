#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export V3_BASE_OUTPUT_ROOT="${V3_BASE_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export V3_GREEDY_ROOT="${V3_GREEDY_ROOT:-$V3_BASE_OUTPUT_ROOT/greedy}"
export V3_PIPELINE_CONFIG="${V3_PIPELINE_CONFIG:-$REPO_ROOT/config/pipeline.yaml}"
export V3_SMOKE_CONFIG="${V3_SMOKE_CONFIG:-$REPO_ROOT/config/smoke.yaml}"

python -m scripts.run_full_analysis_v3 --variant a
