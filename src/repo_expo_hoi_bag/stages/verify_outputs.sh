#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/outputs/variant_a}"
python -m scripts.verify_outputs --output-root "$OUTPUT_ROOT"
