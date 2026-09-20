#!/bin/bash
# Submit the historical five-country-exclusion nested-LOCO tuning reproduction.
# France, Italy, Egypt, Greece, and Poland are excluded via the versioned
# `historical_a` BAG policy; the underlying launcher creates a new immutable run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_xgb_nested_loco_tuning.sh" historical_a
