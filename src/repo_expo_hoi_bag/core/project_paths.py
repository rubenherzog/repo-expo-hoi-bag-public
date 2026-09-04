from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))

# Keep legacy V3 env names for compatibility.
BASE_OUTPUT_ROOT = Path(os.environ.get("V3_BASE_OUTPUT_ROOT", str(OUTPUT_ROOT)))
GREEDY_ROOT = Path(os.environ.get("V3_GREEDY_ROOT", str(BASE_OUTPUT_ROOT / "greedy")))

RAW_DATA_CSV = DATA_ROOT / "all_exposome_bag_clean_expo63_countryyear_only_complete_cases.csv"
FEATURE_NAMES_CSV = DATA_ROOT / "exposome_feature_names.csv"
