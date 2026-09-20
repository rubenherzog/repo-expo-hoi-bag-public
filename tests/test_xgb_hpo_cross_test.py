from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "src" / "repo_expo_hoi_bag" / "stages"
CORE = ROOT / "src" / "repo_expo_hoi_bag" / "core"
for path in (STAGES, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_xgb_hpo_cross_test import resolve_cross_artifacts


def test_cross_artifacts_require_three_distinct_completed_local_selected_files(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    root = checkout / "outputs" / "xgb_nested_loco_tuning"
    paths = []
    for label in ("baseline", "single", "full"):
        path = root / label / "selected_xgb_configs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"selected": []}))
        paths.append(path.relative_to(checkout).as_posix())
    assert resolve_cross_artifacts(checkout, ",".join(paths)) == [path.resolve() for path in [root / "baseline" / "selected_xgb_configs.json", root / "single" / "selected_xgb_configs.json", root / "full" / "selected_xgb_configs.json"]]
    with pytest.raises(ValueError, match="distinct"):
        resolve_cross_artifacts(checkout, ",".join([paths[0], paths[0], paths[2]]))
