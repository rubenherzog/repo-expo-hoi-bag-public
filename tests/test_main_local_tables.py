from pathlib import Path


def test_main_local_workbook_has_explicit_completion_status() -> None:
    source = Path("src/repo_expo_hoi_bag/stages/build_main_local_tables.py").read_text(encoding="utf-8")
    assert "Supplementary_Tables_main_k10.xlsx" in source
    assert "pending_cluster" in source
    assert "outputs/dedup" not in source
