from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_main_k10_diagnosis_balance_fix.sh"


def test_diagnosis_balance_launcher_is_narrow_and_scheduler_driven() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "diagnosis-balance structural 40" in text
    assert "diagnosis-balance functional 40" in text
    assert text.count("\nrun -t 12:00") == 2
    assert 'RUN_ID="${MAIN_K10_CLUSTER_RUN_ID:-main_k10_release_20260916}"' in text
    assert "main_k10_adapter_manifest.json" in text
    assert "prepare_main_k10_cluster_adapter.sh" not in text
    assert "Refusing to replace completed diagnosis-balance summaries" in text


def test_diagnosis_balance_dry_run_never_contacts_scheduler() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    dry_run = text.split('if [[ "$MODE" == "--dry-run" ]]; then', 1)[1].split(
        "\nfi", 1
    )[0]
    assert "--preflight" in dry_run
    assert "\nrun -t " not in dry_run
