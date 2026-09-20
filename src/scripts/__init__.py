"""Compatibility namespace for preserved stage modules.

The legacy numerical modules refer to one another as ``scripts.*``.  The
cluster workers import them in fresh Python processes, so this namespace must
be importable from the active checkout as well as from the copied runtime.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "repo_expo_hoi_bag" / "stages")]
