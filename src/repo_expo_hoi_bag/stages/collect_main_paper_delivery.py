#!/usr/bin/env python3
"""Collect lightweight main-k10 sensitivity deliveries into the paper package."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_SUFFIXES = {".csv", ".json", ".pdf", ".png", ".svg", ".tiff", ".txt", ".xlsx"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed main-k10 sensitivity run identifier.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _copy_tree(source: Path, destination: Path, *, dry_run: bool) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite delivery artifact: {target}")
        copied.append(target)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return copied


def main() -> None:
    args = _args()
    staging = CHECKOUT_ROOT / "outputs" / "main" / args.run_id / "sensitivity"
    destination = CHECKOUT_ROOT / "outputs" / "main" / "paper" / "complete"
    figures = staging / "figures"
    if not staging.is_dir():
        raise FileNotFoundError(f"Missing local sensitivity delivery staging directory: {staging}")
    copied: list[Path] = []
    if figures.is_dir():
        copied.extend(_copy_tree(figures, destination / "figures" / "supplementary", dry_run=args.dry_run))
    table_sources = [path for path in staging.iterdir() if path.name != "figures"]
    for source in table_sources:
        if source.is_dir():
            copied.extend(_copy_tree(source, destination / "tables" / "source_data" / source.name, dry_run=args.dry_run))
        elif source.is_file() and source.suffix.lower() in ALLOWED_SUFFIXES:
            target = destination / "tables" / "source_data" / source.name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite delivery artifact: {target}")
            copied.append(target)
            if not args.dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    mode = "would collect" if args.dry_run else "collected"
    print(f"{mode} {len(copied)} lightweight artifacts from {staging}")


if __name__ == "__main__":
    main()
