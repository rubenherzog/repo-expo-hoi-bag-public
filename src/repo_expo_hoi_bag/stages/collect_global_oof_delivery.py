#!/usr/bin/env python3
"""Assemble the global-OOF paper delivery from its staged run outputs.

Mirrors collect_main_paper_delivery.py, but reads the global-OOF staging run
and writes into the separate global delivery root declared in
config/main_paper_delivery_global_oof.yaml.  It copies only lightweight file
types and refuses to overwrite an existing delivery artifact.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_SUFFIXES = {".csv", ".json", ".pdf", ".png", ".svg", ".tiff", ".txt", ".xlsx"}
CONFIG = CHECKOUT_ROOT / "config" / "main_paper_delivery_global_oof.yaml"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed global-OOF staging run identifier.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _copy(source: Path, target: Path, *, dry_run: bool, copied: list[Path]) -> None:
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite delivery artifact: {target}")
    copied.append(target)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("r2_mode") != "global_oof":
        raise ValueError(f"{args.config} is not a global_oof delivery contract")
    destination = CHECKOUT_ROOT / config["delivery_root"]
    if destination.resolve() == (CHECKOUT_ROOT / "outputs/main/paper/complete").resolve():
        raise ValueError("Refusing to write the global delivery into the country-balanced root")
    staging = CHECKOUT_ROOT / "outputs" / "main" / args.run_id
    sensitivity = staging / "sensitivity"
    if not sensitivity.is_dir():
        raise FileNotFoundError(f"Missing global staging directory: {sensitivity}")

    copied: list[Path] = []
    main_stems = set(config["main_figures"])
    supplementary_stems = set(config["supplementary_figures"])

    # Main figures and their source data.
    for path in sorted((staging / "figures" / "main").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging / "figures" / "main")
        _copy(path, destination / "figures" / "main" / relative, dry_run=args.dry_run, copied=copied)

    # Supplementary figures: every rendered stem the contract asks for.
    figure_roots = [sensitivity / "figures"]
    for root in figure_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stem = path.name.split(".")[0].replace("_source_data", "")
            keep = any(path.name.startswith(s) for s in supplementary_stems | main_stems)
            if not keep and "source_data" not in str(path):
                continue
            relative = path.relative_to(root)
            # Flatten one level of per-figure nesting (e.g. residual_confounds/).
            _copy(path, destination / "figures" / "supplementary" / relative, dry_run=args.dry_run, copied=copied)

    # Table source data: every lightweight table directory in the staging run.
    for source_dir in sorted(sensitivity.iterdir()):
        if source_dir.name == "figures" or not source_dir.is_dir():
            continue
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir)
            _copy(
                path,
                destination / "tables" / "source_data" / source_dir.name / relative,
                dry_run=args.dry_run,
                copied=copied,
            )

    mode = "would collect" if args.dry_run else "collected"
    print(f"{mode} {len(copied)} lightweight artifacts into {destination}")


if __name__ == "__main__":
    main()
