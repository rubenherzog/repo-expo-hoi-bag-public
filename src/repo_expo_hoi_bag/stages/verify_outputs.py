#!/usr/bin/env python3
"""Verify pipeline outputs against the consolidated manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.pipeline_config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify pipeline outputs using the final pipeline manifest.")
    parser.add_argument("--output-root", type=str, required=False,
                        help="Variant output root containing pipeline_manifest.json.")
    parser.add_argument("--manifest", type=str, required=False,
                        help="Path to a pipeline manifest JSON file.")
    parser.add_argument("--config", type=str, required=False,
                        help="Optional pipeline config YAML file for audit contracts.")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(manifest: dict) -> tuple[int, list[str]]:
    missing_files: list[str] = []
    invalid_entries: list[str] = []
    for artifact in manifest.get("artifacts", []):
        path_text = artifact.get("path")
        if not path_text:
            invalid_entries.append(str(artifact))
            continue
        path = Path(path_text)
        if not path.exists():
            missing_files.append(str(path))
    return missing_files, invalid_entries


def verify_audit_contract(config_path: Path, output_root: Path) -> tuple[int, list[str]]:
    config = load_pipeline_config(config_path)
    missing_files: list[str] = []
    for artifact in config.get("audit", {}).get("required_outputs", []):
        path_template = artifact.get("path")
        if not path_template:
            continue
        path = Path(path_template.format(output_root=str(output_root.resolve())))
        if not path.exists():
            missing_files.append(str(path))
    return missing_files, []


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.output_root:
        manifest_path = Path(args.output_root) / "pipeline_manifest.json"
    else:
        raise ValueError("Either --output-root or --manifest must be provided.")

    manifest = load_manifest(manifest_path)
    missing_files, invalid_entries = verify_manifest(manifest)
    if invalid_entries:
        print("[ERROR] Invalid artifact entries in manifest:")
        for entry in invalid_entries:
            print(f"  - {entry}")
    if missing_files:
        print("[ERROR] Missing required manifest files:")
        for path in missing_files:
            print(f"  - {path}")

    audit_missing: list[str] = []
    if args.config and args.output_root:
        audit_missing, _ = verify_audit_contract(Path(args.config), Path(args.output_root))
        if audit_missing:
            print("[ERROR] Missing required audit contract files:")
            for path in audit_missing:
                print(f"  - {path}")

    if invalid_entries or missing_files or audit_missing:
        sys.exit(1)
    print("[OK] All manifest artifacts and audit contract outputs are present.")
    sys.exit(0)


if __name__ == "__main__":
    main()
