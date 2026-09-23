"""Command-line interface for the public reproducibility workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from repo_expo_hoi_bag.config.models import ConfigurationError, RuntimePaths, load_paper_config
from repo_expo_hoi_bag.data.contracts import validate_inputs
from repo_expo_hoi_bag.figures.registry import active_targets, load_figure_sets, load_manifest
from repo_expo_hoi_bag.legacy_runtime import LegacyStageError, render_target, run_stage


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repo-expo-hoi-bag")
    parser.add_argument("--config", type=Path, default=Path("config/paper.yaml"))
    parser.add_argument("--repro-data-root", help="External runtime root; defaults to REPRO_DATA_ROOT")
    parser.add_argument(
        "--analysis-run-id",
        help="Required for main run stages; creates an isolated results/analysis_runs/<id> namespace",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-data")
    run = subparsers.add_parser("run")
    run_subparsers = run.add_subparsers(dest="stage", required=True)
    for stage in ("greedy", "evaluate", "analyses"):
        run_subparsers.add_parser(stage)
    sensitivity = run_subparsers.add_parser("sensitivity")
    sensitivity.add_argument(
        "name",
        choices=(
            "domain-imbalance",
            "country-region",
            "whole-exposome-pca",
            "order-cap",
            "cooccurrence-network",
            "education-scanner-baseline",
            "residual-confounds",
            "residualized-bag-target",
            "residualized-bag",
            "residualized-bag-clean",
            "diagnosis-balance",
            "negative-o-arm-comparison",
            "feature-ablation",
            "country-block-null",
            "xgb-nested-loco-tuning",
            "xgb-hpo-cap500-selection",
            "xgb-hpo-cross-test",
            "xgb-tuned-top50-comparison",
            "xgb-frozen-cap500-top50",
            "normative-transfer-summary",
            "normative-transfer-ols",
            "normative-transfer-xgb",
            "normative-transfer-single-xgb",
        ),
    )
    render = subparsers.add_parser("render")
    render.add_argument("--manifest", type=Path, default=Path("outputs/figures/manifest.yaml"))
    render.add_argument("--target", action="append", help="Render only a named manifest target; repeatable")
    verify = subparsers.add_parser("verify")
    verify.add_argument("scope", nargs="?", choices=("manifest",), default="manifest")
    verify.add_argument("--manifest", type=Path, default=Path("outputs/figures/manifest.yaml"))
    return parser


def _write_input_manifest(runtime: RuntimePaths, hashes: dict[str, str]) -> Path:
    output = runtime.results_root / "manifests" / "input_hashes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _repository_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = load_paper_config(config_path)
        runtime = RuntimePaths.from_environment(args.repro_data_root)
        if args.command == "validate-data":
            hashes = validate_inputs(config, root)
            destination = _write_input_manifest(runtime, hashes)
            print(f"Validated public inputs; wrote {destination}")
            return 0
        if args.command == "run":
            run_stage(
                root,
                runtime,
                args.stage,
                config.selected_bags(),
                sensitivity=getattr(args, "name", None),
                analysis_run_id=args.analysis_run_id,
            )
            return 0
        manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
        targets = active_targets(load_manifest(manifest_path))
        if args.command == "render":
            requested = set(args.target or ())
            if requested:
                available = {target.name for target in targets}
                unknown = requested.difference(available)
                if unknown:
                    raise ConfigurationError(f"Unknown or disabled figure target(s): {sorted(unknown)}")
                targets = tuple(target for target in targets if target.name in requested)
            for target in targets:
                print(f"Rendering {target.name} -> {runtime.figures_root}")
                render_target(root, runtime, target.name, config.selected_bags())
            return 0
        if args.command == "verify":
            hashes = validate_inputs(config, root)
            for target in targets:
                reference = root / target.reference
                if not reference.exists():
                    raise ConfigurationError(f"Missing delivered reference for {target.name}: {reference}")
            for figure_set in load_figure_sets(manifest_path):
                missing = [
                    asset for asset in figure_set.assets
                    if not (root / figure_set.root / asset).is_file()
                ]
                if missing:
                    raise ConfigurationError(
                        f"Missing figure-set assets for {figure_set.name}: {missing}"
                    )
            print(f"Verified {len(targets)} figure contracts and {len(hashes)} public input hashes.")
            return 0
    except (ConfigurationError, LegacyStageError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
