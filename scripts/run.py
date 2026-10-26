#!/usr/bin/env python3
"""Run one DiscernMem experiment with the reported settings.

Examples
--------
    # Main result: PDDL on AutoGen with the full method
    python scripts/run.py --benchmark pddl --mas autogen

    # Ablation and hyper-parameter variants
    python scripts/run.py --benchmark kodcode --mas camel --variant w_o_dmr
    python scripts/run.py --benchmark pddl --mas autogen --set max_evolution_rounds=1

    # Smoke test on the first three tasks, and inspecting the expanded command
    python scripts/run.py --benchmark pddl --mas autogen --num_tasks 3 --run_id smoke
    python scripts/run.py --benchmark popqa --mas macnet --dry-run

Service endpoints and model names are read from ``.env`` (see ``.env.example``).
Anything after ``--`` is passed unchanged to ``scripts/run_benchmarks.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import launch  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", required=True, choices=launch.BENCHMARKS)
    parser.add_argument("--mas", required=True, choices=launch.MAS_TYPES)
    parser.add_argument(
        "--variant", default="full", choices=launch.available_variants(),
        help="Method variant (default: full)",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override one method-config field, e.g. --set max_injected_memories=2 (repeatable)",
    )
    parser.add_argument("--num_tasks", type=int, help="Run only the first N tasks (smoke tests)")
    parser.add_argument("--run_id", help="Default: <benchmark>_<mas>_<variant>[_<overrides>]")
    parser.add_argument("--results_root", default="results")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the expanded runner command and method-config fingerprint, then exit",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Do not ping the services first")
    return parser


def prepare(args: argparse.Namespace, *, require_env: bool) -> dict:
    """Resolve everything needed to launch one run."""
    env = launch.load_environment(require=require_env)
    for key in launch.REQUIRED_ENV:
        env.setdefault(key, f"${key}")
    protocol = launch.load_protocol(args.benchmark)
    if args.num_tasks is not None:
        protocol["num_tasks"] = int(args.num_tasks)
    overrides = launch.parse_overrides(args.overrides)
    config = launch.build_method_config(
        args.benchmark,
        variant=args.variant,
        overrides=overrides,
        data_split=protocol["data_split"],
        start_from=int(protocol["start_from"]),
        num_tasks=int(protocol["num_tasks"]),
    )
    run_id = args.run_id or launch.default_run_id(
        args.benchmark, args.mas, launch.label_for(args.variant, overrides)
    )
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    config_path = launch.method_config_path(results_root, run_id, args.benchmark, args.mas)
    runner_args = launch.build_runner_args(
        benchmark=args.benchmark,
        mas=args.mas,
        run_id=run_id,
        results_root=results_root,
        method_config=config_path,
        protocol=protocol,
        env=env,
        extra_args=getattr(args, "passthrough", []),
    )
    return {
        "env": env,
        "config": config,
        "config_path": config_path,
        "command": launch.runner_command(runner_args),
        "run_id": run_id,
        "leaf": launch.leaf_dir(results_root, run_id, args.benchmark, args.mas),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.passthrough = passthrough
    try:
        plan = prepare(args, require_env=not args.dry_run)
    except (launch.LaunchError, ValueError) as exc:
        parser.error(str(exc))

    summary = {
        "run_id": plan["run_id"],
        "variant": args.variant,
        "method_config_changes": launch.config_changes(plan["config"]),
        "method_config_fingerprint": plan["config"].fingerprint,
        "tasks": len(plan["config"].split_manifest.learning_task_ids),
        "results": str(plan["leaf"].relative_to(PROJECT_ROOT))
        if plan["leaf"].is_relative_to(PROJECT_ROOT) else str(plan["leaf"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(launch.format_command(plan["command"]))
    if args.dry_run:
        return 0

    if plan["config_path"].exists():
        existing = launch.DiscernMemConfig.from_yaml(plan["config_path"])
        if existing.fingerprint != plan["config"].fingerprint:
            parser.error(
                f"{plan['config_path']} already holds a different configuration; "
                "choose another --run_id"
            )
    plan["config"].export_yaml(plan["config_path"])
    if not args.skip_preflight:
        try:
            launch.preflight(plan["env"])
        except launch.LaunchError as exc:
            parser.error(str(exc))
    return launch.run_command(plan["command"], plan["env"])


if __name__ == "__main__":
    raise SystemExit(main())
