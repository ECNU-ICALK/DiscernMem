#!/usr/bin/env python3
"""Run a group of DiscernMem experiments (ablations or hyper-parameter sweeps).

Groups are defined in ``configs/discernmem/sweeps.yaml``; every run uses the
same settings as ``scripts/run.py`` plus the group's overrides.

Examples
--------
    # Ablation table (Full, w/o DMR, w/o BLM, w/o both) for one benchmark + MAS
    python scripts/run_sweep.py ablation --benchmark pddl --mas autogen --jobs 4

    # Sensitivity sweeps (AutoGen); restrict to one sweep or one benchmark
    python scripts/run_sweep.py hparam --only repair_rounds --jobs 5
    python scripts/run_sweep.py hparam --benchmark kodcode --dry-run

Results go to ``<results_root>/<run_id>/...``; a ``summary.md`` per group is
written under ``<results_root>/_sweeps/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import launch  # noqa: E402


def load_groups() -> dict:
    return yaml.safe_load(launch.SWEEPS_FILE.read_text(encoding="utf-8")) or {}


def plan_ablation(group: dict, args: argparse.Namespace) -> list[dict]:
    if not args.benchmark or not args.mas:
        raise launch.LaunchError("ablation requires --benchmark and --mas")
    return [
        {"benchmark": args.benchmark, "mas": args.mas, "variant": variant, "overrides": {},
         "label": f"ablation_{variant}"}
        for variant in group["variants"]
    ]


def plan_hparam(group: dict, args: argparse.Namespace) -> list[dict]:
    mas = args.mas or group.get("mas", "autogen")
    only = set(args.only or [])
    entries = []
    for name, sweep in group["sweeps"].items():
        if only and name not in only:
            continue
        for benchmark in sweep["benchmarks"]:
            if args.benchmark and benchmark != args.benchmark:
                continue
            for value in sweep["values"]:
                entries.append({
                    "benchmark": benchmark, "mas": mas, "variant": "full",
                    "overrides": {sweep["key"]: value},
                    "label": f"{name}_{value}", "sweep": name, "value": value,
                })
    return entries


def expand(entry: dict, group: dict, args: argparse.Namespace, env: dict, results_root: Path) -> dict:
    protocol = launch.load_protocol(entry["benchmark"])
    if args.num_tasks is not None:
        protocol["num_tasks"] = int(args.num_tasks)
    config = launch.build_method_config(
        entry["benchmark"], variant=entry["variant"], overrides=entry["overrides"],
        data_split=protocol["data_split"], start_from=int(protocol["start_from"]),
        num_tasks=int(protocol["num_tasks"]),
    )
    run_id = f"{entry['benchmark']}_{entry['mas']}_{entry['label']}"
    config_path = launch.method_config_path(results_root, run_id, entry["benchmark"], entry["mas"])
    runner_args = launch.build_runner_args(
        benchmark=entry["benchmark"], mas=entry["mas"], run_id=run_id,
        results_root=results_root, method_config=config_path, protocol=protocol,
        env=env, runner_overrides=group.get("runner"),
    )
    return {**entry, "run_id": run_id, "config": config, "config_path": config_path,
            "command": launch.runner_command(runner_args),
            "leaf": launch.leaf_dir(results_root, run_id, entry["benchmark"], entry["mas"])}


def read_score(leaf: Path) -> float | None:
    path = leaf / "results" / "eval_results.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("pass_rate")
    return 100.0 * float(value) if isinstance(value, (int, float)) else None


def run_one(job: dict, env: dict, log_dir: Path, resume: bool) -> dict:
    if resume and read_score(job["leaf"]) is not None:
        print(f"[skip ] {job['run_id']} (finished)", flush=True)
        return {**job, "returncode": 0, "score": read_score(job["leaf"]), "hours": 0.0}
    if job["leaf"].exists():
        print(f"[block] {job['run_id']}: {job['leaf']} exists; remove it or use --resume", flush=True)
        return {**job, "returncode": None, "score": None, "hours": 0.0}
    job["config"].export_yaml(job["config_path"])
    started = time.time()
    print(f"[start] {job['run_id']}", flush=True)
    code = launch.run_command(job["command"], env, log_dir / f"{job['run_id']}.log")
    hours = (time.time() - started) / 3600.0
    score = read_score(job["leaf"])
    print(f"[done ] {job['run_id']} rc={code} {hours:.2f}h score={score}", flush=True)
    return {**job, "returncode": code, "score": score, "hours": hours}


def write_summary(path: Path, name: str, rows: list[dict]) -> None:
    lines = [f"# DiscernMem {name}", "", "| run_id | variant | overrides | score (%) | rc |", "|---|---|---|---:|---|"]
    for row in rows:
        score = "" if row["score"] is None else f"{row['score']:.2f}"
        overrides = ", ".join(f"{k}={v}" for k, v in row["overrides"].items())
        lines.append(f"| {row['run_id']} | {row['variant']} | {overrides} | {score} | {row['returncode']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    groups = load_groups()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", choices=sorted(groups))
    parser.add_argument("--benchmark", choices=launch.BENCHMARKS)
    parser.add_argument("--mas", choices=launch.MAS_TYPES)
    parser.add_argument("--only", action="append", help="hparam: run only this sweep (repeatable)")
    parser.add_argument("--num_tasks", type=int, help="Run only the first N tasks (smoke tests)")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent runs")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already finished")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    group = groups[args.group]
    try:
        env = launch.load_environment(require=not args.dry_run)
        for key in launch.REQUIRED_ENV:
            env.setdefault(key, f"${key}")
        results_root = Path(args.results_root)
        if not results_root.is_absolute():
            results_root = PROJECT_ROOT / results_root
        planner = plan_ablation if args.group == "ablation" else plan_hparam
        jobs = [expand(entry, group, args, env, results_root) for entry in planner(group, args)]
    except launch.LaunchError as exc:
        parser.error(str(exc))
    if not jobs:
        parser.error("no runs selected")

    for job in jobs:
        print(f"[{job['run_id']}] fingerprint={job['config'].fingerprint} "
              f"changes={launch.config_changes(job['config'])}")
        if args.dry_run:
            print("    " + launch.format_command(job["command"]))
    if args.dry_run:
        return 0

    if not args.skip_preflight:
        try:
            launch.preflight(env)
        except launch.LaunchError as exc:
            parser.error(str(exc))
    log_dir = results_root / "_sweeps" / args.group / "logs"
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda job: run_one(job, env, log_dir, args.resume), jobs))
    write_summary(results_root / "_sweeps" / args.group / "summary.md", args.group, rows)
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
