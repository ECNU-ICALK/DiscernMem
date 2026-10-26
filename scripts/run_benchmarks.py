#!/usr/bin/env python3
"""
Benchmark runner for the memory method on multi-agent systems.

  Datasets: kodcode, triviaqa, popqa, pddl
  MAS:      autogen, camel, macnet

Prerequisite: an OpenAI-compatible LLM service and an embedding service.
Most users should launch experiments with scripts/run.py; see docs/reproduce.md.
"""
import argparse
import copy
import datetime as _dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

_PROFILE = os.environ.get("DISCERNMEM_PROFILE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_benchmarks")
_RUN_FILE_HANDLER = None
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_RUN_STDIO_FILE = None


class _TeeStream:
    def __init__(self, stream, file_obj):
        self.stream = stream
        self.file_obj = file_obj

    def write(self, data):
        self.stream.write(data)
        self.file_obj.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file_obj.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _restore_run_stdio_tee():
    global _RUN_STDIO_FILE
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    if _RUN_STDIO_FILE is not None:
        _RUN_STDIO_FILE.close()
        _RUN_STDIO_FILE = None


def _enable_run_stdio_tee(log_file: str):
    global _RUN_STDIO_FILE
    _restore_run_stdio_tee()
    _RUN_STDIO_FILE = open(log_file, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(_ORIGINAL_STDOUT, _RUN_STDIO_FILE)
    sys.stderr = _TeeStream(_ORIGINAL_STDERR, _RUN_STDIO_FILE)


def _sanitize_run_segment(value, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"--{field_name} cannot be empty")
    if text in {".", ".."} or os.sep in text or (os.altsep and os.altsep in text):
        raise ValueError(f"--{field_name} must be a single path segment: {text!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(
            f"--{field_name} may only contain letters, numbers, dot, dash, and underscore: {text!r}"
        )
    return text


def _resolve_results_root(results_root: str) -> str:
    root = results_root or "./results"
    if os.path.isabs(root):
        return os.path.abspath(root)
    return os.path.abspath(os.path.join(PROJECT_ROOT, root))


def _run_artifact_paths(args, benchmark: str) -> dict:
    run_id = _sanitize_run_segment(args.run_id, "run_id")
    bench = _sanitize_run_segment(benchmark, "benchmark")
    mas_type = _sanitize_run_segment(args.mas_type, "mas_type")
    memory_backend = _sanitize_run_segment(args.memory_backend, "memory_backend")
    run_root = os.path.join(_resolve_results_root(args.results_root), run_id)
    leaf_dir = os.path.join(run_root, bench, mas_type, memory_backend)
    results_dir = os.path.join(leaf_dir, "results")
    memory_dir = os.path.join(leaf_dir, "memory")
    logs_dir = os.path.join(leaf_dir, "logs")
    return {
        "run_root_dir": run_root,
        "leaf_dir": leaf_dir,
        "results_dir": results_dir,
        "memory_dir": memory_dir,
        "logs_dir": logs_dir,
        "log_file": os.path.join(logs_dir, "run.log"),
        "run_config_file": os.path.join(results_dir, "run_config.json"),
        "eval_results_file": os.path.join(results_dir, "eval_results.json"),
        "partial_eval_results_file": os.path.join(
            results_dir, "partial_eval_results.json"
        ),
        "task_results_file": os.path.join(results_dir, "task_results.jsonl"),
        "discernmem_storage_dir": os.path.join(memory_dir, "discernmem"),
    }


def _prepare_run_artifacts(args, benchmark: str):
    if not getattr(args, "run_id", None):
        return
    paths = _run_artifact_paths(args, benchmark)
    leaf_dir = paths["leaf_dir"]
    if os.path.exists(leaf_dir) and os.listdir(leaf_dir):
        if not getattr(args, "overwrite_run", False):
            raise FileExistsError(
                f"Run artifact directory already exists and is not empty: {leaf_dir}. "
                "Use a new --run_id, or pass --overwrite_run to replace this leaf experiment."
            )
        shutil.rmtree(leaf_dir)

    for key in ("results_dir", "memory_dir", "logs_dir"):
        os.makedirs(paths[key], exist_ok=True)

    args.storage_dir = paths["results_dir"]
    if not args.discernmem_storage_dir:
        args.discernmem_storage_dir = paths["discernmem_storage_dir"]

    args._artifact_paths = paths
    args._run_root_dir = paths["run_root_dir"]
    args._run_leaf_dir = paths["leaf_dir"]
    args._run_results_dir = paths["results_dir"]
    args._run_log_file = paths["log_file"]
    args._run_config_file = paths["run_config_file"]


def _setup_run_file_logging(log_file: str | None):
    global _RUN_FILE_HANDLER
    _restore_run_stdio_tee()
    root_logger = logging.getLogger()
    if _RUN_FILE_HANDLER is not None:
        root_logger.removeHandler(_RUN_FILE_HANDLER)
        _RUN_FILE_HANDLER.close()
        _RUN_FILE_HANDLER = None
    if not log_file:
        return
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(handler)
    _RUN_FILE_HANDLER = handler
    _enable_run_stdio_tee(log_file)


def _redact_argv(argv: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for item in argv:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        name = item.split("=", 1)[0].lower()
        if any(token in name for token in ("key", "token", "password", "secret")):
            if "=" in item:
                redacted.append(f"{item.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(item)
                skip_next = True
            continue
        redacted.append(item)
    return redacted


def _jsonable_args(args) -> dict:
    out = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if any(
            token in key.lower() for token in ("key", "token", "password", "secret")
        ):
            out[key] = "<redacted>" if value else value
        else:
            out[key] = value
    return out


def _git_metadata() -> dict:
    def run_git(*parts):
        try:
            proc = subprocess.run(
                ["git", "-C", PROJECT_ROOT, *parts],
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except Exception:
            return None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
    }


def _write_run_config(args, benchmark: str):
    config_file = getattr(args, "_run_config_file", None)
    if not config_file:
        return
    started_at = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()
    payload = {
        "run_id": args.run_id,
        "benchmark": benchmark,
        "mas_type": args.mas_type,
        "memory_backend": args.memory_backend,
        "started_at_utc": started_at,
        "project_root": PROJECT_ROOT,
        "command": _redact_argv(sys.argv),
        "args": _jsonable_args(args),
        "resolved_paths": getattr(args, "_artifact_paths", {}),
        "git": _git_metadata(),
    }
    if benchmark == "kodcode":
        from common.utils.code_utils import get_kodcode_executor_metadata

        payload["kodcode_executor"] = get_kodcode_executor_metadata(
            args.kodcode_env_mode,
            timeout=args.kodcode_test_timeout_seconds,
            cuda_visible_devices=args.kodcode_cuda_visible_devices,
        )
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _write_partial_eval_results(
    results_dir: str,
    *,
    benchmark: str,
    mas_type: str,
    memory_backend: str,
    run_id: str,
    metrics: dict,
    complete: bool = False,
):
    payload = {
        "benchmark": benchmark,
        "mas_type": mas_type,
        "memory_backend": memory_backend,
        "run_id": run_id,
        "complete": bool(complete),
        **metrics,
        "updated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(results_dir, exist_ok=True)
    destination = os.path.join(results_dir, "partial_eval_results.json")
    temporary = f"{destination}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def _append_task_result(results_dir: str, record: dict) -> str:
    os.makedirs(results_dir, exist_ok=True)
    destination = os.path.join(results_dir, "task_results.jsonl")
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return destination


def _consume_retrieval_usage(centralized_memory):
    records = centralized_memory.get_and_reset_retrieval_usage() or []
    injected_count = int(
        sum(int(record.get("injected_cards", 0) or 0) for record in records)
    )
    return injected_count, records


def _consume_memory_usage(centralized_memory, task_id: str, task_index: int):
    injected_card_occurrences, retrieval_usage = _consume_retrieval_usage(
        centralized_memory
    )
    return (
        injected_card_occurrences,
        retrieval_usage,
        _consume_update_usage(centralized_memory, task_id, task_index),
    )


def _count_active_memory_cards(centralized_memory) -> int:
    if centralized_memory is None:
        return 0
    return int(centralized_memory.storage.count_active_cards())


def _record_task_trajectory(
    centralized_memory,
    read_only_memory: bool,
    *,
    trajectory: dict,
    task_id: str,
    task_description: str,
    outcome: str,
    score: float = 0.0,
    task_index: int = -1,
    metadata: dict | None = None,
):
    completion_kwargs = dict(metadata or {})
    completion_kwargs.update({"score": float(score), "task_index": int(task_index)})
    if read_only_memory:
        logger.info("  [Read-Only Mode] Skipping memory storage")
    else:
        centralized_memory.add_memory(
            trajectory=trajectory,
            task_id=task_id,
            task_description=task_description,
            outcome=outcome,
            **completion_kwargs,
        )


def _source_entities_for_task(
    benchmark: str,
    *payloads,
) -> list[str]:
    text = json.dumps(payloads, ensure_ascii=False, default=str)
    entities = {
        match.group(0) for match in re.finditer(r"\b[A-Za-z_]+\d+[A-Za-z0-9_]*\b", text)
    }
    if str(benchmark).lower() == "pddl":
        entities.update(
            match.group(0)
            for match in re.finditer(
                r"\b(?:ball\d+|room[a-z0-9_]+)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    return sorted(entities, key=lambda value: (value.lower(), value))


def _research_task_metadata(
    args,
    benchmark: str,
    model_name: str,
    generation_config: dict,
    raw_task,
    ground_truth,
    task_family: str,
    environment_config: dict | None = None,
    evaluator_config: dict | None = None,
) -> dict:
    return {
        "execution_type": "official",
        "benchmark": benchmark,
        "data_split": args.data_split,
        "task_family": task_family,
        "raw_task": copy.deepcopy(raw_task),
        "ground_truth": copy.deepcopy(ground_truth),
        "model_name": model_name,
        "mas_type": args.mas_type,
        "memory_backend": args.memory_backend,
        "generation_config": copy.deepcopy(generation_config),
        "environment_config": copy.deepcopy(environment_config or {}),
        "evaluator_config": copy.deepcopy(evaluator_config or {}),
        "source_entities": _source_entities_for_task(
            benchmark,
            raw_task,
            ground_truth,
        ),
    }


def _zero_update_usage(task_id: str, task_index: int) -> dict:
    return {
        "task_id": task_id,
        "task_index": task_index,
        "status": "not_run",
        "candidate_cards": 0,
        "admitted_cards": 0,
        "committed_cards": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_time_seconds": 0.0,
        "latency_ms": 0.0,
    }


def _consume_update_usage(centralized_memory, task_id: str, task_index: int) -> dict:
    record = dict(getattr(centralized_memory, "_last_update_usage", {}) or {})
    if not record:
        return _zero_update_usage(task_id, task_index)
    record.setdefault("task_id", task_id)
    record["task_index"] = task_index
    if hasattr(centralized_memory, "_last_update_usage"):
        centralized_memory._last_update_usage = {}
    return record


def _summarize_memory_cost_stats(
    retrieval_records: list[dict],
    update_records: list[dict],
    api_stats: dict,
    evaluated_tasks: int,
) -> dict:
    flat_retrievals = [
        record
        for task_record in retrieval_records
        for record in task_record.get("retrievals", [])
    ]
    retrieval_calls = len(flat_retrievals)

    def total_retrieval(field: str) -> float:
        return sum(float(record.get(field, 0) or 0) for record in flat_retrievals)

    def avg_per_task(total_value: float) -> float:
        return total_value / evaluated_tasks if evaluated_tasks else 0.0

    def avg_per_retrieval(total_value: float) -> float:
        return total_value / retrieval_calls if retrieval_calls else 0.0

    total_update_calls = sum(
        int(record.get("llm_calls", 0) or 0) for record in update_records
    )
    total_update_prompt = sum(
        int(record.get("prompt_tokens", 0) or 0) for record in update_records
    )
    total_update_completion = sum(
        int(record.get("completion_tokens", 0) or 0) for record in update_records
    )
    total_update_latency = sum(
        float(record.get("latency_ms", 0) or 0) for record in update_records
    )

    total_prompt_tokens = int(api_stats.get("total_prompt_tokens", 0) or 0)
    total_completion_tokens = int(api_stats.get("total_completion_tokens", 0) or 0)
    total_api_calls = int(api_stats.get("total_calls", 0) or 0)

    totals = {
        "retrieved_cards": total_retrieval("retrieved_cards"),
        "expanded_cards": total_retrieval("expanded_cards"),
        "coordinated_cards": total_retrieval("coordinated_cards"),
        "injected_cards": total_retrieval("injected_cards"),
        "injected_tokens_est": total_retrieval("injected_tokens_est"),
        "subgraph_edges": total_retrieval("subgraph_edges"),
        "skipped_over_budget_cards": total_retrieval("skipped_over_budget_cards"),
        "compacted_cards": total_retrieval("compacted_cards"),
        "controller_latency_ms": total_retrieval("latency_ms"),
        "retrieve_latency_ms": total_retrieval("retrieve_latency_ms"),
        "expand_latency_ms": total_retrieval("expand_latency_ms"),
        "coordinate_latency_ms": total_retrieval("coordinate_latency_ms"),
        "serialize_latency_ms": total_retrieval("serialize_latency_ms"),
    }

    per_task = []
    update_by_task = {record.get("task_id"): record for record in update_records}
    for task_record in retrieval_records:
        task_retrievals = task_record.get("retrievals", [])
        task_id = task_record.get("task_id", "")
        update = update_by_task.get(task_id, {})

        def task_total(field: str) -> float:
            return sum(float(record.get(field, 0) or 0) for record in task_retrievals)

        per_task.append(
            {
                "task_id": task_id,
                "task_index": task_record.get("task_index"),
                "retrieval_calls": len(task_retrievals),
                "retrieved_cards": task_total("retrieved_cards"),
                "expanded_cards": task_total("expanded_cards"),
                "coordinated_cards": task_total("coordinated_cards"),
                "injected_cards": task_total("injected_cards"),
                "injected_card_occurrences": task_total("injected_cards"),
                "injected_tokens_est": task_total("injected_tokens_est"),
                "memory_controller_latency_ms": task_total("latency_ms"),
                "memory_update_llm_calls": int(update.get("llm_calls", 0) or 0),
                "memory_update_prompt_tokens": int(update.get("prompt_tokens", 0) or 0),
                "memory_update_latency_ms": float(update.get("latency_ms", 0) or 0),
            }
        )

    return {
        "token_count_note": "injected_tokens_est is a character-based estimate (len/4); total_prompt_tokens uses provider-reported API usage.",
        "injected_card_count_note": "Injected card totals count injection occurrences across retrieval calls; repeated injection of the same memory is counted again.",
        "retrieval_calls": retrieval_calls,
        "avg_retrieval_calls_per_task": avg_per_task(retrieval_calls),
        "total_retrieved_cards": totals["retrieved_cards"],
        "avg_retrieved_cards_per_task": avg_per_task(totals["retrieved_cards"]),
        "avg_retrieved_cards_per_retrieval": avg_per_retrieval(
            totals["retrieved_cards"]
        ),
        "total_expanded_cards": totals["expanded_cards"],
        "avg_expanded_cards_per_task": avg_per_task(totals["expanded_cards"]),
        "avg_expanded_cards_per_retrieval": avg_per_retrieval(totals["expanded_cards"]),
        "total_coordinated_cards": totals["coordinated_cards"],
        "avg_coordinated_cards_per_task": avg_per_task(totals["coordinated_cards"]),
        "avg_coordinated_cards_per_retrieval": avg_per_retrieval(
            totals["coordinated_cards"]
        ),
        "total_injected_cards": totals["injected_cards"],
        "total_injected_card_occurrences": totals["injected_cards"],
        "avg_injected_cards_per_task": avg_per_task(totals["injected_cards"]),
        "avg_injected_card_occurrences_per_task": avg_per_task(
            totals["injected_cards"]
        ),
        "avg_injected_cards_per_retrieval": avg_per_retrieval(totals["injected_cards"]),
        "avg_injected_card_occurrences_per_retrieval": avg_per_retrieval(
            totals["injected_cards"]
        ),
        "total_injected_tokens_est": totals["injected_tokens_est"],
        "avg_injected_tokens_per_task_est": avg_per_task(totals["injected_tokens_est"]),
        "avg_injected_tokens_per_retrieval_est": avg_per_retrieval(
            totals["injected_tokens_est"]
        ),
        "total_skipped_over_budget_cards": totals["skipped_over_budget_cards"],
        "avg_skipped_over_budget_cards_per_task": avg_per_task(
            totals["skipped_over_budget_cards"]
        ),
        "total_compacted_cards": totals["compacted_cards"],
        "avg_memory_controller_latency_ms_per_task": avg_per_task(
            totals["controller_latency_ms"]
        ),
        "avg_memory_controller_latency_ms_per_retrieval": avg_per_retrieval(
            totals["controller_latency_ms"]
        ),
        "avg_retrieve_latency_ms_per_retrieval": avg_per_retrieval(
            totals["retrieve_latency_ms"]
        ),
        "avg_expand_latency_ms_per_retrieval": avg_per_retrieval(
            totals["expand_latency_ms"]
        ),
        "avg_coordinate_latency_ms_per_retrieval": avg_per_retrieval(
            totals["coordinate_latency_ms"]
        ),
        "avg_serialize_latency_ms_per_retrieval": avg_per_retrieval(
            totals["serialize_latency_ms"]
        ),
        "memory_update_llm_calls": total_update_calls,
        "avg_memory_update_llm_calls_per_task": avg_per_task(total_update_calls),
        "memory_update_prompt_tokens": total_update_prompt,
        "avg_memory_update_prompt_tokens_per_task": avg_per_task(total_update_prompt),
        "memory_update_completion_tokens": total_update_completion,
        "avg_memory_update_completion_tokens_per_task": avg_per_task(
            total_update_completion
        ),
        "avg_memory_update_latency_ms_per_task": avg_per_task(total_update_latency),
        "total_prompt_tokens": total_prompt_tokens,
        "avg_total_prompt_tokens_per_task": avg_per_task(total_prompt_tokens),
        "total_completion_tokens": total_completion_tokens,
        "avg_total_completion_tokens_per_task": avg_per_task(total_completion_tokens),
        "total_api_calls": total_api_calls,
        "avg_api_calls_per_task": avg_per_task(total_api_calls),
        "non_update_prompt_tokens": max(total_prompt_tokens - total_update_prompt, 0),
        "avg_non_update_prompt_tokens_per_task": avg_per_task(
            max(total_prompt_tokens - total_update_prompt, 0)
        ),
        "per_task": per_task,
    }


def _summarize_memory_injection_stats(
    injected_card_occurrences_by_task: list[int],
    retrieval_records: list[dict],
    evaluated_tasks: int,
) -> dict:
    tasks_with_memory_injection = sum(
        1 for count in injected_card_occurrences_by_task if count > 0
    )
    total_injected_card_occurrences = int(sum(injected_card_occurrences_by_task))
    identity_stats_available = any(
        "injected_dynamic_memories" in retrieval
        or "injected_static_insights" in retrieval
        for task_record in retrieval_records
        for retrieval in task_record.get("retrievals", [])
    )
    dynamic_refs = {
        (str(item.get("memory_id")), str(item.get("version", 0)))
        for task_record in retrieval_records
        for retrieval in task_record.get("retrievals", [])
        for item in retrieval.get("injected_dynamic_memories", [])
        if item.get("memory_id") is not None
    }
    static_ids = {
        str(item.get("insight_id"))
        for task_record in retrieval_records
        for retrieval in task_record.get("retrievals", [])
        for item in retrieval.get("injected_static_insights", [])
        if item.get("insight_id") is not None
    }
    return {
        "counting_unit": "card injection occurrences across retrieval calls; repeated injection of the same memory counts again",
        "tasks_with_memory_injection": tasks_with_memory_injection,
        "tasks_without_memory_injection": max(
            evaluated_tasks - tasks_with_memory_injection, 0
        ),
        "memory_injection_task_rate": (
            tasks_with_memory_injection / evaluated_tasks if evaluated_tasks else 0.0
        ),
        "total_injected_card_occurrences": total_injected_card_occurrences,
        "avg_injected_card_occurrences_per_task": (
            total_injected_card_occurrences / evaluated_tasks
            if evaluated_tasks
            else 0.0
        ),
        "injected_card_occurrences_by_task": injected_card_occurrences_by_task,
        "unique_memory_identity_stats_available": identity_stats_available,
        "unique_dynamic_memory_ids_injected": (
            len({memory_id for memory_id, _ in dynamic_refs})
            if identity_stats_available
            else None
        ),
        "unique_dynamic_memory_versions_injected": (
            len(dynamic_refs) if identity_stats_available else None
        ),
        "unique_static_insights_injected": (
            len(static_ids) if identity_stats_available else None
        ),
    }


def _replay_runs_from_update_usage(update_usage: dict | None) -> int:
    update_usage = update_usage or {}
    direct_count = update_usage.get("replay_runs")
    if direct_count is not None:
        return int(direct_count or 0)
    return sum(
        int(item.get("role_replays", 0) or 0) + int(item.get("version_replays", 0) or 0)
        for item in update_usage.get("replay", []) or []
        if isinstance(item, dict)
    )


class IncrementalExperimentMetrics:
    """Accumulate progress metrics once per completed task without rescanning history."""

    _RETRIEVAL_FIELDS = (
        "retrieved_cards",
        "expanded_cards",
        "coordinated_cards",
        "injected_cards",
        "injected_tokens_est",
        "subgraph_edges",
        "skipped_over_budget_cards",
        "compacted_cards",
        "latency_ms",
        "retrieve_latency_ms",
        "expand_latency_ms",
        "coordinate_latency_ms",
        "serialize_latency_ms",
    )
    _BACKEND_USAGE_FIELDS = (
        "active_memories",
        "provisional_memories",
        "suspended_memories",
        "accepted_version_updates",
    )

    def __init__(self, benchmark: str, planned_tasks: int, started_at: float):
        self.benchmark = benchmark
        self.planned_tasks = int(planned_tasks)
        self.started_at = float(started_at)
        self.completed_tasks = 0
        self.score_sum = 0.0
        self.won_count = 0
        self.task_elapsed_seconds = 0.0
        self.retrieval_calls = 0
        self.retrieval_totals = {field: 0.0 for field in self._RETRIEVAL_FIELDS}
        self.update_totals = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
        }
        self.replay_runs = 0
        self.backend_usage = {}
        self.latest_task_id = ""
        self.latest_task_index = -1
        self.latest_score = 0.0
        self.latest_won = False
        self.latest_task_elapsed_seconds = 0.0

    def update(
        self,
        *,
        task_id: str,
        task_index: int,
        score: float,
        won: bool,
        task_elapsed_seconds: float,
        retrieval_usage: list[dict],
        update_usage: dict,
    ) -> None:
        self.completed_tasks += 1
        self.latest_task_id = str(task_id)
        self.latest_task_index = int(task_index)
        self.latest_score = float(score)
        self.latest_won = bool(won)
        self.latest_task_elapsed_seconds = float(task_elapsed_seconds)
        self.score_sum += self.latest_score
        self.won_count += int(self.latest_won)
        self.task_elapsed_seconds += self.latest_task_elapsed_seconds

        for record in retrieval_usage or []:
            self.retrieval_calls += 1
            for field in self._RETRIEVAL_FIELDS:
                self.retrieval_totals[field] += float(record.get(field, 0) or 0)

        update_usage = update_usage or {}
        self.update_totals["llm_calls"] += int(update_usage.get("llm_calls", 0) or 0)
        self.update_totals["prompt_tokens"] += int(
            update_usage.get("prompt_tokens", 0) or 0
        )
        self.update_totals["completion_tokens"] += int(
            update_usage.get("completion_tokens", 0) or 0
        )
        self.update_totals["latency_ms"] += float(
            update_usage.get("latency_ms", 0) or 0
        )
        self.replay_runs += _replay_runs_from_update_usage(update_usage)
        for field in self._BACKEND_USAGE_FIELDS:
            if update_usage.get(field) is not None:
                self.backend_usage[field] = int(update_usage[field])

    @staticmethod
    def _average(total: float, count: int) -> float:
        return total / count if count else 0.0

    def memory_cost_stats(self, api_stats: dict) -> dict:
        tasks = self.completed_tasks
        retrievals = self.retrieval_calls
        r = self.retrieval_totals
        u = self.update_totals
        total_prompt = int(api_stats.get("total_prompt_tokens", 0) or 0)
        total_completion = int(api_stats.get("total_completion_tokens", 0) or 0)
        total_calls = int(api_stats.get("total_calls", 0) or 0)
        per_task = lambda value: self._average(value, tasks)
        per_retrieval = lambda value: self._average(value, retrievals)
        return {
            "token_count_note": "injected_tokens_est is a character-based estimate (len/4); total_prompt_tokens uses provider-reported API usage.",
            "injected_card_count_note": "Injected card totals count injection occurrences across retrieval calls; repeated injection of the same memory is counted again.",
            "retrieval_calls": retrievals,
            "avg_retrieval_calls_per_task": per_task(retrievals),
            "total_retrieved_cards": r["retrieved_cards"],
            "avg_retrieved_cards_per_task": per_task(r["retrieved_cards"]),
            "avg_retrieved_cards_per_retrieval": per_retrieval(r["retrieved_cards"]),
            "total_expanded_cards": r["expanded_cards"],
            "avg_expanded_cards_per_task": per_task(r["expanded_cards"]),
            "avg_expanded_cards_per_retrieval": per_retrieval(r["expanded_cards"]),
            "total_coordinated_cards": r["coordinated_cards"],
            "avg_coordinated_cards_per_task": per_task(r["coordinated_cards"]),
            "avg_coordinated_cards_per_retrieval": per_retrieval(
                r["coordinated_cards"]
            ),
            "total_injected_cards": r["injected_cards"],
            "total_injected_card_occurrences": r["injected_cards"],
            "avg_injected_cards_per_task": per_task(r["injected_cards"]),
            "avg_injected_card_occurrences_per_task": per_task(r["injected_cards"]),
            "avg_injected_cards_per_retrieval": per_retrieval(r["injected_cards"]),
            "avg_injected_card_occurrences_per_retrieval": per_retrieval(
                r["injected_cards"]
            ),
            "total_injected_tokens_est": r["injected_tokens_est"],
            "avg_injected_tokens_per_task_est": per_task(r["injected_tokens_est"]),
            "avg_injected_tokens_per_retrieval_est": per_retrieval(
                r["injected_tokens_est"]
            ),
            "total_skipped_over_budget_cards": r["skipped_over_budget_cards"],
            "avg_skipped_over_budget_cards_per_task": per_task(
                r["skipped_over_budget_cards"]
            ),
            "total_compacted_cards": r["compacted_cards"],
            "avg_memory_controller_latency_ms_per_task": per_task(r["latency_ms"]),
            "avg_memory_controller_latency_ms_per_retrieval": per_retrieval(
                r["latency_ms"]
            ),
            "avg_retrieve_latency_ms_per_retrieval": per_retrieval(
                r["retrieve_latency_ms"]
            ),
            "avg_expand_latency_ms_per_retrieval": per_retrieval(
                r["expand_latency_ms"]
            ),
            "avg_coordinate_latency_ms_per_retrieval": per_retrieval(
                r["coordinate_latency_ms"]
            ),
            "avg_serialize_latency_ms_per_retrieval": per_retrieval(
                r["serialize_latency_ms"]
            ),
            "memory_update_llm_calls": u["llm_calls"],
            "avg_memory_update_llm_calls_per_task": per_task(u["llm_calls"]),
            "memory_update_prompt_tokens": u["prompt_tokens"],
            "avg_memory_update_prompt_tokens_per_task": per_task(u["prompt_tokens"]),
            "memory_update_completion_tokens": u["completion_tokens"],
            "avg_memory_update_completion_tokens_per_task": per_task(
                u["completion_tokens"]
            ),
            "avg_memory_update_latency_ms_per_task": per_task(u["latency_ms"]),
            "total_prompt_tokens": total_prompt,
            "avg_total_prompt_tokens_per_task": per_task(total_prompt),
            "total_completion_tokens": total_completion,
            "avg_total_completion_tokens_per_task": per_task(total_completion),
            "total_api_calls": total_calls,
            "avg_api_calls_per_task": per_task(total_calls),
            "non_update_prompt_tokens": max(total_prompt - u["prompt_tokens"], 0),
            "avg_non_update_prompt_tokens_per_task": per_task(
                max(total_prompt - u["prompt_tokens"], 0)
            ),
        }

    def snapshot(
        self, *, api_stats: dict, memory_cards: int, task_results_file: str
    ) -> dict:
        elapsed = max(time.time() - self.started_at, 0.0)
        average_score = self._average(self.score_sum, self.completed_tasks)
        payload = {
            "completed_tasks": self.completed_tasks,
            "planned_tasks": self.planned_tasks,
            "latest_task_id": self.latest_task_id,
            "latest_task_index": self.latest_task_index,
            "latest_score": self.latest_score,
            "score_sum": self.score_sum,
            "average_score": average_score,
            "pass_rate": average_score,
            "elapsed_seconds": round(elapsed, 3),
            "average_task_seconds": self._average(
                self.task_elapsed_seconds, self.completed_tasks
            ),
            "latest_task_elapsed_seconds": self.latest_task_elapsed_seconds,
            "memory_cards": int(memory_cards),
            "api_stats": api_stats,
            "memory_cost_stats": self.memory_cost_stats(api_stats),
            "backend_usage": dict(self.backend_usage),
            "replay_runs": self.replay_runs,
            "task_results_file": os.path.basename(task_results_file),
        }
        if self.benchmark == "pddl":
            payload.update(
                {
                    "latest_won": self.latest_won,
                    "won_count": self.won_count,
                    "win_rate": self._average(self.won_count, self.completed_tasks),
                }
            )
        return payload


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _log_incremental_progress(benchmark: str, metrics: dict) -> None:
    progress = (
        f"  [{benchmark}][PROGRESS] task={metrics['completed_tasks']}/{metrics['planned_tasks']} "
        f"latest={metrics['latest_score']:.4f} avg={metrics['average_score']:.4f}"
    )
    if benchmark == "pddl":
        progress += f" won={metrics['won_count']} win_rate={metrics['win_rate']:.4f}"
    progress += (
        f" elapsed={_format_duration(metrics['elapsed_seconds'])} "
        f"avg_task={metrics['average_task_seconds']:.1f}s"
    )
    logger.info(progress)

    api = metrics["api_stats"]
    cost = metrics["memory_cost_stats"]
    logger.info(
        "  [%s][USAGE] api_calls=%d prompt_tokens=%d completion_tokens=%d "
        "memory_cards=%d retrievals=%d injected_occurrences=%.0f update_calls=%d replay_runs=%d",
        benchmark,
        int(api.get("total_calls", 0) or 0),
        int(api.get("total_prompt_tokens", 0) or 0),
        int(api.get("total_completion_tokens", 0) or 0),
        int(metrics.get("memory_cards", 0) or 0),
        int(cost.get("retrieval_calls", 0) or 0),
        float(
            cost.get(
                "total_injected_card_occurrences", cost.get("total_injected_cards", 0)
            )
            or 0
        ),
        int(cost.get("memory_update_llm_calls", 0) or 0),
        int(metrics.get("replay_runs", 0) or 0),
    )


# ============================================================
# Lazy imports
# ============================================================
def _import_discernmem_centralized_memory():
    from discernmem.centralized_adapter import (
        DiscernMemCentralizedMemory,
    )

    return DiscernMemCentralizedMemory


def _import_discernmem_replay():
    from discernmem.replay import ReplayExecution

    return ReplayExecution


def _load_task_positions_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("indices") or data.get("task_indices") or data.get("positions")
    if not isinstance(data, list):
        raise ValueError(
            "--task_indices_file must contain a JSON list or an object with indices"
        )
    positions = []
    for value in data:
        idx = int(value)
        if idx < 0:
            raise ValueError(f"Negative task index in --task_indices_file: {idx}")
        positions.append(idx)
    return positions


def _select_task_positions(tasks, positions: list[int]):
    if not positions:
        return tasks.select([]) if hasattr(tasks, "select") else []
    max_pos = len(tasks) - 1
    out_of_range = [idx for idx in positions if idx > max_pos]
    if out_of_range:
        raise ValueError(
            f"--task_indices_file contains positions outside loaded task range "
            f"0..{max_pos}: {out_of_range[:5]}"
        )
    if hasattr(tasks, "select"):
        return tasks.select(positions)
    return [tasks[idx] for idx in positions]


def _pddl_task_game_name(task):
    additional = task.get("additional_info") or {}
    return task.get("game_name") or additional.get("subtask") or "unknown"


def _pddl_task_problem_index(task, default):
    if task.get("problem_index") is not None:
        return task.get("problem_index")
    additional = task.get("additional_info") or {}
    value = additional.get("problem_index")
    return default if value is None else value


def _import_mas_class(mas_type):
    if mas_type == "camel":
        from mas.camel.camel_main import CamelMemoryMAS

        return CamelMemoryMAS
    elif mas_type == "macnet":
        from mas.macnet.macnet_main import MacNetMemoryMAS

        return MacNetMemoryMAS
    elif mas_type == "autogen":
        from mas.autogen.autogen_main import AutoGenMemoryMAS

        return AutoGenMemoryMAS
    else:
        raise ValueError(f"Unsupported MAS type: {mas_type}")


# ============================================================
# Dataset loading.
# ============================================================
def _split_kodcode_dataset(split_seed: int = 42):
    from data.kodcode.builder import load_kodcode_splits

    return load_kodcode_splits(split_seed=split_seed)


def load_benchmark(
    name: str,
    num_tasks: int,
    start_from: int,
    data_split: str = "test",
    popqa_sampling: str = "ordered",
    popqa_sample_seed: int = 42,
):
    """Load benchmark datasets and return (tasks, dataset_info)."""
    if name == "kodcode":
        split_seed = 42
        splits = _split_kodcode_dataset(split_seed=split_seed)
        if data_split not in splits:
            raise ValueError(f"KodCode only supports data_split in {list(splits)}")
        ds = splits[data_split]
        end = min(start_from + num_tasks, len(ds))
        tasks = ds.select(range(start_from, end))
        return tasks, {
            "type": "static",
            "total": len(ds),
            "split": data_split,
            "split_seed": split_seed,
        }

    elif name == "triviaqa":
        from datasets import load_dataset

        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
        if data_split == "train":
            selected = ds["train"].select(range(5000))
        elif data_split == "valid":
            selected = ds["validation"].select(range(1000))
        elif data_split == "test":
            selected = ds["validation"].select(range(1000, len(ds["validation"])))
        else:
            raise ValueError(
                "TriviaQA only supports data_split in ['train', 'valid', 'test']"
            )
        end = min(start_from + num_tasks, len(selected))
        tasks = selected.select(range(start_from, end))
        return tasks, {"type": "dynamic", "total": len(selected), "split": data_split}

    elif name == "popqa":
        from datasets import load_dataset
        from data.popqa.sampling import select_popqa_tasks

        ds = load_dataset("akariasai/PopQA")
        if data_split == "valid":
            selected = ds["test"].select(range(6000, 7000))
        elif data_split == "test":
            selected = ds["test"].select(range(7000, len(ds["test"])))
        else:
            raise ValueError("PopQA only supports data_split in ['valid', 'test']")
        tasks, _, sampling_info = select_popqa_tasks(
            selected,
            num_tasks=num_tasks,
            start_from=start_from,
            sampling=popqa_sampling,
            seed=popqa_sample_seed,
        )
        return tasks, {
            "type": "dynamic",
            "total": len(selected),
            "split": data_split,
            **sampling_info,
        }

    elif name == "pddl":
        data_path = os.path.join(PROJECT_ROOT, "data", "pddl", "test.jsonl")
        with open(data_path) as f:
            all_tasks = [json.loads(line) for line in f]
        per_game_indices = {}
        for task in all_tasks:
            additional = task.setdefault("additional_info", {})
            game = additional.get("subtask") or task.get("game_name") or "unknown"
            if additional.get("problem_index") is None:
                additional["problem_index"] = per_game_indices.get(game, 0)
            per_game_indices[game] = int(additional["problem_index"]) + 1
            task["game_name"] = game
            task["problem_index"] = additional["problem_index"]
        if data_split != "test":
            raise ValueError("PDDL only supports data_split='test'")
        end = min(start_from + num_tasks, len(all_tasks))
        tasks = all_tasks[start_from:end]
        return tasks, {"type": "dynamic", "total": len(all_tasks), "split": data_split}

    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ============================================================
# KodCode evaluation using KodCodeEnv.
# ============================================================
def eval_kodcode_task(
    code,
    test_code,
    test_info,
    env_mode="pytest",
    timeout_seconds=None,
    cuda_visible_devices=None,
    return_detail=False,
):
    from data.kodcode.env import KodCodeEnv

    env = KodCodeEnv(
        {
            "kodcode_env_mode": env_mode,
            "kodcode_test_timeout_seconds": timeout_seconds,
            "kodcode_cuda_visible_devices": cuda_visible_devices,
        }
    )
    env.set_env(
        {
            "prompt": "",
            "test": test_code,
            "test_info": test_info,
        }
    )

    try:
        _, score, _ = env.step(code or "")
    except Exception as e:
        logger.warning(f"KodCode evaluation failed: {e}")
        detail = {
            "summary": f"Error: {e}",
            "score": 0.0,
            "full_feedback": f"Error: {e}",
            "test_passed": False,
            "test_results": False,
            "kodcode_env_mode": env_mode,
            "executor_detail": {"failure_category": "executor_internal_error"},
        }
        return (
            (0.0, detail["summary"], detail)
            if return_detail
            else (0.0, detail["summary"])
        )

    detail = env.feedback_detail or {}
    summary = detail.get("summary")
    if summary:
        return (score, summary, detail) if return_detail else (score, summary)
    if score >= 1.0:
        summary = "All tests passed."
        return (score, summary, detail) if return_detail else (score, summary)
    if score > 0:
        summary = f"Partial pass: {score:.1%}"
        return (score, summary, detail) if return_detail else (score, summary)
    summary = "All tests failed."
    return (score, summary, detail) if return_detail else (score, summary)


# ============================================================
# MAS trajectory conversion
# ============================================================
def compose_prompt_record(
    system_prompt: str, user_prompt: str, memory_context: str = ""
) -> str:
    user_parts = []
    if memory_context:
        user_parts.append(memory_context)
    if user_prompt:
        user_parts.append(user_prompt)
    full_user = "\n\n".join(part for part in user_parts if part)
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    if full_user:
        prompt_parts.append(full_user)
    return "\n\n".join(prompt_parts)


def _format_message_node_input(msg, fallback_user_prompt: str = "") -> str:
    if msg is None:
        return fallback_user_prompt
    prompt_parts = []
    formatted_system = getattr(msg, "formatted_system_prompt", None) or ""
    formatted_user = getattr(msg, "formatted_user_prompt", None) or ""
    if formatted_system:
        prompt_parts.append(formatted_system)
    if formatted_user:
        prompt_parts.append(formatted_user)
    if prompt_parts:
        return "\n\n".join(prompt_parts)
    return fallback_user_prompt


def message_graph_to_trajectory(
    question, message_graph, outcome, feedback, system_prompt=""
):
    steps = []
    if message_graph.mas_message_graph is not None:
        import networkx as nx

        try:
            topo_order = list(nx.topological_sort(message_graph.mas_message_graph))
        except Exception:
            topo_order = list(message_graph.mas_message_graph.nodes())
        for idx, node_id in enumerate(topo_order, 1):
            node_data = message_graph.mas_message_graph.nodes[node_id]
            msg = node_data.get("message")
            extra_fields = (
                (
                    (getattr(msg, "user_prompt_fields", None) or {}).get("extra_fields")
                    or {}
                )
                if msg
                else {}
            )
            steps.append(
                {
                    "step_index": idx,
                    "agent": str(node_id),
                    "input": _format_message_node_input(
                        msg, compose_prompt_record(system_prompt, question)
                    ),
                    "output": msg.response if msg else "",
                    "tool_calls": "",
                    "feedback": feedback if idx == len(topo_order) else "",
                    "turn": 1,
                    "upstream_roles": [
                        str(role)
                        for role in message_graph.mas_message_graph.predecessors(
                            node_id
                        )
                    ],
                    "injected_memories": list(
                        extra_fields.get("injected_memories") or []
                    ),
                    "memory_retrieval": list(
                        extra_fields.get("retrieval_details") or []
                    ),
                }
            )
    else:
        steps.append(
            {
                "step_index": 1,
                "agent": "mas",
                "input": compose_prompt_record(system_prompt, question),
                "output": message_graph.action or "",
                "tool_calls": "",
                "feedback": feedback,
            }
        )
    return {
        "task_description": question,
        "outcome": outcome,
        "evaluator_feedback": feedback,
        "steps": steps,
    }


# ============================================================
# Main evaluation logic
# ============================================================
def get_benchmark_config(benchmark):
    """Return the system prompt and evaluator for a benchmark."""
    configs = {
        "kodcode": {
            "system_prompt": "You are an expert Python programmer. Write a correct Python function. Output ONLY Python code.",
            "task_domain": "kodcode",
        },
        "triviaqa": {
            "system_prompt": (
                "Answer the given question. You must conduct reasoning inside <think> and </think>. "
                "If you need more info, use <search> query </search>. "
                "Provide your final answer inside <answer> and </answer>."
            ),
            "task_domain": "triviaqa",
        },
        "popqa": {
            "system_prompt": (
                "Answer the given question. You must conduct reasoning inside <think> and </think>. "
                "If you need more info, use <search> query </search>. "
                "Provide your final answer inside <answer> and </answer>."
            ),
            "task_domain": "popqa",
        },
        "pddl": {
            "system_prompt": "You are a planning assistant. Given a planning problem, provide a step-by-step plan to achieve the goal.",
            "task_domain": "pddl",
        },
    }
    return configs[benchmark]


def get_task_question(benchmark, task):
    """Extract the question text from a task record."""
    if benchmark == "kodcode":
        return task["question"].strip()
    elif benchmark in ("triviaqa", "popqa"):
        return task["question"].strip()
    elif benchmark == "pddl":
        return task.get("task", "") or json.dumps(task)
    return str(task)


def get_task_answer(benchmark, task):
    """Extract the gold answer from a task record."""
    if benchmark == "kodcode":
        return {"test": task["test"].strip(), "test_info": task["test_info"]}
    elif benchmark == "triviaqa":
        return task["answer"]["normalized_aliases"]
    elif benchmark == "popqa":
        import ast

        return ast.literal_eval(task["possible_answers"])
    elif benchmark == "pddl":
        return task
    return None


def run_single_benchmark(benchmark, args):
    """Run evaluation for a single benchmark."""
    from common.qa_config import QAConfig
    from common.config_loader import load_benchmark_config

    memory_backend = args.memory_backend
    memory_label = "DiscernMem"
    bench_cfg = get_benchmark_config(benchmark)
    is_dynamic_benchmark = benchmark in {"triviaqa", "popqa", "pddl"}

    # ---- Load config file ----
    if hasattr(args, "config") and args.config:
        config_path = args.config
    else:
        config_path = None

    bench_config = load_benchmark_config(benchmark, config_path)
    logger.info(
        f"[{benchmark}] Loaded config: max_new_tokens={bench_config.generation.max_new_tokens}, "
        f"temperature={bench_config.generation.temperature}, seed={bench_config.seed}"
    )

    # Use the config model_name when the CLI does not specify one.
    model_name = (
        args.model_name if args.model_name else bench_config.model.llm_name_or_path
    )

    results_dir = (
        args.storage_dir
        if getattr(args, "run_id", None)
        else os.path.join(args.storage_dir, benchmark)
    )
    env_path = os.path.join(PROJECT_ROOT, ".env")

    # Initialize memory and MAS.
    config = QAConfig.from_env(env_path if os.path.exists(env_path) else None)
    dynamic_max_turns = (
        args.max_turns
        if getattr(args, "max_turns", None)
        else bench_config.interaction.max_turns
    )
    dynamic_timeout_seconds = (
        args.timeout_seconds
        if getattr(args, "timeout_seconds", None) is not None
        else getattr(bench_config.interaction, "timeout_seconds", None)
    )
    pddl_action_guard = getattr(args, "pddl_action_guard", "strict")
    max_obs_length = getattr(bench_config.interaction, "max_obs_length", None)

    dynamic_env_kwargs = {}
    run_dynamic_mas = None
    dynamic_env_cls = None
    if benchmark == "triviaqa":
        from data.triviaqa.env import TriviaQAEnv
        from data.triviaqa.runner import run_multiturn_mas as run_triviaqa_mas

        dynamic_env_cls = TriviaQAEnv
        run_dynamic_mas = run_triviaqa_mas
        dynamic_env_kwargs = config.factual_qa_retriever_kwargs_for_domain("triviaqa")
    elif benchmark == "popqa":
        from data.popqa.env import PopQAEnv
        from data.popqa.runner import run_multiturn_mas as run_popqa_mas

        dynamic_env_cls = PopQAEnv
        run_dynamic_mas = run_popqa_mas
        dynamic_env_kwargs = config.factual_qa_retriever_kwargs_for_domain("popqa")
    elif benchmark == "pddl":
        from data.pddl.env.pddl_env import PDDLEnv, get_pddl_backend_metadata
        from data.pddl.runner import run_multiturn_mas as run_pddl_mas

        dynamic_env_cls = PDDLEnv
        run_dynamic_mas = run_pddl_mas
        dynamic_env_kwargs["pddl_backend_metadata"] = get_pddl_backend_metadata(
            args.pddl_env_mode
        )

    if benchmark in {"triviaqa", "popqa"} and getattr(args, "search_url", None):
        dynamic_env_kwargs["search_url"] = args.search_url

    qa_replay_search_cache = None

    DiscernMemCentralizedMemory = _import_discernmem_centralized_memory()
    discernmem_storage_dir = args.discernmem_storage_dir or os.path.join(
        results_dir, "discernmem_memory"
    )
    centralized_memory = DiscernMemCentralizedMemory(
        storage_dir=discernmem_storage_dir,
        config_path=args.discernmem_config,
        task_domain=benchmark,
        read_only=args.read_only_memory,
        llm_api_base=args.api_base,
        llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        llm_model_name=model_name,
        embedding_api_base=args.discernmem_embedding_api_base,
        embedding_api_key=args.discernmem_embedding_api_key,
        embedding_model_name=args.discernmem_embedding_model,
        llm_timeout=args.discernmem_llm_timeout,
        embedding_timeout=args.discernmem_embedding_timeout,
        replay_workers=args.discernmem_replay_workers,
    )
    logger.info(
        f"[{benchmark}] Using DiscernMem JSON store: {discernmem_storage_dir}; "
        f"config={args.discernmem_config}"
    )
    MASClass = _import_mas_class(args.mas_type)
    mas = MASClass(
        llm_name_or_path=model_name,
        centralized_memory=centralized_memory,
        share_llm=True,
        task_domain=bench_cfg["task_domain"],
        api_base=args.api_base,
        model_name=model_name,
    )
    # Use generation settings from the config file.
    generation_config = {
        "max_new_tokens": bench_config.generation.max_new_tokens,
        "temperature": bench_config.generation.temperature,
        "top_p": 0.95,
        "do_sample": bench_config.generation.temperature > 0,
        "seed": int(bench_config.seed),
    }
    logger.info(
        f"[{benchmark}] {args.mas_type.upper()} MAS initialized ({len(mas.agents_list)} agents)"
    )
    logger.info(f"[{benchmark}] DiscernMem JSON storage: {discernmem_storage_dir}")

    # Load data
    logger.info(f"[{benchmark}] Loading dataset...")
    tasks, ds_info = load_benchmark(
        benchmark,
        args.num_tasks,
        args.start_from,
        args.data_split,
        popqa_sampling=args.popqa_sampling,
        popqa_sample_seed=args.popqa_sample_seed,
    )
    task_positions = _load_task_positions_file(getattr(args, "task_indices_file", None))
    if task_positions is not None:
        tasks = _select_task_positions(tasks, task_positions)
        ds_info["selected_task_positions_file"] = os.path.abspath(
            args.task_indices_file
        )
        ds_info["selected_task_positions"] = task_positions
        task_source_indices = [
            args.start_from + position for position in task_positions
        ]
    else:
        task_source_indices = [
            args.start_from + position for position in range(len(tasks))
        ]
    num_tasks = len(tasks)
    logger.info(
        f"[{benchmark}] {num_tasks} tasks loaded from split={ds_info.get('split', 'n/a')} "
        f"(total available: {ds_info['total']})"
    )
    if benchmark == "pddl":
        for check_i, check_task in enumerate(tasks):
            check_task_idx = task_source_indices[check_i]
            check_game_name = _pddl_task_game_name(check_task)
            check_env_config = {
                "game_name": check_game_name,
                "problem_index": _pddl_task_problem_index(check_task, check_i),
                "goal": check_task.get("goal"),
                "subgoals": check_task.get("subgoals", []),
                "difficulty": check_task.get("difficulty", "unknown"),
                "id": check_task.get("id"),
                "pddl_env_mode": args.pddl_env_mode,
            }
            try:
                check_env = dynamic_env_cls({})
                check_env.set_env(check_env_config)
            except Exception as exc:
                raise RuntimeError(
                    f"PDDL env precheck failed for task_index={check_task_idx}, "
                    f"game={check_game_name}, config={check_env_config}: {exc}"
                ) from exc
        logger.info(f"[{benchmark}] PDDL env precheck passed for {num_tasks} tasks")

    ReplayExecution = _import_discernmem_replay()
    if benchmark in {"triviaqa", "popqa"}:
        from common.utils.qa_search_cache import QASearchCache

        qa_replay_search_cache = QASearchCache()
    task_registry = {
        f"{benchmark}_{args.data_split}_{source_index}": {
            "task": task,
            "position": source_index,
        }
        for source_index, task in zip(task_source_indices, tasks)
    }
    unassigned = sorted(
        task_id
        for task_id in task_registry
        if centralized_memory.config.split_manifest.partition_of(task_id) is None
    )
    if unassigned:
        raise ValueError(
            "DiscernMem frozen split manifest does not assign loaded task IDs: "
            f"{unassigned}. Update --discernmem_config before the run."
        )

    def execute_discernmem_replay(request, replay_memory):
        replay_started = time.time()
        registry_item = task_registry.get(request.task_id)
        if registry_item is None:
            raise KeyError(f"Unknown DiscernMem replay task: {request.task_id}")
        replay_task = copy.deepcopy(registry_item["task"])
        replay_position = int(registry_item["position"])
        replay_mas = MASClass(
            llm_name_or_path=model_name,
            centralized_memory=replay_memory,
            share_llm=True,
            task_domain=bench_cfg["task_domain"],
            api_base=args.api_base,
            model_name=model_name,
        )
        replay_generation_config = dict(generation_config)
        replay_generation_config["seed"] = int(request.seed)
        replay_answer = None
        replay_feedback = ""
        replay_environment_config = {}
        replay_evaluator_config = {}

        logger.debug(
            "[DiscernMem][REPLAY][trigger=%s][task=%s][condition=%s][role=%s][seed=%s] START replay=%s",
            request.trigger_task_id or "-",
            request.task_id,
            request.condition.value,
            request.role or "-",
            request.seed,
            request.replay_id,
        )
        if benchmark == "kodcode":
            replay_question = get_task_question(benchmark, replay_task)
            replay_answer = get_task_answer(benchmark, replay_task)
            replay_test_info = replay_answer["test_info"]
            replay_function_name = ""
            if replay_test_info:
                replay_function_name = str(
                    replay_test_info[0].get("function_name", "")
                ).strip()
            graphs = replay_mas.generate(
                task_domain_instructions=[bench_cfg["system_prompt"]],
                user_inputs=[replay_question],
                generation_config=replay_generation_config,
                function_names=[replay_function_name],
            )
            replay_response = graphs[0].action or ""
            replay_score, replay_feedback = eval_kodcode_task(
                replay_response,
                replay_answer["test"],
                replay_answer["test_info"],
                env_mode=args.kodcode_env_mode,
                timeout_seconds=args.kodcode_test_timeout_seconds,
                cuda_visible_devices=args.kodcode_cuda_visible_devices,
            )
            replay_outcome = (
                "success"
                if replay_score >= 1.0
                else ("partial" if replay_score > 0 else "failure")
            )
            replay_trajectory = message_graph_to_trajectory(
                replay_question,
                graphs[0],
                replay_outcome,
                replay_feedback,
                system_prompt=bench_cfg["system_prompt"],
            )
            replay_environment_config = {
                "system_prompt": bench_cfg["system_prompt"],
                "function_name": replay_function_name,
            }
            replay_evaluator_config = {
                "mode": args.kodcode_env_mode,
                "timeout_seconds": args.kodcode_test_timeout_seconds,
                "cuda_visible_devices": args.kodcode_cuda_visible_devices,
                "feedback": replay_feedback,
            }
        elif benchmark in {"triviaqa", "popqa"}:
            replay_question = get_task_question(benchmark, replay_task)
            replay_answer = get_task_answer(benchmark, replay_task)
            replay_env = dynamic_env_cls(dynamic_env_kwargs)
            replay_system_prompt, replay_env_question = replay_env.set_env(
                {"prompt": replay_question, "answer": replay_answer}
            )
            if qa_replay_search_cache is not None:
                qa_replay_search_cache.wrap_retriever(
                    replay_env.explorer, task_id=request.task_id
                )
            replay_score, replay_response, replay_steps = run_dynamic_mas(
                replay_mas,
                replay_env,
                replay_system_prompt,
                replay_env_question,
                replay_answer,
                replay_generation_config,
                dynamic_max_turns,
                action_resolver=replay_env.explorer.batch_search,
                max_obs_length=max_obs_length,
            )
            replay_outcome = (
                "success"
                if replay_score >= 1.0
                else ("partial" if replay_score > 0 else "failure")
            )
            replay_trajectory = {
                "task_description": replay_question,
                "outcome": replay_outcome,
                "evaluator_feedback": (
                    f"score={replay_score:.4f}; ground_truth={replay_answer}"
                ),
                "final_response": replay_response,
                "steps": replay_steps,
            }
            replay_environment_config = {
                "env_class": dynamic_env_cls.__name__,
                "env_kwargs": dynamic_env_kwargs,
                "task_config": {"prompt": replay_question, "answer": replay_answer},
                "max_turns": dynamic_max_turns,
                "max_obs_length": max_obs_length,
            }
            replay_evaluator_config = {
                "mode": "environment_feedback",
                "final_score": replay_score,
            }
        elif benchmark == "pddl":
            replay_game_name = _pddl_task_game_name(replay_task)
            replay_environment_config = {
                "game_name": replay_game_name,
                "problem_index": _pddl_task_problem_index(
                    replay_task, replay_position
                ),
                "goal": replay_task.get("goal"),
                "subgoals": replay_task.get("subgoals", []),
                "difficulty": replay_task.get("difficulty", "unknown"),
                "id": replay_task.get("id"),
                "pddl_env_mode": args.pddl_env_mode,
            }
            replay_env = dynamic_env_cls({})
            replay_system_prompt, replay_init_prompt = replay_env.set_env(
                replay_environment_config
            )
            replay_description = f"{replay_game_name}: {replay_env._get_goal()}"
            replay_score, replay_won, replay_steps = run_dynamic_mas(
                replay_mas,
                replay_env,
                replay_system_prompt,
                replay_init_prompt,
                replay_generation_config,
                dynamic_max_turns,
                timeout_seconds=(
                    args.discernmem_replay_timeout_seconds
                    if args.discernmem_replay_timeout_seconds > 0
                    else dynamic_timeout_seconds
                ),
                memory_task_description=replay_description,
                execution_label=(
                    f"DISCERNMEM_REPLAY trigger={request.trigger_task_id or '-'} "
                    f"condition={request.condition.value} seed={request.seed} "
                    f"replay={request.replay_id}"
                ),
                execution_task_id=request.task_id,
            )
            replay_outcome = (
                "success"
                if replay_won
                else ("partial" if replay_score > 0 else "failure")
            )
            replay_rounds = max(
                (int(step.get("turn", 0) or 0) for step in replay_steps),
                default=0,
            )
            replay_trajectory = {
                "task_description": replay_description,
                "outcome": replay_outcome,
                "evaluator_feedback": (
                    f"score={replay_score:.4f}; won={replay_won}; "
                    f"interaction_rounds={replay_rounds}; "
                    f"role_steps={len(replay_steps)}"
                ),
                "interaction_rounds": replay_rounds,
                "role_steps": len(replay_steps),
                "steps": replay_steps,
            }
            replay_evaluator_config = {
                "mode": "environment_feedback",
                "final_score": replay_score,
                "won": replay_won,
            }
        else:
            raise ValueError(f"Unsupported DiscernMem replay benchmark: {benchmark}")

        replay_latency = time.time() - replay_started
        replay_trajectory.update(
            {
                "execution_type": "discernmem_replay",
                "replay_id": request.replay_id,
                "trigger_task_id": request.trigger_task_id,
                "validation_task_id": request.task_id,
                "condition": request.condition.value,
                "target_role": request.role or None,
                "seed": request.seed,
            }
        )
        prompt_refs = tuple(
            f"memory_text_sha256:{record['memory_text_sha256']}"
            for record in replay_memory.retrieval_records
        )
        logger.debug(
            "[DiscernMem][REPLAY][task=%s][condition=%s][role=%s][seed=%s] END score=%.4f outcome=%s latency=%.2fs replay=%s",
            request.task_id,
            request.condition.value,
            request.role or "-",
            request.seed,
            float(replay_score),
            replay_outcome,
            replay_latency,
            request.replay_id,
        )
        return ReplayExecution(
            score=float(replay_score),
            outcome=replay_outcome,
            trajectory=replay_trajectory,
            runtime_signature=request.runtime_signature,
            evaluator_metadata={
                "benchmark": benchmark,
                "mas_type": args.mas_type,
                "model_name": model_name,
                "generation_config": replay_generation_config,
                "raw_task": replay_task,
                "ground_truth": replay_answer,
                "environment_config": replay_environment_config,
                "evaluator_config": replay_evaluator_config,
            },
            prompt_audit_refs=prompt_refs,
        )

    trace_replay_process_isolation = benchmark == "pddl"
    centralized_memory.set_replay_runner(
        execute_discernmem_replay,
        process_isolation=trace_replay_process_isolation,
        replay_timeout_seconds=args.discernmem_replay_timeout_seconds,
    )
    logger.info(
        "[%s] DiscernMem replay enabled: registered_tasks=%d workers=%d mode=%s seeds=%s",
        benchmark,
        len(task_registry),
        args.discernmem_replay_workers,
        "process_isolated" if trace_replay_process_isolation else "threaded",
        ",".join(str(value) for value in centralized_memory.config.seeds),
    )

    # Evaluation loop
    results = []
    kodcode_executor_records = []
    injected_card_occurrences_by_task = []
    memory_retrieval_usage_records = []
    memory_update_usage_records = []
    won_count = 0
    t_start = time.time()
    progress_metrics = IncrementalExperimentMetrics(benchmark, num_tasks, t_start)
    task_results_file = os.path.join(results_dir, "task_results.jsonl")
    os.makedirs(results_dir, exist_ok=True)
    with open(task_results_file, "w", encoding="utf-8"):
        pass
    latest_progress_snapshot = None
    from common.stats import stats

    profiler = None
    if _PROFILE:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    for i in range(num_tasks):
        task_started = time.time()
        won_before_task = won_count
        task = tasks[i]
        task_idx = task_source_indices[i]
        question = get_task_question(benchmark, task)
        answer = get_task_answer(benchmark, task)
        task_id = f"{benchmark}_{args.data_split}_{task_idx}"
        centralized_memory.set_current_task_context(
            task_id=task_id,
            task_index=task_idx,
            execution_type="official",
        )
        centralized_memory._last_update_usage = {}

        logger.info(
            "\n[%s][OFFICIAL][task=%s] Task %d (%d/%d): %s...",
            benchmark,
            task_id,
            task_idx,
            i + 1,
            num_tasks,
            question[:80],
        )
        code_function_name = ""
        if benchmark == "kodcode":
            test_info = answer["test_info"]
            if test_info:
                code_function_name = str(test_info[0].get("function_name", "")).strip()

        if benchmark == "kodcode":
            graphs = None
            try:
                graphs = mas.generate(
                    task_domain_instructions=[bench_cfg["system_prompt"]],
                    user_inputs=[question],
                    generation_config=generation_config,
                    function_names=[code_function_name],
                )
                response = graphs[0].action or ""
            except Exception as e:
                logger.error(f"  [MAS] Generation failed: {e}")
                response = ""
                graphs = None

            score, feedback, executor_detail = eval_kodcode_task(
                response,
                answer["test"],
                answer["test_info"],
                env_mode=args.kodcode_env_mode,
                timeout_seconds=args.kodcode_test_timeout_seconds,
                cuda_visible_devices=args.kodcode_cuda_visible_devices,
                return_detail=True,
            )
            results.append(score)
            kodcode_executor_records.append(
                {
                    "task_id": task_id,
                    "task_index": task_idx,
                    "score": score,
                    "executor_detail": executor_detail.get("executor_detail"),
                    "solution_normalizations": executor_detail.get(
                        "solution_normalizations", []
                    ),
                }
            )
            logger.info(
                f"  [OFFICIAL][task={task_id}][{args.mas_type.upper()}+{memory_label}] score={score:.1f} {feedback[:60]}"
            )

            outcome = (
                "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            )
            # A failed generation (graphs is None) is scored but not learned from.
            if graphs is not None:
                traj = message_graph_to_trajectory(
                    question,
                    graphs[0],
                    outcome,
                    feedback,
                    system_prompt=bench_cfg["system_prompt"],
                )
                _record_task_trajectory(
                    centralized_memory,
                    getattr(args, "read_only_memory", False),
                    trajectory=traj,
                    task_id=task_id,
                    task_description=question,
                    outcome=outcome,
                    score=score,
                    task_index=task_idx,
                    metadata=_research_task_metadata(
                        args,
                        benchmark,
                        model_name,
                        generation_config,
                        task,
                        answer,
                        benchmark,
                        environment_config={
                            "system_prompt": bench_cfg["system_prompt"],
                            "function_name": code_function_name,
                        },
                        evaluator_config={
                            "mode": args.kodcode_env_mode,
                            "timeout_seconds": args.kodcode_test_timeout_seconds,
                            "cuda_visible_devices": args.kodcode_cuda_visible_devices,
                            "executor_detail": executor_detail,
                        },
                    ),
                )
            injected_card_occurrences, retrieval_usage, update_usage = (
                _consume_memory_usage(
                    centralized_memory, task_id, task_idx
                )
            )
            injected_card_occurrences_by_task.append(injected_card_occurrences)
            memory_retrieval_usage_records.append(
                {
                    "task_id": task_id,
                    "task_index": task_idx,
                    "retrievals": retrieval_usage,
                }
            )
            memory_update_usage_records.append(update_usage)
        elif benchmark == "triviaqa":
            env_task_config = {"prompt": question, "answer": answer}

            env = dynamic_env_cls(dynamic_env_kwargs)
            system_prompt, env_question = env.set_env(env_task_config)

            score, response, steps = run_dynamic_mas(
                mas,
                env,
                system_prompt,
                env_question,
                answer,
                generation_config,
                dynamic_max_turns,
                action_resolver=env.explorer.batch_search,
                max_obs_length=max_obs_length,
            )

            results.append(score)
            feedback = f"score={score:.1f} (turns={len(steps)}, gt={answer[0] if answer else 'N/A'})"
            logger.info(
                f"  [OFFICIAL][task={task_id}][{args.mas_type.upper()}+{memory_label}] {feedback}"
            )

            outcome = (
                "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            )
            traj_data = {
                "task_description": question,
                "outcome": outcome,
                "evaluator_feedback": feedback,
                "final_response": response,
                "steps": steps,
            }
            _record_task_trajectory(
                centralized_memory,
                getattr(args, "read_only_memory", False),
                trajectory=traj_data,
                task_id=task_id,
                task_description=question,
                outcome=outcome,
                score=score,
                task_index=task_idx,
                metadata=_research_task_metadata(
                    args,
                    benchmark,
                    model_name,
                    generation_config,
                    task,
                    answer,
                    benchmark,
                    environment_config={
                        "env_class": dynamic_env_cls.__name__,
                        "env_kwargs": dynamic_env_kwargs,
                        "task_config": env_task_config,
                        "max_turns": dynamic_max_turns,
                        "max_obs_length": max_obs_length,
                    },
                    evaluator_config={
                        "mode": "environment_feedback",
                        "final_score": score,
                    },
                ),
            )
            injected_card_occurrences, retrieval_usage, update_usage = (
                _consume_memory_usage(
                    centralized_memory, task_id, task_idx
                )
            )
            injected_card_occurrences_by_task.append(injected_card_occurrences)
            memory_retrieval_usage_records.append(
                {
                    "task_id": task_id,
                    "task_index": task_idx,
                    "retrievals": retrieval_usage,
                }
            )
            memory_update_usage_records.append(update_usage)
        elif benchmark == "popqa":
            env_task_config = {
                "prompt": question,
                "answer": answer,
            }

            env = dynamic_env_cls(dynamic_env_kwargs)
            system_prompt, env_question = env.set_env(env_task_config)

            score, response, steps = run_dynamic_mas(
                mas,
                env,
                system_prompt,
                env_question,
                answer,
                generation_config,
                dynamic_max_turns,
                action_resolver=env.explorer.batch_search,
                max_obs_length=max_obs_length,
            )

            results.append(score)
            feedback = f"score={score:.1f} (turns={len(steps)}, gt={answer[0] if answer else 'N/A'})"
            logger.info(
                f"  [OFFICIAL][task={task_id}][{args.mas_type.upper()}+{memory_label}] {feedback}"
            )

            outcome = (
                "success" if score >= 1.0 else ("partial" if score > 0 else "failure")
            )
            traj_data = {
                "task_description": question,
                "outcome": outcome,
                "evaluator_feedback": feedback,
                "final_response": response,
                "steps": steps,
            }
            _record_task_trajectory(
                centralized_memory,
                getattr(args, "read_only_memory", False),
                trajectory=traj_data,
                task_id=task_id,
                task_description=question,
                outcome=outcome,
                score=score,
                task_index=task_idx,
                metadata=_research_task_metadata(
                    args,
                    benchmark,
                    model_name,
                    generation_config,
                    task,
                    answer,
                    benchmark,
                    environment_config={
                        "env_class": dynamic_env_cls.__name__,
                        "env_kwargs": dynamic_env_kwargs,
                        "task_config": env_task_config,
                        "max_turns": dynamic_max_turns,
                        "max_obs_length": max_obs_length,
                    },
                    evaluator_config={
                        "mode": "environment_feedback",
                        "final_score": score,
                    },
                ),
            )
            injected_card_occurrences, retrieval_usage, update_usage = (
                _consume_memory_usage(
                    centralized_memory, task_id, task_idx
                )
            )
            injected_card_occurrences_by_task.append(injected_card_occurrences)
            memory_retrieval_usage_records.append(
                {
                    "task_id": task_id,
                    "task_index": task_idx,
                    "retrievals": retrieval_usage,
                }
            )
            memory_update_usage_records.append(update_usage)
        elif benchmark == "pddl":
            game_name = _pddl_task_game_name(task)
            difficulty = task.get("difficulty", "unknown")
            env_task_config = {
                "game_name": game_name,
                "problem_index": _pddl_task_problem_index(task, i),
                "goal": task.get("goal"),
                "subgoals": task.get("subgoals", []),
                "difficulty": difficulty,
                "id": task.get("id"),
                "pddl_env_mode": args.pddl_env_mode,
            }
            logger.info(f"  [PDDL] game={game_name} difficulty={difficulty}")

            env = dynamic_env_cls({})
            try:
                system_prompt, init_user_prompt = env.set_env(env_task_config)
            except (ValueError, IndexError) as e:
                raise RuntimeError(
                    f"PDDL env setup failed for strict mode {args.pddl_env_mode}: "
                    f"task={task_idx}, game={game_name}, problem_index={env_task_config.get('problem_index')}: {e}"
                ) from e
            task_description = f"{game_name}: {env._get_goal()}"

            score, won, steps = run_dynamic_mas(
                mas,
                env,
                system_prompt,
                init_user_prompt,
                generation_config,
                dynamic_max_turns,
                timeout_seconds=dynamic_timeout_seconds,
                memory_task_description=task_description,
                execution_label="OFFICIAL",
                execution_task_id=task_id,
                pddl_action_guard=pddl_action_guard,
            )

            results.append(score)
            if won:
                won_count += 1
            interaction_rounds = max(
                (int(step.get("turn", 0) or 0) for step in steps), default=0
            )
            feedback = (
                f"score={score:.2f}, won={won}, interaction_rounds={interaction_rounds}, "
                f"role_steps={len(steps)}"
            )
            logger.info(
                f"  [OFFICIAL][task={task_id}][{args.mas_type.upper()}+{memory_label}] {feedback}"
            )

            outcome = "success" if won else ("partial" if score > 0 else "failure")
            traj_data = {
                "task_description": task_description,
                "outcome": outcome,
                "pddl_action_guard": pddl_action_guard,
                "evaluator_feedback": feedback,
                "interaction_rounds": interaction_rounds,
                "role_steps": len(steps),
                "steps": steps,
            }
            _record_task_trajectory(
                centralized_memory,
                getattr(args, "read_only_memory", False),
                trajectory=traj_data,
                task_id=task_id,
                task_description=task_description,
                outcome=outcome,
                score=score,
                task_index=task_idx,
                metadata={
                    **_research_task_metadata(
                        args,
                        benchmark,
                        model_name,
                        generation_config,
                        task,
                        None,
                        game_name,
                        environment_config={
                            **env_task_config,
                            "env_class": dynamic_env_cls.__name__,
                            "max_turns": dynamic_max_turns,
                            "timeout_seconds": dynamic_timeout_seconds,
                        },
                        evaluator_config={
                            "mode": "environment_feedback",
                            "pddl_action_guard": pddl_action_guard,
                            "won": won,
                            "final_score": score,
                        },
                    ),
                    "game_name": game_name,
                },
            )
            injected_card_occurrences, retrieval_usage, update_usage = (
                _consume_memory_usage(
                    centralized_memory, task_id, task_idx
                )
            )
            injected_card_occurrences_by_task.append(injected_card_occurrences)
            memory_retrieval_usage_records.append(
                {
                    "task_id": task_id,
                    "task_index": task_idx,
                    "retrievals": retrieval_usage,
                }
            )
            memory_update_usage_records.append(update_usage)

        # Incremental progress and recovery artifacts.
        latest_score = float(results[-1])
        latest_won = benchmark == "pddl" and won_count > won_before_task
        task_elapsed_seconds = time.time() - task_started
        task_retrieval_record = memory_retrieval_usage_records[-1]
        task_update_record = memory_update_usage_records[-1]
        progress_metrics.update(
            task_id=task_id,
            task_index=task_idx,
            score=latest_score,
            won=latest_won,
            task_elapsed_seconds=task_elapsed_seconds,
            retrieval_usage=task_retrieval_record.get("retrievals", []),
            update_usage=task_update_record,
        )
        _append_task_result(
            results_dir,
            {
                "position": i,
                "task_id": task_id,
                "task_index": task_idx,
                "score": latest_score,
                "won": bool(latest_won) if benchmark == "pddl" else None,
                "task_elapsed_seconds": round(task_elapsed_seconds, 3),
                "retrieval_calls": len(task_retrieval_record.get("retrievals", [])),
                "memory_update_llm_calls": int(
                    task_update_record.get("llm_calls", 0) or 0
                ),
                "replay_runs": _replay_runs_from_update_usage(task_update_record),
                "completed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
        )
        current_memory_cards = (
            int(task_update_record["active_memories"])
            if task_update_record.get("active_memories") is not None
            else _count_active_memory_cards(centralized_memory)
        )
        latest_progress_snapshot = progress_metrics.snapshot(
            api_stats=stats.to_dict(),
            memory_cards=current_memory_cards,
            task_results_file=task_results_file,
        )
        _write_partial_eval_results(
            results_dir,
            benchmark=benchmark,
            mas_type=args.mas_type,
            memory_backend=memory_backend,
            run_id=getattr(args, "run_id", None),
            metrics=latest_progress_snapshot,
        )
        _log_incremental_progress(benchmark, latest_progress_snapshot)

    if profiler is not None:
        profiler.disable()
        import io
        import pstats

        os.makedirs(results_dir, exist_ok=True)
        profile_path = os.path.join(
            results_dir, f"profile_{benchmark}_{args.mas_type}.prof"
        )
        profiler.dump_stats(profile_path)
        logger.info(f"[{benchmark}] cProfile saved to {profile_path}")
        for sort_key in ("cumulative", "tottime"):
            buf = io.StringIO()
            pstats.Stats(profile_path, stream=buf).strip_dirs().sort_stats(
                sort_key
            ).print_stats(30)
            logger.info(
                f"[{benchmark}] ===== cProfile top 30 ({sort_key}) =====\n{buf.getvalue()}"
            )

    elapsed = time.time() - t_start
    pass_rate = sum(results) / len(results) if results else 0
    evaluated_tasks = len(results)

    # Statistics
    memory_cards = _count_active_memory_cards(centralized_memory)
    discernmem_usage_stats = centralized_memory.get_usage_stats()
    discernmem_trajectory_files = centralized_memory.get_saved_trajectory_files()
    api_stats = stats.to_dict()
    memory_cost_stats = _summarize_memory_cost_stats(
        memory_retrieval_usage_records,
        memory_update_usage_records,
        api_stats,
        evaluated_tasks,
    )
    memory_injection_stats = _summarize_memory_injection_stats(
        injected_card_occurrences_by_task,
        memory_retrieval_usage_records,
        evaluated_tasks,
    )
    incremental_cost_stats = progress_metrics.memory_cost_stats(api_stats)
    incremental_mismatches = {}
    for key, incremental_value in incremental_cost_stats.items():
        full_value = memory_cost_stats.get(key)
        if isinstance(incremental_value, (int, float)) and isinstance(
            full_value, (int, float)
        ):
            if abs(float(incremental_value) - float(full_value)) > 1e-9:
                incremental_mismatches[key] = {
                    "incremental": incremental_value,
                    "full": full_value,
                }
        elif incremental_value != full_value:
            incremental_mismatches[key] = {
                "incremental": incremental_value,
                "full": full_value,
            }
    incremental_validation = {
        "matched": not incremental_mismatches
        and abs(progress_metrics.score_sum - sum(results)) <= 1e-9
        and progress_metrics.won_count == won_count,
        "memory_cost_mismatches": incremental_mismatches,
        "score_sum_incremental": progress_metrics.score_sum,
        "score_sum_full": sum(results),
        "won_count_incremental": progress_metrics.won_count,
        "won_count_full": won_count,
    }
    if not incremental_validation["matched"]:
        logger.warning(
            "[%s] Incremental metric validation mismatch: %s",
            benchmark,
            incremental_validation,
        )
    summary = {
        "benchmark": benchmark,
        "mas_type": args.mas_type,
        "memory_backend": memory_backend,
        "run_id": getattr(args, "run_id", None),
        "artifact_leaf_dir": getattr(args, "_run_leaf_dir", None),
        "storage_dir": os.path.abspath(args.storage_dir),
        "log_file": getattr(args, "_run_log_file", None),
        "run_config_file": getattr(args, "_run_config_file", None),
        "task_results_file": task_results_file,
        "discernmem_storage_dir": discernmem_storage_dir,
        "discernmem_config": os.path.abspath(args.discernmem_config),
        "discernmem_replay_workers": args.discernmem_replay_workers,
        "discernmem_replay_timeout_seconds": args.discernmem_replay_timeout_seconds,
        "discernmem_embedding_model": args.discernmem_embedding_model,
        "model_name": model_name,
        "config_model": bench_config.model.llm_name_or_path,
        "max_new_tokens": bench_config.generation.max_new_tokens,
        "temperature": bench_config.generation.temperature,
        "data_split": args.data_split,
        "num_tasks": evaluated_tasks,
        "start_from": args.start_from,
        "task_indices_file": (
            os.path.abspath(args.task_indices_file) if args.task_indices_file else None
        ),
        "pass_rate": pass_rate,
        "scores": results,
        "elapsed_seconds": round(elapsed, 1),
        "memory_cards": memory_cards,
        "dataset_total": ds_info["total"],
        "api_stats": api_stats,
        "discernmem_usage_stats": discernmem_usage_stats,
        "discernmem_trajectory_files": discernmem_trajectory_files,
        "discernmem_qa_replay_search_cache_stats": (
            qa_replay_search_cache.stats()
            if qa_replay_search_cache is not None
            else None
        ),
        "memory_injection_stats": memory_injection_stats,
        "memory_cost_stats": memory_cost_stats,
        "incremental_metrics_validation": incremental_validation,
    }
    if "split_seed" in ds_info:
        summary["split_seed"] = ds_info["split_seed"]
    if "sampling" in ds_info:
        summary["sampling"] = ds_info["sampling"]
    if "sample_seed" in ds_info:
        summary["sample_seed"] = ds_info["sample_seed"]
    if "selected_indices" in ds_info:
        summary["selected_indices"] = ds_info["selected_indices"]
    if "selected_task_positions" in ds_info:
        summary["selected_task_positions"] = ds_info["selected_task_positions"]
    if is_dynamic_benchmark:
        summary["max_turns"] = dynamic_max_turns
        if dynamic_timeout_seconds is not None:
            summary["timeout_seconds"] = dynamic_timeout_seconds
    if benchmark == "pddl":
        summary["pddl_env_mode"] = args.pddl_env_mode
        summary["pddl_action_guard"] = pddl_action_guard
        summary["pddl_backend_metadata"] = dynamic_env_kwargs.get(
            "pddl_backend_metadata", {}
        )
        summary["avg_score"] = pass_rate
        summary["win_rate"] = won_count / evaluated_tasks if evaluated_tasks else 0
        summary["won"] = won_count
    if benchmark == "kodcode":
        from common.utils.code_utils import get_kodcode_executor_metadata

        summary["kodcode_env_mode"] = args.kodcode_env_mode
        summary["kodcode_executor_metadata"] = get_kodcode_executor_metadata(
            args.kodcode_env_mode,
            timeout=args.kodcode_test_timeout_seconds,
            cuda_visible_devices=args.kodcode_cuda_visible_devices,
        )
        summary["kodcode_executor_records"] = kodcode_executor_records

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    _write_partial_eval_results(
        results_dir,
        benchmark=benchmark,
        mas_type=args.mas_type,
        memory_backend=memory_backend,
        run_id=getattr(args, "run_id", None),
        metrics=progress_metrics.snapshot(
            api_stats=api_stats,
            memory_cards=memory_cards,
            task_results_file=task_results_file,
        ),
        complete=True,
    )

    logger.info(f"\n{'='*60}")
    if benchmark == "pddl":
        logger.info(
            f"[{benchmark}] RESULTS: avg_score={pass_rate:.2%} win_rate={summary['win_rate']:.2%} "
            f"({won_count}/{evaluated_tasks}) time={elapsed:.1f}s cards={memory_cards}"
        )
    else:
        logger.info(
            f"[{benchmark}] RESULTS: pass_rate={pass_rate:.2%} ({int(sum(results))}/{evaluated_tasks}) "
            f"time={elapsed:.1f}s cards={memory_cards}"
        )
    logger.info(
        f"[{benchmark}] COST: injected_tokens/task~{memory_cost_stats['avg_injected_tokens_per_task_est']:.1f} "
        f"retrieved/expanded/coordinated card_occurrences/task="
        f"{memory_cost_stats['avg_retrieved_cards_per_task']:.2f}/"
        f"{memory_cost_stats['avg_expanded_cards_per_task']:.2f}/"
        f"{memory_cost_stats['avg_coordinated_cards_per_task']:.2f} "
        f"controller_ms/retrieval={memory_cost_stats['avg_memory_controller_latency_ms_per_retrieval']:.1f} "
        f"update_llm_calls/task={memory_cost_stats['avg_memory_update_llm_calls_per_task']:.2f} "
        f"prompt_tokens/task={memory_cost_stats['avg_total_prompt_tokens_per_task']:.1f}"
    )
    logger.info(f"[{benchmark}] Saved to {results_file}")

    return summary


# ============================================================
# Main entry point
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiscernMem benchmark runner")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config directory or file path (default: configs/benchmarks/*.yaml).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="kodcode",
        choices=["kodcode", "triviaqa", "popqa", "pddl", "all"],
        help="Benchmark to run (default: kodcode)",
    )
    parser.add_argument(
        "--mas_type",
        type=str,
        default="autogen",
        choices=["autogen", "camel", "macnet"],
        help="MAS type (default: autogen)",
    )
    parser.add_argument("--api_base", type=str, default="http://localhost:8100/v1")
    parser.add_argument(
        "--search_url",
        type=str,
        default=None,
        help="Optional override for QA retrieval endpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (default: value from config).",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=20,
        help="Number of tasks per benchmark (default: 20)",
    )
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to use when supported (default: test)",
    )
    parser.add_argument(
        "--popqa_sampling",
        type=str,
        default="ordered",
        choices=["ordered", "stratified_by_prop"],
        help="PopQA task selection strategy (default: ordered)",
    )
    parser.add_argument(
        "--popqa_sample_seed",
        type=int,
        default=42,
        help="Seed for PopQA stratified sampling",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier for isolated artifacts under --results_root/<run_id>/<benchmark>/<mas_type>/<memory_backend>",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="./results",
        help="Root directory for --run_id artifacts. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--overwrite_run",
        action="store_true",
        help="Replace an existing non-empty leaf artifact directory for this run_id/benchmark/mas/memory backend",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="./results",
        help="Benchmark result output directory. Overridden by --run_id.",
    )
    parser.add_argument(
        "--task_indices_file",
        type=str,
        default=None,
        help="JSON list of task positions to evaluate after loading the requested split/sample",
    )
    parser.add_argument(
        "--memory_backend",
        type=str,
        default="discernmem",
        choices=["discernmem"],
        help="Memory backend (only discernmem is provided)",
    )
    parser.add_argument(
        "--discernmem_storage_dir",
        type=str,
        default=None,
        help="Independent DiscernMem JSON storage directory. With --run_id: <run leaf>/memory/discernmem",
    )
    parser.add_argument(
        "--discernmem_config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "discernmem", "default.yaml"),
        help="Frozen DiscernMem YAML configuration including learning/H/final task IDs",
    )
    parser.add_argument(
        "--discernmem_replay_workers",
        type=int,
        default=8,
        help="Maximum concurrent DiscernMem isolated replay runs",
    )
    parser.add_argument(
        "--discernmem_replay_timeout_seconds",
        type=float,
        default=600.0,
        help="Per-replay wall-clock timeout in seconds (0 disables watchdog; default: 600)",
    )
    parser.add_argument(
        "--discernmem_llm_timeout",
        type=float,
        default=180.0,
        help="Timeout for DiscernMem extraction, identity, validation, and repair LLM calls",
    )
    parser.add_argument(
        "--discernmem_embedding_api_base",
        type=str,
        default=None,
        help="OpenAI-compatible DiscernMem embedding API base",
    )
    parser.add_argument(
        "--discernmem_embedding_api_key",
        type=str,
        default=None,
        help="DiscernMem embedding API key; defaults to DISCERNMEM_EMBED_API_KEY or EMPTY",
    )
    parser.add_argument(
        "--discernmem_embedding_model",
        type=str,
        default=None,
        help="DiscernMem embedding model; defaults to DISCERNMEM_EMBED_MODEL",
    )
    parser.add_argument(
        "--discernmem_embedding_timeout",
        type=float,
        default=60.0,
        help="Timeout for DiscernMem embedding calls",
    )
    parser.add_argument(
        "--read_only_memory",
        action="store_true",
        help="Read from existing memory storage without writing new cards",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Optional override for dynamic benchmark max turns",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=None,
        help="Optional per-task wall-clock timeout for dynamic benchmarks",
    )
    parser.add_argument(
        "--pddl_env_mode",
        type=str,
        default="agentboard",
        choices=["agentboard"],
        help="PDDL backend: the AgentBoard PDDLGym environments (60 problems).",
    )
    parser.add_argument(
        "--pddl_action_guard",
        type=str,
        default="strict",
        choices=["strict"],
        help="PDDL action policy: canonicalize exact legal equivalents; the environment validates everything else.",
    )
    parser.add_argument(
        "--kodcode_env_mode",
        type=str,
        default="pytest",
        choices=["pytest"],
        help="KodCode evaluator: isolated pytest subprocess per task.",
    )
    parser.add_argument(
        "--kodcode_test_timeout_seconds",
        type=int,
        default=None,
        help="Per-task KodCode pytest timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--kodcode_cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value applied only to KodCode pytest subprocesses.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    benchmarks = (
        ["kodcode", "triviaqa", "popqa", "pddl"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    all_summaries = []
    has_errors = False
    for bench in benchmarks:
        bench_args = copy.copy(args)
        _setup_run_file_logging(None)
        try:
            if bench_args.run_id:
                _prepare_run_artifacts(bench_args, bench)
                _setup_run_file_logging(bench_args._run_log_file)
                _write_run_config(bench_args, bench)
                logger.info("Run artifacts: %s", bench_args._run_leaf_dir)
                logger.info("  logs:   %s", bench_args._artifact_paths["logs_dir"])
                logger.info("  memory: %s", bench_args._artifact_paths["memory_dir"])
                logger.info("  results:%s", bench_args._artifact_paths["results_dir"])
                logger.info("Run log: %s", bench_args._run_log_file)
            else:
                _setup_run_file_logging(None)

            logger.info(f"\n{'#'*60}")
            logger.info(
                f"# Benchmark: {bench.upper()} | MAS: {bench_args.mas_type.upper()} | Memory: {bench_args.memory_backend}"
            )
            logger.info(f"{'#'*60}")

            # Reset stats for each benchmark.
            from common.stats import stats

            stats.reset()

            summary = run_single_benchmark(bench, bench_args)
            all_summaries.append(summary)
        except Exception as e:
            has_errors = True
            logger.error(f"[{bench}] Failed: {e}", exc_info=True)
            error_summary = {
                "benchmark": bench,
                "mas_type": bench_args.mas_type,
                "memory_backend": bench_args.memory_backend,
                "run_id": getattr(bench_args, "run_id", None),
                "artifact_leaf_dir": getattr(bench_args, "_run_leaf_dir", None),
                "log_file": getattr(bench_args, "_run_log_file", None),
                "error": str(e),
            }
            error_dir = getattr(bench_args, "_run_results_dir", None)
            if error_dir:
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(error_dir, "eval_error.json")
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump(error_summary, f, ensure_ascii=False, indent=2)
                error_summary["error_file"] = error_file
            all_summaries.append(error_summary)

    _setup_run_file_logging(None)

    # Summary
    if len(benchmarks) > 1:
        logger.info(f"\n{'#'*60}")
        logger.info("# OVERALL SUMMARY")
        logger.info(f"{'#'*60}\n")
        for s in all_summaries:
            if "error" in s:
                logger.info(f"  {s['benchmark']:12s}  ERROR: {s['error']}")
            else:
                logger.info(
                    f"  {s['benchmark']:12s}  pass_rate={s['pass_rate']:.2%}  "
                    f"tasks={s['num_tasks']}  time={s['elapsed_seconds']}s  "
                    f"cards={s['memory_cards']}"
                )

        if args.run_id:
            run_id = _sanitize_run_segment(args.run_id, "run_id")
            summary_dir = os.path.join(
                _resolve_results_root(args.results_root),
                run_id,
                "summaries",
                _sanitize_run_segment(args.mas_type, "mas_type"),
                _sanitize_run_segment(args.memory_backend, "memory_backend"),
            )
            os.makedirs(summary_dir, exist_ok=True)
            overall_file = os.path.join(summary_dir, "overall_results.json")
        else:
            overall_file = os.path.join(args.storage_dir, "overall_results.json")
        with open(overall_file, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        logger.info(f"\nOverall results saved to {overall_file}")

    if args.run_id and has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
