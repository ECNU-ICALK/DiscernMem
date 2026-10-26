from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Optional

from common.memory_interface import BaseCentralizedMemory, Memory

from .ids import stable_id
from .schema import (
    HLabel,
    MetaMemory,
    ReplayCondition,
    ReplayRecord,
    ReplayStatus,
)
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now


def runtime_signature(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayRequest:
    task_id: str
    task_index: int
    seed: int
    condition: ReplayCondition
    config_fingerprint: str
    runtime_signature: str
    role: str = ""
    memory_id: str = ""
    version: Optional[int] = None
    body: str = ""
    meta: MetaMemory = field(default_factory=MetaMemory)
    candidate_id: str = ""
    validation_plan_id: str = ""
    h_label: Optional[HLabel] = None
    is_source_task: bool = False
    trigger_task_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    replay_id: str = ""

    @property
    def memory(self) -> str:
        return self.body

    def __post_init__(self) -> None:
        if self.condition is ReplayCondition.NO_MEMORY:
            if self.memory_id or self.candidate_id or self.body or self.role:
                raise ValueError("NO_MEMORY request cannot contain an injection target")
        elif self.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE:
            if (
                not self.memory_id
                or self.version is None
                or not self.body
                or not self.role
            ):
                raise ValueError(
                    "version replay requires body, memory version, and one role"
                )
            if self.candidate_id:
                raise ValueError("version replay cannot include a candidate")
        elif self.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE:
            if not self.candidate_id or not self.body or not self.role:
                raise ValueError(
                    "candidate replay requires body, candidate id, and one role"
                )
            if self.memory_id or self.version is not None:
                raise ValueError("candidate replay cannot target a formal version")
        else:
            raise ValueError(
                "ReplayRequest supports isolated replay conditions only"
            )
        if not self.config_fingerprint or not self.runtime_signature:
            raise ValueError(
                "replay requires frozen configuration and runtime signatures"
            )
        if not self.replay_id:
            object.__setattr__(
                self,
                "replay_id",
                stable_id(
                    "rpl",
                    {
                        "task_id": self.task_id,
                        "seed": self.seed,
                        "condition": self.condition.value,
                        "role": self.role,
                        "memory_id": self.memory_id,
                        "version": self.version,
                        "candidate_id": self.candidate_id,
                        "config_fingerprint": self.config_fingerprint,
                        "runtime_signature": self.runtime_signature,
                    },
                ),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayRequest":
        label = data.get("h_label")
        version = data.get("version")
        return cls(
            task_id=str(data.get("task_id") or ""),
            task_index=int(data.get("task_index", -1)),
            seed=int(data.get("seed", 0)),
            condition=ReplayCondition(data.get("condition")),
            config_fingerprint=str(data.get("config_fingerprint") or ""),
            runtime_signature=str(data.get("runtime_signature") or ""),
            role=str(data.get("role") or ""),
            memory_id=str(data.get("memory_id") or ""),
            version=None if version is None else int(version),
            body=str(data.get("memory") or ""),
            meta=MetaMemory.from_dict(data.get("meta")),
            candidate_id=str(data.get("candidate_id") or ""),
            validation_plan_id=str(data.get("validation_plan_id") or ""),
            h_label=HLabel(label) if label else None,
            is_source_task=bool(data.get("is_source_task", False)),
            trigger_task_id=str(data.get("trigger_task_id") or ""),
            metadata=dict(data.get("metadata") or {}),
            replay_id=str(data.get("replay_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **asdict(self),
            "condition": self.condition.value,
            "meta": self.meta.to_dict(),
            "h_label": self.h_label.value if self.h_label else None,
            "metadata": dict(self.metadata),
        }
        payload["memory"] = payload.pop("body")
        return payload


@dataclass(frozen=True)
class ReplayExecution:
    score: float
    outcome: str
    trajectory: Mapping[str, Any]
    runtime_signature: str
    evaluator_metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_audit_refs: tuple[str, ...] = ()


class ReplayExecutionError(RuntimeError):
    pass


def format_memory_block(request: ReplayRequest) -> str:
    if request.condition is ReplayCondition.NO_MEMORY:
        return ""
    label = (
        f"Memory {request.memory_id} / version {request.version}"
        if request.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE
        else f"Temporary candidate {request.candidate_id}"
    )
    lines = [
        "# DiscernMem",
        "",
        f"## {label}",
        "Memory:",
        request.memory,
        "Roles:",
        f"- {request.role}",
        "Use when:",
        *(
            [f"- {value}" for value in request.meta.use_when]
            or ["- It is relevant to the current observable state."]
        ),
        "Avoid when:",
        *(
            [f"- {value}" for value in request.meta.avoid_when]
            or ["- No verified avoid condition."]
        ),
    ]
    return "\n".join(lines).strip()


class DiscernMemReplayView(BaseCentralizedMemory):
    """A memory backend containing zero or one isolated replay artifact."""

    def __init__(self, request: ReplayRequest):
        super().__init__()
        self.request = request
        self._agent_roles: dict[str, str] = {}
        self.retrieval_records: list[dict[str, Any]] = []
        self._role_memory: dict[str, Memory] = {}

    @classmethod
    def from_config(cls, config, working_dir):
        raise RuntimeError("DiscernMemReplayView must be created from ReplayRequest")

    def register_agents(self, agents_list: list):
        for agent in agents_list:
            agent_id = getattr(agent, "id", None) or str(id(agent))
            self._agent_roles[agent_id] = getattr(agent, "role", None) or "default"

    def set_runtime_context(self, model_name=None, mas_architecture=None):
        return None

    def add_memory(self, **kwargs):
        return None

    def process_memory(self, text_memory: str, **kwargs) -> Memory:
        return Memory(
            text_memory=text_memory or None,
            extra_fields=kwargs.get("extra_fields") or {},
        )

    def _role(self, agent=None) -> str:
        if agent is None:
            return "default"
        agent_id = getattr(agent, "id", None) or str(id(agent))
        return (
            self._agent_roles.get(agent_id) or getattr(agent, "role", None) or "default"
        )

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        role = self._role(agent)
        if role in self._role_memory:
            return self._role_memory[role]
        inject = (
            self.request.condition is not ReplayCondition.NO_MEMORY
            and role == self.request.role
        )
        text = format_memory_block(self.request) if inject else ""
        selected = []
        if inject:
            selected.append(
                {
                    "memory_id": self.request.memory_id or None,
                    "version": self.request.version,
                    "candidate_id": self.request.candidate_id or None,
                    "injected_role": role,
                    "memory_sha256": hashlib.sha256(
                        self.request.memory.encode("utf-8")
                    ).hexdigest(),
                }
            )
        record = {
            "execution_type": "discernmem_replay",
            "replay_id": self.request.replay_id,
            "condition": self.request.condition.value,
            "task_id": self.request.task_id,
            "seed": self.request.seed,
            "role": role,
            "selected": selected,
            "memory_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        self.retrieval_records.append(record)
        memory = Memory(
            text_memory=text or None,
            extra_fields={
                "memory_backend": "discernmem_replay",
                "replay_id": self.request.replay_id,
                "condition": self.request.condition.value,
                "injected_memories": selected,
            },
        )
        self._role_memory[role] = memory
        return memory

    def assert_isolation(self) -> None:
        injected = [
            item for record in self.retrieval_records for item in record["selected"]
        ]
        if self.request.condition is ReplayCondition.NO_MEMORY:
            if injected:
                raise ReplayExecutionError("NO_MEMORY replay leaked a memory injection")
            return
        if not injected:
            raise ReplayExecutionError(
                "isolated replay never delivered the target artifact to its role"
            )
        if any(item["injected_role"] != self.request.role for item in injected):
            raise ReplayExecutionError("isolated replay injected a non-target role")
        if self.request.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE:
            if any(
                item["memory_id"] != self.request.memory_id
                or item["version"] != self.request.version
                or item["candidate_id"] is not None
                for item in injected
            ):
                raise ReplayExecutionError(
                    "isolated version replay injected another artifact"
                )
        if self.request.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE:
            if any(
                item["candidate_id"] != self.request.candidate_id
                or item["memory_id"] is not None
                for item in injected
            ):
                raise ReplayExecutionError(
                    "isolated candidate replay injected another artifact"
                )


ReplayRunner = Callable[[ReplayRequest, DiscernMemReplayView], ReplayExecution]


def _isolated_replay_entry(executor, request: ReplayRequest, connection) -> None:
    try:
        connection.send(("ok", executor.run(request)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class ReplayExecutor:
    def __init__(
        self,
        store: DiscernMemStore,
        runner: ReplayRunner,
        *,
        max_workers: int = 1,
        max_attempts: int = 2,
        process_isolation: bool = False,
        crash_retries: int = 1,
        replay_timeout_seconds: Optional[float] = None,
    ):
        self.store = store
        self.runner = runner
        self.max_workers = max(int(max_workers), 1)
        self.max_attempts = max(int(max_attempts), 1)
        self.process_isolation = bool(process_isolation)
        self.crash_retries = max(int(crash_retries), 0)
        self.replay_timeout_seconds = (
            None
            if replay_timeout_seconds is None or float(replay_timeout_seconds) <= 0
            else float(replay_timeout_seconds)
        )

    def _request_path(self, request: ReplayRequest) -> Path:
        return (
            self.store.root
            / "replays"
            / "requests"
            / f"{safe_component(request.replay_id)}.json"
        )

    def _attempt_path(self, request: ReplayRequest, attempt: int) -> Path:
        return (
            self.store.root
            / "replays"
            / "attempts"
            / safe_component(request.replay_id)
            / f"attempt_{attempt:03d}.json"
        )

    def _next_attempt_number(self, request: ReplayRequest) -> int:
        directory = (
            self.store.root / "replays" / "attempts" / safe_component(request.replay_id)
        )
        existing = []
        for path in directory.glob("attempt_*.json"):
            try:
                existing.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return max(existing, default=0) + 1

    def _save_request(self, request: ReplayRequest) -> None:
        with self.store._exclusive():
            self.store._write_once(self._request_path(request), request.to_dict())

    def _save_attempt(
        self,
        request: ReplayRequest,
        attempt: int,
        *,
        status: str,
        error: str = "",
        latency_seconds: float = 0.0,
    ) -> None:
        with self.store._exclusive():
            self.store._write_once(
                self._attempt_path(request, attempt),
                {
                    "replay_id": request.replay_id,
                    "attempt": attempt,
                    "status": status,
                    "error": error or None,
                    "latency_seconds": latency_seconds,
                    "recorded_at": utc_now(),
                },
            )

    def run(self, request: ReplayRequest) -> ReplayRecord:
        cached = self.store.get_replay(request.replay_id)
        if cached is not None:
            if cached.status is ReplayStatus.TIMEOUT:
                return cached
            if cached.status is not ReplayStatus.SUCCEEDED:
                raise ReplayExecutionError(
                    f"cached replay is not successful: {request.replay_id}"
                )
            return cached
        self._save_request(request)
        errors = []
        first_attempt = self._next_attempt_number(request)
        for attempt in range(first_attempt, first_attempt + self.max_attempts):
            view = DiscernMemReplayView(request)
            started = time.perf_counter()
            try:
                execution = self.runner(request, view)
                latency = time.perf_counter() - started
                if execution.runtime_signature != request.runtime_signature:
                    raise ReplayExecutionError(
                        "runner runtime signature differs from the frozen replay request"
                    )
                view.assert_isolation()
                trajectory_path = (
                    self.store.root
                    / "trajectories"
                    / "replay"
                    / f"{safe_component(request.replay_id)}.json"
                )
                trajectory_payload = {
                    **dict(execution.trajectory),
                    "execution_type": "discernmem_replay",
                    "replay_id": request.replay_id,
                    "condition": request.condition.value,
                    "target_role": request.role or None,
                    "seed": request.seed,
                    "memory_retrievals": list(view.retrieval_records),
                }
                _atomic_write_json(trajectory_path, trajectory_payload)
                record = ReplayRecord(
                    replay_id=request.replay_id,
                    task_id=request.task_id,
                    seed=request.seed,
                    condition=request.condition,
                    role=request.role,
                    config_fingerprint=request.config_fingerprint,
                    runtime_signature=request.runtime_signature,
                    status=ReplayStatus.SUCCEEDED,
                    score=float(execution.score),
                    outcome=str(execution.outcome),
                    memory_id=request.memory_id,
                    version=request.version,
                    candidate_id=request.candidate_id,
                    validation_plan_id=request.validation_plan_id,
                    h_label=request.h_label,
                    is_source_task=request.is_source_task,
                    trace_ref=str(trajectory_path),
                    prompt_audit_refs=tuple(execution.prompt_audit_refs),
                    created_at=utc_now(),
                    metadata={
                        **dict(request.metadata),
                        "evaluator": dict(execution.evaluator_metadata),
                        "latency_seconds": latency,
                    },
                )
                self.store.record_replay(record)
                self._save_attempt(
                    request,
                    attempt,
                    status="succeeded",
                    latency_seconds=latency,
                )
                return record
            except Exception as exc:
                latency = time.perf_counter() - started
                errors.append(f"{type(exc).__name__}: {exc}")
                self._save_attempt(
                    request,
                    attempt,
                    status="failed",
                    error=errors[-1],
                    latency_seconds=latency,
                )
        raise ReplayExecutionError(
            f"replay {request.replay_id} failed after {self.max_attempts} attempts: "
            + " | ".join(errors)
        )

    def run_many(self, requests: Iterable[ReplayRequest]) -> list[ReplayRecord]:
        requests = list(requests)
        if self.process_isolation:
            return self._run_many_isolated(requests)
        if self.max_workers == 1 or len(requests) <= 1:
            return [self.run(request) for request in requests]
        completed: dict[int, ReplayRecord] = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(requests))
        ) as pool:
            futures = {
                pool.submit(self.run, request): index
                for index, request in enumerate(requests)
            }
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        return [completed[index] for index in range(len(requests))]

    def _run_many_isolated(
        self, requests: list[ReplayRequest]
    ) -> list[ReplayRecord]:
        import multiprocessing as mp
        from multiprocessing.connection import wait

        try:
            context = mp.get_context("fork")
        except ValueError:
            return [self.run(request) for request in requests]

        pending = [(index, request, 0) for index, request in enumerate(requests)]
        active = {}
        completed: dict[int, ReplayRecord] = {}
        errors: dict[int, str] = {}

        def start_one(index, request, attempt):
            receive_conn, send_conn = context.Pipe(duplex=False)
            process = context.Process(
                target=_isolated_replay_entry,
                args=(self, request, send_conn),
                daemon=False,
            )
            process.start()
            send_conn.close()
            active[process.pid] = (
                process,
                receive_conn,
                index,
                request,
                attempt,
                time.monotonic(),
            )

        def finish_one(process_id, *, result=None, error=None):
            process, connection, index, request, attempt, _started = active.pop(
                process_id
            )
            connection.close()
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if result is not None:
                completed[index] = result
            elif attempt < self.crash_retries:
                pending.insert(0, (index, request, attempt + 1))
            else:
                errors[index] = str(error)

        def timeout_one(process_id):
            process, connection, index, request, attempt, started = active.pop(
                process_id
            )
            latency = max(0.0, time.monotonic() - started)
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
            connection.close()
            error = f"replay exceeded timeout of {self.replay_timeout_seconds:.1f}s"
            self._save_attempt(
                request,
                self._next_attempt_number(request),
                status="timeout",
                error=error,
                latency_seconds=latency,
            )
            record = ReplayRecord(
                replay_id=request.replay_id,
                task_id=request.task_id,
                seed=request.seed,
                condition=request.condition,
                role=request.role,
                config_fingerprint=request.config_fingerprint,
                runtime_signature=request.runtime_signature,
                status=ReplayStatus.TIMEOUT,
                memory_id=request.memory_id,
                version=request.version,
                candidate_id=request.candidate_id,
                validation_plan_id=request.validation_plan_id,
                h_label=request.h_label,
                is_source_task=request.is_source_task,
                error=error,
                created_at=utc_now(),
                metadata={
                    **dict(request.metadata),
                    "timeout_seconds": self.replay_timeout_seconds,
                    "latency_seconds": latency,
                    "exitcode": process.exitcode,
                },
            )
            self.store.record_replay(record)
            completed[index] = record

        while pending or active:
            while pending and len(active) < min(self.max_workers, len(requests)):
                index, request, attempt = pending.pop(0)
                try:
                    start_one(index, request, attempt)
                except Exception as exc:
                    if attempt < self.crash_retries:
                        pending.insert(0, (index, request, attempt + 1))
                    else:
                        errors[index] = f"process start failed: {exc}"

            if not active:
                continue

            ready = set(wait([item[1] for item in active.values()], timeout=0.2))
            for process_id, (
                process,
                connection,
                _index,
                _request,
                _attempt,
                started,
            ) in list(active.items()):
                if (
                    self.replay_timeout_seconds is not None
                    and process.is_alive()
                    and time.monotonic() - started >= self.replay_timeout_seconds
                ):
                    timeout_one(process_id)
                    continue
                if connection in ready:
                    try:
                        status, payload = connection.recv()
                    except (EOFError, OSError) as exc:
                        finish_one(
                            process_id,
                            error=f"worker pipe closed: {exc}; exitcode={process.exitcode}",
                        )
                    else:
                        if status == "ok":
                            finish_one(process_id, result=payload)
                        else:
                            finish_one(process_id, error=payload)
                elif not process.is_alive():
                    if connection.poll():
                        continue
                    finish_one(
                        process_id,
                        error=f"worker exited without result; exitcode={process.exitcode}",
                    )

        if errors:
            details = " | ".join(
                f"{requests[index].replay_id}: {error}"
                for index, error in sorted(errors.items())
            )
            raise ReplayExecutionError(f"process-isolated replay failed: {details}")
        return [completed[index] for index in range(len(requests))]


def official_run_is_reusable(
    *,
    condition: ReplayCondition,
    target_role: str,
    memory_id: str,
    version: int,
    actual_injections: Iterable[Mapping[str, Any]],
) -> bool:
    if condition is not ReplayCondition.ONLY_VERSION_FOR_ROLE:
        return False
    unique = {
        (
            str(item.get("memory_id") or ""),
            int(item.get("version", -1)),
            str(item.get("injected_role") or ""),
        )
        for item in actual_injections
        if item.get("memory_id")
    }
    return unique == {(str(memory_id), int(version), str(target_role))}
