from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable, Mapping, Optional

from .config import DiscernMemConfig
from .invariants import (
    InvariantViolation,
    MemoryState,
    assert_linear_parent,
    require_single_role,
)
from .schema import (
    AdmissionRecord,
    CapabilityConstraint,
    MemoryIdentity,
    MemoryVersion,
    MetaMemory,
    MetaUsageEvent,
    RepairCandidate,
    ReplayRecord,
    TaskEvidence,
    ValidationPlan,
    VersionBase,
    VersionEvidence,
)


def _validate_single_role_meta(meta: MetaMemory, *, context: str) -> str:
    owner_role = require_single_role(meta.roles, context=context)
    unexpected_effect_roles = set(meta.effect) - {owner_role}
    if unexpected_effect_roles:
        raise InvariantViolation(
            f"{context} has effect evidence outside owner role {owner_role!r}: "
            f"{sorted(unexpected_effect_roles)!r}"
        )
    return owner_role


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_component(value: str) -> str:
    value = str(value or "").strip()
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("JSON object id cannot be empty")
    return cleaned[:220]


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DiscernMemStorageError(RuntimeError):
    pass


class NamespaceCollisionError(DiscernMemStorageError):
    pass


class ImmutableRecordError(DiscernMemStorageError):
    pass


class ConcurrentUpdateError(DiscernMemStorageError):
    pass


class DiscernMemStore:
    """Human-inspectable, JSON-first DiscernMem persistence.

    Immutable objects are stored once. Mutable pointers are replaced atomically
    while holding a process-wide file lock. Version bodies and evidence
    revisions remain append-only, so every deployed state is auditable without
    querying a database.
    """

    STORAGE_TYPE = "discernmem_json_store"
    CURRENT_LAYOUT = "records_reports_v3"
    _DATA_DIRS = (
        "identities",
        "versions",
        "replays",
        "validation_plans",
        "repair_candidates",
        "admissions",
        "tasks",
        "events",
        "trajectories",
        "embeddings",
        "audit",
    )

    def __init__(self, root: str | Path, config: DiscernMemConfig):
        self.output_root = Path(root).resolve()
        self.root = self.output_root / "records"
        self.records_root = self.root
        self.reports_root = self.output_root / "reports"
        self.config = config
        self._thread_lock = threading.RLock()
        self.lock_path = self.root / ".locks" / "store.lock"
        self.manifest_path = self.root / "manifest.json"
        self.config_path = self.root / "config.json"
        self.full_export_path = self.reports_root / "exports" / "memory_evolution.json"
        self.report_catalog_path = self.reports_root / "memories.json"
        self.report_summary_path = self.reports_root / "summary.json"
        self.report_index_path = self.reports_root / "index.md"
        self.report_memory_dir = self.reports_root / "memories"
        self.progress_path = self.output_root / "progress.json"
        self._initialize()

    @contextmanager
    def _exclusive(self):
        with self._thread_lock:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _initialize(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._exclusive():
            if self.manifest_path.exists():
                manifest = read_json(self.manifest_path)
                if manifest.get("storage_type") != self.STORAGE_TYPE:
                    raise NamespaceCollisionError(
                        f"storage root belongs to {manifest.get('storage_type')!r}, not DiscernMem"
                    )
                if manifest.get("namespace") != "discernmem":
                    raise NamespaceCollisionError(
                        "DiscernMem namespace marker is invalid"
                    )
                if manifest.get("config_fingerprint") != self.config.fingerprint:
                    raise NamespaceCollisionError(
                        "DiscernMem store configuration differs from its frozen manifest"
                    )
                if manifest.get("layout") != self.CURRENT_LAYOUT:
                    raise NamespaceCollisionError(
                        "DiscernMem storage layout differs from its manifest"
                    )
            else:
                _atomic_write_json(
                    self.manifest_path,
                    {
                        "storage_type": self.STORAGE_TYPE,
                        "namespace": "discernmem",
                        "layout": self.CURRENT_LAYOUT,
                        "schema_version": self.config.schema_version,
                        "config_fingerprint": self.config.fingerprint,
                        "created_at": utc_now(),
                    },
                )
                _atomic_write_json(self.config_path, self.config.to_dict())
            for directory in self._DATA_DIRS:
                (self.root / directory).mkdir(parents=True, exist_ok=True)
            for directory in (
                self.root / "replays" / "requests",
                self.root / "replays" / "attempts",
                self.root / "replays" / "results",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            self.report_memory_dir.mkdir(parents=True, exist_ok=True)
            self.full_export_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.report_summary_path.exists():
                _atomic_write_json(
                    self.report_summary_path,
                    {
                        "storage_type": "discernmem_report_summary",
                        "layout": self.CURRENT_LAYOUT,
                        "updated_at": utc_now(),
                        "memories": 0,
                        "states": {},
                        "versions": 0,
                        "accepted_version_updates": 0,
                        "admission_decisions": {},
                        "records_dir": os.path.relpath(
                            self.records_root, self.reports_root
                        ),
                        "complete": False,
                    },
                )
            if not self.report_index_path.exists():
                _atomic_write_text(
                    self.report_index_path,
                    "# DiscernMem Reports\n\n"
                    "No memory reports have been generated yet.\n",
                )

    @staticmethod
    def _event_memory_refs(payload: Any, known_memory_ids: set[str]) -> set[str]:
        refs: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple, set)):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and value in known_memory_ids:
                refs.add(value)

        visit(payload)
        return refs

    def _read_event_records(
        self, known_memory_ids: set[str]
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        records: list[dict[str, Any]] = []
        refs: dict[str, list[str]] = {memory_id: [] for memory_id in known_memory_ids}
        event_root = self.root / "events"
        for event_path in sorted(event_root.glob("*.json")):
            record = read_json(event_path)
            records.append(record)
            event_id = str(record.get("event_id") or event_path.stem)
            for memory_id in self._event_memory_refs(record, known_memory_ids):
                refs[memory_id].append(event_id)
        for event_dir in sorted(event_root.iterdir()):
            if not event_dir.is_dir() or not (event_dir / "event.json").exists():
                continue
            record = read_json(event_dir / "event.json")
            completion_path = event_dir / "completion.json"
            record["completion"] = (
                read_json(completion_path) if completion_path.exists() else None
            )
            record["stages"] = [
                read_json(path)
                for path in sorted((event_dir / "stages").glob("*.json"))
            ]
            records.append(record)
            event_id = str(record.get("event_id") or event_dir.name)
            for memory_id in self._event_memory_refs(record, known_memory_ids):
                refs[memory_id].append(event_id)
        return records, refs

    def _memory_snapshot(
        self,
        identity: MemoryIdentity,
        *,
        admissions: Iterable[Mapping[str, Any]] = (),
        repair_candidates: Iterable[Mapping[str, Any]] = (),
        event_refs: Iterable[str] = (),
        complete: bool,
    ) -> dict[str, Any]:
        versions = []
        admission_ids = set()
        for version_number in range(identity.current_version + 1):
            base = read_json(self._base_path(identity.memory_id, version_number))
            admission_id = str((base.get("audit") or {}).get("admission_id") or "")
            if admission_id:
                admission_ids.add(admission_id)
            evidence_history = [
                read_json(path)
                for path in sorted(
                    (
                        self._version_dir(identity.memory_id, version_number)
                        / "evidence"
                    ).glob("rev_*.json")
                )
            ]
            versions.append(
                {
                    "version": version_number,
                    "parent_version": base.get("parent_version"),
                    "memory": base.get("memory", ""),
                    "source_task_id": base.get("source_task_id", ""),
                    "decision": base.get("decision", ""),
                    "created_at": base.get("created_at", ""),
                    "audit": dict(base.get("audit") or {}),
                    "evidence_history": evidence_history,
                }
            )
        memory_admissions = [
            dict(item)
            for item in admissions
            if item.get("admission_id") in admission_ids
            or item.get("target_memory_id") == identity.memory_id
        ]
        memory_candidates = [
            dict(item)
            for item in repair_candidates
            if item.get("memory_id") == identity.memory_id
        ]
        current = versions[-1]["evidence_history"][-1]
        return {
            "memory_id": identity.memory_id,
            "source_task_id": identity.source_task_id,
            "current_version": identity.current_version,
            "current_evidence_revision": identity.current_evidence_revision,
            "current_state": current.get("state", ""),
            "created_at": identity.created_at,
            "updated_at": identity.updated_at,
            "semantic_key": identity.semantic_key,
            "imported_from": identity.imported_from,
            "versions": versions,
            "admission_history": memory_admissions,
            "repair_candidates": memory_candidates,
            "event_refs": list(dict.fromkeys(str(value) for value in event_refs)),
            "complete": bool(complete),
        }

    @staticmethod
    def _readable_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
        readable_versions = []
        for version in memory.get("versions", []):
            evidence_history = list(version.get("evidence_history") or [])
            latest = evidence_history[-1] if evidence_history else {}
            transitions = []
            previous_state = None
            for evidence in evidence_history:
                state = str(evidence.get("state") or "")
                if state == previous_state:
                    continue
                transitions.append(
                    {
                        "revision": evidence.get("revision"),
                        "state": state,
                        "reason": evidence.get("reason", ""),
                        "updated_at": evidence.get("updated_at", ""),
                    }
                )
                previous_state = state
            meta = dict(latest.get("meta") or {})
            effects = {
                role: {
                    "n_tasks": stats.get("n_tasks", 0),
                    "mean_delta": stats.get("mean_delta", 0.0),
                    "positive_count": stats.get("positive_count", 0),
                    "negative_count": stats.get("negative_count", 0),
                }
                for role, stats in dict(meta.get("effect") or {}).items()
            }
            readable_versions.append(
                {
                    "version": version.get("version"),
                    "parent_version": version.get("parent_version"),
                    "memory": version.get("memory", ""),
                    "source_task_id": version.get("source_task_id", ""),
                    "decision": version.get("decision", ""),
                    "created_at": version.get("created_at", ""),
                    "final_state": latest.get("state", ""),
                    "final_state_reason": latest.get("reason", ""),
                    "roles": list(meta.get("roles") or []),
                    "use_when": list(meta.get("use_when") or []),
                    "avoid_when": list(meta.get("avoid_when") or []),
                    "effect_summary": effects,
                    "evidence_revision_count": len(evidence_history),
                    "state_transitions": transitions,
                    "replay_refs": list(latest.get("replay_ids") or []),
                }
            )
        return {
            "memory_id": memory.get("memory_id", ""),
            "source_task_id": memory.get("source_task_id", ""),
            "current_version": memory.get("current_version", 0),
            "current_evidence_revision": memory.get("current_evidence_revision", 0),
            "current_state": memory.get("current_state", ""),
            "created_at": memory.get("created_at", ""),
            "updated_at": memory.get("updated_at", ""),
            "versions": readable_versions,
            "admission_history": [
                {
                    "source_task_id": item.get("source_task_id", ""),
                    "decision": item.get("decision", ""),
                    "reason": item.get("reason", ""),
                    "created_at": item.get("created_at", ""),
                }
                for item in memory.get("admission_history", [])
            ],
            "repair_candidate_refs": [
                item.get("candidate_id", "")
                for item in memory.get("repair_candidates", [])
            ],
            "event_refs": list(memory.get("event_refs") or []),
            "complete": bool(memory.get("complete", False)),
        }

    @staticmethod
    def _memory_markdown(memory: Mapping[str, Any]) -> str:
        lines = [
            f"# Memory {memory.get('memory_id', '')}",
            "",
            f"- Current state: `{memory.get('current_state', '')}`",
            f"- Current version: `v{memory.get('current_version', 0)}`",
            f"- Source task: `{memory.get('source_task_id', '')}`",
            f"- Updated at: `{memory.get('updated_at', '')}`",
            f"- Complete materialization: `{str(bool(memory.get('complete', False))).lower()}`",
        ]
        for version in memory.get("versions", []):
            lines.extend(
                [
                    "",
                    f"## Version v{version.get('version', 0)}",
                    "",
                    f"- Parent: `{version.get('parent_version')}`",
                    f"- Source task: `{version.get('source_task_id', '')}`",
                    f"- Decision: `{version.get('decision', '')}`",
                    f"- Final state: `{version.get('final_state', '')}`",
                    "",
                    "### Memory",
                    "",
                    str(version.get("memory") or ""),
                    "",
                    "### Deployment",
                    "",
                    "Roles: " + (", ".join(version.get("roles") or []) or "none"),
                    "",
                    "Use when:",
                    *(
                        [f"- {item}" for item in version.get("use_when") or []]
                        or ["- No condition recorded."]
                    ),
                    "",
                    "Avoid when:",
                    *(
                        [f"- {item}" for item in version.get("avoid_when") or []]
                        or ["- No condition recorded."]
                    ),
                    "",
                    "### Evidence history",
                    "",
                    *(
                        [
                            "- Revision {revision}: `{state}` - {reason}".format(
                                revision=item.get("revision"),
                                state=item.get("state", ""),
                                reason=item.get("reason", ""),
                            )
                            for item in version.get("state_transitions") or []
                        ]
                        or ["- No state transition recorded."]
                    ),
                    "",
                    "Replay references:",
                    *(
                        [f"- `{item}`" for item in version.get("replay_refs") or []]
                        or ["- No replay reference recorded."]
                    ),
                ]
            )
        if memory.get("admission_history"):
            lines.extend(
                [
                    "",
                    "## Admission history",
                    "",
                    *[
                        "- `{decision}` from `{task}` - {reason}".format(
                            decision=item.get("decision", ""),
                            task=item.get("source_task_id", ""),
                            reason=item.get("reason", ""),
                        )
                        for item in memory.get("admission_history") or []
                    ],
                ]
            )
        if memory.get("repair_candidate_refs"):
            lines.extend(
                [
                    "",
                    "## Repair candidate references",
                    "",
                    *[
                        f"- `{item}`"
                        for item in memory.get("repair_candidate_refs") or []
                    ],
                ]
            )
        if memory.get("event_refs"):
            lines.extend(
                [
                    "",
                    "## Event references",
                    "",
                    *[f"- `{item}`" for item in memory.get("event_refs") or []],
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _write_memory_report(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        readable = self._readable_memory(memory)
        memory_id = safe_component(str(readable["memory_id"]))
        payload = {
            "storage_type": "discernmem_memory_report",
            "generated_at": utc_now(),
            **readable,
        }
        _atomic_write_json(self.report_memory_dir / f"{memory_id}.json", payload)
        _atomic_write_text(
            self.report_memory_dir / f"{memory_id}.md",
            self._memory_markdown(payload),
        )
        return payload

    def refresh_memory_report(self, memory_id: str) -> dict[str, Any]:
        identity = self.get_identity(memory_id)
        if identity is None:
            raise KeyError(memory_id)
        snapshot = self._memory_snapshot(identity, complete=False)
        return self._write_memory_report(snapshot)

    def _write_report_index(
        self,
        memories: Iterable[Mapping[str, Any]],
        *,
        complete: bool,
        updated_at: str,
    ) -> None:
        items = sorted(memories, key=lambda item: str(item.get("memory_id") or ""))
        lines = [
            "# DiscernMem Reports",
            "",
            f"Generated at: `{updated_at}`",
            f"Complete materialization: `{str(bool(complete)).lower()}`",
            "",
            "| Memory | State | Version | Source task |",
            "|---|---|---:|---|",
        ]
        for item in items:
            memory_id = str(item.get("memory_id") or "")
            safe_id = safe_component(memory_id)
            lines.append(
                f"| [{memory_id}](memories/{safe_id}.md) | "
                f"{item.get('current_state', '')} | "
                f"{item.get('current_version', 0)} | "
                f"{item.get('source_task_id', '')} |"
            )
        _atomic_write_text(self.report_index_path, "\n".join(lines) + "\n")

    def materialize_views(self, *, complete: bool = False) -> dict[str, Any]:
        with self._exclusive():
            watermark = (
                read_json(self.progress_path) if self.progress_path.exists() else {}
            )
            admissions = [
                read_json(path)
                for path in sorted((self.root / "admissions").glob("*.json"))
            ]
            repair_candidates = [
                read_json(path)
                for path in sorted((self.root / "repair_candidates").glob("*.json"))
            ]
            identities = self.iter_identities()
            known_ids = {identity.memory_id for identity in identities}
            _events, event_refs = self._read_event_records(known_ids)
            memories = [
                self._memory_snapshot(
                    identity,
                    admissions=admissions,
                    repair_candidates=repair_candidates,
                    event_refs=event_refs.get(identity.memory_id, ()),
                    complete=complete,
                )
                for identity in identities
            ]
            updated_at = utc_now()
            _atomic_write_json(
                self.full_export_path,
                {
                    "storage_type": "discernmem_aggregated_memory_evolution",
                    "namespace": "discernmem",
                    "layout": self.CURRENT_LAYOUT,
                    "config_fingerprint": self.config.fingerprint,
                    "generated_at": updated_at,
                    "complete": bool(complete),
                    "last_completed_task": watermark.get("last_completed_task"),
                    "last_completed_task_index": watermark.get(
                        "last_completed_task_index"
                    ),
                    "completed_tasks": watermark.get("completed_tasks"),
                    "memories": memories,
                },
            )
            readable_memories = [
                self._write_memory_report(memory) for memory in memories
            ]
            _atomic_write_json(
                self.report_catalog_path,
                {
                    "storage_type": "discernmem_report_catalog",
                    "layout": self.CURRENT_LAYOUT,
                    "generated_at": updated_at,
                    "complete": bool(complete),
                    "last_completed_task": watermark.get("last_completed_task"),
                    "last_completed_task_index": watermark.get(
                        "last_completed_task_index"
                    ),
                    "completed_tasks": watermark.get("completed_tasks"),
                    "memories": readable_memories,
                },
            )
            summary = self._write_report_summary(
                readable_memories,
                admissions=admissions,
                complete=complete,
                updated_at=updated_at,
                watermark=watermark,
            )
            self._write_report_index(
                readable_memories, complete=complete, updated_at=updated_at
            )
            return summary

    def _write_report_summary(
        self,
        memories: Iterable[Mapping[str, Any]],
        *,
        admissions: Iterable[Mapping[str, Any]] = (),
        complete: bool,
        updated_at: Optional[str] = None,
        watermark: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        memory_items = list(memories)
        admission_items = list(admissions)
        state_counts: dict[str, int] = {}
        versions = 0
        for memory in memory_items:
            state = str(memory.get("current_state") or "")
            state_counts[state] = state_counts.get(state, 0) + 1
            versions += len(memory.get("versions") or [])
        decision_counts: dict[str, int] = {}
        for item in admission_items:
            decision = str(item.get("decision") or "")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        payload = {
            "memory_backend": "discernmem",
            "storage_type": "discernmem_report_summary",
            "layout": self.CURRENT_LAYOUT,
            "updated_at": updated_at or utc_now(),
            "complete": bool(complete),
            "last_completed_task": (watermark or {}).get("last_completed_task"),
            "last_completed_task_index": (watermark or {}).get(
                "last_completed_task_index"
            ),
            "completed_tasks": (watermark or {}).get("completed_tasks"),
            "memories": len(memory_items),
            "states": state_counts,
            "versions": versions,
            "accepted_version_updates": max(versions - len(memory_items), 0),
            "admission_decisions": decision_counts,
            "accepted_admissions": sum(
                count
                for decision, count in decision_counts.items()
                if decision in {"create", "attach"}
            ),
            "records_dir": os.path.relpath(self.records_root, self.reports_root),
            "memory_catalog": "memories.json",
            "memory_index": "index.md",
            "full_export": "exports/memory_evolution.json",
        }
        _atomic_write_json(self.report_summary_path, payload)
        return payload

    def refresh_report_index(
        self, *, watermark: Optional[Mapping[str, Any]] = None
    ) -> dict[str, Any]:
        reports = [
            read_json(path) for path in sorted(self.report_memory_dir.glob("*.json"))
        ]
        admissions = [
            read_json(path)
            for path in sorted((self.root / "admissions").glob("*.json"))
        ]
        updated_at = utc_now()
        summary = self._write_report_summary(
            reports,
            admissions=admissions,
            complete=False,
            updated_at=updated_at,
            watermark=watermark,
        )
        self._write_report_index(reports, complete=False, updated_at=updated_at)
        return summary

    def record_progress(
        self,
        *,
        task_id: str,
        status: str,
        task_index: int = -1,
        memory_id: str = "",
        replay_count: int = 0,
        details: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._exclusive():
            watermark = {
                "last_completed_task": str(task_id),
                "last_completed_task_index": int(task_index),
                "completed_tasks": (
                    int(task_index) + 1 if int(task_index) >= 0 else None
                ),
            }
            summary = self.refresh_report_index(watermark=watermark)
            payload = {
                "storage_type": "discernmem_progress",
                "layout": self.CURRENT_LAYOUT,
                **watermark,
                "status": str(status),
                "memory_id": str(memory_id),
                "replay_count": int(replay_count),
                "memories": summary["memories"],
                "states": summary["states"],
                "versions": summary["versions"],
                "updated_at": utc_now(),
                "details": dict(details or {}),
            }
            _atomic_write_json(self.progress_path, payload)
        return payload

    def _write_once(self, path: Path, payload: Mapping[str, Any]) -> bool:
        if path.exists():
            if _canonical(read_json(path)) == _canonical(payload):
                return False
            raise ImmutableRecordError(f"immutable JSON record already exists: {path}")
        _atomic_write_json(path, payload)
        return True

    def _identity_path(self, memory_id: str) -> Path:
        return self.root / "identities" / f"{safe_component(memory_id)}.json"

    def _version_dir(self, memory_id: str, version: int) -> Path:
        return self.root / "versions" / safe_component(memory_id) / f"v{int(version)}"

    def _base_path(self, memory_id: str, version: int) -> Path:
        return self._version_dir(memory_id, version) / "base.json"

    def _evidence_path(self, memory_id: str, version: int, revision: int) -> Path:
        return (
            self._version_dir(memory_id, version)
            / "evidence"
            / f"rev_{int(revision):04d}.json"
        )

    def get_identity(self, memory_id: str) -> Optional[MemoryIdentity]:
        path = self._identity_path(memory_id)
        return MemoryIdentity.from_dict(read_json(path)) if path.exists() else None

    def iter_identities(self) -> list[MemoryIdentity]:
        return [
            MemoryIdentity.from_dict(read_json(path))
            for path in sorted((self.root / "identities").glob("*.json"))
        ]

    def get_version(
        self,
        memory_id: str,
        version: Optional[int] = None,
        evidence_revision: Optional[int] = None,
    ) -> Optional[MemoryVersion]:
        identity = self.get_identity(memory_id)
        if not identity:
            return None
        selected_version = identity.current_version if version is None else int(version)
        if evidence_revision is None:
            if selected_version == identity.current_version:
                selected_revision = identity.current_evidence_revision
            else:
                revision_paths = sorted(
                    (self._version_dir(memory_id, selected_version) / "evidence").glob(
                        "rev_*.json"
                    )
                )
                if not revision_paths:
                    return None
                selected_revision = int(revision_paths[-1].stem.split("_")[-1])
        else:
            selected_revision = int(evidence_revision)
        base_path = self._base_path(memory_id, selected_version)
        evidence_path = self._evidence_path(
            memory_id, selected_version, selected_revision
        )
        if not base_path.exists() or not evidence_path.exists():
            return None
        return MemoryVersion(
            base=VersionBase.from_dict(read_json(base_path)),
            evidence=VersionEvidence.from_dict(read_json(evidence_path)),
        )

    def iter_current_versions(
        self,
        states: Optional[Iterable[MemoryState | str]] = None,
    ) -> list[MemoryVersion]:
        allowed = (
            {MemoryState(value) for value in states} if states is not None else None
        )
        versions = []
        for identity in self.iter_identities():
            version = self.get_version(identity.memory_id)
            if version and (allowed is None or version.state in allowed):
                versions.append(version)
        return versions

    def create_memory(
        self,
        *,
        memory_id: str,
        body: str,
        meta: MetaMemory,
        source_task_id: str,
        source_trajectory_ref: str = "",
        source_event_ids: Iterable[str] = (),
        semantic_key: str = "",
        state: MemoryState = MemoryState.PROVISIONAL,
        capabilities: Iterable[CapabilityConstraint] = (),
        created_at: Optional[str] = None,
        imported_from: Optional[str] = None,
        audit: Optional[Mapping[str, Any]] = None,
    ) -> MemoryVersion:
        if not str(body).strip():
            raise ValueError("memory body cannot be empty")
        _validate_single_role_meta(meta, context="created memory")
        created_at = created_at or utc_now()
        base = VersionBase(
            memory_id=memory_id,
            version=0,
            body=str(body).strip(),
            parent_version=None,
            source_task_id=str(source_task_id),
            created_at=created_at,
            source_trajectory_ref=str(source_trajectory_ref or ""),
            source_event_ids=tuple(dict.fromkeys(str(value) for value in source_event_ids if str(value))),
            decision="created",
            audit=dict(audit or {}),
        )
        evidence = VersionEvidence(
            memory_id=memory_id,
            version=0,
            revision=0,
            meta=meta,
            capabilities=tuple(capabilities),
            state=MemoryState(state),
            updated_at=created_at,
            reason="initial",
            audit=dict(audit or {}),
        )
        identity = MemoryIdentity(
            memory_id=memory_id,
            current_version=0,
            current_evidence_revision=0,
            source_task_id=str(source_task_id),
            created_at=created_at,
            updated_at=created_at,
            semantic_key=str(semantic_key),
            imported_from=imported_from,
        )
        with self._exclusive():
            existing = self.get_identity(memory_id)
            if existing:
                current = self.get_version(memory_id)
                if current and _canonical(current.to_dict()) == _canonical(
                    MemoryVersion(base, evidence).to_dict()
                ):
                    return current
                raise ImmutableRecordError(
                    f"memory identity already exists: {memory_id}"
                )
            self._write_once(self._base_path(memory_id, 0), base.to_dict())
            self._write_once(self._evidence_path(memory_id, 0, 0), evidence.to_dict())
            _atomic_write_json(self._identity_path(memory_id), identity.to_dict())
            self.refresh_memory_report(memory_id)
        return MemoryVersion(base=base, evidence=evidence)

    def record_meta_usage_event(self, event: MetaUsageEvent) -> Path:
        """Persist one formal-use observation without waiting for aggregation."""
        path = self.root / "audit" / "meta_usage" / f"{safe_component(event.event_id)}.json"
        payload = event.to_dict()
        with self._exclusive():
            if path.exists():
                existing = read_json(path)
                if _canonical(existing) != _canonical(payload):
                    raise ImmutableRecordError(
                        f"meta usage event already exists with different content: {event.event_id}"
                    )
            else:
                _atomic_write_json(path, payload)
        return path

    def update_evidence(
        self,
        *,
        memory_id: str,
        expected_version: int,
        expected_revision: int,
        meta: MetaMemory,
        capabilities: Iterable[CapabilityConstraint],
        state: MemoryState,
        replay_ids: Iterable[str],
        reason: str,
        audit: Optional[Mapping[str, Any]] = None,
    ) -> VersionEvidence:
        _validate_single_role_meta(meta, context="updated memory evidence")
        with self._exclusive():
            identity = self.get_identity(memory_id)
            if not identity:
                raise KeyError(memory_id)
            if identity.current_version != int(expected_version):
                raise ConcurrentUpdateError(
                    f"expected version {expected_version}, current is {identity.current_version}"
                )
            if identity.current_evidence_revision != int(expected_revision):
                raise ConcurrentUpdateError(
                    f"expected evidence revision {expected_revision}, current is "
                    f"{identity.current_evidence_revision}"
                )
            revision = identity.current_evidence_revision + 1
            evidence = VersionEvidence(
                memory_id=memory_id,
                version=identity.current_version,
                revision=revision,
                meta=meta,
                capabilities=tuple(capabilities),
                state=MemoryState(state),
                replay_ids=tuple(
                    dict.fromkeys(str(value) for value in replay_ids if str(value))
                ),
                updated_at=utc_now(),
                reason=str(reason),
                audit=dict(audit or {}),
            )
            self._write_once(
                self._evidence_path(memory_id, identity.current_version, revision),
                evidence.to_dict(),
            )
            updated = MemoryIdentity(
                **{
                    **identity.to_dict(),
                    "current_evidence_revision": revision,
                    "updated_at": evidence.updated_at,
                }
            )
            _atomic_write_json(self._identity_path(memory_id), updated.to_dict())
            self.refresh_memory_report(memory_id)
            return evidence

    def commit_version(
        self,
        *,
        memory_id: str,
        expected_parent_version: int,
        body: str,
        meta: MetaMemory,
        capabilities: Iterable[CapabilityConstraint],
        source_task_id: str,
        replay_ids: Iterable[str],
        source_trajectory_ref: str = "",
        source_event_ids: Iterable[str] = (),
        state: MemoryState = MemoryState.PROVISIONAL,
        decision: str = "accepted",
        audit: Optional[Mapping[str, Any]] = None,
    ) -> MemoryVersion:
        _validate_single_role_meta(meta, context="committed memory version")
        with self._exclusive():
            identity = self.get_identity(memory_id)
            if not identity:
                raise KeyError(memory_id)
            try:
                assert_linear_parent(
                    current_version=identity.current_version,
                    expected_parent=expected_parent_version,
                )
            except ValueError as exc:
                raise ConcurrentUpdateError(str(exc)) from exc
            next_version = identity.current_version + 1
            now = utc_now()
            base = VersionBase(
                memory_id=memory_id,
                version=next_version,
                body=str(body).strip(),
                parent_version=identity.current_version,
                source_task_id=str(source_task_id),
                created_at=now,
                source_trajectory_ref=str(source_trajectory_ref or ""),
                source_event_ids=tuple(dict.fromkeys(str(value) for value in source_event_ids if str(value))),
                decision=str(decision),
                audit=dict(audit or {}),
            )
            evidence = VersionEvidence(
                memory_id=memory_id,
                version=next_version,
                revision=0,
                meta=meta,
                capabilities=tuple(capabilities),
                state=MemoryState(state),
                replay_ids=tuple(
                    dict.fromkeys(str(value) for value in replay_ids if str(value))
                ),
                updated_at=now,
                reason="version_commit",
                audit=dict(audit or {}),
            )
            if not base.body:
                raise ValueError("memory body cannot be empty")
            self._write_once(self._base_path(memory_id, next_version), base.to_dict())
            self._write_once(
                self._evidence_path(memory_id, next_version, 0), evidence.to_dict()
            )
            updated = MemoryIdentity(
                **{
                    **identity.to_dict(),
                    "current_version": next_version,
                    "current_evidence_revision": 0,
                    "updated_at": now,
                }
            )
            _atomic_write_json(self._identity_path(memory_id), updated.to_dict())
            self.refresh_memory_report(memory_id)
            return MemoryVersion(base=base, evidence=evidence)

    def freeze_validation_plan(self, plan: ValidationPlan) -> bool:
        path = self.root / "validation_plans" / f"{safe_component(plan.plan_id)}.json"
        with self._exclusive():
            return self._write_once(path, plan.to_dict())

    def get_validation_plan(self, plan_id: str) -> Optional[ValidationPlan]:
        path = self.root / "validation_plans" / f"{safe_component(plan_id)}.json"
        return ValidationPlan.from_dict(read_json(path)) if path.exists() else None

    def record_replay(self, record: ReplayRecord) -> bool:
        if record.validation_plan_id:
            plan = self.get_validation_plan(record.validation_plan_id)
            if not plan:
                raise ImmutableRecordError(
                    f"replay references unknown validation plan {record.validation_plan_id}"
                )
            labels = {item.task_id: item.label for item in plan.assignments}
            if labels.get(record.task_id) != record.h_label:
                raise ImmutableRecordError(
                    "replay H label differs from its frozen validation plan"
                )
        path = (
            self.root
            / "replays"
            / "results"
            / f"{safe_component(record.replay_id)}.json"
        )
        with self._exclusive():
            return self._write_once(path, record.to_dict())

    def get_replay(self, replay_id: str) -> Optional[ReplayRecord]:
        path = self.root / "replays" / "results" / f"{safe_component(replay_id)}.json"
        return ReplayRecord.from_dict(read_json(path)) if path.exists() else None

    def iter_replays(self) -> list[ReplayRecord]:
        return [
            ReplayRecord.from_dict(read_json(path))
            for path in sorted((self.root / "replays" / "results").glob("*.json"))
        ]

    def record_task(self, task: TaskEvidence) -> bool:
        self.config.split_manifest.assert_learning_allowed(task.task_id)
        path = self.root / "tasks" / f"{safe_component(task.task_id)}.json"
        with self._exclusive():
            return self._write_once(path, task.to_dict())

    def get_task(self, task_id: str) -> Optional[TaskEvidence]:
        path = self.root / "tasks" / f"{safe_component(task_id)}.json"
        return TaskEvidence.from_dict(read_json(path)) if path.exists() else None

    def record_candidate(self, candidate: RepairCandidate) -> bool:
        path = (
            self.root
            / "repair_candidates"
            / f"{safe_component(candidate.candidate_id)}.json"
        )
        with self._exclusive():
            return self._write_once(path, candidate.to_dict())

    def get_candidate(self, candidate_id: str) -> Optional[RepairCandidate]:
        path = self.root / "repair_candidates" / f"{safe_component(candidate_id)}.json"
        return RepairCandidate.from_dict(read_json(path)) if path.exists() else None

    def record_admission(self, admission: AdmissionRecord) -> bool:
        path = (
            self.root / "admissions" / f"{safe_component(admission.admission_id)}.json"
        )
        with self._exclusive():
            return self._write_once(path, admission.to_dict())

    def record_event(
        self,
        event_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        status: str = "completed",
    ) -> bool:
        path = self.root / "events" / f"{safe_component(event_id)}.json"
        record = {
            "event_id": event_id,
            "event_type": event_type,
            "status": status,
            "payload": dict(payload),
            "recorded_at": utc_now(),
        }
        with self._exclusive():
            return self._write_once(path, record)

