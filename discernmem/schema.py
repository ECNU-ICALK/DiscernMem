from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from .invariants import MemoryState


def _strings(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


class HLabel(str, Enum):
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


class ReplayCondition(str, Enum):
    NO_MEMORY = "NO_MEMORY"
    ONLY_VERSION_FOR_ROLE = "ONLY_VERSION_FOR_ROLE"
    ONLY_CANDIDATE_FOR_ROLE = "ONLY_CANDIDATE_FOR_ROLE"
    NORMAL_RETRIEVAL = "NORMAL_RETRIEVAL"


class ReplayStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AdmissionDecision(str, Enum):
    DISCARD = "discard"
    ATTACH = "attach"
    CREATE = "create"


class EditOperation(str, Enum):
    ADD = "ADD"
    REPLACE = "REPLACE"
    DELETE = "DELETE"
    REORDER = "REORDER"
    NO_UPDATE = "NO_UPDATE"


class CandidateDecision(str, Enum):
    PENDING = "pending"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


@dataclass(frozen=True)
class RoleEffectStats:
    role: str
    n_tasks: int = 0
    mean_delta: Optional[float] = None
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    task_deltas: Mapping[str, float] = field(default_factory=dict)
    replay_ids: tuple[str, ...] = ()

    @property
    def tested(self) -> bool:
        return self.n_tasks > 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoleEffectStats":
        mean = data.get("mean_delta")
        return cls(
            role=str(data.get("role") or ""),
            n_tasks=int(data.get("n_tasks", 0)),
            mean_delta=None if mean is None else float(mean),
            positive_count=int(data.get("positive_count", 0)),
            negative_count=int(data.get("negative_count", 0)),
            neutral_count=int(data.get("neutral_count", 0)),
            task_deltas={
                str(key): float(value)
                for key, value in (data.get("task_deltas") or {}).items()
            },
            replay_ids=_strings(data.get("replay_ids")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_deltas"] = dict(self.task_deltas)
        payload["replay_ids"] = list(self.replay_ids)
        payload["tested"] = self.tested
        return payload


@dataclass(frozen=True)
class MetaUsageEvent:
    event_id: str
    task_id: str
    memory_id: str
    version: int
    role: str
    retrieval_ids: tuple[str, ...] = ()
    use_when_matches: Optional[bool] = None
    avoid_when_matches: Optional[bool] = None
    isolated_delta: Optional[float] = None
    behavior_changed: Optional[bool] = None
    routing: str = ""
    routing_reason: str = ""
    status: str = ""
    replay_ids: tuple[str, ...] = ()
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetaUsageEvent":
        return cls(
            event_id=str(data.get("event_id") or ""),
            task_id=str(data.get("task_id") or ""),
            memory_id=str(data.get("memory_id") or ""),
            version=int(data.get("version", 0)),
            role=str(data.get("role") or ""),
            retrieval_ids=_strings(data.get("retrieval_ids")),
            use_when_matches=(
                None if data.get("use_when_matches") is None else bool(data.get("use_when_matches"))
            ),
            avoid_when_matches=(
                None if data.get("avoid_when_matches") is None else bool(data.get("avoid_when_matches"))
            ),
            isolated_delta=(
                None if data.get("isolated_delta") is None else float(data.get("isolated_delta"))
            ),
            behavior_changed=(
                None if data.get("behavior_changed") is None else bool(data.get("behavior_changed"))
            ),
            routing=str(data.get("routing") or ""),
            routing_reason=str(data.get("routing_reason") or ""),
            status=str(data.get("status") or ""),
            replay_ids=_strings(data.get("replay_ids")),
            created_at=str(data.get("created_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieval_ids"] = list(self.retrieval_ids)
        payload["replay_ids"] = list(self.replay_ids)
        return payload


@dataclass(frozen=True)
class MetaMemory:
    roles: tuple[str, ...] = ()
    use_when: tuple[str, ...] = ()
    avoid_when: tuple[str, ...] = ()
    effect: Mapping[str, RoleEffectStats] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MetaMemory":
        data = data or {}
        effects = {
            str(role): RoleEffectStats.from_dict({"role": role, **dict(stats)})
            for role, stats in (data.get("effect") or {}).items()
        }
        return cls(
            roles=_strings(data.get("roles")),
            use_when=_strings(data.get("use_when")),
            avoid_when=_strings(data.get("avoid_when")),
            effect=effects,
        )

    @classmethod
    def deployment_from_llm(cls, data: Mapping[str, Any] | None) -> "MetaMemory":
        data = data or {}
        if "effect" in data:
            raise ValueError("LLM output cannot create or modify DiscernMem effect")
        return cls(
            roles=_strings(data.get("roles")),
            use_when=_strings(data.get("use_when")),
            avoid_when=_strings(data.get("avoid_when")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "effect": {role: stats.to_dict() for role, stats in self.effect.items()},
        }


@dataclass(frozen=True)
class CapabilityConstraint:
    capability_id: str
    claim: str
    introduced_version: int
    supported_roles: tuple[str, ...] = ()
    replay_ids: tuple[str, ...] = ()
    status: str = "supported"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityConstraint":
        return cls(
            capability_id=str(data.get("capability_id") or ""),
            claim=str(data.get("claim") or ""),
            introduced_version=int(data.get("introduced_version", 0)),
            supported_roles=_strings(data.get("supported_roles")),
            replay_ids=_strings(data.get("replay_ids")),
            status=str(data.get("status") or "supported"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supported_roles"] = list(self.supported_roles)
        payload["replay_ids"] = list(self.replay_ids)
        return payload


@dataclass(frozen=True)
class MemoryIdentity:
    memory_id: str
    current_version: int
    current_evidence_revision: int
    source_task_id: str
    created_at: str
    updated_at: str
    semantic_key: str = ""
    imported_from: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryIdentity":
        return cls(
            memory_id=str(data.get("memory_id") or ""),
            current_version=int(data.get("current_version", 0)),
            current_evidence_revision=int(data.get("current_evidence_revision", 0)),
            source_task_id=str(data.get("source_task_id") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            semantic_key=str(data.get("semantic_key") or ""),
            imported_from=(
                str(data["imported_from"]) if data.get("imported_from") else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VersionBase:
    memory_id: str
    version: int
    body: str
    parent_version: Optional[int]
    source_task_id: str
    created_at: str
    decision: str = "created"
    source_trajectory_ref: str = ""
    source_event_ids: tuple[str, ...] = ()
    audit: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionBase":
        parent = data.get("parent_version")
        return cls(
            memory_id=str(data.get("memory_id") or ""),
            version=int(data.get("version", 0)),
            body=str(data.get("memory") or ""),
            parent_version=None if parent is None else int(parent),
            source_task_id=str(data.get("source_task_id") or ""),
            created_at=str(data.get("created_at") or ""),
            source_trajectory_ref=str(data.get("source_trajectory_ref") or ""),
            source_event_ids=_strings(data.get("source_event_ids")),
            decision=str(data.get("decision") or "created"),
            audit=dict(data.get("audit") or {}),
        )

    @property
    def memory(self) -> str:
        return self.body

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory"] = payload.pop("body")
        payload["source_event_ids"] = list(self.source_event_ids)
        payload["audit"] = dict(self.audit)
        return payload


@dataclass(frozen=True)
class VersionEvidence:
    memory_id: str
    version: int
    revision: int
    meta: MetaMemory
    capabilities: tuple[CapabilityConstraint, ...]
    state: MemoryState
    replay_ids: tuple[str, ...] = ()
    updated_at: str = ""
    reason: str = "initial"
    audit: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionEvidence":
        return cls(
            memory_id=str(data.get("memory_id") or ""),
            version=int(data.get("version", 0)),
            revision=int(data.get("revision", 0)),
            meta=MetaMemory.from_dict(data.get("meta")),
            capabilities=tuple(
                CapabilityConstraint.from_dict(item)
                for item in data.get("capabilities", [])
            ),
            state=MemoryState(data.get("state", MemoryState.PROVISIONAL.value)),
            replay_ids=_strings(data.get("replay_ids")),
            updated_at=str(data.get("updated_at") or ""),
            reason=str(data.get("reason") or "initial"),
            audit=dict(data.get("audit") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "version": self.version,
            "revision": self.revision,
            "meta": self.meta.to_dict(),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "state": self.state.value,
            "replay_ids": list(self.replay_ids),
            "updated_at": self.updated_at,
            "reason": self.reason,
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class MemoryVersion:
    base: VersionBase
    evidence: VersionEvidence

    @property
    def memory_id(self) -> str:
        return self.base.memory_id

    @property
    def version(self) -> int:
        return self.base.version

    @property
    def memory(self) -> str:
        return self.base.memory

    @property
    def body(self) -> str:
        return self.memory

    @property
    def meta(self) -> MetaMemory:
        return self.evidence.meta

    @property
    def state(self) -> MemoryState:
        return self.evidence.state

    @property
    def capabilities(self) -> tuple[CapabilityConstraint, ...]:
        return self.evidence.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base.to_dict(), "evidence": self.evidence.to_dict()}


@dataclass(frozen=True)
class ReplayRecord:
    replay_id: str
    task_id: str
    seed: int
    condition: ReplayCondition
    role: str
    config_fingerprint: str
    runtime_signature: str
    status: ReplayStatus
    score: Optional[float] = None
    outcome: str = ""
    memory_id: str = ""
    version: Optional[int] = None
    candidate_id: str = ""
    validation_plan_id: str = ""
    h_label: Optional[HLabel] = None
    is_source_task: bool = False
    trace_ref: str = ""
    prompt_audit_refs: tuple[str, ...] = ()
    error: Optional[str] = None
    created_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition is ReplayCondition.NO_MEMORY:
            if self.memory_id or self.candidate_id:
                raise ValueError("NO_MEMORY replay cannot target a memory or candidate")
        elif self.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE:
            if (
                not self.memory_id
                or self.version is None
                or self.candidate_id
                or not self.role
            ):
                raise ValueError(
                    "version replay requires one memory version and one role"
                )
        elif self.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE:
            if not self.candidate_id or self.memory_id or not self.role:
                raise ValueError(
                    "candidate replay requires one temporary candidate and one role"
                )
        if self.status is ReplayStatus.SUCCEEDED and self.score is None:
            raise ValueError("successful replay requires a score")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayRecord":
        label = data.get("h_label")
        version = data.get("version")
        score = data.get("score")
        return cls(
            replay_id=str(data.get("replay_id") or ""),
            task_id=str(data.get("task_id") or ""),
            seed=int(data.get("seed", 0)),
            condition=ReplayCondition(data.get("condition")),
            role=str(data.get("role") or ""),
            config_fingerprint=str(data.get("config_fingerprint") or ""),
            runtime_signature=str(data.get("runtime_signature") or ""),
            status=ReplayStatus(data.get("status")),
            score=None if score is None else float(score),
            outcome=str(data.get("outcome") or ""),
            memory_id=str(data.get("memory_id") or ""),
            version=None if version is None else int(version),
            candidate_id=str(data.get("candidate_id") or ""),
            validation_plan_id=str(data.get("validation_plan_id") or ""),
            h_label=HLabel(label) if label else None,
            is_source_task=bool(data.get("is_source_task", False)),
            trace_ref=str(data.get("trace_ref") or ""),
            prompt_audit_refs=_strings(data.get("prompt_audit_refs")),
            error=(str(data["error"]) if data.get("error") else None),
            created_at=str(data.get("created_at") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "condition": self.condition.value,
            "status": self.status.value,
            "h_label": self.h_label.value if self.h_label else None,
            "prompt_audit_refs": list(self.prompt_audit_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationAssignment:
    task_id: str
    label: HLabel
    capability_ids: tuple[str, ...] = ()
    selection_reason: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationAssignment":
        return cls(
            task_id=str(data.get("task_id") or ""),
            label=HLabel(data.get("label")),
            capability_ids=_strings(data.get("capability_ids")),
            selection_reason=str(data.get("selection_reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label.value,
            "capability_ids": list(self.capability_ids),
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class ValidationPlan:
    plan_id: str
    trigger_task_id: str
    memory_id: str
    version: int
    assignments: tuple[ValidationAssignment, ...]
    roles: tuple[str, ...]
    selected_capability_ids: tuple[str, ...]
    config_fingerprint: str
    frozen_at: str
    source_task_ids: tuple[str, ...] = ()
    final_test_task_ids: tuple[str, ...] = ()
    purpose: str = "repair_validation"
    trigger_task_index: int = -1
    assignment_task_indices: Mapping[str, int] = field(default_factory=dict)
    retention_pool_task_ids: tuple[str, ...] = ()
    retention_sampling_key: str = ""

    def __post_init__(self) -> None:
        task_ids = [item.task_id for item in self.assignments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "a validation task can have only one frozen H label per plan"
            )
        retention_pool = set(self.retention_pool_task_ids)
        if retention_pool:
            invalid_h1 = {
                item.task_id
                for item in self.assignments
                if item.label is HLabel.H1 and item.task_id not in retention_pool
            }
            invalid_h2 = {
                item.task_id
                for item in self.assignments
                if item.label is HLabel.H2 and item.task_id in retention_pool
            }
            if invalid_h1 or invalid_h2:
                raise ValueError(
                    "validation assignments violate the frozen retention pool: "
                    f"H1 outside={sorted(invalid_h1)}, H2 inside={sorted(invalid_h2)}"
                )
        forbidden = (
            set(self.source_task_ids)
            | {self.trigger_task_id}
            | set(self.final_test_task_ids)
        )
        leaked = forbidden & set(task_ids)
        if leaked:
            raise ValueError(
                f"validation plan contains forbidden tasks: {sorted(leaked)}"
            )
        if self.trigger_task_index >= 0:
            assignment_indices = {
                str(task_id): int(task_index)
                for task_id, task_index in self.assignment_task_indices.items()
            }
            missing = set(task_ids) - set(assignment_indices)
            if missing:
                raise ValueError(
                    "validation plan lacks task indices for past-only audit: "
                    f"{sorted(missing)}"
                )
            leaked_future = {
                task_id: assignment_indices[task_id]
                for task_id in task_ids
                if assignment_indices[task_id] < 0
                or assignment_indices[task_id] >= self.trigger_task_index
            }
            if leaked_future:
                raise ValueError(
                    "validation plan contains non-past tasks: "
                    f"{sorted(leaked_future.items())}; trigger_index={self.trigger_task_index}"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationPlan":
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            trigger_task_id=str(data.get("trigger_task_id") or ""),
            memory_id=str(data.get("memory_id") or ""),
            version=int(data.get("version", 0)),
            assignments=tuple(
                ValidationAssignment.from_dict(item)
                for item in data.get("assignments", [])
            ),
            roles=_strings(data.get("roles")),
            selected_capability_ids=_strings(data.get("selected_capability_ids")),
            config_fingerprint=str(data.get("config_fingerprint") or ""),
            frozen_at=str(data.get("frozen_at") or ""),
            source_task_ids=_strings(data.get("source_task_ids")),
            final_test_task_ids=_strings(data.get("final_test_task_ids")),
            purpose=str(data.get("purpose") or "repair_validation"),
            trigger_task_index=int(data.get("trigger_task_index", -1)),
            assignment_task_indices={
                str(task_id): int(task_index)
                for task_id, task_index in dict(
                    data.get("assignment_task_indices") or {}
                ).items()
            },
            retention_pool_task_ids=_strings(data.get("retention_pool_task_ids")),
            retention_sampling_key=str(data.get("retention_sampling_key") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "trigger_task_id": self.trigger_task_id,
            "memory_id": self.memory_id,
            "version": self.version,
            "assignments": [item.to_dict() for item in self.assignments],
            "roles": list(self.roles),
            "selected_capability_ids": list(self.selected_capability_ids),
            "config_fingerprint": self.config_fingerprint,
            "frozen_at": self.frozen_at,
            "source_task_ids": list(self.source_task_ids),
            "final_test_task_ids": list(self.final_test_task_ids),
            "purpose": self.purpose,
            "trigger_task_index": self.trigger_task_index,
            "assignment_task_indices": {
                str(task_id): int(task_index)
                for task_id, task_index in self.assignment_task_indices.items()
            },
            "retention_pool_task_ids": list(self.retention_pool_task_ids),
            "retention_sampling_key": self.retention_sampling_key,
        }


@dataclass(frozen=True)
class RepairCandidate:
    candidate_id: str
    memory_id: str
    parent_version: int
    diagnosis: str
    edit_operation: EditOperation
    edited_body: str
    evidence_refs: tuple[str, ...]
    claimed_preserved_capability_ids: tuple[str, ...]
    expected_change: str
    decision: CandidateDecision = CandidateDecision.PENDING
    created_at: str = ""
    audit: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepairCandidate":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            memory_id=str(data.get("memory_id") or ""),
            parent_version=int(data.get("parent_version", 0)),
            diagnosis=str(data.get("diagnosis") or ""),
            edit_operation=EditOperation(data.get("edit_operation")),
            edited_body=str(data.get("edited_memory") or ""),
            evidence_refs=_strings(data.get("evidence_refs")),
            claimed_preserved_capability_ids=_strings(
                data.get("claimed_preserved_capability_ids")
            ),
            expected_change=str(data.get("expected_change") or ""),
            decision=CandidateDecision(
                data.get("decision", CandidateDecision.PENDING.value)
            ),
            created_at=str(data.get("created_at") or ""),
            audit=dict(data.get("audit") or {}),
        )

    @property
    def edited_memory(self) -> str:
        return self.edited_body

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **asdict(self),
            "edit_operation": self.edit_operation.value,
            "decision": self.decision.value,
            "evidence_refs": list(self.evidence_refs),
            "claimed_preserved_capability_ids": list(
                self.claimed_preserved_capability_ids
            ),
            "audit": dict(self.audit),
        }
        payload["edited_memory"] = payload.pop("edited_body")
        return payload


@dataclass(frozen=True)
class AdmissionRecord:
    admission_id: str
    source_task_id: str
    candidate: Mapping[str, Any]
    decision: AdmissionDecision
    reason: str
    compared_memory_ids: tuple[str, ...] = ()
    target_memory_id: str = ""
    validation_status: str = "not_evaluated"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate": dict(self.candidate),
            "decision": self.decision.value,
            "compared_memory_ids": list(self.compared_memory_ids),
        }


@dataclass(frozen=True)
class TaskEvidence:
    task_id: str
    task_index: int
    partition: str
    task_description: str
    task_family: str
    mas_spec: Mapping[str, Any]
    outcome: str
    score: float
    evaluator_feedback: str
    trajectory_ref: str
    observable_state_refs: tuple[str, ...] = ()
    injected_memories: tuple[Mapping[str, Any], ...] = ()
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskEvidence":
        return cls(
            task_id=str(data.get("task_id") or ""),
            task_index=int(data.get("task_index", -1)),
            partition=str(data.get("partition") or ""),
            task_description=str(data.get("task_description") or ""),
            task_family=str(data.get("task_family") or ""),
            mas_spec=dict(data.get("mas_spec") or {}),
            outcome=str(data.get("outcome") or ""),
            score=float(data.get("score", 0.0)),
            evaluator_feedback=str(data.get("evaluator_feedback") or ""),
            trajectory_ref=str(data.get("trajectory_ref") or ""),
            observable_state_refs=tuple(
                str(value) for value in data.get("observable_state_refs", [])
            ),
            injected_memories=tuple(
                dict(item) for item in data.get("injected_memories", [])
            ),
            created_at=str(data.get("created_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "mas_spec": dict(self.mas_spec),
            "observable_state_refs": list(self.observable_state_refs),
            "injected_memories": [dict(item) for item in self.injected_memories],
        }
