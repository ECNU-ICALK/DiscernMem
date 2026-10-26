from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import math
from typing import Any, Callable, Iterable, Mapping, Optional

from .config import TaskPartition
from .ids import stable_id
from .invariants import InvariantViolation, require_single_role
from .schema import (
    CapabilityConstraint,
    HLabel,
    MetaMemory,
    ReplayCondition,
    ReplayStatus,
    ValidationAssignment,
    ValidationPlan,
)
from .storage import DiscernMemStore, read_json, safe_component, utc_now


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalTaskSummary:
    task_id: str
    partition: TaskPartition
    mas_type: str
    observable_trigger: str
    key_operation: str
    expected_state_change: str
    surface_summary: str
    task_index: int = -1
    source_entities: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HistoricalTaskSummary":
        return cls(
            task_id=str(data.get("task_id") or ""),
            partition=TaskPartition(data.get("partition")),
            mas_type=str(data.get("mas_type") or ""),
            observable_trigger=str(data.get("observable_trigger") or ""),
            key_operation=str(data.get("key_operation") or ""),
            expected_state_change=str(data.get("expected_state_change") or ""),
            surface_summary=str(data.get("surface_summary") or ""),
            task_index=int(data.get("task_index", -1)),
            source_entities=tuple(
                str(value) for value in data.get("source_entities", [])
            ),
        )

    def semantic_text(self) -> str:
        return " ".join(
            (
                self.observable_trigger,
                self.key_operation,
                self.expected_state_change,
                self.surface_summary,
            )
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "partition": self.partition.value,
            "source_entities": list(self.source_entities),
        }


@dataclass(frozen=True)
class ValidationPlanningResult:
    plan: Optional[ValidationPlan]
    missing_evidence: tuple[str, ...]
    recalled_task_ids: tuple[str, ...]


EmbeddingFn = Callable[[str], list[float]]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


class ValidationTaskIndex:
    def __init__(self, store: DiscernMemStore, embed: EmbeddingFn, *, model_name: str):
        self.store = store
        self.embed = embed
        self.model_name = model_name

    def _cached_vector(self, path, *, digest: str) -> list[float]:
        payload = read_json(path)
        if payload.get("model_name") != self.model_name:
            raise ValueError("validation task embedding model mismatch")
        if payload.get("text_sha256") != digest:
            raise ValueError("validation task embedding text mismatch")
        vector = [float(value) for value in payload["vector"]]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("cached validation embedding is empty or non-finite")
        return vector

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = (
            self.store.root
            / "embeddings"
            / "validation_tasks"
            / f"{safe_component(digest)}.json"
        )
        if path.exists():
            return self._cached_vector(path, digest=digest)
        vector = [float(value) for value in self.embed(text)]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("validation embedding is empty or non-finite")
        payload = {
            "namespace": "discernmem",
            "model_name": self.model_name,
            "text_sha256": digest,
            "vector": vector,
        }
        with self.store._exclusive():
            # Another formal-use group may have filled this cache after our
            # initial existence check. The first completed embedding becomes
            # canonical; concurrent callers reuse it instead of comparing
            # independently generated floating-point vectors as immutable data.
            if path.exists():
                return self._cached_vector(path, digest=digest)
            self.store._write_once(path, payload)
        return vector

    def retrieve(
        self,
        query: str,
        tasks: Iterable[HistoricalTaskSummary],
        *,
        top_k: int,
    ) -> list[tuple[HistoricalTaskSummary, float]]:
        query_vector = self._vector(query)
        scored = [
            (task, _cosine(query_vector, self._vector(task.semantic_text())))
            for task in tasks
        ]
        scored.sort(key=lambda item: (-item[1], item[0].task_id))
        return scored[: max(int(top_k), 0)]


class DiscernMemValidationPlanner:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        task_index: ValidationTaskIndex,
        recall_top_k: int = 12,
    ):
        self.store = store
        self.task_index = task_index
        self.recall_top_k = max(int(recall_top_k), 1)

    def _sample_retention_tasks(
        self,
        *,
        task_ids: Iterable[str],
        limit: int,
        trigger_task_id: str,
        memory_id: str,
        version: int,
        purpose: str,
    ) -> tuple[tuple[str, ...], str]:
        pool = tuple(sorted({str(task_id) for task_id in task_ids if str(task_id)}))
        if not pool or limit <= 0:
            return (), ""
        sampling_key = stable_id(
            "h1sample",
            {
                "config": self.store.config.fingerprint,
                "memory_id": memory_id,
                "parent_version": int(version),
                "trigger_task_id": trigger_task_id,
                "purpose": purpose,
            },
        )
        ranked = sorted(
            pool,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{sampling_key}\0{task_id}".encode("utf-8")
                ).hexdigest(),
                task_id,
            ),
        )
        return tuple(ranked[: min(int(limit), len(ranked))]), sampling_key

    def plan(
        self,
        *,
        trigger_task_id: str,
        trigger_task_index: int = -1,
        memory_id: str,
        version: int,
        body: str,
        meta: MetaMemory,
        capabilities: Iterable[CapabilityConstraint],
        problem_summary: str,
        roles: Iterable[str],
        source_task_ids: Iterable[str],
        historical_tasks: Iterable[HistoricalTaskSummary],
        purpose: str,
        strict_h2_roles: Iterable[str] = (),
    ) -> ValidationPlanningResult:
        owner_role = require_single_role(
            roles, context="validation plan memory"
        )
        source_ids = tuple(dict.fromkeys(str(value) for value in source_task_ids))
        trigger_task_index = int(trigger_task_index)
        final_ids = self.store.config.split_manifest.final_test_task_ids
        forbidden = set(source_ids) | {str(trigger_task_id)} | set(final_ids)
        eligible = []
        for task in historical_tasks:
            if task.task_id in forbidden:
                continue
            manifest_partition = self.store.config.split_manifest.partition_of(
                task.task_id
            )
            self.store.config.split_manifest.assert_history_eligible(task.task_id)
            if task.partition is not manifest_partition:
                raise ValueError(
                    f"historical task {task.task_id} partition differs from frozen manifest"
                )
            if trigger_task_index >= 0:
                if task.task_index < 0:
                    continue
                if task.task_index >= trigger_task_index:
                    continue
            eligible.append(task)
        memory = self.store.get_version(memory_id, version)
        capability_tuple = tuple(capabilities)
        query = " ".join(
            (
                body,
                *meta.use_when,
                *meta.avoid_when,
                *(capability.claim for capability in capability_tuple),
                problem_summary,
            )
        )
        eligible_by_id = {task.task_id: task for task in eligible}
        replay_task_by_id = {
            record.replay_id: record.task_id for record in self.store.iter_replays()
        }
        capabilities_by_task: dict[str, set[str]] = {}
        for capability in capability_tuple:
            for replay_id in capability.replay_ids:
                task_id = replay_task_by_id.get(replay_id, "")
                if task_id in eligible_by_id:
                    capabilities_by_task.setdefault(task_id, set()).add(
                        capability.capability_id
                    )
        if memory is not None:
            for item in memory.evidence.audit.get("qualified_positive_states", []):
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("role") or "") != owner_role:
                    continue
                task_id = str(item.get("task_id") or "")
                if task_id in eligible_by_id:
                    capabilities_by_task.setdefault(task_id, set())

        retention_pool_task_ids = tuple(sorted(capabilities_by_task))
        retention_limit = (
            self.store.config.h1_max_tasks
            if self.store.config.enable_h1_validation
            else 0
        )
        sampled_retention_ids, retention_sampling_key = self._sample_retention_tasks(
            task_ids=retention_pool_task_ids,
            limit=retention_limit,
            trigger_task_id=trigger_task_id,
            memory_id=memory_id,
            version=version,
            purpose=purpose,
        )
        assignments: list[ValidationAssignment] = []
        for task_id in sampled_retention_ids:
            capability_ids = tuple(sorted(capabilities_by_task[task_id]))
            assignments.append(
                ValidationAssignment(
                    task_id=task_id,
                    label=HLabel.H1,
                    capability_ids=capability_ids,
                    selection_reason=(
                        "seeded sample from verified capability retention pool"
                        if capability_ids
                        else "seeded sample from verified positive-use retention pool"
                    ),
                )
            )

        transfer_candidates = [
            task for task in eligible if task.task_id not in capabilities_by_task
        ]
        recalled = self.task_index.retrieve(
            query, transfer_candidates, top_k=self.recall_top_k
        )
        recalled_ids = tuple(task.task_id for task, _ in recalled)
        strict_h2_roles = tuple(
            dict.fromkeys(str(value) for value in strict_h2_roles if str(value))
        )
        if any(role != owner_role for role in strict_h2_roles):
            raise InvariantViolation(
                "strict H2 validation crosses memory role namespaces"
            )
        strict_h2_eligible = {
            task.task_id: not self._known_saturated_strict_h2(
                task_id=task.task_id,
                memory_id=memory_id,
                version=version,
                roles=strict_h2_roles,
            )
            for task, _ in recalled
        }

        transfer_limit = (
            self.store.config.h2_tasks if self.store.config.enable_h2_validation else 0
        )
        for task, _score in recalled:
            if len([a for a in assignments if a.label is HLabel.H2]) >= transfer_limit:
                break
            if not strict_h2_eligible.get(task.task_id, True):
                continue
            assignments.append(
                ValidationAssignment(
                    task_id=task.task_id,
                    label=HLabel.H2,
                    capability_ids=(),
                    selection_reason="nearest eligible historical transfer probe",
                )
            )
        selected_capability_ids = {
            capability_id
            for assignment in assignments
            if assignment.label is HLabel.H1
            for capability_id in assignment.capability_ids
        }
        plan_id = stable_id(
            "plan",
            {
                "trigger": trigger_task_id,
                "trigger_task_index": trigger_task_index,
                "memory": memory_id,
                "version": version,
                "role": owner_role,
                "purpose": purpose,
                "assignments": [item.to_dict() for item in assignments],
                "retention_pool_task_ids": list(retention_pool_task_ids),
                "retention_sampling_key": retention_sampling_key,
                "config": self.store.config.fingerprint,
            },
        )
        existing = self.store.get_validation_plan(plan_id)
        if existing is not None:
            return ValidationPlanningResult(
                plan=existing,
                missing_evidence=(),
                recalled_task_ids=recalled_ids,
            )
        plan = ValidationPlan(
            plan_id=plan_id,
            trigger_task_id=trigger_task_id,
            memory_id=memory_id,
            version=int(version),
            assignments=tuple(assignments),
            roles=(owner_role,),
            selected_capability_ids=tuple(sorted(selected_capability_ids)),
            config_fingerprint=self.store.config.fingerprint,
            frozen_at=utc_now(),
            source_task_ids=source_ids,
            final_test_task_ids=final_ids,
            purpose=purpose,
            trigger_task_index=trigger_task_index,
            assignment_task_indices={
                item.task_id: next(
                    task.task_index for task in eligible if task.task_id == item.task_id
                )
                for item in assignments
            },
            retention_pool_task_ids=retention_pool_task_ids,
            retention_sampling_key=retention_sampling_key,
        )
        self.store.freeze_validation_plan(plan)
        return ValidationPlanningResult(
            plan=plan,
            missing_evidence=(),
            recalled_task_ids=recalled_ids,
        )

    def _known_saturated_strict_h2(
        self,
        *,
        task_id: str,
        memory_id: str,
        version: int,
        roles: tuple[str, ...],
    ) -> bool:
        if not roles:
            return False
        baselines = {}
        versions = {}
        for record in self.store.iter_replays():
            if (
                record.task_id != task_id
                or record.status is not ReplayStatus.SUCCEEDED
                or record.score is None
            ):
                continue
            key = (
                record.task_id,
                record.seed,
                record.config_fingerprint,
                record.runtime_signature,
            )
            if record.condition is ReplayCondition.NO_MEMORY:
                baselines[key] = float(record.score)
            elif (
                record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE
                and record.memory_id == memory_id
                and record.version == int(version)
                and record.role in roles
            ):
                versions[(*key, record.role)] = float(record.score)
        for role in roles:
            paired = [
                (baseline, versions[(*key, role)])
                for key, baseline in baselines.items()
                if (*key, role) in versions
            ]
            if not paired or any(
                baseline < 1.0 - 1e-12 or old_version < 1.0 - 1e-12
                for baseline, old_version in paired
            ):
                return False
        return True
