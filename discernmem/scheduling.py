from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .counterexamples import ContentCounterexample
from .invariants import InvariantViolation, require_single_role
from .replay import ReplayRequest
from .schema import (
    HLabel,
    MemoryVersion,
    RepairCandidate,
    ReplayCondition,
    ValidationPlan,
)


@dataclass(frozen=True)
class ReplayTaskContext:
    task_id: str
    task_index: int
    runtime_signature: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayTaskContext":
        return cls(
            task_id=str(data.get("task_id") or ""),
            task_index=int(data.get("task_index", -1)),
            runtime_signature=str(data.get("runtime_signature") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_index": self.task_index,
            "runtime_signature": self.runtime_signature,
            "metadata": dict(self.metadata),
        }


class ReplayScheduler:
    def __init__(self, *, config_fingerprint: str, seeds: tuple[int, ...]):
        self.config_fingerprint = config_fingerprint
        self.seeds = tuple(int(value) for value in seeds)

    def formal_use_pair(
        self,
        *,
        version: MemoryVersion,
        role: str,
        task_context: ReplayTaskContext,
    ) -> tuple[ReplayRequest, ...]:
        owner_role = require_single_role(
            version.meta.roles, context="formal-use memory"
        )
        if role != owner_role:
            raise InvariantViolation(
                f"formal-use role {role!r} does not own memory role {owner_role!r}"
            )
        source_task = task_context.task_id == version.base.source_task_id
        requests = []
        for seed in self.seeds:
            requests.extend(
                [
                    ReplayRequest(
                        task_id=task_context.task_id,
                        task_index=task_context.task_index,
                        seed=seed,
                        condition=ReplayCondition.NO_MEMORY,
                        config_fingerprint=self.config_fingerprint,
                        runtime_signature=task_context.runtime_signature,
                        is_source_task=source_task,
                        trigger_task_id=task_context.task_id,
                        metadata=dict(task_context.metadata),
                    ),
                    ReplayRequest(
                        task_id=task_context.task_id,
                        task_index=task_context.task_index,
                        seed=seed,
                        condition=ReplayCondition.ONLY_VERSION_FOR_ROLE,
                        config_fingerprint=self.config_fingerprint,
                        runtime_signature=task_context.runtime_signature,
                        role=role,
                        memory_id=version.memory_id,
                        version=version.version,
                        body=version.body,
                        meta=version.meta,
                        is_source_task=source_task,
                        trigger_task_id=task_context.task_id,
                        metadata=dict(task_context.metadata),
                    ),
                ]
            )
        return tuple(requests)

    def birth_comparison(self, *, parent: MemoryVersion, candidates: Iterable[RepairCandidate], role: str, validation_contexts: Iterable[ReplayTaskContext], trigger_task_id: str) -> tuple[ReplayRequest, ...]:
        owner_role = require_single_role(parent.meta.roles, context="birth CPE memory")
        if role != owner_role:
            raise InvariantViolation(f"birth CPE role {role!r} does not own memory role {owner_role!r}")
        requests = []
        for context in validation_contexts:
            for seed in self.seeds:
                common = dict(task_id=context.task_id, task_index=context.task_index, seed=seed, config_fingerprint=self.config_fingerprint, runtime_signature=context.runtime_signature, trigger_task_id=trigger_task_id, metadata=dict(context.metadata))
                requests.append(ReplayRequest(**common, condition=ReplayCondition.NO_MEMORY))
                requests.append(ReplayRequest(**common, condition=ReplayCondition.ONLY_VERSION_FOR_ROLE, role=role, memory_id=parent.memory_id, version=parent.version, body=parent.body, meta=parent.meta))
                for candidate in candidates:
                    requests.append(ReplayRequest(**common, condition=ReplayCondition.ONLY_CANDIDATE_FOR_ROLE, role=role, candidate_id=candidate.candidate_id, body=candidate.edited_body, meta=parent.meta))
        return tuple(requests)

    def repair_comparison(
        self,
        *,
        plan: ValidationPlan,
        parent: MemoryVersion,
        candidates: Iterable[RepairCandidate],
        counterexample: ContentCounterexample,
        problem_context: ReplayTaskContext,
        task_contexts: Mapping[str, ReplayTaskContext],
    ) -> tuple[ReplayRequest, ...]:
        candidates = tuple(candidates)
        owner_role = require_single_role(
            parent.meta.roles, context="repair parent memory"
        )
        if counterexample.role != owner_role:
            raise InvariantViolation(
                f"repair role {counterexample.role!r} does not own memory role "
                f"{owner_role!r}"
            )
        if any(role != owner_role for role in plan.roles):
            raise InvariantViolation(
                "repair validation plan crosses memory role namespaces"
            )
        tasks = [(problem_context, None)] + [
            (task_contexts[item.task_id], item.label)
            for item in plan.assignments
            if item.label in {HLabel.H1, HLabel.H2}
        ]
        requests = []
        for context, label in tasks:
            for seed in self.seeds:
                requests.append(
                    ReplayRequest(
                        task_id=context.task_id,
                        task_index=context.task_index,
                        seed=seed,
                        condition=ReplayCondition.NO_MEMORY,
                        config_fingerprint=self.config_fingerprint,
                        runtime_signature=context.runtime_signature,
                        validation_plan_id=plan.plan_id if label else "",
                        h_label=label,
                        trigger_task_id=counterexample.task_id,
                        metadata=dict(context.metadata),
                    )
                )
                for role in (owner_role,):
                    requests.append(
                        ReplayRequest(
                            task_id=context.task_id,
                            task_index=context.task_index,
                            seed=seed,
                            condition=ReplayCondition.ONLY_VERSION_FOR_ROLE,
                            config_fingerprint=self.config_fingerprint,
                            runtime_signature=context.runtime_signature,
                            role=role,
                            memory_id=parent.memory_id,
                            version=parent.version,
                            body=parent.body,
                            meta=parent.meta,
                            validation_plan_id=plan.plan_id if label else "",
                            h_label=label,
                            trigger_task_id=counterexample.task_id,
                            metadata=dict(context.metadata),
                        )
                    )
                    for candidate in candidates:
                        requests.append(
                            ReplayRequest(
                                task_id=context.task_id,
                                task_index=context.task_index,
                                seed=seed,
                                condition=ReplayCondition.ONLY_CANDIDATE_FOR_ROLE,
                                config_fingerprint=self.config_fingerprint,
                                runtime_signature=context.runtime_signature,
                                role=role,
                                candidate_id=candidate.candidate_id,
                                body=candidate.edited_body,
                                meta=parent.meta,
                                validation_plan_id=plan.plan_id if label else "",
                                h_label=label,
                                trigger_task_id=counterexample.task_id,
                                metadata=dict(context.metadata),
                            )
                        )
        return tuple(requests)
