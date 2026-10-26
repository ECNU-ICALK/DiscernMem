from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .counterexamples import CounterexampleRouting
from .effect import EffectAggregation, aggregate_effect
from .invariants import InvariantViolation, MemoryState, require_single_role
from .schema import AdmissionDecision, AdmissionRecord, MetaMemory, ReplayRecord, RoleEffectStats
from .state_engine import StateEvidence, DiscernMemStateEngine
from .storage import DiscernMemStore


@dataclass(frozen=True)
class OnlineEventResult:
    memory_id: str
    version: int
    evidence_revision: int
    state: MemoryState
    body_changed: bool
    routing: Optional[CounterexampleRouting] = None


def _merge_stats(
    historical: Mapping[str, RoleEffectStats],
    update: Mapping[str, RoleEffectStats],
    *,
    epsilon: float,
) -> dict[str, RoleEffectStats]:
    merged = {}
    for role in sorted(set(historical) | set(update)):
        task_deltas = dict(historical.get(role, RoleEffectStats(role)).task_deltas)
        for task_id, delta in update.get(role, RoleEffectStats(role)).task_deltas.items():
            if task_id in task_deltas and abs(task_deltas[task_id] - delta) > 1e-12:
                raise ValueError(f"conflicting effect for existing task {task_id}/{role}")
            task_deltas[task_id] = delta
        values = list(task_deltas.values())
        replay_ids = tuple(
            dict.fromkeys(
                (*historical.get(role, RoleEffectStats(role)).replay_ids,
                 *update.get(role, RoleEffectStats(role)).replay_ids)
            )
        )
        merged[role] = RoleEffectStats(
            role=role,
            n_tasks=len(values),
            mean_delta=sum(values) / len(values) if values else None,
            positive_count=sum(value > epsilon for value in values),
            negative_count=sum(value < -epsilon for value in values),
            neutral_count=sum(-epsilon <= value <= epsilon for value in values),
            task_deltas=task_deltas,
            replay_ids=replay_ids,
        )
    return merged


class DiscernMemOnlineEvents:
    def __init__(self, store: DiscernMemStore):
        self.store = store
        self.state_engine = DiscernMemStateEngine(
            epsilon=store.config.epsilon,
            min_stable_tasks=store.config.min_stable_tasks,
        )

    def attach(
        self,
        admission: AdmissionRecord,
        *,
        shadow_replays: Iterable[ReplayRecord] = (),
    ) -> OnlineEventResult:
        if admission.decision is not AdmissionDecision.ATTACH or not admission.target_memory_id:
            raise ValueError("attach event requires an attach admission target")
        current = self.store.get_version(admission.target_memory_id)
        if current is None:
            raise KeyError(admission.target_memory_id)
        owner_role = require_single_role(
            current.meta.roles, context="attached memory"
        )
        candidate_roles = admission.candidate.get("roles", [])
        candidate_role = require_single_role(
            candidate_roles, context="attached candidate"
        )
        if candidate_role != owner_role:
            raise InvariantViolation(
                f"candidate role {candidate_role!r} cannot attach to memory role "
                f"{owner_role!r}"
            )
        if current.state is not MemoryState.INACTIVE:
            return OnlineEventResult(
                memory_id=current.memory_id,
                version=current.version,
                evidence_revision=current.evidence.revision,
                state=current.state,
                body_changed=False,
            )
        shadow_replays = tuple(shadow_replays)
        if not shadow_replays:
            return OnlineEventResult(
                memory_id=current.memory_id,
                version=current.version,
                evidence_revision=current.evidence.revision,
                state=current.state,
                body_changed=False,
            )
        update = aggregate_effect(
            shadow_replays,
            memory_id=current.memory_id,
            version=current.version,
            epsilon=self.store.config.epsilon,
            min_tasks=1,
        )
        unexpected_roles = set(update.role_stats) - {owner_role}
        if unexpected_roles:
            raise InvariantViolation(
                "attach shadow replay crossed memory role namespace: "
                f"{sorted(unexpected_roles)!r}"
            )
        historical_stats = {}
        if owner_role in current.meta.effect:
            historical_stats[owner_role] = current.meta.effect[owner_role]
        merged_stats = _merge_stats(
            historical_stats,
            update.role_stats,
            epsilon=self.store.config.epsilon,
        )
        merged_aggregation = EffectAggregation(
            memory_id=current.memory_id,
            version=current.version,
            epsilon=self.store.config.epsilon,
            role_stats=merged_stats,
        )
        shadow_deltas = {
            role: dict(stats.task_deltas) for role, stats in update.role_stats.items()
        }
        decision = self.state_engine.decide(
            current.state,
            StateEvidence(
                aggregation=merged_aggregation,
                shadow_role_task_deltas=shadow_deltas,
            ),
        )
        meta = MetaMemory(
            roles=(owner_role,),
            use_when=current.meta.use_when,
            avoid_when=current.meta.avoid_when,
            effect=merged_stats,
        )
        evidence = self.store.update_evidence(
            memory_id=current.memory_id,
            expected_version=current.version,
            expected_revision=current.evidence.revision,
            meta=meta,
            capabilities=current.capabilities,
            state=decision.next,
            replay_ids=tuple(record.replay_id for record in shadow_replays),
            reason="inactive attach shadow validation",
            audit={"admission_id": admission.admission_id},
        )
        return OnlineEventResult(
            memory_id=current.memory_id,
            version=current.version,
            evidence_revision=evidence.revision,
            state=evidence.state,
            body_changed=False,
        )

