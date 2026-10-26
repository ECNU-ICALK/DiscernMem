from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .effect import EffectAggregation, role_effect_class
from .invariants import MemoryState


@dataclass(frozen=True)
class StateEvidence:
    aggregation: EffectAggregation
    h1_task_ids: tuple[str, ...] = ()
    h2_task_ids: tuple[str, ...] = ()
    all_receiving_roles: tuple[str, ...] = ()
    deployment_boundary_explains_negative: bool = False
    has_content_counterexample: bool = False
    content_repair_attempted: bool = False
    content_repair_failed: bool = False
    h3_only_update: bool = False
    shadow_role_task_deltas: Mapping[str, Mapping[str, float]] = None
    behavior_changed_task_ids_by_role: Mapping[str, tuple[str, ...]] = None
    accepted_repair_version: bool = False


@dataclass(frozen=True)
class StateDecision:
    previous: MemoryState
    next: MemoryState
    reason: str

    @property
    def changed(self) -> bool:
        return self.previous is not self.next


_ALLOWED = {
    MemoryState.PROVISIONAL: {
        MemoryState.PROVISIONAL,
        MemoryState.ACTIVE,
        MemoryState.INACTIVE,
    },
    MemoryState.ACTIVE: {MemoryState.ACTIVE, MemoryState.INACTIVE},
    MemoryState.INACTIVE: {MemoryState.INACTIVE, MemoryState.PROVISIONAL},
}


def assert_state_transition(previous: MemoryState, next_state: MemoryState) -> None:
    if next_state not in _ALLOWED[previous]:
        raise ValueError(
            f"illegal DiscernMem state transition: {previous.value}->{next_state.value}"
        )


class DiscernMemStateEngine:
    def __init__(
        self,
        *,
        epsilon: float,
        min_stable_tasks: int,
        max_provisional_neutral_tasks: int = 5,
    ):
        self.epsilon = float(epsilon)
        self.min_stable_tasks = int(min_stable_tasks)
        self.max_provisional_neutral_tasks = int(max_provisional_neutral_tasks)

    def _classes(self, aggregation: EffectAggregation) -> dict[str, str]:
        return {
            role: role_effect_class(
                stats,
                epsilon=self.epsilon,
                min_tasks=self.min_stable_tasks,
            )
            for role, stats in aggregation.role_stats.items()
        }

    def _sustained_negative_roles(self, aggregation: EffectAggregation) -> set[str]:
        return {
            role
            for role, stats in aggregation.role_stats.items()
            if stats.n_tasks >= max(3, self.min_stable_tasks)
            and stats.mean_delta is not None
            and stats.mean_delta < -self.epsilon
            and stats.negative_count >= 2
            and stats.positive_count == 0
        }

    def _recent_negative_roles(self, aggregation: EffectAggregation) -> set[str]:
        roles: set[str] = set()
        for role, stats in aggregation.role_stats.items():
            recent = list(stats.task_deltas.values())[-3:]
            if len(recent) == 3 and all(
                float(delta) < -self.epsilon for delta in recent
            ):
                roles.add(role)
        return roles

    def _neutral_trial_exhausted(
        self, evidence: StateEvidence, receiving: set[str]
    ) -> bool:
        changed = evidence.behavior_changed_task_ids_by_role or {}
        if not receiving or not all(
            len(set(changed.get(role, ()))) >= self.max_provisional_neutral_tasks
            for role in receiving
        ):
            return False
        for role in receiving:
            stats = evidence.aggregation.role_stats.get(role)
            if stats is None or not stats.task_deltas:
                return False
            if any(
                abs(float(delta)) > self.epsilon for delta in stats.task_deltas.values()
            ):
                return False
        return True

    def decide(self, previous: MemoryState, evidence: StateEvidence) -> StateDecision:
        previous = MemoryState(previous)
        if evidence.h3_only_update:
            return StateDecision(
                previous, previous, "H3 boundary evidence cannot change state"
            )
        classes = self._classes(evidence.aggregation)
        stable_positive = {
            role for role, label in classes.items() if label == "stable_positive"
        }
        sustained_negative = self._sustained_negative_roles(evidence.aggregation)

        if previous is MemoryState.INACTIVE:
            if evidence.accepted_repair_version:
                return StateDecision(
                    previous,
                    MemoryState.PROVISIONAL,
                    "accepted repair re-enters optimistic validation",
                )
            shadow = evidence.shadow_role_task_deltas or {}
            positive_shadow = any(
                any(float(delta) > self.epsilon for delta in task_deltas.values())
                for task_deltas in shadow.values()
            )
            if positive_shadow:
                return StateDecision(
                    previous,
                    MemoryState.PROVISIONAL,
                    "independent shadow pair has positive benefit",
                )
            return StateDecision(
                previous, previous, "shadow evidence is absent, neutral, or negative"
            )

        if previous is MemoryState.PROVISIONAL:
            receiving = set(evidence.all_receiving_roles)
            if (
                len(receiving) == 1
                and evidence.has_content_counterexample
                and evidence.content_repair_attempted
                and evidence.content_repair_failed
                and not evidence.deployment_boundary_explains_negative
            ):
                return StateDecision(
                    previous,
                    MemoryState.INACTIVE,
                    "verified single-role content counterexample could not be repaired",
                )
            if stable_positive:
                return StateDecision(
                    previous,
                    MemoryState.ACTIVE,
                    "independent paired tasks show stable positive effect",
                )
            all_negative = bool(receiving) and receiving.issubset(sustained_negative)
            repair_gate = not evidence.has_content_counterexample or (
                evidence.content_repair_attempted and evidence.content_repair_failed
            )
            if (
                all_negative
                and not evidence.deployment_boundary_explains_negative
                and repair_gate
            ):
                return StateDecision(
                    previous,
                    MemoryState.INACTIVE,
                    "all receiving roles are stably negative and repair cannot recover content",
                )
            if self._neutral_trial_exhausted(evidence, receiving):
                return StateDecision(
                    previous,
                    MemoryState.INACTIVE,
                    "behavior-changing neutral trials reached the provisional limit",
                )
            return StateDecision(
                previous,
                previous,
                "evidence is insufficient, neutral, conflicting, or still repairable",
            )

        # Active memories retain deployment while any supported role remains positive.
        receiving = set(evidence.all_receiving_roles)
        if (
            len(receiving) == 1
            and evidence.has_content_counterexample
            and evidence.content_repair_attempted
            and evidence.content_repair_failed
            and not evidence.deployment_boundary_explains_negative
        ):
            return StateDecision(
                previous,
                MemoryState.INACTIVE,
                "verified single-role content counterexample could not be repaired",
            )
        if stable_positive:
            return StateDecision(
                previous, previous, "at least one role retains stable positive effect"
            )
        if (
            receiving
            and receiving.issubset(self._recent_negative_roles(evidence.aggregation))
            and not evidence.deployment_boundary_explains_negative
        ):
            return StateDecision(
                previous,
                MemoryState.INACTIVE,
                "all formerly supported roles are stably negative in applicable states",
            )
        return StateDecision(
            previous,
            previous,
            "single, boundary, neutral, or conflicting negative evidence cannot deactivate",
        )


def new_memory_state() -> MemoryState:
    return MemoryState.PROVISIONAL


def repaired_version_state() -> MemoryState:
    return MemoryState.PROVISIONAL
