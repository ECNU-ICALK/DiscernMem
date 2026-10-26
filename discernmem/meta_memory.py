from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from .effect import (
    EffectAggregation,
    apply_controller_effect,
)
from .invariants import require_single_role
from .schema import HLabel, MetaMemory


@dataclass(frozen=True)
class PositiveStateEvidence:
    task_id: str
    role: str
    observable_state: str
    replay_ids: tuple[str, ...]
    source_entities: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PositiveStateEvidence":
        return cls(
            task_id=str(data.get("task_id") or ""),
            role=str(data.get("role") or ""),
            observable_state=str(data.get("observable_state") or ""),
            replay_ids=tuple(str(value) for value in data.get("replay_ids", [])),
            source_entities=tuple(
                str(value) for value in data.get("source_entities", [])
            ),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["replay_ids"] = list(self.replay_ids)
        payload["source_entities"] = list(self.source_entities)
        return payload


@dataclass(frozen=True)
class BoundaryEvidence:
    task_id: str
    role: str
    h_label: HLabel
    missing_key_condition: str
    delta: float
    caused_interference: bool = False
    expected_behavior_absent: bool = False
    explicit_state_mismatch: bool = False
    replay_ids: tuple[str, ...] = ()
    source_entities: tuple[str, ...] = ()

    def qualifies(self, epsilon: float) -> bool:
        return (
            self.h_label is HLabel.H3
            and bool(self.missing_key_condition.strip())
            and (
                self.delta < -epsilon
                or self.caused_interference
                or self.expected_behavior_absent
                or self.explicit_state_mismatch
            )
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["h_label"] = self.h_label.value
        payload["replay_ids"] = list(self.replay_ids)
        payload["source_entities"] = list(self.source_entities)
        return payload


@dataclass(frozen=True)
class MetaUpdateResult:
    meta: MetaMemory
    qualifying_boundary_task_ids: tuple[str, ...]
    rejected_boundary_task_ids: tuple[str, ...]


MetaSummarizer = Callable[..., Mapping[str, Any]]



def sanitize_meta_conditions(
    conditions: Iterable[str],
    *,
    source_entities: Iterable[str] = (),
    fallback: Iterable[str] = (),
) -> tuple[str, ...]:
    """Preserve replay-derived conditions without semantic rewriting."""
    del source_entities

    def preserve(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                value.strip()
                for value in (str(item) for item in values)
                if value.strip()
            )
        )

    proposed = preserve(conditions)
    return proposed or preserve(fallback)

def update_meta_memory(
    current: MetaMemory,
    *,
    aggregation: EffectAggregation,
    positive_states: Iterable[PositiveStateEvidence],
    boundaries: Iterable[BoundaryEvidence],
    summarizer: MetaSummarizer,
    min_stable_tasks: int,
) -> MetaUpdateResult:
    del min_stable_tasks
    owner_role = require_single_role(current.roles, context="meta memory update")
    positive_states = tuple(item for item in positive_states if item.role == owner_role)
    boundaries = tuple(item for item in boundaries if item.role == owner_role)
    qualifying = tuple(
        item for item in boundaries if item.qualifies(aggregation.epsilon)
    )
    rejected = tuple(item for item in boundaries if item not in qualifying)
    controller_meta = apply_controller_effect(current, aggregation)
    positive_task_ids = {item.task_id for item in positive_states}
    summary = {}
    if len(positive_task_ids) >= 2 or qualifying:
        summary = dict(
            summarizer(
                positive_states=[item.to_dict() for item in positive_states],
                qualifying_boundaries=[item.to_dict() for item in qualifying],
            )
        )
        forbidden = {"effect", "roles", "state", "status"} & set(summary)
        if forbidden:
            raise ValueError(
                f"meta summarizer attempted controller-owned fields: {sorted(forbidden)}"
            )
    source_entities = tuple(
        dict.fromkeys(
            entity
            for item in (*positive_states, *qualifying)
            for entity in item.source_entities
        )
    )
    if len(positive_task_ids) >= 2:
        use_when = sanitize_meta_conditions(
            (*current.use_when, *(str(value) for value in summary.get("use_when", []))),
            source_entities=source_entities,
            fallback=current.use_when,
        )
    else:
        use_when = current.use_when
    avoid_when = current.avoid_when
    if qualifying:
        avoid_when = sanitize_meta_conditions(
            (
                *current.avoid_when,
                *(str(value) for value in summary.get("avoid_when", [])),
            ),
            source_entities=source_entities,
            fallback=current.avoid_when,
        )
    return MetaUpdateResult(
        meta=MetaMemory(
            roles=(owner_role,),
            use_when=use_when,
            avoid_when=avoid_when,
            effect=controller_meta.effect,
        ),
        qualifying_boundary_task_ids=tuple(item.task_id for item in qualifying),
        rejected_boundary_task_ids=tuple(item.task_id for item in rejected),
    )
