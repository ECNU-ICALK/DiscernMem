from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from .invariants import InvariantViolation, require_single_role
from .schema import (
    MetaMemory,
    ReplayCondition,
    ReplayRecord,
    ReplayStatus,
    RoleEffectStats,
)


@dataclass(frozen=True)
class EffectAggregation:
    memory_id: str
    version: int
    epsilon: float
    role_stats: Mapping[str, RoleEffectStats]
    unknown_roles: tuple[str, ...] = ()
    duplicate_logical_replays: int = 0
    excluded_source_records: int = 0
    excluded_h3_records: int = 0
    missing_baselines: int = 0
    excluded_invalid_records: int = 0

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "version": self.version,
            "epsilon": self.epsilon,
            "role_stats": {
                role: stats.to_dict() for role, stats in self.role_stats.items()
            },
            "unknown_roles": list(self.unknown_roles),
            "duplicate_logical_replays": self.duplicate_logical_replays,
            "excluded_source_records": self.excluded_source_records,
            "excluded_h3_records": self.excluded_h3_records,
            "missing_baselines": self.missing_baselines,
            "excluded_invalid_records": self.excluded_invalid_records,
        }


def _logical_key(record: ReplayRecord) -> tuple:
    return (
        record.task_id,
        int(record.seed),
        record.config_fingerprint,
        record.runtime_signature,
    )


def _task_key(record: ReplayRecord) -> tuple:
    return (record.task_id, record.config_fingerprint, record.runtime_signature)


def role_effect_class(stats: RoleEffectStats, *, epsilon: float, min_tasks: int) -> str:
    if stats.n_tasks < int(min_tasks) or stats.mean_delta is None:
        return "unknown"
    if stats.mean_delta > epsilon and stats.negative_count == 0:
        return "stable_positive"
    if stats.mean_delta < -epsilon and stats.positive_count == 0:
        return "stable_negative"
    return "uncertain"


def aggregate_effect(
    records: Iterable[ReplayRecord],
    *,
    memory_id: str,
    version: int,
    epsilon: float,
    min_tasks: int = 2,
    include_h3: bool = False,
) -> EffectAggregation:
    """Compute controller-owned independent role effects.

    A baseline and target must share task, seed, runtime signature, and config.
    Seed observations are averaged within a task before tasks receive equal
    weight. H3 and source-task evidence never enter the main effect.
    """

    records = list(records)
    baselines: dict[tuple, ReplayRecord] = {}
    targets: dict[tuple[tuple, str], ReplayRecord] = {}
    all_target_roles: set[str] = set()
    duplicate_count = 0
    excluded_source = 0
    excluded_h3 = 0
    excluded_invalid = 0

    for record in records:
        if record.status is not ReplayStatus.SUCCEEDED or record.score is None:
            excluded_invalid += 1
            continue
        logical = _logical_key(record)
        if record.condition is ReplayCondition.NO_MEMORY:
            if logical in baselines:
                duplicate_count += 1
            else:
                baselines[logical] = record
            continue
        if record.condition is not ReplayCondition.ONLY_VERSION_FOR_ROLE:
            excluded_invalid += 1
            continue
        if (
            record.memory_id != memory_id
            or record.version != int(version)
            or not record.role
        ):
            continue
        all_target_roles.add(record.role)
        if record.is_source_task:
            excluded_source += 1
            continue
        if (
            record.h_label is not None
            and record.h_label.value == "H3"
            and not include_h3
        ):
            excluded_h3 += 1
            continue
        key = (logical, record.role)
        if key in targets:
            duplicate_count += 1
        else:
            targets[key] = record

    task_role_seed_deltas: dict[tuple[str, str, str, str], dict[int, float]] = {}
    replay_ids_by_task_role: dict[tuple[str, str, str, str], list[str]] = {}
    missing_baselines = 0
    for (logical, role), target in targets.items():
        baseline = baselines.get(logical)
        if baseline is None:
            missing_baselines += 1
            continue
        task_key = (*_task_key(target), role)
        task_role_seed_deltas.setdefault(task_key, {})[target.seed] = float(
            target.score
        ) - float(baseline.score)
        replay_ids_by_task_role.setdefault(task_key, []).extend(
            [target.replay_id, baseline.replay_id]
        )

    by_role: dict[str, dict[tuple, float]] = {}
    by_role_replays: dict[str, list[str]] = {}
    for (
        task_id,
        config_fp,
        runtime_fp,
        role,
    ), seed_deltas in task_role_seed_deltas.items():
        task_mean = sum(seed_deltas.values()) / len(seed_deltas)
        by_role.setdefault(role, {})[(task_id, config_fp, runtime_fp)] = task_mean
        by_role_replays.setdefault(role, []).extend(
            replay_ids_by_task_role[(task_id, config_fp, runtime_fp, role)]
        )

    stats: dict[str, RoleEffectStats] = {}
    for role in sorted(all_target_roles | set(by_role)):
        task_values = by_role.get(role, {})
        values = list(task_values.values())
        positive = sum(value > epsilon for value in values)
        negative = sum(value < -epsilon for value in values)
        neutral = len(values) - positive - negative
        stats[role] = RoleEffectStats(
            role=role,
            n_tasks=len(values),
            mean_delta=(sum(values) / len(values) if values else None),
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            task_deltas={
                task_id: value for (task_id, _, _), value in task_values.items()
            },
            replay_ids=tuple(dict.fromkeys(by_role_replays.get(role, []))),
        )

    return EffectAggregation(
        memory_id=memory_id,
        version=int(version),
        epsilon=float(epsilon),
        role_stats=stats,
        unknown_roles=tuple(
            role
            for role, role_stat in stats.items()
            if role_effect_class(role_stat, epsilon=epsilon, min_tasks=min_tasks)
            == "unknown"
        ),
        duplicate_logical_replays=duplicate_count,
        excluded_source_records=excluded_source,
        excluded_h3_records=excluded_h3,
        missing_baselines=missing_baselines,
        excluded_invalid_records=excluded_invalid,
    )


def apply_controller_effect(
    meta: MetaMemory, aggregation: EffectAggregation
) -> MetaMemory:
    """Return a deployment meta snapshot with effect owned by the controller."""

    owner_role = require_single_role(meta.roles, context="effect target memory")
    unexpected_roles = set(aggregation.role_stats) - {owner_role}
    if unexpected_roles:
        raise InvariantViolation(
            "effect aggregation crossed memory role namespace: "
            f"{sorted(unexpected_roles)!r}"
        )
    return MetaMemory(
        roles=(owner_role,),
        use_when=meta.use_when,
        avoid_when=meta.avoid_when,
        effect={owner_role: aggregation.role_stats[owner_role]}
        if owner_role in aggregation.role_stats
        else {},
    )


def supported_roles(
    aggregation: EffectAggregation,
    *,
    min_tasks: int,
) -> tuple[str, ...]:
    return tuple(
        role
        for role in sorted(aggregation.role_stats)
        if role_effect_class(
            aggregation.role_stats[role],
            epsilon=aggregation.epsilon,
            min_tasks=min_tasks,
        )
        == "stable_positive"
    )
