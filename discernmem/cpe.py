from __future__ import annotations

from dataclasses import asdict, dataclass
import difflib
from typing import Any, Iterable, Mapping, Optional

from .counterexamples import ContentCounterexample
from .effect import aggregate_effect
from .ids import stable_id
from .invariants import InvariantViolation, MemoryState, require_single_role
from .schema import (
    CapabilityConstraint,
    HLabel,
    MemoryVersion,
    MetaMemory,
    RepairCandidate,
    ReplayCondition,
    ReplayRecord,
    ReplayStatus,
    ValidationPlan,
)
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now


@dataclass(frozen=True)
class TaskComparison:
    task_id: str
    role: str
    stage: str
    baseline_score: float
    old_score: float
    candidate_score: float

    @property
    def candidate_vs_baseline(self) -> float:
        return self.candidate_score - self.baseline_score

    @property
    def candidate_vs_old(self) -> float:
        return self.candidate_score - self.old_score

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_vs_baseline": self.candidate_vs_baseline,
            "candidate_vs_old": self.candidate_vs_old,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    accepted: bool
    failed_stage: str
    reasons: tuple[str, ...]
    edit_cost: int
    worst_h_margin: float
    comparisons: tuple[TaskComparison, ...]
    decision_kind: str = "rejected"
    q_margin: float = float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "failed_stage": self.failed_stage,
            "reasons": list(self.reasons),
            "edit_cost": self.edit_cost,
            "worst_h_margin": self.worst_h_margin,
            "decision_kind": self.decision_kind,
            "q_margin": self.q_margin,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass(frozen=True)
class CPESelection:
    winner: Optional[RepairCandidate]
    evaluations: tuple[CandidateEvaluation, ...]


def edit_cost(parent: str, candidate: str) -> int:
    parent_words = parent.split()
    candidate_words = candidate.split()
    cost = 0
    for opcode, left_start, left_end, right_start, right_end in difflib.SequenceMatcher(
        a=parent_words,
        b=candidate_words,
    ).get_opcodes():
        if opcode != "equal":
            cost += max(left_end - left_start, right_end - right_start)
    return cost


def _shared_control_conflicts(records: Iterable[ReplayRecord]) -> tuple[str, ...]:
    observed: dict[tuple[Any, ...], tuple[float, str]] = {}
    conflicts = []
    for record in records:
        if record.status is not ReplayStatus.SUCCEEDED or record.score is None:
            continue
        common = (
            record.task_id,
            record.seed,
            record.config_fingerprint,
            record.runtime_signature,
        )
        if record.condition is ReplayCondition.NO_MEMORY:
            key = ("B", *common)
        elif record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE:
            key = (
                "O",
                *common,
                record.role,
                record.memory_id,
                record.version,
            )
        else:
            continue
        score = float(record.score)
        previous = observed.get(key)
        if previous is None:
            observed[key] = (score, record.replay_id)
        elif previous[0] != score:
            conflicts.append(
                f"shared {key[0]} control {key[1:]} has conflicting scores "
                f"{previous[0]} ({previous[1]}) and {score} ({record.replay_id})"
            )
    return tuple(conflicts)


def _means_by_task_role(
    records: Iterable[ReplayRecord],
    *,
    candidate_id: str,
    memory_id: str,
    old_version: int,
    stage_by_task: Mapping[str, str],
) -> tuple[TaskComparison, ...]:
    records = [
        record
        for record in records
        if record.status is ReplayStatus.SUCCEEDED and record.score is not None
    ]
    baselines = {}
    old = {}
    candidate = {}
    for record in records:
        common = (
            record.task_id,
            record.seed,
            record.config_fingerprint,
            record.runtime_signature,
        )
        if record.condition is ReplayCondition.NO_MEMORY:
            baselines.setdefault(common, float(record.score))
        elif (
            record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE
            and record.memory_id == memory_id
            and record.version == old_version
        ):
            old.setdefault((*common, record.role), float(record.score))
        elif (
            record.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE
            and record.candidate_id == candidate_id
        ):
            candidate.setdefault((*common, record.role), float(record.score))
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    for key, candidate_score in candidate.items():
        task_id, seed, config_fp, runtime_fp, role = key
        common = (task_id, seed, config_fp, runtime_fp)
        if common not in baselines or key not in old:
            continue
        grouped.setdefault(
            (task_id, role, stage_by_task.get(task_id, "unknown")), []
        ).append((baselines[common], old[key], candidate_score))
    comparisons = []
    for (task_id, role, stage), triples in sorted(grouped.items()):
        comparisons.append(
            TaskComparison(
                task_id=task_id,
                role=role,
                stage=stage,
                baseline_score=sum(value[0] for value in triples) / len(triples),
                old_score=sum(value[1] for value in triples) / len(triples),
                candidate_score=sum(value[2] for value in triples) / len(triples),
            )
        )
    return tuple(comparisons)


class CapabilityPreservingEvolution:
    def __init__(self, *, store: DiscernMemStore):
        self.store = store

    def evaluate_candidate(
        self,
        *,
        parent: MemoryVersion,
        candidate: RepairCandidate,
        counterexample: ContentCounterexample,
        plan: ValidationPlan,
        records: Iterable[ReplayRecord],
        allow_neutral: bool = False,
    ) -> CandidateEvaluation:
        assignment_by_task = {item.task_id: item for item in plan.assignments}
        stage_by_task = {
            counterexample.task_id: "q",
            **{
                task_id: assignment.label.value
                for task_id, assignment in assignment_by_task.items()
            },
        }
        comparisons = _means_by_task_role(
            records,
            candidate_id=candidate.candidate_id,
            memory_id=parent.memory_id,
            old_version=parent.version,
            stage_by_task=stage_by_task,
        )
        indexed = {(item.task_id, item.role): item for item in comparisons}
        owner_role = require_single_role(
            parent.meta.roles, context="CPE parent memory"
        )
        if counterexample.role != owner_role:
            raise InvariantViolation(
                f"counterexample role {counterexample.role!r} cannot evaluate memory "
                f"role {owner_role!r}"
            )
        supported = (owner_role,)
        epsilon = self.store.config.epsilon
        reasons: list[str] = []
        failed_stage = ""
        decision_kind = "rejected"
        q_margin = float("-inf")
        neutral_safe = False

        q = indexed.get((counterexample.task_id, counterexample.role))
        if q is None:
            reasons.append("q is missing a complete B/O/N comparison")
        else:
            q_margin = q.candidate_vs_old
            if q.candidate_vs_baseline < -epsilon:
                reasons.append("q candidate is significantly below no-memory baseline")
            elif q.candidate_vs_old > epsilon:
                decision_kind = "improved"
            else:
                neutral_safe = (
                    allow_neutral
                    and q.candidate_vs_old >= -epsilon
                    and bool(candidate.evidence_refs)
                    and bool(counterexample.trajectory_evidence_refs)
                    and bool(candidate.expected_change.strip())
                    and candidate.edited_body.strip() != parent.body.strip()
                )
                if neutral_safe:
                    decision_kind = "neutral_safe"
                else:
                    reasons.append("q candidate does not improve over old by epsilon")
        if reasons:
            failed_stage = "q"

        if not reasons:
            for assignment in plan.assignments:
                if assignment.label is not HLabel.H1:
                    continue
                for role in supported:
                    item = indexed.get((assignment.task_id, role))
                    if item is None:
                        reasons.append(f"H1 {assignment.task_id}/{role} lacks B/O/N")
                    elif item.candidate_vs_old < -epsilon:
                        reasons.append(f"H1 {assignment.task_id}/{role} regresses old")
                    elif item.candidate_vs_baseline < -epsilon:
                        reasons.append(
                            f"H1 {assignment.task_id}/{role} regresses baseline"
                        )
            if reasons:
                failed_stage = "H1"

        if not reasons:
            for assignment in plan.assignments:
                if assignment.label is not HLabel.H2:
                    continue
                for role in supported:
                    item = indexed.get((assignment.task_id, role))
                    if item is None:
                        reasons.append(f"H2 {assignment.task_id}/{role} lacks B/O/N")
                    elif role == counterexample.role:
                        # H2 is a transfer-retention probe. A strict gain over
                        # an already-perfect old version is not required; it must
                        # only avoid regressing old or no-memory behavior.
                        if item.candidate_vs_old < -epsilon:
                            reasons.append(f"H2 {assignment.task_id}/{role} regresses old")
                        elif item.candidate_vs_baseline < -epsilon:
                            reasons.append(
                                f"H2 {assignment.task_id}/{role} regresses baseline"
                            )
                    elif item.candidate_vs_old < -epsilon:
                        reasons.append(f"H2 {assignment.task_id}/{role} regresses old")
            if reasons:
                failed_stage = "H2"

        relevant = [item for item in comparisons if item.stage in {"q", "H1", "H2"}]
        if not reasons:
            old_failures = sum(item.old_score <= 0 for item in relevant)
            candidate_failures = sum(item.candidate_score <= 0 for item in relevant)
            if candidate_failures > old_failures:
                reasons.append(
                    "candidate increases zero-score or complete-failure comparisons"
                )
                failed_stage = "failure_count"
        if reasons:
            decision_kind = "rejected"
        h_margins = [
            min(item.candidate_vs_old, item.candidate_vs_baseline)
            for item in relevant
            if item.stage in {"H1", "H2"}
        ]
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            accepted=not reasons,
            failed_stage=failed_stage,
            reasons=tuple(reasons),
            edit_cost=edit_cost(parent.body, candidate.edited_body),
            worst_h_margin=min(h_margins) if h_margins else float("-inf"),
            comparisons=comparisons,
            decision_kind=decision_kind,
            q_margin=q_margin,
        )

    def evaluate_birth_candidate(self, *, parent: MemoryVersion, candidate: RepairCandidate, records: Iterable[ReplayRecord], validation_task_ids: Iterable[str], role: str) -> CandidateEvaluation:
        allowed = set(validation_task_ids); baselines = {}; old = {}; proposed = {}
        for record in records:
            if record.task_id not in allowed or record.status is not ReplayStatus.SUCCEEDED or record.score is None: continue
            key = (record.task_id, record.seed, record.config_fingerprint, record.runtime_signature, role)
            if record.condition is ReplayCondition.NO_MEMORY: baselines[key] = float(record.score)
            elif record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE and record.role == role and record.version == parent.version: old[key] = float(record.score)
            elif record.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE and record.role == role and record.candidate_id == candidate.candidate_id: proposed[key] = float(record.score)
        comparisons = []
        for task_id in sorted(allowed):
            vals = [(baselines[k], old[k], proposed[k]) for k in proposed if k[0] == task_id and k in baselines and k in old]
            if vals:
                b, o, c = zip(*vals); comparisons.append(TaskComparison(task_id, role, "birth", sum(b)/len(b), sum(o)/len(o), sum(c)/len(c)))
        eps = self.store.config.epsilon; reasons = []
        for item in comparisons:
            if item.candidate_vs_baseline < -eps: reasons.append(f"birth {item.task_id} candidate regresses no-memory")
            if item.candidate_vs_old < -eps: reasons.append(f"birth {item.task_id} candidate regresses parent")
        if not comparisons: reasons.append("birth lacks complete comparisons")
        return CandidateEvaluation(candidate_id=candidate.candidate_id, accepted=not reasons, failed_stage="birth" if reasons else "", reasons=tuple(reasons), edit_cost=edit_cost(parent.body, candidate.edited_body), worst_h_margin=min((min(i.candidate_vs_old, i.candidate_vs_baseline) for i in comparisons), default=float("-inf")), comparisons=tuple(comparisons), decision_kind="improved" if any(i.candidate_vs_old > eps for i in comparisons) and not reasons else "rejected")

    def select(
        self,
        *,
        parent: MemoryVersion,
        candidates: Iterable[RepairCandidate],
        counterexample: ContentCounterexample,
        plan: ValidationPlan,
        records: Iterable[ReplayRecord],
        event_id: str,
        allow_neutral: bool = False,
    ) -> CPESelection:
        candidates = tuple(candidates)
        records = tuple(records)
        consistency_errors = _shared_control_conflicts(records)
        if consistency_errors:
            evaluations = tuple(
                CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    accepted=False,
                    failed_stage="replay_consistency",
                    reasons=consistency_errors,
                    edit_cost=edit_cost(parent.body, candidate.edited_body),
                    worst_h_margin=float("-inf"),
                    comparisons=(),
                )
                for candidate in candidates
            )
        else:
            evaluations = tuple(
                self.evaluate_candidate(
                    parent=parent,
                    candidate=candidate,
                    counterexample=counterexample,
                    plan=plan,
                    records=records,
                    allow_neutral=allow_neutral,
                )
                for candidate in candidates
            )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        accepted = [item for item in evaluations if item.accepted]
        accepted.sort(
            key=lambda item: (
                0 if item.decision_kind == "improved" else 1,
                -item.q_margin,
                -item.worst_h_margin,
                item.edit_cost,
                item.candidate_id,
            )
        )
        winner = by_id[accepted[0].candidate_id] if accepted else None
        _atomic_write_json(
            self.store.root
            / "audit"
            / "cpe_selections"
            / f"{safe_component(event_id)}.json",
            {
                "event_id": event_id,
                "memory_id": parent.memory_id,
                "parent_version": parent.version,
                "counterexample": counterexample.to_dict(),
                "validation_plan_id": plan.plan_id,
                "evaluations": [item.to_dict() for item in evaluations],
                "winner_candidate_id": winner.candidate_id if winner else None,
                "recorded_at": utc_now(),
            },
        )
        return CPESelection(winner=winner, evaluations=evaluations)

    def commit_winner(
        self,
        *,
        parent: MemoryVersion,
        selection: CPESelection,
        counterexample: ContentCounterexample,
        plan: ValidationPlan,
        records: Iterable[ReplayRecord],
        avoid_when: Optional[tuple[str, ...]] = None,
    ) -> Optional[MemoryVersion]:
        winner = selection.winner
        decision_dir = self.store.root / "audit" / "candidate_decisions"
        for evaluation in selection.evaluations:
            _atomic_write_json(
                decision_dir / f"{safe_component(evaluation.candidate_id)}.json",
                {
                    **evaluation.to_dict(),
                    "decision": (
                        "accepted"
                        if winner and evaluation.candidate_id == winner.candidate_id
                        else "rejected"
                    ),
                    "formal_version_created": bool(
                        winner and evaluation.candidate_id == winner.candidate_id
                    ),
                    "recorded_at": utc_now(),
                },
            )
        if winner is None:
            return None
        winner_evaluation = next(
            (
                evaluation
                for evaluation in selection.evaluations
                if evaluation.candidate_id == winner.candidate_id
            ),
            None,
        )
        neutral_safe = bool(
            winner_evaluation and winner_evaluation.decision_kind == "neutral_safe"
        )
        new_version_number = parent.version + 1
        converted = []
        h2_replay_ids = []
        for record in records:
            if record.condition is ReplayCondition.NO_MEMORY:
                converted.append(record)
                if record.h_label is HLabel.H2:
                    h2_replay_ids.append(record.replay_id)
            elif (
                record.condition is ReplayCondition.ONLY_CANDIDATE_FOR_ROLE
                and record.candidate_id == winner.candidate_id
            ):
                data = record.to_dict()
                data.update(
                    {
                        "condition": ReplayCondition.ONLY_VERSION_FOR_ROLE.value,
                        "memory_id": parent.memory_id,
                        "version": new_version_number,
                        "candidate_id": "",
                        "is_source_task": record.task_id == counterexample.task_id,
                    }
                )
                converted_record = ReplayRecord.from_dict(data)
                converted.append(converted_record)
                if record.h_label is HLabel.H2:
                    h2_replay_ids.append(record.replay_id)
        aggregation = aggregate_effect(
            converted,
            memory_id=parent.memory_id,
            version=new_version_number,
            epsilon=self.store.config.epsilon,
            min_tasks=self.store.config.min_stable_tasks,
        )
        committed_state = MemoryState.PROVISIONAL
        new_meta = MetaMemory(
            roles=parent.meta.roles,
            use_when=parent.meta.use_when,
            avoid_when=avoid_when if avoid_when is not None else parent.meta.avoid_when,
            effect={},
        )
        preserved = (
            list(parent.capabilities)
            if self.store.config.enable_capability_constraints
            else []
        )
        new_capability = CapabilityConstraint(
            capability_id=stable_id(
                "cap",
                {
                    "memory_id": parent.memory_id,
                    "version": new_version_number,
                    "expected_change": winner.expected_change,
                },
            ),
            claim=winner.expected_change,
            introduced_version=new_version_number,
            supported_roles=(counterexample.role,),
            replay_ids=tuple(dict.fromkeys(h2_replay_ids)),
        )
        if (
            not neutral_safe
            and self.store.config.enable_capability_constraints
            and new_capability.replay_ids
        ):
            preserved.append(new_capability)
        return self.store.commit_version(
            memory_id=parent.memory_id,
            expected_parent_version=parent.version,
            body=winner.edited_body,
            meta=new_meta,
            capabilities=tuple(preserved),
            source_task_id=parent.base.source_task_id,
            source_trajectory_ref=parent.base.source_trajectory_ref,
            source_event_ids=parent.base.source_event_ids,
            replay_ids=tuple(
                dict.fromkeys(
                    replay_id
                    for stats in aggregation.role_stats.values()
                    for replay_id in stats.replay_ids
                )
            ),
            state=committed_state,
            decision=("cpe_accepted_neutral_safe" if neutral_safe else "cpe_accepted"),
            audit={
                "candidate_id": winner.candidate_id,
                "decision_kind": (
                    winner_evaluation.decision_kind if winner_evaluation else "improved"
                ),
                "counterexample_task_id": counterexample.task_id,
                "validation_plan_id": plan.plan_id,
                "qualified_positive_states": list(
                    parent.evidence.audit.get("qualified_positive_states", [])
                ),
            },
        )
