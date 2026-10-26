from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class CounterexampleKind(str, Enum):
    NONE = "none"
    DEPLOYMENT = "deployment"
    CONTENT = "content"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class UsageAssessment:
    task_id: str
    memory_id: str
    version: int
    role: str
    role_allowed: bool
    use_when_matches: bool
    avoid_when_matches: bool
    behavior_changed: bool
    isolated_delta: Optional[float]
    baseline_replay_id: str
    version_replay_id: str
    specific_problem_type: str = ""
    specific_problem: str = ""
    trajectory_evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContentCounterexample:
    task_id: str
    memory_id: str
    version: int
    role: str
    problem_type: str
    problem: str
    isolated_delta: float
    replay_ids: tuple[str, str]
    trajectory_evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["replay_ids"] = list(self.replay_ids)
        payload["trajectory_evidence_refs"] = list(self.trajectory_evidence_refs)
        return payload


@dataclass(frozen=True)
class CounterexampleRouting:
    kind: CounterexampleKind
    reason: str
    content: Optional[ContentCounterexample] = None


_CONTENT_PROBLEMS = {
    "missing_step",
    "wrong_condition",
    "wrong_order",
    "unexecutable_action",
    "missing_progress_check",
    "wrong_stop_condition",
}


def route_counterexample(
    assessment: UsageAssessment,
    *,
    epsilon: float,
) -> CounterexampleRouting:
    if not assessment.role_allowed:
        return CounterexampleRouting(
            CounterexampleKind.DEPLOYMENT, "memory was deployed to an unsupported role"
        )
    if assessment.avoid_when_matches:
        return CounterexampleRouting(
            CounterexampleKind.DEPLOYMENT, "observable state matched avoid_when"
        )
    if not assessment.behavior_changed:
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS,
            "memory did not measurably alter behavior; content attribution is unavailable",
        )
    if assessment.isolated_delta is None:
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS, "isolated paired replay is missing"
        )
    if assessment.isolated_delta > epsilon:
        return CounterexampleRouting(
            CounterexampleKind.NONE, "isolated use was beneficial"
        )
    if assessment.isolated_delta >= -epsilon:
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS,
            "isolated use was neutral; content repair requires a significant negative effect",
        )
    if not assessment.use_when_matches:
        return CounterexampleRouting(
            CounterexampleKind.DEPLOYMENT,
            "paired replay was harmful because the observable use conditions did not match",
        )
    if assessment.specific_problem_type not in _CONTENT_PROBLEMS:
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS,
            "non-positive score lacks a supported content-level diagnosis",
        )
    if (
        not assessment.specific_problem.strip()
        or not assessment.trajectory_evidence_refs
    ):
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS,
            "content diagnosis lacks a specific problem or trajectory evidence",
        )
    if not assessment.baseline_replay_id or not assessment.version_replay_id:
        return CounterexampleRouting(
            CounterexampleKind.AMBIGUOUS, "paired replay IDs are incomplete"
        )
    content = ContentCounterexample(
        task_id=assessment.task_id,
        memory_id=assessment.memory_id,
        version=assessment.version,
        role=assessment.role,
        problem_type=assessment.specific_problem_type,
        problem=assessment.specific_problem,
        isolated_delta=float(assessment.isolated_delta),
        replay_ids=(assessment.baseline_replay_id, assessment.version_replay_id),
        trajectory_evidence_refs=assessment.trajectory_evidence_refs,
    )
    return CounterexampleRouting(
        CounterexampleKind.CONTENT,
        "correct deployment, behavior change, paired negative effect, and a specific trace problem",
        content,
    )
