from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .admission_controller import DiscernMemAdmission
from .counterexamples import (
    ContentCounterexample,
    CounterexampleKind,
    route_counterexample,
)
from .cpe import CapabilityPreservingEvolution
from .effect import EffectAggregation, aggregate_effect
from .events import EventJournal
from .ids import stable_id
from .invariants import InvariantViolation, MemoryState, require_single_role
from .meta_memory import (
    PositiveStateEvidence,
    sanitize_meta_conditions,
)
from .online_events import _merge_stats
from .pipeline import DiscernMemPromptError, DiscernMemPromptPipeline
from .replay import ReplayExecutor
from .repair_generator import RepairGenerationError, DiscernMemRepairGenerator
from .scheduling import ReplayTaskContext, ReplayScheduler
from .schema import (
    AdmissionDecision,
    MetaMemory,
    MetaUsageEvent,
    ReplayCondition,
    ReplayRecord,
    TaskEvidence,
)
from .state_engine import StateEvidence, DiscernMemStateEngine
from .storage import (
    ConcurrentUpdateError,
    DiscernMemStore,
    _atomic_write_json,
    read_json,
    safe_component,
    utc_now,
)
from .trajectory_projection import (
    compact_text,
    extract_initial_observable_state,
    project_assessment_trajectory,
    project_extraction_trajectory,
)
from .usage_assessor import DiscernMemUsageAssessor, UsageAssessmentError
from .validation import HistoricalTaskSummary, DiscernMemValidationPlanner


logger = logging.getLogger(__name__)


def _log_summary(value: object, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text or "-"
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{clipped or text[:limit]}..."


@dataclass(frozen=True)
class ControllerResult:
    task_id: str
    status: str
    admission_decision: str = ""
    memory_id: str = ""
    validation_plan_id: str = ""
    replay_ids: tuple[str, ...] = ()
    state: str = ""
    event_id: str = ""
    formal_use_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "admission_decision": self.admission_decision,
            "memory_id": self.memory_id,
            "validation_plan_id": self.validation_plan_id,
            "replay_ids": list(self.replay_ids),
            "state": self.state,
            "event_id": self.event_id,
            "formal_use_event_ids": list(self.formal_use_event_ids),
        }


class DiscernMemController:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        extraction: DiscernMemPromptPipeline,
        admission: DiscernMemAdmission,
        validation_planner: DiscernMemValidationPlanner,
        replay_executor: Optional[ReplayExecutor],
        meta_summarizer,
        usage_assessor: Optional[DiscernMemUsageAssessor] = None,
        repair_generator: Optional[DiscernMemRepairGenerator] = None,
        cpe: Optional[CapabilityPreservingEvolution] = None,
        mas_spec: Optional[Mapping[str, Any]] = None,
    ):
        self.store = store
        self.extraction = extraction
        self.admission = admission
        self.validation_planner = validation_planner
        self.replay_executor = replay_executor
        self.meta_summarizer = meta_summarizer
        self.usage_assessor = usage_assessor
        self.repair_generator = repair_generator
        self.cpe = cpe
        self.mas_spec = dict(
            mas_spec or {"mas_type": "unknown", "roles": [], "edges": []}
        )
        self.journal = EventJournal(store)
        self.scheduler = ReplayScheduler(
            config_fingerprint=store.config.fingerprint,
            seeds=store.config.seeds,
        )
        self.state_engine = DiscernMemStateEngine(
            epsilon=store.config.epsilon,
            min_stable_tasks=store.config.min_stable_tasks,
        )
        self.historical_tasks: dict[str, HistoricalTaskSummary] = {}
        self.task_contexts: dict[str, ReplayTaskContext] = {}
        self._load_historical_tasks()

    def _historical_index_dir(self) -> Path:
        return self.store.root / "tasks" / "historical_index"

    def _load_historical_tasks(self) -> None:
        for path in sorted(self._historical_index_dir().glob("*.json")):
            payload = read_json(path)
            if payload.get("config_fingerprint") != self.store.config.fingerprint:
                raise ValueError(
                    f"historical task config fingerprint mismatch in {path}"
                )
            summary = HistoricalTaskSummary.from_dict(payload["summary"])
            context = ReplayTaskContext.from_dict(payload["replay_context"])
            if summary.task_index < 0 and context.task_index >= 0:
                summary = replace(summary, task_index=context.task_index)
            self.store.config.split_manifest.assert_history_eligible(summary.task_id)
            if summary.task_id != context.task_id:
                raise ValueError(f"historical task context mismatch in {path}")
            if summary.task_index != context.task_index:
                raise ValueError(f"historical task index mismatch in {path}")
            self.historical_tasks[summary.task_id] = summary
            self.task_contexts[summary.task_id] = context

    def set_mas_spec(self, mas_spec: Mapping[str, Any]) -> None:
        self.mas_spec = dict(mas_spec)

    def set_replay_executor(self, executor: ReplayExecutor) -> None:
        self.replay_executor = executor

    def register_historical_task(
        self,
        summary: HistoricalTaskSummary,
        replay_context: ReplayTaskContext,
    ) -> None:
        self.store.config.split_manifest.assert_history_eligible(summary.task_id)
        if summary.task_id != replay_context.task_id:
            raise ValueError("historical summary and replay context task IDs differ")
        if summary.task_index != replay_context.task_index:
            raise ValueError(
                "historical summary and replay context task indices differ"
            )
        path = self._historical_index_dir() / f"{safe_component(summary.task_id)}.json"
        with self.store._exclusive():
            self.store._write_once(
                path,
                {
                    "summary": summary.to_dict(),
                    "replay_context": replay_context.to_dict(),
                    "config_fingerprint": self.store.config.fingerprint,
                },
            )
        self.historical_tasks[summary.task_id] = summary
        self.task_contexts[summary.task_id] = replay_context

    @staticmethod
    def summarize_historical_task(
        *,
        task_id: str,
        task_index: int,
        task_description: str,
        trajectory: Mapping[str, Any],
        mas_type: str,
        partition,
    ) -> HistoricalTaskSummary:
        steps = list(trajectory.get("steps") or [])
        first = steps[0] if steps else {}
        last = steps[-1] if steps else {}
        first_context = first.get("state_summary") or first.get("input") or ""
        observable_trigger = extract_initial_observable_state(first_context)
        return HistoricalTaskSummary(
            task_id=task_id,
            partition=partition,
            mas_type=mas_type,
            observable_trigger=str(
                observable_trigger or first_context or task_description
            )[:1200],
            key_operation=str(
                last.get("output") or trajectory.get("final_response") or ""
            )[:1200],
            expected_state_change=str(
                last.get("feedback") or trajectory.get("evaluator_feedback") or ""
            )[:1200],
            surface_summary=task_description[:1600],
            task_index=int(task_index),
            source_entities=tuple(
                str(value) for value in trajectory.get("source_entities", [])
            ),
        )

    def _memory_for_admission(self, admission_id: str):
        for identity in self.store.iter_identities():
            version = self.store.get_version(identity.memory_id, 0)
            if version and version.base.audit.get("admission_id") == admission_id:
                return version
        return None

    @staticmethod
    def _source_event_ids(trajectory: Mapping[str, Any]) -> tuple[str, ...]:
        event_ids = []
        for index, item in enumerate(
            list(trajectory.get("events") or []) + list(trajectory.get("steps") or [])
        ):
            event_ids.append(str(item.get("event_id") or f"event-{index + 1}"))
        return tuple(dict.fromkeys(event_ids))

    @classmethod
    def _formal_evidence_sources(
        cls,
        *,
        official: Mapping[str, Any],
        baseline: ReplayRecord,
        version: ReplayRecord,
    ) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
        sources = {
            "official": cls._normalize_trajectory_events(official),
            "baseline": (
                cls._normalize_trajectory_events(read_json(Path(baseline.trace_ref)))
                if Path(baseline.trace_ref).exists()
                else cls._normalize_trajectory_events({})
            ),
            "version": (
                cls._normalize_trajectory_events(read_json(Path(version.trace_ref)))
                if Path(version.trace_ref).exists()
                else cls._normalize_trajectory_events({})
            ),
        }
        refs: list[str] = []
        for namespace, trajectory in sources.items():
            projected = project_assessment_trajectory(
                trajectory, evidence_namespace=namespace
            )
            refs.extend(
                str(item.get("event_id") or "")
                for item in projected.get("transitions", [])
                if str(item.get("event_id") or "")
            )
            refs.extend(
                f"{namespace}:{item.get('event_id')}"
                for item in trajectory.get("events", []) or []
                if str(item.get("event_id") or "")
            )
        return sources, tuple(dict.fromkeys(refs))

    @classmethod
    def _namespaced_trajectory_evidence(
        cls,
        sources: Mapping[str, Mapping[str, Any]],
        refs: Iterable[str],
    ) -> dict[str, Any]:
        requested = {str(value) for value in refs}
        evidence: dict[str, Any] = {}
        for namespace, trajectory in sources.items():
            local_refs = {
                value.split(":", 1)[1]
                for value in requested
                if value.startswith(f"{namespace}:")
            }
            if not local_refs:
                continue
            local = cls._trajectory_evidence(trajectory, local_refs)
            for event_id, payload in local.items():
                namespaced_id = f"{namespace}:{event_id}"
                value = json.loads(json.dumps(payload, ensure_ascii=False))
                for item in value.values():
                    if isinstance(item, Mapping) and "event_id" in item:
                        item["event_id"] = namespaced_id
                evidence[namespaced_id] = value
        return evidence

    @staticmethod
    def _normalize_trajectory_events(
        trajectory: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(trajectory)
        counter = 0
        for collection_name in ("events", "steps"):
            items = []
            for item in trajectory.get(collection_name, []) or []:
                counter += 1
                value = dict(item)
                value["event_id"] = str(value.get("event_id") or f"event-{counter}")
                items.append(value)
            if items or collection_name in trajectory:
                normalized[collection_name] = items
        feedback = str(trajectory.get("evaluator_feedback") or "").strip()
        events = list(normalized.get("events") or [])
        if feedback and not any(
            item.get("event_id") == "evaluator-feedback" for item in events
        ):
            events.append(
                {
                    "event_id": "evaluator-feedback",
                    "kind": "evaluator_feedback",
                    "feedback": feedback,
                }
            )
            normalized["events"] = events
        return normalized

    @classmethod
    def _extraction_trajectory(cls, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        return project_extraction_trajectory(trajectory)

    def _extract_payload(
        self,
        *,
        task_id: str,
        task_description: str,
        outcome: str,
        score: float,
        trajectory: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "task": {
                "task_id": task_id,
                "description": task_description,
                "task_family": str(trajectory.get("task_family") or ""),
                "outcome": outcome,
                "score": float(score),
                "evaluator_feedback": str(trajectory.get("evaluator_feedback") or ""),
            },
            "mas": dict(self.mas_spec),
            "trajectory": self._extraction_trajectory(trajectory),
        }

    @staticmethod
    def _formal_use_contexts(
        trajectory: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
        for retrieval in trajectory.get("memory_retrievals", []):
            role = str(retrieval.get("agent_role") or "")
            audit_path = str(retrieval.get("retrieval_audit_path") or "")
            pre_action_context: Mapping[str, Any] = {}
            if audit_path and Path(audit_path).exists():
                pre_action_context = dict(
                    read_json(Path(audit_path)).get("pre_action_context") or {}
                )
            for injected in retrieval.get("injected_memories", []):
                memory_id = str(injected.get("memory_id") or "")
                version = injected.get("version")
                injected_role = str(injected.get("injected_role") or role)
                if not memory_id or version is None or not injected_role:
                    continue
                key = (memory_id, int(version), injected_role)
                item = grouped.setdefault(
                    key,
                    {
                        "memory_id": memory_id,
                        "version": int(version),
                        "role": injected_role,
                        "pre_action_contexts": [],
                        "retrieval_ids": [],
                    },
                )
                if pre_action_context:
                    item["pre_action_contexts"].append(dict(pre_action_context))
                retrieval_id = str(retrieval.get("retrieval_id") or "")
                if retrieval_id:
                    item["retrieval_ids"].append(retrieval_id)
        return tuple(
            {
                **item,
                "pre_action_contexts": list(
                    {
                        json.dumps(value, sort_keys=True, default=str): value
                        for value in item["pre_action_contexts"]
                    }.values()
                ),
                "retrieval_ids": list(dict.fromkeys(item["retrieval_ids"])),
            }
            for _, item in sorted(grouped.items())
        )

    def _formal_meta(
        self,
        *,
        memory,
        aggregation: EffectAggregation,
        routing,
        role: str,
        observable_state: str,
        deployment_avoid_when: str,
        assessment,
        source_entities: tuple[str, ...],
    ) -> tuple[MetaMemory, EffectAggregation, tuple[PositiveStateEvidence, ...]]:
        owner_role = require_single_role(
            memory.meta.roles, context="formal-use memory"
        )
        if role != owner_role:
            raise InvariantViolation(
                f"formal-use role {role!r} does not own memory role {owner_role!r}"
            )
        unexpected_roles = set(aggregation.role_stats) - {owner_role}
        if unexpected_roles:
            raise InvariantViolation(
                "formal-use effect crossed memory role namespace: "
                f"{sorted(unexpected_roles)!r}"
            )
        historical_stats = {}
        if owner_role in memory.meta.effect:
            historical_stats[owner_role] = memory.meta.effect[owner_role]
        merged_stats = _merge_stats(
            historical_stats,
            aggregation.role_stats,
            epsilon=self.store.config.epsilon,
        )
        merged = EffectAggregation(
            memory_id=memory.memory_id,
            version=memory.version,
            epsilon=self.store.config.epsilon,
            role_stats=merged_stats,
        )

        prior_positive_states = tuple(
            PositiveStateEvidence.from_dict(item)
            for item in memory.evidence.audit.get("qualified_positive_states", [])
            if isinstance(item, Mapping)
            and str(item.get("role") or "") == owner_role
        )
        positive_states = list(
            {(item.task_id, item.role): item for item in prior_positive_states}.values()
        )
        qualified_positive = (
            assessment.role_allowed
            and not assessment.avoid_when_matches
            and routing.kind is not CounterexampleKind.DEPLOYMENT
            and assessment.isolated_delta is not None
            and assessment.isolated_delta > self.store.config.epsilon
            and bool(observable_state.strip())
        )
        if qualified_positive:
            new_evidence = PositiveStateEvidence(
                task_id=assessment.task_id,
                role=role,
                observable_state=observable_state,
                replay_ids=(
                    aggregation.role_stats[role].replay_ids
                    if aggregation.role_stats.get(role)
                    else ()
                ),
                source_entities=source_entities,
            )
            existing_keys = {(item.task_id, item.role) for item in positive_states}
            if (new_evidence.task_id, new_evidence.role) not in existing_keys:
                positive_states.append(new_evidence)

        use_when = memory.meta.use_when
        cumulative_entities = tuple(
            dict.fromkeys(
                entity for item in positive_states for entity in item.source_entities
            )
        )
        positive_task_ids = {item.task_id for item in positive_states}
        if qualified_positive and len(positive_task_ids) >= 2:
            try:
                summary = dict(
                    self.meta_summarizer(
                        positive_states=[item.to_dict() for item in positive_states],
                        qualifying_boundaries=[],
                    )
                )
                forbidden = {"effect", "roles", "state", "status"} & set(summary)
                if forbidden:
                    raise ValueError(
                        "meta summarizer attempted controller-owned fields: "
                        f"{sorted(forbidden)}"
                    )
                use_when = sanitize_meta_conditions(
                    (
                        *memory.meta.use_when,
                        *(str(value) for value in summary.get("use_when", [])),
                    ),
                    source_entities=cumulative_entities,
                    fallback=memory.meta.use_when,
                )
            except Exception as exc:
                logger.warning(
                    "[DiscernMem][FORMAL-USE] meta summary failed; preserving existing use_when: %s",
                    _log_summary(exc),
                )
        avoid_when = memory.meta.avoid_when
        if routing.kind is CounterexampleKind.DEPLOYMENT and deployment_avoid_when:
            avoid_when = sanitize_meta_conditions(
                (*avoid_when, deployment_avoid_when),
                source_entities=(*cumulative_entities, *source_entities),
                fallback=memory.meta.avoid_when,
            )
        return (
            MetaMemory(
                roles=(owner_role,),
                use_when=use_when,
                avoid_when=avoid_when,
                effect=merged_stats,
            ),
            merged,
            tuple(positive_states),
        )

    @staticmethod
    def _trajectory_evidence(
        trajectory: Mapping[str, Any], refs: Iterable[str]
    ) -> dict[str, Any]:
        requested = set(refs)
        evidence: dict[str, Any] = {}
        for index, item in enumerate(trajectory.get("events") or []):
            event_id = str(item.get("event_id") or f"event-{index + 1}")
            if event_id not in requested:
                continue
            evidence.setdefault(event_id, {})["event"] = {
                "event_id": event_id,
                "kind": str(item.get("kind") or ""),
                "role": str(item.get("agent") or item.get("role") or ""),
                "observation": compact_text(
                    item.get("observation"), head=100, tail=300
                ),
                "action": compact_text(item.get("action"), head=80, tail=220),
                "feedback": compact_text(item.get("feedback"), head=100, tail=300),
            }
        for index, item in enumerate(trajectory.get("steps") or []):
            event_id = str(item.get("event_id") or f"step-{index + 1}")
            if event_id not in requested:
                continue
            transition = dict(item.get("transition") or {})
            action_guard = dict(item.get("action_guard") or {})
            evidence.setdefault(event_id, {})["step"] = {
                "event_id": event_id,
                "turn": int(item.get("turn", item.get("step_index", 0)) or 0),
                "role": str(item.get("agent") or item.get("role") or ""),
                "input": compact_text(item.get("input"), head=80, tail=180),
                "output": compact_text(item.get("output"), head=80, tail=220),
                "feedback": compact_text(item.get("feedback"), head=100, tail=300),
                "transition": {
                    "action": str(transition.get("action") or ""),
                    "valid": bool(transition.get("valid", True)),
                    "state_changed": bool(transition.get("state_changed", True)),
                    "reward": float(transition.get("reward", 0.0) or 0.0),
                    "newly_satisfied": list(
                        transition.get("newly_satisfied", []) or []
                    ),
                    "newly_lost": list(transition.get("newly_lost", []) or []),
                    "invalid_feedback": compact_text(
                        transition.get("invalid_feedback"), head=0, tail=400
                    ),
                },
                "action_guard": {
                    "model_action": str(action_guard.get("model_action") or ""),
                    "executed_action": str(action_guard.get("executed_action") or ""),
                    "was_replaced": bool(action_guard.get("was_replaced", False)),
                    "guard_note": compact_text(
                        action_guard.get("guard_note"), head=0, tail=240
                    ),
                },
                "done": bool(item.get("done", False)),
            }
        return evidence

    @staticmethod
    def _normalized_memory_body(body: str) -> str:
        return " ".join(
            "".join(
                character.lower() if character.isalnum() else " " for character in body
            ).split()
        )

    @classmethod
    def _origin_evidence(cls, parent) -> dict[str, Any]:
        """Load only source events recorded for the memory's origin task."""
        path = str(getattr(parent.base, "source_trajectory_ref", "") or "")
        refs = tuple(getattr(parent.base, "source_event_ids", ()) or ())
        if not path:
            return {}
        try:
            source = read_json(Path(path))
        except (OSError, ValueError, TypeError):
            return {}
        if refs:
            evidence = cls._trajectory_evidence(source, refs)
        else:
            evidence = cls._trajectory_evidence(
                cls._normalize_trajectory_events(source),
                cls._source_event_ids(source),
            )
        return {
            "source_task_id": str(parent.base.source_task_id),
            "source_trajectory_ref": path,
            "source_event_ids": list(refs),
            "evidence": evidence,
        }

    def _repair_content(
        self,
        *,
        parent,
        counterexample: ContentCounterexample,
        replay_context: ReplayTaskContext,
        trajectory: Mapping[str, Any],
        trajectory_evidence: Mapping[str, Any] | None = None,
        origin_evidence: Mapping[str, Any] | None = None,
        event_id: str,
    ) -> dict[str, Any]:
        if self.repair_generator is None or self.cpe is None:
            raise RuntimeError("DiscernMem content repair components are not configured")
        owner_role = require_single_role(
            parent.meta.roles, context="repair parent memory"
        )
        if counterexample.role != owner_role:
            raise InvariantViolation(
                f"counterexample role {counterexample.role!r} cannot repair memory "
                f"role {owner_role!r}"
            )
        roles = (owner_role,)
        capabilities = (
            parent.capabilities
            if self.store.config.enable_capability_constraints
            else ()
        )
        planning = self.validation_planner.plan(
            trigger_task_id=counterexample.task_id,
            trigger_task_index=replay_context.task_index,
            memory_id=parent.memory_id,
            version=parent.version,
            body=parent.body,
            meta=parent.meta,
            capabilities=capabilities,
            problem_summary=counterexample.problem,
            roles=roles,
            source_task_ids=(parent.base.source_task_id,),
            historical_tasks=tuple(self.historical_tasks.values()),
            purpose="content_repair",
            strict_h2_roles=(counterexample.role,),
        )
        if planning.plan is None:
            return {
                "status": "repair_missing_evidence",
                "missing_evidence": list(planning.missing_evidence),
                "replay_ids": [],
            }
        evidence = (
            dict(trajectory_evidence)
            if trajectory_evidence is not None
            else self._trajectory_evidence(
                trajectory, counterexample.trajectory_evidence_refs
            )
        )
        source_entities = tuple(
            str(value) for value in trajectory.get("source_entities", [])
        )
        # CPE generation occurs in the loop below; direct_overwrite stays single-shot.
        candidate_factory = (
            self.repair_generator.generate
            if self.store.config.evolution_mode == "direct_overwrite"
            else lambda **_: ()
        )
        try:
            candidates = candidate_factory(
                memory_id=parent.memory_id,
                parent_version=parent.version,
                body=parent.body,
                meta=parent.meta,
                counterexample=counterexample,
                mas_spec=self.mas_spec,
                capabilities=capabilities,
                trajectory_evidence=evidence,
                origin_evidence=dict(origin_evidence or {}),
                source_entities=source_entities,
                artifact_name=event_id,
            )
        except RepairGenerationError as exc:
            failure_path = (
                self.store.root
                / "audit"
                / "repair_failures"
                / f"{safe_component(event_id)}.json"
            )
            _atomic_write_json(
                failure_path,
                {
                    "event_id": event_id,
                    "memory_id": parent.memory_id,
                    "parent_version": parent.version,
                    "counterexample": counterexample.to_dict(),
                    "validation_plan_id": planning.plan.plan_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "recorded_at": utc_now(),
                },
            )
            logger.warning(
                "[DiscernMem][REPAIR][task=%s][memory=%s@v%s] generation-failed reason=%s",
                counterexample.task_id,
                parent.memory_id,
                parent.version,
                _log_summary(exc),
            )
            return {
                "status": "repair_generation_failed",
                "validation_plan_id": planning.plan.plan_id,
                "replay_ids": [],
            }
        if self.store.config.evolution_mode == "direct_overwrite" and not candidates:
            return {
                "status": "repair_no_update",
                "validation_plan_id": planning.plan.plan_id,
                "replay_ids": [],
            }
        if self.store.config.evolution_mode == "direct_overwrite":
            winner = candidates[0]
            direct_meta = MetaMemory(
                roles=parent.meta.roles,
                use_when=parent.meta.use_when,
                avoid_when=parent.meta.avoid_when,
                effect={},
            )
            try:
                committed = self.store.commit_version(
                    memory_id=parent.memory_id,
                    expected_parent_version=parent.version,
                    body=winner.edited_body,
                    meta=direct_meta,
                    capabilities=capabilities,
                    source_task_id=parent.base.source_task_id,
                    source_trajectory_ref=parent.base.source_trajectory_ref,
                    source_event_ids=parent.base.source_event_ids,
                    replay_ids=counterexample.replay_ids,
                    state=MemoryState.PROVISIONAL,
                    decision="direct_overwrite_ablation",
                    audit={
                        "candidate_id": winner.candidate_id,
                        "counterexample_task_id": counterexample.task_id,
                        "validation_plan_id": planning.plan.plan_id,
                        "ablation": "direct_overwrite",
                    },
                )
            except ConcurrentUpdateError:
                return {
                    "status": "repair_stale_parent",
                    "validation_plan_id": planning.plan.plan_id,
                    "replay_ids": list(counterexample.replay_ids),
                }
            _atomic_write_json(
                self.store.root
                / "audit"
                / "direct_overwrite"
                / f"{safe_component(event_id)}.json",
                {
                    "event_id": event_id,
                    "candidate_id": winner.candidate_id,
                    "memory_id": committed.memory_id,
                    "parent_version": parent.version,
                    "committed_version": committed.version,
                    "validation_plan_id": planning.plan.plan_id,
                    "recorded_at": utc_now(),
                },
            )
            return {
                "status": "direct_overwrite_committed",
                "validation_plan_id": planning.plan.plan_id,
                "winner_candidate_id": winner.candidate_id,
                "committed_version": committed.version,
                "state": committed.state.value,
                "replay_ids": list(counterexample.replay_ids),
            }
        if self.replay_executor is None:
            raise RuntimeError("DiscernMem repair requires a real replay executor")
        rounds = []
        replay_ids = []
        attempted_bodies: set[str] = set()
        audit_path = (
            self.store.root
            / "audit"
            / "repair_rounds"
            / f"{safe_component(event_id)}.json"
        )

        def write_audit(final_status):
            _atomic_write_json(
                audit_path,
                {
                    "event_id": event_id,
                    "memory_id": parent.memory_id,
                    "parent_version": parent.version,
                    "validation_plan_id": planning.plan.plan_id,
                    "max_evolution_rounds": self.store.config.max_evolution_rounds,
                    "rounds": rounds,
                    "final_status": final_status,
                    "recorded_at": utc_now(),
                },
            )

        for round_index in range(1, self.store.config.max_evolution_rounds + 1):
            round_event_id = f"{event_id}-round-{round_index}"
            prior_rounds = tuple(
                {
                    "round_index": item["round_index"],
                    "status": item["status"],
                    "candidate_feedback": item.get("candidate_feedback", []),
                }
                for item in rounds
            )
            try:
                candidates = self.repair_generator.generate(
                    memory_id=parent.memory_id,
                    parent_version=parent.version,
                    body=parent.body,
                    meta=parent.meta,
                    counterexample=counterexample,
                    mas_spec=self.mas_spec,
                    capabilities=capabilities,
                    trajectory_evidence=evidence,
                    origin_evidence=dict(origin_evidence or {}),
                    source_entities=source_entities,
                    artifact_name=round_event_id,
                    round_index=round_index,
                    prior_rounds=prior_rounds,
                )
            except Exception as exc:
                round_item = {
                    "round_index": round_index,
                    "status": "generation_failed",
                    "candidate_ids": [],
                    "candidate_feedback": [],
                    "duplicate_candidate_ids": [],
                    "evaluations": [],
                    "replay_ids": [],
                    "winner_candidate_id": "",
                    "committed_version": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                rounds.append(round_item)
                _atomic_write_json(
                    self.store.root
                    / "audit"
                    / "repair_failures"
                    / f"{safe_component(event_id)}.json",
                    {
                        "event_id": event_id,
                        "memory_id": parent.memory_id,
                        "parent_version": parent.version,
                        "counterexample": counterexample.to_dict(),
                        "validation_plan_id": planning.plan.plan_id,
                        "round_index": round_index,
                        "error": round_item["error"],
                        "failures": [
                            item
                            for item in rounds
                            if item["status"] == "generation_failed"
                        ],
                        "recorded_at": utc_now(),
                    },
                )
                write_audit("repair_in_progress")
                continue
            if not candidates:
                rounds.append(
                    {
                        "round_index": round_index,
                        "status": "no_candidates",
                        "candidate_ids": [],
                        "candidate_feedback": [],
                        "duplicate_candidate_ids": [],
                        "evaluations": [],
                        "replay_ids": [],
                        "winner_candidate_id": "",
                        "committed_version": None,
                    }
                )
                write_audit("repair_in_progress")
                continue

            duplicate_candidate_ids = [
                item.candidate_id
                for item in candidates
                if self._normalized_memory_body(item.edited_body) in attempted_bodies
            ]
            candidates = tuple(
                item
                for item in candidates
                if self._normalized_memory_body(item.edited_body)
                not in attempted_bodies
            )
            if not candidates:
                rounds.append(
                    {
                        "round_index": round_index,
                        "status": "duplicate_candidates",
                        "candidate_ids": [],
                        "candidate_feedback": [],
                        "duplicate_candidate_ids": duplicate_candidate_ids,
                        "evaluations": [],
                        "replay_ids": [],
                        "winner_candidate_id": "",
                        "committed_version": None,
                    }
                )
                write_audit("repair_in_progress")
                continue
            attempted_bodies.update(
                self._normalized_memory_body(item.edited_body) for item in candidates
            )

            requests = self.scheduler.repair_comparison(
                plan=planning.plan,
                parent=parent,
                candidates=candidates,
                counterexample=counterexample,
                problem_context=replay_context,
                task_contexts=self.task_contexts,
            )
            records = tuple(self.replay_executor.run_many(requests))
            round_replay_ids = [record.replay_id for record in records]
            replay_ids.extend(round_replay_ids)
            selection = self.cpe.select(
                parent=parent,
                candidates=candidates,
                counterexample=counterexample,
                plan=planning.plan,
                records=records,
                event_id=round_event_id,
                allow_neutral=self.store.config.enable_neutral_repair,
            )
            candidates_by_id = {item.candidate_id: item for item in candidates}
            candidate_feedback = []
            for evaluation in selection.evaluations:
                candidate = candidates_by_id[evaluation.candidate_id]
                comparisons = [item.to_dict() for item in evaluation.comparisons]
                candidate_feedback.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "diagnosis": candidate.diagnosis,
                        "edit_operation": candidate.edit_operation.value,
                        "old_text_or_anchor": str(
                            candidate.audit.get("old_text_or_anchor") or ""
                        ),
                        "edit_text": str(candidate.audit.get("edit_text") or ""),
                        "edited_memory": candidate.edited_body,
                        "expected_change": candidate.expected_change,
                        "failed_stage": evaluation.failed_stage,
                        "rejection_reasons": list(evaluation.reasons),
                        "trigger_task_comparisons": [
                            item for item in comparisons if item["stage"] == "q"
                        ],
                        "validation_comparisons": [
                            item for item in comparisons if item["stage"] != "q"
                        ],
                    }
                )
            round_item = {
                "round_index": round_index,
                "status": "rejected",
                "candidate_ids": [item.candidate_id for item in candidates],
                "candidate_feedback": candidate_feedback,
                "duplicate_candidate_ids": duplicate_candidate_ids,
                "evaluations": [item.to_dict() for item in selection.evaluations],
                "replay_ids": round_replay_ids,
                "winner_candidate_id": (
                    selection.winner.candidate_id if selection.winner else ""
                ),
                "committed_version": None,
            }
            try:
                committed = self.cpe.commit_winner(
                    parent=parent,
                    selection=selection,
                    counterexample=counterexample,
                    plan=planning.plan,
                    records=records,
                    avoid_when=parent.meta.avoid_when,
                )
            except ConcurrentUpdateError:
                round_item["status"] = "stale_parent"
                rounds.append(round_item)
                write_audit("repair_stale_parent")
                return {
                    "status": "repair_stale_parent",
                    "validation_plan_id": planning.plan.plan_id,
                    "replay_ids": replay_ids,
                    "rounds_attempted": len(rounds),
                }
            if committed is None:
                rounds.append(round_item)
                write_audit("repair_in_progress")
                continue
            round_item["status"] = "committed"
            round_item["committed_version"] = committed.version
            rounds.append(round_item)
            write_audit("repair_committed")
            return {
                "status": "repair_committed",
                "validation_plan_id": planning.plan.plan_id,
                "winner_candidate_id": selection.winner.candidate_id,
                "committed_version": committed.version,
                "state": committed.state.value,
                "replay_ids": replay_ids,
                "rounds_attempted": len(rounds),
            }

        write_audit("repair_exhausted")
        return {
            "status": "repair_exhausted",
            "validation_plan_id": planning.plan.plan_id,
            "winner_candidate_id": "",
            "committed_version": None,
            "state": parent.state.value,
            "replay_ids": replay_ids,
            "rounds_attempted": len(rounds),
        }

    def _split_cross_role_usages(
        self, usages: tuple[dict[str, Any], ...]
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        """Split injections into owner-role ones and cross-role ones.

        Retrieval normally only injects a version to the role that owns it.
        With the role-conditioning ablation the same version can reach another
        role. Paired-replay attribution is defined per owning role, so
        cross-role injections are logged but never evaluated.
        """
        in_contract: list[dict[str, Any]] = []
        cross_role: list[dict[str, Any]] = []
        for usage in usages:
            version = self.store.get_version(
                str(usage.get("memory_id") or ""), int(usage.get("version", -1))
            )
            roles = tuple(version.meta.roles) if version is not None else ()
            if roles and roles != (str(usage.get("role") or ""),):
                cross_role.append({**usage, "owner_role": roles[0]})
            else:
                in_contract.append(dict(usage))
        return tuple(in_contract), tuple(cross_role)

    def _run_formal_use_replay_batch(
        self,
        *,
        entries: Iterable[tuple[str, Mapping[str, Any]]],
        replay_context: ReplayTaskContext,
    ) -> dict[str, tuple[ReplayRecord, ...]]:
        if self.replay_executor is None:
            raise RuntimeError("DiscernMem formal use requires a replay executor")
        request_ids: dict[str, tuple[str, ...]] = {}
        unique_requests = {}
        for event_id, usage in entries:
            memory = self.store.get_version(
                str(usage["memory_id"]), int(usage["version"])
            )
            current = self.store.get_version(str(usage["memory_id"]))
            if memory is None or current is None or current.version != memory.version:
                continue
            requests = self.scheduler.formal_use_pair(
                version=memory,
                role=str(usage["role"]),
                task_context=replay_context,
            )
            request_ids[event_id] = tuple(request.replay_id for request in requests)
            for request in requests:
                unique_requests.setdefault(request.replay_id, request)
        if not unique_requests:
            return {}
        logger.info(
            "[DiscernMem][FORMAL-USE][task=%s] batch replay usages=%d unique_requests=%d workers=%d",
            replay_context.task_id,
            len(request_ids),
            len(unique_requests),
            self.replay_executor.max_workers,
        )
        records = self.replay_executor.run_many(unique_requests.values())
        records_by_id = {record.replay_id: record for record in records}
        return {
            event_id: tuple(records_by_id[replay_id] for replay_id in replay_ids)
            for event_id, replay_ids in request_ids.items()
        }

    def _process_formal_use(
        self,
        *,
        task_id: str,
        usage: Mapping[str, Any],
        trajectory: Mapping[str, Any],
        replay_context: ReplayTaskContext,
        event_id: str,
        prepared_records: Optional[Iterable[ReplayRecord]] = None,
    ) -> dict[str, Any]:
        if self.replay_executor is None or self.usage_assessor is None:
            raise RuntimeError(
                "DiscernMem formal use requires replay and usage assessment"
            )
        memory = self.store.get_version(str(usage["memory_id"]), int(usage["version"]))
        current = self.store.get_version(str(usage["memory_id"]))
        if memory is None or current is None or current.version != memory.version:
            return {"status": "stale_injected_version", "replay_ids": []}
        role = str(usage["role"])
        if prepared_records is None:
            requests = self.scheduler.formal_use_pair(
                version=memory,
                role=role,
                task_context=replay_context,
            )
            records = tuple(self.replay_executor.run_many(requests))
        else:
            records = tuple(prepared_records)
        aggregation = aggregate_effect(
            records,
            memory_id=memory.memory_id,
            version=memory.version,
            epsilon=self.store.config.epsilon,
            min_tasks=self.store.config.min_stable_tasks,
        )
        role_stats = aggregation.role_stats.get(role)
        if role_stats is None or task_id not in role_stats.task_deltas:
            return {
                "status": "formal_use_source_or_incomplete",
                "replay_ids": [record.replay_id for record in records],
            }
        baselines = [
            record
            for record in records
            if record.condition is ReplayCondition.NO_MEMORY
        ]
        versions = [
            record
            for record in records
            if record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE
        ]
        if not baselines or not versions:
            raise RuntimeError("formal-use paired replay is incomplete")
        contexts = list(usage.get("pre_action_contexts") or [])
        pre_action_context = {
            "retrieval_contexts": contexts,
            "retrieval_ids": list(usage.get("retrieval_ids") or []),
        }
        evidence_sources, allowed_refs = self._formal_evidence_sources(
            official=trajectory,
            baseline=baselines[0],
            version=versions[0],
        )
        try:
            assessment, deployment_avoid_when = self.usage_assessor.assess(
                task_id=task_id,
                memory=memory,
                role=role,
                pre_action_context=pre_action_context,
                official_trajectory=trajectory,
                baseline=baselines[0],
                version_replay=versions[0],
                isolated_delta=role_stats.task_deltas[task_id],
                allowed_evidence_refs=allowed_refs,
                artifact_name=event_id,
            )
        except UsageAssessmentError as exc:
            logger.warning(
                "[DiscernMem][FORMAL-USE][task=%s][memory=%s@v%s][role=%s] assessment-failed; preserving memory state reason=%s",
                task_id,
                memory.memory_id,
                memory.version,
                role,
                _log_summary(exc),
            )
            return {
                "status": "formal_use_assessment_failed",
                "reason": str(exc),
                "memory_id": memory.memory_id,
                "version": memory.version,
                "replay_ids": [record.replay_id for record in records],
            }
        routing = route_counterexample(assessment, epsilon=self.store.config.epsilon)
        current = self.store.get_version(memory.memory_id)
        if current.version != memory.version:
            return {
                "status": "stale_after_replay",
                "routing": routing.kind.value,
                "replay_ids": [record.replay_id for record in records],
            }
        self.store.record_meta_usage_event(
            MetaUsageEvent(
                event_id=event_id,
                task_id=task_id,
                memory_id=memory.memory_id,
                version=memory.version,
                role=role,
                retrieval_ids=tuple(str(value) for value in usage.get("retrieval_ids") or []),
                use_when_matches=assessment.use_when_matches,
                avoid_when_matches=assessment.avoid_when_matches,
                isolated_delta=assessment.isolated_delta,
                behavior_changed=assessment.behavior_changed,
                routing=routing.kind.value,
                routing_reason=routing.reason,
                status="assessed",
                replay_ids=tuple(record.replay_id for record in records),
                created_at=utc_now(),
            )
        )
        owner_role = require_single_role(
            current.meta.roles, context="formal-use current memory"
        )
        receiving_roles = (owner_role,)
        observable_state = "\n".join(
            str(item.get("observable_state") or item.get("task_description") or "")
            for item in contexts
        )
        meta, merged, positive_states = self._formal_meta(
            memory=current,
            aggregation=aggregation,
            routing=routing,
            role=role,
            observable_state=observable_state,
            deployment_avoid_when=deployment_avoid_when,
            assessment=assessment,
            source_entities=tuple(
                str(value) for value in trajectory.get("source_entities", [])
            ),
        )
        behavior_changed_task_ids_by_role = {
            str(existing_role): list(dict.fromkeys(str(value) for value in task_ids))
            for existing_role, task_ids in (
                current.evidence.audit.get("behavior_changed_task_ids_by_role") or {}
            ).items()
            if str(existing_role) == owner_role
        }
        if (
            assessment.role_allowed
            and not assessment.avoid_when_matches
            and assessment.behavior_changed
        ):
            role_tasks = behavior_changed_task_ids_by_role.setdefault(role, [])
            if task_id not in role_tasks:
                role_tasks.append(task_id)

        audit = {
            **current.evidence.audit,
            "formal_use_event_id": event_id,
            "routing": routing.kind.value,
            "routing_reason": routing.reason,
            "retrieval_ids": list(usage.get("retrieval_ids") or []),
            "qualified_positive_states": [item.to_dict() for item in positive_states],
            "behavior_changed_task_ids_by_role": behavior_changed_task_ids_by_role,
        }
        already_applied = current.evidence.audit.get("formal_use_event_id") == event_id
        if routing.kind is CounterexampleKind.CONTENT:
            if not already_applied:
                self.store.update_evidence(
                    memory_id=current.memory_id,
                    expected_version=current.version,
                    expected_revision=current.evidence.revision,
                    meta=meta,
                    capabilities=current.capabilities,
                    state=current.state,
                    replay_ids=tuple(record.replay_id for record in records),
                    reason="formal use paired effect before content repair",
                    audit=audit,
                )
            parent = self.store.get_version(current.memory_id)
            repair = self._repair_content(
                parent=parent,
                counterexample=routing.content,
                replay_context=replay_context,
                trajectory=trajectory,
                trajectory_evidence=self._namespaced_trajectory_evidence(
                    evidence_sources, routing.content.trajectory_evidence_refs
                ),
                origin_evidence=self._origin_evidence(parent),
                event_id=event_id,
            )
            repair_replays = tuple(repair.get("replay_ids") or [])
            if repair.get("status") not in {
                "repair_committed",
                "direct_overwrite_committed",
            }:
                parent = self.store.get_version(current.memory_id)
                state_decision = self.state_engine.decide(
                    parent.state,
                    StateEvidence(
                        aggregation=merged,
                        all_receiving_roles=receiving_roles,
                        behavior_changed_task_ids_by_role=behavior_changed_task_ids_by_role,
                        has_content_counterexample=True,
                        content_repair_attempted=True,
                        content_repair_failed=True,
                    ),
                )
                if state_decision.next is not parent.state:
                    self.store.update_evidence(
                        memory_id=parent.memory_id,
                        expected_version=parent.version,
                        expected_revision=parent.evidence.revision,
                        meta=parent.meta,
                        capabilities=parent.capabilities,
                        state=state_decision.next,
                        replay_ids=repair_replays,
                        reason=state_decision.reason,
                        audit={**audit, "repair_status": repair.get("status")},
                    )
            return {
                **repair,
                "routing": routing.kind.value,
                "formal_replay_ids": [record.replay_id for record in records],
                "replay_ids": list(
                    dict.fromkeys(
                        [record.replay_id for record in records] + list(repair_replays)
                    )
                ),
            }

        state_decision = self.state_engine.decide(
            current.state,
            StateEvidence(
                aggregation=merged,
                all_receiving_roles=receiving_roles,
                behavior_changed_task_ids_by_role=behavior_changed_task_ids_by_role,
                deployment_boundary_explains_negative=(
                    routing.kind is CounterexampleKind.DEPLOYMENT
                ),
                shadow_role_task_deltas=(
                    {
                        role: dict(
                            aggregation.role_stats.get(role).task_deltas
                            if aggregation.role_stats.get(role)
                            else {}
                        )
                    }
                    if current.state is MemoryState.INACTIVE
                    else None
                ),
            ),
        )
        if not already_applied:
            evidence = self.store.update_evidence(
                memory_id=current.memory_id,
                expected_version=current.version,
                expected_revision=current.evidence.revision,
                meta=meta,
                capabilities=current.capabilities,
                state=state_decision.next,
                replay_ids=tuple(record.replay_id for record in records),
                reason=f"formal use: {routing.reason}",
                audit=audit,
            )
            state = evidence.state.value
        else:
            state = current.state.value
        return {
            "status": "formal_use_updated",
            "routing": routing.kind.value,
            "routing_reason": routing.reason,
            "memory_id": current.memory_id,
            "version": current.version,
            "state": state,
            "replay_ids": [record.replay_id for record in records],
        }

    def _birth_similar_tasks(
        self,
        *,
        task_id: str,
        task_index: int,
        memory,
        task_description: str,
    ) -> tuple[HistoricalTaskSummary, ...]:
        """Select earlier history tasks for birth calibration and birth CPE.

        The returned pool is wide enough for both consumers: birth calibration
        The pool size is ``h2_tasks``: birth calibration and birth CPE share a
        single transfer-validation knob, which is also what bounds formal CPE's
        H2 probes.
        """
        eligible = []
        forbidden = {str(task_id), str(memory.base.source_task_id)}
        for item in self.historical_tasks.values():
            if item.task_id in forbidden or item.task_index >= int(task_index):
                continue
            self.store.config.split_manifest.assert_history_eligible(item.task_id)
            eligible.append(item)
        if not eligible:
            return ()
        query = " ".join((memory.body, *memory.meta.use_when, task_description)).strip()
        width = max(int(self.store.config.h2_tasks), 1)
        recalled = self.validation_planner.task_index.retrieve(
            query,
            eligible,
            top_k=width,
        )
        return tuple(item for item, _score in recalled)

    def _birth_content_counterexample(
        self,
        *,
        memory,
        role: str,
        task_id: str,
        records: Iterable[ReplayRecord],
    ) -> ContentCounterexample | None:
        """Create a conservative birth counterexample from a negative paired task."""
        baseline = {
            record.task_id: record
            for record in records
            if record.condition is ReplayCondition.NO_MEMORY and record.score is not None
        }
        targets = {
            record.task_id: record
            for record in records
            if record.condition is ReplayCondition.ONLY_VERSION_FOR_ROLE
            and record.role == role
            and record.score is not None
        }
        for candidate_task, target in targets.items():
            control = baseline.get(candidate_task)
            if control is None or float(target.score) >= float(control.score) - self.store.config.epsilon:
                continue
            return ContentCounterexample(
                task_id=candidate_task,
                memory_id=memory.memory_id,
                version=memory.version,
                role=role,
                problem_type="birth_calibration_harm",
                problem=(
                    "The newly created memory version reduced the paired replay score "
                    "relative to no-memory on an earlier similar task."
                ),
                isolated_delta=float(target.score) - float(control.score),
                replay_ids=(control.replay_id, target.replay_id),
                trajectory_evidence_refs=tuple(
                    self._source_event_ids(self._trajectory_for_task(candidate_task))
                ),
            )
        return None

    def _trajectory_for_task(self, task_id: str) -> Mapping[str, Any]:
        task = self.store.get_task(str(task_id))
        if task is None or not task.trajectory_ref:
            return {}
        try:
            return read_json(Path(task.trajectory_ref))
        except (OSError, ValueError, TypeError):
            return {}

    def _birth_content_repair(
        self, *, memory, role: str, counterexample: ContentCounterexample,
        validation_tasks: Iterable[HistoricalTaskSummary],
    ) -> dict[str, Any]:
        if not self.store.config.enable_birth_cpe:
            return {"status": "disabled"}
        if self.repair_generator is None or self.cpe is None or self.replay_executor is None:
            return {"status": "unavailable"}
        task_ids = tuple(
            item.task_id for item in validation_tasks if item.task_id in self.task_contexts
        )
        if not task_ids:
            return {"status": "skipped_no_validation_tasks"}
        rounds = []
        attempted_bodies = set()
        origin = self._origin_evidence(memory)
        for round_index in range(1, self.store.config.max_evolution_rounds + 1):
            history = self._trajectory_for_task(counterexample.task_id)
            evidence = self._trajectory_evidence(
                history, counterexample.trajectory_evidence_refs
            )
            try:
                candidates = self.repair_generator.generate(
                    memory_id=memory.memory_id,
                    parent_version=memory.version,
                    body=memory.body,
                    meta=memory.meta,
                    counterexample=counterexample,
                    mas_spec=self.mas_spec,
                    capabilities=memory.capabilities,
                    trajectory_evidence=evidence,
                    origin_evidence=origin,
                    source_entities=(),
                    artifact_name=f"birth-{memory.memory_id}-round-{round_index}",
                    round_index=round_index,
                    prior_rounds=tuple(rounds),
                )
            except Exception as exc:
                rounds.append({
                    "round_index": round_index,
                    "status": "generation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            candidates = tuple(
                item for item in candidates
                if self._normalized_memory_body(item.edited_body) not in attempted_bodies
            )
            if not candidates:
                rounds.append({"round_index": round_index, "status": "no_candidates"})
                continue
            attempted_bodies.update(
                self._normalized_memory_body(item.edited_body) for item in candidates
            )
            contexts = [self.task_contexts[task_id] for task_id in task_ids]
            requests = self.scheduler.birth_comparison(
                parent=memory,
                candidates=candidates,
                role=role,
                validation_contexts=contexts,
                trigger_task_id=counterexample.task_id,
            )
            records = tuple(self.replay_executor.run_many(requests))
            evaluations = tuple(
                self.cpe.evaluate_birth_candidate(
                    parent=memory,
                    candidate=item,
                    records=records,
                    validation_task_ids=task_ids,
                    role=role,
                )
                for item in candidates
            )
            winner = next(
                (candidates[index] for index, item in enumerate(evaluations) if item.accepted),
                None,
            )
            round_item = {
                "round_index": round_index,
                "status": "rejected",
                "candidate_ids": [item.candidate_id for item in candidates],
                "evaluations": [item.to_dict() for item in evaluations],
                "replay_ids": [item.replay_id for item in records],
            }
            if winner is None:
                rounds.append(round_item)
                continue
            committed = self.store.commit_version(
                memory_id=memory.memory_id,
                expected_parent_version=memory.version,
                body=winner.edited_body,
                meta=memory.meta,
                capabilities=memory.capabilities,
                source_task_id=memory.base.source_task_id,
                source_trajectory_ref=memory.base.source_trajectory_ref,
                source_event_ids=memory.base.source_event_ids,
                replay_ids=[item.replay_id for item in records],
                state=MemoryState.PROVISIONAL,
                decision="birth_cpe_accepted",
                audit={
                    "birth_cpe": True,
                    "birth_validation_task_ids": list(task_ids),
                    "candidate_id": winner.candidate_id,
                    "round_index": round_index,
                },
            )
            round_item.update(
                status="committed",
                winner_candidate_id=winner.candidate_id,
                committed_version=committed.version,
            )
            rounds.append(round_item)
            return {
                "status": "birth_cpe_committed",
                "rounds": rounds,
                "committed_version": committed.version,
                "validation_task_ids": list(task_ids),
                "replay_ids": [item.replay_id for item in records],
            }
        return {
            "status": "birth_cpe_exhausted",
            "rounds": rounds,
            "validation_task_ids": list(task_ids),
        }

    def _birth_calibration(
        self,
        *,
        task_id: str,
        task_index: int,
        task_description: str,
        memory,
        replay_context: ReplayTaskContext,
        source_trajectory: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run optional cold-start calibration without changing memory content."""
        if not self.store.config.enable_birth_calibration:
            return {"status": "disabled", "similar_task_ids": [], "replay_ids": []}
        if self.replay_executor is None:
            return {"status": "unavailable", "similar_task_ids": [], "replay_ids": []}
        similar = self._birth_similar_tasks(
            task_id=task_id,
            task_index=task_index,
            memory=memory,
            task_description=task_description,
        )
        path = self.store.root / "audit" / "birth_calibration" / f"{safe_component(task_id)}.json"
        if not similar:
            payload = {"status": "skipped_no_similar_history", "similar_task_ids": [], "replay_ids": [], "recorded_at": utc_now()}
            _atomic_write_json(path, payload)
            return payload
        contexts = [
            self.task_contexts[item.task_id]
            for item in similar
            if item.task_id in self.task_contexts
        ]
        requests = []
        for context in contexts:
            requests.extend(self.scheduler.formal_use_pair(version=memory, role=require_single_role(memory.meta.roles, context="birth calibration memory"), task_context=context))
        records = tuple(self.replay_executor.run_many(requests))
        aggregation = aggregate_effect(records, memory_id=memory.memory_id, version=memory.version, epsilon=self.store.config.epsilon, min_tasks=1, include_h3=True)
        role = require_single_role(memory.meta.roles, context="birth calibration memory")
        stats = aggregation.role_stats.get(role)
        if stats is not None:
            historical = dict(memory.meta.effect)
            merged_stats = _merge_stats(
                historical,
                aggregation.role_stats,
                epsilon=self.store.config.epsilon,
            )
            self.store.update_evidence(
                memory_id=memory.memory_id,
                expected_version=memory.version,
                expected_revision=memory.evidence.revision,
                meta=MetaMemory(
                    roles=memory.meta.roles,
                    use_when=memory.meta.use_when,
                    avoid_when=memory.meta.avoid_when,
                    effect=merged_stats,
                ),
                capabilities=memory.capabilities,
                state=memory.state,
                replay_ids=tuple(record.replay_id for record in records),
                reason="birth calibration effect",
                audit={"birth_calibration": True, "birth_validation_task_ids": [item.task_id for item in similar]},
            )
        birth_counterexample = self._birth_content_counterexample(
            memory=memory,
            role=role,
            task_id=task_id,
            records=records,
        )
        birth_repair = {"status": "not_attempted"}
        if birth_counterexample is not None and self.store.config.enable_birth_cpe:
            birth_repair = self._birth_content_repair(
                memory=memory,
                role=role,
                counterexample=birth_counterexample,
                validation_tasks=similar,
            )
        payload = {
            "status": "calibrated",
            "similar_task_ids": [item.task_id for item in similar],
            "birth_validation_task_ids": [item.task_id for item in similar],
            "replay_ids": [record.replay_id for record in records],
            "role": role,
            "task_deltas": dict(stats.task_deltas) if stats else {},
            "mean_delta": stats.mean_delta if stats else None,
            "source_task_id": memory.base.source_task_id,
            "birth_counterexample": birth_counterexample.to_dict() if birth_counterexample else None,
            "birth_repair": birth_repair,
            "recorded_at": utc_now(),
        }
        _atomic_write_json(path, payload)
        return payload

    def process_task(
        self,
        *,
        task_id: str,
        task_index: int,
        task_description: str,
        outcome: str,
        score: float,
        trajectory: Mapping[str, Any],
        replay_context: ReplayTaskContext,
        source_entities: tuple[str, ...] = (),
    ) -> ControllerResult:
        trajectory = self._normalize_trajectory_events(trajectory)
        trajectory["source_entities"] = list(source_entities)
        partition = self.store.config.split_manifest.partition_of(task_id)
        if partition is None:
            raise ValueError(
                f"task {task_id!r} is absent from the frozen DiscernMem split manifest"
            )
        if partition.value == "final_test":
            path = (
                self.store.root
                / "audit"
                / "final_test_results"
                / f"{safe_component(task_id)}.json"
            )
            _atomic_write_json(
                path,
                {
                    "task_id": task_id,
                    "task_index": task_index,
                    "score": score,
                    "outcome": outcome,
                    "trajectory": dict(trajectory),
                    "recorded_at": utc_now(),
                },
            )
            return ControllerResult(task_id, "final_test_no_learning")

        trajectory_path = (
            self.store.root
            / "trajectories"
            / "official"
            / f"{safe_component(task_id)}.json"
        )
        _atomic_write_json(trajectory_path, dict(trajectory))
        injected_memories = tuple(
            item
            for retrieval in trajectory.get("memory_retrievals", [])
            for item in retrieval.get("injected_memories", [])
        ) or tuple(
            item
            for step in trajectory.get("steps", [])
            for item in step.get("injected_memories", [])
        )
        task_evidence = TaskEvidence(
            task_id=task_id,
            task_index=task_index,
            partition=partition.value,
            task_description=task_description,
            task_family=str(trajectory.get("task_family") or ""),
            mas_spec=dict(self.mas_spec),
            outcome=outcome,
            score=float(score),
            evaluator_feedback=str(trajectory.get("evaluator_feedback") or ""),
            trajectory_ref=str(trajectory_path),
            injected_memories=injected_memories,
            created_at=utc_now(),
        )
        existing_task = self.store.get_task(task_id)
        if existing_task is not None:
            proposed = task_evidence.to_dict()
            proposed["created_at"] = existing_task.created_at
            if proposed != existing_task.to_dict():
                raise ValueError(
                    f"task {task_id!r} was retried with different evidence"
                )
            task_evidence = existing_task
        else:
            self.store.record_task(task_evidence)
        if partition.value in {"learning", "h_candidate"}:
            summary = self.summarize_historical_task(
                task_id=task_id,
                task_index=task_index,
                task_description=task_description,
                trajectory=trajectory,
                mas_type=str(self.mas_spec.get("mas_type") or "unknown"),
                partition=partition,
            )
            self.register_historical_task(summary, replay_context)
        logger.info(
            "[DiscernMem][LEARN][task=%s] 1/4 trajectory-indexed partition=%s score=%.4f retrievals=%d injected=%d",
            task_id,
            partition.value,
            float(score),
            len(trajectory.get("memory_retrievals", []) or []),
            len(injected_memories),
        )
        if partition.value == "h_candidate":
            return ControllerResult(task_id, "indexed_h_candidate")

        formal_results = []
        formal_event_ids = []
        formal_replay_ids = []
        formal_usages = self._formal_use_contexts(trajectory)
        formal_usages, cross_role_usages = self._split_cross_role_usages(formal_usages)
        if cross_role_usages:
            logger.info(
                "[DiscernMem][FORMAL-USE][task=%s] cross-role injections not evaluated=%d targets=%s",
                task_id,
                len(cross_role_usages),
                [
                    str(item.get("memory_id"))
                    + "@v"
                    + str(item.get("version"))
                    + ":"
                    + str(item.get("owner_role"))
                    + "->"
                    + str(item.get("role"))
                    for item in cross_role_usages
                ],
            )
        formal_entries = []
        for usage in formal_usages:
            formal_inputs = {
                "task_id": task_id,
                "memory_id": usage["memory_id"],
                "version": usage["version"],
                "role": usage["role"],
                "retrieval_ids": usage["retrieval_ids"],
                "runtime_signature": replay_context.runtime_signature,
                "config": self.store.config.fingerprint,
            }
            formal_event = self.journal.begin("formal_use", task_id, formal_inputs)
            formal_entries.append((usage, formal_inputs, formal_event))
        pending_entries = [
            (formal_event.event_id, usage)
            for usage, _formal_inputs, formal_event in formal_entries
            if self.journal.get_stage(
                formal_event.event_id, "paired_assessment_and_update"
            )
            is None
        ]
        prepared_records = (
            self._run_formal_use_replay_batch(
                entries=pending_entries,
                replay_context=replay_context,
            )
            if pending_entries
            else {}
        )
        formal_groups: dict[str, list[tuple[Any, Any, Any]]] = {}
        for entry in formal_entries:
            formal_groups.setdefault(str(entry[0]["memory_id"]), []).append(entry)

        def process_formal_group(entries):
            outputs = {}
            for usage, formal_inputs, formal_event in entries:
                event_records = prepared_records.get(formal_event.event_id)
                formal_output = self.journal.run_stage(
                    formal_event,
                    "paired_assessment_and_update",
                    formal_inputs,
                    lambda usage=usage, formal_event=formal_event, event_records=event_records: self._process_formal_use(
                        task_id=task_id,
                        usage=usage,
                        trajectory=trajectory,
                        replay_context=replay_context,
                        event_id=formal_event.event_id,
                        prepared_records=event_records,
                    ),
                )
                self.journal.complete(formal_event, formal_output)
                outputs[formal_event.event_id] = dict(formal_output)
            return outputs

        formal_outputs = {}
        if len(formal_groups) <= 1:
            for entries in formal_groups.values():
                formal_outputs.update(process_formal_group(entries))
        else:
            formal_workers = min(
                len(formal_groups),
                max(1, int(getattr(self.replay_executor, "max_workers", 1))),
            )
            logger.info(
                "[DiscernMem][FORMAL-USE][task=%s] assessment groups=%d workers=%d same-memory-serial=true",
                task_id,
                len(formal_groups),
                formal_workers,
            )
            with ThreadPoolExecutor(
                max_workers=formal_workers,
                thread_name_prefix="discernmem-formal",
            ) as pool:
                futures = [
                    pool.submit(process_formal_group, tuple(entries))
                    for entries in formal_groups.values()
                ]
                for future in futures:
                    formal_outputs.update(future.result())

        for usage, _formal_inputs, formal_event in formal_entries:
            formal_output = formal_outputs[formal_event.event_id]
            formal_results.append(dict(formal_output))
            formal_event_ids.append(formal_event.event_id)
            formal_replay_ids.extend(formal_output.get("replay_ids") or [])
            logger.info(
                "[DiscernMem][FORMAL-USE][task=%s][memory=%s@v%s][role=%s] status=%s routing=%s replays=%d",
                task_id,
                usage["memory_id"],
                usage["version"],
                usage["role"],
                formal_output.get("status") or "-",
                formal_output.get("routing") or "-",
                len(formal_output.get("replay_ids") or []),
            )

        attributed_content_updates = [
            item
            for item in formal_results
            if item.get("routing") == CounterexampleKind.CONTENT.value
        ]
        if attributed_content_updates:
            logger.info(
                "[DiscernMem][LEARN][task=%s] skipping new-memory extraction; "
                "paired replay attributed the failure to %d existing memory use(s)",
                task_id,
                len(attributed_content_updates),
            )
            return ControllerResult(
                task_id,
                "formal_use_processed",
                replay_ids=tuple(dict.fromkeys(formal_replay_ids)),
                formal_use_event_ids=tuple(formal_event_ids),
            )

        inputs = {
            "task_id": task_id,
            "trajectory_sha256": hashlib.sha256(
                json.dumps(trajectory, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "config": self.store.config.fingerprint,
        }
        event = self.journal.begin("new_memory", task_id, inputs)

        def extract_operation():
            extraction_payload = self._extract_payload(
                task_id=task_id,
                task_description=task_description,
                outcome=outcome,
                score=score,
                trajectory=trajectory,
            )
            extraction_payload["task"]["source_entities"] = list(source_entities)
            path = (
                self.store.root
                / "audit"
                / "extractions"
                / f"{safe_component(event.event_id)}.json"
            )
            supports_target_role = (
                "target_role"
                in inspect.signature(self.extraction.extract_and_validate).parameters
            )
            role_specs = [
                dict(item)
                for item in self.mas_spec.get("roles", []) or []
                if str(item.get("name") or item.get("role") or "")
            ]
            if not supports_target_role:
                role_specs = [{}]
            attempts = []
            candidates = []
            for role_spec in role_specs:
                role = str(role_spec.get("name") or role_spec.get("role") or "")
                call_kwargs = {
                    "artifact_name": (
                        f"{event.event_id}_{safe_component(role)}"
                        if role
                        else event.event_id
                    ),
                    "source_entities": source_entities,
                }
                if supports_target_role:
                    call_kwargs["target_role"] = role
                try:
                    parsed, candidate, _errors = self.extraction.extract_and_validate(
                        extraction_payload,
                        **call_kwargs,
                    )
                    attempt = {
                        "role": role or None,
                        "parsed": parsed,
                        "candidate": candidate.to_dict() if candidate else None,
                        "status": "candidate" if candidate else "no_memory",
                    }
                    if candidate:
                        candidates.append(candidate.to_dict())
                except DiscernMemPromptError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    attempt = {
                        "role": role or None,
                        "parsed": None,
                        "candidate": None,
                        "status": "failed",
                        "error": error,
                    }
                    logger.warning(
                        "[DiscernMem][LEARN][task=%s][role=%s] extraction failed; %s",
                        task_id,
                        role or "legacy",
                        _log_summary(exc),
                    )
                attempts.append(attempt)
            _atomic_write_json(
                path,
                {
                    "role_extractions": attempts,
                    "candidates": candidates,
                    "candidate": candidates[0] if len(candidates) == 1 else None,
                    "error": next(
                        (
                            str(item.get("error") or "")
                            for item in attempts
                            if item.get("error")
                        ),
                        "",
                    ),
                    "prompt_quality_mode": "prompt_only",
                    "extraction_status": (
                        "failed"
                        if attempts
                        and all(item["status"] == "failed" for item in attempts)
                        else "completed"
                    ),
                },
            )
            return {
                "candidates": candidates,
                "prompt_quality_mode": "prompt_only",
                "extraction_status": (
                    "failed"
                    if attempts and all(item["status"] == "failed" for item in attempts)
                    else "completed"
                ),
                "role_statuses": [
                    {"role": item["role"], "status": item["status"]}
                    for item in attempts
                ],
            }

        extraction_output = self.journal.run_stage(
            event, "extract", inputs, extract_operation
        )
        candidate_payloads = list(extraction_output.get("candidates") or [])
        if candidate_payloads:
            logger.info(
                "[DiscernMem][LEARN][task=%s] 2/4 extraction candidates=%d roles=%s",
                task_id,
                len(candidate_payloads),
                [tuple(item.get("roles") or ()) for item in candidate_payloads],
            )
        else:
            logger.info(
                "[DiscernMem][LEARN][task=%s] 2/4 extraction candidates=0 statuses=%s",
                task_id,
                extraction_output.get("role_statuses") or [],
            )
        if not candidate_payloads:
            status = (
                "formal_use_processed"
                if formal_results
                else (
                    "extraction_failed"
                    if extraction_output.get("extraction_status") == "failed"
                    else "candidate_rejected"
                )
            )
            summary = {
                "status": status,
                "formal_use_results": formal_results,
                "formal_use_event_ids": formal_event_ids,
            }
            self.journal.complete(event, summary)
            return ControllerResult(
                task_id,
                status,
                replay_ids=tuple(dict.fromkeys(formal_replay_ids)),
                event_id=event.event_id,
                formal_use_event_ids=tuple(formal_event_ids),
            )

        from .extraction import ExtractionCandidate

        admissions = []
        for candidate_index, candidate_payload in enumerate(candidate_payloads):
            candidate = ExtractionCandidate.from_dict(candidate_payload)

            def admission_operation(
                candidate=candidate, candidate_index=candidate_index
            ):
                record = self.admission.admit(
                    candidate,
                    source_task_id=task_id,
                    source_event_ids=self._source_event_ids(trajectory),
                    mas_spec=self.mas_spec,
                    source_entities=source_entities,
                    source_trajectory=trajectory,
                    source_task={
                        "task_id": task_id,
                        "description": task_description,
                        "task_family": str(trajectory.get("task_family") or ""),
                        "outcome": outcome,
                        "score": float(score),
                        "source_entities": list(source_entities),
                        "trajectory_ref": str(trajectory_path),
                    },
                    artifact_name=f"{event.event_id}_{candidate_index}",
                )
                return record.to_dict()

            admission_output = self.journal.run_stage(
                event,
                "admit" if len(candidate_payloads) == 1 else f"admit_{candidate_index}",
                {"candidate": candidate.to_dict()},
                admission_operation,
            )
            decision = str(admission_output["decision"])
            memory_id = str(admission_output.get("target_memory_id") or "")
            state = ""
            birth_output = {}
            if decision == AdmissionDecision.CREATE.value:
                memory = self._memory_for_admission(
                    str(admission_output["admission_id"])
                )
                if memory is None:
                    raise RuntimeError(
                        "create admission did not produce a DiscernMem identity"
                    )
                memory_id = memory.memory_id
                state = memory.state.value
                birth_output = self._birth_calibration(
                    task_id=task_id,
                    task_index=task_index,
                    task_description=task_description,
                    memory=memory,
                    replay_context=replay_context,
                    source_trajectory=trajectory,
                )
                if birth_output.get("birth_repair", {}).get("status") == "birth_cpe_committed":
                    refreshed = self.store.get_version(memory_id)
                    if refreshed is not None:
                        memory = refreshed
                        state = memory.state.value
            elif decision == AdmissionDecision.ATTACH.value:
                attached = self.store.get_version(memory_id)
                attached_version = attached.version
                role = candidate.roles[0]
                already_processed = any(
                    str(usage.get("memory_id") or "") == memory_id
                    and int(usage.get("version", -1)) == attached.version
                    and str(usage.get("role") or "") == role
                    for usage in formal_usages
                )
                if not already_processed:
                    attach_artifact_id = stable_id(
                        "attach-followup",
                        {
                            "event_id": event.event_id,
                            "admission_id": admission_output["admission_id"],
                            "memory_id": memory_id,
                            "version": attached.version,
                            "role": role,
                        },
                    )
                    attach_inputs = {
                        "admission_id": admission_output["admission_id"],
                        "memory_id": memory_id,
                        "version": attached.version,
                        "role": role,
                        "runtime_signature": replay_context.runtime_signature,
                    }
                    attach_output = self.journal.run_stage(
                        event,
                        f"attach_followup_{candidate_index}",
                        attach_inputs,
                        lambda role=role, attach_artifact_id=attach_artifact_id, candidate=candidate, attached=attached: self._process_formal_use(
                            task_id=task_id,
                            usage={
                                "memory_id": memory_id,
                                "version": attached.version,
                                "role": role,
                                "pre_action_contexts": [
                                    {
                                        "task_description": task_description,
                                        "observable_state": " ".join(
                                            candidate.use_when
                                        ),
                                        "decision_point": candidate.body,
                                    }
                                ],
                                "retrieval_ids": [
                                    f"admission:{admission_output['admission_id']}"
                                ],
                            },
                            trajectory=trajectory,
                            replay_context=replay_context,
                            event_id=attach_artifact_id,
                        ),
                    )
                    formal_event_ids.append(attach_artifact_id)
                    formal_replay_ids.extend(attach_output.get("replay_ids") or [])
                state = self.store.get_version(memory_id).state.value
            admissions.append(
                {
                    **dict(admission_output),
                    "memory_id": memory_id,
                    "state": state,
                    "birth_calibration": locals().get("birth_output", {}),
                }
            )
            logger.info(
                "[DiscernMem][LEARN][task=%s][role=%s] 3/4 admission=%s target=%s",
                task_id,
                candidate.roles[0],
                decision,
                memory_id or "-",
            )

        primary = admissions[0]

        result = ControllerResult(
            task_id=task_id,
            status="processed",
            admission_decision=str(primary["decision"]),
            memory_id=str(primary["memory_id"]),
            replay_ids=tuple(dict.fromkeys(formal_replay_ids)),
            state=str(primary["state"]),
            event_id=event.event_id,
            formal_use_event_ids=tuple(formal_event_ids),
        )
        self.journal.complete(
            event,
            {
                **result.to_dict(),
                "admissions": admissions,
                "role_statuses": extraction_output.get("role_statuses") or [],
            },
        )
        return result
