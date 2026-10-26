from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import uuid

from .ids import stable_id
from .invariants import retrieval_allowed
from .schema import MemoryVersion
from .semantic_index import DiscernMemSemanticIndex
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now
from .task_family import infer_task_family, memory_source_task_family, task_families_compatible


_FORBIDDEN_CONTEXT_KEYS = {
    "final_score",
    "score",
    "evaluator_feedback",
    "ground_truth",
    "hidden_answer",
    "future_messages",
    "future_trajectory",
    "outcome",
}


@dataclass(frozen=True)
class PreActionContext:
    task_description: str
    role: str
    observable_state: str = ""
    received_messages: tuple[str, ...] = ()
    intermediate_results: tuple[str, ...] = ()
    decision_point: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PreActionContext":
        forbidden = sorted(_FORBIDDEN_CONTEXT_KEYS & set(data))
        if forbidden:
            raise ValueError(
                f"pre-action retrieval contains forbidden future fields: {forbidden}"
            )
        metadata = dict(data.get("metadata") or {})
        nested_forbidden = sorted(_FORBIDDEN_CONTEXT_KEYS & set(metadata))
        if nested_forbidden:
            raise ValueError(
                f"pre-action metadata contains forbidden future fields: {nested_forbidden}"
            )
        return cls(
            task_description=str(data.get("task_description") or ""),
            role=str(data.get("role") or ""),
            observable_state=str(data.get("observable_state") or ""),
            received_messages=tuple(
                str(value) for value in data.get("received_messages", [])
            ),
            intermediate_results=tuple(
                str(value) for value in data.get("intermediate_results", [])
            ),
            decision_point=str(data.get("decision_point") or ""),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "role": self.role,
            "observable_state": self.observable_state,
            "received_messages": list(self.received_messages),
            "intermediate_results": list(self.intermediate_results),
            "decision_point": self.decision_point,
            "metadata": dict(self.metadata),
        }

    def semantic_text(self) -> str:
        visible_values = (
            f"Current MAS role: {self.role}",
            self.task_description,
            self.observable_state,
            *self.received_messages,
            *self.intermediate_results,
            self.decision_point,
        )
        return "\n".join(dict.fromkeys(value for value in visible_values if value))


@dataclass(frozen=True)
class RetrievalResult:
    retrieval_id: str
    selected: tuple[MemoryVersion, ...]
    memory_text: str
    audit_path: str


def format_deployed_memories(versions: tuple[MemoryVersion, ...]) -> str:
    if not versions:
        return ""
    lines = [
        "# DiscernMem retrieved references",
        "",
        "These memories were selected by semantic similarity as optional auxiliary references.",
        "Use semantic judgment to decide whether each memory fits the current observable",
        "state. The current task state and environment feedback are authoritative.",
    ]
    for version in versions:
        lines.extend(
            [
                "",
                f"## Memory {version.memory_id} / version {version.version}",
                "Memory:",
                version.memory,
                "Roles:",
                *(
                    [f"- {value}" for value in version.meta.roles]
                    or ["- No role recorded."]
                ),
                "Use when:",
                *(
                    [f"- {value}" for value in version.meta.use_when]
                    or ["- Relevant to the observable state."]
                ),
                "Avoid when:",
                *(
                    [f"- {value}" for value in version.meta.avoid_when]
                    or ["- No verified avoid condition."]
                ),
            ]
        )
    return "\n".join(lines).strip()


def format_deployed_bodies(versions: tuple[MemoryVersion, ...]) -> str:
    """Render memory bodies only; used by the meta-memory ablation switch."""
    if not versions:
        return ""
    lines = [
        "# DiscernMem retrieved references",
        "",
        "These memories were selected by semantic similarity as optional auxiliary references.",
        "Use semantic judgment to decide whether each memory fits the current observable",
        "state. The current task state and environment feedback are authoritative.",
    ]
    for version in versions:
        lines.extend(
            [
                "",
                f"## Memory {version.memory_id} / version {version.version}",
                "Memory:",
                version.memory,
            ]
        )
    return "\n".join(lines).strip()


class DiscernMemRetriever:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        semantic_index: DiscernMemSemanticIndex,
        selector=None,
        top_k: Optional[int] = None,
        max_injected: Optional[int] = None,
    ):
        self.store = store
        self.semantic_index = semantic_index
        self.selector = selector
        self.top_k = int(top_k or store.config.retrieval_top_k)
        self.max_injected = int(max_injected or store.config.max_injected_memories)

    def retrieve(self, context: PreActionContext) -> RetrievalResult:
        if not context.role:
            raise ValueError("pre-action retrieval requires a real role")
        deployable_pool = [
            version
            for version in self.store.iter_current_versions()
            if retrieval_allowed(
                version.state,
                provisional_enabled=self.store.config.enable_provisional_retrieval,
            )
            and (
                not self.store.config.enable_role_conditioning
                or tuple(version.meta.roles) == (context.role,)
            )
        ]
        current_task_family = infer_task_family(
            context.task_description, context.metadata
        )
        family_filter_enabled = self.store.config.enable_task_family_filter
        if current_task_family and family_filter_enabled:
            pool = [
                version
                for version in deployable_pool
                if task_families_compatible(
                    current_task_family,
                    memory_source_task_family(self.store, version),
                )
            ]
        else:
            pool = deployable_pool
        selector_enabled = (
            self.selector is not None
            and self.store.config.enable_retrieval_selector
        )
        recall_deployment = (
            self.semantic_index.retrieve_for_deployment
            if self.store.config.enable_use_avoid_conditions
            else self.semantic_index.retrieve_for_deployment_body_only
        )
        recalled = recall_deployment(
            context.semantic_text(),
            pool,
            top_k=(self.top_k if selector_enabled else min(self.top_k, self.max_injected)),
        )
        min_similarity = float(self.store.config.min_retrieval_similarity)
        recalled = tuple(item for item in recalled if item.score >= min_similarity)
        versions = {version.memory_id: version for version in pool}
        candidate_payload = [
            {
                "memory_id": item.memory_id,
                "version": item.version,
                "memory": item.memory,
                "use_when": list(item.use_when),
                "avoid_when": list(versions[item.memory_id].meta.avoid_when),
                "semantic_score": item.score,
            }
            for item in recalled
        ]
        selector_status = "disabled"
        selector_raw_output = ""
        selector_usage: Mapping[str, Any] = {}
        if selector_enabled:
            decision = self.selector.select(
                context=context.to_dict(), candidates=candidate_payload
            )
            selected_ids = set(decision.selected_ids)
            selector_status = decision.status
            selector_raw_output = decision.raw_output
            selector_usage = dict(decision.actual_usage or {})
            recalled = tuple(
                item for item in recalled if item.memory_id in selected_ids
            )[: self.max_injected]
        else:
            recalled = recalled[: self.max_injected]
        logical_retrieval_id = stable_id(
            "ret",
            {
                "context": context.to_dict(),
                "candidate_versions": [
                    (item["memory_id"], item["version"]) for item in candidate_payload
                ],
                "config": self.store.config.fingerprint,
            },
        )
        retrieval_id = stable_id(
            "retcall",
            {
                "logical_retrieval_id": logical_retrieval_id,
                "occurrence_nonce": uuid.uuid4().hex,
            },
        )
        selected = tuple(versions[item.memory_id] for item in recalled)
        memory_text = (
            format_deployed_memories(selected)
            if self.store.config.enable_use_avoid_conditions
            else format_deployed_bodies(selected)
        )
        audit_path = (
            self.store.root
            / "audit"
            / "retrievals"
            / f"{safe_component(retrieval_id)}.json"
        )
        _atomic_write_json(
            audit_path,
            {
                "retrieval_id": retrieval_id,
                "logical_retrieval_id": logical_retrieval_id,
                "timestamp": utc_now(),
                "pre_action_context": context.to_dict(),
                "selection_policy": (
                    "role_then_semantic_top_k_then_llm"
                    if selector_enabled
                    else "role_then_semantic_top_k"
                ),
                "min_similarity": min_similarity,
                "role_filter": context.role,
                "task_family_filter": (
                    current_task_family if family_filter_enabled else None
                ),
                "task_family_filter_enabled": family_filter_enabled,
                "candidate_pool": [
                    {
                        "memory_id": version.memory_id,
                        "version": version.version,
                        "state": version.state.value,
                    }
                    for version in pool
                ],
                "semantic_recall": candidate_payload,
                "selector": {
                    "status": selector_status,
                    "selected_ids": [item.memory_id for item in recalled],
                    "raw_output": selector_raw_output,
                    "actual_usage": dict(selector_usage),
                },
                "selected": [
                    {
                        "memory_id": version.memory_id,
                        "version": version.version,
                        "semantic_score": next(
                            item.score
                            for item in recalled
                            if item.memory_id == version.memory_id
                            and item.version == version.version
                        ),
                    }
                    for version in selected
                ],
            },
        )
        return RetrievalResult(
            retrieval_id=retrieval_id,
            selected=selected,
            memory_text=memory_text,
            audit_path=str(audit_path),
        )
