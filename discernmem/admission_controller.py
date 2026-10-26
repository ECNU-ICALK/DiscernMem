from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional

from .extraction import ExtractionCandidate
from .ids import stable_id
from .invariants import InvariantViolation, require_single_role
from .semantic_index import RetrievedIdentity, DiscernMemSemanticIndex
from .schema import AdmissionDecision, AdmissionRecord
from .storage import (
    DiscernMemStore,
    _atomic_write_json,
    read_json,
    safe_component,
    utc_now,
)

IdentityJudge = Callable[..., Mapping[str, Any]]


class DiscernMemAdmission:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        semantic_index: DiscernMemSemanticIndex,
        identity_judge: IdentityJudge,
        top_k: Optional[int] = None,
    ):
        self.store = store
        self.semantic_index = semantic_index
        self.identity_judge = identity_judge
        self.top_k = int(top_k or store.config.admission_top_k)

    def _admission_id(self, source_task_id: str, candidate: ExtractionCandidate) -> str:
        return stable_id(
            "adm",
            {
                "admission_policy": "identity_similarity_first_v2",
                "source_task_id": source_task_id,
                "body": candidate.memory,
                "roles": candidate.roles,
                "use_when": candidate.use_when,
                "evidence_refs": candidate.evidence_refs,
                "config": self.store.config.fingerprint,
            },
        )

    def admit(
        self,
        candidate: ExtractionCandidate,
        *,
        source_task_id: str,
        source_event_ids: Iterable[str],
        mas_spec: Mapping[str, Any],
        source_entities: tuple[str, ...] = (),
        source_trajectory: Optional[Mapping[str, Any]] = None,
        source_task: Optional[Mapping[str, Any]] = None,
        artifact_name: Optional[str] = None,
    ) -> AdmissionRecord:
        candidate_role = require_single_role(
            candidate.roles, context="admission candidate"
        )
        admission_id = self._admission_id(source_task_id, candidate)
        admission_path = (
            self.store.root / "admissions" / f"{safe_component(admission_id)}.json"
        )
        if admission_path.exists():
            data = read_json(admission_path)
            cached = AdmissionRecord(
                admission_id=data["admission_id"],
                source_task_id=data["source_task_id"],
                candidate=dict(data["candidate"]),
                decision=AdmissionDecision(data["decision"]),
                reason=data["reason"],
                compared_memory_ids=tuple(data.get("compared_memory_ids", [])),
                target_memory_id=data.get("target_memory_id", ""),
                validation_status=data.get("validation_status", "not_evaluated"),
                created_at=data.get("created_at", ""),
            )
            if cached.decision is AdmissionDecision.ATTACH:
                target = self.store.get_version(cached.target_memory_id)
                if target is None or tuple(target.meta.roles) != (candidate_role,):
                    raise InvariantViolation(
                        "cached admission attachment violates the single-role "
                        f"identity namespace for role {candidate_role!r}"
                    )
            return cached
        compared: list[RetrievedIdentity] = []
        target = ""
        reason = "prompt-produced candidate; identity recall pending"
        decision = AdmissionDecision.CREATE
        versions = self.store.iter_current_versions(
            states={"active", "provisional", "inactive"}
        )
        versions = [
            version
            for version in versions
            if tuple(version.meta.roles) == (candidate_role,)
        ]
        query = " ".join((candidate.memory, *candidate.use_when))
        if self.store.config.identity_recall_mode == "lexical":
            compared = self.semantic_index.retrieve_lexical(
                query, versions, top_k=self.top_k
            )
        else:
            compared = self.semantic_index.retrieve(query, versions, top_k=self.top_k)
        comparison_payload = [
            {
                "memory_id": item.memory_id,
                "version": item.version,
                "state": item.state,
                "memory": item.memory,
                "use_when": list(item.use_when),
                "semantic_score": item.score,
                "recall_mode": self.store.config.identity_recall_mode,
            }
            for item in compared
        ]
        if comparison_payload:
            judgment = dict(
                self.identity_judge(
                    candidate,
                    comparison_payload,
                    artifact_name=artifact_name or admission_id,
                )
            )
            if judgment.get("_judgment_valid") is False:
                decision = AdmissionDecision.DISCARD
                reason = str(
                    judgment.get("reason")
                    or "identity judgment failed; candidate was not admitted"
                )
            elif bool(judgment.get("same_operation")):
                target = str(judgment.get("target_memory_id") or "")
                recalled_ids = {item["memory_id"] for item in comparison_payload}
                if target in recalled_ids:
                    decision = AdmissionDecision.ATTACH
                    reason = str(judgment.get("reason") or "same executable operation")
                else:
                    target = ""
                    reason = (
                        "identity judge selected a target outside semantic recall; "
                        "treated as a novel identity"
                    )
            elif judgment.get("target_memory_id"):
                reason = (
                    "identity judge returned a target for a novel rule; "
                    "treated as a novel identity"
                )
            else:
                reason = str(judgment.get("reason") or "novel operation")
        else:
            reason = "prompt-produced candidate has no recalled identity"

        record = AdmissionRecord(
            admission_id=admission_id,
            source_task_id=source_task_id,
            candidate=candidate.to_dict(),
            decision=decision,
            reason=reason,
            compared_memory_ids=tuple(item.memory_id for item in compared),
            target_memory_id=target,
            validation_status="not_evaluated",
            created_at=utc_now(),
        )
        self.store.record_admission(record)
        if decision is AdmissionDecision.CREATE:
            self.store.create_memory(
                memory_id=stable_id(
                    "mem",
                    {"admission_id": admission_id, "candidate_body": candidate.memory},
                ),
                body=candidate.memory,
                meta=candidate.initial_meta,
                source_task_id=source_task_id,
                source_trajectory_ref=str(
                    (source_task or {}).get("trajectory_ref") or ""
                ),
                source_event_ids=source_event_ids,
                semantic_key=" ".join((candidate.memory, *candidate.use_when)),
                audit={"admission_id": admission_id},
            )
        elif decision is AdmissionDecision.ATTACH:
            _atomic_write_json(
                self.store.root
                / "audit"
                / "attachments"
                / f"{safe_component(admission_id)}.json",
                {
                    "admission_id": admission_id,
                    "memory_id": target,
                    "source_task_id": source_task_id,
                    "candidate": candidate.to_dict(),
                    "recorded_at": utc_now(),
                },
            )
        return record
