from __future__ import annotations

import copy
import json
import time
from typing import Any, Mapping

import jsonschema

from common.llm_client import LLMClient

from .counterexamples import ContentCounterexample
from .extraction import RequiredHandoff
from .ids import stable_id
from .schema import CapabilityConstraint, EditOperation, MetaMemory, RepairCandidate
from .schema_compat import guided_decoding_schema, normalize_unique_items
from .prompting import load_prompt, load_schema, render_user_prompt
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now


class RepairGenerationError(RuntimeError):
    pass


def materialize_repair_body(
    body: str,
    *,
    operation: EditOperation,
    anchor: str,
    edit_text: str,
) -> str:
    """Apply one unambiguous local edit and return the complete candidate body."""
    parent = str(body).strip()
    target = str(anchor).strip()
    replacement = str(edit_text).strip()
    occurrences = parent.count(target) if target else 0
    if occurrences == 0:
        raise ValueError("edit target is absent from the parent body")
    if occurrences != 1:
        raise ValueError("edit target is ambiguous in the parent body")

    if operation is EditOperation.DELETE:
        if replacement:
            raise ValueError("DELETE requires empty edit_text")
        materialized = parent.replace(target, "", 1)
    elif operation is EditOperation.ADD:
        if not replacement:
            raise ValueError("ADD requires non-empty edit_text")
        insertion = parent.index(target) + len(target)
        materialized = f"{parent[:insertion]} {replacement} {parent[insertion:]}"
    else:
        if not replacement:
            raise ValueError(f"{operation.value} requires non-empty edit_text")
        materialized = parent.replace(target, replacement, 1)

    materialized = " ".join(materialized.split())
    if not materialized:
        raise ValueError("materialized body must not be empty")
    if materialized == parent:
        raise ValueError("materialized body is unchanged from the parent body")
    return materialized


class DiscernMemRepairGenerator:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: float = 180.0,
        max_tokens: int = 2200,
    ):
        self.store = store
        self.schema = load_schema("memory_repair.schema.json")
        jsonschema.Draft202012Validator.check_schema(self.schema)
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            retry_count=1,
            timeout=timeout,
            logger_name="discernmem.repair",
            stats_source="discernmem/repair_generator",
        )

    def generate(
        self,
        *,
        memory_id: str,
        parent_version: int,
        body: str,
        meta: MetaMemory,
        counterexample: ContentCounterexample,
        mas_spec: Mapping[str, Any],
        capabilities: tuple[CapabilityConstraint, ...],
        trajectory_evidence: Mapping[str, Any],
        origin_evidence: Mapping[str, Any] | None = None,
        source_entities: tuple[str, ...] = (),
        artifact_name: str,
        round_index: int = 1,
        prior_rounds: tuple[Mapping[str, Any], ...] = (),
    ) -> tuple[RepairCandidate, ...]:
        payload = {
            "memory": {
                "memory_id": memory_id,
                "parent_version": parent_version,
                "memory": body,
                "roles": list(meta.roles),
                "use_when": list(meta.use_when),
                "avoid_when": list(meta.avoid_when),
            },
            "content_counterexample": counterexample.to_dict(),
            "mas": dict(mas_spec),
            "capabilities": [item.to_dict() for item in capabilities],
            "trajectory_evidence": dict(trajectory_evidence),
            "origin_evidence": dict(origin_evidence or {}),
            "source_entities": list(source_entities),
            "evolution": {
                "round_index": int(round_index),
                "prior_rounds": [dict(item) for item in prior_rounds],
            },
        }
        system = load_prompt(
            "memory_repair.md",
            mas_type=str(mas_spec.get("mas_type") or ""),
            include_mas_guidance=True,
            enable_mas_guidance=self.store.config.enable_mas_guidance,
        )
        base_user = render_user_prompt("memory_repair.md", payload)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "discernmem_memory_repair",
                "schema": guided_decoding_schema(self.schema),
                "strict": True,
            },
        }
        candidate_schema = self.schema["properties"]["candidates"]["items"]
        envelope_schema = copy.deepcopy(self.schema)
        envelope_schema["properties"]["candidates"]["items"] = {"type": "object"}
        evidence_allowed = set(counterexample.trajectory_evidence_refs)
        capability_ids = {item.capability_id for item in capabilities}
        roles = {
            str(item.get("name") or item.get("role") or "")
            for item in mas_spec.get("roles", [])
        }
        edges = {
            (str(edge[0]), str(edge[1]))
            for edge in mas_spec.get("edges", [])
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        }
        audit_path = (
            self.store.root
            / "audit"
            / "repair_generators"
            / f"{safe_component(artifact_name)}.json"
        )
        attempts: list[dict[str, Any]] = []
        retry_errors: list[str] = []

        def schema_error(exc: jsonschema.ValidationError) -> str:
            location = ".".join(str(value) for value in exc.absolute_path)
            return f"{location or '<root>'}: {exc.message}"

        def write_audit(status: str) -> None:
            _atomic_write_json(
                audit_path,
                {
                    "timestamp": utc_now(),
                    "input": payload,
                    "system_prompt": system,
                    "schema": self.schema,
                    "attempts": attempts,
                    "status": status,
                    "model": self.llm.model,
                },
            )

        for attempt in range(2):
            retry_note = ""
            if retry_errors:
                retry_note = (
                    "\n\nThe previous repair output was rejected. Return a complete "
                    "replacement JSON object. Keep any sound repair idea, but correct "
                    "each exact validation error below."
                    "\nValidation errors:\n- " + "\n- ".join(retry_errors)
                )
            current_user = base_user + retry_note
            started = time.perf_counter()
            raw = ""
            parsed: dict[str, Any] | None = None
            rejected: list[dict[str, Any]] = []
            accepted: list[RepairCandidate] = []
            overall_errors: list[str] = []
            try:
                raw = self.llm.chat(
                    system,
                    current_user,
                    temperature=0.0,
                    response_format=response_format,
                )
                parsed_value = normalize_unique_items(json.loads(raw), self.schema)
                jsonschema.Draft202012Validator(envelope_schema).validate(parsed_value)
                parsed = dict(parsed_value)
                raw_candidates = list(parsed.get("candidates") or [])
                if len(raw_candidates) > self.store.config.max_repair_candidates:
                    overall_errors.append(
                        "candidates: exceeds configured candidate budget "
                        f"{self.store.config.max_repair_candidates}"
                    )
            except jsonschema.ValidationError as exc:
                overall_errors.append(schema_error(exc))
                raw_candidates = []
            except Exception as exc:
                overall_errors.append(f"{type(exc).__name__}: {exc}")
                raw_candidates = []

            if not overall_errors:
                seen_bodies: set[str] = set()
                for index, item in enumerate(raw_candidates):
                    errors = [
                        schema_error(exc)
                        for exc in sorted(
                            jsonschema.Draft202012Validator(
                                candidate_schema
                            ).iter_errors(item),
                            key=lambda exc: list(exc.absolute_path),
                        )
                    ]
                    if errors:
                        rejected.append({"index": index, "errors": errors})
                        continue
                    operation = EditOperation(item["edit_operation"])
                    edit_text = str(item["edit_text"]).strip()
                    anchor = str(item["old_text_or_anchor"]).strip()
                    try:
                        edited_body = materialize_repair_body(
                            body,
                            operation=operation,
                            anchor=anchor,
                            edit_text=edit_text,
                        )
                    except ValueError as exc:
                        errors.append(str(exc))
                        edited_body = ""
                    if edited_body and edited_body in seen_bodies:
                        errors.append(
                            "materialized body duplicates an earlier valid candidate"
                        )
                    evidence_refs = tuple(
                        str(value) for value in item.get("evidence_refs", [])
                    )
                    if not set(evidence_refs).issubset(evidence_allowed):
                        errors.append(
                            "evidence_refs cite events outside the counterexample"
                        )
                    claimed = tuple(
                        str(value)
                        for value in item.get("claimed_preserved_capability_ids", [])
                    )
                    if not capability_ids:
                        # Initial v0 memories have no verified capabilities;
                        # ignore spurious model claims rather than blocking the edit.
                        claimed = ()
                    elif not set(claimed).issubset(capability_ids):
                        errors.append("claims an unknown preserved capability")
                    try:
                        handoffs = tuple(
                            RequiredHandoff.from_dict(value)
                            for value in item.get("required_handoffs", [])
                        )
                    except Exception as exc:
                        errors.append(f"invalid required_handoffs: {exc}")
                        handoffs = ()
                    for handoff in handoffs:
                        if (
                            handoff.from_role not in roles
                            or handoff.to_role not in roles
                        ):
                            errors.append(
                                "uses an unknown handoff endpoint "
                                f"{handoff.from_role}->{handoff.to_role}"
                            )
                        elif (
                            self.store.config.enable_required_handoffs
                            and (
                                handoff.from_role,
                                handoff.to_role,
                            )
                            not in edges
                        ):
                            errors.append(
                                "uses an absent handoff edge "
                                f"{handoff.from_role}->{handoff.to_role}"
                            )
                    if errors:
                        rejected.append({"index": index, "errors": errors})
                        continue
                    candidate_id = stable_id(
                        "candidate",
                        {
                            "memory_id": memory_id,
                            "parent_version": parent_version,
                            "diagnosis": item["diagnosis"],
                            "operation": operation.value,
                            "edited_body": edited_body,
                            "counterexample": counterexample.task_id,
                        },
                    )
                    candidate = self.store.get_candidate(candidate_id)
                    if candidate is None:
                        candidate = RepairCandidate(
                            candidate_id=candidate_id,
                            memory_id=memory_id,
                            parent_version=parent_version,
                            diagnosis=str(item["diagnosis"]),
                            edit_operation=operation,
                            edited_body=edited_body,
                            evidence_refs=evidence_refs,
                            claimed_preserved_capability_ids=claimed,
                            expected_change=str(item["expected_change"]),
                            created_at=utc_now(),
                            audit={
                                "round_index": int(round_index),
                                "old_text_or_anchor": anchor,
                                "edit_text": edit_text,
                                "required_handoffs": [
                                    handoff.__dict__ for handoff in handoffs
                                ],
                                "counterexample_task_id": counterexample.task_id,
                            },
                        )
                        self.store.record_candidate(candidate)
                    accepted.append(candidate)
                    seen_bodies.add(edited_body)

            attempts.append(
                {
                    "timestamp": utc_now(),
                    "attempt": attempt + 1,
                    "user_prompt": current_user,
                    "raw_output": raw,
                    "parsed_output": parsed,
                    "overall_errors": overall_errors,
                    "rejected_candidates": rejected,
                    "accepted_candidate_ids": [item.candidate_id for item in accepted],
                    "latency_seconds": time.perf_counter() - started,
                    "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                }
            )
            if accepted:
                write_audit("accepted")
                return tuple(accepted)
            if (
                not overall_errors
                and not raw_candidates
                and str((parsed or {}).get("no_update_reason") or "").strip()
            ):
                write_audit("no_update")
                return ()
            retry_errors = list(overall_errors)
            retry_errors.extend(
                f"candidate {item['index']}: " + "; ".join(item["errors"])
                for item in rejected
            )
            if not retry_errors:
                retry_errors.append(
                    "empty repair set requires a concrete no_update_reason"
                )
            write_audit("retrying" if attempt == 0 else "failed")

        raise RepairGenerationError(
            "repair generation failed after 2 attempts: " + "; ".join(retry_errors)
        )
