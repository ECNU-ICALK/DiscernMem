from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import jsonschema

from common.llm_client import LLMClient

from .counterexamples import UsageAssessment
from .schema import MemoryVersion, ReplayRecord
from .schema_compat import guided_decoding_schema, normalize_unique_items
from .storage import (
    DiscernMemStore,
    _atomic_write_json,
    read_json,
    safe_component,
    utc_now,
)
from .trajectory_projection import (
    project_assessment_trajectory,
    project_pre_action_context,
)
from .prompting import load_prompt, load_schema, render_user_prompt


logger = logging.getLogger(__name__)


class UsageAssessmentError(RuntimeError):
    def __init__(self, message: str, *, attempts=()):
        super().__init__(message)
        self.attempts = tuple(dict(item) for item in attempts)


class DiscernMemUsageAssessor:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: float = 180.0,
        max_tokens: int = 1200,
        max_input_chars: int = 120000,
    ):
        self.store = store
        self.deployment_system_prompt = load_prompt("deployment_assessment.md")
        self.content_system_prompt = load_prompt("formal_use_content_diagnosis.md")
        self.output_schema = load_schema("formal_use_assessment.schema.json")
        self.deployment_schema = load_schema("formal_use_deployment.schema.json")
        self.content_schema = load_schema("formal_use_content_diagnosis.schema.json")
        for schema in (
            self.output_schema,
            self.deployment_schema,
            self.content_schema,
        ):
            jsonschema.Draft202012Validator.check_schema(schema)
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            retry_count=1,
            timeout=timeout,
            max_input_chars=max_input_chars,
            logger_name="discernmem.usage_assessor",
            stats_source="discernmem/usage_assessor",
        )
        self.max_input_chars = int(max_input_chars)

    @staticmethod
    def _trajectory(record: ReplayRecord) -> Mapping[str, Any]:
        path = Path(record.trace_ref)
        return read_json(path) if path.exists() else {}

    @staticmethod
    def _response_format(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": guided_decoding_schema(schema),
                "strict": True,
            },
        }

    @staticmethod
    def _validate_schema(raw: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        try:
            parsed = normalize_unique_items(json.loads(raw), schema)
            jsonschema.Draft202012Validator(schema).validate(parsed)
        except Exception as exc:
            raise UsageAssessmentError(f"invalid formal-use assessment: {exc}") from exc
        return parsed

    def _validate_deployment(self, raw: str) -> dict[str, Any]:
        return self._validate_schema(raw, self.deployment_schema)

    def _validate_content(
        self,
        raw: str,
        *,
        allowed_evidence_refs: tuple[str, ...],
    ) -> dict[str, Any]:
        parsed = self._validate_schema(raw, self.content_schema)
        refs = tuple(str(value) for value in parsed["trajectory_evidence_refs"])
        if not set(refs).issubset(set(allowed_evidence_refs)):
            raise UsageAssessmentError(
                "formal-use assessment cited unknown trajectory evidence"
            )
        problem_type = str(parsed["specific_problem_type"])
        problem = str(parsed["specific_problem"]).strip()
        if bool(problem_type) != bool(problem):
            raise UsageAssessmentError(
                "content problem type and description must be present together"
            )
        if problem_type and not refs:
            raise UsageAssessmentError("a content problem requires trajectory evidence")
        if not problem_type and refs:
            raise UsageAssessmentError(
                "trajectory evidence requires a specific content problem"
            )
        return parsed

    @staticmethod
    def _normalization_notes(
        raw: str,
        parsed: Mapping[str, Any],
    ) -> list[str]:
        try:
            original = json.loads(raw)
        except Exception:
            return []
        if not isinstance(original, Mapping):
            return []
        before = original.get("trajectory_evidence_refs")
        after = parsed.get("trajectory_evidence_refs")
        if isinstance(before, list) and isinstance(after, list) and before != after:
            return ["removed duplicate trajectory_evidence_refs while preserving order"]
        return []

    def _call_stage(
        self,
        *,
        stage: str,
        system_prompt: str,
        schema: Mapping[str, Any],
        base_prompt: str,
        validator: Callable[[str], dict[str, Any]],
        attempts: list[dict[str, Any]],
        compact_retry_prompt: str = "",
    ) -> tuple[dict[str, Any], str]:
        response_format = self._response_format(f"discernmem_{stage}", schema)
        validation_error = ""
        previous_length_limited = False
        raw = ""
        for format_attempt in range(2):
            current_prompt = base_prompt
            current_kind = stage
            if format_attempt:
                if previous_length_limited and compact_retry_prompt:
                    current_prompt = compact_retry_prompt
                    current_kind = f"{stage}_compact_retry"
                else:
                    current_kind = f"{stage}_format_retry"
                current_prompt += (
                    "\n\nThe previous output was rejected. Return one complete "
                    "corrected JSON object matching the schema. Validation error: "
                    + validation_error
                )
            if len(system_prompt) + len(current_prompt) > self.max_input_chars:
                raise UsageAssessmentError(
                    f"DiscernMem {stage} prompt exceeds the input budget",
                    attempts=attempts,
                )
            started = time.perf_counter()
            raw = self.llm.chat(
                system_prompt,
                current_prompt,
                temperature=0.0,
                response_format=response_format,
            )
            last_call = dict(self.llm.last_call or {})
            usage = dict(last_call.get("usage") or {})
            finish_reason = str(last_call.get("finish_reason") or "")
            previous_length_limited = finish_reason == "length" or (
                int(usage.get("completion_tokens", 0) or 0)
                >= int(getattr(self.llm, "max_tokens", 0) or 0)
                and len(raw.strip()) < 128
            )
            record = {
                "timestamp": utc_now(),
                "attempt": len(attempts) + 1,
                "stage": stage,
                "kind": current_kind,
                "user_prompt": current_prompt,
                "prompt_chars": len(system_prompt) + len(current_prompt),
                "raw_output": raw,
                "latency_seconds": time.perf_counter() - started,
                "actual_usage": usage,
                "finish_reason": finish_reason,
            }
            try:
                parsed = validator(raw)
            except UsageAssessmentError as exc:
                validation_error = str(exc)
                record["status"] = "invalid"
                record["validation_error"] = validation_error
                attempts.append(record)
                if format_attempt:
                    raise UsageAssessmentError(
                        f"{stage} remained invalid after format retry: "
                        + validation_error,
                        attempts=attempts,
                    ) from exc
                continue
            record["status"] = "valid"
            record["parsed_output"] = parsed
            record["normalizations"] = self._normalization_notes(raw, parsed)
            attempts.append(record)
            return parsed, raw
        raise AssertionError("formal-use format retry loop did not resolve")

    @staticmethod
    def _transition_signature(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(item.get("executed_action") or item.get("action") or "").lower(),
            bool(item.get("valid", True)),
            bool(item.get("state_changed", True)),
            tuple(str(value) for value in item.get("newly_satisfied", []) or []),
            tuple(str(value) for value in item.get("newly_lost", []) or []),
            bool(item.get("done", False)),
        )

    @classmethod
    def _first_divergence_turn(
        cls,
        baseline: Mapping[str, Any],
        version: Mapping[str, Any],
    ) -> int:
        baseline_items = list(baseline.get("transitions", []) or [])
        version_items = list(version.get("transitions", []) or [])
        for index in range(max(len(baseline_items), len(version_items))):
            if index >= len(baseline_items) or index >= len(version_items):
                item = (
                    baseline_items[index]
                    if index < len(baseline_items)
                    else version_items[index]
                )
                return int(item.get("turn", index + 1) or index + 1)
            if cls._transition_signature(
                baseline_items[index]
            ) != cls._transition_signature(version_items[index]):
                return int(
                    version_items[index].get(
                        "turn", baseline_items[index].get("turn", index + 1)
                    )
                    or index + 1
                )
        return 0

    @classmethod
    def _behavior_changed(
        cls,
        baseline: Mapping[str, Any],
        version: Mapping[str, Any],
        *,
        measured_delta: float,
        epsilon: float,
    ) -> bool:
        if abs(measured_delta) > epsilon:
            return True
        if baseline.get("final_score") != version.get("final_score"):
            return True
        return cls._first_divergence_turn(baseline, version) > 0

    @staticmethod
    def _focused_trajectory(
        value: Mapping[str, Any],
        *,
        highlight_turn: int = 0,
        limit: int = 16,
    ) -> dict[str, Any]:
        transitions = [dict(item) for item in value.get("transitions", []) or []]
        if len(transitions) <= limit:
            return dict(value)
        selected: set[int] = {0, 1, len(transitions) - 2, len(transitions) - 1}
        for index, item in enumerate(transitions):
            if (
                not bool(item.get("valid", True))
                or not bool(item.get("state_changed", True))
                or item.get("newly_satisfied")
                or item.get("newly_lost")
                or item.get("guard_replaced")
                or item.get("done")
                or (
                    highlight_turn
                    and abs(int(item.get("turn", 0) or 0) - highlight_turn) <= 1
                )
            ):
                selected.add(index)
        priority = sorted(
            selected,
            key=lambda index: (
                (
                    0
                    if highlight_turn
                    and abs(
                        int(transitions[index].get("turn", 0) or 0) - highlight_turn
                    )
                    <= 1
                    else 1
                ),
                index,
            ),
        )[:limit]
        focused = dict(value)
        focused["transitions"] = [transitions[index] for index in sorted(priority)]
        focused["focused_from_transition_count"] = len(transitions)
        return focused

    @staticmethod
    def _evidence_refs(*trajectories: Mapping[str, Any]) -> tuple[str, ...]:
        refs = []
        for trajectory in trajectories:
            refs.extend(
                str(item.get("event_id") or "")
                for item in trajectory.get("transitions", []) or []
                if str(item.get("event_id") or "")
            )
        return tuple(dict.fromkeys(refs))

    @staticmethod
    def _official_use_summary(
        trajectory: Mapping[str, Any],
        *,
        memory_id: str,
        version: int,
        role: str,
    ) -> dict[str, Any]:
        turns = []
        for step in trajectory.get("steps", []) or []:
            if str(step.get("agent") or "") != role:
                continue
            matched = any(
                str(item.get("memory_id") or "") == memory_id
                and int(item.get("version", -1)) == version
                for item in step.get("injected_memories", []) or []
            )
            if matched:
                turns.append(int(step.get("turn", step.get("step_index", 0)) or 0))
        return {
            "target_memory_injection_count": len(turns),
            "injection_turns": list(dict.fromkeys(turns))[:8],
            "interaction_rounds": int(trajectory.get("interaction_rounds", 0) or 0),
            "evaluator_feedback": str(trajectory.get("evaluator_feedback") or "")[:600],
        }

    def assess(
        self,
        *,
        task_id: str,
        memory: MemoryVersion,
        role: str,
        pre_action_context: Mapping[str, Any],
        official_trajectory: Mapping[str, Any],
        baseline: ReplayRecord,
        version_replay: ReplayRecord,
        isolated_delta: float | None = None,
        allowed_evidence_refs: tuple[str, ...],
        artifact_name: str,
    ) -> tuple[UsageAssessment, str]:
        if baseline.score is None or version_replay.score is None:
            raise UsageAssessmentError(
                "formal-use assessment requires successful scored replays"
            )
        measured_delta = (
            float(isolated_delta)
            if isolated_delta is not None
            else float(version_replay.score) - float(baseline.score)
        )
        epsilon = self.store.config.epsilon
        baseline_projection = project_assessment_trajectory(
            self._trajectory(baseline), evidence_namespace="baseline"
        )
        version_projection = project_assessment_trajectory(
            self._trajectory(version_replay), evidence_namespace="version"
        )
        behavior_changed = self._behavior_changed(
            baseline_projection,
            version_projection,
            measured_delta=measured_delta,
            epsilon=epsilon,
        )
        effect_class = (
            "beneficial"
            if measured_delta > epsilon
            else "harmful" if measured_delta < -epsilon else "neutral"
        )
        deployment_payload = {
            "memory": {
                "memory": memory.memory,
                "roles": list(memory.meta.roles),
                "use_when": list(memory.meta.use_when),
                "avoid_when": list(memory.meta.avoid_when),
            },
            "receiving_role": role,
            "pre_action_context": project_pre_action_context(pre_action_context),
        }
        official_summary = self._official_use_summary(
            official_trajectory,
            memory_id=memory.memory_id,
            version=memory.version,
            role=role,
        )
        audit_input: dict[str, Any] = {
            "controller_identity": {
                "task_id": task_id,
                "memory_id": memory.memory_id,
                "version": memory.version,
                "role": role,
            },
            "controller_measurements": {
                "baseline_score": baseline.score,
                "version_score": version_replay.score,
                "isolated_delta": measured_delta,
                "effect_class": effect_class,
                "behavior_changed": behavior_changed,
                "baseline_replay_id": baseline.replay_id,
                "version_replay_id": version_replay.replay_id,
            },
            "official_use_summary": official_summary,
            "deployment_input": deployment_payload,
            "content_input": None,
        }
        total_started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        audit_path = (
            self.store.root
            / "audit"
            / "formal_use_assessments"
            / f"{safe_component(artifact_name)}.json"
        )
        deployment_raw = ""
        try:
            deployment, deployment_raw = self._call_stage(
                stage="deployment_assessment",
                system_prompt=self.deployment_system_prompt,
                schema=self.deployment_schema,
                base_prompt=render_user_prompt(
                    "deployment_assessment.md", deployment_payload, indent=None
                ),
                validator=self._validate_deployment,
                attempts=attempts,
            )
        except UsageAssessmentError as exc:
            attempts = [dict(item) for item in exc.attempts]
            _atomic_write_json(
                audit_path,
                {
                    "timestamp": utc_now(),
                    "status": "failed",
                    "failed_stage": "deployment_assessment",
                    "input": audit_input,
                    "allowed_evidence_refs": [],
                    "system_prompts": {
                        "deployment": self.deployment_system_prompt,
                        "content": self.content_system_prompt,
                    },
                    "schemas": {
                        "deployment": self.deployment_schema,
                        "content": self.content_schema,
                        "combined_output": self.output_schema,
                    },
                    "attempts": attempts,
                    "raw_output": (
                        str(attempts[-1].get("raw_output") or "") if attempts else ""
                    ),
                    "error": str(exc),
                    "model": self.llm.model,
                    "latency_seconds": time.perf_counter() - total_started,
                    "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                    "finish_reason": str(
                        (self.llm.last_call or {}).get("finish_reason") or ""
                    ),
                },
            )
            raise

        content = {
            "specific_problem_type": "",
            "specific_problem": "",
            "trajectory_evidence_refs": [],
        }
        content_raw = ""
        content_status = "not_required"
        content_error = ""
        displayed_refs: tuple[str, ...] = ()
        deployment_correct = bool(deployment["use_when_matches"]) and not bool(
            deployment["avoid_when_matches"]
        )
        if effect_class == "harmful" and deployment_correct and behavior_changed:
            divergence_turn = self._first_divergence_turn(
                baseline_projection, version_projection
            )
            content_payload = {
                "memory": memory.memory,
                "receiving_role": role,
                "controller_measurements": {
                    "baseline_score": baseline.score,
                    "version_score": version_replay.score,
                    "isolated_delta": measured_delta,
                    "first_behavior_divergence_turn": divergence_turn,
                },
                "baseline_no_memory": baseline_projection,
                "isolated_memory_version": version_projection,
            }
            focused_payload = {
                **content_payload,
                "baseline_no_memory": self._focused_trajectory(
                    baseline_projection, highlight_turn=divergence_turn
                ),
                "isolated_memory_version": self._focused_trajectory(
                    version_projection, highlight_turn=divergence_turn
                ),
            }
            audit_input["content_input"] = content_payload
            projected_refs = self._evidence_refs(
                baseline_projection, version_projection
            )
            controller_refs = set(allowed_evidence_refs)
            displayed_refs = tuple(
                ref for ref in projected_refs if ref in controller_refs
            )
            try:
                content, content_raw = self._call_stage(
                    stage="harmful_content_diagnosis",
                    system_prompt=self.content_system_prompt,
                    schema=self.content_schema,
                    base_prompt=render_user_prompt(
                        "formal_use_content_diagnosis.md", content_payload, indent=None
                    ),
                    compact_retry_prompt=render_user_prompt(
                        "formal_use_content_diagnosis.md", focused_payload, indent=None
                    ),
                    validator=lambda raw: self._validate_content(
                        raw, allowed_evidence_refs=displayed_refs
                    ),
                    attempts=attempts,
                )
                content_status = "completed"
            except UsageAssessmentError as exc:
                attempts = [dict(item) for item in exc.attempts]
                content_status = "failed_preserved_as_ambiguous"
                content_error = str(exc)
                logger.warning(
                    "[DiscernMem][FORMAL-USE][task=%s][memory=%s@v%s][role=%s] "
                    "content-diagnosis-failed; preserving effect and deployment evidence",
                    task_id,
                    memory.memory_id,
                    memory.version,
                    role,
                )
                content = {
                    "specific_problem_type": "",
                    "specific_problem": "",
                    "trajectory_evidence_refs": [],
                }

        parsed = {
            "use_when_matches": bool(deployment["use_when_matches"]),
            "avoid_when_matches": bool(deployment["avoid_when_matches"]),
            "behavior_changed": behavior_changed,
            "specific_problem_type": str(content["specific_problem_type"]),
            "specific_problem": str(content["specific_problem"]).strip(),
            "trajectory_evidence_refs": list(content["trajectory_evidence_refs"]),
            "deployment_avoid_when": str(deployment["deployment_avoid_when"]).strip(),
        }
        jsonschema.Draft202012Validator(self.output_schema).validate(parsed)
        assessment = UsageAssessment(
            task_id=task_id,
            memory_id=memory.memory_id,
            version=memory.version,
            role=role,
            role_allowed=role in memory.meta.roles,
            use_when_matches=parsed["use_when_matches"],
            avoid_when_matches=parsed["avoid_when_matches"],
            behavior_changed=behavior_changed,
            isolated_delta=measured_delta,
            baseline_replay_id=baseline.replay_id,
            version_replay_id=version_replay.replay_id,
            specific_problem_type=parsed["specific_problem_type"],
            specific_problem=parsed["specific_problem"],
            trajectory_evidence_refs=tuple(
                str(value) for value in parsed["trajectory_evidence_refs"]
            ),
        )
        _atomic_write_json(
            audit_path,
            {
                "timestamp": utc_now(),
                "status": "completed",
                "content_status": content_status,
                "content_error": content_error,
                "input": audit_input,
                "allowed_evidence_refs": list(displayed_refs),
                "system_prompts": {
                    "deployment": self.deployment_system_prompt,
                    "content": self.content_system_prompt,
                },
                "schemas": {
                    "deployment": self.deployment_schema,
                    "content": self.content_schema,
                    "combined_output": self.output_schema,
                },
                "attempts": attempts,
                "stage_raw_outputs": {
                    "deployment": deployment_raw,
                    "content": content_raw,
                },
                "raw_output": content_raw or deployment_raw,
                "parsed_output": parsed,
                "controller_assessment": {
                    **assessment.__dict__,
                    "trajectory_evidence_refs": list(
                        assessment.trajectory_evidence_refs
                    ),
                },
                "model": self.llm.model,
                "latency_seconds": time.perf_counter() - total_started,
                "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                "finish_reason": str(
                    (self.llm.last_call or {}).get("finish_reason") or ""
                ),
            },
        )
        return assessment, parsed["deployment_avoid_when"]
