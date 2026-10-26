from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping

import jsonschema

from common.llm_client import LLMClient, estimate_tokens
from common.memory_guidance import mas_role_guidance, supported_mas_types, task_guidance

from .extraction import (
    ExtractionCandidate,
    candidate_from_response,
)
from .prompting import (
    load_prompt,
    load_schema,
    render_user_prompt,
)
from .schema_compat import guided_decoding_schema, normalize_unique_items
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now
from .trajectory_projection import (
    _execution_diagnostics,
    _representative_environment_trace,
    compact_text,
)


def _role_guidance(mas_type: str, role: str, duty: str) -> str:
    architecture = str(mas_type or "").casefold()
    if architecture not in supported_mas_types():
        raise ValueError(f"unsupported DiscernMem MAS guidance: {architecture!r}")
    return mas_role_guidance(architecture, role, duty)


def _task_family_guidance(
    task_family: str, *, enable_dataset_guidance: bool = True
) -> str:
    return task_guidance(task_family, task_specific=enable_dataset_guidance)


class DiscernMemPromptError(RuntimeError):
    pass


class DiscernMemPromptPipeline:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        api_base: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 180.0,
        retry_count: int = 2,
        max_input_chars: int = 120000,
    ):
        if not model_name:
            raise ValueError("DiscernMem prompt pipeline requires a real model name")
        self.store = store
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=0,
            max_input_chars=max_input_chars,
            logger_name="discernmem.llm",
            stats_source="discernmem/method_llm",
        )
        self.retry_count = max(int(retry_count), 0)
        self.max_input_chars = int(max_input_chars)
        self._call_lock = threading.Lock()
        self._call_index = 0
        self.usage = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_time_seconds": 0.0,
        }
        self.schema = load_schema("memory_extraction.schema.json")
        jsonschema.Draft202012Validator.check_schema(self.schema)

    def _next_call_index(self) -> int:
        with self._call_lock:
            self._call_index += 1
            return self._call_index

    def _system_prompt(
        self,
        mas_type: str,
        *,
        role_guidance: str = "",
        task_family_guidance: str = "",
    ) -> str:
        try:
            return load_prompt(
                "memory_extraction.md",
                mas_type=mas_type,
                include_mas_guidance=True,
                enable_mas_guidance=self.store.config.enable_mas_guidance,
                system_addenda=(
                    role_guidance if self.store.config.enable_mas_guidance else "",
                    task_family_guidance,
                ),
            )
        except (OSError, ValueError) as exc:
            raise DiscernMemPromptError(
                f"cannot load DiscernMem extraction prompt: {exc}"
            ) from exc

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("DiscernMem extraction response must be a JSON object")
        return payload


    @staticmethod
    def _kodcode_candidate_allowed(
        candidate: ExtractionCandidate, target_role: str, trajectory: Mapping[str, Any]
    ) -> bool:
        """Apply deterministic safety gates after prompt extraction."""
        role = str(target_role or "").casefold()
        messages = [
            item for item in trajectory.get("role_messages", []) or []
            if str(item.get("role") or "") == target_role
        ]
        if ("actor" in role or "implementation" in role) and len(messages) < 2:
            return False
        if "critic" in role and not trajectory.get("outgoing_context"):
            return False
        text = " ".join(
            [candidate.memory, *candidate.use_when, *candidate.avoid_when]
        ).casefold()
        forbidden = (
            r"```",
            r"\bdef\s+\w+",
            r"\bo\([^)]*\)",
            r"\b(?:time|space) complexity\b",
            r"\b(?:binomial|coefficient)\b",
            r"\b(?:dynamic programming|binary search|hash table)\b",
            r"\buse(?: a| an)? set\b",
            r"\bset for\b",
            r"\b(?:iterate|loop) through tokens\b",
            r"\bindex to recheck\b",
            r"\breplace .* tokens\b",
            r"\b(?:single pass|left to right|same index|replacing each)\b",
            r"\b(?:token|multiplication|addition)\b",
        )
        return not any(re.search(pattern, text) for pattern in forbidden)

    def _audit_path(
        self, artifact_name: str, call_index: int, suffix: str = ""
    ) -> Path:
        name = safe_component(f"{artifact_name}_memory_extraction_{call_index}{suffix}")
        return self.store.root / "audit" / "llm_calls" / f"{name}.json"

    @staticmethod
    def _grounding_evidence_refs(payload: Mapping[str, Any]) -> tuple[str, ...]:
        trajectory = dict(payload.get("trajectory") or {})
        return tuple(
            dict.fromkeys(
                str(value)
                for value in trajectory.get("grounding_evidence_refs", []) or []
                if str(value)
            )
        )

    @staticmethod
    def _model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        task = dict(payload.get("task") or {})
        mas = dict(payload.get("mas") or {})
        trajectory = dict(payload.get("trajectory") or {})

        def selected(
            source: Mapping[str, Any], keys: tuple[str, ...]
        ) -> dict[str, Any]:
            return {
                key: source[key]
                for key in keys
                if key in source and source[key] not in (None, "", [], {})
            }

        model_trajectory = selected(trajectory, ("initial_observable_state",))
        diagnostics = selected(
            dict(trajectory.get("execution_diagnostics") or {}),
            (
                "executed_action_count",
                "repeated_actions",
                "short_action_cycles",
                "repeated_post_action_states",
                "last_improvement_turn",
                "stagnant_turns",
            ),
        )
        if diagnostics:
            model_trajectory["execution_diagnostics"] = diagnostics
        environment_transitions = []
        seen_transitions: set[str] = set()
        for item in trajectory.get("environment_trace", []) or []:
            transition = selected(
                dict(item),
                (
                    "turn",
                    "step_index",
                    "role",
                    "upstream_roles",
                    "valid_actions",
                    "pre_action_state",
                    "submitted_action",
                    "executed_action",
                    "post_action_state",
                    "transition",
                    "score",
                    "step_reward",
                    "done",
                ),
            )
            signature = json.dumps(
                {key: value for key, value in transition.items() if key != "turn"},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if signature in seen_transitions:
                continue
            seen_transitions.add(signature)
            environment_transitions.append(transition)
        model_trajectory["environment_transitions"] = environment_transitions

        role_messages = []
        for item in trajectory.get("role_messages", []) or []:
            message = selected(
                dict(item),
                ("turn", "step_index", "role", "upstream_roles", "output"),
            )
            # Keep action-only messages as causal context. The environment
            # transition is complementary state/result evidence; dropping the
            # action here would hide an upstream proposal from its critic.
            role_messages.append(message)
        model_trajectory["role_messages"] = role_messages
        if not model_trajectory["environment_transitions"]:
            model_trajectory["event_summaries"] = [
                selected(
                    dict(item),
                    ("kind", "role", "observation", "action", "result", "feedback"),
                )
                for item in trajectory.get("events", []) or []
            ]
        model_trajectory = {
            key: value
            for key, value in model_trajectory.items()
            if value not in (None, "", [], {})
        }

        roles = [
            selected(
                dict(item),
                ("name", "role", "duty", "order"),
            )
            for item in mas.get("roles", []) or []
        ]
        model_mas = selected(mas, ("mas_type",))
        if roles:
            model_mas["roles"] = roles
        edges = list(mas.get("edges", []) or [])
        if edges:
            model_mas["edges"] = edges

        return {
            "task": selected(
                task,
                (
                    "description",
                    "task_family",
                    "outcome",
                    "score",
                    "evaluator_feedback",
                    "unsatisfied_goals",
                ),
            ),
            "trajectory": model_trajectory,
            "mas": model_mas,
        }


    @staticmethod
    def _causal_role_context(
        trajectory: Mapping[str, Any], mas: Mapping[str, Any], target_role: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return target messages plus topology-linked input/output context."""
        messages = [dict(item) for item in trajectory.get("role_messages", []) or []]
        target = [
            item for item in messages if str(item.get("role") or "") == target_role
        ]
        if not target:
            return [], [], []
        predecessors: set[str] = set()
        successors: set[str] = set()
        for edge in mas.get("edges", []) or []:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            source, destination = str(edge[0]), str(edge[1])
            if destination == target_role:
                predecessors.add(source)
            if source == target_role:
                successors.add(destination)
        context_in: dict[tuple[Any, ...], dict[str, Any]] = {}
        context_out: dict[tuple[Any, ...], dict[str, Any]] = {}

        def key(item: Mapping[str, Any]) -> tuple[int, int]:
            return (
                int(item.get("turn", 0) or 0),
                int(item.get("step_index", 0) or 0),
            )

        for current in target:
            current_key = key(current)
            declared = {
                str(value)
                for value in current.get("upstream_roles", []) or []
                if str(value)
            }
            for role in predecessors | declared:
                candidates = [
                    item for item in messages
                    if str(item.get("role") or "") == role and key(item) <= current_key
                ]
                if candidates:
                    item = max(candidates, key=key)
                    context_in[(role, *key(item), str(item.get("output") or ""))] = item
            for role in successors:
                candidates = [
                    item for item in messages
                    if str(item.get("role") or "") == role and key(item) >= current_key
                ]
                if candidates:
                    item = min(candidates, key=key)
                    context_out[(role, *key(item), str(item.get("output") or ""))] = item

        return (
            sorted(target, key=key),
            sorted(context_in.values(), key=key),
            sorted(context_out.values(), key=key),
        )

    @staticmethod
    def _direct_downstream_outcomes(
        transitions: list[dict[str, Any]],
        mas: Mapping[str, Any],
        target_role: str,
    ) -> list[dict[str, Any]]:
        """Return outcomes emitted by a direct successor from the target output."""
        successors = {
            str(edge[1])
            for edge in mas.get("edges", []) or []
            if isinstance(edge, (list, tuple))
            and len(edge) >= 2
            and str(edge[0]) == target_role
        }
        return [
            dict(item)
            for item in transitions
            if str(item.get("role") or "") in successors
            and target_role
            in {
                str(value)
                for value in item.get("upstream_roles", []) or []
                if str(value)
            }
        ]

    def _prepare_user_prompt(
        self, system_prompt: str, model_payload: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Render a complete payload, then compact only if the frozen budget is exceeded."""
        working = json.loads(json.dumps(model_payload, ensure_ascii=False, default=str))
        prompt = render_user_prompt("memory_extraction.md", working)
        info: dict[str, Any] = {"strategy": "complete", "omitted": {}}
        if len(system_prompt) + len(prompt) <= self.max_input_chars:
            return prompt, info

        trajectory = dict(working.get("trajectory") or {})
        # First remove exact duplicate context messages. Target messages are
        # never removed because they are the attribution evidence.
        target_outputs = {
            str(item.get("output") or "")
            for item in trajectory.get("role_messages", []) or []
        }
        for field in ("incoming_context", "outgoing_context"):
            values = list(trajectory.get(field) or [])
            kept = []
            seen: set[tuple[str, str]] = set()
            for item in values:
                value = dict(item)
                output = str(value.get("output") or "")
                key = (str(value.get("role") or ""), output)
                if not output or key in seen or output in target_outputs:
                    continue
                seen.add(key)
                kept.append(value)
            if len(kept) != len(values):
                info["omitted"][field] = len(values) - len(kept)
            trajectory[field] = kept
        working["trajectory"] = trajectory
        prompt = render_user_prompt("memory_extraction.md", working)
        if len(system_prompt) + len(prompt) <= self.max_input_chars:
            info["strategy"] = "deduplicate_context"
            return prompt, info

        # Preserve all target messages and transitions while shortening only
        # verbose context/state strings as a last textual compaction step.
        for field in ("incoming_context", "outgoing_context"):
            compacted = []
            for item in trajectory.get(field, []) or []:
                value = dict(item)
                if value.get("output"):
                    value["output"] = compact_text(value["output"], head=1200, tail=2400)
                compacted.append(value)
            trajectory[field] = compacted
        for transition_field in ("environment_transitions", "downstream_outcomes"):
            for item in trajectory.get(transition_field, []) or []:
                for field in ("pre_action_state", "post_action_state", "valid_actions"):
                    if isinstance(item.get(field), str):
                        item[field] = compact_text(item[field], head=1000, tail=3000)
        working["trajectory"] = trajectory
        prompt = render_user_prompt("memory_extraction.md", working)
        if len(system_prompt) + len(prompt) <= self.max_input_chars:
            info["strategy"] = "context_and_state_text_compaction"
            return prompt, info

        # Only when textual compaction cannot fit do we reduce transitions. The
        # representative projector preserves anchors, errors, changes, and ends.
        transitions = list(trajectory.get("environment_transitions", []) or [])
        if transitions:
            fixed = dict(working)
            fixed_trajectory = dict(trajectory)
            fixed_trajectory["environment_transitions"] = []
            fixed["trajectory"] = fixed_trajectory
            fixed_prompt = render_user_prompt("memory_extraction.md", fixed)
            remaining = max(self.max_input_chars - len(system_prompt) - len(fixed_prompt), 1)
            average = max(len(json.dumps(transitions[0], ensure_ascii=False)), 1)
            limit = max(1, min(len(transitions), remaining // average))
            selected = _representative_environment_trace(transitions, limit=limit)
            fixed_trajectory["environment_transitions"] = selected
            trajectory = fixed_trajectory
            working = fixed
            prompt = render_user_prompt("memory_extraction.md", working)
            info["strategy"] = "representative_transition_fallback"
            info["omitted"]["environment_transitions"] = len(transitions) - len(selected)
        if len(system_prompt) + len(prompt) > self.max_input_chars:
            # A very long direct context can dominate the budget even after
            # transition reduction. Compress context text progressively; target
            # messages remain intact for attribution.
            for head, tail in ((480, 960), (160, 320), (40, 80)):
                for field in ("incoming_context", "outgoing_context"):
                    compacted = []
                    for item in trajectory.get(field, []) or []:
                        value = dict(item)
                        if value.get("output"):
                            value["output"] = compact_text(
                                value["output"], head=head, tail=tail
                            )
                        compacted.append(value)
                    trajectory[field] = compacted
                working["trajectory"] = trajectory
                prompt = render_user_prompt("memory_extraction.md", working)
                info["strategy"] = "progressive_context_compaction"
                if len(system_prompt) + len(prompt) <= self.max_input_chars:
                    return prompt, info

            # Last resort for an unusually small budget: retain the complete
            # target-role evidence and omit only topology context. This is
            # explicitly audited rather than silently dropping information.
            for field in ("outgoing_context", "incoming_context"):
                values = trajectory.pop(field, []) or []
                if values:
                    info["omitted"][field] = len(values)
                working["trajectory"] = trajectory
                prompt = render_user_prompt("memory_extraction.md", working)
                if len(system_prompt) + len(prompt) <= self.max_input_chars:
                    info["strategy"] = "context_omission_last_resort"
                    return prompt, info
            # Final bounded projection for unusually long traces. Keep every
            # target-role message for attribution, but compact its verbose text
            # and retain only salient transition anchors. This prevents a hard
            # extraction failure while preserving the causal evidence needed by
            # the small extractor model.
            target_messages = []
            for item in trajectory.get("role_messages", []) or []:
                value = dict(item)
                if value.get("output"):
                    value["output"] = compact_text(value["output"], head=320, tail=640)
                target_messages.append(value)
            trajectory["role_messages"] = target_messages
            transitions = list(trajectory.get("environment_transitions", []) or [])
            if len(transitions) > 8:
                trajectory["environment_transitions"] = _representative_environment_trace(
                    transitions, limit=8
                )
            for item in trajectory.get("environment_transitions", []) or []:
                for field in ("pre_action_state", "post_action_state", "valid_actions"):
                    if isinstance(item.get(field), str):
                        item[field] = compact_text(item[field], head=320, tail=640)
            trajectory.pop("events", None)
            trajectory.pop("injected_memories", None)
            working["trajectory"] = trajectory
            prompt = render_user_prompt("memory_extraction.md", working)
            info["strategy"] = "bounded_target_and_transition_projection"
            if len(system_prompt) + len(prompt) <= self.max_input_chars:
                return prompt, info
            # Keep the target attribution but progressively shorten only its
            # textual payload until the frozen request budget is met.
            for head, tail in ((160, 320), (80, 160), (40, 80)):
                for item in trajectory.get("role_messages", []) or []:
                    if item.get("output"):
                        item["output"] = compact_text(item["output"], head=head, tail=tail)
                working["trajectory"] = trajectory
                prompt = render_user_prompt("memory_extraction.md", working)
                info["strategy"] = "bounded_target_text_compaction"
                if len(system_prompt) + len(prompt) <= self.max_input_chars:
                    return prompt, info
            raise DiscernMemPromptError(
                "DiscernMem extraction prompt exceeds the frozen input budget after bounded projection"
            )
        return prompt, info

    def extract(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_name: str,
        validation_feedback: tuple[str, ...] = (),
        source_entities: tuple[str, ...] = (),
        target_role: str,
    ) -> tuple[dict[str, Any], ExtractionCandidate | None]:
        mas = dict(payload.get("mas") or {})
        output_schema = self.schema
        model_payload = self._model_payload(payload)
        role_spec = next(
            (
                dict(item)
                for item in mas.get("roles", []) or []
                if str(item.get("name") or item.get("role") or "") == target_role
            ),
            {},
        )
        task_family = str(
            (payload.get("task") or {}).get("task_family")
            or (payload.get("trajectory") or {}).get("task_family")
            or ""
        ).casefold()
        model_payload["target_role"] = {
            "name": target_role,
            "duty": str(role_spec.get("duty") or ""),
        }
        role_guidance = _role_guidance(
            str(mas.get("mas_type") or ""),
            target_role,
            str(role_spec.get("duty") or ""),
        )
        task_family_guidance = _task_family_guidance(
            task_family,
            enable_dataset_guidance=(
                self.store.config.enable_dataset_task_guidance
            ),
        )
        system_prompt = self._system_prompt(
            str(mas.get("mas_type") or ""),
            role_guidance=role_guidance,
            task_family_guidance=task_family_guidance,
        )
        role_trajectory = dict(model_payload.get("trajectory") or {})
        role_messages, incoming_context, outgoing_context = self._causal_role_context(
            role_trajectory, mas, target_role
        )
        if role_messages:
            role_trajectory["role_messages"] = role_messages
        else:
            role_trajectory.pop("role_messages", None)
        if incoming_context:
            role_trajectory["incoming_context"] = incoming_context
        else:
            role_trajectory.pop("incoming_context", None)
        if outgoing_context:
            role_trajectory["outgoing_context"] = outgoing_context
        else:
            role_trajectory.pop("outgoing_context", None)

        all_transitions = list(
            role_trajectory.get("environment_transitions", []) or []
        )
        downstream_outcomes = self._direct_downstream_outcomes(
            all_transitions, mas, target_role
        )
        if downstream_outcomes:
            role_trajectory["downstream_outcomes"] = downstream_outcomes
        else:
            role_trajectory.pop("downstream_outcomes", None)
        transitions = all_transitions
        if all_transitions:
            if any(str(item.get("role") or "") for item in all_transitions):
                transitions = [
                    item
                    for item in all_transitions
                    if str(item.get("role") or "") == target_role
                ]
            else:
                role_step_indices = {
                    int(item.get("step_index", 0) or 0)
                    for item in role_messages
                    if int(item.get("step_index", 0) or 0) > 0
                }
                if role_step_indices and any(
                    int(item.get("step_index", 0) or 0) > 0 for item in transitions
                ):
                    transitions = [
                        item
                        for item in transitions
                        if int(item.get("step_index", 0) or 0) in role_step_indices
                    ]
                else:
                    transitions = []
            if transitions:
                role_trajectory["environment_transitions"] = transitions
            else:
                role_trajectory.pop("environment_transitions", None)
        # Recompute diagnostics only from transitions attributable to this role.
        # For a proposal-only role, direct successor outcomes are its causal evidence.
        role_trajectory.pop("execution_diagnostics", None)
        diagnostic_transitions = transitions or downstream_outcomes
        if diagnostic_transitions:
            diagnostics = _execution_diagnostics(diagnostic_transitions)
            has_failure_pattern = bool(
                diagnostics.get("short_action_cycles")
                or diagnostics.get("repeated_post_action_states")
                or int(diagnostics.get("stagnant_turns", 0) or 0) >= 3
            )
            if has_failure_pattern:
                role_trajectory["role_local_diagnostics"] = diagnostics
        model_payload["trajectory"] = role_trajectory
        if source_entities:
            model_payload["source_entities"] = list(source_entities)
        if validation_feedback:
            roles = [
                str(item.get("name") or item.get("role") or "")
                for item in mas.get("roles", [])
                if item.get("name") or item.get("role")
            ]
            model_payload["controller_feedback"] = {
                "validation_errors": list(validation_feedback),
                "allowed_mas_roles": roles,
                "instruction": (
                    "Correct only errors supported by observed evidence. Preserve "
                    "grounded causal order and topology, generalize source entities, "
                    "and do not add operations or effects. Always return one corrected "
                    "candidate using the strongest supported observed transition."
                ),
            }
        user_prompt, compaction_info = self._prepare_user_prompt(
            system_prompt, model_payload
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "discernmem_memory_extraction",
                "schema": guided_decoding_schema(output_schema),
                "strict": True,
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            call_index = self._next_call_index()
            started = time.perf_counter()
            raw = ""
            attempt_user_prompt = user_prompt
            try:
                attempt_payload = dict(model_payload)
                if attempt:
                    feedback = dict(attempt_payload.get("controller_feedback") or {})
                    feedback["format_error"] = (
                        "The previous output was invalid. Return one complete JSON "
                        "object matching the response schema."
                    )
                    attempt_payload["controller_feedback"] = feedback
                attempt_user_prompt = (
                    user_prompt
                    if not attempt
                    else render_user_prompt("memory_extraction.md", attempt_payload)
                )
                raw = self.llm.chat(
                    system_prompt,
                    attempt_user_prompt,
                    temperature=0.0,
                    response_format=response_format,
                )
                parsed = normalize_unique_items(self._parse_json(raw), self.schema)
                jsonschema.Draft202012Validator(self.schema).validate(parsed)
                candidate = candidate_from_response(parsed, target_role=target_role)
                if (
                    candidate is not None
                    and task_family == "kodcode"
                    and not self._kodcode_candidate_allowed(
                        candidate, target_role, role_trajectory
                    )
                ):
                    parsed = {
                        "decision": "NO_MEMORY",
                        "memory": "NO_MEMORY",
                        "use_when": [],
                        "avoid_when": [],
                    }
                    candidate = None
                elapsed = time.perf_counter() - started
                last_call = dict(self.llm.last_call or {})
                usage = dict(last_call.get("usage") or {})
                prompt_tokens = int(
                    usage.get("prompt_tokens")
                    or estimate_tokens(system_prompt + attempt_user_prompt)
                )
                completion_tokens = int(
                    usage.get("completion_tokens") or estimate_tokens(raw)
                )
                self.usage["llm_calls"] += 1
                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
                self.usage["llm_time_seconds"] += elapsed
                _atomic_write_json(
                    self._audit_path(artifact_name, call_index),
                    {
                        "timestamp": utc_now(),
                        "stage": "memory_extraction",
                        "attempt": attempt,
                        "model": self.llm.model,
                        "api_base": self.llm.base_url,
                        "temperature": 0.0,
                        "max_tokens": self.llm.max_tokens,
                        "system_prompt": system_prompt,
                        "user_prompt": attempt_user_prompt,
                        "input": model_payload,
                        "output_schema": output_schema,
                        "raw_output": raw,
                        "parsed_output": parsed,
                        "schema_valid": True,
                        "request_payload": last_call.get("request_payload"),
                        "actual_usage": usage,
                        "latency_seconds": elapsed,
                        "prompt_compaction": compaction_info,
                    },
                )
                return parsed, candidate
            except Exception as exc:
                elapsed = time.perf_counter() - started
                last_error = exc
                self.usage["llm_calls"] += 1
                self.usage["prompt_tokens"] += estimate_tokens(
                    system_prompt + attempt_user_prompt
                )
                self.usage["completion_tokens"] += estimate_tokens(raw)
                self.usage["llm_time_seconds"] += elapsed
                _atomic_write_json(
                    self._audit_path(artifact_name, call_index, "_error"),
                    {
                        "timestamp": utc_now(),
                        "stage": "memory_extraction",
                        "attempt": attempt,
                        "model": self.llm.model,
                        "api_base": self.llm.base_url,
                        "system_prompt": system_prompt,
                        "user_prompt": attempt_user_prompt,
                        "input": model_payload,
                        "raw_output": raw,
                        "schema_valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_seconds": elapsed,
                    },
                )
        raise DiscernMemPromptError(
            f"memory extraction failed after {self.retry_count + 1} attempts: {last_error}"
        )

    def extract_and_validate(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_name: str,
        source_entities: tuple[str, ...] = (),
        target_role: str,
    ) -> tuple[dict[str, Any], ExtractionCandidate | None, tuple[str, ...]]:
        parsed, candidate = self.extract(
            payload,
            artifact_name=artifact_name,
            source_entities=source_entities,
            target_role=target_role,
        )
        # Provenance is attached from the normalized trajectory; it is not a
        # candidate-quality gate.
        grounding_refs = self._grounding_evidence_refs(payload)
        if candidate is not None:
            candidate = replace(candidate, evidence_refs=grounding_refs)
        return dict(parsed), candidate, ()
