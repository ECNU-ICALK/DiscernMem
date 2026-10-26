from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping


def normalize_trajectory_event_ids(
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with stable fallback IDs for every event and role step."""
    normalized = dict(trajectory)
    counter = 0
    used_ids = {
        str(item.get("event_id") or "")
        for collection_name in ("events", "steps")
        for item in trajectory.get(collection_name, []) or []
        if str(item.get("event_id") or "")
    }
    for collection_name in ("events", "steps"):
        items = []
        for item in trajectory.get(collection_name, []) or []:
            counter += 1
            value = dict(item)
            event_id = str(value.get("event_id") or "")
            if not event_id:
                event_id = f"event-{counter}"
                while event_id in used_ids:
                    counter += 1
                    event_id = f"event-{counter}"
                used_ids.add(event_id)
            value["event_id"] = event_id
            items.append(value)
        if items or collection_name in trajectory:
            normalized[collection_name] = items
    return normalized


def compact_text(value: object, *, head: int, tail: int) -> str:
    text = str(value or "")
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n...[middle omitted]...\n{text[-tail:]}"


_INITIAL_OBSERVATION_RE = re.compile(
    r"Here is your initial observation:\s*(.*?)\s*\*\*Here is your task:",
    flags=re.IGNORECASE | re.DOTALL,
)
_CURRENT_OBSERVATION_RE = re.compile(
    r"Current observation:\s*(.*?)(?:\n\s*What is your next action\?|\n\s*<goal_progress>)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CURRENT_STATE_TAG_RE = re.compile(
    r"<(current_observation|current_state)>\s*(.*?)\s*</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_VALID_ACTION_BLOCK_RE = re.compile(
    r"^[ \t]*<valid_action_constraint>[ \t]*\r?\n"
    r"(.*?)^[ \t]*</valid_action_constraint>[ \t]*$",
    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_FEEDBACK_ACTION_RE = re.compile(r"actions=\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_FEEDBACK_OBSERVATION_RE = re.compile(
    r"\bobservation=(.*?)(?:\s+guarded_action:|$)",
    re.IGNORECASE | re.DOTALL,
)
_FEEDBACK_SCORE_RE = re.compile(
    r"\bscore=([-+]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE
)


def extract_initial_observable_state(value: object) -> str:
    text = str(value or "")
    match = _INITIAL_OBSERVATION_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    tagged_states = _CURRENT_STATE_TAG_RE.findall(text)
    if tagged_states:
        return " ".join(tagged_states[0][1].split())
    match = _CURRENT_OBSERVATION_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    return ""


def _extract_pre_action_state(value: object) -> str:
    text = str(value or "")
    matches = _CURRENT_STATE_TAG_RE.findall(text)
    if matches:
        return " ".join(matches[-1][1].split())
    match = _CURRENT_OBSERVATION_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    return extract_initial_observable_state(text)


def _extract_valid_actions(value: object) -> list[str]:
    blocks = _VALID_ACTION_BLOCK_RE.findall(str(value or ""))
    if not blocks:
        return []
    actions = []
    for line in blocks[-1].splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            actions.append(stripped[2:].strip())
    return actions


def _extract_submitted_action(value: object) -> str:
    text = str(value or "")
    tagged = _ACTION_TAG_RE.search(text)
    return " ".join((tagged.group(1) if tagged else text).split())


def _extract_feedback_action(value: object) -> str:
    text = str(value or "")
    match = _FEEDBACK_ACTION_RE.search(text)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _extract_feedback_observation(value: object) -> str:
    text = str(value or "")
    match = _FEEDBACK_OBSERVATION_RE.search(text)
    return " ".join(match.group(1).split()) if match else text


def _extract_feedback_score(value: object) -> float | None:
    match = _FEEDBACK_SCORE_RE.search(str(value or ""))
    return float(match.group(1)) if match else None


def _canonical_action(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _canonical_state(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" .")


def _short_action_cycles(actions: list[str]) -> list[dict[str, Any]]:
    cycles = []
    for width in (2, 3):
        counts: Counter[tuple[str, ...]] = Counter()
        starts: dict[tuple[str, ...], list[int]] = {}
        for index in range(0, len(actions) - (2 * width) + 1):
            first = tuple(actions[index : index + width])
            second = tuple(actions[index + width : index + (2 * width)])
            if first != second or not all(first):
                continue
            counts[first] += 1
            starts.setdefault(first, []).append(index + 1)
        for sequence, repeat_windows in counts.items():
            cycles.append(
                {
                    "length": width,
                    "actions": list(sequence),
                    "repeat_windows": repeat_windows,
                    "first_turn": starts[sequence][0],
                }
            )
    cycles.sort(
        key=lambda item: (
            -item["repeat_windows"],
            item["length"],
            item["first_turn"],
        )
    )
    return cycles[:8]


def _execution_diagnostics(environment_trace: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [
        _canonical_action(item.get("executed_action") or item.get("submitted_action"))
        for item in environment_trace
    ]
    action_counts = Counter(action for action in actions if action)
    repeated_actions = [
        {"action": action, "count": count}
        for action, count in action_counts.most_common()
        if count > 1
    ][:12]

    state_turns: dict[str, list[int]] = {}
    for item in environment_trace:
        state = _canonical_state(
            item.get("post_action_state") or item.get("observation")
        )
        if state:
            state_turns.setdefault(state, []).append(int(item.get("turn", 0) or 0))
    repeated_states = [
        {
            "state": compact_text(state, head=160, tail=240),
            "count": len(turns),
            "turns": turns[:12],
        }
        for state, turns in state_turns.items()
        if len(turns) > 1
    ]
    repeated_states.sort(key=lambda item: (-item["count"], item["turns"][0]))

    best_score = float("-inf")
    last_improvement_turn = 0
    for item in environment_trace:
        score = item.get("score")
        if score is None:
            continue
        score = float(score)
        if score > best_score + 1e-9:
            best_score = score
            last_improvement_turn = int(item.get("turn", 0) or 0)
    last_turn = max(
        (int(item.get("turn", 0) or 0) for item in environment_trace),
        default=0,
    )

    ignored: dict[str, dict[str, Any]] = {}
    for item in environment_trace:
        executed = _canonical_action(
            item.get("executed_action") or item.get("submitted_action")
        )
        for action in item.get("valid_actions", []) or []:
            canonical = _canonical_action(action)
            if not canonical or canonical == executed:
                continue
            record = ignored.setdefault(
                canonical,
                {
                    "action": str(action),
                    "available_count": 0,
                    "turns": [],
                    "pre_action_states": [],
                },
            )
            record["available_count"] += 1
            if len(record["turns"]) < 12:
                record["turns"].append(int(item.get("turn", 0) or 0))
            state = compact_text(item.get("pre_action_state"), head=140, tail=260)
            if state and state not in record["pre_action_states"]:
                record["pre_action_states"].append(state)
                record["pre_action_states"] = record["pre_action_states"][:3]
    ignored_alternatives = sorted(
        ignored.values(),
        key=lambda item: (-item["available_count"], item["action"].lower()),
    )[:12]

    return {
        "executed_action_count": len(environment_trace),
        "repeated_actions": repeated_actions,
        "short_action_cycles": _short_action_cycles(actions),
        "repeated_post_action_states": repeated_states[:8],
        "ignored_valid_alternatives": ignored_alternatives,
        "last_improvement_turn": last_improvement_turn,
        "stagnant_turns": max(last_turn - last_improvement_turn, 0),
    }


def _representative_environment_trace(
    environment_trace: list[dict[str, Any]], *, limit: int = 16
) -> list[dict[str, Any]]:
    if len(environment_trace) <= limit:
        return environment_trace

    selected: set[int] = {
        0,
        1,
        len(environment_trace) - 2,
        len(environment_trace) - 1,
    }
    seen_actions = set()
    seen_states = set()
    for index, item in enumerate(environment_trace):
        action = _canonical_action(
            item.get("executed_action") or item.get("submitted_action")
        )
        state = _canonical_state(item.get("post_action_state"))
        transition = dict(item.get("transition") or {})
        important = (
            not bool(transition.get("valid", True))
            or not bool(transition.get("state_changed", True))
            or bool(transition.get("newly_satisfied"))
            or bool(transition.get("newly_lost"))
            or float(transition.get("reward", 0.0) or 0.0) != 0.0
            or bool(item.get("done", False))
        )
        if important or action not in seen_actions or state not in seen_states:
            selected.add(index)
        if action:
            seen_actions.add(action)
        if state:
            seen_states.add(state)

    if len(selected) < limit:
        remaining = [
            index for index in range(len(environment_trace)) if index not in selected
        ]
        slots = limit - len(selected)
        for offset in range(slots):
            position = round(offset * (len(remaining) - 1) / max(slots - 1, 1))
            selected.add(remaining[position])
    ordered = sorted(selected)
    if len(ordered) > limit:
        anchors = {0, 1, len(environment_trace) - 2, len(environment_trace) - 1}
        priority = [index for index in ordered if index in anchors]
        priority.extend(index for index in ordered if index not in anchors)
        ordered = sorted(priority[:limit])
    return [environment_trace[index] for index in ordered]


def _grounding_evidence_refs(
    environment_trace: list[dict[str, Any]], *, limit: int = 8
) -> tuple[str, ...]:
    """Select compact causal transition IDs for machine-side provenance."""
    if not environment_trace:
        return ()
    selected: list[int] = []
    important = []
    for index, item in enumerate(environment_trace):
        transition = dict(item.get("transition") or {})
        if (
            not bool(transition.get("valid", True))
            or not bool(transition.get("state_changed", True))
            or bool(transition.get("newly_satisfied"))
            or bool(transition.get("newly_lost"))
            or float(transition.get("reward", 0.0) or 0.0) != 0.0
            or bool(item.get("done", False))
        ):
            important.append(index)
    selected.extend(important)
    selected.extend(range(min(4, len(environment_trace))))
    if len(environment_trace) > 4:
        selected.extend(
            range(max(0, len(environment_trace) - 2), len(environment_trace))
        )
    refs = []
    for index in selected:
        if len(refs) >= limit:
            break
        event_id = str(environment_trace[index].get("event_id") or "")
        if event_id and event_id not in refs:
            refs.append(event_id)
    return tuple(refs)


def _injected_memories(
    trajectory: Mapping[str, Any], steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    injected = []
    seen = set()
    sources = [
        (item, str(retrieval.get("agent_role") or retrieval.get("role") or ""))
        for retrieval in trajectory.get("memory_retrievals", []) or []
        for item in (
            retrieval.get("injected_memories") or retrieval.get("selected") or []
        )
    ] + [
        (item, str(step.get("agent") or ""))
        for step in steps
        for item in step.get("injected_memories", []) or []
    ]
    for item, fallback_role in sources:
        record = dict(item)
        memory_id = str(record.get("memory_id") or "")
        candidate_id = str(record.get("candidate_id") or "")
        version = record.get("version")
        role = str(record.get("injected_role") or fallback_role)
        key = (memory_id, candidate_id, version, role)
        if (not memory_id and not candidate_id) or key in seen:
            continue
        seen.add(key)
        injected.append(
            {
                "memory_id": memory_id or None,
                "candidate_id": candidate_id or None,
                "version": version,
                "injected_role": role,
            }
        )
    return injected


def project_extraction_trajectory(
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory = normalize_trajectory_event_ids(trajectory)
    steps = [dict(item) for item in trajectory.get("steps", []) or []]
    events = []
    for item in trajectory.get("events", []) or []:
        event = dict(item)
        events.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "kind": str(event.get("kind") or ""),
                "observation": compact_text(
                    event.get("observation"), head=120, tail=480
                ),
                "action": compact_text(event.get("action"), head=80, tail=320),
                "feedback": compact_text(event.get("feedback"), head=120, tail=480),
            }
        )

    all_role_messages = []
    full_environment_trace = []
    for step in steps:
        event_id = str(step.get("event_id") or "")
        all_role_messages.append(
            {
                "event_id": event_id,
                "turn": int(step.get("turn", step.get("step_index", 0)) or 0),
                "step_index": int(step.get("step_index", 0) or 0),
                "role": str(step.get("agent") or "unknown"),
                "upstream_roles": [
                    str(value) for value in step.get("upstream_roles", []) or []
                ],
                # Preserve the complete role output; prompt boilerplate is not
                # copied into this projection, so truncating code or a plan here
                # would remove causal evidence before token budgeting runs.
                "output": " ".join(str(step.get("output") or "").split()),
            }
        )
        if step.get("feedback"):
            feedback = str(step.get("feedback") or "")
            transition = dict(step.get("transition") or {})
            post_action_state = str(
                transition.get("environment_observation")
                or _extract_feedback_observation(feedback)
            )
            full_environment_trace.append(
                {
                    "event_id": event_id,
                    "turn": int(step.get("turn", step.get("step_index", 0)) or 0),
                    "step_index": int(step.get("step_index", 0) or 0),
                    "role": str(step.get("agent") or "unknown"),
                    "upstream_roles": [
                        str(value) for value in step.get("upstream_roles", []) or []
                    ],
                    "valid_actions": _extract_valid_actions(step.get("input")),
                    "pre_action_state": " ".join(
                        str(_extract_pre_action_state(step.get("input")) or "").split()
                    ),
                    "submitted_action": _extract_submitted_action(step.get("output")),
                    "executed_action": str(
                        transition.get("action") or _extract_feedback_action(feedback)
                    ),
                    "post_action_state": " ".join(
                        str(post_action_state or "").split()
                    ),
                    "transition": {
                        "valid": bool(transition.get("valid", True)),
                        "state_changed": bool(transition.get("state_changed", True)),
                        "reward": float(
                            transition.get("reward", step.get("step_reward", 0.0))
                            or 0.0
                        ),
                        "newly_satisfied": list(
                            transition.get("newly_satisfied", []) or []
                        ),
                        "newly_lost": list(transition.get("newly_lost", []) or []),
                        "invalid_feedback": compact_text(
                            transition.get("invalid_feedback"), head=0, tail=400
                        ),
                    },
                    "score": _extract_feedback_score(feedback),
                    "step_reward": float(step.get("step_reward", 0.0) or 0.0),
                    "done": bool(step.get("done", False)),
                }
            )

    # Keep every projected event. Redundancy is removed at field level; event
    # count is reduced only by the token-budget fallback in the prompt pipeline.
    environment_trace = list(full_environment_trace)
    role_messages = list(all_role_messages)
    first_input = next(
        (step.get("input") for step in steps if step.get("input")),
        "",
    )
    grounding_evidence_refs = _grounding_evidence_refs(environment_trace)

    return {
        "initial_observable_state": extract_initial_observable_state(first_input),
        "events": events,
        "role_messages": role_messages,
        "environment_trace": environment_trace,
        "grounding_evidence_refs": list(grounding_evidence_refs),
        "execution_diagnostics": _execution_diagnostics(full_environment_trace),
        "injected_memories": _injected_memories(trajectory, steps),
        "interaction_rounds": int(trajectory.get("interaction_rounds", 0) or 0),
        "role_steps": int(trajectory.get("role_steps", len(steps)) or len(steps)),
        "compaction": {
            "strategy": "loss-minimizing_full_event_projection_v4",
            "original_step_count": len(steps),
            "original_environment_event_count": len(full_environment_trace),
            "projected_role_message_count": len(role_messages),
            "projected_environment_event_count": len(environment_trace),
        },
    }


def project_assessment_trajectory(
    trajectory: Mapping[str, Any],
    *,
    evidence_namespace: str = "",
) -> dict[str, Any]:
    trajectory = normalize_trajectory_event_ids(trajectory)

    def evidence_id(value: object) -> str:
        event_id = str(value or "")
        if not evidence_namespace or not event_id:
            return event_id
        return f"{evidence_namespace}:{event_id}"

    def state_facts(value: object) -> list[str]:
        text = " ".join(str(value or "").split())
        if not text:
            return []
        facts = []
        for item in re.split(r"(?<=[.!?])\s+|\s*;\s*", text):
            fact = item.strip()
            if fact and fact not in facts:
                facts.append(fact)
        return facts

    def state_delta(before: object, after: object) -> tuple[list[str], list[str]]:
        before_facts = state_facts(before)
        after_facts = state_facts(after)
        before_keys = {_canonical_state(item): item for item in before_facts}
        after_keys = {_canonical_state(item): item for item in after_facts}
        added = [
            item for item in after_facts if _canonical_state(item) not in before_keys
        ]
        removed = [
            item for item in before_facts if _canonical_state(item) not in after_keys
        ]
        return added[:16], removed[:16]

    steps = [dict(item) for item in trajectory.get("steps", []) or []]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        turn = int(step.get("turn", step.get("step_index", 0)) or 0)
        grouped.setdefault(turn, []).append(step)

    transitions = []
    initial_state = ""
    final_state = ""
    previous_score: float | None = None
    for turn, turn_steps in grouped.items():
        execution = next(
            (
                step
                for step in reversed(turn_steps)
                if step.get("transition") is not None
                or step.get("feedback")
                or step.get("action_guard") is not None
            ),
            turn_steps[-1],
        )
        proposal = next(
            (
                step
                for step in turn_steps
                if step is not execution and str(step.get("output") or "").strip()
            ),
            None,
        )
        transition = dict(execution.get("transition") or {})
        action_guard = dict(execution.get("action_guard") or {})
        feedback = str(execution.get("feedback") or "")
        proposed_action = _extract_submitted_action(
            (proposal or execution).get("output")
        )
        executed_action = str(
            transition.get("action")
            or action_guard.get("executed_action")
            or _extract_feedback_action(feedback)
            or proposed_action
        ).strip()
        pre_action_state = _extract_pre_action_state(
            (proposal or execution).get("input")
        )
        post_action_state = str(
            transition.get("environment_observation")
            or _extract_feedback_observation(feedback)
        ).strip()
        if not initial_state:
            initial_state = pre_action_state
        if post_action_state:
            final_state = post_action_state
        added, removed = state_delta(pre_action_state, post_action_state)
        valid = bool(transition.get("valid", True))
        changed = bool(transition.get("state_changed", bool(added or removed)))
        reward = float(
            transition.get("reward", execution.get("step_reward", 0.0)) or 0.0
        )
        score = _extract_feedback_score(feedback)
        guard_replaced = bool(action_guard.get("was_replaced", False))
        if not action_guard:
            guard_replaced = (
                bool(proposed_action)
                and bool(executed_action)
                and _canonical_action(proposed_action)
                != _canonical_action(executed_action)
            )
        item: dict[str, Any] = {
            "event_id": evidence_id(execution.get("event_id")),
            "turn": turn,
        }
        if guard_replaced:
            item["proposed_action"] = compact_text(proposed_action, head=80, tail=220)
            item["executed_action"] = compact_text(executed_action, head=80, tail=220)
            item["guard_replaced"] = True
        else:
            item["action"] = compact_text(
                executed_action or proposed_action, head=80, tail=220
            )
        item["valid"] = valid
        item["state_changed"] = changed
        if reward:
            item["reward"] = reward
        newly_satisfied = list(transition.get("newly_satisfied", []) or [])
        newly_lost = list(transition.get("newly_lost", []) or [])
        if newly_satisfied:
            item["newly_satisfied"] = newly_satisfied
        if newly_lost:
            item["newly_lost"] = newly_lost
        if added:
            item["state_added"] = added
        if removed:
            item["state_removed"] = removed
        invalid_feedback = compact_text(
            transition.get("invalid_feedback"), head=0, tail=400
        )
        if invalid_feedback:
            item["invalid_feedback"] = invalid_feedback
        if action_guard.get("guard_note"):
            item["guard_note"] = compact_text(
                action_guard.get("guard_note"), head=0, tail=240
            )
        if score is not None and (
            previous_score is None or abs(score - previous_score) > 1e-9
        ):
            item["score"] = score
        if score is not None:
            previous_score = score
        if execution.get("done"):
            item["done"] = True
        transitions.append(item)

    actions = [
        _canonical_action(item.get("executed_action") or item.get("action"))
        for item in transitions
    ]
    action_counts = Counter(action for action in actions if action)
    repeated_actions = [
        {"action": action, "count": count}
        for action, count in action_counts.most_common()
        if count > 1
    ][:8]
    progress_turns = [
        int(item["turn"])
        for item in transitions
        if item.get("newly_satisfied")
        or float(item.get("reward", 0.0) or 0.0) > 0.0
        or item.get("done")
    ]
    last_turn = max((int(item["turn"]) for item in transitions), default=0)
    last_progress_turn = max(progress_turns, default=0)
    diagnostics: dict[str, Any] = {
        "transition_count": len(transitions),
        "last_progress_turn": last_progress_turn,
        "stagnant_turns": max(last_turn - last_progress_turn, 0),
    }
    if repeated_actions:
        diagnostics["repeated_actions"] = repeated_actions
    cycles = _short_action_cycles(actions)
    if cycles:
        diagnostics["short_action_cycles"] = cycles
    invalid_turns = [item["turn"] for item in transitions if not item["valid"]]
    if invalid_turns:
        diagnostics["invalid_turns"] = invalid_turns
    replaced_turns = [
        item["turn"] for item in transitions if item.get("guard_replaced")
    ]
    if replaced_turns:
        diagnostics["guard_replaced_turns"] = replaced_turns

    result: dict[str, Any] = {
        "interaction_rounds": int(trajectory.get("interaction_rounds", 0) or 0),
        "initial_state": compact_text(initial_state, head=240, tail=720),
        "final_state": compact_text(final_state, head=240, tail=720),
        "transitions": transitions,
        "diagnostics": diagnostics,
    }
    evaluator_feedback = str(trajectory.get("evaluator_feedback") or "")
    final_score = _extract_feedback_score(evaluator_feedback)
    if final_score is not None:
        result["final_score"] = final_score
    if trajectory.get("outcome"):
        result["outcome"] = str(trajectory.get("outcome"))
    return result


def project_pre_action_context(context: Mapping[str, Any]) -> dict[str, Any]:
    contexts = []
    seen = set()
    for item in context.get("retrieval_contexts", []) or []:
        task_description = str(item.get("task_description") or "")
        observable_state = str(item.get("observable_state") or "")
        if not observable_state:
            observable_state = _extract_pre_action_state(task_description)
        value = {
            "task_description": compact_text(task_description, head=120, tail=420),
            "observable_state": compact_text(observable_state, head=180, tail=620),
            "decision_point": compact_text(
                item.get("decision_point"), head=80, tail=180
            ),
            "role": str(item.get("role") or ""),
        }
        key = tuple(value.items())
        if key in seen:
            continue
        seen.add(key)
        contexts.append(value)
    return {"retrieval_contexts": contexts}
