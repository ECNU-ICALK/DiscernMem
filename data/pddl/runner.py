"""
Multi-agent interaction loop for the PDDL planning benchmark.

PDDL is a dynamic environment: each turn the MAS proposes one action, the
AgentBoard PDDLGym backend executes it, and the next turn sees the new state.
The loop runs until the goal is reached, the wall-clock timeout expires, or
max_turns is reached.
"""
import logging
import re
import time

logger = logging.getLogger(__name__)


VALID_ACTIONS_PREFIX = "Valid actions are:"


META_ACTIONS = {"check valid actions", "look around"}


PDDL_ACTION_GUARD_MODES = ("strict",)


DEFAULT_PDDL_ACTION_GUARD = "strict"


def extract_action_details(response):
    """Extract one candidate while preserving malformed model text for audit."""
    raw_output = str(response or "")
    match = re.search(
        r"<action>(.*?)</action>",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        action = match.group(1).strip()
        return {
            "model_action": action,
            "extracted_action": action,
            "cleanup_steps": [],
        }

    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    model_action = lines[-1] if lines else raw_output.strip()
    extracted_action = model_action
    cleanup_steps = []

    malformed_closing = re.fullmatch(
        r"<(.+)</[a-z][a-z0-9_:-]*>\s*",
        extracted_action,
        re.DOTALL | re.IGNORECASE,
    )
    if malformed_closing:
        extracted_action = malformed_closing.group(1).strip()
        cleanup_steps.append("removed_malformed_angle_wrapper")
    else:
        pseudo_tag = re.fullmatch(r"<([^<>]+)>\s*", extracted_action, re.DOTALL)
        if pseudo_tag:
            extracted_action = pseudo_tag.group(1).strip()
            cleanup_steps.append("removed_pseudo_action_wrapper")

    without_prefix = re.sub(
        r"^action\s*:\s*",
        "",
        extracted_action,
        flags=re.IGNORECASE,
    )
    if without_prefix != extracted_action:
        extracted_action = without_prefix.strip()
        cleanup_steps.append("removed_action_prefix")

    unquoted = extracted_action.strip("`\"' ")
    if unquoted != extracted_action:
        extracted_action = unquoted
        cleanup_steps.append("removed_outer_quotes")

    return {
        "model_action": model_action,
        "extracted_action": extracted_action,
        "cleanup_steps": cleanup_steps,
    }


def extract_action(response):
    """Return the controlled action candidate extracted from model output."""
    return extract_action_details(response)["extracted_action"]


def extract_valid_actions_observation(observation):
    """Return the valid action list from a `check valid actions` observation."""
    if not observation:
        return []
    pattern = rf"{re.escape(VALID_ACTIONS_PREFIX)}\s*(.*)"
    match = re.search(pattern, observation, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    raw_actions = match.group(1).strip()
    if not raw_actions:
        return []
    return [part.strip() for part in raw_actions.split(",") if part.strip()]


def format_valid_action_constraint(valid_actions):
    if not valid_actions:
        return ""
    action_lines = "\n".join(f"- {action}" for action in valid_actions)
    return (
        "<valid_action_constraint>\n"
        "The environment lists the currently valid actions for this exact state. "
        "For this turn, you must output exactly one action copied from the list below. "
        "Do not translate it to another action syntax, invent a new action, "
        "or change object names, rooms, hands, or order. Prefer a progress action "
        "over diagnostic actions such as `check valid actions` or `look around`.\n"
        f"{action_lines}\n"
        "</valid_action_constraint>"
    )


def detect_no_progress_cycle(state_path, satisfied_path, action_path):
    """Return the most recent exact-state cycle for background diagnostics."""
    if len(state_path) < 2 or len(state_path) != len(satisfied_path):
        return None
    if len(action_path) + 1 != len(state_path):
        return None

    current_state = state_path[-1]
    previous_indices = [
        index
        for index, state in enumerate(state_path[:-1])
        if state == current_state
    ]
    if not previous_indices:
        return None

    # The latest occurrence identifies the shortest directly observed closed path.
    previous_index = previous_indices[-1]
    previous_satisfied = frozenset(satisfied_path[previous_index])
    current_satisfied = frozenset(satisfied_path[-1])
    if current_satisfied != previous_satisfied:
        return None

    cycle_actions = list(action_path[previous_index:])
    if not cycle_actions:
        return None
    return {
        "start_turn": previous_index + 1,
        "end_turn": len(action_path),
        "actions": cycle_actions,
        "satisfied_goals": sorted(current_satisfied),
        "self_loop": len(cycle_actions) == 1,
    }


def new_cycle_diagnostics():
    return {
        "exact_cycles": 0,
        "self_loops": 0,
        "two_step_cycles": 0,
        "longer_cycles": 0,
        "first_detected_turn": None,
    }


def update_cycle_diagnostics(diagnostics, cycle):
    if not cycle:
        return diagnostics
    diagnostics["exact_cycles"] += 1
    action_count = len(cycle["actions"])
    if action_count == 1:
        diagnostics["self_loops"] += 1
    elif action_count == 2:
        diagnostics["two_step_cycles"] += 1
    else:
        diagnostics["longer_cycles"] += 1
    if diagnostics["first_detected_turn"] is None:
        diagnostics["first_detected_turn"] = cycle["end_turn"]
    return diagnostics


def log_cycle_diagnostics(prefix, diagnostics):
    logger.debug(
        "    %s CYCLE_DIAGNOSTICS exact_cycles=%d self_loops=%d "
        "two_step_cycles=%d longer_cycles=%d first_detected_turn=%s",
        prefix,
        diagnostics["exact_cycles"],
        diagnostics["self_loops"],
        diagnostics["two_step_cycles"],
        diagnostics["longer_cycles"],
        diagnostics["first_detected_turn"]
        if diagnostics["first_detected_turn"] is not None
        else "none",
    )


def build_pddl_memory_query(task_description, state_observation, recent_feedback, goal_progress):
    observations = re.findall(
        r"<observation>(.*?)</observation>",
        str(state_observation or ""),
        re.DOTALL | re.IGNORECASE,
    )
    current_state = observations[-1].strip() if observations else str(state_observation or "").strip()
    parts = [
        str(task_description or "").strip(),
        f"Current observable state: {current_state}" if current_state else "",
        str(goal_progress or "").strip(),
        f"Most recent environment feedback: {str(recent_feedback or '').strip()}"
        if recent_feedback
        else "",
    ]
    return "\n\n".join(part for part in parts if part)


def normalize_action_text(action):
    text = extract_action(action or "").lower()
    text = re.sub(r"[`\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".")


def is_meta_action(action):
    return normalize_action_text(action) in META_ACTIONS


def filter_decision_actions(valid_actions):
    """Keep the prompt focused on executable progress actions when possible."""
    progress_actions = [action for action in valid_actions if not is_meta_action(action)]
    return progress_actions or valid_actions


def get_current_valid_actions(env):
    """Ask the environment for the current valid action space, if available."""
    action_space_getter = getattr(env, "_get_action_space", None)
    if not callable(action_space_getter):
        return []
    try:
        return filter_decision_actions(sorted(str(action) for action in action_space_getter()))
    except Exception as exc:
        logger.debug("Unable to get PDDL action space for prompt constraint: %s", exc)
        return []


def gripper_action_signature(action):
    """Parse canonical and rendered gripper actions into comparable signatures."""
    text = normalize_action_text(action)
    if text == "check valid actions":
        return ("check",)

    move = re.fullmatch(
        r"move(?:\s+from)?\s+([a-z0-9_-]+)\s+(?:to\s+)?([a-z0-9_-]+)",
        text,
    )
    if move:
        return ("move", move.group(1), move.group(2))

    pick = re.fullmatch(
        r"pick(?:\s+up)?\s+([a-z0-9_-]+)(?:\s+at)?\s+([a-z0-9_-]+)"
        r"(?:\s+with\s+arm)?\s+([a-z0-9_-]+)",
        text,
    )
    if pick:
        return ("pick", pick.group(1), pick.group(2), pick.group(3))

    drop = re.fullmatch(
        r"drop\s+([a-z0-9_-]+)(?:\s+at)?\s+([a-z0-9_-]+)"
        r"(?:\s+with\s+arm)?\s+([a-z0-9_-]+)",
        text,
    )
    if drop:
        return ("drop", drop.group(1), drop.group(2), drop.group(3))
    return None


def _find_valid_action_match(action, valid_actions):
    normalized = normalize_action_text(action)
    for valid in valid_actions:
        if normalize_action_text(valid) == normalized:
            return valid

    signature = gripper_action_signature(action)
    if signature is not None:
        for valid in valid_actions:
            if gripper_action_signature(valid) == signature:
                return valid
    return None


def goal_progress_snapshot(env):
    satisfied = []
    unsatisfied = []
    try:
        last_obs = getattr(env, "last_obs", None)
        goal_literals = getattr(env, "goal_literals", None) or []
        if last_obs is None:
            return satisfied, unsatisfied
        obs_literals = getattr(last_obs, "literals", set())
        literal_to_text = getattr(env, "_literal_to_text", None)
        for literal in goal_literals:
            text = literal_to_text(literal).strip() if callable(literal_to_text) else str(literal).strip()
            (satisfied if literal in obs_literals else unsatisfied).append(text)
    except Exception as exc:
        logger.debug("Unable to inspect PDDL goal progress: %s", exc)
    return satisfied, unsatisfied


def format_goal_progress_context(env):
    """Expose the complete satisfied/unsatisfied goal ledger."""
    satisfied, unsatisfied = goal_progress_snapshot(env)
    if not satisfied and not unsatisfied:
        return ""
    lines = [
        "<goal_progress>",
        "Satisfied goal conditions:",
    ]
    lines.extend(f"- {goal}" for goal in satisfied)
    if not satisfied:
        lines.append("- none")
    lines.append("Unsatisfied goal conditions:")
    lines.extend(f"- {goal}" for goal in unsatisfied)
    if not unsatisfied:
        lines.append("- none")
    lines.append("</goal_progress>")
    return "\n".join(lines)


def pddl_state_signature(env):
    last_obs = getattr(env, "last_obs", None)
    literals = getattr(last_obs, "literals", None)
    if literals is not None:
        return tuple(sorted(str(literal) for literal in literals))
    return str(getattr(env, "_get_obs", lambda: "")()).strip().lower()


def record_pddl_transition(
    action,
    before_signature,
    before_satisfied,
    after_signature,
    after_satisfied,
    state_path,
    satisfied_path,
    action_path,
    valid=True,
    reward=0.0,
    invalid_feedback="",
    environment_observation="",
):
    """Record one real transition and return its summary and diagnostic cycle."""
    before_satisfied = set(before_satisfied)
    after_satisfied = set(after_satisfied)
    action_path.append(action)
    state_path.append(after_signature)
    satisfied_path.append(frozenset(after_satisfied))
    transition = {
        "action": action,
        "valid": bool(valid),
        "state_changed": after_signature != before_signature,
        "reward": reward,
        "newly_satisfied": sorted(after_satisfied - before_satisfied),
        "newly_lost": sorted(before_satisfied - after_satisfied),
        "invalid_feedback": invalid_feedback,
        "environment_observation": environment_observation,
    }
    return transition, detect_no_progress_cycle(state_path, satisfied_path, action_path)


def _format_transition_values(values):
    return ", ".join(str(value) for value in values) if values else "none"


def format_action_history(transitions, limit=6):
    earlier_progress = [
        transition
        for transition in list(transitions or [])[:-1]
        if transition.get("valid") is True and transition.get("state_changed") is True
    ][-limit:]
    lines = ["<action_history>"]
    lines.extend(f"- {transition['action']}" for transition in earlier_progress)
    if not earlier_progress:
        lines.append("- none")
    lines.append("</action_history>")
    return "\n".join(lines)


def format_action_repetition_summary(transitions, minimum_attempts=3):
    summaries = {}
    action_order = []
    for transition in transitions or []:
        action = str(transition.get("action", ""))
        action_key = normalize_action_text(action)
        if action_key not in summaries:
            summaries[action_key] = {
                "action": action,
                "attempts": 0,
                "invalid": 0,
                "state_changes": 0,
                "goal_condition_changes": 0,
            }
            action_order.append(action_key)
        summary = summaries[action_key]
        summary["attempts"] += 1
        summary["invalid"] += int(transition.get("valid") is False)
        summary["state_changes"] += int(bool(transition.get("state_changed")))
        summary["goal_condition_changes"] += int(bool(
            transition.get("newly_satisfied") or transition.get("newly_lost")
        ))
    repeated = [
        summaries[action_key]
        for action_key in action_order
        if summaries[action_key]["attempts"] >= minimum_attempts
    ]
    if not repeated:
        return ""
    lines = ["<action_repetition_summary>"]
    for summary in repeated:
        lines.extend([
            f"- action: {summary['action']}",
            f"  attempts: {summary['attempts']}",
            f"  invalid: {summary['invalid']}",
            f"  state_changes: {summary['state_changes']}",
            f"  goal_condition_changes: {summary['goal_condition_changes']}",
        ])
    lines.append("</action_repetition_summary>")
    return "\n".join(lines)


def format_last_transition(transitions):
    if not transitions:
        return ""
    transition = transitions[-1]
    invalid_feedback = transition.get("invalid_feedback") or "none"
    return "\n".join([
        "<last_transition>",
        f"action: {transition.get('action', '')}",
        f"valid: {str(bool(transition.get('valid'))).lower()}",
        f"state_changed: {str(bool(transition.get('state_changed'))).lower()}",
        f"reward: {transition.get('reward', 0.0)}",
        f"newly_satisfied: {_format_transition_values(transition.get('newly_satisfied', []))}",
        f"newly_lost: {_format_transition_values(transition.get('newly_lost', []))}",
        f"invalid_feedback: {invalid_feedback}",
        "</last_transition>",
    ])


def pddl_action_is_valid(env, observation):
    feedback = "" if observation is None else str(observation)
    infos = getattr(env, "infos", None)
    if isinstance(infos, dict):
        action_is_valid = infos.get("action_is_valid")
    else:
        action_is_valid = getattr(infos, "action_is_valid", None)
    invalid_markers = (
        "not valid",
        "invalid action",
        "action is invalid",
        "illegal action",
    )
    return action_is_valid is not False and not any(
        marker in feedback.lower() for marker in invalid_markers
    )


def get_invalid_environment_feedback(env, observation):
    """Return bounded invalid feedback only when the environment marks it invalid."""
    feedback = "" if observation is None else str(observation)
    if pddl_action_is_valid(env, observation):
        return ""
    compact = re.sub(r"\s+", " ", feedback).strip()
    return compact[:800]


def build_pddl_task_context(
    init_user_prompt,
    initial_state_observation,
    latest_state_observation,
    transition_history,
    goal_progress_context="",
    valid_action_constraint="",
):
    """Build the full agent context from the current transition ledger."""
    del initial_state_observation
    return "\n\n".join(
        part
        for part in (
            str(init_user_prompt or "").strip(),
            (
                "<current_observation>\n"
                f"{latest_state_observation}\n"
                "</current_observation>"
            ),
            format_action_history(transition_history),
            format_action_repetition_summary(transition_history),
            format_last_transition(transition_history),
            goal_progress_context,
            valid_action_constraint,
        )
        if part
    )


def build_pddl_validator_context(
    task_goal,
    current_state,
    transition_history,
    goal_progress_context="",
):
    """Build a compact PDDL validator context without examples or valid actions."""
    return "\n\n".join(
        part
        for part in (
            f"<task_goal>\n{task_goal}\n</task_goal>",
            f"<current_state>\n{current_state}\n</current_state>",
            format_action_history(transition_history),
            format_action_repetition_summary(transition_history),
            format_last_transition(transition_history),
            goal_progress_context,
        )
        if part
    )


def _guard_decision(
    action,
    executed_action,
    match_mode,
    reason,
    guard_note,
    valid_actions,
    model_action=None,
    cleanup_steps=None,
    guard_mode=DEFAULT_PDDL_ACTION_GUARD,
    was_replaced=False,
):
    raw_action = action if model_action is None else model_action
    return {
        "guard_mode": guard_mode,
        "model_action": raw_action,
        "extracted_action": action,
        "executed_action": executed_action,
        "match_mode": match_mode,
        "reason": reason,
        "replacement_reason": reason if was_replaced else "",
        "was_normalized": raw_action != executed_action,
        "was_replaced": bool(was_replaced),
        "cleanup_steps": list(cleanup_steps or []),
        "valid_action_count": len(valid_actions),
        "guard_note": guard_note,
    }


def guard_action_decision(
    action,
    valid_actions,
    state_observation="",
    action_resolver=None,
    model_action=None,
    cleanup_steps=None,
    pddl_action_guard=DEFAULT_PDDL_ACTION_GUARD,
):
    """Canonicalize only actions that resolve to one current legal action."""
    if pddl_action_guard not in PDDL_ACTION_GUARD_MODES:
        raise ValueError(
            f"Unsupported pddl_action_guard={pddl_action_guard!r}; "
            f"supported modes: {PDDL_ACTION_GUARD_MODES}"
        )
    decision_kwargs = {
        "model_action": model_action,
        "cleanup_steps": cleanup_steps,
        "guard_mode": pddl_action_guard,
    }
    if not valid_actions:
        return _guard_decision(
            action,
            action,
            "unconstrained",
            "no valid-action list was available",
            "",
            valid_actions,
            **decision_kwargs,
        )

    matched = _find_valid_action_match(action, valid_actions)
    if matched:
        if normalize_action_text(matched) == normalize_action_text(action):
            return _guard_decision(
                action,
                matched,
                "normalized_exact",
                "normalized model output exactly matched a listed valid action",
                "",
                valid_actions,
                **decision_kwargs,
            )
        return _guard_decision(
            action,
            matched,
            "gripper_signature",
            "Gripper operator and all ordered parameters matched a listed action",
            "",
            valid_actions,
            **decision_kwargs,
        )

    if callable(action_resolver):
        resolved = action_resolver(action)
        resolved_match = next(
            (
                valid
                for valid in valid_actions
                if normalize_action_text(valid) == normalize_action_text(resolved or "")
            ),
            None,
        )
        if resolved_match is not None:
            return _guard_decision(
                action,
                resolved_match,
                "environment_literal",
                "operator and all ordered parameters resolved to one listed action",
                "",
                valid_actions,
                **decision_kwargs,
            )

    return _guard_decision(
        action,
        action,
        "environment_validation",
        "no exact listed equivalent; the environment validates the extracted action",
        "",
        valid_actions,
        **decision_kwargs,
    )


def guard_action_with_valid_actions(
    action,
    valid_actions,
    state_observation="",
    action_resolver=None,
    model_action=None,
    cleanup_steps=None,
    pddl_action_guard=DEFAULT_PDDL_ACTION_GUARD,
):
    """Canonicalize exact current-state equivalents before environment execution."""
    decision = guard_action_decision(
        action,
        valid_actions,
        state_observation,
        action_resolver=action_resolver,
        model_action=model_action,
        cleanup_steps=cleanup_steps,
        pddl_action_guard=pddl_action_guard,
    )
    return decision["executed_action"], decision["guard_note"]


def _compact_action_for_log(action, limit=80):
    compact = re.sub(r"\s+", " ", str(action or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _deadline_from_timeout(timeout_seconds):
    if timeout_seconds is None:
        return None
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return None
    return time.monotonic() + timeout


def _deadline_expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _append_timeout_feedback(feedback, timeout_seconds):
    if timeout_seconds is None:
        return feedback
    return f"{feedback} timeout=True budget_seconds={float(timeout_seconds):.1f}"


def _format_env_feedback(actions, observation, reward, score, done):
    if not actions:
        return "no action executed"
    joined = " | ".join(a[:80] for a in actions)
    return (
        f"actions=[{joined}] step_reward={reward} score={score:.3f} "
        f"done={done} observation={(observation or '')[:200]}"
    )


def run_multiturn_mas(
    mas,
    env,
    system_prompt,
    init_user_prompt,
    generation_config,
    max_turns,
    timeout_seconds=None,
    memory_task_description=None,
    execution_label="OFFICIAL",
    execution_task_id="",
    initial_state_observation=None,
    pddl_action_guard=DEFAULT_PDDL_ACTION_GUARD,
):
    """Run multi-turn MAS interaction with PDDLEnv.

    Each turn calls MAS generation, extracts the final action, and executes one
    env.step. The trajectory schema matches KodCode: each MAS agent node emits
    one step. Only the final agent receives the environment feedback summary;
    other agent feedback fields stay empty. If no MAS graph exists, a single
    agent="mas" fallback step is emitted.
    """
    import networkx as nx

    all_steps = []
    transition_history = []
    deadline = _deadline_from_timeout(timeout_seconds)
    pending_valid_actions = []
    fixed_initial_state_observation = str(
        env._get_obs() if initial_state_observation is None else initial_state_observation
    )
    latest_state_observation = fixed_initial_state_observation
    initial_signature = pddl_state_signature(env)
    state_path = [initial_signature]
    satisfied_path = [frozenset(goal_progress_snapshot(env)[0])]
    action_path = []
    cycle_diagnostics = new_cycle_diagnostics()
    memory_task_description = memory_task_description or init_user_prompt
    execution_prefix = f"[{execution_label}][task={execution_task_id or 'unknown'}]"
    verbose_turn_logging = str(execution_label).upper() == "OFFICIAL"
    stop_reason = "max_turns"
    turns_executed = 0

    for turn in range(max_turns):
        if _deadline_expired(deadline):
            logger.info(
                f"    {execution_prefix} PDDL timeout before turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            stop_reason = "timeout"
            break

        current_valid_actions = get_current_valid_actions(env) or pending_valid_actions
        goal_progress_context = format_goal_progress_context(env)
        valid_action_constraint = format_valid_action_constraint(current_valid_actions)
        current_task_context = build_pddl_task_context(
            init_user_prompt,
            fixed_initial_state_observation,
            latest_state_observation,
            transition_history,
            goal_progress_context,
            valid_action_constraint,
        )
        validator_task_context = build_pddl_validator_context(
            memory_task_description,
            latest_state_observation,
            transition_history,
            goal_progress_context,
        )
        step_memory_query = build_pddl_memory_query(
            memory_task_description,
            latest_state_observation,
            last_observation if turn > 0 else "",
            goal_progress_context,
        )

        try:
            try:
                generate_kwargs = {
                    "memory_task_descriptions": [memory_task_description],
                    "memory_queries": [step_memory_query],
                    "validator_task_descriptions": [validator_task_context],
                    "validator_valid_action_constraints": [valid_action_constraint],
                }
                graphs = mas.generate(
                    [system_prompt],
                    [current_task_context],
                    generation_config,
                    **generate_kwargs,
                )
            except TypeError as e:
                error_text = str(e)
                if (
                    "validator_task_descriptions" not in error_text
                    and "validator_valid_action_constraints" not in error_text
                    and "memory_queries" not in error_text
                    and "memory_task_descriptions" not in error_text
                ):
                    raise
                generate_kwargs.pop("validator_task_descriptions", None)
                generate_kwargs.pop("validator_valid_action_constraints", None)
                try:
                    graphs = mas.generate(
                        [system_prompt],
                        [current_task_context],
                        generation_config,
                        **generate_kwargs,
                    )
                except TypeError as fallback_error:
                    if "memory_queries" not in str(fallback_error):
                        raise
                    generate_kwargs.pop("memory_queries", None)
                    graphs = mas.generate(
                        [system_prompt],
                        [current_task_context],
                        generation_config,
                        **generate_kwargs,
                    )
            msg_graph = graphs[0]
            response = msg_graph.action or ""
        except Exception as e:
            logger.warning(f"    {execution_prefix} Turn {turn+1}: MAS generate failed: {e}")
            response = ""
            msg_graph = None

        if _deadline_expired(deadline):
            feedback_str = _append_timeout_feedback(
                _format_env_feedback([], "", 0.0, env.feedback(), False),
                timeout_seconds,
            )
            all_steps.append({
                "step_index": len(all_steps) + 1,
                "agent": "mas",
                "input": f"{system_prompt}\n\n{current_task_context}" if system_prompt else current_task_context,
                "output": response,
                "tool_calls": "",
                "feedback": feedback_str,
                "action_guard": None,
                "transition": None,
            })
            turns_executed = turn + 1
            stop_reason = "timeout"
            break

        action_details = extract_action_details(response)
        actions = (
            [action_details]
            if action_details["extracted_action"]
            else []
        )

        # Execute the single action first, then attach one environment feedback summary.
        done = False
        last_observation = ""
        last_reward = 0.0
        last_score = env.feedback()
        executed_actions = []
        guard_notes = []
        guard_decisions = []
        last_transition = None
        for details in actions:
            if _deadline_expired(deadline):
                break
            action = details["extracted_action"]
            guard_decision = guard_action_decision(
                action,
                current_valid_actions,
                latest_state_observation,
                action_resolver=getattr(env, "resolve_action_text", None),
                model_action=details["model_action"],
                cleanup_steps=details["cleanup_steps"],
                pddl_action_guard=pddl_action_guard,
            )
            action_to_execute = guard_decision["executed_action"]
            guard_note = guard_decision["guard_note"]
            guard_decisions.append(guard_decision)
            if guard_note:
                guard_notes.append(guard_note)
                if verbose_turn_logging:
                    logger.debug(
                        "    %s PDDL_GUARD guard=%s match=%s replaced=%s model=%r executed=%r",
                        execution_prefix,
                        guard_decision["guard_mode"],
                        guard_decision["match_mode"],
                        guard_decision["was_replaced"],
                        _compact_action_for_log(action),
                        _compact_action_for_log(action_to_execute),
                    )
            before_signature = pddl_state_signature(env)
            before_satisfied = set(goal_progress_snapshot(env)[0])
            observation, reward, done = env.step(action_to_execute)
            current_score = env.feedback()
            after_signature = pddl_state_signature(env)
            after_satisfied = set(goal_progress_snapshot(env)[0])
            invalid_feedback = get_invalid_environment_feedback(env, observation)
            last_transition, detected_cycle = record_pddl_transition(
                action_to_execute,
                before_signature,
                before_satisfied,
                after_signature,
                after_satisfied,
                state_path,
                satisfied_path,
                action_path,
                valid=pddl_action_is_valid(env, observation),
                reward=reward,
                invalid_feedback=invalid_feedback,
                environment_observation=observation,
            )
            transition_history.append(last_transition)
            update_cycle_diagnostics(cycle_diagnostics, detected_cycle)
            last_observation = observation
            last_reward = reward
            last_score = current_score
            executed_actions.append(action_to_execute)
            if verbose_turn_logging:
                logger.info(
                    f"    {execution_prefix} Turn {turn+1} action='{action_to_execute[:60]}' "
                    f"step_reward={reward} score={current_score:.3f} done={done}"
                )
            if done:
                break
        feedback_str = _format_env_feedback(
            executed_actions, last_observation, last_reward, last_score, done
        )
        if guard_notes:
            feedback_str = f"{feedback_str} {'; '.join(guard_notes)}"
        timed_out = _deadline_expired(deadline)
        if timed_out and not done:
            feedback_str = _append_timeout_feedback(feedback_str, timeout_seconds)

        # Split one step per agent role; attach feedback only to the final agent.
        if msg_graph and msg_graph.mas_message_graph is not None:
            graph = msg_graph.mas_message_graph
            try:
                topo_order = list(nx.topological_sort(graph))
            except Exception:
                topo_order = list(graph.nodes())
            total = len(topo_order)
            for idx, agent_role in enumerate(topo_order):
                node_data = graph.nodes[agent_role]
                msg_node = node_data.get("message")
                extra_fields = ((getattr(msg_node, "user_prompt_fields", None) or {}).get("extra_fields") or {}) if msg_node else {}
                system_part = (
                    getattr(msg_node, "formatted_system_prompt", None)
                    or (msg_node.system_prompt_template if msg_node else "")
                    or system_prompt
                )
                user_part = (
                    getattr(msg_node, "formatted_user_prompt", None)
                    or (msg_node.user_prompt_template if msg_node else "")
                    or current_task_context
                )
                agent_input = f"{system_part}\n\n{user_part}" if system_part else user_part
                agent_output = msg_node.response if (msg_node and msg_node.response) else ""
                is_last = idx == total - 1
                all_steps.append({
                    "step_index": len(all_steps) + 1,
                    "agent": str(agent_role),
                    "input": agent_input,
                    "output": agent_output,
                    "tool_calls": "",
                    "feedback": feedback_str if is_last else "",
                    "turn": turn + 1,
                    "upstream_roles": [str(role) for role in graph.predecessors(agent_role)],
                    "injected_memories": list(extra_fields.get("injected_memories") or []),
                    "memory_retrieval": list(extra_fields.get("retrieval_details") or []),
                    "state_summary": current_task_context,
                    "step_reward": last_reward if is_last else 0.0,
                    "done": bool(done) if is_last else False,
                    "action_guard": (
                        guard_decisions[-1]
                        if is_last and guard_decisions
                        else None
                    ),
                    "transition": last_transition if is_last else None,
                })
        else:
            # Fallback: synthesize one aggregate step when no MAS graph is available.
            all_steps.append({
                "step_index": len(all_steps) + 1,
                "agent": "mas",
                "input": f"{system_prompt}\n\n{current_task_context}" if system_prompt else current_task_context,
                "output": response,
                "tool_calls": "",
                "feedback": feedback_str,
                "action_guard": guard_decisions[-1] if guard_decisions else None,
                "transition": last_transition,
            })

        turns_executed = turn + 1
        if done:
            stop_reason = "goal_reached"
            break
        if timed_out:
            logger.info(
                f"    {execution_prefix} PDDL timeout after turn {turn+1} "
                f"(budget={float(timeout_seconds):.1f}s)"
            )
            stop_reason = "timeout"
            break

        pending_valid_actions = extract_valid_actions_observation(last_observation)
        if (
            last_observation
            and not pending_valid_actions
            and (last_transition is None or last_transition["valid"])
        ):
            latest_state_observation = last_observation

    final_reward = env.feedback()
    won = env.won
    log_cycle_diagnostics(execution_prefix, cycle_diagnostics)
    logger.info(
        "    %s STOP reason=%s interaction_rounds=%d max_turns=%d timeout_seconds=%s won=%s final_score=%.3f",
        execution_prefix,
        stop_reason,
        turns_executed,
        max_turns,
        "disabled" if timeout_seconds is None or float(timeout_seconds) <= 0 else float(timeout_seconds),
        won,
        final_reward,
    )
    return final_reward, won, all_steps
