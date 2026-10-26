def get_domain_prompt(task_domain: str):

    if task_domain == "kodcode":
        from mas.autogen.prompts import kodcode_prompt
        return kodcode_prompt
    elif task_domain == "triviaqa":
        from mas.autogen.prompts import triviaqa_prompt
        return triviaqa_prompt
    elif task_domain == "popqa":
        from mas.autogen.prompts import popqa_prompt
        return popqa_prompt
    elif task_domain == "pddl":
        from mas.autogen.prompts import pddl_prompt
        return pddl_prompt
    else:
        raise ValueError(f"Unsupported task domina: {task_domain}")


# --- guidance for memory methods (looked up via common/memory_guidance.py) ---
# MEMORY_ROLE_GUIDANCE keys are matched in order as case-insensitive substrings
# of an agent's role name; MEMORY_DUTY_FALLBACK (with {duty}) and
# MEMORY_DEFAULT_ROLE_GUIDANCE cover unmatched roles.

MEMORY_SYSTEM_ADDENDUM = """\
Distinguish the assistant proposal from proxy validation and environment action.
Context supports evidence, not ownership."""

MEMORY_ROLE_GUIDANCE = {
    "assistant": """\
Store a planning rule when observable evidence explains a goal-progressing action
or next subgoal. If repeated proposals recreate a prior state and task evidence
identifies a valid alternative at that branch, store the concrete transferable
next action; use an abstract rejection only when no alternative is supported. A
linked `downstream_outcomes` record may verify the proposal without making the
assistant the environment actor. Put conditions in `use_when`. Routine proposals
and source solutions are `NO_MEMORY`.""",
    "user proxy": """\
### Eligibility Gate

Require a proxy output with a demonstrated validation or correction
decision. It qualifies when the proxy rejects/revises the assistant proposal, chooses a
different listed valid action, or selects a goal-progressing action after a state check. Do not store unchanged forwarding, agreement, formatting/paraphrase, or success without a
proxy decision. Wrappers are forwarding. A loop is failure evidence for planning-role extraction, not proxy correction;
examples alone cannot satisfy this gate.

### Memory

After the gate passes, store only demonstrated validation or correction:
reject an invalid proposal, revise after invalid/no-op feedback, reject immediate
restoration of a removed relation, or choose a listed goal-progressing action.
Store the boundary/selection rule, not a complete plan or another role's action.
Use `Reject`, `Revise`, `Check`, or `Choose`.""",
}

MEMORY_DUTY_FALLBACK = """\
Store only a non-routine decision demonstrated within this duty: {duty}"""

MEMORY_DEFAULT_ROLE_GUIDANCE = """\
Store only a non-routine decision demonstrated by the target role."""
