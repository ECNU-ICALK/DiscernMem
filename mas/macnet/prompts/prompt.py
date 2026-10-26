def get_domain_prompt(task_domain: str):
    if task_domain == "kodcode":
        from mas.macnet.prompts import kodcode_prompt
        return kodcode_prompt
    elif task_domain == "triviaqa":
        from mas.macnet.prompts import triviaqa_prompt
        return triviaqa_prompt
    elif task_domain == "popqa":
        from mas.macnet.prompts import popqa_prompt
        return popqa_prompt
    elif task_domain == "pddl":
        from mas.macnet.prompts import pddl_prompt
        return pddl_prompt
    else:
        raise ValueError(f"Unsupported task domina: {task_domain}")


# --- guidance for memory methods (looked up via common/memory_guidance.py) ---
# MEMORY_ROLE_GUIDANCE keys are matched in order as case-insensitive substrings
# of an agent's role name; MEMORY_DUTY_FALLBACK (with {duty}) and
# MEMORY_DEFAULT_ROLE_GUIDANCE cover unmatched roles.

MEMORY_SYSTEM_ADDENDUM = """\
When interpreting MacNet traces, keep evidence and decisions within the observed
branch. Only a role that receives multiple branches may compare them."""

MEMORY_ROLE_GUIDANCE = {
    "actor": """\
Store a branch-local candidate or execution rule only when an observed constraint
or result selects among plausible alternatives. Routine generation and claims
based on another branch are `NO_MEMORY`.""",
    "critic": """\
Store a branch-local check only when it catches or rules out a specific plausible
error. Repeated requirements, generic review, agreement, and evidence visible
only to another branch are `NO_MEMORY`.""",
    "summarizer": """\
Store a cross-branch synthesis rule only when it resolves an observed conflict
or preserves a non-obvious verified constraint. Copying agreeing branches is
`NO_MEMORY`.""",
}

MEMORY_DUTY_FALLBACK = """\
Store only a non-routine decision demonstrated within this duty: {duty}"""

MEMORY_DEFAULT_ROLE_GUIDANCE = """\
Store only a non-routine decision demonstrated by the target role."""
