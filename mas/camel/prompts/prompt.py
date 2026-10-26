def get_domain_prompt(task_domain: str):
    if task_domain == "kodcode":
        from mas.camel.prompts import kodcode_prompt
        return kodcode_prompt  
    elif task_domain == "triviaqa":
        from mas.camel.prompts import triviaqa_prompt
        return triviaqa_prompt
    elif task_domain == "popqa":
        from mas.camel.prompts import popqa_prompt
        return popqa_prompt
    elif task_domain == "pddl":
        from mas.camel.prompts import pddl_prompt
        return pddl_prompt
    else:
        raise ValueError(f"Unsupported task domina: {task_domain}")


# --- guidance for memory methods (looked up via common/memory_guidance.py) ---
# MEMORY_ROLE_GUIDANCE keys are matched in order as case-insensitive substrings
# of an agent's role name; MEMORY_DUTY_FALLBACK (with {duty}) and
# MEMORY_DEFAULT_ROLE_GUIDANCE cover unmatched roles.

MEMORY_SYSTEM_ADDENDUM = """\
When interpreting CAMEL traces, respect the observed specialist handoffs.
Received context may support a decision but does not transfer decision ownership.
Task-family guidance is the final authority for what qualifies as transferable."""

MEMORY_ROLE_GUIDANCE = {
    "user proxy": """\
Store a requirement analysis, query reformulation, or ambiguity-resolution rule
only when the target's change is observable and changes downstream reasoning.
Restatement, search, forwarding, and source-specific answers are `NO_MEMORY`.""",
    "actor": """\
Store a target-owned decision only when an observed constraint or result selects
among plausible alternatives and the task-family gate is satisfied. Routine
generation, implementation details, and complete source solutions are
`NO_MEMORY`.""",
    "critic": """\
Store a concrete check only when it catches or rules out a plausible error in a
received candidate. Repeating a task requirement, generic review, agreement, or
an untested critique is `NO_MEMORY`.""",
    "summarizer": """\
Store a synthesis rule only when it resolves an observed conflict or preserves a
non-obvious verified constraint. Copying an agreed answer is `NO_MEMORY`.""",
}

MEMORY_DUTY_FALLBACK = """\
Store only a non-routine decision demonstrated within this duty: {duty}"""

MEMORY_DEFAULT_ROLE_GUIDANCE = """\
Store only a non-routine decision demonstrated by the target role."""
