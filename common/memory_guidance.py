"""MAS and task guidance that memory methods can add to their prompts.

MAS guidance lives with each MAS in ``mas/<mas>/prompts/prompt.py`` as the
``MEMORY_*`` constants. Task guidance for every benchmark is defined below in
TASK_GUIDANCE; DEFAULT_TASK_GUIDANCE covers task families without an entry.
Each memory method decides whether and where it uses these texts.

To support a new MAS, add the ``MEMORY_*`` constants to its ``prompt.py`` and
list the module in MAS_PROMPT_MODULES. To support a new benchmark, add an entry
to TASK_GUIDANCE.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


MAS_PROMPT_MODULES = {
    "autogen": "mas.autogen.prompts.prompt",
    "camel": "mas.camel.prompts.prompt",
    "macnet": "mas.macnet.prompts.prompt",
}


TASK_GUIDANCE = {
    "pddl": {
        "families": ("pddl", "gripper", "blockworld", "barman", "tyreworld"),
        "extraction": """\
### Selection Priority

Store one goal-conditioned decision: an action or subgoal for a planning role, and a check or correction boundary for a validating role. Rank candidates:

1. an action that newly satisfies a goal or observably increases progress;
2. a demonstrated recovery from an invalid action or unchanged state;
3. a listed valid alternative that breaks an observed short action/state cycle;
4. an operator ordering that establishes the next required subgoal;
5. a prerequisite check that changes the next decision.

A state-changing action is not progress inside a loop. For a repeated two-state
cycle, treat the repeated branch and its valid-action list as local counterfactual
evidence. Select one listed action that avoids the inverse transition. Store only
this executable choice; do not infer later plan steps.

### Relational Abstraction

- Inspect `newly_satisfied`, score changes, and state changes before drafting.
- Replace source objects, locations, tools, and values with task relations such as carried object, target location, required support, or available tool. Never retain constants such as `ball3`, `rooma`, `b5`, or `r1`.
- In blockworld, do not immediately restore a support relation that was removed because it conflicts with the goal. Keep the obstructing block separated while clearing required supports, then build goal supports from the bottom upward.
- In a blockworld stack/unstack cycle, prefer a valid `Putdown` over stack-back when it keeps a just-cleared wrong support separated.
- In gripper, track undelivered objects, preserve delivered objects, and reuse a carrying trip for compatible undelivered objects when capacity permits.
- Preserve observed causal order. Put preconditions in `use_when`, keep all conditions jointly satisfiable, and keep conditions out of `memory`.
- gripper: acquire an unsatisfied object, move while carrying it, then release it at its target;
- blockworld: remove a wrong support or obstruction before placing a block on its required clear support;
- barman: preserve observed holding, container, cleanliness, fill, pour, shake, and serve dependencies;
- tyreworld: preserve observed lifted, loose, mounted, fastened, inflated, tool, and boot-storage dependencies.""",
        "examples": """\
These forms define role and abstraction level, not answers to copy.

Planning cycle recovery:
`{"decision":"STORE","memory":"Put down the held obstruction instead of stacking it back on the support just cleared.","use_when":["The obstruction is held after removal from a wrong support","Stacking it back would recreate the prior state","Putdown is listed as valid"],"avoid_when":[]}`

Validation: `Reject a proposal that recreates the state observed before the preceding action when another valid action is available.`

Do not copy source entities or expand either form into a complete plan.

### insight
- Among listed valid actions, prefer one that directly changes an unsatisfied goal relation over auxiliary grasp/leave actions.
- After an action leaves the relevant state unchanged, do not repeat that action or its inverse; choose another listed action that advances the goal.
- First reject a proposal that directly reverses <last_transition> without satisfying a new goal condition. In blockworld, after `Unstack X from Y`, reject `Stack X on Y`; if `Putdown X` is valid, output it instead.""",
    },
    "triviaqa": {
        "families": ("triviaqa",),
        "extraction": """\
Default to the exact `NO_MEMORY` object. Store at most one transferable rule
from target-owned `role_messages` only when:

1. The target makes an explicit retrieval, entity-selection, relation-checking,
   or query-repair decision.
2. Retrieved evidence or feedback shows that the decision resolves ambiguity,
   fixes a weak search, or prevents an evidenced wrong match.
3. The rule remains useful after removing names, titles, dates, values, and
   answers.

Two searches are not required. A single search qualifies when the target uses
an explicit role, time, entity, or answer-type constraint to select evidence.
Do not infer a decision merely from a correct final answer. The target must
explicitly state or perform the selection/correction; do not invent ambiguity
from a passage or turn an answer/non-answer into a memory rule.

Use only target-owned `role_messages`: incoming context is received information,
and downstream actions or later outcomes cannot be used to invent the target's
reasoning. A failed trajectory qualifies only when the target demonstrates an
evidence-supported correction or preventive choice.

Prefer rules about:

- bridging descriptive clues (occupation, event, organization, work) to the
  intended entity;
- separating similarly named entities and historical time contexts;
- checking that evidence states the requested relation, not merely the same
  entity;
- matching the answer type requested by the question.

Write one concise imperative rule beginning with an action verb, using the
pattern `trigger -> action -> evidence test -> error prevented`. The action
must be a retrieval or evidence-selection action; never write a rule about
accepting/rejecting an answer string or about the final answer state. Keep
conditions out of `memory`; make `use_when` observable and `avoid_when` an
evidenced, mutually exclusive exception.

Before storing, perform a literal overlap check against the question and evidence.
If any field copies a concrete token or phrase (especially a proper name, title,
place, year, acronym, quoted string, value, or answer), return `NO_MEMORY`; use
only abstract roles. Never store literal answers,
names, titles, dates, source facts, full queries, query templates, popularity
priors, or generic advice such as “search more” or “verify carefully”. A rule that only says to verify or accept explicit evidence, without an
observed ambiguity, wrong candidate, or query repair, is `NO_MEMORY`. Never
write “accept the answer as correct”; describe a retrieval action or return
`NO_MEMORY`. A direct one-hop extraction is `NO_MEMORY` unless the target also makes an
explicit ambiguity resolution, wrong-candidate rejection, or query repair. A
plain lookup, repeated query without a changed clue,
answer derivation, answer-state claims, forwarding, formatting, or finalization is `NO_MEMORY`.""",
        "examples": """\
```json
{"decision":"STORE","memory":"Use temporal and role constraints to identify the intended entity before extracting the requested relation.","use_when":["Several entities fit the description or historical context"],"avoid_when":["The entity is uniquely identified and the source directly states the relation"]}
```

```json
{"decision":"NO_MEMORY","memory":"NO_MEMORY","use_when":[],"avoid_when":[]}
```""",
    },
    "popqa": {
        "families": ("popqa",),
        "extraction": """\
Default to the exact `NO_MEMORY` object. Store at most one transferable rule
from target-owned `role_messages` only when:

1. The target makes an explicit subject, entity-type, relation, direction,
   context, or source-selection decision.
2. Retrieved evidence or feedback supports that decision and shows that it
   resolves ambiguity, rejects a neighboring relation, or prevents a wrong
   match.
3. The rule remains useful after removing concrete subjects, objects, titles,
   people, and answers.

A direct one-hop lookup or extraction of an explicitly stated relation is
`NO_MEMORY` by itself. It qualifies only when the target also resolves an
ambiguity, rejects a wrong candidate/relation, or repairs the query. Two searches
are not required for that decision. Do not infer a decision merely from a correct
answer.

Use only target-owned `role_messages`: incoming context is received information,
and downstream actions or later outcomes cannot be used to invent the target's
reasoning. A failed trajectory qualifies only when the target demonstrates an
evidence-supported correction or preventive choice.

Prefer rules about:

- matching the subject to the correct entity or media type;
- preserving `subject -> relation -> object` direction;
- distinguishing neighboring relations such as author, director, producer,
  actor, capital, located-in, and member-of;
- using time, geography, or organizational scope;
- accepting direct relation evidence, not a name merely mentioned in a source.

Write one concise imperative rule beginning with an action verb, using the
pattern `trigger -> action -> evidence test -> error prevented`. The action
must be a retrieval or evidence-selection action; never write a rule about
accepting/rejecting an answer string or about the final answer state. Keep
conditions out of `memory`; make `use_when` observable and `avoid_when` an
evidenced, mutually exclusive exception.

Before storing, perform a literal overlap check against the question and evidence.
If any field copies a concrete token or phrase (especially a proper name, title,
place, year, acronym, quoted string, value, or answer), return `NO_MEMORY`; use
only abstract roles. Never store facts, answers,
names, titles, source wording, full queries, query templates, or
popularity/fame/model-familiarity tie-breakers. A rule that only says to verify
or accept explicit evidence, without an observed ambiguity, wrong candidate, or
query repair, is `NO_MEMORY`. Never write “accept the answer as correct”;
describe a retrieval action or return `NO_MEMORY`. A plain lookup, mere
corroboration, repeated query without a changed clue, unresolved search,
answer-state claims, forwarding, formatting, or generic “verify/search” advice is `NO_MEMORY`.""",
        "examples": """\
```json
{"decision":"STORE","memory":"Match the subject and work type before accepting direct evidence for the requested relation.","use_when":["A label can refer to multiple media or entity types"],"avoid_when":["The subject is uniquely typed and the source explicitly states the relation"]}
```

```json
{"decision":"NO_MEMORY","memory":"NO_MEMORY","use_when":[],"avoid_when":[]}
```""",
    },
    "kodcode": {
        "families": ("kodcode",),
        "extraction": """\
Task-specific rules below override generic MAS role guidance. Default to the
exact `NO_MEMORY` object; decide eligibility before writing any memory.

### Evidence Gate

- For an implementation/actor role, `STORE` requires two target-owned solutions with
  specific failure feedback between them and a passing test after the second.
  One target message before feedback is always `NO_MEMORY`.
- For a critic, `STORE` requires a specific defect in the incoming solution,
  a downstream change at exactly that point, and a passing test afterward. If
  the requested behavior was already present, return `NO_MEMORY`.
- A requirement-analysis message, code-only summarizer, approval, or formatting
  step is `NO_MEMORY`.

For every role, `All tests failed` means `NO_MEMORY`; never salvage the
proposed fix. An overall pass without the required before/feedback/change chain
does not validate implementation choices.

### Selection Priority

- After the gate passes, store one smallest causal rule: a specification contract,
edge-case partition, representation or mutation invariant, ordering constraint,
or error/output semantic.

### Relational Abstraction

Describe input/state relations and required behavior. If the rule requires a
function or variable name, code, pseudocode, formula, complexity notation,
library, data structure, named algorithm, exact test, or complete solution,
return `NO_MEMORY` instead. Put observable applicability in `use_when`.""",
        "examples": """\
Observed: a critic identifies lost duplicate occurrences, downstream changes
membership checking to occurrence counting, and the failing duplicate case
passes.

```json
{"decision":"STORE","memory":"Track multiplicity when repeated values must be consumed separately.","use_when":["Duplicates represent distinct consumable occurrences"],"avoid_when":["Only membership matters"]}
```

Observed: an actor emits one complete solution before feedback; or a proposed
fix ends with `All tests failed`.

```json
{"decision":"NO_MEMORY","memory":"NO_MEMORY","use_when":[],"avoid_when":[]}
```""",
    },
}

DEFAULT_TASK_GUIDANCE = {
        "extraction": """\
Store only an observed decision that transfers across instances of this task
family. Prefer a demonstrated strategy change or evidenced correction; instance
outputs and surface-specific details are `NO_MEMORY`.""",
        "examples": None,
}


def _family_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for benchmark, entry in TASK_GUIDANCE.items():
        for family in entry["families"]:
            if family in index:
                raise ValueError(
                    f"task family {family!r} is listed by {index[family]!r} and {benchmark!r}"
                )
            index[family] = benchmark
    return index


TASK_FAMILY_TO_BENCHMARK = _family_index()


def supported_mas_types() -> frozenset[str]:
    return frozenset(MAS_PROMPT_MODULES)


def _mas_prompt_module(mas_type: str) -> ModuleType:
    module_name = MAS_PROMPT_MODULES.get(mas_type)
    if module_name is None:
        raise ValueError(f"no MAS guidance for {mas_type!r}")
    return import_module(module_name)


def mas_system_addendum(mas_type: str) -> str:
    """Architecture description of a MAS."""
    return _mas_prompt_module(mas_type).MEMORY_SYSTEM_ADDENDUM


def mas_role_guidance(mas_type: str, role: str, duty: str = "") -> str:
    """Guidance for one agent: first role key contained in its name, else duty, else default."""
    module = _mas_prompt_module(mas_type)
    normalized = str(role).casefold()
    for label, text in module.MEMORY_ROLE_GUIDANCE.items():
        if label in normalized:
            return text
    if duty:
        return module.MEMORY_DUTY_FALLBACK.replace("{duty}", str(duty))
    return module.MEMORY_DEFAULT_ROLE_GUIDANCE


def task_guidance(task_family: str, *, task_specific: bool = True) -> str:
    """Guidance for a task family, with transfer-level examples when present.

    Unknown families, and every family when ``task_specific`` is False, get
    DEFAULT_TASK_GUIDANCE.
    """
    entry = DEFAULT_TASK_GUIDANCE
    if task_specific:
        benchmark = TASK_FAMILY_TO_BENCHMARK.get(str(task_family or "").casefold())
        if benchmark is not None:
            entry = TASK_GUIDANCE[benchmark]
    if entry["examples"] is None:
        return entry["extraction"]
    return f"{entry['extraction']}\n\n### Transfer-Level Examples\n{entry['examples']}"


def guidance_snapshot() -> dict[str, Any]:
    """All guidance text as plain data (for experiment fingerprints)."""
    mas = {}
    for name in sorted(MAS_PROMPT_MODULES):
        module = _mas_prompt_module(name)
        mas[name] = {
            "system_addendum": module.MEMORY_SYSTEM_ADDENDUM,
            "roles": [[label, text] for label, text in module.MEMORY_ROLE_GUIDANCE.items()],
            "duty_fallback": module.MEMORY_DUTY_FALLBACK,
            "default_role": module.MEMORY_DEFAULT_ROLE_GUIDANCE,
        }
    return {"mas": mas, "task": TASK_GUIDANCE, "default_task": DEFAULT_TASK_GUIDANCE}
