# Memory Extraction

## System

### Objective

Extract at most one transferable behavioral rule demonstrated by the target
role. Decide `STORE` or `NO_MEMORY` before drafting the memory.

### Storage Decision

Choose `STORE` only when all three conditions hold:

1. The target role made an identifiable decision, check, or revision.
2. An observed state, result, or feedback item links that choice to progress or
   prevention of an evidenced error.
3. The rule remains useful after removing source-specific entities, values,
   answers, code, and wording.

A failed trajectory qualifies when the target role demonstrates a supported
correction, a valid alternative selection, or an evidenced preventive decision.
Do not infer an unseen solution or recovery plan. A bare score, unexplained
failure, routine proposal, forwarding/agreement, formatting step, source answer,
complete solution, algorithm, formula, data structure, complexity claim, code, or
query template is `NO_MEMORY`. Do not invent hidden facts, requirements,
effects, or repairs.

### Evidence Attribution

- Attribute a decision only from `role_messages`.
- Treat `incoming_context` as received information.
- Use `outgoing_context` and `downstream_outcomes` only as evidence of what
  followed the target decision; never reassign another role's action or solution.
- Do not use a later outcome to invent reasoning unavailable when the role acted.

### Memory Form

For `STORE`:

- Prefer the smallest causal decision that explains progress, correction, or
  prevention of a repeated/no-op transition.
- Write `memory` as one concise imperative behavioral rule, not a full procedure.
  Begin with an action verb; keep When, If, After, and Unless conditions out of it.
- Generalize source names, values, answers, function names, and constants into
  relational descriptions in `memory`, `use_when`, and `avoid_when`. Do not
  copy task IDs, object/location names, literal answers, or constants into any
  field. Do not store implementation choices or a complete answer, program,
  plan, or action sequence.
- Write `use_when` as observable requirements for the action.
- Write `avoid_when` only as observable exceptions supported by the trajectory;
  otherwise return an empty list. Keep the two condition sets mutually
  exclusive.

## User

Input JSON:
{input_json}

## Output Contract

Return one JSON object matching the response schema.

For `NO_MEMORY`, return:
`{"decision":"NO_MEMORY","memory":"NO_MEMORY","use_when":[],"avoid_when":[]}`
