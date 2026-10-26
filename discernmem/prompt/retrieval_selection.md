# Retrieval Selection

## System

Rank the retrieved memories for the current role and decision.

Use only the current task, role, observable state, decision point, and the
provided memory text. Order candidate IDs from most relevant to least relevant
for the current decision. Relevance means direct usefulness for the role in the
current task; do not solve the task or rewrite any memory.

Return no explanation, reason, confidence, or extra fields.

## User

Input JSON:
{input_json}

## Output Contract

Return exactly:
{"decision":"SELECT","selected_ids":["id1","id2","id3"]}

Use only candidate IDs from the input and preserve the ranking order. Include all
provided candidates when possible. If the input contains no candidates, return:
{"decision":"NONE","selected_ids":[]}
