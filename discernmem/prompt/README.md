# DiscernMem Prompts

This directory holds the prompts of the DiscernMem method and the JSON schemas
of their structured outputs. Prompt text lives in Markdown; Python selects and
renders the relevant sections (`discernmem/prompting.py`).

## Prompts and Output Schemas

| Prompt | Purpose | Output schema (`schemas/`) |
|---|---|---|
| `memory_extraction.md` | extract one role-owned procedural memory, or `NO_MEMORY` | `memory_extraction.schema.json` |
| `identity_judge.md` | attach a candidate to an existing memory identity or create a new one | `identity_judge.schema.json` |
| `retrieval_selection.md` | select among role-filtered retrieval candidates before injection | `retrieval_selection.schema.json` |
| `deployment_assessment.md` | decide whether a used memory fit the observed deployment context | `formal_use_deployment.schema.json` |
| `formal_use_content_diagnosis.md` | diagnose a content defect after a correctly scoped harmful use | `formal_use_content_diagnosis.schema.json` |
| `memory_repair.md` | propose a minimal, evidence-supported revision of a memory | `memory_repair.schema.json` |
| `meta_summary.md` | update a memory's accumulated `use_when` and `avoid_when` conditions | `meta_summary.schema.json` |

`formal_use_assessment.schema.json` validates the combined result of the two
formal-use stages.

## File Format

Each prompt file has exactly one `## System`, `## User`, and `## Output Contract`
section; runtime selects them by these headings. The `## User` section contains
one `{input_json}` placeholder, which is replaced by a single structured JSON
payload. This README is not loaded into any prompt.

The method-config fingerprint covers these prompts and schemas, so every run
records the exact model instructions it used.
