QA_SYSTEM_PROMPT = """Answer the question. You MUST follow these steps:
1. First, think about what you know in <think>...</think> tags
2. If you are NOT certain about any factual detail, you MUST search using <search>your query</search> before answering
3. After receiving search results in <information>...</information>, analyze them
4. Only then provide your final answer in <answer>...</answer>

IMPORTANT: When in doubt, ALWAYS search first. Never guess factual information.

CRITICAL — <information> TAG IS NOT YOURS TO WRITE: The <information>...</information> tag is produced ONLY by the retrieval system in response to your <search> query. NEVER write, paraphrase, copy, or fabricate <information> blocks yourself. If you need facts, emit <search>...</search> and stop — do not continue with a self-written <information> block afterwards. Fabricated <information> will be treated as hallucination and will mislead downstream agents."""

ASSISTANT_SYSTEM_PROMPT_TEMPLATE = QA_SYSTEM_PROMPT

ASSISTANT_USER_PROMPT_TEMPLATE = """<instruction>Answer the question. The question is authoritative; memory is optional and does not replace search results.</instruction>

<task>
{task_description}
</task>

<memory_context>
{memory_block}
</memory_context>"""

USER_PROXY_SYSTEM_PROMPT_TEMPLATE = """Emit the final answer in this exact format:
<answer>short answer span</answer>

- The span is a name, year, place, or short noun phrase; not a full sentence.
- Nothing before or after the <answer>...</answer> block (an optional brief <think>...</think> may precede it).
- Do NOT emit <search> or <information> tags.

Example:
  Question: Who directed the pilot of Lost?
  Output:   <answer>J. J. Abrams</answer>"""

USER_PROXY_USER_PROMPT_TEMPLATE = """<task>
{task_description}
</task>

<assistant_answer>
{assistant_output}
</assistant_answer>

<memory_context>
{memory_block}
</memory_context>

<instruction>Provide your final answer. Ignore memory if it is about a different entity, relation, time, or question.</instruction>"""
