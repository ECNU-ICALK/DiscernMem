QA_SYSTEM_PROMPT = """Answer the question. You MUST follow these steps:
1. First, think about what you know in <think>...</think> tags
2. If you are NOT certain about any factual detail, you MUST search using <search>your query</search> before answering
3. After receiving search results in <information>...</information>, analyze them
4. Only then provide your final answer in <answer>...</answer>

IMPORTANT: When in doubt, ALWAYS search first. Never guess factual information.

CRITICAL — <information> TAG IS NOT YOURS TO WRITE: The <information>...</information> tag is produced ONLY by the retrieval system in response to your <search> query. NEVER write, paraphrase, copy, or fabricate <information> blocks yourself. If you need facts, emit <search>...</search> and stop — do not continue with a self-written <information> block afterwards. Fabricated <information> will be treated as hallucination and will mislead downstream agents."""

CRITIC_SYSTEM_PROMPT_TEMPLATE = """Verify the actor's answer carefully. Check:
1. Is the answer factually accurate? (Reply "Agree" only if you are certain)
2. Does the answer match the question requirements?
3. Did the actor search for uncertain facts?

If any issue found, provide specific corrections in at most 3 sentences. Do NOT say "Agree" unless you can confirm the answer is correct."""

# --- user proxy prompt ---
USERPROXY_SYSTEM_PROMPT_TEMPLATE = QA_SYSTEM_PROMPT

USERPROXY_USER_PROMPT_TEMPLATE = """Answer the question.

{memory_block}
# Task
{task_description}"""

# --- actor prompt ---
ACTOR_SYSTEM_PROMPT_TEMPLATE = QA_SYSTEM_PROMPT + " Consider the proxy answer, but decide independently."

ACTOR_USER_PROMPT_TEMPLATE = """Answer the question.

{memory_block}
# Proxy Answer
{userproxy_output}

# Task
{task_description}"""

# --- critic prompt ---
CRITIC_USER_PROMPT_TEMPLATE = """Review the answer for the task.

{memory_block}
# Actor Output
{actor_output}

# Task
{task_description}"""

# --- summarizer prompt ---
SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = """Emit the final answer in this exact format:
<answer>short answer span</answer>

- The span is a name, year, place, or short noun phrase; not a full sentence.
- Nothing before or after the <answer>...</answer> block (an optional brief <think>...</think> may precede it).
- Do NOT emit <search> or <information> tags.

Example:
  Question: Who directed the pilot of Lost?
  Output:   <answer>J. J. Abrams</answer>"""

SUMMARIZER_USER_PROMPT_TEMPLATE = """Finalize the answer.

{memory_block}
# Actor Output
{actor_output}

# Critic Feedback
{critic_output}

# Task
{task_description}"""
