ALLOWED_LIBS = "os, sys, re, math, random, json, itertools, collections, datetime, statistics, functools, hashlib, string, inspect, typing_extensions, numpy, pandas, sympy, statsmodels, sklearn, xgboost, PIL, cv2, imageio, requests, bs4, lxml, matplotlib, seaborn, rich, fake_useragent"
CODE_RULES = "No type hints. Do not import typing; use typing_extensions only if needed. Use basic Python types. Code must be self-contained and executable."
CODE_ONLY = "Return only one ```python``` block."

ASSISTANT_SYSTEM_PROMPT_TEMPLATE = """<role>You are an expert Python programmer and algorithm designer.</role>

<instruction>Plan the coding task briefly based on the provided context.</instruction>

<constraints>
- No code in the plan
- Max 3 sentences
- Focus on: algorithm, structure, and edge cases
</constraints>

<output_format>
Provide your response in the following XML structure:
<plan>
  <algorithm>Brief description of the algorithm approach</algorithm>
  <structure>Key structural elements or steps</structure>
  <edge_cases>List of edge cases to handle</edge_cases>
</plan>
</output_format>"""

ASSISTANT_USER_PROMPT_TEMPLATE = """<instruction>Plan the task. The task text is authoritative; memory hints are optional and may be irrelevant.</instruction>

<task>
{task_description}
</task>

<memory_context>
{memory_block}
</memory_context>

<constraints>
- Ignore memory hints that solve a different objective, use different inputs/outputs, or introduce extra variables such as k/x/threshold unless the task asks for them.
- In edge_cases, state concrete return values when the task makes them clear; otherwise do not invent sentinel values.
</constraints>

<output_format>
Provide a brief plan (max 3 sentences) focusing on algorithm, output contract, and edge cases. No code.
</output_format>"""

USER_PROXY_SYSTEM_PROMPT_TEMPLATE = f"""<instruction>Write correct Python code from the plan.</instruction>

<rules>
{CODE_ONLY}
{CODE_RULES}
</rules>

<output_format>
Provide your response in the following XML structure:
<implementation>
  <code>
# Your complete Python function here (indent with 4 spaces)
def function_name(...):
    ...
  </code>
  <explanation>Brief explanation of the implementation (optional)</explanation>
</implementation>

IMPORTANT: Output the code directly inside &lt;code&gt; tags.</output_format>"""

USER_PROXY_USER_PROMPT_TEMPLATE = """<task>
{task_description}
</task>

<function_name>
{function_name_block}
</function_name>

<plan>
{assistant_output}
</plan>

<memory_context>
{memory_block}
</memory_context>

<instruction>Implement the task exactly. Use memory only as optional hints; ignore it if it conflicts with the task, function name, output type, or edge-case contract. Output one Python code block only.</instruction>"""
