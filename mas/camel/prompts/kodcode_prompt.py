# --- common definitions ---
ALLOWED_LIBS = "os, sys, re, math, random, json, itertools, collections, datetime, statistics, functools, hashlib, string, inspect, typing_extensions, numpy, pandas, sympy, statsmodels, sklearn, xgboost, PIL, cv2, imageio, requests, bs4, lxml, matplotlib, seaborn, rich, fake_useragent"
CODE_RULES = "No type hints. Do not import typing; use typing_extensions only if needed. Use basic Python types. Code must be self-contained and executable."
CODE_ONLY = "Return only one ```python``` block."

# --- user proxy prompt ---
USERPROXY_SYSTEM_PROMPT_TEMPLATE = "Plan the coding task briefly. No code. Max 3 sentences. Focus on algorithm, structure, and edge cases."

USERPROXY_USER_PROMPT_TEMPLATE = """Plan the task.

{memory_block}
# Task
{task_description}"""

# --- actor prompt ---
ACTOR_SYSTEM_PROMPT_TEMPLATE = f"""Write correct Python from the strategy. {CODE_ONLY} {CODE_RULES}"""

ACTOR_USER_PROMPT_TEMPLATE = f"""Implement the task from the strategy.

{{memory_block}}
{{function_name_block}}
# Strategy
{{userproxy_output}}

# Task
{{task_description}}"""

# --- critic prompt ---
CRITIC_SYSTEM_PROMPT_TEMPLATE = 'Review the code against the SPECIFIC task description below. Reply only "Agree" if the code correctly solves the described task; otherwise give at most 3 short sentences about what is wrong. Do NOT invent requirements not stated in the task.'

CRITIC_USER_PROMPT_TEMPLATE = """Review the implementation for the task.

{memory_block}
# Actor Output
{actor_output}

# Task
{task_description}"""

# --- summarizer prompt ---
SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = f"""Produce the best final Python solution from the draft and critique. {CODE_ONLY} {CODE_RULES}"""

SUMMARIZER_USER_PROMPT_TEMPLATE = f"""Finalize the implementation.

{{memory_block}}
{{function_name_block}}
# Actor Code
{{actor_output}}

# Critic Feedback
{{critic_output}}

# Task
{{task_description}}"""
