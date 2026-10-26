# --- common definitions ---
ALLOWED_LIBS = "os, sys, re, math, random, json, itertools, collections, datetime, statistics, functools, hashlib, string, inspect, typing_extensions, numpy, pandas, sympy, statsmodels, sklearn, xgboost, PIL, cv2, imageio, requests, bs4, lxml, matplotlib, seaborn, rich, fake_useragent"
CODE_RULES = """- Do NOT import from typing module (e.g., List, Dict, Tuple).
- Built-in generic types like list[int], dict[str, int] are allowed.
- Use basic Python types only.
- The function must be self-contained and directly executable."""
CODE_ONLY = "Return only one ```python``` block."

# --- actor prompt ---
ACTOR_SYSTEM_PROMPT_TEMPLATE = f"""Write correct Python for the task. {CODE_ONLY} {CODE_RULES} Handle edge cases."""

ACTOR_USER_PROMPT_TEMPLATE = f"""Implement the task using any useful memory.

{{memory_block}}
{{function_name_block}}
# Allowed Libraries
{ALLOWED_LIBS}

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
SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = f"""Produce the best final Python solution from the drafts and critiques. {CODE_ONLY} {CODE_RULES}"""

SUMMARIZER_USER_PROMPT_TEMPLATE = f"""Finalize the implementation.

{{memory_block}}
{{function_name_block}}
# Allowed Libraries
{ALLOWED_LIBS}

# Draft 1
{{feedback_page1}}

# Draft 2
{{feedback_page2}}

# Task
{{task_description}}"""

# --- actor code & critic feedback ---
FEEDBACK_PAGE = """# Actor Code
{actor_output}

# Critic Feedback
{critic_output}"""
