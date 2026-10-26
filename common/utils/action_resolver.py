"""
Utilities for resolving intermediate MAS agent actions.

Intermediate agents in a MAS generate() pipeline may emit
<search>query</search>, while only the final agent output is passed to
env.step(). This module runs those searches immediately and appends the
results as <information>...</information> so downstream agents can use them.

It also provides invoke_with_iterative_search, which lets a single agent
perform repeated <search> -> <information> -> continued reasoning -> <answer>
inside one generate call instead of rerunning the outer turn loop.
"""
import re
from typing import Callable, List, Optional


_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>.*?</search>", re.DOTALL | re.IGNORECASE)


def extract_search_query(text: str) -> Optional[str]:
    """Extract the first <search>...</search> query from text."""
    match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
    if match:
        return match.group(1).strip().split("\n", 1)[0].strip()
    return None


def extract_search_tool_calls(text: str, fallback_result: str = "") -> list[dict]:
    """Return every tagged search and its following information as structured trace data."""
    content = text or ""
    searches = list(re.finditer(r"<search>(.*?)</search>", content, re.DOTALL | re.IGNORECASE))
    calls = []
    for index, match in enumerate(searches):
        query = match.group(1).strip().split("\n", 1)[0].strip()
        segment_end = searches[index + 1].start() if index + 1 < len(searches) else len(content)
        segment = content[match.end():segment_end]
        info_match = re.search(r"<information>(.*?)</information>", segment, re.DOTALL | re.IGNORECASE)
        result = info_match.group(1).strip() if info_match else str(fallback_result or "").strip()
        calls.append({"tool": "search", "query": query, "result": result})
    return calls


def strip_search_tags(text: str) -> str:
    """Remove all <search>...</search> blocks.

    This is a guardrail for final agents such as summarizers or user proxies:
    if they emit an invalid search, env.step() still receives an answer-style
    response instead of taking the search path.
    """
    return _SEARCH_RE.sub("", text or "").strip()


def resolve_intermediate_actions(
    responses: List[str],
    resolver: Callable[[List[str]], List[str]],
) -> List[str]:
    """Resolve <search> blocks in intermediate agent outputs.

    Args:
        responses: Raw intermediate-agent outputs for the batch.
        resolver: Search function with signature
            (queries: list[str]) -> list[str], compatible with
            Retriever.batch_search and PopQARetriever.batch_search.

    Returns:
        Outputs with <information>...</information> appended for responses
        that contained a search query. Responses without search are returned
        unchanged.
    """
    queries_with_indices: List[tuple[int, str]] = []
    for i, resp in enumerate(responses):
        query = extract_search_query(resp)
        if query:
            queries_with_indices.append((i, query))

    if not queries_with_indices:
        return responses

    indices, queries = zip(*queries_with_indices)
    try:
        results = resolver(list(queries))
    except Exception:
        results = ["Cannot find corresponding pages."] * len(queries)

    result_map = dict(zip(indices, results))

    resolved = []
    for i, resp in enumerate(responses):
        if i in result_map:
            resolved.append(f"{resp}\n<information>{result_map[i]}</information>")
        else:
            resolved.append(resp)
    return resolved


def invoke_with_iterative_search(
    agent,
    system_inputs: dict,
    user_inputs: dict,
    generation_config: dict,
    resolver: Optional[Callable[[List[str]], List[str]]] = None,
    max_search_iters: int = 3,
    task_description_key: str = "task_description",
):
    """Invoke an agent iteratively while resolving search actions.

    Each round resolves <search>, injects <information> back into the task
    description, and continues until the agent emits <answer> or reaches
    max_search_iters.

    The returned Message `.response` is the concatenated iterative trace:
    `"<search>Q1</search>\\n<information>I1</information>\\n<search>Q2</search>\\n"
    `...`\\n<answer>X</answer>"`. Downstream compact_qa_exchange splits this
    tagged trace.

    Args:
        agent: LLMAgent-like instance with `agent.invoke(system, user, cfg)`.
        system_inputs: Batched system-side fields.
        user_inputs: Batched user-side fields.
        generation_config: Generation config dictionary.
        resolver: Search resolver. If None, the function degrades to one
            plain invoke call.
        max_search_iters: Maximum number of search actions allowed per agent
            inside one generate call.
        task_description_key: Key in user_inputs carrying the task text or
            conversation context; iterative search appends search traces there.

    Returns:
        list[Message] with the same length as `agent.invoke`; each response is
        the complete iterative trace for that batch item.
    """
    batch_size = len(user_inputs[task_description_key])
    cur_task_desc: list[str] = list(user_inputs[task_description_key])
    cumulative: list[str] = ["" for _ in range(batch_size)]
    final_msgs: list = [None] * batch_size
    done: list[bool] = [False] * batch_size

    for iter_idx in range(max_search_iters + 1):
        active = [i for i in range(batch_size) if not done[i]]
        if not active:
            break

        sub_sys = {k: [v[i] for i in active] for k, v in system_inputs.items()}
        sub_usr = {k: [v[i] for i in active] for k, v in user_inputs.items()}
        sub_usr[task_description_key] = [cur_task_desc[i] for i in active]

        msgs = agent.invoke(sub_sys, sub_usr, generation_config)

        for li, gi in enumerate(active):
            msg = msgs[li]
            resp = msg.response or ""
            has_answer = bool(_ANSWER_RE.search(resp))
            query = extract_search_query(resp) if resolver is not None else None

            if has_answer or query is None or iter_idx >= max_search_iters:
                # Finalize: concatenate prior search history and mark done.
                if cumulative[gi]:
                    msg.response = cumulative[gi] + "\n" + resp
                final_msgs[gi] = msg
                done[gi] = True
                continue

            # Continue searching: resolve this query and append information.
            try:
                info = resolver([query])[0]
            except Exception:
                info = "Cannot find corresponding pages."
            step_text = f"{resp}\n<information>{info}</information>"
            cumulative[gi] = (cumulative[gi] + "\n" + step_text).strip() if cumulative[gi] else step_text
            cur_task_desc[gi] = (
                cur_task_desc[gi]
                + f"\n\nPrevious response:\n{resp}\nSearch result:\n<information>{info}</information>\nPlease continue."
            )
            final_msgs[gi] = msg

    for gi in range(batch_size):
        if final_msgs[gi] is None:
            raise RuntimeError(f"invoke_with_iterative_search: batch item {gi} produced no message")

    return final_msgs
