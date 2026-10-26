"""Helpers for keeping factual-QA prompts compact and evidence-focused."""

from __future__ import annotations

import re


FACTUAL_QA_DOMAINS = {"triviaqa", "popqa"}

_DOC_SPLIT_RE = re.compile(r"(?=Doc\s+\d+\(Title:)")
_TITLE_RE = re.compile(r"Doc\s+(\d+)\(Title:\s*(.*?)\)\s*(.*)", re.DOTALL)
_TAG_RE = re.compile(r"<(search|information|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def is_factual_qa_domain(task_domain: str | None) -> bool:
    return (task_domain or "").strip().lower() in FACTUAL_QA_DOMAINS


def truncate_text(text: str, max_chars: int) -> str:
    cleaned = normalize_whitespace(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return (truncated or cleaned[:max_chars]).rstrip() + "..."


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact_search_result(
    text: str,
    *,
    max_total_chars: int,
    max_doc_chars: int,
    title_chars: int,
    doc_slack_chars: int,
    remaining_floor_chars: int,
) -> str:
    """Pack retrieved passages under a prompt budget without re-summarizing them."""
    normalized = str(text or "").strip()
    if not normalized:
        return ""

    docs = [chunk.strip() for chunk in _DOC_SPLIT_RE.split(normalized) if chunk.strip()]
    if not docs:
        return truncate_text(normalized, max_total_chars)

    parsed_docs: list[dict[str, str]] = []
    for raw_doc in docs:
        match = _TITLE_RE.match(raw_doc)
        if match:
            _, title, body = match.groups()
            parsed_docs.append({
                "title": title,
                "body": normalize_whitespace(body),
            })
        else:
            parsed_docs.append({
                "title": "",
                "body": normalize_whitespace(raw_doc),
            })

    compact_docs = _pack_docs(
        parsed_docs,
        max_total_chars=max_total_chars,
        max_doc_chars=max_doc_chars,
        title_chars=title_chars,
        doc_slack_chars=doc_slack_chars,
        remaining_floor_chars=remaining_floor_chars,
    )
    if not compact_docs:
        return truncate_text(normalized, max_total_chars)
    return compact_docs


def compact_retrieval_result(
    retrieval_result: list[dict],
    *,
    max_total_chars: int,
    max_doc_chars: int,
    title_chars: int,
    doc_slack_chars: int,
    remaining_floor_chars: int,
    max_chunks_per_source: int,
) -> str:
    """Pack ranked retrieval results while limiting same-source domination."""
    parsed_docs: list[dict[str, str]] = []
    seen_chunks: set[tuple[str, int | None, str]] = set()
    source_counts: dict[str, int] = {}

    for item in retrieval_result or []:
        document = item.get("document") or {}
        contents = str(document.get("contents") or "")
        title, body = _split_document_contents(contents, fallback_title=document.get("title") or "Untitled")
        body = normalize_whitespace(body)
        if not body:
            continue

        source_key = str(document.get("source_id") or title or "unknown")
        if source_counts.get(source_key, 0) >= max(1, max_chunks_per_source):
            continue

        chunk_index = document.get("chunk_index")
        dedupe_key = (source_key, chunk_index, body)
        if dedupe_key in seen_chunks:
            continue
        seen_chunks.add(dedupe_key)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

        chunk_count = document.get("chunk_count")
        passage_prefix = ""
        if isinstance(chunk_index, int) and isinstance(chunk_count, int) and chunk_count > 1:
            passage_prefix = f"[Passage {chunk_index + 1}/{chunk_count}] "

        parsed_docs.append({
            "title": title,
            "body": f"{passage_prefix}{body}".strip(),
        })

    if not parsed_docs:
        return ""

    return _pack_docs(
        parsed_docs,
        max_total_chars=max_total_chars,
        max_doc_chars=max_doc_chars,
        title_chars=title_chars,
        doc_slack_chars=doc_slack_chars,
        remaining_floor_chars=remaining_floor_chars,
    )


def compact_qa_exchange(
    text: str,
    *,
    max_total_chars: int,
    max_info_chars: int,
    max_doc_chars: int,
    query_chars: int,
    answer_chars: int,
    min_info_budget_chars: int,
    max_info_blocks: int,
    title_chars: int,
    doc_slack_chars: int,
    remaining_floor_chars: int,
) -> str:
    """Keep only compact QA evidence that downstream reviewers actually need."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    stripped = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    stripped = stripped.strip()
    tags = _TAG_RE.findall(stripped)
    if not tags:
        return truncate_text(stripped, max_total_chars)

    searches: list[str] = []
    infos: list[str] = []
    answers: list[str] = []

    for tag, content in tags:
        tag_lower = tag.lower()
        if tag_lower == "search":
            query = truncate_text(content.splitlines()[0], query_chars)
            if query and query not in searches:
                searches.append(query)
        elif tag_lower == "information":
            compact_info = compact_search_result(
                content,
                max_total_chars=max_info_chars,
                max_doc_chars=max_doc_chars,
                title_chars=title_chars,
                doc_slack_chars=doc_slack_chars,
                remaining_floor_chars=remaining_floor_chars,
            )
            if compact_info:
                infos.append(compact_info)
        elif tag_lower == "answer":
            answer = truncate_text(content, answer_chars)
            if answer:
                answers.append(answer)

    lines: list[str] = []
    if searches:
        lines.append(f"<search>{searches[0]}</search>")
    if infos:
        info_budget = max(min_info_budget_chars, max_info_chars // max(1, min(len(infos), max_info_blocks)))
        for info in infos[:max_info_blocks]:
            lines.append(f"<information>{truncate_text(info, info_budget)}</information>")
    if answers:
        lines.append(f"<answer>{answers[-1]}</answer>")

    if not lines:
        return truncate_text(stripped, max_total_chars)
    return truncate_text("\n".join(lines), max_total_chars)


def _split_document_contents(contents: str, fallback_title: str = "Untitled") -> tuple[str, str]:
    parts = str(contents or "").split("\n", 1)
    if len(parts) == 2:
        title, body = parts
        return (title or fallback_title).strip(), body
    return fallback_title.strip(), contents


def _render_full_doc(doc: dict[str, str], index: int, title_chars: int) -> str:
    title = truncate_text(doc.get("title", ""), title_chars)
    body = normalize_whitespace(doc.get("body", ""))
    if title:
        return f"Doc {index}(Title: {title}) {body}".strip()
    return body


def _render_budgeted_doc(
    doc: dict[str, str],
    index: int,
    *,
    title_chars: int,
    max_doc_chars: int,
    doc_slack_chars: int,
    remaining: int,
) -> str:
    title = truncate_text(doc.get("title", ""), title_chars)
    body = normalize_whitespace(doc.get("body", ""))
    header = f"Doc {index}(Title: {title})" if title else ""

    doc_budget = min(max_doc_chars + doc_slack_chars, remaining)
    if doc_budget <= 0:
        return ""
    if not header:
        return truncate_text(body, doc_budget)
    if doc_budget <= len(header) + 1:
        return truncate_text(header, doc_budget)

    body_budget = max(0, doc_budget - len(header) - 1)
    compact_body = truncate_text(body, body_budget) if body_budget else ""
    return f"{header} {compact_body}".strip()


def _pack_docs(
    docs: list[dict[str, str]],
    *,
    max_total_chars: int,
    max_doc_chars: int,
    title_chars: int,
    doc_slack_chars: int,
    remaining_floor_chars: int,
) -> str:
    full_docs = [_render_full_doc(doc, index + 1, title_chars) for index, doc in enumerate(docs)]
    full_docs = [doc for doc in full_docs if doc]
    if not full_docs:
        return ""

    full_render = "\n".join(full_docs)
    if len(full_render) <= max_total_chars:
        return full_render

    packed_docs: list[str] = []
    remaining = max_total_chars
    for index, doc in enumerate(docs, start=1):
        if remaining <= 0:
            break
        candidate = _render_budgeted_doc(
            doc,
            index,
            title_chars=title_chars,
            max_doc_chars=max_doc_chars,
            doc_slack_chars=doc_slack_chars,
            remaining=remaining,
        )
        if not candidate:
            continue
        separator = 1 if packed_docs else 0
        if len(candidate) + separator > remaining:
            break
        packed_docs.append(candidate)
        remaining -= len(candidate) + separator
        if remaining <= remaining_floor_chars:
            break

    return "\n".join(packed_docs)
