from __future__ import annotations

from typing import Iterable


def build_role_memory_query(
    base_query: str,
    role: str,
    duty: str,
    visible_messages: Iterable[str] | None = None,
) -> str:
    parts = [
        str(base_query or "").strip(),
        f"Current MAS role: {str(role or '').strip()}",
        f"Role duty: {str(duty or '').strip()}",
    ]
    messages = [str(value or "").strip() for value in (visible_messages or [])]
    messages = [value for value in messages if value]
    if messages:
        parts.append("Visible upstream messages:\n" + "\n".join(messages))
    return "\n\n".join(part for part in parts if part)
