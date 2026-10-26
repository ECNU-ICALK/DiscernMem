from __future__ import annotations

import re
from typing import Any, Mapping

from .schema import MemoryVersion
from .storage import DiscernMemStore


_FAMILY_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:")


def normalize_task_family(value: object) -> str:
    return re.sub(r"[^a-z0-9_-]+", "", str(value or "").strip().lower())


def infer_task_family(
    task_description: object,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    explicit = normalize_task_family((metadata or {}).get("task_family"))
    if explicit:
        return explicit
    match = _FAMILY_PREFIX_RE.match(str(task_description or ""))
    return normalize_task_family(match.group(1)) if match else ""


def memory_source_task_family(store: DiscernMemStore, version: MemoryVersion) -> str:
    task_ids = [version.base.source_task_id]
    identity = store.get_identity(version.memory_id)
    if identity is not None:
        task_ids.append(identity.source_task_id)
    for task_id in dict.fromkeys(task_ids):
        if not task_id:
            continue
        evidence = store.get_task(task_id)
        if evidence is None:
            continue
        family = normalize_task_family(evidence.task_family) or infer_task_family(
            evidence.task_description
        )
        if family:
            return family
    return ""


def task_families_compatible(current: str, source: str) -> bool:
    current = normalize_task_family(current)
    source = normalize_task_family(source)
    return not current or not source or current == source
