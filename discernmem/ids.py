from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    clean_prefix = "".join(char for char in prefix.lower() if char.isalnum() or char == "_")
    if not clean_prefix:
        raise ValueError("stable id prefix cannot be empty")
    return f"{clean_prefix}_{content_hash(payload)[:24]}"


def event_id(event_type: str, task_id: str, inputs: Mapping[str, Any]) -> str:
    return stable_id(
        "evt",
        {"event_type": str(event_type), "task_id": str(task_id), "inputs": dict(inputs)},
    )


def replay_id(
    *,
    task_id: str,
    seed: int,
    role: str,
    condition: str,
    config_fingerprint: str,
    memory_id: str = "",
    version: int | None = None,
    candidate_id: str = "",
) -> str:
    if bool(memory_id) == bool(candidate_id) and condition != "NO_MEMORY":
        raise ValueError("a non-baseline replay targets exactly one version or candidate")
    return stable_id(
        "rpl",
        {
            "task_id": str(task_id),
            "seed": int(seed),
            "role": str(role),
            "condition": str(condition),
            "config_fingerprint": str(config_fingerprint),
            "memory_id": str(memory_id),
            "version": version,
            "candidate_id": str(candidate_id),
        },
    )
