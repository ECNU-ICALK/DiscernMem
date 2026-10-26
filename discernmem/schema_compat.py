from __future__ import annotations

import json
from typing import Any, Mapping


_GUIDED_DECODING_UNSUPPORTED = {"uniqueItems", "maxLength"}


def guided_decoding_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a provider-compatible guide without weakening local validation.

    Some vLLM guided-decoding backends reject valid Draft 2020-12 keywords.
    The authoritative schema remains unchanged and is always applied locally
    after generation; this copy only constrains server-side decoding.
    """

    def clean(value):
        if isinstance(value, Mapping):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in _GUIDED_DECODING_UNSUPPORTED
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(schema)


def normalize_unique_items(value: Any, schema: Mapping[str, Any]) -> Any:
    """Stably deduplicate arrays explicitly marked as unique by the schema."""

    def matching_branch(current: Any, branches: Any) -> Mapping[str, Any] | None:
        if not isinstance(branches, list):
            return None
        for branch in branches:
            if not isinstance(branch, Mapping):
                continue
            branch_type = branch.get("type")
            if branch_type == "object" and isinstance(current, Mapping):
                return branch
            if branch_type == "array" and isinstance(current, list):
                return branch
            if branch_type == "null" and current is None:
                return branch
            if branch_type == "string" and isinstance(current, str):
                return branch
            if branch_type == "boolean" and isinstance(current, bool):
                return branch
            if (
                branch_type == "integer"
                and isinstance(current, int)
                and not isinstance(current, bool)
            ):
                return branch
            if (
                branch_type == "number"
                and isinstance(current, (int, float))
                and not isinstance(current, bool)
            ):
                return branch
        return None

    for combinator in ("oneOf", "anyOf"):
        branch = matching_branch(value, schema.get(combinator))
        if branch is not None:
            value = normalize_unique_items(value, branch)

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for branch in all_of:
            if isinstance(branch, Mapping):
                value = normalize_unique_items(value, branch)

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        additional = schema.get("additionalProperties")
        return {
            key: normalize_unique_items(
                item,
                (
                    properties.get(key)
                    if isinstance(properties.get(key), Mapping)
                    else additional if isinstance(additional, Mapping) else {}
                ),
            )
            for key, item in value.items()
        }

    if not isinstance(value, list):
        return value

    item_schema = schema.get("items")
    item_schema = item_schema if isinstance(item_schema, Mapping) else {}
    normalized = [normalize_unique_items(item, item_schema) for item in value]
    if schema.get("uniqueItems") is not True:
        return normalized

    result: list[Any] = []
    seen: set[str] = set()
    for item in normalized:
        canonical = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(item)
    return result
