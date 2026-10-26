from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping

import jsonschema

from common.llm_client import LLMClient, estimate_tokens

from .extraction import ExtractionCandidate
from .prompting import load_prompt, load_schema, render_user_prompt
from .storage import DiscernMemStore, _atomic_write_json, safe_component, utc_now
from .schema_compat import guided_decoding_schema, normalize_unique_items


logger = logging.getLogger(__name__)

IDENTITY_SCHEMA = load_schema("identity_judge.schema.json")
_IDENTITY_REASON_MAX_LENGTH = int(
    IDENTITY_SCHEMA["properties"]["reason"].get("maxLength", 240)
)


def _compact_reason(reason: object) -> tuple[object, bool]:
    """Keep audit-only explanations bounded without changing the decision."""
    if not isinstance(reason, str):
        return reason, False
    if len(reason) <= _IDENTITY_REASON_MAX_LENGTH:
        return reason, False
    compact = " ".join(reason.split())
    if len(compact) <= _IDENTITY_REASON_MAX_LENGTH:
        return compact, True
    suffix = "..."
    limit = max(_IDENTITY_REASON_MAX_LENGTH - len(suffix), 1)
    boundary = compact.rfind(" ", 0, limit + 1)
    if boundary <= 0:
        boundary = limit
    return compact[:boundary].rstrip() + suffix, True


def _schema_error(exc: Exception) -> str:
    if isinstance(exc, jsonschema.ValidationError):
        path = ".".join(str(item) for item in exc.absolute_path)
        return f"{path or '<root>'}: {exc.message}"
    return f"{type(exc).__name__}: {exc}"


class DiscernMemIdentityJudge:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: float = 180.0,
        max_tokens: int = 256,
    ):
        if not model_name:
            raise ValueError("DiscernMem identity judge requires a real model name")
        self.store = store
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            retry_count=1,
            timeout=timeout,
            logger_name="discernmem.identity",
            stats_source="discernmem/identity_judge",
        )

    def judge(
        self,
        candidate: ExtractionCandidate,
        comparisons: list[Mapping[str, Any]],
        *,
        artifact_name: str,
    ) -> dict[str, Any]:
        system_prompt = load_prompt("identity_judge.md")
        user_payload = {
            "candidate": candidate.to_dict(),
            "comparisons": [dict(item) for item in comparisons],
        }
        user_prompt = render_user_prompt("identity_judge.md", user_payload)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "discernmem_identity_judge",
                "schema": guided_decoding_schema(IDENTITY_SCHEMA),
                "strict": True,
            },
        }
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        parsed: dict[str, Any] | None = None
        raw = ""
        normalized_reason = False
        fallback_error = ""
        started = time.perf_counter()
        for attempt in range(2):
            attempt_prompt = user_prompt
            if attempt:
                attempt_prompt += (
                    "\n\nThe previous response failed validation: "
                    + _schema_error(
                        last_error or ValueError("unknown validation error")
                    )
                    + ". Return one JSON object with exactly the three required fields. "
                    "Use double quotes, close every string and brace, keep reason under "
                    "160 characters, and add no text outside the JSON object."
                )
            started = time.perf_counter()
            raw = ""
            try:
                raw = self.llm.chat(
                    system_prompt,
                    attempt_prompt,
                    temperature=0.0,
                    response_format=response_format,
                )
                value = normalize_unique_items(json.loads(raw), IDENTITY_SCHEMA)
                if isinstance(value, Mapping):
                    compacted, normalized_reason = _compact_reason(value.get("reason"))
                    if normalized_reason:
                        value = dict(value)
                        value["reason"] = compacted
                jsonschema.Draft202012Validator(IDENTITY_SCHEMA).validate(value)
                if value["same_operation"] != bool(value["target_memory_id"]):
                    raise ValueError(
                        "identity decision and target_memory_id are inconsistent"
                    )
                parsed = dict(value)
            except Exception as exc:
                last_error = exc
                attempt_record = {
                    "attempt": attempt + 1,
                    "user_prompt": attempt_prompt,
                    "raw_output": raw,
                    "error": _schema_error(exc),
                    "latency_seconds": time.perf_counter() - started,
                    "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                }
                attempts.append(attempt_record)
                _atomic_write_json(
                    self.store.root
                    / "audit"
                    / "identity_judges"
                    / (
                        f"{safe_component(artifact_name)}_attempt_"
                        f"{attempt + 1}_error.json"
                    ),
                    {
                        "timestamp": utc_now(),
                        "system_prompt": system_prompt,
                        "schema": IDENTITY_SCHEMA,
                        "model": self.llm.model,
                        **attempt_record,
                    },
                )
                if attempt == 0:
                    continue
                fallback_error = _schema_error(exc)
                parsed = {
                    "same_operation": False,
                    "target_memory_id": None,
                    "reason": (
                        "Identity judgment failed structured output; candidate was not admitted."
                    ),
                    "_judgment_valid": False,
                }
            break

        if parsed is None:
            fallback_error = _schema_error(
                last_error or ValueError("identity judge returned no parsed decision")
            )
            parsed = {
                "same_operation": False,
                "target_memory_id": None,
                "reason": (
                    "Identity judgment failed structured output; candidate was not admitted."
                ),
                "_judgment_valid": False,
            }
        if fallback_error:
            logger.warning(
                "[DiscernMem][IDENTITY][artifact=%s] invalid structured output; "
                "fallback=discard reason=%s",
                artifact_name,
                fallback_error,
            )
        _atomic_write_json(
            self.store.root
            / "audit"
            / "identity_judges"
            / f"{safe_component(artifact_name)}.json",
            {
                "timestamp": utc_now(),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": IDENTITY_SCHEMA,
                "raw_output": raw,
                "parsed_output": parsed,
                "attempt": len(attempts) if fallback_error else len(attempts) + 1,
                "fallback": bool(fallback_error),
                "fallback_error": fallback_error or None,
                "normalized_reason": normalized_reason,
                "previous_attempts": attempts,
                "model": self.llm.model,
                "latency_seconds": time.perf_counter() - started,
                "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                "prompt_tokens_est": estimate_tokens(system_prompt + user_prompt),
            },
        )
        return parsed
