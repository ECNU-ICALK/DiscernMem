from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping
import uuid

import jsonschema

from common.llm_client import LLMClient

from .schema_compat import guided_decoding_schema, normalize_unique_items
from .prompting import load_prompt, load_schema, render_user_prompt
from .storage import DiscernMemStore, _atomic_write_json, utc_now


META_SUMMARY_SCHEMA = load_schema("meta_summary.schema.json")


class MetaSummarizerError(RuntimeError):
    """Raised when the model cannot produce a valid meta summary."""


def _parse_meta_summary(raw: str) -> dict[str, Any]:
    """Parse only safe, common formatting errors around a JSON object."""
    text = str(raw or "").strip()
    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE
        ).strip()

    candidates = [text]
    start = text.find("{")
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            _, end = decoder.raw_decode(text[start:])
            candidates.append(text[start : start + end])
        except json.JSONDecodeError:
            candidates.append(
                text[start : text.rfind("}") + 1] if "}" in text else text[start:]
            )

    last_error: Exception | None = None
    for candidate in dict.fromkeys(candidates):
        for value in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, dict):
                    raise ValueError("meta summary must be a JSON object")
                return parsed
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
    raise MetaSummarizerError(f"invalid meta summary JSON: {last_error}")


class DiscernMemMetaSummarizer:
    def __init__(
        self,
        *,
        store: DiscernMemStore,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: float = 180.0,
    ):
        self.store = store
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=900,
            retry_count=1,
            timeout=timeout,
            logger_name="discernmem.meta",
            stats_source="discernmem/meta_summarizer",
        )

    def summarize(self, *, positive_states, qualifying_boundaries) -> dict[str, Any]:
        system = load_prompt("meta_summary.md")
        payload = {
            "positive_states": positive_states,
            "qualifying_boundaries": qualifying_boundaries,
        }
        user = render_user_prompt("meta_summary.md", payload)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "discernmem_meta_summary",
                "schema": guided_decoding_schema(META_SUMMARY_SCHEMA),
                "strict": True,
            },
        }
        last_error: Exception | None = None
        for attempt in range(2):
            attempt_user = user
            if attempt:
                attempt_user += (
                    "\n\nPrevious output was invalid. Return exactly one compact JSON "
                    "object only: double-quoted keys and strings, no markdown, "
                    "commentary, or trailing commas."
                )
            started = time.perf_counter()
            raw = ""
            try:
                raw = self.llm.chat(
                    system,
                    attempt_user,
                    temperature=0.0,
                    response_format=response_format,
                )
                parsed = normalize_unique_items(
                    _parse_meta_summary(raw), META_SUMMARY_SCHEMA
                )
                jsonschema.Draft202012Validator(META_SUMMARY_SCHEMA).validate(parsed)
            except Exception as exc:
                last_error = exc
                _atomic_write_json(
                    self.store.root
                    / "audit"
                    / "meta_summaries"
                    / f"meta_{uuid.uuid4().hex}_attempt_{attempt + 1}_error.json",
                    {
                        "timestamp": utc_now(),
                        "attempt": attempt + 1,
                        "system_prompt": system,
                        "user_prompt": attempt_user,
                        "schema": META_SUMMARY_SCHEMA,
                        "input": payload,
                        "raw_output": raw,
                        "error": f"{type(exc).__name__}: {exc}",
                        "model": self.llm.model,
                        "latency_seconds": time.perf_counter() - started,
                        "actual_usage": dict(
                            (self.llm.last_call or {}).get("usage") or {}
                        ),
                    },
                )
                continue
            _atomic_write_json(
                self.store.root
                / "audit"
                / "meta_summaries"
                / f"meta_{uuid.uuid4().hex}.json",
                {
                    "timestamp": utc_now(),
                    "attempt": attempt + 1,
                    "system_prompt": system,
                    "user_prompt": attempt_user,
                    "schema": META_SUMMARY_SCHEMA,
                    "input": payload,
                    "raw_output": raw,
                    "parsed_output": parsed,
                    "model": self.llm.model,
                    "latency_seconds": time.perf_counter() - started,
                    "actual_usage": dict((self.llm.last_call or {}).get("usage") or {}),
                },
            )
            return parsed
        raise MetaSummarizerError(
            f"invalid meta summary after 2 attempts: {last_error}"
        )
