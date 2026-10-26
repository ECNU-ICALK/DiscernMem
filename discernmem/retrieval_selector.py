from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

import jsonschema

from common.llm_client import LLMClient

from .prompting import load_prompt, load_schema, render_user_prompt
from .schema_compat import guided_decoding_schema


@dataclass(frozen=True)
class RetrievalSelection:
    selected_ids: tuple[str, ...]
    status: str
    raw_output: str = ""
    actual_usage: Mapping[str, Any] | None = None


class DiscernMemRetrievalSelector:
    """One short role-local compatibility decision after semantic Top-K."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: float = 180.0,
    ):
        self.system_prompt = load_prompt("retrieval_selection.md")
        self.schema = load_schema("retrieval_selection.schema.json")
        jsonschema.Draft202012Validator.check_schema(self.schema)
        self.llm = LLMClient(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            model=model_name,
            temperature=0.0,
            max_tokens=64,
            timeout=timeout,
            retry_count=0,
            logger_name="discernmem.retrieval_selector",
            stats_source="discernmem/retrieval_selector",
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("retrieval selector output must be a JSON object")
        return value

    def select(
        self,
        *,
        context: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> RetrievalSelection:
        if not candidates:
            return RetrievalSelection((), "no_candidates")
        payload = {
            "context": dict(context),
            "candidates": [
                {
                    "id": str(item["memory_id"]),
                    "memory": str(item["memory"]),
                    "use_when": list(item.get("use_when") or []),
                    "avoid_when": list(item.get("avoid_when") or []),
                }
                for item in candidates
            ],
        }
        user_prompt = render_user_prompt("retrieval_selection.md", payload, indent=None)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "discernmem_retrieval_selection",
                "schema": guided_decoding_schema(self.schema),
                "strict": True,
            },
        }
        raw = ""
        try:
            raw = self.llm.chat(
                self.system_prompt,
                user_prompt,
                temperature=0.0,
                max_tokens=64,
                response_format=response_format,
            )
            parsed = self._parse(raw)
            jsonschema.Draft202012Validator(self.schema).validate(parsed)
            allowed = [str(item["memory_id"]) for item in candidates]
            requested = set(str(value) for value in parsed["selected_ids"])
            decision = str(parsed["decision"])
            if decision == "NONE" and requested:
                raise ValueError("NONE retrieval decision must return no IDs")
            if decision == "SELECT" and not requested:
                raise ValueError(
                    "SELECT retrieval decision must return at least one ID"
                )
            if not requested <= set(allowed):
                raise ValueError("retrieval selector returned an unknown candidate ID")
            # Preserve the LLM ranking; max_injected_memories is applied by
            # the retriever after this ordered selection.
            selected = tuple(
                value for value in parsed["selected_ids"] if str(value) in set(allowed)
            )
            return RetrievalSelection(
                selected,
                "selected" if selected else "none_selected",
                raw,
                dict((self.llm.last_call or {}).get("usage") or {}),
            )
        except Exception:
            return RetrievalSelection(
                (),
                "failed_closed",
                raw,
                dict((self.llm.last_call or {}).get("usage") or {}),
            )
