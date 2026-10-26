"""
OpenAI-compatible LLM and embedding clients.

Used by the memory method for extraction, judging, repair, and retrieval
selection calls, and for embedding computation.
"""
import hashlib
import logging
import threading
import time
from typing import Optional

import numpy as np
import requests

from common.stats import stats

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retry_count: int = 1,
        timeout: float = 120.0,
        max_input_chars: int = 120000,
        logger_name: Optional[str] = None,
        stats_source: str = "llm",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.logger = logging.getLogger(logger_name) if logger_name else logger
        self.stats_source = stats_source
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.timeout = timeout
        self._call_state = threading.local()
        self.last_call = None

    @property
    def last_call(self):
        return getattr(self._call_state, "last_call", None)

    @last_call.setter
    def last_call(self, value):
        self._call_state.last_call = value

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_input_chars: int = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        if max_input_chars is None:
            max_input_chars = self.max_input_chars
        # Truncate oversized prompts to avoid exceeding the vLLM max-model-len.
        total_len = len(system_prompt or "") + len(user_prompt)
        if total_len > max_input_chars:
            sys_len = len(system_prompt or "")
            sys_budget = min(sys_len, max_input_chars // 4)
            usr_budget = max_input_chars - sys_budget
            if system_prompt and sys_len > sys_budget:
                system_prompt = system_prompt[:sys_budget] + "\n...[truncated]"
            if len(user_prompt) > usr_budget:
                user_prompt = user_prompt[:usr_budget] + "\n...[truncated]"
            self.logger.warning(f"Prompt truncated to ~{max_input_chars} chars to fit context window")

        self.last_call = None
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1 + self.retry_count):
            try:
                t0 = time.time()
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                elapsed = time.time() - t0
                data = response.json()
                choice = data["choices"][0]
                raw_content = choice["message"]["content"]
                content = self._strip_think_tags(raw_content)
                usage = data.get("usage", {})
                self.last_call = {
                    "url": url,
                    "request_payload": payload,
                    "raw_content": raw_content,
                    "returned_content": content,
                    "usage": usage,
                    "finish_reason": str(choice.get("finish_reason") or ""),
                    "latency_seconds": elapsed,
                }
                # Record stats
                stats.record(
                    source=self.stats_source,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    elapsed=elapsed,
                )
                return content.strip()
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count:
                    time.sleep(1)
                else:
                    self.logger.error(f"LLM call failed after {1 + self.retry_count} attempts")
                    raise

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from Qwen3 responses."""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class EmbeddingClient:
    """OpenAI-compatible Embedding API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        device: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.timeout = timeout
        self.device = device
        self._cache: dict[str, list[float]] = {}  # text_hash -> embedding
        self._local_model = None

    def _uses_local_sentence_transformer(self) -> bool:
        model_name = (self.model or "").strip()
        return model_name.startswith("sentence-transformers/")

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.device} if self.device else {}
            self._local_model = SentenceTransformer(self.model, **kwargs)
        return self._local_model

    def embed(self, text: str) -> list[float]:
        """Compute embedding for a single text (with cache)."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = self.embed_batch([text])[0]
        self._cache[key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of texts."""
        if not texts:
            return []

        if self._uses_local_sentence_transformer():
            model = self._get_local_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.tolist()

        payload = {
            "model": self.model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise


def estimate_tokens(text: str) -> int:
    """Estimate token count using the BPE heuristic: ~4 chars per token."""
    return max(1, len(text) // 4)
