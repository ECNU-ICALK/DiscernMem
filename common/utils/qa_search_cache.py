"""Thread-safe cache for deterministic QA retrieval results."""

from __future__ import annotations

import threading
from typing import Callable, Hashable


class QASearchCache:
    """Share successful exact-query results across concurrent replay runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[Hashable, str] = {}
        self._inflight: dict[Hashable, threading.Event] = {}
        self._hits = 0
        self._misses = 0

    def get_or_fetch(self, key: Hashable, fetch: Callable[[], str]) -> str:
        """Fetch once for concurrent callers; failed requests are not cached."""
        while True:
            with self._lock:
                if key in self._values:
                    self._hits += 1
                    return self._values[key]
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    self._misses += 1
                    owner = True
                else:
                    owner = False

            if owner:
                try:
                    value = fetch()
                except BaseException:
                    with self._lock:
                        self._inflight.pop(key, None)
                        event.set()
                    raise
                with self._lock:
                    self._values[key] = value
                    self._inflight.pop(key, None)
                    event.set()
                return value

            event.wait()

    def wrap_retriever(self, retriever, *, task_id: str = ""):
        """Wrap one retriever instance without changing its public API."""
        original = retriever.batch_search
        config = getattr(retriever, "config", {})
        compact_config = tuple(
            (name, getattr(retriever, name, None))
            for name in (
                "max_doc_chars", "max_total_chars", "title_chars",
                "doc_slack_chars", "remaining_floor_chars",
                "max_chunks_per_source",
            )
        )
        namespace = (
            task_id, config.get("search_url"), config.get("topk"), compact_config
        )

        def cached_batch_search(queries=None):
            queries = list(queries or [])
            if not queries:
                return []
            return [
                self.get_or_fetch(
                    (namespace, query),
                    lambda query=query: original([query])[0],
                )
                for query in queries
            ]

        retriever.batch_search = cached_batch_search
        return retriever

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values),
                "hits": self._hits,
                "misses": self._misses,
                "requests_avoided": self._hits,
            }
