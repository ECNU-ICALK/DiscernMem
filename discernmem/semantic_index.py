from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import threading
from typing import Callable, Iterable

from .schema import MemoryVersion
from .storage import DiscernMemStore, read_json, safe_component


EmbeddingFn = Callable[[str], list[float]]


@dataclass(frozen=True)
class RetrievedIdentity:
    memory_id: str
    version: int
    state: str
    memory: str
    use_when: tuple[str, ...]
    score: float

    @property
    def body(self) -> str:
        return self.memory


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class DiscernMemSemanticIndex:
    """Embedding-backed identity recall with an inspectable JSON cache."""

    def __init__(self, store: DiscernMemStore, embed: EmbeddingFn, *, model_name: str):
        if not model_name:
            raise ValueError(
                "DiscernMem semantic index requires an embedding model name"
            )
        self.store = store
        self.embed = embed
        self.model_name = model_name
        self._vector_cache: dict[str, list[float]] = {}
        self._vector_cache_lock = threading.Lock()

    @staticmethod
    def identity_text(version: MemoryVersion) -> str:
        return " ".join((version.memory, *version.meta.use_when)).strip()

    @staticmethod
    def deployment_text(version: MemoryVersion) -> str:
        return "\n\n".join(
            (
                "Memory:\n" + version.memory,
                "Use when:\n" + "\n".join(version.meta.use_when),
            )
        ).strip()

    def _get_or_create(self, text: str) -> list[float]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._vector_cache_lock:
            cached = self._vector_cache.get(text_hash)
        if cached is not None:
            return list(cached)
        path = self.store.root / "embeddings" / f"{safe_component(text_hash)}.json"
        if path.exists():
            payload = read_json(path)
            if payload.get("namespace") != "discernmem":
                raise ValueError(f"embedding cache namespace mismatch: {path}")
            if (
                payload.get("model_name") != self.model_name
                or payload.get("text_sha256") != text_hash
            ):
                raise ValueError(f"embedding cache metadata mismatch: {path}")
            vector = [float(value) for value in payload.get("vector", [])]
        else:
            vector = [float(value) for value in self.embed(text)]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    "embedding provider returned an empty or non-finite vector"
                )
            with self.store._exclusive():
                self.store._write_once(
                    path,
                    {
                        "namespace": "discernmem",
                        "model_name": self.model_name,
                        "text_sha256": text_hash,
                        "vector": vector,
                    },
                )
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("cached embedding is empty or non-finite")
        with self._vector_cache_lock:
            self._vector_cache[text_hash] = list(vector)
        return vector

    def _retrieve_with_text(
        self,
        candidate_text: str,
        versions: Iterable[MemoryVersion],
        *,
        top_k: int,
        memory_text,
    ) -> list[RetrievedIdentity]:
        query_vector = self._get_or_create(candidate_text)
        results = []
        for version in versions:
            vector = self._get_or_create(memory_text(version))
            results.append(
                RetrievedIdentity(
                    memory_id=version.memory_id,
                    version=version.version,
                    state=version.state.value,
                    memory=version.memory,
                    use_when=version.meta.use_when,
                    score=cosine_similarity(query_vector, vector),
                )
            )
        results.sort(key=lambda item: (-item.score, item.memory_id))
        return results[: max(int(top_k), 0)]

    def retrieve(
        self,
        candidate_text: str,
        versions: Iterable[MemoryVersion],
        *,
        top_k: int,
    ) -> list[RetrievedIdentity]:
        return self._retrieve_with_text(
            candidate_text,
            versions,
            top_k=top_k,
            memory_text=self.identity_text,
        )

    
    @staticmethod
    def body_text(version: MemoryVersion) -> str:
        """Deployment text without meta-memory conditions (ablation switch)."""
        return version.memory

    def retrieve_for_deployment_body_only(
        self,
        candidate_text: str,
        versions: Iterable[MemoryVersion],
        *,
        top_k: int,
    ) -> list[RetrievedIdentity]:
        return self._retrieve_with_text(
            candidate_text,
            versions,
            top_k=top_k,
            memory_text=self.body_text,
        )

    def retrieve_for_deployment(
        self,
        candidate_text: str,
        versions: Iterable[MemoryVersion],
        *,
        top_k: int,
    ) -> list[RetrievedIdentity]:
        return self._retrieve_with_text(
            candidate_text,
            versions,
            top_k=top_k,
            memory_text=self.deployment_text,
        )

    def retrieve_lexical(
        self,
        candidate_text: str,
        versions: Iterable[MemoryVersion],
        *,
        top_k: int,
    ) -> list[RetrievedIdentity]:
        query_tokens = set(re.findall(r"[A-Za-z0-9_]+", candidate_text.lower()))
        results = []
        for version in versions:
            tokens = set(
                re.findall(r"[A-Za-z0-9_]+", self.identity_text(version).lower())
            )
            union = query_tokens | tokens
            score = len(query_tokens & tokens) / len(union) if union else 0.0
            results.append(
                RetrievedIdentity(
                    memory_id=version.memory_id,
                    version=version.version,
                    state=version.state.value,
                    memory=version.memory,
                    use_when=version.meta.use_when,
                    score=score,
                )
            )
        results.sort(
            key=lambda item: (item.score, item.state == "active", item.memory_id),
            reverse=True,
        )
        return results[: max(int(top_k), 0)]
