from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .prompting import prompt_bundle_fingerprint


class ConfigurationError(ValueError):
    """Raised when a DiscernMem experiment configuration is not reproducible."""


class DataLeakageError(ValueError):
    """Raised when a task crosses an experiment data boundary."""


MAX_EVOLUTION_ROUNDS_LIMIT = 5
"""Upper bound accepted for ``max_evolution_rounds`` in sweep experiments."""


class TaskPartition(str, Enum):
    LEARNING = "learning"
    H_CANDIDATE = "h_candidate"
    FINAL_TEST = "final_test"


def _clean_task_ids(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class TaskSplitManifest:
    """Immutable task identity boundaries fixed before an experiment starts."""

    learning_task_ids: tuple[str, ...] = ()
    h_candidate_task_ids: tuple[str, ...] = ()
    final_test_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_task_ids", _clean_task_ids(self.learning_task_ids)
        )
        object.__setattr__(
            self, "h_candidate_task_ids", _clean_task_ids(self.h_candidate_task_ids)
        )
        object.__setattr__(
            self, "final_test_task_ids", _clean_task_ids(self.final_test_task_ids)
        )
        groups = {
            TaskPartition.LEARNING: set(self.learning_task_ids),
            TaskPartition.H_CANDIDATE: set(self.h_candidate_task_ids),
            TaskPartition.FINAL_TEST: set(self.final_test_task_ids),
        }
        for left, left_ids in groups.items():
            for right, right_ids in groups.items():
                if left.value >= right.value:
                    continue
                overlap = left_ids & right_ids
                if overlap:
                    raise DataLeakageError(
                        f"tasks cannot belong to both {left.value} and {right.value}: "
                        f"{sorted(overlap)}"
                    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TaskSplitManifest":
        data = data or {}
        return cls(
            learning_task_ids=_clean_task_ids(data.get("learning_task_ids")),
            h_candidate_task_ids=_clean_task_ids(data.get("h_candidate_task_ids")),
            final_test_task_ids=_clean_task_ids(data.get("final_test_task_ids")),
        )

    def partition_of(self, task_id: str) -> TaskPartition | None:
        task_id = str(task_id)
        if task_id in self.learning_task_ids:
            return TaskPartition.LEARNING
        if task_id in self.h_candidate_task_ids:
            return TaskPartition.H_CANDIDATE
        if task_id in self.final_test_task_ids:
            return TaskPartition.FINAL_TEST
        return None

    def assert_learning_allowed(self, task_id: str) -> None:
        partition = self.partition_of(task_id)
        if partition is TaskPartition.FINAL_TEST:
            raise DataLeakageError(f"final-test task {task_id!r} cannot enter learning")

    def assert_h_candidate(self, task_id: str) -> None:
        partition = self.partition_of(task_id)
        if partition is not TaskPartition.H_CANDIDATE:
            label = partition.value if partition else "unassigned"
            raise DataLeakageError(
                f"task {task_id!r} is {label}, not an internal H candidate"
            )

    def assert_history_eligible(self, task_id: str) -> None:
        partition = self.partition_of(task_id)
        if partition not in {TaskPartition.LEARNING, TaskPartition.H_CANDIDATE}:
            label = partition.value if partition else "unassigned"
            raise DataLeakageError(
                f"task {task_id!r} is {label}, not eligible for causal history"
            )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "learning_task_ids": list(self.learning_task_ids),
            "h_candidate_task_ids": list(self.h_candidate_task_ids),
            "final_test_task_ids": list(self.final_test_task_ids),
        }


@dataclass(frozen=True)
class DiscernMemConfig:
    """Configuration whose fingerprint defines a DiscernMem experiment."""

    schema_version: int = 1
    method_name: str = "DiscernMem"
    namespace: str = "discernmem"
    epsilon: float = 0.01
    seeds: tuple[int, ...] = (42,)
    max_repair_candidates: int = 3
    max_evolution_rounds: int = 3
    enable_neutral_repair: bool = True
    admission_top_k: int = 5
    retrieval_top_k: int = 3
    max_injected_memories: int = 1
    min_retrieval_similarity: float = 0.25
    retrieval_policy: str = "semantic_top_k"
    h1_max_tasks: int = 2
    h2_tasks: int = 1
    h3_tasks: int = 1
    min_stable_tasks: int = 2
    enable_h3: bool = True
    enable_provisional_retrieval: bool = True
    enable_task_family_filter: bool = True
    enable_retrieval_selector: bool = True
    enable_required_handoffs: bool = False
    enable_capability_constraints: bool = True
    enable_h1_validation: bool = True
    enable_h2_validation: bool = True
    enable_mas_guidance: bool = True
    enable_role_conditioning: bool = True
    enable_use_avoid_conditions: bool = True
    enable_dataset_task_guidance: bool = True
    identity_recall_mode: str = "semantic"
    evolution_mode: str = "cpe"
    enable_birth_calibration: bool = True
    enable_birth_cpe: bool = True
    split_manifest: TaskSplitManifest = field(default_factory=TaskSplitManifest)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if self.schema_version < 1:
            raise ConfigurationError("schema_version must be positive")
        if self.method_name != "DiscernMem" or self.namespace != "discernmem":
            raise ConfigurationError(
                "DiscernMem method and namespace identifiers are fixed"
            )
        if not 0 < float(self.epsilon) < 1:
            raise ConfigurationError("epsilon must be between zero and one")
        if not 0.0 <= float(self.min_retrieval_similarity) <= 1.0:
            raise ConfigurationError("min_retrieval_similarity must be between zero and one")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError(
                "seeds must be a non-empty sequence of unique integers"
            )
        if not 1 <= int(self.max_repair_candidates) <= 3:
            raise ConfigurationError("max_repair_candidates must be in [1, 3]")
        if not 1 <= int(self.max_evolution_rounds) <= MAX_EVOLUTION_ROUNDS_LIMIT:
            raise ConfigurationError(
                f"max_evolution_rounds must be in [1, {MAX_EVOLUTION_ROUNDS_LIMIT}]"
            )
        for name in (
            "retrieval_top_k",
            "h1_max_tasks",
            "h2_tasks",
            "min_stable_tasks",
        ):
            if int(getattr(self, name)) < 1:
                raise ConfigurationError(f"{name} must be positive")
        # Sweep endpoints: `admission_top_k = 0` means "never recall an existing
        # identity" (every candidate becomes a new memory) and
        # `max_injected_memories = 0` means "never inject any memory".
        for name in ("admission_top_k", "max_injected_memories"):
            if int(getattr(self, name)) < 0:
                raise ConfigurationError(f"{name} cannot be negative")
        if int(self.h3_tasks) < 0:
            raise ConfigurationError("h3_tasks cannot be negative")
        if self.enable_h3 and self.h3_tasks < 1:
            raise ConfigurationError("h3_tasks must be positive when H3 is enabled")
        if self.identity_recall_mode not in {"semantic", "lexical"}:
            raise ConfigurationError("identity_recall_mode must be semantic or lexical")
        if self.retrieval_policy != "semantic_top_k":
            raise ConfigurationError("retrieval_policy must be semantic_top_k")
        if self.evolution_mode not in {"cpe", "direct_overwrite"}:
            raise ConfigurationError("evolution_mode must be cpe or direct_overwrite")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "DiscernMemConfig":
        data = dict(data or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigurationError(f"unknown DiscernMem configuration keys: {unknown}")
        if "seeds" in data:
            data["seeds"] = tuple(int(value) for value in data["seeds"])
        if "split_manifest" in data and not isinstance(
            data["split_manifest"], TaskSplitManifest
        ):
            data["split_manifest"] = TaskSplitManifest.from_mapping(
                data["split_manifest"]
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiscernMemConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ConfigurationError("DiscernMem YAML root must be a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["split_manifest"] = self.split_manifest.to_dict()
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.canonical_json().encode("utf-8"))
        digest.update(b"\0")
        digest.update(prompt_bundle_fingerprint().encode("ascii"))
        return digest.hexdigest()

    def export_yaml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=False),
            encoding="utf-8",
        )
        return destination
