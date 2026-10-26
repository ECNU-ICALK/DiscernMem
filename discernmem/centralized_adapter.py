"""DiscernMem centralized memory adapter for the repository MAS runners."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Optional

from common.registry import registry
from common.memory_interface import BaseCentralizedMemory, Memory
from common.llm_client import EmbeddingClient
from common.stats import stats

from .admission_controller import DiscernMemAdmission
from .config import DiscernMemConfig
from .controller import DiscernMemController
from .cpe import CapabilityPreservingEvolution
from .identity_judge import DiscernMemIdentityJudge
from .meta_summarizer import DiscernMemMetaSummarizer
from .pipeline import DiscernMemPromptPipeline
from .repair_generator import DiscernMemRepairGenerator
from .replay import ReplayExecutor, runtime_signature
from .retrieval import PreActionContext, DiscernMemRetriever
from .retrieval_selector import DiscernMemRetrievalSelector
from .scheduling import ReplayTaskContext
from .semantic_index import DiscernMemSemanticIndex
from .storage import DiscernMemStore
from .validation import DiscernMemValidationPlanner, ValidationTaskIndex
from .usage_assessor import DiscernMemUsageAssessor


logger = logging.getLogger(__name__)


@registry.register_memory("discernmem")
class DiscernMemCentralizedMemory(BaseCentralizedMemory):
    def __init__(
        self,
        *,
        storage_dir: str,
        config_path: str,
        task_domain: str = "unknown",
        read_only: bool = False,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        llm_timeout: float = 180.0,
        embedding_timeout: float = 60.0,
        replay_workers: int = 16,
    ):
        super().__init__()
        self.task_domain = str(task_domain)
        self.read_only = bool(read_only)
        self.storage_dir = Path(storage_dir)
        self.config_path = Path(config_path)
        self.config = DiscernMemConfig.from_yaml(self.config_path)
        self.store = DiscernMemStore(self.storage_dir, self.config)
        self.storage = self
        self.model_name = llm_model_name or os.getenv("LLM_MODEL") or ""
        if not self.model_name:
            raise ValueError("DiscernMem requires --model_name or LLM_MODEL")
        llm_base = (
            llm_api_base or os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8100/v1"
        )
        llm_key = llm_api_key or os.getenv("LLM_API_KEY") or "EMPTY"
        embedding_base = (
            embedding_api_base
            or os.getenv("DISCERNMEM_EMBED_BASE_URL")
            or "http://127.0.0.1:8002/v1"
        )
        embedding_key = (
            embedding_api_key or os.getenv("DISCERNMEM_EMBED_API_KEY") or "EMPTY"
        )
        embedding_model = (
            embedding_model_name
            or os.getenv("DISCERNMEM_EMBED_MODEL")
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.embedding_client = EmbeddingClient(
            api_key=embedding_key,
            base_url=embedding_base,
            model=embedding_model,
            timeout=embedding_timeout,
        )
        self.semantic_index = DiscernMemSemanticIndex(
            self.store,
            self.embedding_client.embed,
            model_name=embedding_model,
        )
        self.extraction = DiscernMemPromptPipeline(
            store=self.store,
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.identity_judge = DiscernMemIdentityJudge(
            store=self.store,
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.meta_summarizer = DiscernMemMetaSummarizer(
            store=self.store,
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.usage_assessor = DiscernMemUsageAssessor(
            store=self.store,
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.repair_generator = DiscernMemRepairGenerator(
            store=self.store,
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.cpe = CapabilityPreservingEvolution(store=self.store)
        self.admission = DiscernMemAdmission(
            store=self.store,
            semantic_index=self.semantic_index,
            identity_judge=self.identity_judge.judge,
        )
        self.retrieval_selector = DiscernMemRetrievalSelector(
            api_base=llm_base,
            api_key=llm_key,
            model_name=self.model_name,
            timeout=llm_timeout,
        )
        self.retriever = DiscernMemRetriever(
            store=self.store,
            semantic_index=self.semantic_index,
            selector=self.retrieval_selector,
        )
        self.validation_planner = DiscernMemValidationPlanner(
            store=self.store,
            task_index=ValidationTaskIndex(
                self.store,
                self.embedding_client.embed,
                model_name=embedding_model,
            ),
        )
        self.mas_spec: dict[str, Any] = {
            "mas_type": "unknown",
            "roles": [],
            "edges": [],
        }
        self.controller = DiscernMemController(
            store=self.store,
            extraction=self.extraction,
            admission=self.admission,
            validation_planner=self.validation_planner,
            replay_executor=None,
            meta_summarizer=self.meta_summarizer.summarize,
            usage_assessor=self.usage_assessor,
            repair_generator=self.repair_generator,
            cpe=self.cpe,
            mas_spec=self.mas_spec,
        )
        self.replay_workers = max(int(replay_workers), 1)
        self._agent_roles: dict[str, str] = {}
        self._current_task_context = {
            "task_id": "",
            "task_index": -1,
            "execution_type": "official",
        }
        self._retrieval_usage: list[dict[str, Any]] = []
        self._retrieval_log_signatures: dict[
            tuple[str, str], tuple[tuple[str, int], ...]
        ] = {}
        self._task_role_memory: dict[tuple[str, str, str], Memory] = {}
        self._last_update_usage: dict[str, Any] = {}
        self._saved_trajectory_files: list[str] = []

    @classmethod
    def from_config(cls, config: dict, working_dir: str) -> "DiscernMemCentralizedMemory":
        cfg = dict(config or {})
        config_path = cfg.get("config_path")
        if not config_path:
            raise ValueError("DiscernMem from_config requires config_path")
        return cls(
            storage_dir=cfg.get("storage_dir") or str(Path(working_dir) / "discernmem"),
            config_path=config_path,
            task_domain=cfg.get("task_domain", "unknown"),
            read_only=cfg.get("read_only", False),
            llm_api_base=cfg.get("llm_api_base"),
            llm_api_key=cfg.get("llm_api_key"),
            llm_model_name=cfg.get("llm_model_name"),
            embedding_api_base=cfg.get("embedding_api_base"),
            embedding_api_key=cfg.get("embedding_api_key"),
            embedding_model_name=cfg.get("embedding_model_name"),
            llm_timeout=cfg.get("llm_timeout", 180.0),
            embedding_timeout=cfg.get("embedding_timeout", 60.0),
            replay_workers=cfg.get("replay_workers", 16),
        )

    def register_agents(self, agents_list: list):
        roles = []
        for order, agent in enumerate(agents_list):
            agent_id = getattr(agent, "id", None) or str(id(agent))
            role = getattr(agent, "role", None) or f"agent_{order}"
            self._agent_roles[agent_id] = role
            roles.append(
                {
                    "name": role,
                    "duty": str(getattr(agent, "system_prompt_template", ""))[:1200],
                    "order": order,
                    "agent_id": agent_id,
                }
            )
        self.mas_spec["roles"] = roles
        self.controller.set_mas_spec(self.mas_spec)

    def set_mas_spec(self, mas_spec: Mapping[str, Any]):
        provided = dict(mas_spec or {})
        if provided.get("mas_type"):
            self.mas_spec["mas_type"] = provided["mas_type"]
        if provided.get("roles"):
            self.mas_spec["roles"] = list(provided["roles"])
        if "edges" in provided:
            self.mas_spec["edges"] = [list(edge) for edge in provided["edges"]]
        self.controller.set_mas_spec(self.mas_spec)

    def set_runtime_context(self, model_name=None, mas_architecture=None):
        if model_name and model_name != self.model_name:
            raise ValueError(
                "DiscernMem model cannot change after configuration is frozen"
            )
        if mas_architecture:
            self.mas_spec["mas_type"] = str(mas_architecture)
            self.controller.set_mas_spec(self.mas_spec)

    def set_current_task_context(
        self, task_id: str, task_index: int = -1, execution_type: str = "official"
    ):
        next_context = {
            "task_id": str(task_id),
            "task_index": int(task_index),
            "execution_type": str(execution_type),
        }
        if next_context != self._current_task_context:
            self._task_role_memory.clear()
        self._current_task_context = next_context

    def set_replay_runner(
        self,
        runner,
        *,
        process_isolation: bool = False,
        replay_timeout_seconds: float | None = None,
    ) -> None:
        executor = ReplayExecutor(
            self.store,
            runner,
            max_workers=self.replay_workers,
            max_attempts=2,
            process_isolation=process_isolation,
            crash_retries=1,
            replay_timeout_seconds=replay_timeout_seconds,
        )
        self.controller.set_replay_executor(executor)

    def _role(self, agent=None) -> str:
        if agent is None:
            return "default"
        agent_id = getattr(agent, "id", None) or str(id(agent))
        return (
            self._agent_roles.get(agent_id) or getattr(agent, "role", None) or "default"
        )

    @staticmethod
    def _method_usage_snapshot() -> dict[str, float]:
        sources = stats.to_dict().get("by_source", {})
        trace_sources = [
            values
            for source, values in sources.items()
            if str(source).startswith("discernmem/")
        ]
        return {
            "llm_calls": sum(int(item.get("calls", 0) or 0) for item in trace_sources),
            "prompt_tokens": sum(
                int(item.get("prompt_tokens", 0) or 0) for item in trace_sources
            ),
            "completion_tokens": sum(
                int(item.get("completion_tokens", 0) or 0) for item in trace_sources
            ),
            "llm_time_seconds": sum(
                float(item.get("time", 0.0) or 0.0) for item in trace_sources
            ),
        }

    def retrieve_memory(self, task_description: str, agent=None) -> Memory:
        started = time.perf_counter()
        role = self._role(agent)
        task_id = str(self._current_task_context.get("task_id") or "")
        execution_type = str(
            self._current_task_context.get("execution_type") or "official"
        )
        cache_key = (execution_type, task_id, role)
        if task_id and cache_key in self._task_role_memory:
            return self._task_role_memory[cache_key]
        retrieval_query, separator, upstream = str(task_description).partition(
            "\n\nVisible upstream messages:\n"
        )
        retrieval_query = retrieval_query.strip()
        received_messages = (
            (upstream.strip(),) if separator and upstream.strip() else ()
        )

        result = self.retriever.retrieve(
            PreActionContext.from_mapping(
                {
                    "task_description": retrieval_query,
                    "role": role,
                    "observable_state": "",
                    "received_messages": received_messages,
                    "decision_point": "task-level guidance for this MAS role",
                    "metadata": {
                        "task_id": self._current_task_context.get("task_id"),
                        "execution_type": self._current_task_context.get(
                            "execution_type"
                        ),
                    },
                }
            )
        )
        injected = [
            {
                "memory_id": version.memory_id,
                "version": version.version,
                "injected_role": role,
            }
            for version in result.selected
        ]
        usage = {
            "memory_backend": "discernmem",
            **self._current_task_context,
            "agent_role": role,
            "retrieval_id": result.retrieval_id,
            "retrieved_cards": len(result.selected),
            "injected_cards": len(result.selected),
            "injected_memory_count": len(result.selected),
            "injected_card_occurrence_count": len(result.selected),
            "injected_memories": injected,
            "retrieval_audit_path": result.audit_path,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self._retrieval_usage.append(usage)
        execution_label = execution_type.upper()
        task_label = task_id or "unknown"
        signature = tuple(
            (str(item["memory_id"]), int(item["version"])) for item in injected
        )
        log_key = (f"{execution_label}:{task_label}", role)
        previous_signature = self._retrieval_log_signatures.get(log_key)
        if previous_signature != signature:
            logger.info(
                "[DiscernMem][RETRIEVE][%s task=%s][role=%s] %s selected=%d memories=%s latency_ms=%.1f",
                execution_label,
                task_label,
                role,
                "first" if previous_signature is None else "changed",
                len(signature),
                [f"{memory_id}@v{version}" for memory_id, version in signature],
                usage["latency_ms"],
            )
            self._retrieval_log_signatures[log_key] = signature
        memory = Memory(
            text_memory=result.memory_text or None,
            extra_fields={
                "memory_backend": "discernmem",
                "retrieval_id": result.retrieval_id,
                "injected_memories": injected,
            },
        )
        if task_id:
            self._task_role_memory[cache_key] = memory
        return memory

    def process_memory(self, text_memory: str, **kwargs) -> Memory:
        return Memory(
            text_memory=text_memory or None,
            extra_fields=kwargs.get("extra_fields") or {},
        )

    def add_memory(self, **kwargs):
        return self.on_task_complete(**kwargs)

    def on_task_complete(
        self,
        *,
        task_id: str,
        task_description: str,
        outcome: str,
        trajectory: Mapping[str, Any],
        score: float,
        task_index: int = -1,
        **metadata,
    ):
        if self.read_only:
            self._last_update_usage = {"task_id": task_id, "status": "read_only"}
            return None
        task_retrievals = [
            dict(item)
            for item in self._retrieval_usage
            if str(item.get("task_id") or "") == str(task_id)
        ]
        trajectory = dict(trajectory)
        trajectory["memory_retrievals"] = task_retrievals
        trajectory["task_family"] = str(
            metadata.get("task_family") or metadata.get("game_name") or ""
        )
        signature_payload = {
            "task_id": task_id,
            "model_name": self.model_name,
            "mas_spec": self.mas_spec,
            "generation_config": metadata.get("generation_config"),
            "environment_config": metadata.get("environment_config"),
            "evaluator_config": metadata.get("evaluator_config"),
            "benchmark": metadata.get("benchmark", self.task_domain),
        }
        replay_context = ReplayTaskContext(
            task_id=task_id,
            task_index=task_index,
            runtime_signature=runtime_signature(signature_payload),
            metadata={**metadata, "runtime_signature_payload": signature_payload},
        )
        before_usage = self._method_usage_snapshot()
        started = time.perf_counter()
        source_entities = {
            str(value)
            for value in metadata.get("source_entities", [])
            if str(value).strip()
        }
        trajectory["source_entities"] = sorted(
            source_entities, key=lambda value: (value.lower(), value)
        )
        result = self.controller.process_task(
            task_id=task_id,
            task_index=task_index,
            task_description=task_description,
            outcome=outcome,
            score=float(score),
            trajectory=trajectory,
            replay_context=replay_context,
            source_entities=tuple(
                sorted(source_entities, key=lambda value: (value.lower(), value))
            ),
        )
        after_usage = self._method_usage_snapshot()
        usage_delta = {key: after_usage[key] - before_usage[key] for key in after_usage}
        self._last_update_usage = {
            "task_id": task_id,
            "status": result.status,
            "candidate_cards": int(bool(result.admission_decision)),
            "admitted_cards": int(result.admission_decision in {"create", "attach"}),
            "committed_cards": int(result.admission_decision == "create"),
            "replay_runs": len(result.replay_ids),
            **usage_delta,
            "total_tokens": (
                usage_delta["prompt_tokens"] + usage_delta["completion_tokens"]
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            **result.to_dict(),
        }
        self.store.record_progress(
            task_id=task_id,
            status=result.status,
            task_index=task_index,
            memory_id=result.memory_id,
            replay_count=len(result.replay_ids),
            details={
                "admission_decision": result.admission_decision,
                "state": result.state,
                "event_id": result.event_id,
                "formal_use_event_ids": list(result.formal_use_event_ids),
            },
        )
        logger.info(
            "[DiscernMem][COMPLETE][task=%s] status=%s decision=%s memory=%s state=%s replays=%d latency_ms=%.1f",
            task_id,
            result.status,
            result.admission_decision or "-",
            result.memory_id or "-",
            result.state or "-",
            len(result.replay_ids),
            self._last_update_usage["latency_ms"],
        )
        return result.to_dict()

    def get_and_reset_retrieval_usage(self) -> list[dict[str, Any]]:
        value = list(self._retrieval_usage)
        self._retrieval_usage.clear()
        return value

    def count_active_cards(self) -> int:
        return len(self.store.iter_current_versions(states={"active"}))

    def get_usage_stats(self) -> dict[str, Any]:
        identities = self.store.iter_identities()
        versions = [self.store.get_version(item.memory_id) for item in identities]
        return {
            "memory_backend": "discernmem",
            "storage_dir": str(self.store.output_root),
            "records_storage_dir": str(self.store.records_root),
            "reports_storage_dir": str(self.store.reports_root),
            "config_fingerprint": self.config.fingerprint,
            "memories": len(identities),
            "active_memories": sum(item.state.value == "active" for item in versions),
            "provisional_memories": sum(
                item.state.value == "provisional" for item in versions
            ),
            "inactive_memories": sum(
                item.state.value == "inactive" for item in versions
            ),
            "retrieval_calls_total": len(
                list((self.store.root / "audit/retrievals").glob("*.json"))
            ),
        }

    def get_saved_trajectory_files(self) -> list[str]:
        return [
            str(path)
            for path in sorted((self.store.root / "trajectories").rglob("*.json"))
        ]
