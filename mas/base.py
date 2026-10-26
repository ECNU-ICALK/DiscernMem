from abc import ABC, abstractmethod
import logging

from typing import Optional, Literal, Callable

from common.registry import registry
from common.memory_interface import BaseCentralizedMemory
from mas.message import MessageNode, MessageGraph
from mas.agent import LLMAgent


class BaseMemoryMAS(ABC):
    """Memory-based Multi-Agent System"""

    def __init__(
        self,
        llm_name_or_path: str,
        centralized_memory: Optional[BaseCentralizedMemory] = None,
        share_llm: bool = True,
        task_domain: Optional[str] = None,
        **kwargs
    ):
        # mas centralized memory: connecting all agents
        self.centralized_memory = centralized_memory
        self.llm_name_or_path = llm_name_or_path
        self.share_llm = share_llm
        self.task_domain = task_domain
        self.agents_list: list[LLMAgent] = list()

    @abstractmethod
    def generate(self, task_domain_instructions: list[str], user_inputs: list[str], generation_config: dict, action_resolver: Callable[[list[str]], list[str]] | None = None) -> list[MessageGraph]:
        ...

    @classmethod
    def from_config(cls, config, working_dir: str, task_domain: str):
        mas_cfg = config.get("mas", dict())
        memory_cfg = config.get("memory", dict())

        memory_name = memory_cfg.get("name")
        memory_cls = registry.get_memory_class(memory_name)
        centralized_memory = memory_cls.from_config(memory_cfg, working_dir)

        memory_mas = cls(centralized_memory=centralized_memory, task_domain=task_domain, **mas_cfg)

        return memory_mas
