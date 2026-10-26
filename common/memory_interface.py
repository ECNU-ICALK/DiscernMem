from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Optional, Union


@dataclass
class Memory:  # self-evolving memory
    text_memory: Optional[str] = None
    extra_fields: dict = field(default_factory=dict)

class BaseCentralizedMemory(ABC):

    def __init__(self):
        self.warnings_issued = {}

    def register_agents(self, agents_list: list["LLMAgent"]):
        logging.info(f"Register {len(agents_list)} agents into the memory.")
    
    @abstractmethod
    def add_memory(self, **kwargs):
        ...

    @abstractmethod
    def retrieve_memory(self, agent_uuid: str) -> Memory:
        ...

    @abstractmethod
    def process_memory(self, text_memory: str, **kwargs) -> Memory:
        ...
    
    @abstractmethod
    def from_config(self, config, working_dir):
        ...