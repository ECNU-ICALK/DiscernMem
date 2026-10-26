from abc import ABC, abstractmethod
from typing import Type

from datasets import DatasetDict

from data.base_env import BaseEnv

class BaseDataBuilder(ABC):

    def __init__(self, cfg: dict = None):
        super().__init__()

        cfg = cfg or {}
        self.mode = cfg.get("mode", "eval")
        self.config = cfg.get(self.mode)
    
    def get_dataset_dict(self) -> DatasetDict:
        return self._build_datasets()
    
    @abstractmethod
    def get_env_cls(self) -> Type[BaseEnv]:
        ...

    @abstractmethod
    def _build_datasets(self):
        ...

    
