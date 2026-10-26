"""DiscernMem memory evolution backend.

Method configuration, prompts, storage, and persisted artifacts live under
the ``discernmem`` namespace; the benchmark runner and MAS hosts interact with
it through ``DiscernMemCentralizedMemory``.
"""

from .config import TaskPartition, TaskSplitManifest, DiscernMemConfig
from .centralized_adapter import DiscernMemCentralizedMemory

__all__ = [
    "TaskPartition",
    "TaskSplitManifest",
    "DiscernMemCentralizedMemory",
    "DiscernMemConfig",
]
