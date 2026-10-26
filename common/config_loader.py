"""
Configuration loading helpers for YAML benchmark configs.

Supports configs/benchmarks/*.yaml and exposes a small typed access layer for
common runtime parameters.
"""
import os
import yaml
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Generation settings."""
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.0
    early_stopping: bool = True
    use_cache: bool = True


@dataclass
class InteractionConfig:
    """Interaction settings."""
    batch_size: int = 16
    max_turns: int = 1
    max_obs_length: int = 2048
    timeout_seconds: Optional[float] = None


@dataclass
class ModelConfig:
    """Model settings."""
    load_model_path: Optional[str] = None
    structure: str = "autogen"
    llm_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration with the common runtime fields."""
    
    # Model settings.
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Generation settings.
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Interaction settings.
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    
    # Runtime settings.
    seed: int = 42
    device: int = 0
    working_dir: Optional[str] = None
    
    # Raw config dictionary for fields not mapped above.
    raw_config: dict = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load a config from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "BenchmarkConfig":
        """Create a config object from a dictionary."""
        raw_config = config_dict.copy()
        
        # Parse model settings.
        model_dict = config_dict.get('model', {})
        mas_dict = model_dict.get('mas', {})
        model = ModelConfig(
            load_model_path=model_dict.get('load_model_path'),
            structure=mas_dict.get('structure', 'autogen'),
            llm_name_or_path=mas_dict.get('llm_name_or_path', 'Qwen/Qwen3-4B-Instruct-2507')
        )
        
        # Parse generation settings.
        gen_dict = config_dict.get('run', {}).get('generation', {})
        generation = GenerationConfig(
            max_new_tokens=gen_dict.get('max_new_tokens', 1024),
            do_sample=gen_dict.get('do_sample', False),
            temperature=gen_dict.get('temperature', 0.0),
            early_stopping=gen_dict.get('early_stopping', True),
            use_cache=gen_dict.get('use_cache', True)
        )
        
        # Parse interaction settings.
        inter_dict = config_dict.get('run', {}).get('interaction', {})
        interaction = InteractionConfig(
            batch_size=inter_dict.get('batch_size', 16),
            max_turns=inter_dict.get('max_turns', 1),
            max_obs_length=inter_dict.get('max_obs_length', 2048),
            timeout_seconds=inter_dict.get('timeout_seconds')
        )
        
        # Parse runtime settings.
        run_dict = config_dict.get('run', {})
        seed = run_dict.get('seed', 42)
        device = run_dict.get('device', 0)
        working_dir = run_dict.get('working_dir')
        
        return cls(
            model=model,
            generation=generation,
            interaction=interaction,
            seed=seed,
            device=device,
            working_dir=working_dir,
            raw_config=raw_config
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested config value with dot notation, e.g. 'run.generation.max_new_tokens'."""
        keys = key.split('.')
        value = self.raw_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
def get_default_config_path(benchmark: str) -> str:
    """Return the default config path for a benchmark."""
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'benchmarks')
    return os.path.join(config_dir, f"{benchmark}.yaml")


def load_benchmark_config(benchmark: str, config_path: Optional[str] = None) -> BenchmarkConfig:
    """
    Load the configuration for a benchmark.
    
    Args:
        benchmark: Benchmark name (kodcode, pddl, popqa, triviaqa).
        config_path: Optional custom config path. If None, use the default path.
    
    Returns:
        BenchmarkConfig object.
    """
    if config_path is None:
        config_path = get_default_config_path(benchmark)
    
    if not os.path.exists(config_path):
        # Fall back to defaults when the config file is missing.
        return BenchmarkConfig()
    
    return BenchmarkConfig.from_yaml(config_path)


# Convenience helpers for individual benchmarks.


