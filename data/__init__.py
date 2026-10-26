from data.base_env import (
    BaseEnv,
    StaticEnv,
    DynamicEnv,
)

# Lazy imports to avoid triggering heavy dependencies (datasets, nltk, pddlgym)
# at module load time. Builders are registered via @registry.register_builder
# and accessed through registry.get_builder_class(), not direct import.
# BaseDataBuilder is also lazy so that `from data.base_env import DynamicEnv`
# does not pull in `datasets` through this package's __init__.
def __getattr__(name):
    if name == "BaseDataBuilder":
        from data.base_builder import BaseDataBuilder
        return BaseDataBuilder
    elif name == "KodCodeBuilder":
        from data.kodcode.builder import KodCodeBuilder
        return KodCodeBuilder
    elif name == "TriviaQABuilder":
        from data.triviaqa.builder import TriviaQABuilder
        return TriviaQABuilder
    elif name == "PopQABuilder":
        from data.popqa.builder import PopQABuilder
        return PopQABuilder
    elif name == "PDDLBuilder":
        from data.pddl.builder import PDDLBuilder
        return PDDLBuilder
    raise AttributeError(f"module 'data' has no attribute {name!r}")
