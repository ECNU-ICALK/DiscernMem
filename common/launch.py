"""Shared launcher logic for ``scripts/run.py`` and ``scripts/run_sweep.py``.

A short experiment specification (benchmark, MAS, method variant, overrides)
is expanded into the full ``scripts/run_benchmarks.py`` command used for the
reported experiments:

* runner settings come from the ``protocol`` section of
  ``configs/benchmarks/<benchmark>.yaml``;
* the method configuration is ``configs/discernmem/default.yaml``, optionally
  patched by ``configs/discernmem/variants/<variant>.yaml`` and ``--set``
  overrides, with the online task manifest generated for the selected tasks;
* service endpoints and model names are read from the environment / ``.env``.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import urllib.request
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discernmem.config import DiscernMemConfig  # noqa: E402

RUNNER = PROJECT_ROOT / "scripts" / "run_benchmarks.py"
BENCHMARKS = ("kodcode", "triviaqa", "popqa", "pddl")
MAS_TYPES = ("autogen", "camel", "macnet")
BENCHMARK_CONFIG_DIR = PROJECT_ROOT / "configs" / "benchmarks"
METHOD_DEFAULT = PROJECT_ROOT / "configs" / "discernmem" / "default.yaml"
VARIANT_DIR = PROJECT_ROOT / "configs" / "discernmem" / "variants"
SWEEPS_FILE = PROJECT_ROOT / "configs" / "discernmem" / "sweeps.yaml"

# Identifiers that define the method or the task stream; never overridable.
FROZEN_FIELDS = frozenset({"split_manifest", "method_name", "namespace", "schema_version"})
REQUIRED_ENV = (
    "LLM_BASE_URL",
    "LLM_MODEL",
    "DISCERNMEM_EMBED_BASE_URL",
    "DISCERNMEM_EMBED_MODEL",
)


class LaunchError(ValueError):
    """Raised for an invalid experiment specification."""


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def load_environment(require: bool = True) -> dict[str, str]:
    """Return the process environment with ``.env`` applied (shell wins)."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    env = dict(os.environ)
    env.setdefault("LLM_API_KEY", "EMPTY")
    env.setdefault("DISCERNMEM_EMBED_API_KEY", "EMPTY")
    missing = [key for key in REQUIRED_ENV if not env.get(key)]
    if require and missing:
        raise LaunchError(
            "missing environment variables: "
            + ", ".join(missing)
            + " (copy .env.example to .env and fill in the service endpoints)"
        )
    return env


# --------------------------------------------------------------------------
# Benchmark protocol
# --------------------------------------------------------------------------
def benchmark_config_path(benchmark: str) -> Path:
    if benchmark not in BENCHMARKS:
        raise LaunchError(f"unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")
    return BENCHMARK_CONFIG_DIR / f"{benchmark}.yaml"


def load_protocol(benchmark: str) -> dict[str, Any]:
    """Runner settings reported in the paper for ``benchmark``."""
    payload = yaml.safe_load(benchmark_config_path(benchmark).read_text(encoding="utf-8")) or {}
    protocol = dict(payload.get("protocol") or {})
    for key in ("data_split", "start_from", "num_tasks"):
        if key not in protocol:
            raise LaunchError(f"configs/benchmarks/{benchmark}.yaml protocol lacks {key!r}")
    return protocol


# --------------------------------------------------------------------------
# Method configuration
# --------------------------------------------------------------------------
def available_variants() -> tuple[str, ...]:
    return ("full",) + tuple(sorted(path.stem for path in VARIANT_DIR.glob("*.yaml")))


def _overridable_fields() -> set[str]:
    return {item.name for item in fields(DiscernMemConfig)} - FROZEN_FIELDS


_FIELD_TYPES = {"bool": bool, "int": int, "float": (int, float), "str": str}


def _field_types() -> dict[str, str]:
    return {item.name: str(item.type) for item in fields(DiscernMemConfig)}


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` method-config overrides (values are YAML)."""
    allowed = _overridable_fields()
    overrides: dict[str, Any] = {}
    for item in items:
        key, separator, raw = str(item).partition("=")
        key = key.strip()
        if not separator or not key:
            raise LaunchError(f"--set expects KEY=VALUE, received {item!r}")
        if key not in allowed:
            raise LaunchError(
                f"--set cannot override {key!r}; allowed keys: {', '.join(sorted(allowed))}"
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise LaunchError(f"--set {item!r} is not valid YAML: {exc}") from exc
        expected = _FIELD_TYPES.get(_field_types()[key])
        if expected is not None and (
            not isinstance(value, expected) or (expected is not bool and isinstance(value, bool))
        ):
            raise LaunchError(f"--set {key} expects {_field_types()[key]}, received {raw!r}")
        overrides[key] = value
    return overrides


def variant_patch(variant: str) -> dict[str, Any]:
    if variant == "full":
        return {}
    path = VARIANT_DIR / f"{variant}.yaml"
    if not path.exists():
        raise LaunchError(f"unknown variant {variant!r}; choose from {available_variants()}")
    patch = dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    frozen = sorted(set(patch) & FROZEN_FIELDS)
    if frozen:
        raise LaunchError(f"variant {variant!r} must not set {frozen}")
    return patch


def task_ids(benchmark: str, data_split: str, start_from: int, num_tasks: int) -> list[str]:
    if num_tasks < 1 or start_from < 0:
        raise LaunchError("num_tasks must be positive and start_from non-negative")
    return [f"{benchmark}_{data_split}_{index}" for index in range(start_from, start_from + num_tasks)]


def build_method_config(
    benchmark: str,
    *,
    variant: str = "full",
    overrides: Mapping[str, Any] | None = None,
    data_split: str = "test",
    start_from: int = 0,
    num_tasks: int,
) -> DiscernMemConfig:
    """Full-method config + variant + overrides, with an all-learning online manifest."""
    payload = dict(yaml.safe_load(METHOD_DEFAULT.read_text(encoding="utf-8")) or {})
    payload.update(variant_patch(variant))
    payload.update(dict(overrides or {}))
    payload["split_manifest"] = {
        "learning_task_ids": task_ids(benchmark, data_split, start_from, num_tasks),
        "h_candidate_task_ids": [],
        "final_test_task_ids": [],
    }
    return DiscernMemConfig.from_mapping(payload)


def config_changes(config: DiscernMemConfig) -> dict[str, Any]:
    """Method settings that differ from the full-method default."""
    default = DiscernMemConfig.from_yaml(METHOD_DEFAULT).to_dict()
    current = config.to_dict()
    return {
        key: value
        for key, value in current.items()
        if key != "split_manifest" and default.get(key) != value
    }


# --------------------------------------------------------------------------
# Runner command
# --------------------------------------------------------------------------
def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _flag(key: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return [f"--{key}"] if value else []
    return [f"--{key}", str(value)]


def default_run_id(benchmark: str, mas: str, label: str) -> str:
    return f"{benchmark}_{mas}_{label}"


def label_for(variant: str, overrides: Mapping[str, Any]) -> str:
    parts = [variant] + [f"{key}-{value}" for key, value in sorted(overrides.items())]
    return "_".join(str(part).replace("/", "-") for part in parts)


def leaf_dir(results_root: Path, run_id: str, benchmark: str, mas: str) -> Path:
    return results_root / run_id / benchmark / mas / "discernmem"


def method_config_path(results_root: Path, run_id: str, benchmark: str, mas: str) -> Path:
    return results_root / run_id / "configs" / f"{benchmark}_{mas}_discernmem.yaml"


def build_runner_args(
    *,
    benchmark: str,
    mas: str,
    run_id: str,
    results_root: Path,
    method_config: Path,
    protocol: Mapping[str, Any],
    env: Mapping[str, str],
    runner_overrides: Mapping[str, Any] | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Arguments for ``scripts/run_benchmarks.py`` (without the interpreter)."""
    if mas not in MAS_TYPES:
        raise LaunchError(f"unknown MAS {mas!r}; choose from {MAS_TYPES}")
    settings = dict(protocol)
    if "search_url" in settings and env.get("SEARCH_URL"):
        settings["search_url"] = env["SEARCH_URL"]
    settings.update(dict(runner_overrides or {}))
    args = [
        "--run_id", run_id,
        "--results_root", _relative(results_root),
        "--benchmark", benchmark,
        "--mas_type", mas,
        "--memory_backend", "discernmem",
        "--config", _relative(benchmark_config_path(benchmark)),
        "--discernmem_config", _relative(method_config),
    ]
    for key, value in settings.items():
        args += _flag(key, value)
    args += [
        "--discernmem_embedding_api_base", env["DISCERNMEM_EMBED_BASE_URL"],
        "--discernmem_embedding_model", env["DISCERNMEM_EMBED_MODEL"],
        "--api_base", env["LLM_BASE_URL"],
        "--model_name", env["LLM_MODEL"],
    ]
    return args + list(extra_args)


def runner_command(runner_args: Sequence[str]) -> list[str]:
    return [sys.executable, str(RUNNER), *runner_args]


def format_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


# --------------------------------------------------------------------------
# Execution helpers
# --------------------------------------------------------------------------
def preflight(env: Mapping[str, str], *, need_search: bool = False) -> None:
    """Fail early when an OpenAI-compatible endpoint is unreachable."""
    targets = [("LLM", env["LLM_BASE_URL"]), ("Embedding", env["DISCERNMEM_EMBED_BASE_URL"])]
    for label, base in targets:
        url = base.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as exc:  # noqa: BLE001 - reported to the operator
            raise LaunchError(f"{label} service not reachable at {url}: {exc}") from exc


def run_command(command: Sequence[str], env: Mapping[str, str], log_path: Path | None = None) -> int:
    """Run the runner from the project root, optionally teeing into a log file."""
    if log_path is None:
        return subprocess.call(list(command), cwd=PROJECT_ROOT, env=dict(env))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(format_command(command) + "\n\n")
        handle.flush()
        return subprocess.call(
            list(command), cwd=PROJECT_ROOT, env=dict(env), stdout=handle, stderr=subprocess.STDOUT
        )
