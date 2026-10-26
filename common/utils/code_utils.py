#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

ExecuteResult = Tuple[bool, str, Tuple[bool]]

KODCODE_ENV_MODES = ("pytest",)


def resolve_kodcode_timeout(mode: str, timeout: Optional[int] = None) -> int:
    if timeout is not None:
        return int(timeout)
    return 30


def _strip_code_wrappers(block: str) -> str:
    """Remove common model-output wrappers while preserving the Python source."""
    text = (block or "").strip()
    if not text:
        return ""

    # Models sometimes put a markdown code block inside XML <code> tags, or
    # forget the closing fence. Strip repeated outer wrappers before parsing.
    while True:
        stripped = text.strip()
        fence_match = re.fullmatch(
            r"```(?:python|py)?[ \t\r\n]*(.*?)(?:```)?[ \t\r\n]*",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            text = fence_match.group(1).strip()
            continue

        xml_match = re.fullmatch(
            r"(?:<implementation>\s*)?<code>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</code>\s*(?:</implementation>)?",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if xml_match:
            text = xml_match.group(1).strip()
            continue
        break

    # Remove stray fence-only lines that can remain after partial extraction.
    text = re.sub(r"(?im)^\s*```(?:python|py)?\s*$", "", text)
    text = re.sub(r"(?im)^\s*```\s*$", "", text)
    return text.strip()


def _extract_python_blocks(text_string: str) -> List[str]:
    xml_code_blocks = re.findall(
        r"<code>\s*<!\[CDATA\[(.*?)\]\]>\s*</code>",
        text_string,
        re.DOTALL,
    )
    if not xml_code_blocks:
        xml_code_blocks = re.findall(r"<code>(.*?)</code>", text_string, re.DOTALL)
    if xml_code_blocks:
        blocks = xml_code_blocks
    else:
        blocks = re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text_string,
            re.DOTALL | re.IGNORECASE,
        )
        if not blocks:
            blocks = [text_string]

    cleaned_blocks = [_strip_code_wrappers(block) for block in blocks if block and block.strip()]
    cleaned_blocks = [block for block in cleaned_blocks if block]
    if cleaned_blocks:
        return cleaned_blocks
    stripped = text_string.strip()
    return [_strip_code_wrappers(stripped)] if stripped else []


def extract_python_programs(text_string: str) -> List[str]:
    """Return full extracted Python program blocks without dropping structure."""
    return _extract_python_blocks(text_string)


def collect_python_program(text_string: str) -> str:
    """Collect extracted program blocks into one executable source string."""
    return "\n\n".join(extract_python_programs(text_string))


def rename_function(function: str, function_name: str) -> str:
    """
    Replace the name of the first function in `answer` with `function_name`.
    Only modifies the function name, keeps everything else intact.
    """
    pattern = r"def\s+(\w+)\s*\("

    new_answer = re.sub(pattern, f"def {function_name}(", function, count=1)
    return new_answer


def prepare_kodcode_solution(source: str, function_name: str, mode: str) -> tuple[str, List[str]]:
    if not function_name:
        return source, []
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return source, []
    has_target = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        for node in tree.body
    )
    if has_target:
        return source, []
    renamed = rename_function(source, function_name)
    if renamed != source:
        return renamed, ["rename_first_function_when_target_missing"]
    return source, []


@dataclass
class KodCodeExecutionResult:
    passed: bool
    feedback: str
    collected: int = 0
    passed_count: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    timeout: bool = False
    exit_code: Optional[int] = None
    failure_category: Optional[str] = None
    test_normalizations: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class _GuardTopLevelPytestMain(ast.NodeTransformer):
    def __init__(self):
        self.changed = False

    @staticmethod
    def _is_pytest_main_call(node: ast.stmt) -> bool:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        func = node.value.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
            and func.attr == "main"
        )

    def visit_Module(self, node: ast.Module):
        guarded_calls = []
        retained = []
        for statement in node.body:
            if self._is_pytest_main_call(statement):
                guarded_calls.append(statement)
                self.changed = True
            else:
                retained.append(statement)
        if guarded_calls:
            retained.append(
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id="__name__", ctx=ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(value="__main__")],
                    ),
                    body=guarded_calls,
                    orelse=[],
                )
            )
        node.body = retained
        return node


def normalize_kodcode_test_source(test_source: str) -> tuple[str, List[str]]:
    source = test_source or ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    transformer = _GuardTopLevelPytestMain()
    tree = transformer.visit(tree)
    if not transformer.changed:
        return source, []
    ast.fix_missing_locations(tree)
    normalized = ast.unparse(tree)
    return normalized + ("\n" if source.endswith("\n") else ""), ["guard_top_level_pytest_main"]


def add_official_kodcode_test_harness(test_source: str) -> tuple[str, List[str]]:
    normalized, normalizations = normalize_kodcode_test_source(test_source)
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return normalized, normalizations
    has_solution_star_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "solution"
        and any(alias.name == "*" for alias in node.names)
        for node in tree.body
    )
    if has_solution_star_import:
        return normalized, normalizations
    return "from solution import *\n\n" + normalized, [
        "inject_official_solution_star_import",
        *normalizations,
    ]


_PYTEST_RESULT_PLUGIN = r'''
import json
import os

_RESULTS = {
    "collected": 0,
    "passed": 0,
    "failed": 0,
    "errors": 0,
    "skipped": 0,
}

def pytest_collection_finish(session):
    _RESULTS["collected"] = len(session.items)

def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            _RESULTS["passed"] += 1
        elif report.failed:
            _RESULTS["failed"] += 1
        elif report.skipped:
            _RESULTS["skipped"] += 1
    elif report.when in ("setup", "teardown"):
        if report.failed:
            _RESULTS["errors"] += 1
        elif report.skipped and report.when == "setup":
            _RESULTS["skipped"] += 1

def pytest_sessionfinish(session, exitstatus):
    _RESULTS["exit_code"] = int(exitstatus)
    path = os.environ.get("KODCODE_PYTEST_RESULT_FILE")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_RESULTS, handle)
'''


def _classify_pytest_failure(
    exit_code: Optional[int],
    stdout: str,
    stderr: str,
    timeout: bool,
) -> Optional[str]:
    if timeout:
        return "timeout"
    combined = f"{stdout}\n{stderr}"
    if exit_code == 0:
        return None
    if "ModuleNotFoundError" in combined or "No module named" in combined:
        if "solution.py" in combined:
            return "candidate_import_error"
        return "missing_environment_dependency"
    if "optional dependency" in combined and "requires" in combined:
        return "missing_environment_dependency"
    if "DeferredCudaCallError" in combined or "CUDAContext.cpp" in combined:
        return "environment_runtime_error"
    if "SyntaxError" in combined or "IndentationError" in combined:
        if "solution.py" in combined:
            return "candidate_syntax_error"
        return "test_collection_error"
    if exit_code == 5 or "collected 0 items" in combined or "no tests ran" in combined:
        return "test_collection_error"
    if "INTERNALERROR" in combined:
        return "executor_internal_error"
    if "ERROR" in combined and "test_solution.py" in combined:
        return "test_collection_error"
    if exit_code is not None and exit_code < 0:
        return "process_crash"
    return "test_assertion_failure"


class FixedKodCodePytestExecutor:
    def __init__(
        self,
        python_executable: Optional[str] = None,
        timeout: int = 30,
        cuda_visible_devices: Optional[str] = None,
    ):
        self.python_executable = python_executable or sys.executable
        self.timeout = timeout
        self.cuda_visible_devices = cuda_visible_devices

    def execute_detailed(
        self,
        func: str,
        test: str,
        timeout: Optional[int] = None,
    ) -> KodCodeExecutionResult:
        timeout_seconds = timeout if timeout is not None else self.timeout
        normalized_test, normalizations = add_official_kodcode_test_harness(test)
        with tempfile.TemporaryDirectory(prefix="kodcode_pytest_") as task_dir:
            solution_path = os.path.join(task_dir, "solution.py")
            test_path = os.path.join(task_dir, "test_solution.py")
            result_path = os.path.join(task_dir, "pytest_result.json")
            conftest_path = os.path.join(task_dir, "conftest.py")
            with open(solution_path, "w", encoding="utf-8") as handle:
                handle.write((func or "") + "\n")
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write(normalized_test)
            with open(conftest_path, "w", encoding="utf-8") as handle:
                handle.write(_PYTEST_RESULT_PLUGIN)

            env = os.environ.copy()
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            env["KODCODE_PYTEST_RESULT_FILE"] = result_path
            if self.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            command = [
                self.python_executable,
                "-m",
                "pytest",
                "--rootdir=.",
                "-q",
                "-p",
                "no:cacheprovider",
                "test_solution.py",
            ]
            process = subprocess.Popen(
                command,
                cwd=task_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                exit_code = process.returncode
                stdout = stdout or ""
                stderr = stderr or ""
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                stdout = stdout or ""
                stderr = stderr or ""
                timed_out = True
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.communicate()
                raise

            counts = {}
            if os.path.exists(result_path):
                try:
                    with open(result_path, encoding="utf-8") as handle:
                        counts = json.load(handle)
                except (OSError, ValueError):
                    counts = {}
            passed = not timed_out and exit_code == 0 and counts.get("collected", 0) > 0
            failure_category = _classify_pytest_failure(exit_code, stdout, stderr, timed_out)
            feedback = stdout
            if stderr:
                feedback += ("\n" if feedback else "") + stderr
            if timed_out:
                feedback += ("\n" if feedback else "") + f"Test execution timed out after {timeout_seconds}s."
            return KodCodeExecutionResult(
                passed=passed,
                feedback=feedback,
                collected=int(counts.get("collected", 0) or 0),
                passed_count=int(counts.get("passed", 0) or 0),
                failed=int(counts.get("failed", 0) or 0),
                errors=int(counts.get("errors", 0) or 0),
                skipped=int(counts.get("skipped", 0) or 0),
                timeout=timed_out,
                exit_code=exit_code,
                failure_category=failure_category,
                test_normalizations=normalizations,
                stdout=stdout,
                stderr=stderr,
            )

    def execute(self, func: str, tests: List[str], timeout: int = 30, verbose: bool = True) -> ExecuteResult:
        detailed = [self.execute_detailed(func, test, timeout=timeout) for test in tests]
        state = tuple(result.passed for result in detailed)
        feedback = "\n\n".join(result.feedback for result in detailed)
        return all(state), feedback, state


def create_kodcode_executor(
    mode: str,
    timeout: Optional[int] = None,
    cuda_visible_devices: Optional[str] = None,
):
    if mode not in KODCODE_ENV_MODES:
        raise ValueError(f"Unsupported KodCode environment mode: {mode}")
    return FixedKodCodePytestExecutor(
        timeout=resolve_kodcode_timeout(mode, timeout),
        cuda_visible_devices=cuda_visible_devices,
    )


def get_kodcode_executor_metadata(
    mode: str,
    timeout: Optional[int] = None,
    cuda_visible_devices: Optional[str] = None,
) -> dict:
    resolved_timeout = resolve_kodcode_timeout(mode, timeout)
    metadata = {
        "kodcode_env_mode": mode,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "kodcode_test_timeout_seconds": resolved_timeout,
        "kodcode_cuda_visible_devices": cuda_visible_devices,
    }
    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = None
    metadata.update({
        "kodcode_executor_backend": "pytest_subprocess",
        "pytest_version": pytest_version,
        "pytest_plugin_autoload": False,
        "test_normalizations": [
            "inject_official_solution_star_import",
            "guard_top_level_pytest_main",
        ],
    })
    return metadata
