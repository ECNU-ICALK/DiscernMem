from common.utils.code_utils import (
    collect_python_program,
    create_kodcode_executor,
    prepare_kodcode_solution,
    resolve_kodcode_timeout,
)
from data.base_env import StaticEnv

KODCODE_INSTRUCTIONS = "Write a Python function according to the given task"


def _resolve_function_name(test_info) -> str:
    if not test_info or not isinstance(test_info, list):
        return ""
    first = test_info[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("function_name", "")).strip()

class KodCodeEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)
        self.kodcode_env_mode = config.get("kodcode_env_mode", "pytest")
        self.test_timeout_seconds = resolve_kodcode_timeout(
            self.kodcode_env_mode,
            config.get("kodcode_test_timeout_seconds"),
        )
        self.cuda_visible_devices = config.get("kodcode_cuda_visible_devices")
        self.feedback_detail = None

    def set_env(self, task_config: dict) -> tuple[str, str]:

        self.prompt = task_config.get("prompt")
        self.test = task_config.get("test")
        self.test_info = task_config.get("test_info")

        return KODCODE_INSTRUCTIONS, self.prompt

    def step(self, action: str) -> tuple[str, float, bool]:

        py_executor = create_kodcode_executor(
            self.kodcode_env_mode,
            timeout=self.test_timeout_seconds,
            cuda_visible_devices=self.cuda_visible_devices,
        )
        collected_answer = collect_python_program((action or "").strip())
        function_name = _resolve_function_name(self.test_info)
        renamed_answer, solution_normalizations = prepare_kodcode_solution(
            collected_answer,
            function_name,
            self.kodcode_env_mode,
        )

        execution_detail = py_executor.execute_detailed(
            renamed_answer,
            self.test,
            timeout=self.test_timeout_seconds,
        )
        is_passing = execution_detail.passed
        feedback = execution_detail.feedback
        results = (execution_detail.passed,)
        score = 1.0 if results and all(results) else 0.0

        if score >= 1.0:
            summary = "All tests passed."
        elif score > 0:
            summary = f"Partial pass: {score:.1%}"
        else:
            summary = "All tests failed."

        self.feedback_detail = {
            "summary": summary,
            "score": score,
            "full_feedback": feedback,
            "test_passed": is_passing,
            "test_results": results[0] if results else False,
            "kodcode_env_mode": self.kodcode_env_mode,
            "executor_detail": execution_detail.to_dict(),
            "solution_normalizations": solution_normalizations,
            "kodcode_cuda_visible_devices": self.cuda_visible_devices,
        }

        self.reward = score
        if self.reward == 1.0:
            self.observation = "True!"
            self.done = True
        else:
            self.observation = "False!"
            self.done = False

        return self.observation, self.reward, self.done

    @classmethod
    def compute_reward(cls, completions: list[str], test: list[str], test_info: list, **kwargs) -> list[float]:

        mode = kwargs.get("kodcode_env_mode", "pytest")
        timeout = resolve_kodcode_timeout(mode, kwargs.get("kodcode_test_timeout_seconds"))
        py_executor = create_kodcode_executor(
            mode,
            timeout=timeout,
            cuda_visible_devices=kwargs.get("kodcode_cuda_visible_devices"),
        )
        scores = []
        for completion, t, tf in zip(completions, test, test_info):
            collected_answer = collect_python_program((completion or "").strip())
            function_name = _resolve_function_name(tf)
            renamed_answer, _ = prepare_kodcode_solution(collected_answer, function_name, mode)
            result = py_executor.execute_detailed(renamed_answer, t, timeout=timeout)
            score = 1.0 if result.passed else 0.0
            scores.append(score)

        return scores
