import re

from data.base_env import DynamicEnv
from data.popqa.retriever import PopQARetriever
from common.qa_config import QAConfig

class PopQAEnv(DynamicEnv):
   
    def __init__(self, configs: dict):
        super().__init__(configs)
        defaults = QAConfig.from_env()
        search_url = configs.get("search_url", defaults.qa_search_url)
        topk = configs.get("topk", defaults.qa_search_topk)
        self.explorer = PopQARetriever(
            search_url=search_url,
            topk=topk,
            timeout_seconds=configs.get("timeout_seconds", defaults.qa_search_timeout_seconds),
            max_doc_chars=configs.get("max_doc_chars", defaults.qa_compaction_max_doc_chars),
            max_total_chars=configs.get("max_total_chars", defaults.qa_compaction_max_total_chars),
            title_chars=configs.get("title_chars", defaults.qa_compaction_title_chars),
            doc_slack_chars=configs.get("doc_slack_chars", defaults.qa_compaction_doc_slack_chars),
            remaining_floor_chars=configs.get("remaining_floor_chars", defaults.qa_compaction_remaining_floor_chars),
            max_chunks_per_source=configs.get(
                "max_chunks_per_source",
                defaults.qa_compaction_max_chunks_per_source,
            ),
        )

    def set_env(self, task_config: dict) -> None:
        if task_config.get('answer') is None:
            raise ValueError('Please provide the answer for the task')
        if task_config.get("prompt") is None:
            raise ValueError('Please provide the prompt for the task')

        self.task_config = task_config
        
        self._reset()

        from data.popqa.builder import POPQA_SYSTEM_PROMPT
        return POPQA_SYSTEM_PROMPT, task_config["prompt"]
    
    def _reset(self):
        self.done = False
        self.reward = 0.0

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.preprocess_action(action)
        action_type, action_content = self._process_action(action)
        observation = None
        
        if action_type == "search":
            try:
                observation = self.explorer.batch_search([action_content])[0]
            except Exception as e:
                observation = f'Cannot find corresponding pages.'     
            self.done = False
            self.reward = 0.0

        elif action_type == "answer":
            observation = ""
            self.done = True
            self.reward = 1.0 if self._check_answer(action_content, self.task_config["answer"]) else 0.0
        else:
            observation = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
            self.done = False
            self.reward = 0.0

        return observation, self.reward, self.done
    
    @classmethod
    def preprocess_action(cls, action: str) -> str:
        search_end = action.find("</search>")
        answer_end = action.find("</answer>")

        if search_end != -1 and (answer_end == -1 or search_end < answer_end):
            return action[: search_end + len("</search>")]
        if answer_end != -1:
            return action[: answer_end + len("</answer>")]
        return action

    @classmethod
    def _process_action(cls, action: str):
        action = action.strip()

        if "<search>" in action and "</search>" in action:
            start = action.index("<search>") + len("<search>")
            end = action.index("</search>")
            content = action[start:end].strip()
            return "search", content

        if "<answer>" in action and "</answer>" in action:
            start = action.index("<answer>") + len("<answer>")
            end = action.index("</answer>")
            content = action[start:end].strip()
            return "answer", content

        return "think", action
    
    def _check_answer(self, answer: str, ground_truth: list[str]):
        answer = answer.lower()
        for gt in ground_truth:
            if gt.lower() in answer:
                return True
        
        return False
    
    def feedback(self) -> float:
        return self.reward

    @classmethod
    def compute_reward(cls, completions: list[str], envs: list['PopQAEnv'], **kwargs) -> list[float]:
        scores = []
        for completion, env in zip(completions, envs):
            solution = env.task_config['answer']  

            matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            
            if not matches:  
                scores.append(0.0)
                continue

            extracted = matches[-1].strip()

            correct = False
            for s in solution:
                if s.lower() in extracted.lower():
                    correct = True
                    break

            if correct:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return scores
