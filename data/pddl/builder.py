from datasets import DatasetDict, Dataset
import json

from common.registry import registry
from data.base_builder import BaseDataBuilder
from data.pddl.env.pddl_env import PDDLEnv

PDDL_SYSTEM_MESSAGE = (
    "You are a PDDL planning agent. Output exactly one next environment action "
    "inside one <action>...</action> tag. Do not output a multi-step plan. "
    "If a <valid_action_constraint> block is present, copy exactly one listed "
    "action. If no valid-action constraint is present and the current observation "
    "says the previous action was invalid, output <action>check valid actions</action>."
)

PDDL_TASK_NAMES = ["gripper", "blockworld", "barman", "tyreworld"]
PDDL_NUM_PROBLEMS = {
    "barman": 20,
    "blockworld": 10,
    "gripper": 20,
    "tyreworld": 10,
}


def _load_annotations_by_game(label_path: str) -> dict[str, list[dict]]:
    annotations: dict[str, list[dict]] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line.strip())
            game_name = record.get("additional_info", {}).get("subtask")
            if game_name:
                annotations.setdefault(game_name, []).append(record)
    return annotations


def get_all_environment_configs(game_names: list[str], label_path: str):
    requested_games = set(game_names)
    env_configs = []
    annotations = _load_annotations_by_game(label_path)

    for game_name in game_names:
        if game_name not in requested_games:
            continue
        num_problems = PDDL_NUM_PROBLEMS.get(game_name, 0)
        game_annotations = annotations.get(game_name, [])
        for problem_index in range(num_problems):
            record = (
                game_annotations[problem_index]
                if problem_index < len(game_annotations)
                else {}
            )
            env_configs.append({
                "game_name": game_name,
                "problem_index": problem_index,
                "difficulty": record.get("difficulty", "unknown"),
                "goal": record.get("goal"),
                "subgoals": record.get("subgoals", []),
                "id": record.get("id", f"{game_name}_{problem_index}"),
            })

    return env_configs

@registry.register_builder("pddl")
class PDDLBuilder(BaseDataBuilder): 
    
    def get_env_cls(self):
        return PDDLEnv
 
    def _build_datasets(self) -> DatasetDict:        
        
        pddl_tasks: list[dict] = get_all_environment_configs(
            PDDL_TASK_NAMES, "data/pddl/test.jsonl"
        )
        test_dataset = Dataset.from_list(pddl_tasks)

        empty_dataset = Dataset.from_dict({k: [] for k in test_dataset.column_names})

        dataset_dict = DatasetDict({
            "train": empty_dataset,
            "valid": empty_dataset,
            "test": test_dataset
        })

        return dataset_dict
