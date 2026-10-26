from datasets import DatasetDict, load_dataset
from typing import Dict
 
from common.registry import registry
from data.base_builder import BaseDataBuilder
from data.kodcode.env import KodCodeEnv


def load_kodcode_splits(
    split_seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> DatasetDict:
    all_dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
    all_correct_dataset = all_dataset["train"]

    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("KodCode split ratios must sum to 1.0")

    all_size = len(all_correct_dataset)
    test_size = int(all_size * test_ratio)
    split = all_correct_dataset.train_test_split(
        test_size=test_size,
        shuffle=True,
        seed=split_seed,
    )
    train_valid_dataset, test_dataset = split["train"], split["test"]

    valid_size = int(len(train_valid_dataset) * valid_ratio / (train_ratio + valid_ratio))
    split = train_valid_dataset.train_test_split(
        test_size=valid_size,
        shuffle=True,
        seed=split_seed,
    )
    train_dataset, valid_dataset = split["train"], split["test"]

    dataset_dict = DatasetDict()
    dataset_dict["train"] = train_dataset
    dataset_dict["valid"] = valid_dataset
    dataset_dict["test"] = test_dataset
    return dataset_dict

@registry.register_builder("kodcode")
class KodCodeBuilder(BaseDataBuilder):  
    
    def get_env_cls(self):
        return KodCodeEnv

    def _build_datasets(self) -> DatasetDict:
        train_ratio, valid_ratio, test_ratio = self.config.get("train_ratio"), self.config.get("valid_ratio"), self.config.get("test_ratio")
        split_seed = self.config.get("split_seed", 42)
        dataset_dict = load_kodcode_splits(
            split_seed=split_seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
        train_dataset = dataset_dict["train"]
        valid_dataset = dataset_dict["valid"]
        test_dataset = dataset_dict["test"]
        
        # preprocess
        train_dataset = train_dataset.map(self._preprocess).select_columns(self._keep_keys())
        valid_dataset = valid_dataset.map(self._preprocess).select_columns(self._keep_keys())
        test_dataset = test_dataset.map(self._preprocess).select_columns(self._keep_keys())
        
        # build dataset dict
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict

    @classmethod
    def _preprocess(cls, example: Dict):
        
        format_template = r""
        prompt_template = "{prompt}\n"

        question = example["question"].strip()
        solution = example["solution"].strip()

        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = solution

        text_output = {
            "prompt": processed_prompt,
            "completion": processed_label,
            "solution": processed_label,
            "test": example["test"].strip(),  
            "test_info": example["test_info"]
        }

        return text_output
    
    @classmethod
    def _keep_keys(cls):
        return ["prompt", "completion", "solution", "test", "test_info"]
