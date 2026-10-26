from typing import Any
import importlib
import re
import sys
import warnings
from pathlib import Path


from data.base_env import DynamicEnv
from data.pddl.prompts import PDDL_PRORMPTS

_AGENTBOARD_PDDLGYM_PARENT = Path(__file__).resolve().parents[1] / "agentboard_pddlgym"
_AGENTBOARD_PDDLGYM_PACKAGE = _AGENTBOARD_PDDLGYM_PARENT / "pddlgym"

PDDL_ENV_MODES = ("agentboard",)
PDDL_TASK_ORDER = ("gripper", "blockworld", "barman", "tyreworld")

PDDL_ENV_IDS = {
    "gripper": "PDDLEnvGripper-v0",
    "blockworld": "PDDLEnvBlockworld-v0",
    "barman": "PDDLEnvBarman-v0",
    "tyreworld": "PDDLEnvTyreworld-v0",
}

EXPECTED_PDDL_PROBLEM_COUNTS = {
    "gripper": 20,
    "blockworld": 10,
    "barman": 20,
    "tyreworld": 10,
}

INVALID_ACTION_OBSERVATION = (
    "The action is not valid and therefore takes no effect. "
    "Use `check valid actions` to list currently valid actions before trying another action."
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _clear_pddlgym_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "pddlgym" or module_name.startswith("pddlgym."):
            del sys.modules[module_name]


def _import_pddlgym():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Overriding environment .* already in registry.*",
            category=UserWarning,
        )
        return importlib.import_module("pddlgym")


def load_agentboard_pddlgym():
    """Import the AgentBoard-bundled pddlgym package and reject stock pddlgym."""
    if not (_AGENTBOARD_PDDLGYM_PACKAGE / "__init__.py").exists():
        raise RuntimeError(
            "AgentBoard bundled pddlgym is missing. Expected package at "
            f"{_AGENTBOARD_PDDLGYM_PACKAGE}"
        )

    current = sys.modules.get("pddlgym")
    if current is not None and getattr(current, "__file__", None):
        current_file = Path(current.__file__).resolve()
        if _is_relative_to(current_file, _AGENTBOARD_PDDLGYM_PACKAGE):
            return current
        _clear_pddlgym_modules()

    parent = str(_AGENTBOARD_PDDLGYM_PARENT)
    sys.path = [item for item in sys.path if item != parent]
    sys.path.insert(0, parent)
    module = _import_pddlgym()
    module_file = Path(module.__file__).resolve()
    if not _is_relative_to(module_file, _AGENTBOARD_PDDLGYM_PACKAGE):
        raise RuntimeError(
            "Loaded wrong pddlgym package. Expected AgentBoard package under "
            f"{_AGENTBOARD_PDDLGYM_PACKAGE}, got {module_file}"
        )
    return module


def get_agentboard_pddlgym_metadata() -> dict[str, Any]:
    pddlgym = load_agentboard_pddlgym()
    counts: dict[str, int] = {}
    problem_files: dict[str, list[str]] = {}
    for game_name, env_id in PDDL_ENV_IDS.items():
        env = pddlgym.make(env_id)
        problems = list(getattr(env, "problems", []))
        counts[game_name] = len(problems)
        problem_files[game_name] = [Path(problem.problem_fname).name for problem in problems]
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "pddl_env_mode": "agentboard",
        "pddl_env_backend": "agentboard_pddlgym",
        "pddlgym_source": str(Path(pddlgym.__file__).resolve()),
        "pddl_env_ids": dict(PDDL_ENV_IDS),
        "pddl_problem_counts": counts,
        "pddl_problem_files": problem_files,
        "expected_pddl_problem_counts": dict(EXPECTED_PDDL_PROBLEM_COUNTS),
        "pddl_task_order": list(PDDL_TASK_ORDER),
        "pddl_theoretical_max_score": 100.0,
        "pddl_initialization_failures_score_zero": False,
    }


def get_pddl_backend_metadata(pddl_env_mode: str = "agentboard") -> dict[str, Any]:
    if pddl_env_mode == "agentboard":
        return get_agentboard_pddlgym_metadata()
    raise ValueError(f"Unsupported pddl_env_mode={pddl_env_mode!r}; supported modes: {PDDL_ENV_MODES}")


def transform_few_shot_to_chatml(raw_text: str) -> str:
    text = raw_text.strip()

    user_tpl = "<|im_start|>user\n{}<|im_end|>"
    assistant_tpl = "<|im_start|>assistant\n{}<|im_end|>"

    pattern = re.compile(r'(Goal:|Observation:|Action:)\s*(.*?)(?=\n(?:Goal:|Observation:|Action:)|$)', re.DOTALL)
    matches = pattern.findall(text)

    transformed_blocks = []
    for prefix, content in matches:
        content = content.strip()
        if prefix == "Goal:":
            block = user_tpl.format(f"<observation>Goal: {content}</observation>")
        elif prefix == "Observation:":
            block = user_tpl.format(f"<observation>{content}</observation>")
        elif prefix == "Action:":
            block = assistant_tpl.format(f"<action>{content}</action>")
        else:
            continue
        transformed_blocks.append(block)

    return "\n".join(transformed_blocks)


PDDL_FEW_SHOTS_TEMPLATE = """
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.  
Please use these as references to guide your thinking and approach to the current task:

{few_shots}
---

Use the above examples and insights as a foundation, and now work on the following task:
{task_description}
"""


class PDDLEnv(DynamicEnv):
    def __init__(
        self,
        env_config: dict[str, Any],
    ):
        super().__init__(env_config)
        self.last_obs = None
        self.env = None
        self.game_name = None
        self.problem_index = None
        self.pddl_env_id = None
        self.problem_count = None
        self.problem_file = None
        self.pddlgym_source = None
        self.pddl_env_mode = "agentboard"
        self.pddl_env_backend = "agentboard_pddlgym"
        self._effective_action_cache_state = None
        self._effective_action_cache = None

    def set_env(self, configs: dict) -> tuple[str, str]:
        self.pddl_env_mode = configs.get("pddl_env_mode", "agentboard")
        if self.pddl_env_mode not in PDDL_ENV_MODES:
            raise ValueError(
                f"Unsupported pddl_env_mode={self.pddl_env_mode!r}; "
                f"supported modes: {PDDL_ENV_MODES}."
            )
        self.game_name = configs.get("game_name")
        raw_problem_index = configs.get("problem_index")
        if self.game_name is None or raw_problem_index is None:
            raise ValueError("Missing attributes in configs.")
        try:
            self.problem_index = int(raw_problem_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid problem_index: {raw_problem_index!r}") from exc

        self._set_agentboard_backend()
        self._reset()

        task_instruction = self._get_task_instruction()
        task_description = (
            f"{task_instruction}\n"
            f"**Task goal:** {self._get_goal()}"
        )
        few_shot = PDDL_PRORMPTS[self.game_name]["examples"][0]
        few_shot = transform_few_shot_to_chatml(few_shot)
        user_prompt = PDDL_FEW_SHOTS_TEMPLATE.format(
            few_shots=few_shot,
            task_description=task_description,
        )

        from data.pddl.builder import PDDL_SYSTEM_MESSAGE
        return PDDL_SYSTEM_MESSAGE, user_prompt

    def _get_task_instruction(self) -> str:
        return PDDL_PRORMPTS[self.game_name]["instruction"]

    def _set_agentboard_backend(self) -> None:
        if self.game_name not in PDDL_ENV_IDS:
            raise ValueError(
                f"Unsupported PDDL game '{self.game_name}'. "
                f"Supported games: {sorted(PDDL_ENV_IDS)}."
            )
        pddlgym = load_agentboard_pddlgym()
        self.pddlgym_source = str(Path(pddlgym.__file__).resolve())
        self.pddl_env_backend = "agentboard_pddlgym"
        self.pddl_env_id = PDDL_ENV_IDS[self.game_name]
        self.env = pddlgym.make(self.pddl_env_id)
        self.problem_count = len(getattr(self.env, "problems", []))
        expected_count = EXPECTED_PDDL_PROBLEM_COUNTS[self.game_name]
        if self.problem_count != expected_count:
            raise RuntimeError(
                f"AgentBoard PDDL problem count mismatch for {self.game_name}: "
                f"expected {expected_count}, got {self.problem_count} ({self.pddlgym_source})"
            )
        if self.problem_index < 0 or self.problem_index >= self.problem_count:
            raise IndexError(
                f"No AgentBoard PDDL problem for {self.game_name} problem_index={self.problem_index}; "
                f"available range is 0..{self.problem_count - 1}"
            )
        self.env.fix_problem_index(self.problem_index)

    def _reset(self):
        self._effective_action_cache_state = None
        self._effective_action_cache = None
        obs, debug_info = self.env.reset()
        self.problem_file = debug_info.get("problem_file") if isinstance(debug_info, dict) else None
        self.goal, self.init_obs, self.goal_literals = self._get_goal_and_obs(obs)
        self.last_obs = obs
        self.infos = {
            "goal_literal": obs.goal,
            "goal": self.goal,
            "pddl_env_mode": self.pddl_env_mode,
            "pddl_env_backend": self.pddl_env_backend,
            "pddl_env_id": self.pddl_env_id,
            "pddlgym_source": self.pddlgym_source,
            "problem_index": self.problem_index,
            "problem_count": self.problem_count,
            "problem_file": self.problem_file,
        }
        self.states = [self.init_obs]
        self.history = [("state", self.init_obs)]
        self.steps = 0

        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

        self.reward = 0
        self.done = False
        self.won = False

    def step(self, action: str) -> tuple[str, float, bool]:
        stripped = self._normalize_action_text(action)
        if stripped.startswith("think:") or stripped.startswith("think "):
            self._update_info(action, "(thinking step recorded)")
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done

        if stripped == "check valid actions":
            obs = "Valid actions are: " + ", ".join(self._get_action_space())
            self._update_info(action, obs)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done

        if stripped == "look around":
            obs = [self._literal_to_text(literal).capitalize() for literal in self.last_obs.literals]
            obs = " ".join(obs)
            self._update_info(action, obs)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done

        action_literal = self._text_to_action(action)

        if action_literal is not None:
            step_result = self.env.step(action_literal)
            if len(step_result) == 5:
                obs_temp, _reward, terminated, truncated, infos = step_result
                done = terminated or truncated
            else:
                obs_temp, _reward, done, infos = step_result
            reward = max(self.reward, self._constraint_satisfaction_metric(obs_temp.literals, self.goal_literals))

            _, obs, _ = self._get_goal_and_obs(obs_temp)
            self.last_obs = obs_temp
            self._update(action, obs, reward, done, infos)
            self.infos["action_is_valid"] = True
            return self._get_obs(), 1 if self.done else 0, self.done

        self._update_info(action, INVALID_ACTION_OBSERVATION)
        self.infos["action_is_valid"] = False
        return self._get_obs(), -1, self.done

    def feedback(self):
        return self.reward

    @classmethod
    def preprocess_action(cls, action: str) -> str:
        match = re.search(r"<action>(.*?)</action>", action, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in action.strip().splitlines() if line.strip()]
        return lines[-1] if lines else action.strip()

    @classmethod
    def compute_reward(cls, envs: list["PDDLEnv"], *args, **kwargs) -> list[float]:
        return [env.feedback() for env in envs]

    def _get_obs(self):
        return self.states[-1]

    def _get_goal(self):
        return self.goal

    def _get_effective_ground_actions(self):
        """Return the applicable ground actions in the current state."""
        if self.last_obs is None:
            return []
        if (
            self._effective_action_cache_state == self.last_obs
            and self._effective_action_cache is not None
        ):
            return list(self._effective_action_cache)

        ground_actions = self.env.action_space.all_ground_literals(self.last_obs)
        self._effective_action_cache_state = self.last_obs
        self._effective_action_cache = tuple(ground_actions)
        return list(ground_actions)

    def _get_action_space(self):
        actions = [
            self._literal_to_text(literal)
            for literal in self._get_effective_ground_actions()
        ]
        actions.append("check valid actions")
        if self.game_name in {"barman", "tyreworld"}:
            actions.append("look around")
        return actions

    def _update(self, action, obs, reward, done, infos):
        for k, v in infos.items():
            self.infos[k] = v

        goal_literals = self.infos["goal_literal"].literals
        obs_literals = self.last_obs.literals
        self.won = True
        for literal in goal_literals:
            if literal not in obs_literals:
                self.won = False
                break

        self.steps += 1
        self.reward = reward
        self.done = done

        if self.won:
            obs += " The goal is satisfied."

        self.history.append(("action", action))
        self.history.append(("reward", reward))
        self.history.append(("state", obs))
        self.states.append(obs)

        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

    def _update_info(self, action, info):
        self.history.append(("action", action))
        self.history.append(("reward", self.reward))
        self.history.append(("state", info))
        self.states.append(info)

        self.steps += 1
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]

    def _constraint_satisfaction_metric(self, obs_literals, goal_literals):
        satisfied = 0
        total = 0
        for literal in goal_literals:
            if literal in obs_literals:
                satisfied += 1
            total += 1
        return satisfied / total if total else 0.0

    def _get_goal_and_obs(self, obs):
        goal = [self._literal_to_text(literal) for literal in obs.goal.literals]
        goal.sort()
        goal_text = "The goal is to satisfy the following conditions: " + ", ".join(goal)

        state = obs.literals
        state_text = [self._literal_to_text(literal).capitalize() for literal in state]
        state_text.sort()
        state_text = " ".join(state_text)
        return goal_text, state_text, obs.goal.literals

    def _literal_to_text(self, literal):
        predicate_name = literal.predicate.name
        objects = literal.variables
        game_map = _GAME_PREDICATE_MAPS.get(self.game_name, {})
        predicate_format = game_map.get(predicate_name)
        objects_name = [str(obj.name) for obj in objects]
        if predicate_format is not None:
            try:
                return predicate_format.format(*objects_name)
            except IndexError:
                pass
        return predicate_name + " " + " ".join(objects_name)

    @staticmethod
    def _normalize_action_text(text: str) -> str:
        tag_match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
        if tag_match:
            text = tag_match.group(1)
        text = text.lower().strip()
        text = re.sub(r"^action\s*:\s*", "", text)
        text = text.strip("`\"' ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,;:])", r"\1", text)
        return text.rstrip(" .,;")

    def _valid_action_text_map(self, all_valid_actions):
        text_to_action = {}
        for literal in all_valid_actions:
            rendered = self._normalize_action_text(self._literal_to_text(literal))
            if rendered:
                text_to_action[rendered] = literal
            compact = " ".join(
                [literal.predicate.name, *[str(obj.name) for obj in literal.variables]]
            )
            text_to_action[self._normalize_action_text(compact)] = literal
        return text_to_action

    def _text_to_action(self, text):
        load_agentboard_pddlgym()

        normalized = self._normalize_action_text(text)
        valid_actions = self._get_effective_ground_actions()
        return self._valid_action_text_map(valid_actions).get(normalized)

    def resolve_action_text(self, text: str) -> str | None:
        """Resolve an exact current-state action and return its canonical rendering."""
        literal = self._text_to_action(text)
        return self._literal_to_text(literal) if literal is not None else None


# Per-game predicate rendering — previously a single global dict with
# duplicated keys that silently overrode each other across games.
_BLOCKS_PREDICATES = {
    "on": "{} is on {}.",
    "clear": "{} is clear.",
    "arm-empty": "Your arm is empty.",
    "holding": "You are holding {}.",
    "on-table": "{} is on the table.",
    "putdown": "Putdown {}.",
    "stack": "Stack {} on {}.",
    "pickup": "Pickup {}.",
    "unstack": "Unstack {} from {}.",
}

_GRIPPER_PREDICATES = {
    "ball": "{} is a ball. ",
    "gripper": "{} is a gripper. ",
    "at-robby": "Robby is at {}. ",
    "at": "{} is at {}. ",
    "free": "{} is free. ",
    "carry": "{} is carrying {}. ",
    "move": "Move from {} to {}. ",
    "pick": "Pick up {} at {} with arm {}. ",
}

_TYREWORLD_PREDICATES = {
    "vehicle-at": "The vehicle is at {}.",
    "changetire": "Change tire at {}",
    "spare-in": "The spare tire is in {}.",
    "road": "There is a road from {} to {}.",
    "movecar": "Move the car to {}.",
    "isopen": "{} is open.",
    "closed": "{} is closed.",
    "have": "You have {}.",
    "in": "{} is in {}.",
    "loose": "The nut {} on the hub {} is loose.",
    "tight": "The nut {} on the hub {} is tight.",
    "unlocked": "{} is unlocked.",
    "on-ground": "Hub {} is on the ground.",
    "not-on-ground": "Hub {} is not on the ground.",
    "inflated": "Wheel {} is inflated.",
    "not-inflated": "Wheel {} is not inflated.",
    "fastened": "Hub {} is fastened.",
    "unfastened": "Hub {} is unfastened.",
    "free": "Hub {} is free.",
    "intact": "Wheel {} is intact.",
    "open": "Open {}.",
    "close": "Close {}.",
    "fetch": "Fetch {} from {}.",
    "put-away": "Put-away {} in {}.",
    "loosen": "Loosen the nut {} on the hub {}.",
    "tighten": "Tighten the nut {} on the hub {}.",
    "jack-up": "jack-up the hub {}.",
    "jack-down": "Jack-down the hub {}.",
    "undo": "Undo the fastening of the nut {} on the hub {}.",
    "do-up": "Do-up the nut {} on the hub {}.",
    "remove-wheel": "Remove-wheel {} from the hub {}.",
    "put-on-wheel": "put-on-wheel {} on the hub {}.",
    "inflate": "Inflate the wheel {}.",
    "holding": "You have {}.",
}

_BARMAN_PREDICATES = {
    "ontable": "{} is on the table. ",
    "holding": "{} hand is holding {}. ",
    "handempty": "{} hand is empty. ",
    "empty": "{} is empty. ",
    "contains": "{} contains {}. ",
    "clean": "{} is clean. ",
    "used": "{} has previously been used for {}. ",
    "dispenses": "{} dispenses {}. ",
    "shaker-empty-level": "{} is at empty level {}. ",
    "shaker-level": "{} is at level {}. ",
    "next": "level {} is next to level {}. ",
    "unshaked": "{} is unshaked. ",
    "shaked": "{} is shaked. ",
    "cocktail-part1": "{} part1 ingredient is {}. ",
    "cocktail-part2": "{} part2 ingredient is {}. ",
    "grasp": "{} grasp {}. ",
    "leave": "{} leave {}. ",
    "fill-shot": "fill-shot glass {} with {} with {} and {} holding {}. ",
    "refill-shot": "refill-shot {} with {} with {} and {} holding {}. ",
    "empty-shot": "use hand {} to empty-shot glass {} with beverage {}. ",
    "clean-shot": "clean-shot glass {} with {} with hand {} holding shot glass and {}. ",
    "pour-shot-to-clean-shaker": "pour-shot-to-clean-shaker from a shot glass {} with {} to a clean shaker {} with hand {} from level {} to level {}",
    "pour-shot-to-used-shaker": "pour-shot-to-used-shaker from a shot glass {} with {} to a used shaker {} with hand {} from level {} to level {}",
    "empty-shaker": "use hand {} to empty-shaker {} with beverage {} from level {} to level {}.",
    "clean-shaker": "use hand {} and hand {} to clean-shaker {}",
    "shake": "shake a cocktail {} with ingredient {} and ingredient {} in a shaker {} with hand {} and hand {}",
    "pour-shaker-to-shot": "pour-shaker-to-shot beverage {} to shot glass {} with hand {} from shaker {} from level {} to level {}",
}

_HANOI_PREDICATES = {
    "clear": "The {} is clear.",
    "move": "Move {} to {}.",
    "smaller": "{} is smaller than {}.",
    "on": "{} is on {}.",
}

_GAME_PREDICATE_MAPS = {
    "blocks": _BLOCKS_PREDICATES,
    "blockworld": _BLOCKS_PREDICATES,
    "gripper": _GRIPPER_PREDICATES,
    "barman": _BARMAN_PREDICATES,
    "tyreworld": _TYREWORLD_PREDICATES,
    "hanoi": _HANOI_PREDICATES,
}
