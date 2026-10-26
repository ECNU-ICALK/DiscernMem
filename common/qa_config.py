"""
Factual-QA retrieval and prompt-compaction settings.

Shared by the TriviaQA/PopQA environments, the retriever, and the MAS hosts.
The defaults are the values used in all reported experiments.
"""
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class QAConfig:
    # --- Retrieval endpoint ---
    qa_search_url: str = "http://127.0.0.1:8000/retrieve"
    qa_search_topk: int = 3
    qa_search_timeout_seconds: float = 30.0

    # --- Retrieved-document compaction ---
    qa_compaction_max_total_chars: int = 1800
    qa_compaction_max_doc_chars: int = 520
    qa_compaction_title_chars: int = 80
    qa_compaction_doc_slack_chars: int = 40
    qa_compaction_remaining_floor_chars: int = 120
    qa_compaction_max_chunks_per_source: int = 1

    # --- Per-exchange compaction inside MAS prompts ---
    qa_exchange_query_chars: int = 120
    qa_exchange_answer_chars: int = 240
    qa_exchange_min_info_budget_chars: int = 220
    qa_exchange_max_info_blocks: int = 2
    qa_actor_exchange_total_chars: int = 1600
    qa_actor_exchange_info_chars: int = 900
    qa_actor_max_search_iters: int = 4  # Search-R1 interaction budget B=4
    qa_critic_exchange_total_chars: int = 900
    qa_critic_exchange_info_chars: int = 500
    qa_proxy_exchange_total_chars: int = 1400
    qa_proxy_exchange_info_chars: int = 800

    qa_search_profiles: dict = field(default_factory=lambda: {
        "triviaqa": {
            "search_url": "http://127.0.0.1:8000/retrieve",
        },
        "popqa": {
            "search_url": "http://127.0.0.1:8000/retrieve",
        },
    })

    def factual_qa_retriever_kwargs_for_domain(self, task_domain: str | None = None) -> dict:
        profile = self.qa_search_profiles.get(task_domain, {}) if task_domain else {}
        return {
            "search_url": profile.get("search_url", self.qa_search_url),
            "topk": profile.get("topk", self.qa_search_topk),
            "timeout_seconds": profile.get("timeout_seconds", self.qa_search_timeout_seconds),
            "max_doc_chars": self.qa_compaction_max_doc_chars,
            "max_total_chars": self.qa_compaction_max_total_chars,
            "title_chars": self.qa_compaction_title_chars,
            "doc_slack_chars": self.qa_compaction_doc_slack_chars,
            "remaining_floor_chars": self.qa_compaction_remaining_floor_chars,
            "max_chunks_per_source": self.qa_compaction_max_chunks_per_source,
        }

    @classmethod
    def from_env(cls, env_path: str = None) -> "QAConfig":
        """Load ``.env`` into the process environment and return the defaults."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        return cls()
