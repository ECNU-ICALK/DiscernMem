from typing import List
import requests

from common.utils.factual_qa import compact_retrieval_result
from common.qa_config import QAConfig

class Retriever:

    def __init__(
        self,
        search_url: str | None = None,
        topk: int | None = None,
        timeout_seconds: float | None = None,
        max_doc_chars: int | None = None,
        max_total_chars: int | None = None,
        title_chars: int | None = None,
        doc_slack_chars: int | None = None,
        remaining_floor_chars: int | None = None,
        max_chunks_per_source: int | None = None,
    ):
        defaults = QAConfig.from_env()
        self.config = {
            "search_url": search_url or defaults.qa_search_url,
            "topk": topk if topk is not None else defaults.qa_search_topk,
            "timeout_seconds": timeout_seconds if timeout_seconds is not None else defaults.qa_search_timeout_seconds,
        }
        self.max_doc_chars = max_doc_chars if max_doc_chars is not None else defaults.qa_compaction_max_doc_chars
        self.max_total_chars = max_total_chars if max_total_chars is not None else defaults.qa_compaction_max_total_chars
        self.title_chars = title_chars if title_chars is not None else defaults.qa_compaction_title_chars
        self.doc_slack_chars = doc_slack_chars if doc_slack_chars is not None else defaults.qa_compaction_doc_slack_chars
        self.remaining_floor_chars = (
            remaining_floor_chars if remaining_floor_chars is not None else defaults.qa_compaction_remaining_floor_chars
        )
        self.max_chunks_per_source = (
            max_chunks_per_source
            if max_chunks_per_source is not None
            else defaults.qa_compaction_max_chunks_per_source
        )

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        formatted_results = [self._passages2string(result) for result in results]
        for i, (query, result) in enumerate(zip(queries, formatted_results)):
            print(f"[Retriever] Query {i+1}: {query}")
            print(f"[Retriever] Formatted result:\n{result}")
        return formatted_results

    def _batch_search(self, queries):
        print(f"[Retriever] Searching for queries: {queries}")
        
        payload = {
            "queries": queries,
            "topk": self.config["topk"],
            "return_scores": True
        }
        
        response = requests.post(
            self.config["search_url"],
            json=payload,
            timeout=self.config["timeout_seconds"],
        )
        response.raise_for_status()
        data = response.json()
        print(f"[Retriever] Raw response: {data}")
        return data

    def _passages2string(self, retrieval_result):
        return compact_retrieval_result(
            retrieval_result,
            max_total_chars=self.max_total_chars,
            max_doc_chars=self.max_doc_chars,
            title_chars=self.title_chars,
            doc_slack_chars=self.doc_slack_chars,
            remaining_floor_chars=self.remaining_floor_chars,
            max_chunks_per_source=self.max_chunks_per_source,
        )
