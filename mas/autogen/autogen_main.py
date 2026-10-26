import logging
from typing import Optional

import uuid

from common.registry import registry
from common.utils.action_resolver import (
    invoke_with_iterative_search,
    resolve_intermediate_actions,
    strip_search_tags,
)
from common.utils.factual_qa import compact_qa_exchange, is_factual_qa_domain
from common.qa_config import QAConfig
from common.memory_interface import BaseCentralizedMemory
from mas.base import BaseMemoryMAS
from mas.memory_query import build_role_memory_query
from mas.autogen.prompts import prompt
from mas.agent import LLMAgent, DEFAULT_API_BASE, DEFAULT_MODEL_NAME
from mas.message import (
    MessageNode,
    MessageGraph,
)

@registry.register_mas("autogen")
class AutoGenMemoryMAS(BaseMemoryMAS):
    def __init__(
        self,
        llm_name_or_path: str,
        centralized_memory: Optional[BaseCentralizedMemory] = None,
        share_llm: bool = True,
        task_domain: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        model_name: str = DEFAULT_MODEL_NAME,
        **kwargs
    ):
        super().__init__(
            centralized_memory=centralized_memory,
            llm_name_or_path=llm_name_or_path,
            share_llm=share_llm,
            task_domain=task_domain
        )

        self.api_base = api_base
        self.model_name = model_name
        self.prompt_compaction_config = QAConfig.from_env()

        domain_prompt = prompt.get_domain_prompt(self.task_domain)

        self.assistant_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="assistant agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=domain_prompt.ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=domain_prompt.ASSISTANT_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )
        self.user_proxy_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="user proxy agent",
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=domain_prompt.USER_PROXY_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=domain_prompt.USER_PROXY_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.agents_list.extend([self.assistant_agent, self.user_proxy_agent])
        if self.centralized_memory is not None:
            self.centralized_memory.register_agents(self.agents_list)
            if hasattr(self.centralized_memory, "set_runtime_context"):
                self.centralized_memory.set_runtime_context(
                    model_name=self.model_name,
                    mas_architecture="autogen",
                )
            if hasattr(self.centralized_memory, "set_mas_spec"):
                self.centralized_memory.set_mas_spec({
                    "mas_type": "autogen",
                    "roles": [
                        {"name": self.assistant_agent.role, "duty": "propose the next task action", "order": 0},
                        {"name": self.user_proxy_agent.role, "duty": "validate the proposal and emit the final action", "order": 1},
                    ],
                    "edges": [[self.assistant_agent.role, self.user_proxy_agent.role]],
                })

    def generate(
        self,
        task_domain_instructions: list[str],
        user_inputs: list[str],
        generation_config: dict = None,
        function_names: list[str] = None,
        action_resolver=None,
        memory_task_descriptions: list[str] = None,
        memory_queries: list[str] = None,
        validator_task_descriptions: list[str] = None,
        validator_valid_action_constraints: list[str] = None,
    ) -> list[MessageGraph]:
        if generation_config is None:
            generation_config = {}
        elif hasattr(generation_config, 'to_dict'):
            generation_config = generation_config.to_dict()
        elif not isinstance(generation_config, dict):
            try:
                generation_config = vars(generation_config)
            except TypeError:
                generation_config = {}
        if function_names is None:
            function_names = [""] * len(user_inputs)
        memory_task_descriptions = list(memory_task_descriptions or user_inputs)
        memory_queries = list(memory_queries or memory_task_descriptions)
        validator_task_descriptions = list(validator_task_descriptions or user_inputs)
        validator_valid_action_constraints = list(
            validator_valid_action_constraints or [""] * len(user_inputs)
        )

        batch_message_graphs = [MessageGraph(state=input) for input in user_inputs]
        factual_qa = is_factual_qa_domain(self.task_domain)
        compaction = self.prompt_compaction_config
        max_search_iters = getattr(compaction, "qa_actor_max_search_iters", 4)

        # --- Assistant iteratively searches and refines the answer. ---
        assistant_user_inputs = {
            "task_description": list(user_inputs),
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.assistant_agent.role,
                    "propose the next task action",
                )
                for query in memory_queries
            ],
        }
        assistant_system_inputs = {"task_domain_instructions": task_domain_instructions}
        assistant_messages = invoke_with_iterative_search(
            self.assistant_agent, assistant_system_inputs, assistant_user_inputs, generation_config,
            resolver=action_resolver, max_search_iters=max_search_iters,
        )

        for ast_msg, mas_graph in zip(assistant_messages, batch_message_graphs):
            mas_graph.update_message_graph(ast_msg, self.assistant_agent.role, None)
            mas_graph.action = ast_msg.response

        assistant_responses = [msg.response for msg in assistant_messages]
        assistant_forwarded = [
            compact_qa_exchange(
                resp,
                max_total_chars=compaction.qa_actor_exchange_total_chars,
                max_info_chars=compaction.qa_actor_exchange_info_chars,
                max_doc_chars=compaction.qa_compaction_max_doc_chars,
                query_chars=compaction.qa_exchange_query_chars,
                answer_chars=compaction.qa_exchange_answer_chars,
                min_info_budget_chars=compaction.qa_exchange_min_info_budget_chars,
                max_info_blocks=compaction.qa_exchange_max_info_blocks,
                title_chars=compaction.qa_compaction_title_chars,
                doc_slack_chars=compaction.qa_compaction_doc_slack_chars,
                remaining_floor_chars=compaction.qa_compaction_remaining_floor_chars,
            )
            for resp in assistant_responses
        ] if factual_qa else assistant_responses

        # --- User proxy finalizes; search is not allowed. ---
        userproxy_user_inputs = {
            "task_description": validator_task_descriptions,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.user_proxy_agent.role,
                    "validate the proposal and emit the final action",
                    [assistant_output],
                )
                for query, assistant_output in zip(memory_queries, assistant_forwarded)
            ],
            "function_name": function_names,
            "assistant_output": assistant_forwarded,
            "valid_action_constraint": validator_valid_action_constraints,
        }
        userproxy_system_inputs = {"task_domain_instructions": task_domain_instructions}
        user_proxy_messages = self.user_proxy_agent.invoke(userproxy_system_inputs, userproxy_user_inputs, generation_config)

        for i, msg in enumerate(user_proxy_messages):
            final_response = strip_search_tags(msg.response or "") if factual_qa else (msg.response or "")
            msg.response = final_response
            batch_message_graphs[i].update_message_graph(msg, self.user_proxy_agent.role, [self.assistant_agent.role])
            batch_message_graphs[i].action = final_response

        return batch_message_graphs
