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
from mas.camel.prompts import prompt
from mas.agent import LLMAgent, DEFAULT_API_BASE, DEFAULT_MODEL_NAME
from mas.message import (
    MessageNode,
    MessageGraph,
)

@registry.register_mas("camel")
class CamelMemoryMAS(BaseMemoryMAS):
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
        self.domain_prompt = prompt.get_domain_prompt(self.task_domain)
        self.prompt_compaction_config = QAConfig.from_env()

        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents with API config."""
        self.userproxy_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="user proxy agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.USERPROXY_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.USERPROXY_USER_PROMPT_TEMPLATE,
            centralized_memory=self.centralized_memory
        )

        self.actor_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="actor agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.ACTOR_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.ACTOR_USER_PROMPT_TEMPLATE,
            centralized_memory=self.centralized_memory
        )

        self.critic_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="critic agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.CRITIC_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.CRITIC_USER_PROMPT_TEMPLATE,
            centralized_memory=self.centralized_memory
        )

        self.summarizer_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="summarizer agent",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.SUMMARIZER_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.SUMMARIZER_USER_PROMPT_TEMPLATE,
            centralized_memory=self.centralized_memory
        )

        self.agents_list.extend([
            self.userproxy_agent,
            self.actor_agent,
            self.critic_agent,
            self.summarizer_agent
        ])
        if self.centralized_memory is not None:
            self.centralized_memory.register_agents(self.agents_list)
            if hasattr(self.centralized_memory, "set_runtime_context"):
                self.centralized_memory.set_runtime_context(
                    model_name=self.model_name,
                    mas_architecture="camel",
                )
            if hasattr(self.centralized_memory, "set_mas_spec"):
                self.centralized_memory.set_mas_spec({
                    "mas_type": "camel",
                    "roles": [
                        {"name": self.userproxy_agent.role, "duty": "analyze task requirements", "order": 0},
                        {"name": self.actor_agent.role, "duty": "produce a candidate solution", "order": 1},
                        {"name": self.critic_agent.role, "duty": "critique the candidate solution", "order": 2},
                        {"name": self.summarizer_agent.role, "duty": "finalize the team output", "order": 3},
                    ],
                    "edges": [
                        [self.userproxy_agent.role, self.actor_agent.role],
                        [self.actor_agent.role, self.critic_agent.role],
                        [self.actor_agent.role, self.summarizer_agent.role],
                        [self.critic_agent.role, self.summarizer_agent.role],
                    ],
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
    ) -> list[MessageGraph]:
        # Handle both dict and GenerationConfig objects
        if generation_config is None:
            generation_config = {}
        elif hasattr(generation_config, 'to_dict'):
            # Convert GenerationConfig to dict
            generation_config = generation_config.to_dict()
        elif not isinstance(generation_config, dict):
            # Fallback: try to convert using vars or __dict__
            try:
                generation_config = vars(generation_config)
            except TypeError:
                generation_config = {}

        if function_names is None:
            function_names = [""] * len(user_inputs)
        memory_task_descriptions = list(memory_task_descriptions or user_inputs)
        memory_queries = list(memory_queries or memory_task_descriptions)

        batch_message_graphs = [MessageGraph(state=input) for input in user_inputs]
        factual_qa = is_factual_qa_domain(self.task_domain)
        compaction = self.prompt_compaction_config
        max_search_iters = getattr(compaction, "qa_actor_max_search_iters", 4)

        def _compact(responses, *, total_chars, info_chars):
            if not factual_qa:
                return responses
            return [
                compact_qa_exchange(
                    resp,
                    max_total_chars=total_chars,
                    max_info_chars=info_chars,
                    max_doc_chars=compaction.qa_compaction_max_doc_chars,
                    query_chars=compaction.qa_exchange_query_chars,
                    answer_chars=compaction.qa_exchange_answer_chars,
                    min_info_budget_chars=compaction.qa_exchange_min_info_budget_chars,
                    max_info_blocks=compaction.qa_exchange_max_info_blocks,
                    title_chars=compaction.qa_compaction_title_chars,
                    doc_slack_chars=compaction.qa_compaction_doc_slack_chars,
                    remaining_floor_chars=compaction.qa_compaction_remaining_floor_chars,
                )
                for resp in responses
            ]

        # --- user proxy drafts (iterative search) ---
        userproxy_user_inputs = {
            "task_description": list(user_inputs),
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.userproxy_agent.role,
                    "analyze task requirements",
                )
                for query in memory_queries
            ],
        }
        userproxy_system_inputs = {"task_domain_instructions": task_domain_instructions}
        userproxy_messages = invoke_with_iterative_search(
            self.userproxy_agent, userproxy_system_inputs, userproxy_user_inputs, generation_config,
            resolver=action_resolver, max_search_iters=max_search_iters,
        )
        for up_msg, mas_graph in zip(userproxy_messages, batch_message_graphs):
            mas_graph.update_message_graph(up_msg, self.userproxy_agent.role, None)
        userproxy_forwarded = _compact(
            [msg.response for msg in userproxy_messages],
            total_chars=compaction.qa_proxy_exchange_total_chars,
            info_chars=compaction.qa_proxy_exchange_info_chars,
        )

        # --- actor refines (iterative search) ---
        actor_user_inputs = {
            "task_description": list(user_inputs),
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.actor_agent.role,
                    "produce a candidate solution",
                    [upstream],
                )
                for query, upstream in zip(memory_queries, userproxy_forwarded)
            ],
            "function_name": function_names,
            "userproxy_output": userproxy_forwarded,
        }
        actor_system_inputs = {"task_domain_instructions": task_domain_instructions}
        actor_messages = invoke_with_iterative_search(
            self.actor_agent, actor_system_inputs, actor_user_inputs, generation_config,
            resolver=action_resolver, max_search_iters=max_search_iters,
        )
        for act_msg, mas_graph in zip(actor_messages, batch_message_graphs):
            mas_graph.update_message_graph(act_msg, self.actor_agent.role, [self.userproxy_agent.role])
        actor_forwarded = _compact(
            [msg.response for msg in actor_messages],
            total_chars=compaction.qa_actor_exchange_total_chars,
            info_chars=compaction.qa_actor_exchange_info_chars,
        )

        # --- Critic reviews actor with a single invoke; one-shot resolve is preserved. ---
        critic_user_inputs = {
            "task_description": user_inputs,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.critic_agent.role,
                    "critique the candidate solution",
                    [actor_output],
                )
                for query, actor_output in zip(memory_queries, actor_forwarded)
            ],
            "actor_output": actor_forwarded,
        }
        critic_system_inputs = {"task_domain_instructions": task_domain_instructions}
        critic_messages = self.critic_agent.invoke(critic_system_inputs, critic_user_inputs, generation_config)
        for cri_msg, mas_graph in zip(critic_messages, batch_message_graphs):
            mas_graph.update_message_graph(cri_msg, self.critic_agent.role, [self.actor_agent.role])
        critic_responses = [msg.response for msg in critic_messages]
        if action_resolver is not None:
            critic_responses = resolve_intermediate_actions(critic_responses, action_resolver)
            for message, resolved_response in zip(critic_messages, critic_responses):
                message.response = resolved_response
        critic_forwarded = _compact(
            critic_responses,
            total_chars=compaction.qa_critic_exchange_total_chars,
            info_chars=compaction.qa_critic_exchange_info_chars,
        )

        # --- Summarizer finalizes; search is not allowed. ---
        summarizer_user_inputs = {
            "task_description": user_inputs,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.summarizer_agent.role,
                    "finalize the team output",
                    [actor_output, critic_output],
                )
                for query, actor_output, critic_output in zip(
                    memory_queries, actor_forwarded, critic_forwarded
                )
            ],
            "function_name": function_names,
            "actor_output": actor_forwarded,
            "critic_output": critic_forwarded,
        }
        summarizer_system_inputs = {"task_domain_instructions": task_domain_instructions}
        summarizer_messages = self.summarizer_agent.invoke(summarizer_system_inputs, summarizer_user_inputs, generation_config)

        for sum_msg, mas_graph in zip(summarizer_messages, batch_message_graphs):
            mas_graph.update_message_graph(sum_msg, self.summarizer_agent.role, [self.actor_agent.role, self.critic_agent.role])
            final_response = strip_search_tags(sum_msg.response or "") if factual_qa else (sum_msg.response or "")
            sum_msg.response = final_response
            mas_graph.action = final_response

        return batch_message_graphs
