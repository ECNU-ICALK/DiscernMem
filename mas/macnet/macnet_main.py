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
from mas.macnet.prompts import prompt
from mas.agent import LLMAgent, DEFAULT_API_BASE, DEFAULT_MODEL_NAME
from mas.message import (
    MessageGraph,
)

@registry.register_mas("macnet")
class MacNetMemoryMAS(BaseMemoryMAS):
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

        self.actor_agent_1 = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="actor agent 1",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.ACTOR_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.ACTOR_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )
        self.actor_agent_2 = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="actor agent 2",
            id=str(uuid.uuid4()),
            topology_node_id=0,
            system_prompt_template=self.domain_prompt.ACTOR_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.ACTOR_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.critic_agent_1 = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="critic agent 1",
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=self.domain_prompt.CRITIC_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.CRITIC_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.critic_agent_2 = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="critic agent 2",
            id=str(uuid.uuid4()),
            topology_node_id=1,
            system_prompt_template=self.domain_prompt.CRITIC_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.CRITIC_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.summarizer_agent = LLMAgent(
            api_base=self.api_base,
            model_name=self.model_name,
            role="summarizer agent",
            id=str(uuid.uuid4()),
            topology_node_id=2,
            system_prompt_template=self.domain_prompt.SUMMARIZER_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=self.domain_prompt.SUMMARIZER_USER_PROMPT_TEMPLATE,
            centralized_memory=centralized_memory
        )

        self.agents_list.extend([
            self.actor_agent_1,
            self.actor_agent_2,
            self.critic_agent_1,
            self.critic_agent_2,
            self.summarizer_agent
        ])
        if self.centralized_memory is not None:
            self.centralized_memory.register_agents(self.agents_list)
            if hasattr(self.centralized_memory, "set_runtime_context"):
                self.centralized_memory.set_runtime_context(
                    model_name=self.model_name,
                    mas_architecture="macnet",
                )
            if hasattr(self.centralized_memory, "set_mas_spec"):
                self.centralized_memory.set_mas_spec({
                    "mas_type": "macnet",
                    "roles": [
                        {"name": self.actor_agent_1.role, "duty": "produce candidate solution branch one", "order": 0},
                        {"name": self.actor_agent_2.role, "duty": "produce candidate solution branch two", "order": 0},
                        {"name": self.critic_agent_1.role, "duty": "critique actor branch one", "order": 1},
                        {"name": self.critic_agent_2.role, "duty": "critique actor branch two", "order": 1},
                        {"name": self.summarizer_agent.role, "duty": "aggregate and finalize the team output", "order": 2},
                    ],
                    "edges": [
                        [self.actor_agent_1.role, self.critic_agent_1.role],
                        [self.actor_agent_2.role, self.critic_agent_2.role],
                        [self.critic_agent_1.role, self.summarizer_agent.role],
                        [self.critic_agent_2.role, self.summarizer_agent.role],
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

        batch_message_graphs = [MessageGraph(state=input) for input in user_inputs]
        factual_qa = is_factual_qa_domain(self.task_domain)
        compaction = self.prompt_compaction_config
        max_search_iters = getattr(compaction, "qa_actor_max_search_iters", 4)

        # Use role-specific temperatures.
        actor_config = {**generation_config, "temperature": 0.1}
        critic_config = {**generation_config, "temperature": 0.0}
        summarizer_config = {**generation_config, "temperature": 0.0}

        def _compact_actor(responses):
            if not factual_qa:
                return responses
            return [
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
                for resp in responses
            ]

        def _compact_critic(responses):
            if not factual_qa:
                return responses
            return [
                compact_qa_exchange(
                    resp,
                    max_total_chars=compaction.qa_critic_exchange_total_chars,
                    max_info_chars=compaction.qa_critic_exchange_info_chars,
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

        # --- actor 1 outputs (iterative <search> → <information> loop) ---
        actor1_user_inputs = {
            "task_description": list(user_inputs),
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.actor_agent_1.role,
                    "produce candidate solution branch one",
                )
                for query in memory_queries
            ],
            "function_name": function_names,
        }
        actor1_system_inputs = {"task_domain_instructions": task_domain_instructions}
        actor1_messages = invoke_with_iterative_search(
            self.actor_agent_1, actor1_system_inputs, actor1_user_inputs, actor_config,
            resolver=action_resolver, max_search_iters=max_search_iters,
        )
        for act_msg, mas_graph in zip(actor1_messages, batch_message_graphs):
            mas_graph.update_message_graph(act_msg, self.actor_agent_1.role, None)
        actor1_forwarded = _compact_actor([msg.response for msg in actor1_messages])

        # --- Actor 2 outputs an independent parallel draft with iterative search. ---
        actor2_user_inputs = {
            "task_description": list(user_inputs),
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.actor_agent_2.role,
                    "produce candidate solution branch two",
                )
                for query in memory_queries
            ],
            "function_name": function_names,
        }
        actor2_system_inputs = {"task_domain_instructions": task_domain_instructions}
        actor2_messages = invoke_with_iterative_search(
            self.actor_agent_2, actor2_system_inputs, actor2_user_inputs, actor_config,
            resolver=action_resolver, max_search_iters=max_search_iters,
        )
        for act_msg, mas_graph in zip(actor2_messages, batch_message_graphs):
            mas_graph.update_message_graph(act_msg, self.actor_agent_2.role, None)
        actor2_forwarded = _compact_actor([msg.response for msg in actor2_messages])

        # --- Critic 1 reviews actor 1 with a single invoke; optional one-shot resolve is preserved. ---
        critic1_user_inputs = {
            "task_description": user_inputs,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.critic_agent_1.role,
                    "critique actor branch one",
                    [actor_output],
                )
                for query, actor_output in zip(memory_queries, actor1_forwarded)
            ],
            "actor_output": actor1_forwarded,
        }
        critic1_system_inputs = {"task_domain_instructions": task_domain_instructions}
        critic1_messages = self.critic_agent_1.invoke(critic1_system_inputs, critic1_user_inputs, critic_config)
        for cri_msg, mas_graph in zip(critic1_messages, batch_message_graphs):
            mas_graph.update_message_graph(cri_msg, self.critic_agent_1.role, [self.actor_agent_1.role])
        critic1_responses = [msg.response for msg in critic1_messages]
        if action_resolver is not None:
            critic1_responses = resolve_intermediate_actions(critic1_responses, action_resolver)
            for message, resolved_response in zip(critic1_messages, critic1_responses):
                message.response = resolved_response
        critic1_forwarded = _compact_critic(critic1_responses)

        # --- critic 2 reviews actor 2 ---
        critic2_user_inputs = {
            "task_description": user_inputs,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.critic_agent_2.role,
                    "critique actor branch two",
                    [actor_output],
                )
                for query, actor_output in zip(memory_queries, actor2_forwarded)
            ],
            "actor_output": actor2_forwarded,
        }
        critic2_system_inputs = {"task_domain_instructions": task_domain_instructions}
        critic2_messages = self.critic_agent_2.invoke(critic2_system_inputs, critic2_user_inputs, critic_config)
        for cri_msg, mas_graph in zip(critic2_messages, batch_message_graphs):
            mas_graph.update_message_graph(cri_msg, self.critic_agent_2.role, [self.actor_agent_2.role])
        critic2_responses = [msg.response for msg in critic2_messages]
        if action_resolver is not None:
            critic2_responses = resolve_intermediate_actions(critic2_responses, action_resolver)
            for message, resolved_response in zip(critic2_messages, critic2_responses):
                message.response = resolved_response
        critic2_forwarded = _compact_critic(critic2_responses)

        # --- Summarizer finalizes; search is blocked by prompts and output sanitization. ---
        feedback_page1 = [
            self.domain_prompt.FEEDBACK_PAGE.format(actor_output=act_r, critic_output=cri_r)
            for act_r, cri_r in zip(actor1_forwarded, critic1_forwarded)
        ]
        feedback_page2 = [
            self.domain_prompt.FEEDBACK_PAGE.format(actor_output=act_r, critic_output=cri_r)
            for act_r, cri_r in zip(actor2_forwarded, critic2_forwarded)
        ]
        summarizer_user_inputs = {
            "task_description": user_inputs,
            "memory_task_description": memory_task_descriptions,
            "memory_query": [
                build_role_memory_query(
                    query,
                    self.summarizer_agent.role,
                    "aggregate and finalize the team output",
                    [page1, page2],
                )
                for query, page1, page2 in zip(memory_queries, feedback_page1, feedback_page2)
            ],
            "function_name": function_names,
            "feedback_page1": feedback_page1,
            "feedback_page2": feedback_page2,
        }
        summarizer_system_inputs = {"task_domain_instructions": task_domain_instructions}
        summarizer_messages = self.summarizer_agent.invoke(summarizer_system_inputs, summarizer_user_inputs, summarizer_config)

        for sum_msg, mas_graph in zip(summarizer_messages, batch_message_graphs):
            mas_graph.update_message_graph(sum_msg, self.summarizer_agent.role, [self.critic_agent_1.role, self.critic_agent_2.role])
            # Sanitize: the summarizer must not search. Strip unexpected
            # <search> tags so env.step treats the output as an answer.
            final_response = strip_search_tags(sum_msg.response or "") if factual_qa else (sum_msg.response or "")
            sum_msg.response = final_response
            mas_graph.action = final_response

        return batch_message_graphs
