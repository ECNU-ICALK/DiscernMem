"""
Multi-agent interaction loop for PopQA (same interaction flow as TriviaQA).
"""
import logging

from common.utils.action_resolver import extract_search_tool_calls

logger = logging.getLogger(__name__)


def run_multiturn_mas(mas, env, system_prompt, question, ground_truth, generation_config, max_turns, action_resolver=None, max_obs_length=None):
    """Run one MAS generation call.

    The actor/user proxy performs iterative <search> -> <information> loops
    inside mas.generate(); the summarizer/user proxy emits the final <answer>.
    The trajectory schema matches TriviaQA: each MAS role emits one step with
    step_index, turn, agent, input, output, tool_calls, and feedback. The
    answer-producing role feedback includes observation, reward, answer_correct,
    extracted_answer, and done.
    """
    del max_turns  # one MAS call per question; part of the shared runner signature
    task_context = question
    all_steps = []
    agent_roles = [agent.role for agent in mas.agents_list] if hasattr(mas, 'agents_list') else []

    try:
        graphs = mas.generate([system_prompt], [task_context], generation_config, action_resolver=action_resolver)
        msg_graph = graphs[0]
        response = msg_graph.action or ""
    except Exception as e:
        logger.warning(f"    MAS generate failed: {e}")
        response = ""
        msg_graph = None

    observation, reward, done = env.step(response)

    if msg_graph and msg_graph.mas_message_graph:
        for agent_role in agent_roles:
            if agent_role in msg_graph.mas_message_graph:
                node_data = msg_graph.mas_message_graph.nodes[agent_role]
                msg_node = node_data.get("message")
                if msg_node:
                    extra_fields = ((getattr(msg_node, "user_prompt_fields", None) or {}).get("extra_fields") or {})
                    system_part = getattr(msg_node, "formatted_system_prompt", None) or msg_node.system_prompt_template or ""
                    user_part = getattr(msg_node, "formatted_user_prompt", None) or ""
                    full_input = f"{system_part}\n\n{user_part}" if system_part else user_part
                    agent_response = msg_node.response if msg_node.response else ""

                    is_answer_agent = "actor" in agent_role or "summarizer" in agent_role or "user proxy" in agent_role
                    if is_answer_agent:
                        import re
                        matches = re.findall(r"<answer>(.*?)</answer>", agent_response, re.DOTALL)
                        extracted_answer = matches[-1].strip() if matches else ""

                        answer_correct = False
                        if extracted_answer and ground_truth:
                            for gt in ground_truth:
                                if gt.lower() in extracted_answer.lower():
                                    answer_correct = True
                                    break

                        agent_reward = 0.0
                        if matches:
                            agent_reward = 1.0 if answer_correct else 0.0

                        feedback_obj = {
                            "observation": observation or "",
                            "reward": agent_reward,
                            "answer_correct": answer_correct,
                            "extracted_answer": extracted_answer,
                            "done": done if matches else False,
                        }
                    else:
                        feedback_obj = {
                            "observation": None,
                            "reward": None,
                            "answer_correct": None,
                        }

                    all_steps.append({
                        "step_index": len(all_steps) + 1,
                        "turn": 1,
                        "agent": agent_role,
                        "input": full_input,
                        "output": agent_response,
                        "tool_calls": extract_search_tool_calls(agent_response, observation or ""),
                        "feedback": feedback_obj,
                        "upstream_roles": [str(role) for role in msg_graph.mas_message_graph.predecessors(agent_role)],
                        "injected_memories": list(extra_fields.get("injected_memories") or []),
                        "memory_retrieval": list(extra_fields.get("retrieval_details") or []),
                        "state_summary": task_context,
                        "step_reward": feedback_obj.get("reward") or 0.0,
                        "done": bool(feedback_obj.get("done", False)),
                    })
    else:
        all_steps.append({
            "step_index": len(all_steps) + 1,
            "agent": "mas",
            "input": task_context,
            "output": response,
            "tool_calls": extract_search_tool_calls(response, observation or ""),
            "upstream_roles": [],
            "injected_memories": [],
            "memory_retrieval": [],
            "state_summary": task_context,
            "step_reward": reward,
            "done": bool(done),
            "feedback": {
                "observation": observation or "",
                "reward": reward,
                "done": done,
            },
        })

    logger.info(f"    done={done}, reward={reward:.1f}")
    final_reward = env.feedback()
    return final_reward, response, all_steps
