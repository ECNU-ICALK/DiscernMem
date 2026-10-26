"""
Multi-Agent System (MAS) message and trajectory module.

This module defines the message structures and state trajectory representations used when a MAS interacts with the environment. It mainly includes three data classes:
- MessageNode: represents the LLM input and output of a single agent.
- MessageGraph: represents the message-passing topology (DAG) of the MAS in a single interaction step,
  containing each agent's MessageNode as well as the global state, actions, and environment feedback.
- Trajectory: represents the complete sequence of state transitions between the MAS and the environment, composed of multiple MessageGraphs.

These classes can be used to:
- Record internal message flows and state evolution in MAS
- Construct traceable interaction trajectories
- Support subsequent analysis or training of LLM/agent policy models
"""
from dataclasses import dataclass, asdict
import json
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph

@dataclass
class MessageNode:
    # --- system prompt ---
    system_prompt_template: Optional[str] = None
    system_prompt_fields: Optional[dict] = None

    # --- user prompt ---
    user_prompt_template: Optional[str] = None
    user_prompt_fields: Optional[dict] = None

    # --- formatted prompts (as actually sent to LLM) ---
    formatted_system_prompt: Optional[str] = None
    formatted_user_prompt: Optional[str] = None

    # --- agent response ---
    response: Optional[str] = None

    # --- agent state shots ---
    state: Optional[dict] = None

@dataclass
class MessageGraph:
    """
    In a single interaction step between the MAS and the environment, 
    the message-passing topology among internal agents forms a directed acyclic graph, 
    where each node is an instance of the MessageNode class.
    """
    # --- state transition (MAS as a whole) --- 
    state: Optional[str] = None  # input prompt of mas
    action: Optional[str] = None  # final output of mas
    observation: Optional[str] = None  # env's feedback
    
    # --- detailed memssages in MAS ---
    mas_message_graph: Optional[nx.DiGraph] = None
    
    def update_message_graph(
        self, 
        message_node: MessageNode, 
        current_agent_id: str, 
        upstream_agent_ids: Optional[list[str]] = None
    ):
        if self.mas_message_graph is None:
            self.mas_message_graph = nx.DiGraph()
        
        self.mas_message_graph.add_node(current_agent_id, message=message_node)
        
        if upstream_agent_ids:
            for upstream_agent_id in upstream_agent_ids:
                
                if upstream_agent_id not in self.mas_message_graph:
                    raise ValueError(
                        f"Failed to construct MessageGraph: upstream agent (ID: {upstream_agent_id}) "
                        f"not found in the graph. Agent message flow order is incorrect."
                    )
                
                self.mas_message_graph.add_edge(upstream_agent_id, current_agent_id)
    
    def to_serializable(self) -> dict[str, str]:
        data = {
            "state": str(self.state) if self.state else "",
            "action": str(self.action) if self.action else "",
            "observation": str(self.observation) if self.observation else "",
        }

        if self.mas_message_graph:
            graph_data = json_graph.node_link_data(self.mas_message_graph)
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = asdict(node['message'])
            data["mas_message_graph_data"] = json.dumps(graph_data)
        else:
            data["mas_message_graph_data"] = json.dumps({})

        return data

    @classmethod
    def from_serializable(cls, data: dict[str, str]) -> 'MessageGraph':
        instance = cls(
            state=data.get("state"),
            action=data.get("action"),
            observation=data.get("observation"),
            mas_message_graph=None
        )

        graph_json_str = data.get("mas_message_graph_data")
        if graph_json_str:
            graph_data = json.loads(graph_json_str)
            for node in graph_data.get('nodes', []):
                if 'message' in node and node['message'] is not None:
                    node['message'] = MessageNode(**node['message'])
            instance.mas_message_graph = json_graph.node_link_graph(graph_data, directed=True)

        return instance




