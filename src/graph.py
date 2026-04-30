from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import AIMessage
from typing import TYPE_CHECKING

from src.state import AgentState
from src.config import load_config, AppConfig
from src.providers.factory import get_provider

if TYPE_CHECKING:
    from src.nodes.tool_executor import ToolExecutor


def _build_agent_model(config: AppConfig):
    """Build the LLM model from config."""
    provider_name = config.active_provider
    provider_config = config.providers.get(provider_name)
    if provider_config is None:
        raise ValueError(f"Provider '{provider_name}' not found in config")
    provider = get_provider(provider_name, provider_config)
    return provider.get_model()


def build_graph(checkpointer: BaseCheckpointSaver = None) -> StateGraph:
    if checkpointer is None:
        checkpointer = MemorySaver()

    config = load_config()
    model = _build_agent_model(config)

    # Import nodes
    from src.nodes.agent_node import create_agent_node
    from src.nodes.tool_executor import ToolExecutor
    from src.nodes.observe_node import observe_node
    from src.nodes.human_gate import human_gate
    from src.nodes.human_input import human_input_node
    from src.nodes.memory_retrieve import memory_retrieve_node
    from src.nodes.memory_save import memory_save_node
    from src.nodes.nudge_check import nudge_check_node

    # Import all tools
    from src.tools import read_file, write_file, list_dir
    from src.tools import web_search, web_fetch
    from src.tools import get_time, run_shell
    from src.tools import todo_write, memory_search

    all_tools = [read_file, write_file, list_dir, web_search, web_fetch, get_time, run_shell, todo_write, memory_search]
    tools_by_name = {t.name: t for t in all_tools}

    executor = ToolExecutor(tools_by_name)
    agent_node_fn = create_agent_node(model)

    # Build the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("memory_retrieve", lambda s: memory_retrieve_node(s, data_dir=config.memory["data_dir"]))
    builder.add_node("agent", agent_node_fn)
    builder.add_node("tool_executor", _make_tool_node(executor))
    builder.add_node("observe", observe_node)
    builder.add_node("human_gate", human_gate)
    builder.add_node("human_input", lambda s: human_input_node(s))
    builder.add_node("nudge_check", lambda s: nudge_check_node(
        s, nudge_interval=config.agent["memory_nudge_interval"], data_dir=config.memory["data_dir"]
    ))
    builder.add_node("memory_save", lambda s: memory_save_node(s, data_dir=config.memory["data_dir"]))

    # Set entry
    builder.set_entry_point("memory_retrieve")

    # Edges
    builder.add_edge("memory_retrieve", "agent")

    # Conditional edge from agent
    builder.add_conditional_edges("agent", _route_agent, {
        "tools": "tool_executor",
        "end": END,
    })

    builder.add_edge("tool_executor", "observe")
    builder.add_edge("observe", "human_gate")

    # Conditional edge from human_gate
    builder.add_conditional_edges("human_gate", _route_human_gate, {
        "interrupt": "human_input",
        "continue": "nudge_check",
    })

    builder.add_edge("human_input", "nudge_check")

    # Conditional edge from nudge_check (back to agent or end)
    builder.add_conditional_edges("nudge_check", lambda s: _route_iteration(s, config.agent["max_iterations"]), {
        "continue": "memory_save",
        "end": END,
    })

    builder.add_edge("memory_save", "agent")

    return builder.compile(checkpointer=checkpointer)


def _make_tool_node(executor):
    def tool_node(state: AgentState) -> dict:
        last_ai = None
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
                last_ai = m
                break
        if last_ai is None:
            return {}
        results = executor.execute(last_ai.tool_calls)
        return {"messages": results}
    return tool_node


def _route_agent(state: AgentState) -> str:
    max_iter = state.get("max_iterations", 15)
    if state.get("iteration_count", 0) >= max_iter:
        return "end"

    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


def _route_human_gate(state: AgentState) -> str:
    if state.get("pending_human_input", False):
        return "interrupt"
    return "continue"


def _route_iteration(state: AgentState, max_iterations: int) -> str:
    if state.get("iteration_count", 0) >= max_iterations:
        return "end"
    return "continue"