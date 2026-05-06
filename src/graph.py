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


def _route_intent(state: AgentState) -> str:
    """Route based on intent classification.

    Routes:
        - "knowledge": intent is "knowledge_query" -> knowledge_retrieve only
        - "customer": intent is "query_customer" -> mcp_customer only
        - "both": intent is "mixed" -> both parallel
        - "skip": intent is "general" or unknown -> skip both, go to memory_retrieve
    """
    intent = state.get("intent", "general")
    if intent == "knowledge_query":
        return "knowledge"
    elif intent == "query_customer":
        return "customer"
    elif intent == "mixed":
        return "both"
    else:
        return "skip"


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
    from src.nodes.memory_retrieve import memory_retrieve_node
    from src.nodes.memory_save import memory_save_node
    from src.nodes.nudge_check import nudge_check_node

    # Import beauty nodes
    from src.nodes.beauty import intent_classify_node, knowledge_retrieve_node, mcp_customer_node

    # Import all tools
    from src.tools import read_file, write_file, list_dir
    from src.tools import web_search, web_fetch
    from src.tools import get_time, run_shell
    from src.tools import todo_write, memory_search

    all_tools = [read_file, write_file, list_dir, web_search, web_fetch, get_time, run_shell, todo_write, memory_search]
    tools_by_name = {t.name: t for t in all_tools}

    executor = ToolExecutor(tools_by_name)
    agent_node_fn = create_agent_node(model, tools=all_tools)

    # Build the graph
    builder = StateGraph(AgentState)

    # Add beauty nodes (Phase 1)
    builder.add_node("intent_classify", intent_classify_node)

    # Knowledge retrieve node with config
    beauty_config = config.beauty
    if beauty_config:
        kb_dir = beauty_config.knowledge_base.base_dir
        builder.add_node("knowledge_retrieve", lambda s: knowledge_retrieve_node(s, knowledge_dir=kb_dir))

        # MCP customer node with config
        mcp_config = beauty_config.mcp_servers.get("customer")
        if mcp_config:
            builder.add_node("mcp_customer", lambda s: mcp_customer_node(s, mcp_url=mcp_config.url, timeout=mcp_config.timeout))
        else:
            # Add dummy node that returns empty results
            builder.add_node("mcp_customer", lambda s: {"customer_context": {}, "mcp_results": {}})
    else:
        # Add dummy nodes when beauty config not available
        builder.add_node("knowledge_retrieve", lambda s: {"knowledge_results": []})
        builder.add_node("mcp_customer", lambda s: {"customer_context": {}, "mcp_results": {}})

    # Add existing nodes
    builder.add_node("memory_retrieve", lambda s: memory_retrieve_node(s, data_dir=config.memory["data_dir"]))
    builder.add_node("agent", agent_node_fn)
    builder.add_node("tool_executor", _make_tool_node(executor))
    builder.add_node("observe", observe_node)
    builder.add_node("human_gate", human_gate)
    builder.add_node("nudge_check", lambda s: nudge_check_node(
        s, nudge_interval=config.agent["memory_nudge_interval"], data_dir=config.memory["data_dir"]
    ))
    builder.add_node("memory_save", lambda s: memory_save_node(s, data_dir=config.memory["data_dir"]))

    # Set entry point to intent_classify
    builder.set_entry_point("intent_classify")

    # Intent routing: parallel branches to knowledge_retrieve and mcp_customer
    builder.add_conditional_edges("intent_classify", _route_intent, {
        "knowledge": "knowledge_retrieve",
        "customer": "mcp_customer",
        "both": "knowledge_retrieve",  # Will use fan-out for parallel execution
        "skip": "memory_retrieve",
    })

    # For "both" intent, we need parallel execution
    # LangGraph handles this through fan-out pattern: multiple edges from same node
    builder.add_edge("intent_classify", "knowledge_retrieve")  # Fan-out edge for parallel
    builder.add_edge("intent_classify", "mcp_customer")        # Fan-out edge for parallel

    # Both converge to memory_retrieve
    builder.add_edge("knowledge_retrieve", "memory_retrieve")
    builder.add_edge("mcp_customer", "memory_retrieve")

    # Edges
    builder.add_edge("memory_retrieve", "agent")

    # Conditional edge from agent
    builder.add_conditional_edges("agent", _route_agent, {
        "tools": "tool_executor",
        "end": END,
    })

    builder.add_edge("tool_executor", "observe")
    builder.add_edge("observe", "human_gate")
    builder.add_edge("human_gate", "nudge_check")

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


def _route_iteration(state: AgentState, max_iterations: int) -> str:
    if state.get("iteration_count", 0) >= max_iterations:
        return "end"
    return "continue"