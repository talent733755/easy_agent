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


def _build_mcp_intent_routes(mcp_configs: dict) -> dict:
    """从 MCP config 构建意图到节点名的路由映射。"""
    routes = {}
    for name, svc_config in mcp_configs.items():
        intent = getattr(svc_config, "intent", "")
        if intent:
            routes[intent] = f"mcp_{name}"
    return routes


def _route_intent(state: AgentState, mcp_routes: dict = None) -> str:
    """Route based on intent classification.

    Routes:
        - "knowledge": intent is "knowledge_query" -> knowledge_retrieve only
        - MCP route: intent matches a registered MCP service intent -> that MCP node
        - "mixed": intent is "mixed" -> knowledge (fan-out handles MCP nodes)
        - "skip": intent is "general" or unknown -> skip, go to memory_retrieve
    """
    intent = state.get("intent", "general")

    if intent == "knowledge_query":
        return "knowledge"
    elif intent == "mixed":
        return "knowledge"  # Fan-out handles MCP nodes
    elif intent in (mcp_routes or {}):
        return mcp_routes[intent]
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
    from src.nodes.memory_learn_node import memory_learn_node

    # Import beauty nodes
    from src.nodes.beauty import intent_classify_node, knowledge_retrieve_node, create_mcp_service_node

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
    else:
        builder.add_node("knowledge_retrieve", lambda s: {"knowledge_results": []})

    # Dynamic MCP service registration from config
    mcp_configs = beauty_config.mcp_servers if beauty_config else {}
    registered_mcp_nodes = []

    for service_name, svc_config in mcp_configs.items():
        svc_dict = {
            "url": svc_config.url,
            "timeout": svc_config.timeout,
            "intent": svc_config.intent,
            "endpoints": svc_config.endpoints,
            "name": service_name,
        }
        node_fn = create_mcp_service_node(svc_dict, model=model)
        builder.add_node(f"mcp_{service_name}", node_fn)
        registered_mcp_nodes.append(service_name)

    # Add existing nodes
    builder.add_node("memory_retrieve", lambda s: memory_retrieve_node(s, data_dir=config.memory["data_dir"]))
    builder.add_node("agent", agent_node_fn)
    builder.add_node("tool_executor", _make_tool_node(executor))
    builder.add_node("observe", observe_node)
    builder.add_node("human_gate", human_gate)
    builder.add_node("memory_learn", lambda s: memory_learn_node(s, data_dir=config.memory["data_dir"]))
    builder.add_node("nudge_check", lambda s: nudge_check_node(
        s, nudge_interval=config.agent["memory_nudge_interval"], data_dir=config.memory["data_dir"]
    ))
    builder.add_node("memory_save", lambda s: memory_save_node(s, data_dir=config.memory["data_dir"]))

    # Set entry point to intent_classify
    builder.set_entry_point("intent_classify")

    # Build dynamic routes
    mcp_routes = _build_mcp_intent_routes(mcp_configs)

    # Intent routing with dynamic MCP routes
    route_targets = {"knowledge": "knowledge_retrieve", "skip": "memory_retrieve"}
    for svc_name in registered_mcp_nodes:
        route_targets[f"mcp_{svc_name}"] = f"mcp_{svc_name}"
    builder.add_conditional_edges("intent_classify", lambda s: _route_intent(s, mcp_routes), route_targets)

    # Fan-out edges for parallel execution (mixed intent)
    builder.add_edge("intent_classify", "knowledge_retrieve")
    for svc_name in registered_mcp_nodes:
        builder.add_edge("intent_classify", f"mcp_{svc_name}")

    # Convergence to memory_retrieve
    builder.add_edge("knowledge_retrieve", "memory_retrieve")
    for svc_name in registered_mcp_nodes:
        builder.add_edge(f"mcp_{svc_name}", "memory_retrieve")

    # Edges
    builder.add_edge("memory_retrieve", "agent")

    # Conditional edge from agent
    builder.add_conditional_edges("agent", _route_agent, {
        "tools": "tool_executor",
        "end": END,
    })

    builder.add_edge("tool_executor", "observe")
    builder.add_edge("observe", "human_gate")
    builder.add_edge("human_gate", "memory_learn")
    builder.add_edge("memory_learn", "nudge_check")

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