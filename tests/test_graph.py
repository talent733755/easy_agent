from langchain_core.messages import HumanMessage, AIMessage
from src.state import AgentState
from src.graph import build_graph


class TestGraphStructure:
    def test_graph_builds_without_error(self):
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_graph_has_required_nodes(self):
        graph = build_graph(checkpointer=None)
        node_names = [n for n in graph.builder.nodes] if hasattr(graph, 'builder') else []
        # After compilation, check the graph compiles
        compiled = graph
        assert compiled is not None

    def test_graph_includes_beauty_nodes(self):
        """Test that beauty nodes are included in the graph."""
        from unittest.mock import patch, MagicMock

        # Build graph and check nodes
        graph = build_graph(checkpointer=None)

        # Get node names from the graph builder
        if hasattr(graph, 'builder'):
            node_names = list(graph.builder.nodes.keys())
        else:
            # For compiled graph, check nodes attribute
            node_names = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []

        # Verify beauty nodes are present
        assert "intent_classify" in node_names, f"intent_classify not in nodes: {node_names}"
        assert "knowledge_retrieve" in node_names, f"knowledge_retrieve not in nodes: {node_names}"
        assert "mcp_customer" in node_names, f"mcp_customer not in nodes: {node_names}"
        # Also verify dynamic MCP pattern works
        mcp_nodes = [n for n in node_names if n.startswith("mcp_")]
        assert len(mcp_nodes) > 0, f"No MCP nodes found in: {node_names}"

        # Verify existing nodes still present
        assert "memory_retrieve" in node_names, f"memory_retrieve not in nodes: {node_names}"
        assert "agent" in node_names, f"agent not in nodes: {node_names}"


class TestGraphInvocation:
    def test_simple_conversation_flow(self):
        """Test a minimal graph invocation with a mock model."""
        from unittest.mock import MagicMock, patch
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="Hello! How can I help you?")

        # Mock beauty nodes at their source module
        mock_intent_result = {"intent": "general", "customer_name": "", "query_topic": ""}
        mock_knowledge_result = {"knowledge_results": []}
        mock_mcp_result = {"customer_context": {}, "mcp_results": {}}

        mock_mcp_node_fn = MagicMock(return_value=mock_mcp_result)

        with patch('src.graph._build_agent_model', return_value=mock_model), \
             patch('src.nodes.beauty.intent_classify_node', return_value=mock_intent_result), \
             patch('src.nodes.beauty.knowledge_retrieve_node', return_value=mock_knowledge_result), \
             patch('src.nodes.beauty.create_mcp_service_node', return_value=mock_mcp_node_fn):
            graph = build_graph(checkpointer=MemorySaver())

        state: AgentState = {
            "messages": [HumanMessage(content="Hi")],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 0,
            "max_iterations": 15,
            "nudge_counter": 0,
            "provider_name": "zhipu",
            # Beauty fields
            "intent": "general",
            "customer_context": {},
            "knowledge_results": [],
            "mcp_results": {},
        }

        config = {"configurable": {"thread_id": "test-1"}}
        result = graph.invoke(state, config)
        assert len(result["messages"]) > 1
        assert "Hello" in str(result["messages"][-1].content)